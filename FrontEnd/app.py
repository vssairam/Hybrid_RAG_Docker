import os
import requests
import streamlit as st

API_URL = os.getenv(
    "API_URL",
    "http://backend:8000",
)

st.set_page_config(
    page_title="Hybrid RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Hybrid RAG Chatbot")

st.caption(
    "Streamlit Frontend + FastAPI Backend"
)


def api_get(endpoint):

    response = requests.get(
        API_URL + endpoint,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def api_post(endpoint, payload):

    response = requests.post(
        API_URL + endpoint,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    return response.json()


try:

    knowledge_bases = api_get(
        "/knowledge-bases"
    )

except Exception as error:

    st.error(
        f"Unable to connect to backend: {error}"
    )

    st.stop()


if not knowledge_bases:

    st.info(
        "No existing knowledge bases found."
    )

    st.stop()


with st.sidebar:

    st.header("📁 Knowledge Bases")

    kb_map = {
        (
            f"{item['title']} "
            f"[{item['id'][:6]}]"
        ): item["id"]
        for item in knowledge_bases
    }

    selected_label = st.selectbox(
        "Select Knowledge Base",
        list(kb_map.keys()),
    )

    selected_kb = kb_map[
        selected_label
    ]

    st.divider()

    st.header("💬 Topic Discussions")

    chats = api_get(
        f"/knowledge-bases/{selected_kb}/chats"
    )

    chat_map = {
        (
            f"{item['title']} "
            f"[{item['id'][:6]}]"
        ): item["id"]
        for item in chats
    }

    selected_chat = None

    if chat_map:

        chat_label = st.selectbox(
            "Select Topic",
            list(chat_map.keys()),
        )

        selected_chat = chat_map[
            chat_label
        ]

    topic_title = st.text_input(
        "New Topic Name",
        value="New Topic",
    )

    if st.button("Start New Topic"):

        result = api_post(
            "/chats",
            {
                "kb_id": selected_kb,
                "title": topic_title,
            },
        )

        st.session_state[
            "new_chat_id"
        ] = result["chat_id"]

        st.rerun()


if (
    st.session_state.get("new_chat_id")
    and any(
        item["id"] == st.session_state["new_chat_id"]
        for item in chats
    )
):

    selected_chat = st.session_state[
        "new_chat_id"
    ]


details = api_get(
    f"/knowledge-bases/{selected_kb}"
)

st.subheader(
    details["title"]
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "FAISS Chunks",
    details["chunks"],
)

col2.metric(
    "Entities",
    details["entities"],
)

col3.metric(
    "Relationships",
    details["relationships"],
)


if not selected_chat:

    st.info(
        "Create a new topic to begin."
    )

    st.stop()


history = api_get(
    f"/chats/{selected_chat}/messages"
)

for message in history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


question = st.chat_input(
    "Ask a question about your documents..."
)

if question:

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Retrieving answer from backend..."
        ):

            result = api_post(
                "/ask",
                {
                    "kb_id": selected_kb,
                    "chat_id": selected_chat,
                    "question": question,
                },
            )

        st.markdown(
            result["answer"]
        )

        with st.expander(
            "📖 Sources"
        ):

            for source in result["sources"]:

                st.write(source)