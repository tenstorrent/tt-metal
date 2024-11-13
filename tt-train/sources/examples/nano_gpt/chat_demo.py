# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import streamlit as st
import subprocess

START_MESSAGE = "*******************\n"
END_MESSAGE = "*******************\n"


def read_before_start_message(stream):
    output = ""
    while True:
        ch = stream.read(1)
        if not ch:
            break
        output += ch
        if output.endswith(START_MESSAGE):
            break


def read_until_end_message(stream):
    index = 0
    output = ""
    while True:
        ch = stream.read(1)
        if not ch:
            break
        output += ch
        if index + len(END_MESSAGE) < len(output):
            yield output[index]
            index += 1
        if output.endswith(END_MESSAGE):
            break


def stream_executable(process, user_input):
    try:
        # Send input to the executable
        process.stdin.write(user_input + "\n")
        process.stdin.flush()  # Ensure it's sent immediately

        # Read output until the start message
        read_before_start_message(process.stdout)
        # Read output character by character
        yield from read_until_end_message(process.stdout)
    except Exception as e:
        yield f"An error occurred: {e}"


def run_executable(executable_path):
    try:
        # Start the subprocess
        process = subprocess.Popen(
            executable_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Use text mode for strings
            bufsize=1,  # Line buffering
            universal_newlines=True,  # Use universal newlines mode
            shell=True,  # Run the command through the shell
        )
        return process
    except Exception as e:
        st.error(f"Failed to start the executable: {e}")
        return None


def main():
    st.title("Shakespeare Chat")

    # Specify the path to your executable
    executable_path = "TT_METAL_LOGGER_LEVEL=FATAL"
    executable_path += " /home/ubuntu/ML-Framework-CPP/build/sources/examples/nano_gpt/nano_gpt"
    executable_path += " -p transformer.msgpack"
    executable_path += " -s 5489 -e"

    # Initialize session state
    if "process" not in st.session_state:
        st.session_state.process = run_executable(executable_path)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in stream_executable(st.session_state.process, prompt):
                if chunk == "\n":
                    full_response += "  "
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
