import argparse
from dotenv import load_dotenv
import os
import time

from openai import OpenAI, OpenAIError
from transformers import AutoModelForCausalLM


load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = BASE_URL = "https://litellm-proxy--tenstorrent.workload.tenstorrent.com"
MODEL = MODEL = "azure/gpt-5-chat-2025-08-07"
SYSTEM_PROMPT = "You are a senior ML software engineer at Tenstorrent specializing in Python and machine learning. Provide detailed, technically accurate explanations and code examples when helpful. Keep responses clear and structured."

DEBUG = False  # Print headers for debugging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--api_key", type=str, default=API_KEY, help="API key.")
    parser.add_argument("-m", "--model", type=str, default=MODEL, help="Path to model checkpoint.")
    parser.add_argument("-n", "--no_stream", action="store_true", default=False, help="Toggle streaming off.")
    parser.add_argument(
        "-p", "--print_perf", action="store_true", default=False, help="Toggle printing of performance."
    )
    parser.add_argument("-s", "--system_prompt", type=str, default=SYSTEM_PROMPT, help="Chatbot's system prompt.")
    parser.add_argument("-u", "--base_url", type=str, default=BASE_URL, help="Base URL to API.")
    return parser.parse_args()


def call_chat_api(model, messages, client, stream=True, print_perf=False):
    """
    Sends a message to the vLLM API and gets the assistant's response.
    """
    # Configuration
    temperature = 0.7

    # Prepare API request parameters
    request_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }

    # Add stream_options only if stream=True
    if stream and print_perf:
        request_params["stream_options"] = {"include_usage": True}

    # Send request to the API
    start_time = time.time()
    try:
        response = client.chat.completions.create(**request_params)
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None
    if DEBUG:
        print("Response Headers:", response.response.headers)
    content, usage, first_time = parse_response(response, stream, print_perf)
    end_time = time.time()
    if print_perf and usage:
        ttft = first_time - start_time
        generation_rate = (usage.completion_tokens - 1) / (end_time - first_time)
        # subtract 1 from completion_tokens because the generation time is since the first token
        print("Perf: TTFT = %.2f s, rate = %.2f tokens/s/user" % (ttft, generation_rate))
    return content


def parse_response(response, stream=True, print_perf=False):
    """
    Parses the response for streaming and non-streaming cases.
    """

    usage = None
    first_time = None
    if stream:
        print("Assistant: ", end="", flush=True)
        # Parsing the streaming response
        full_content = []
        for chunk in response:
            if first_time is None:
                first_time = time.time()
            if chunk.choices:  # Regular streamed content
                chunk_content = chunk.choices[0].delta.content
                if chunk_content:
                    print(chunk_content, end="", flush=True)
                    full_content.append(chunk_content)
            elif chunk.usage:  # Final chunk with usage data
                usage = chunk.usage
        print("")
        content = "".join(full_content)
    else:
        content = response.choices[0].message.content
        usage = response.usage
        print("Assistant:", content)

    if print_perf and usage:
        print("Usage:", usage)

    return content, usage, first_time


def setup_client(api_key, base_url, system_prompt):
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Initialize the conversation with a system message
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]

    print(f"System Prompt: {system_prompt}\n")
    return client, messages


def load_hf_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model


def main():
    args = parse_args()

    # Setup client with system prompt
    client, messages = setup_client(args.api_key, args.base_url, args.system_prompt)
    breakpoint()

    # Load HF model to be ported to TT-NN
    hf_model = load_hf_model("gpt2")
    print(f"HF model:\n{hf_model}")
    breakpoint()

    # Get TT-NN MLIR Json from TT-Forge
    # TODO
    breakpoint()

    # AI Prompts
    user_input1 = f"Write a Python file containing a Python class skeleton (with un-implemented functions for forward passes) inheriting from LightweightModule for each nn module in the model: {hf_model}. The attributes don't need to be defined. These Python classes will be completed to implement a TT-NN model in a following step. Please do not provide an explanation, only the Python code."
    user_inputs = [user_input1]

    while user_inputs:
        user_input = user_inputs.pop(0)

        # Append the user's message to the conversation history
        messages.append({"role": "user", "content": user_input})

        # Call the vLLM API to generate the assistant's response
        content = call_chat_api(args.model, messages, client, stream=(not args.no_stream), print_perf=args.print_perf)

        if content:
            # Append the assistant's message to the conversation history
            messages.append({"role": "assistant", "content": content})
        else:
            print("An error occurred. Please try again.")
        print("")


if __name__ == "__main__":
    main()
