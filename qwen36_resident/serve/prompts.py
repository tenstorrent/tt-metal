# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The chat prompts of the serving checks: greedy_ref.py decodes them directly, chat_client.py through vllm serve."""

PROMPTS = [
    dict(
        name="transformer",
        content="Explain, step by step, how a transformer neural network processes a sequence of tokens to "
        "predict the next one.",
        thinking=True,
    ),
    dict(
        name="lighthouse",
        content="Write a short story (about 150 words) about a lighthouse keeper who finds a message in a bottle.",
        thinking=True,
    ),
    dict(
        name="planets",
        content="List the planets of the solar system in order from the sun, with one interesting fact about each.",
        thinking=False,
    ),
    dict(
        name="fibonacci",
        content="Write a Python function that returns the n-th Fibonacci number using memoization, and explain "
        "its time and space complexity.",
        thinking=True,
    ),
    dict(
        name="arithmetic",
        content="A train leaves at 09:40 and arrives at 13:05. How long is the trip? Show the calculation.",
        thinking=False,
    ),
]


def messages(p):
    return [{"role": "user", "content": p["content"]}]


def chat_ids(tokenizer, p):
    """the prompt token ids of p as the server renders them: the chat template, generation prompt on"""
    ids = tokenizer.apply_chat_template(
        messages(p), add_generation_prompt=True, tokenize=True, enable_thinking=p["thinking"]
    )
    if hasattr(ids, "keys"):  # transformers 5 returns a BatchEncoding
        ids = ids["input_ids"]
    return [int(t) for t in ids]
