# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The SFT prompts, byte for byte. tt-inference-server checks the same strings: these are the inference contract
in docs/PRODUCTIZATION_OF_TRAINING.md. Device-free."""

from ttml.experimental.datasets.provider.sft.alpaca import AlpacaProvider
from ttml.experimental.datasets.utils.sft.templates.alpaca import alpaca_prompt
from ttml.experimental.datasets.utils.sft.templates.custom import custom_prompt_and_response

ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\nTranslate to French.\n\n"
    "### Input:\nGood morning\n\n"
    "### Response:\n"
)

ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\nGive three tips for staying healthy.\n\n"
    "### Response:\n"
)


def test_alpaca_prompt():
    assert alpaca_prompt("Translate to French.", "Good morning") == ALPACA_WITH_INPUT
    assert alpaca_prompt("Give three tips for staying healthy.") == ALPACA_NO_INPUT
    assert alpaca_prompt("Give three tips for staying healthy.", " \n") == ALPACA_NO_INPUT


def test_alpaca_provider_prompt():
    provider = AlpacaProvider.__new__(AlpacaProvider)
    example = {"instruction": "Translate to French.", "input": "Good morning", "output": "Bonjour"}
    assert provider._prompt_and_response(example) == (ALPACA_WITH_INPUT, "Bonjour")
    example = {"instruction": "Give three tips for staying healthy.", "input": "", "output": "Sleep."}
    assert provider._prompt_and_response(example) == (ALPACA_NO_INPUT, "Sleep.")


def test_custom_prompt_is_alpaca():
    mapping = {"instruction": "question", "input": "context", "output": "answer"}
    example = {"question": "Translate to French.", "context": "Good morning", "answer": "Bonjour"}
    assert custom_prompt_and_response(example, "alpaca", mapping) == (ALPACA_WITH_INPUT, "Bonjour")
    example = {"question": "Give three tips for staying healthy.", "context": None, "answer": "Sleep."}
    assert custom_prompt_and_response(example, "alpaca", mapping) == (ALPACA_NO_INPUT, "Sleep.")
