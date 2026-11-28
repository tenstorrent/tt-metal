# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The Alpaca prompt (Stanford Alpaca), of an instruction with its input or without one.

tt-inference-server sends these exact prompts at inference time; a change here changes the inference contract
in docs/PRODUCTIZATION_OF_TRAINING.md and has to be made there too.
"""

ALPACA_PROMPT_INTRO = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request."
)

ALPACA_PROMPT_INTRO_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)


def alpaca_prompt(instruction: str, input_text: str = "") -> str:
    """The prompt of ``instruction``, with ``input_text`` as its input unless it is blank."""
    if input_text.strip():
        return (
            f"{ALPACA_PROMPT_INTRO}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    return f"{ALPACA_PROMPT_INTRO_NO_INPUT}\n\n### Instruction:\n{instruction}\n\n### Response:\n"
