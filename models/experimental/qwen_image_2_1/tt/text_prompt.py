# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Host tokenization for Qwen-Image 2.1 text-only prompts.

Only tokenization and template metadata execute here. Embedding lookup and
learned text-encoder operations belong to the TT encoder, not this helper.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoProcessor

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
T2I_TEMPLATE = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n" "<|im_start|>user\n{}<|im_end|>\n" "<|im_start|>assistant\n"
)


def tokenize_prompt(checkpoint: Path, prompt: str) -> tuple[torch.Tensor, int]:
    """Return CPU token IDs and system-prefix length for one raw text prompt.

    The raw template and processor settings match the pinned upstream pipeline.
    ``apply_chat_template`` is used only to measure the system-prefix length;
    using it to encode the complete prompt would change the checkpoint's input.
    """
    if not isinstance(prompt, str):
        raise TypeError("text-only batch-one encoding requires a string prompt")
    processor = AutoProcessor.from_pretrained(Path(checkpoint) / "processor", local_files_only=True)
    system_message = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    system_tokens = processor.apply_chat_template(system_message, tokenize=True, return_dict=False)
    drop_idx = len(system_tokens[0])
    inputs = processor(
        text=[T2I_TEMPLATE.format(prompt or " ")],
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
    ids = inputs.input_ids.cpu().contiguous()
    if not bool(inputs.attention_mask.all()):
        raise ValueError("a single text-only prompt should not contain padding")
    return ids, drop_idx
