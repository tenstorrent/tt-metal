# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer helpers for Muse-Glimmer-30B (host only; avoids transformers' unknown-model AutoTokenizer path)."""

import os

from transformers import PreTrainedTokenizerFast


def load_tokenizer(ckpt_dir) -> PreTrainedTokenizerFast:
    ckpt_dir = str(ckpt_dir)
    with open(os.path.join(ckpt_dir, "chat_template.jinja")) as f:
        chat_template = f.read()
    return PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(ckpt_dir, "tokenizer.json"),
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        chat_template=chat_template,
    )


def encode_chat(tokenizer, user_text: str, system_text: str | None = None) -> list[int]:
    """Render the chat template (it already contains BOS) and tokenize without adding special tokens."""
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return list(tokenizer(text, add_special_tokens=False).input_ids)


def decode(tokenizer, ids) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=True)
