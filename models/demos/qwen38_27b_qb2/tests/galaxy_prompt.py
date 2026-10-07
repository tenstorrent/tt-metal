# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the qualification prompt before allocating devices or loading weights."""


def qualification_prompt(tokenizer):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain in one short sentence why leaves are green."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Qualification prompt must contain a flat list of nonnegative integer token IDs")
    return tokens
