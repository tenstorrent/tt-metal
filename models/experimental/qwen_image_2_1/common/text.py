# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prompt templating and tokenization for the T2I path (host side, transformers tokenizer only)."""
from __future__ import annotations

import os
from functools import lru_cache

import torch

from .config import DROP_IDX, PROMPT_TEMPLATE_T2I, snapshot_dir


@lru_cache(maxsize=1)
def get_tokenizer(root: str | None = None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.path.join(root or snapshot_dir(), "processor"))


def tokenize_prompt(prompt: str, root: str | None = None) -> torch.Tensor:
    """input_ids [1, L] for the T2I template (no padding for a single prompt). Empty prompt -> ' '."""
    prompt = prompt if prompt else " "
    text = PROMPT_TEMPLATE_T2I.format(prompt)
    tok = get_tokenizer(root)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    return ids


def drop_system_tokens(x: torch.Tensor) -> torch.Tensor:
    """Drop the DROP_IDX leading system-role tokens from a [L, ...] sequence."""
    return x[DROP_IDX:]
