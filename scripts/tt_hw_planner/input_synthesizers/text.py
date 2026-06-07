# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Text input synthesizer — used by T2T/T2S/LLM/VLM templates.

Most text-input demos just need a tokenizer call; this helper emits
the standard pattern with sensible defaults (prompt template, default
sample text for smoke tests).
"""

from __future__ import annotations

from ..task_templates._base import TemplateContext


DEFAULT_SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."


def emit_source(ctx: TemplateContext) -> str:
    """Return source code for ``demo/text_loader.py``."""
    model_id = ctx.model_id
    tokenizer_id = ctx.tokenizer_name or model_id

    return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Text tokenizer wrapper for the generated demo."""

from __future__ import annotations

from typing import Optional

import torch


_TOKENIZER_ID = {tokenizer_id!r}
DEFAULT_SAMPLE_TEXT = {DEFAULT_SAMPLE_TEXT!r}


def _load_tokenizer():
    import transformers
    return transformers.AutoTokenizer.from_pretrained(_TOKENIZER_ID, trust_remote_code=True)


def tokenize(text: str) -> torch.Tensor:
    """Return ``input_ids`` shape (1, T) as torch.long."""
    tok = _load_tokenizer()
    enc = tok(text, return_tensors="pt")
    return enc["input_ids"]


def detokenize(token_ids) -> str:
    tok = _load_tokenizer()
    return tok.decode(token_ids, skip_special_tokens=True)


__all__ = ["DEFAULT_SAMPLE_TEXT", "tokenize", "detokenize"]
'''
