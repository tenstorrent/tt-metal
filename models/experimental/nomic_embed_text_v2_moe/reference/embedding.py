# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end driver: text in, embeddings out.

encode() runs the three stages in order and owns nothing itself:

    texts             list[str], length B
      -> preprocessing.apply_prompt / tokenize
                      input_ids      (B, S) int64
                      attention_mask (B, S) int64
      -> inference.forward
                      (B, S, 768) fp32
      -> postprocessing.pool_and_normalize
                      (B, matryoshka_dim or 768) fp32, unit norm

The model is a duck-typed argument, not an import: anything accepting (input_ids,
attention_mask) works, so the identical path runs over the vendored reference and the upstream
HF model. That is what makes a parity difference attributable to the backbone rather than to
the pre- or post-processing on either side.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from models.experimental.nomic_embed_text_v2_moe.reference.inference import forward
from models.experimental.nomic_embed_text_v2_moe.reference.postprocessing import pool_and_normalize
from models.experimental.nomic_embed_text_v2_moe.reference.preprocessing import (
    MAX_SEQ_LENGTH,
    apply_prompt,
    tokenize,
)


def encode(
    model,
    tokenizer,
    texts: Sequence[str],
    prompt_name: Optional[str] = None,
    matryoshka_dim: Optional[int] = None,
    max_length: int = MAX_SEQ_LENGTH,
) -> torch.Tensor:
    """Turn text into normalized embeddings, running all three stages.

    Row i is the embedding of texts[i], independent of the other rows: padding is masked out at
    pooling, so a short text gets the same vector whether encoded alone or in a ragged batch.
    The dot product of two rows is their cosine similarity.

    Args:
        model: The vendored NomicBertModel or the upstream HF model.
        tokenizer: The XLMRobertaTokenizerFast loaded from the checkpoint.
        texts: The input strings, length B.
        prompt_name: Task prefix key ("query", "passage", ...), or None for no prefix.
        matryoshka_dim: Target embedding width, at most 768, or None for the full 768.
        max_length: Tokenizer truncation limit, defaulting to MAX_SEQ_LENGTH (512).

    Returns:
        torch.Tensor: (B, matryoshka_dim or 768) fp32, unit norm.
    """
    encoded = tokenize(tokenizer, apply_prompt(texts, prompt_name), max_length=max_length)
    attention_mask = encoded["attention_mask"]

    hidden = forward(model, encoded["input_ids"], attention_mask)

    return pool_and_normalize(hidden, attention_mask, matryoshka_dim=matryoshka_dim)
