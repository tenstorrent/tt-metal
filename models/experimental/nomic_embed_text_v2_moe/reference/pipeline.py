# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sentence-embedding pipeline on top of the encoder backbone.

NomicBertModel returns token-level hidden states. The published embeddings are what the
sentence-transformers stack produces on top, per the checkpoint's modules.json,
1_Pooling/config.json and config_sentence_transformers.json:

    task prefix -> tokenize -> encoder -> mask-weighted mean pool -> L2 normalize

with optional Matryoshka truncation before the final normalize.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

# Verbatim from config_sentence_transformers.json. The trailing space is part of the prefix.
PROMPTS: dict[str, str] = {
    "query": "search_query: ",
    "passage": "search_document: ",
    "Classification": "classification: ",
    "MultilabelClassification": "classification: ",
    "Clustering": "clustering: ",
    "PairClassification": "classification: ",
    "STS": "classification: ",
    "Summarization": "classification: ",
    "Speed": "search_document: ",
}

# From sentence_bert_config.json.
MAX_SEQ_LENGTH = 512

L2_EPS = 1e-12


def apply_prompt(texts: Sequence[str], prompt_name: Optional[str]) -> list[str]:
    """The model was trained with these prefixes; dropping one measurably moves the embedding."""
    if prompt_name is None:
        return list(texts)
    if prompt_name not in PROMPTS:
        raise KeyError(f"unknown prompt {prompt_name!r}; known: {sorted(PROMPTS)}")
    return [PROMPTS[prompt_name] + text for text in texts]


def tokenize(tokenizer, texts: Sequence[str], max_length: int = MAX_SEQ_LENGTH):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask-weighted mean over the sequence axis, not the CLS token.

    Padded positions must be excluded: <pad> carries a non-zero embedding, so including it
    would make a sentence's embedding depend on its batch-mates.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = L2_EPS) -> torch.Tensor:
    return F.normalize(x, p=2.0, dim=dim, eps=eps)


def matryoshka_truncate(embeddings: torch.Tensor, dim: Optional[int]) -> torch.Tensor:
    """Keep the first dim features.

    Feature axis, not sequence. Upstream's NomicBertModel.forward(matryoshka_dim=...) slices
    sequence_output[:, :matryoshka_dim], which drops tokens and keeps full-width features.
    """
    if dim is None:
        return embeddings
    if dim > embeddings.shape[-1]:
        raise ValueError(f"matryoshka dim {dim} exceeds embedding width {embeddings.shape[-1]}")
    return embeddings[..., :dim]


def pool_and_normalize(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    matryoshka_dim: Optional[int] = None,
) -> torch.Tensor:
    """Pool, truncate, normalize.

    Normalizing before truncation instead gives a different norm but the same direction, and
    the model's declared similarity is cosine, so the order does not affect any intended use.
    """
    pooled = mean_pool(last_hidden_state, attention_mask)
    return l2_normalize(matryoshka_truncate(pooled, matryoshka_dim))


@torch.no_grad()
def encode(
    model,
    tokenizer,
    texts: Sequence[str],
    prompt_name: Optional[str] = None,
    matryoshka_dim: Optional[int] = None,
    max_length: int = MAX_SEQ_LENGTH,
) -> torch.Tensor:
    """Text to normalized embedding. Accepts either the vendored or the upstream model."""
    encoded = tokenize(tokenizer, apply_prompt(texts, prompt_name), max_length=max_length)
    attention_mask = encoded["attention_mask"]

    out = model(input_ids=encoded["input_ids"], attention_mask=attention_mask)
    last_hidden_state = out if isinstance(out, torch.Tensor) else out.last_hidden_state

    return pool_and_normalize(last_hidden_state, attention_mask, matryoshka_dim=matryoshka_dim)


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return l2_normalize(a) @ l2_normalize(b).T
