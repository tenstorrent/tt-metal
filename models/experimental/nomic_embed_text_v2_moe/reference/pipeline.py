# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The sentence-embedding pipeline wrapped around the encoder backbone.

`NomicBertModel` returns token-level hidden states; the published embeddings are what the
sentence-transformers stack produces on top of that. From the checkpoint's own
`modules.json` / `1_Pooling/config.json` / `config_sentence_transformers.json`:

    task prefix  ->  tokenize  ->  encoder  ->  mask-weighted mean pool  ->  L2 normalize

with an optional Matryoshka truncation before the final normalize.

The task prefix is mandatory, not decorative: the model was trained with it, and dropping
it measurably moves the embedding.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

# Verbatim from the checkpoint's config_sentence_transformers.json. Note the trailing space.
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


def apply_prompt(texts: Sequence[str], prompt_name: Optional[str]) -> list[str]:
    if prompt_name is None:
        return list(texts)
    if prompt_name not in PROMPTS:
        raise KeyError(f"unknown prompt {prompt_name!r}; known: {sorted(PROMPTS)}")
    prefix = PROMPTS[prompt_name]
    return [prefix + t for t in texts]


def tokenize(tokenizer, texts: Sequence[str], max_length: int = MAX_SEQ_LENGTH):
    """`include_prompt` is true in 1_Pooling/config.json, so prefix tokens are pooled too."""
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask-weighted mean over the sequence axis -- NOT the CLS token.

    `1_Pooling/config.json` sets `pooling_mode_mean_tokens: true` and every other mode
    false. Padded positions must be excluded: they carry a non-zero `<pad>` embedding and
    including them would make the result depend on batch composition.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2.0, dim=dim, eps=eps)


def matryoshka_truncate(embeddings: torch.Tensor, dim: Optional[int]) -> torch.Tensor:
    """Keep the first `dim` FEATURES.

    Note this is the feature axis. Upstream's `NomicBertModel.forward(matryoshka_dim=...)`
    slices `sequence_output[:, :matryoshka_dim]`, which is the SEQUENCE axis -- an upstream
    bug that silently returns a truncated sequence of full-width vectors. Truncation belongs
    after pooling, which is where it is here.
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
    """pool -> truncate -> normalize.

    On ordering: normalising before truncation and normalising after it give vectors of
    different norms (1.0 vs ~0.57 at d=256), but they are positive multiples of the same
    direction, so cosine similarity -- the model's declared `similarity_fn_name` -- is
    identical either way. The tests pin that as a lemma rather than leaving it implied.
    """
    pooled = mean_pool(last_hidden_state, attention_mask)
    pooled = matryoshka_truncate(pooled, matryoshka_dim)
    return l2_normalize(pooled)


@torch.no_grad()
def encode(
    model,
    tokenizer,
    texts: Sequence[str],
    prompt_name: Optional[str] = None,
    matryoshka_dim: Optional[int] = None,
    max_length: int = MAX_SEQ_LENGTH,
) -> torch.Tensor:
    """Full text -> normalized embedding path, for either the vendored or upstream model."""
    prompted = apply_prompt(texts, prompt_name)
    encoded = tokenize(tokenizer, prompted, max_length=max_length)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_state = out if isinstance(out, torch.Tensor) else out.last_hidden_state

    return pool_and_normalize(last_hidden_state, attention_mask, matryoshka_dim=matryoshka_dim)


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`similarity_fn_name` is "cosine" in config_sentence_transformers.json."""
    return l2_normalize(a) @ l2_normalize(b).T
