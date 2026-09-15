# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage 3 of 3: turn model output into an embedding.

    last_hidden_state (B, S, 768) fp32   one vector per token
      -> mean_pool           -> (B, 768)   one vector per text, padding excluded
      -> matryoshka_truncate -> (B, dim)   optional, dim <= 768, feature axis
      -> l2_normalize        -> (B, dim)   unit norm, so dot product is cosine similarity

The sequence axis disappears at mean_pool: that is where B*S token vectors collapse to B text
vectors. These are the stages the checkpoint declares in modules.json as
sentence_transformers.models Pooling and Normalize.

Unlike preprocessing, this is tensor work, so the TTNN port needs device equivalents for all
of it.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

L2_EPS = 1e-12


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average the token vectors of each text, ignoring padding.

    Mask-weighted mean over the sequence axis, not the CLS token: 1_Pooling/config.json sets
    pooling_mode_mean_tokens and disables CLS. Padded positions must be excluded because <pad>
    carries a non-zero embedding, so including it would make a text's embedding depend on how
    long its batch-mates are.

    Args:
        last_hidden_state: Per-token model output, (B, S, 768) fp32.
        attention_mask: (B, S) int64, 1 for real tokens and 0 for padding.

    Returns:
        torch.Tensor: (B, 768) fp32, one vector per text. Not unit norm; l2_normalize does that.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = L2_EPS) -> torch.Tensor:
    """Scale to unit norm along one axis.

    Args:
        x: Any tensor, typically (B, 768) pooled embeddings.
        dim: Axis to normalize along, defaulting to the last.
        eps: Floor on the divisor, guarding a zero-norm row.

    Returns:
        torch.Tensor: Same shape as x, with unit norm along dim.
    """
    return F.normalize(x, p=2.0, dim=dim, eps=eps)


def matryoshka_truncate(embeddings: torch.Tensor, dim: Optional[int]) -> torch.Tensor:
    """Keep only the leading features of each embedding.

    The feature axis, not the sequence axis. Upstream's NomicBertModel.forward(matryoshka_dim=)
    slices sequence_output[:, :matryoshka_dim], which drops tokens and keeps full-width
    features; that is a different operation and is not what the published embeddings use.

    Args:
        embeddings: Pooled embeddings, (B, 768) fp32.
        dim: Target width, at most 768, or None to pass through unchanged.

    Returns:
        torch.Tensor: (B, dim) fp32, or the input unchanged when dim is None.

    Raises:
        ValueError: If dim exceeds the embedding width.
    """
    if dim is None:
        return embeddings
    if dim > embeddings.shape[-1]:
        raise ValueError(f"matryoshka dim {dim} exceeds embedding width {embeddings.shape[-1]}")
    return embeddings[..., :dim]


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity between two sets of embeddings.

    Args:
        a: (N, D) fp32.
        b: (M, D) fp32, same width as a.

    Returns:
        torch.Tensor: (N, M) fp32 in [-1, 1], entry (i, j) the similarity of a[i] and b[j].
    """
    return l2_normalize(a) @ l2_normalize(b).T
