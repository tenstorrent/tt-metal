# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wan 3D RoPE tables. Constants per latent shape, so built on host and reused."""

from __future__ import annotations

import numpy as np
import ttnn

import ttml


def axis_split(head_dim: int) -> tuple[int, int, int]:
    """Per-axis head-dim split: (frames, height, width)."""
    h_dim = w_dim = 2 * (head_dim // 6)
    return head_dim - h_dim - w_dim, h_dim, w_dim


def _axis_tables(dim: int, max_seq_len: int, theta: float):
    # float64 frequencies, then cos/sin repeat-interleaved to `dim`: the interleaved
    # pairing the rotary_embedding_llama trans_mat expects.
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    ang = np.outer(np.arange(max_seq_len, dtype=np.float64), freqs)
    cos = np.repeat(np.cos(ang), 2, axis=1).astype(np.float32)
    sin = np.repeat(np.sin(ang), 2, axis=1).astype(np.float32)
    return cos, sin


def build_tables(
    *,
    head_dim: int,
    patch_size: tuple,
    latent_shape: tuple,
    max_seq_len: int = 1024,
    theta: float = 10000.0,
):
    """Return (cos, sin) numpy tables shaped (1, 1, S, head_dim) for one latent shape.

    latent_shape is the patch-embedder input (B, C, F, H, W).
    """
    t_dim, h_dim, w_dim = axis_split(head_dim)
    _, _, frames, height, width = latent_shape
    p_t, p_h, p_w = patch_size
    ppf, pph, ppw = frames // p_t, height // p_h, width // p_w
    grid = (ppf, pph, ppw)

    cos_parts, sin_parts = [], []
    for axis, (dim, count) in enumerate(((t_dim, ppf), (h_dim, pph), (w_dim, ppw))):
        cos, sin = _axis_tables(dim, max_seq_len, theta)
        if count > max_seq_len:
            raise ValueError(f"axis {axis} needs {count} positions but max_seq_len is {max_seq_len}")
        view = [1, 1, 1]
        view[axis] = count
        cos_parts.append(np.broadcast_to(cos[:count].reshape(*view, dim), (*grid, dim)))
        sin_parts.append(np.broadcast_to(sin[:count].reshape(*view, dim), (*grid, dim)))

    seq_len = ppf * pph * ppw
    cos = np.concatenate(cos_parts, axis=-1).reshape(1, seq_len, 1, head_dim)
    sin = np.concatenate(sin_parts, axis=-1).reshape(1, seq_len, 1, head_dim)
    return cos.transpose(0, 2, 1, 3).copy(), sin.transpose(0, 2, 1, 3).copy()


def _trans_mat(head_dim_tile: int = 32) -> np.ndarray:
    mat = np.zeros((1, 1, head_dim_tile, head_dim_tile), dtype=np.float32)
    idx = np.arange(0, head_dim_tile, 2)
    mat[..., idx, idx + 1] = 1.0
    mat[..., idx + 1, idx] = -1.0
    return mat


def _upload(arr: np.ndarray) -> ttnn.Tensor:
    # The caches are plain ttnn tensors, so take the value out of a ttml tensor.
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return ttml.autograd.Tensor.from_numpy(arr, ttnn.Layout.TILE, ttnn.bfloat16).get_value()


def build_rope_params(
    *,
    head_dim: int,
    patch_size: tuple,
    latent_shape: tuple,
    max_seq_len: int = 1024,
    theta: float = 10000.0,
):
    """RotaryEmbeddingParams for ttml.ops.rope.rope, carrying Wan's 3D tables."""
    cos, sin = build_tables(
        head_dim=head_dim,
        patch_size=patch_size,
        latent_shape=latent_shape,
        max_seq_len=max_seq_len,
        theta=theta,
    )
    params = ttml.ops.rope.RotaryEmbeddingParams()
    params.cos_cache = _upload(cos)
    params.sin_cache = _upload(sin)
    # Backward rotates by -theta: cos is even, sin is odd.
    params.neg_cos_cache = _upload(cos)
    params.neg_sin_cache = _upload(-sin)
    params.trans_mat = _upload(_trans_mat())
    params.sequence_length = cos.shape[2]
    params.head_dim = head_dim
    params.theta = theta
    return params


def apply(x, rope_params):
    """RoPE on a (B, heads, S, head_dim) tensor."""
    return ttml.ops.rope.rope(x, rope_params, 0)
