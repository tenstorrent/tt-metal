# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU-only PyTorch reference implementations.

Used by op-level tests for operations that don't have a clean ``nn.Module``
analog in HF (or where the HF implementation does more than we want to
test). All functions take and return plain ``torch.Tensor`` and are
deterministic given a seeded input.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference RMSNorm (matches Llama/Qwen2 conventions).

    ``out = x * rsqrt(mean(x**2) + eps) * weight``
    """
    orig_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_f32 = x_f32 * torch.rsqrt(var + eps)
    return x_f32.to(orig_dtype) * weight.to(orig_dtype)


# ---------------------------------------------------------------------------
# Rotary embedding (apply only — table construction lives elsewhere)
# ---------------------------------------------------------------------------


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Half-rotate convention: ``y[..., :d/2] = x[..., :d/2]*cos - x[..., d/2:]*sin`` etc.

    ``x`` and ``cos``/``sin`` must be broadcast-compatible. Returns a tensor
    with the same shape as ``x``.
    """
    d = x.shape[-1]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


# ---------------------------------------------------------------------------
# QKV head splitting / merging (matches the TTNN op semantics)
# ---------------------------------------------------------------------------


def nlp_create_qkv_heads(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a fused QKV projection into Q, K, V tensors.

    Input ``qkv`` shape: ``[..., seq, (num_heads_q + 2*num_heads_kv) * head_dim]``.
    Returns three tensors of shape ``[..., heads_*, seq, head_dim]``.
    """
    *prefix, seq, _ = qkv.shape
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.reshape(*prefix, seq, num_heads_q, head_dim).transpose(-3, -2).contiguous()
    k = k.reshape(*prefix, seq, num_heads_kv, head_dim).transpose(-3, -2).contiguous()
    v = v.reshape(*prefix, seq, num_heads_kv, head_dim).transpose(-3, -2).contiguous()
    return q, k, v


def nlp_concat_heads(x: torch.Tensor) -> torch.Tensor:
    """Concat per-head outputs back to a packed channel dim.

    Input ``x`` shape: ``[..., heads, seq, head_dim]``.
    Output shape: ``[..., seq, heads * head_dim]``.
    """
    *prefix, heads, seq, head_dim = x.shape
    return x.transpose(-3, -2).contiguous().reshape(*prefix, seq, heads * head_dim)


# ---------------------------------------------------------------------------
# SDPA
# ---------------------------------------------------------------------------


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Thin wrapper around ``torch.nn.functional.scaled_dot_product_attention``."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)


# ---------------------------------------------------------------------------
# CCL references — semantic equivalents (NOT performance equivalents)
# ---------------------------------------------------------------------------


def reduce_scatter_torch(x: torch.Tensor, *, mesh_axis: int, num_devices: int) -> torch.Tensor:
    """Reference for ``ttnn.reduce_scatter``.

    Semantic: split the (already all-reduced) tensor along ``mesh_axis`` into
    ``num_devices`` equal chunks; each device keeps its chunk. For CPU
    reference we simulate by splitting and stacking back along a new mesh axis.
    """
    chunks = list(torch.chunk(x, num_devices, dim=mesh_axis))
    return torch.stack(chunks, dim=0)


def all_gather_torch(x: torch.Tensor, *, mesh_axis: int, num_devices: int) -> torch.Tensor:
    """Reference for ``ttnn.all_gather``.

    Semantic: replicate ``x`` ``num_devices`` times along ``mesh_axis``. Used
    here only to give a CPU-side comparable; production AG concatenates
    different per-device shards.
    """
    return torch.cat([x] * num_devices, dim=mesh_axis)
