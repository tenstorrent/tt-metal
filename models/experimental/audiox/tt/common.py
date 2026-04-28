# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def to_tt(t: torch.Tensor, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh_device)


def linear_weight(w: torch.Tensor) -> torch.Tensor:
    """ttnn.linear expects weight in [in, out] order; PyTorch stores [out, in]."""
    return w.transpose(0, 1).contiguous()


def split_in_proj(in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor):
    """nn.MultiheadAttention packs Q/K/V into a single in_proj_weight of shape [3*dim, dim]."""
    dim = in_proj_weight.shape[1]
    qw, kw, vw = in_proj_weight.split(dim, dim=0)
    qb, kb, vb = in_proj_bias.split(dim, dim=0) if in_proj_bias is not None else (None, None, None)
    return qw, kw, vw, qb, kb, vb


def attention(query, key, value, qw, kw, vw, qb, kb, vb, ow, ob, num_heads):
    """Multi-head attention. query/key/value are ttnn tensors of shape [B, S, D]."""
    q = ttnn.linear(query, qw, bias=qb)
    k = ttnn.linear(key, kw, bias=kb)
    v = ttnn.linear(value, vw, bias=vb)

    batch, sq, dim = q.shape
    sk = k.shape[1]
    head_dim = dim // num_heads

    q = ttnn.reshape(q, (batch, sq, num_heads, head_dim))
    k = ttnn.reshape(k, (batch, sk, num_heads, head_dim))
    v = ttnn.reshape(v, (batch, sk, num_heads, head_dim))

    q = ttnn.transpose(q, 1, 2)
    k = ttnn.transpose(k, 1, 2)
    v = ttnn.transpose(v, 1, 2)

    out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = ttnn.transpose(out, 1, 2)
    out = ttnn.reshape(out, (batch, sq, dim))
    return ttnn.linear(out, ow, bias=ob)
