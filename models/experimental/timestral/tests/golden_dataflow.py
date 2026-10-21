# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch mirror of the TTNN Timestral dataflow (folded weights, single causal pass, blocked variates).

Golden tests check it against the upstream model in float32, which pins down every
rewrite the device port relies on before any device runs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from models.experimental.timestral.tt.preprocessing import TimestralHostInputs, forecast_token_layout
from models.experimental.timestral.tt.weights import (
    AttentionWeights,
    MlpWeights,
    ResidualBlockWeights,
    TimestralWeights,
)


def rms_unit(x: Tensor, eps: float) -> Tensor:
    """RMSNorm without its gain (folded into the next linears)."""
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def residual_block(w: ResidualBlockWeights, x: Tensor) -> Tensor:
    hidden = F.silu(F.linear(x, w.hidden_w, w.hidden_b))
    return F.linear(hidden, w.output_w, w.output_b) + F.linear(x, w.residual_w, w.residual_b)


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def split_heads(qkv: Tensor, num_heads: int) -> tuple[Tensor, Tensor, Tensor]:
    """``(B, S, 3 * H * Dh)`` with rows ``(qkv, heads, head_dim)`` -> three ``(B, H, S, Dh)``."""
    b, s, _ = qkv.shape
    q, k, v = qkv.reshape(b, s, 3, num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
    return q, k, v


def merge_heads(ctx: Tensor) -> Tensor:
    b, h, s, dh = ctx.shape
    return ctx.transpose(1, 2).reshape(b, s, h * dh)


def attention(w: AttentionWeights, x: Tensor, prepared: TimestralHostInputs) -> Tensor:
    xn = rms_unit(x, w.eps)
    gate = torch.sigmoid(F.linear(xn, w.wg))
    rows, seq, d = x.shape
    if w.is_time:
        q, k, v = split_heads(F.linear(xn, w.wqkv), w.num_heads)
        cos, sin = prepared.cos[:, None], prepared.sin[:, None]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        mask = prepared.time_mask
        merged = merge_heads(F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None))
    elif prepared.group_size == 1:
        # Softmax over a single visible key is 1, so the context is exactly V.
        merged = F.linear(xn, w.wv)
    else:
        g = prepared.group_size
        blocks = xn.transpose(0, 1).reshape(seq * rows // g, g, d)
        q, k, v = split_heads(F.linear(blocks, w.wqkv), w.num_heads)
        ctx = merge_heads(F.scaled_dot_product_attention(q, k, v, attn_mask=prepared.variate_mask))
        merged = ctx.reshape(seq, rows, d).transpose(0, 1)
    return F.linear(merged * gate, w.wo)


def mlp(w: MlpWeights, x: Tensor) -> Tensor:
    xn = rms_unit(x, w.eps)
    return F.linear(F.linear(xn, w.w_gate) * F.silu(F.linear(xn, w.w_up)), w.w_down)


def head_forward(weights: TimestralWeights, prepared: TimestralHostInputs) -> Tensor:
    """Normalized-space head output ``(R, h, patch_size * Q)``, before the monotone transform."""
    ll = prepared.num_context_patches
    keep, add = forecast_token_layout(weights.forecast_query, ll, prepared.horizon_patches)
    x = residual_block(weights.patch_proj, prepared.tokens) * keep + add
    for layer in weights.layers:
        for attn in layer.attns:
            x = x + attention(attn, x, prepared)
        x = x + mlp(layer.mlp, x)
    return residual_block(weights.head, rms_unit(x[:, ll:], weights.out_norm_eps))
