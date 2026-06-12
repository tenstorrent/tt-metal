# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for scaled_dot_product_attention (Flash Attention).

This file is the immutable spec — the implementer MUST NOT modify it.

Reference: torch.nn.functional.scaled_dot_product_attention. The op MUST use
the Flash Attention algorithm (tiled, online softmax, O(S) memory) but the
externally observable result must match the reference within PCC tolerance.

PCC tolerances are keyed by dtype:
    float32     -> 0.999
    bfloat16    -> 0.995
    bfloat8_b   -> 0.99
"""

import math

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


PCC_BY_DTYPE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}


def _torch_reference(q, k, v, attn_mask=None, is_causal=False, scale=None):
    """Mirror torch.nn.functional.scaled_dot_product_attention."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        s_q, s_kv = q.shape[-2], k.shape[-2]
        causal = torch.full((s_q, s_kv), float("-inf"), dtype=scores.dtype)
        causal = torch.triu(causal, diagonal=1)
        scores = scores + causal
    elif attn_mask is not None:
        scores = scores + attn_mask
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


# (B, H, S_q, D) Q-shape ; K/V share (B, H, S_kv, D)
# Covers: single-tile, multi-tile, non-square (S != D), multi-batch, cross-attention.
SHAPES = [
    # (q_shape, kv_shape)
    ((1, 1, 32, 32), (1, 1, 32, 32)),  # single tile, self
    ((1, 8, 128, 64), (1, 8, 128, 64)),  # multi-head multi-tile, self
    ((2, 8, 256, 64), (2, 8, 256, 64)),  # multi-batch, self
    ((2, 16, 512, 128), (2, 16, 512, 128)),  # non-square, head_dim=128, self
    ((1, 8, 128, 64), (1, 8, 256, 64)),  # cross-attention S_q < S_kv
    ((1, 8, 256, 64), (1, 8, 128, 64)),  # cross-attention S_q > S_kv
]


@pytest.mark.parametrize("q_shape, kv_shape", SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("mask_mode", ["none", "custom"])
@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
def test_scaled_dot_product_attention(device, q_shape, kv_shape, dtype, layout, mask_mode, scale_mode):
    torch.manual_seed(42)

    b, h, s_q, d = q_shape
    _, _, s_kv, _ = kv_shape

    q = torch.randn(q_shape, dtype=torch.float32)
    k = torch.randn(kv_shape, dtype=torch.float32)
    v = torch.randn(kv_shape, dtype=torch.float32)

    scale = 0.125 if scale_mode == "explicit" else None

    torch_mask = None
    if mask_mode == "custom":
        # Arbitrary additive mask broadcast across heads: (B, 1, S_q, S_kv).
        torch_mask = torch.zeros((b, 1, s_q, s_kv), dtype=torch.float32)
        # Mask out a band of key positions so the mask is non-trivial.
        torch_mask[..., : s_kv // 4] = float("-inf")

    torch_out = _torch_reference(q, k, v, attn_mask=torch_mask, is_causal=False, scale=scale)

    tt_q = ttnn.from_torch(q, dtype=dtype, layout=layout, device=device)
    tt_k = ttnn.from_torch(k, dtype=dtype, layout=layout, device=device)
    tt_v = ttnn.from_torch(v, dtype=dtype, layout=layout, device=device)

    tt_mask = None
    if torch_mask is not None:
        tt_mask = ttnn.from_torch(torch_mask, dtype=dtype, layout=layout, device=device)

    tt_out = scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        attn_mask=tt_mask,
        is_causal=False,
        scale=scale,
    )

    out = ttnn.to_torch(tt_out)

    assert out.shape == torch_out.shape
    assert_with_pcc(torch_out, out, PCC_BY_DTYPE[dtype])
