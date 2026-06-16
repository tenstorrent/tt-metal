# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the Flash-Attention scaled_dot_product_attention op.

IMMUTABLE SPEC — the implementer must not modify this file.

Covers the Phase 0 capability surface:
  dtype=bfloat16, layout=TILE, tile-aligned, kv_heads=mha,
  attention_kind in {self, cross}, mask_mode in {none, custom},
  scale_mode in {auto, explicit}.

is_causal (mask_mode=causal) and gqa/mqa are refinements and are NOT
exercised here.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# --- PCC tolerances keyed by dtype (same thresholds as the golden suite) ---
PCC_TOLERANCE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
}


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def make_causal_mask(B, S_q, S_kv, torch_dtype):
    """Broadcast (B,1,S_q,S_kv) additive mask: -inf above the diagonal, 0 elsewhere."""
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch_dtype)
    mask.masked_fill_(
        torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask


def reference_sdpa(Q, K, V, *, attn_mask=None, scale=None):
    """Reference SDPA in fp32; mirrors torch.nn.functional.scaled_dot_product_attention."""
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    D = Qf.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(D)
    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    if attn_mask is not None:
        scores = scores + attn_mask.float()
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, Vf).to(Q.dtype)


# (B, H, S_q, S_kv, D) — minimum 4 shapes: single-tile, multi-tile,
# non-square, multi-batch; plus multi-head and cross-attention.
SHAPES = [
    pytest.param((1, 1, 32, 32, 32), id="single_tile_self"),
    pytest.param((1, 1, 128, 128, 64), id="multi_tile_self"),
    pytest.param((1, 1, 128, 256, 64), id="non_square_cross"),
    pytest.param((2, 4, 128, 128, 64), id="multi_batch_multi_head"),
    pytest.param((1, 8, 256, 256, 64), id="multi_head_self"),
    pytest.param((1, 4, 64, 128, 64), id="cross_sq_lt_skv"),
]

MASK_MODES = [
    pytest.param("none", id="mask_none"),
    pytest.param("custom", id="mask_custom"),
]

SCALE_MODES = [
    pytest.param("auto", id="scale_auto"),
    pytest.param("explicit", id="scale_explicit"),
]

EXPLICIT_SCALE = 0.125


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mask_mode", MASK_MODES)
@pytest.mark.parametrize("scale_mode", SCALE_MODES)
def test_scaled_dot_product_attention(device, shape, mask_mode, scale_mode):
    B, H, S_q, S_kv, D = shape
    dtype = ttnn.bfloat16
    torch_dtype = _TORCH_DTYPE[dtype]

    torch.manual_seed(42)
    Q = torch.randn((B, H, S_q, D), dtype=torch_dtype)
    K = torch.randn((B, H, S_kv, D), dtype=torch_dtype)
    V = torch.randn((B, H, S_kv, D), dtype=torch_dtype)

    if mask_mode == "custom":
        torch_mask = make_causal_mask(B, S_q, S_kv, torch_dtype)
    else:
        torch_mask = None

    scale = EXPLICIT_SCALE if scale_mode == "explicit" else None

    expected = reference_sdpa(Q, K, V, attn_mask=torch_mask, scale=scale)

    ttnn_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_mask = (
        ttnn.from_torch(torch_mask, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        if torch_mask is not None
        else None
    )

    ttnn_output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, attn_mask=ttnn_mask, scale=scale)

    assert list(ttnn_output.shape) == [B, H, S_q, D]
    torch_output = ttnn.to_torch(ttnn_output)

    correlation = pcc(torch_output, expected)
    assert correlation >= PCC_TOLERANCE[dtype], f"PCC too low: {correlation:.6f}"
