# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test (immutable spec) for scaled_dot_product_attention (Flash Attention).

This is the contract the implementer must satisfy. DO NOT MODIFY.

Scope = Phase-0 SUPPORTED: dtype=bfloat16, layout=TILE, tile-aligned shapes,
mask_mode ∈ {none, custom}, scale_mode ∈ {auto, explicit}, attention_kind ∈
{self, cross}, kv_heads_mode ∈ {mha, gqa, mqa}. Causal masking and lower-precision
dtypes are refinements and are intentionally not exercised here so that a correct
Phase-0 op passes this file in full.

Reference mirrors torch.nn.functional.scaled_dot_product_attention:
    scores  = Q @ K^T * scale
    scores += attn_mask            # additive (custom mode)
    weights = softmax(scores, -1)
    output  = weights @ V
with K/V heads repeat-interleaved to Q heads for GQA/MQA.
"""

import math

import pytest
import torch

import ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
from tests.ttnn.utils_for_testing import assert_with_pcc


# PCC tolerances keyed by dtype — identical to the golden suite; not derived from
# op "complexity".
PCC_BY_DTYPE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}


def attention_reference(q, k, v, attn_mask=None, scale=None):
    """fp32 PyTorch reference. q:(B,H,Sq,D)  k,v:(B,Hkv,Skv,D)."""
    B, H, Sq, D = q.shape
    Hkv = k.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q = q.float()
    k = k.float()
    v = v.float()

    if Hkv != H:  # GQA / MQA: broadcast the shared KV heads up to Q heads
        rep = H // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask.float()
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def _to_device(t, device, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def run_sdpa(device, q_shape, k_shape, v_shape, *, dtype, mask_mode, scale_mode):
    torch.manual_seed(42)

    q = torch.randn(q_shape)
    k = torch.randn(k_shape)
    v = torch.randn(v_shape)

    D = q_shape[-1]
    # Explicit scale deliberately differs from 1/sqrt(D) (0.125 for D=64) so a
    # test failure catches an op that ignores the `scale` kwarg.
    scale = None if scale_mode == "auto" else 0.25

    attn_mask = None
    tt_mask = None
    if mask_mode == "custom":
        # Arbitrary additive mask broadcast over heads: (B, 1, Sq, Skv). Moderate
        # magnitude so a dropped mask-add would visibly change the softmax weights.
        B, _, Sq, _ = q_shape
        Skv = k_shape[-2]
        attn_mask = torch.randn(B, 1, Sq, Skv) * 2.0
        tt_mask = _to_device(attn_mask, device, dtype)

    ref = attention_reference(q, k, v, attn_mask=attn_mask, scale=scale)

    tq = _to_device(q, device, dtype)
    tk = _to_device(k, device, dtype)
    tv = _to_device(v, device, dtype)

    out = scaled_dot_product_attention(tq, tk, tv, attn_mask=tt_mask, scale=scale)

    assert list(out.shape) == list(q_shape), f"output shape {list(out.shape)} != {list(q_shape)}"
    assert_with_pcc(ref, ttnn.to_torch(out), PCC_BY_DTYPE[dtype])


# (Q_shape, K_shape, V_shape) covering: single-tile, multi-tile, non-square
# (cross-attention S_q != S_kv), multi-batch, and GQA/MQA reduced KV heads.
SHAPES = {
    "single_tile_self": ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
    "multi_tile_self": ((1, 1, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),
    "multi_head_batch": ((2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64)),
    "cross_attention": ((1, 8, 256, 64), (1, 8, 128, 64), (1, 8, 128, 64)),
    "gqa_4to1": ((1, 8, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)),
    "mqa": ((1, 8, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),
    "longer_self": ((1, 2, 512, 64), (1, 2, 512, 64), (1, 2, 512, 64)),
}


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("shape_key", list(SHAPES.keys()))
def test_sdpa_shapes(device, shape_key, dtype):
    q_shape, k_shape, v_shape = SHAPES[shape_key]
    run_sdpa(device, q_shape, k_shape, v_shape, dtype=dtype, mask_mode="none", scale_mode="auto")


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "shape_key",
    ["single_tile_self", "multi_tile_self", "cross_attention", "gqa_4to1"],
)
def test_sdpa_custom_mask(device, shape_key, dtype):
    q_shape, k_shape, v_shape = SHAPES[shape_key]
    run_sdpa(device, q_shape, k_shape, v_shape, dtype=dtype, mask_mode="custom", scale_mode="auto")


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("shape_key", ["multi_tile_self", "multi_head_batch"])
def test_sdpa_explicit_scale(device, shape_key, dtype):
    q_shape, k_shape, v_shape = SHAPES[shape_key]
    run_sdpa(device, q_shape, k_shape, v_shape, dtype=dtype, mask_mode="none", scale_mode="explicit")
