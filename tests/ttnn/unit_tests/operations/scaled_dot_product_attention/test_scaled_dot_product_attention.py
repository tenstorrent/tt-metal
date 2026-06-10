# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for Flash-Attention scaled_dot_product_attention.

IMMUTABLE SPEC — the implementer must not modify this file.

Reference: O = softmax(Q @ K^T * scale + mask, dim=-1) @ V
PCC thresholds: float32 0.999, bfloat16 0.995, bfloat8_b 0.99.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}


def torch_sdpa(q, k, v, mask=None, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def causal_mask(B, S_q, S_kv, dtype=torch.float32):
    m = torch.zeros(B, 1, S_q, S_kv, dtype=dtype)
    m.masked_fill_(torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1), float("-inf"))
    return m


def compute_pcc(golden, actual):
    g = golden.flatten().float()
    a = actual.flatten().float()
    if torch.allclose(g, a, atol=1e-8):
        return 1.0
    return torch.corrcoef(torch.stack([g, a]))[0, 1].item()


def run_case(device, q_shape, kv_shape, mask_mode, scale, dtype=ttnn.bfloat16):
    torch.manual_seed(42)
    q = torch.randn(q_shape)
    k = torch.randn(kv_shape)
    v = torch.randn(kv_shape)
    mask = causal_mask(q_shape[0], q_shape[2], kv_shape[2]) if mask_mode == "causal" else None

    golden = torch_sdpa(q, k, v, mask=mask, scale=scale)

    tt_q = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = {}
    if mask is not None:
        kwargs["attention_mask"] = ttnn.from_torch(mask, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if scale is not None:
        kwargs["scale"] = scale

    out = ttnn.to_torch(scaled_dot_product_attention(tt_q, tt_k, tt_v, **kwargs))

    assert out.shape == golden.shape
    pcc = compute_pcc(golden, out)
    assert (
        pcc >= PCC[dtype]
    ), f"PCC {pcc:.5f} < {PCC[dtype]} for q={q_shape} kv={kv_shape} mask={mask_mode} scale={scale}"


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 1, 32, 32), (1, 1, 32, 32)),  # single tile
        ((1, 1, 128, 64), (1, 1, 128, 64)),  # multi-tile
        ((1, 8, 256, 64), (1, 8, 256, 64)),  # multi-head, non-square S x D
        ((2, 4, 128, 64), (2, 4, 128, 64)),  # multi-batch
        ((1, 1, 128, 256), (1, 1, 128, 256)),  # wide head_dim
    ],
    ids=["1tile", "128x64", "h8_256x64", "b2h4", "d256"],
)
@pytest.mark.parametrize("mask_mode", ["none", "causal"])
def test_sdpa_self_attention(device, q_shape, kv_shape, mask_mode):
    run_case(device, q_shape, kv_shape, mask_mode, scale=None)


@pytest.mark.parametrize(
    "q_shape, kv_shape",
    [
        ((1, 4, 64, 64), (1, 4, 128, 64)),  # S_q < S_kv
        ((1, 4, 128, 64), (1, 4, 64, 64)),  # S_q > S_kv
    ],
    ids=["q_lt_kv", "q_gt_kv"],
)
def test_sdpa_cross_attention(device, q_shape, kv_shape):
    run_case(device, q_shape, kv_shape, mask_mode="none", scale=None)


@pytest.mark.parametrize("scale", [0.125, 0.1])
def test_sdpa_explicit_scale(device, scale):
    run_case(device, (1, 4, 128, 64), (1, 4, 128, 64), mask_mode="none", scale=scale)


def test_sdpa_explicit_scale_with_mask(device):
    run_case(device, (1, 2, 128, 64), (1, 2, 128, 64), mask_mode="causal", scale=0.1)


@pytest.mark.parametrize("seq_len", [2048, 4096])
def test_sdpa_long_context(device, seq_len):
    # Flash Attention's load-bearing case — full S x S scores (>= 16 MB) cannot fit in L1.
    run_case(device, (1, 1, seq_len, 64), (1, 1, seq_len, 64), mask_mode="causal", scale=None)


def test_sdpa_shape_validation(device):
    torch.manual_seed(42)
    good = lambda s: ttnn.from_torch(torch.randn(s), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    q = good((1, 2, 64, 64))
    with pytest.raises((ValueError, RuntimeError)):
        scaled_dot_product_attention(q, good((1, 2, 64, 32)), good((1, 2, 64, 32)))  # head_dim mismatch
    with pytest.raises((ValueError, RuntimeError)):
        scaled_dot_product_attention(q, good((1, 2, 64, 64)), good((1, 2, 128, 64)))  # K/V seq mismatch
    with pytest.raises((ValueError, RuntimeError)):
        scaled_dot_product_attention(q, good((2, 2, 64, 64)), good((2, 2, 64, 64)))  # batch mismatch
    with pytest.raises((ValueError, RuntimeError)):
        scaled_dot_product_attention(q, good((1, 2, 64, 64)), good((1, 2, 64, 64)), attention_mask=good((1, 1, 32, 64)))
