# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the Flash-Attention scaled_dot_product_attention op.

IMMUTABLE SPEC — the implementer must not modify this file. It pins the
Phase-0 support contract (bfloat16, TILE_LAYOUT, tile-aligned, MHA,
self/cross attention, none/causal mask, auto/explicit scale) and the
numerical tolerance.

The op MUST use the Flash-Attention algorithm (tiled online softmax,
O(S) memory) — that constraint is graded by the shared golden suite at
eval/golden_tests/scaled_dot_product_attention/, not re-asserted here.
This file only checks mathematical correctness against a fp32 PyTorch
reference, which is invariant to whether the kernel is two-pass SDPA or
Flash Attention.
"""

from __future__ import annotations

import math

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# PCC tolerance keyed by dtype — same thresholds as the shared golden suite.
PCC = {
    torch.float32: 0.999,
    torch.bfloat16: 0.995,
}


def pytorch_sdpa_reference(Q, K, V, *, attention_mask=None, scale=None):
    """Reference SDPA computed in fp32, returned in the input dtype.

    Mirrors eval/golden_tests/.../helpers.py: softmax(Q·Kᵀ·scale + mask)·V,
    scale defaults to 1/sqrt(D); K/V head-broadcast for GQA/MQA (Phase 0
    uses MHA, so this is a no-op here).
    """
    original_dtype = Q.dtype
    Qf, Kf, Vf = Q.float(), K.float(), V.float()

    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        repeats = H_q // H_kv
        Kf = Kf.repeat_interleave(repeats, dim=1)
        Vf = Vf.repeat_interleave(repeats, dim=1)

    D = Qf.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(D)

    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    if attention_mask is not None:
        scores = scores + attention_mask.float()
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, Vf)
    return output.to(original_dtype)


def make_causal_mask(B, S_q, S_kv, *, torch_dtype=torch.bfloat16):
    """(B,1,S_q,S_kv) additive mask — upper triangle (above diagonal) is -inf."""
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch_dtype)
    mask.masked_fill_(
        torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask


def _to_device(t, device):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run(Q, K, V, device, *, attention_mask=None, scale=None):
    expected = pytorch_sdpa_reference(Q, K, V, attention_mask=attention_mask, scale=scale)

    ttnn_Q = _to_device(Q, device)
    ttnn_K = _to_device(K, device)
    ttnn_V = _to_device(V, device)
    ttnn_mask = _to_device(attention_mask, device) if attention_mask is not None else None

    ttnn_out = scaled_dot_product_attention(
        ttnn_Q,
        ttnn_K,
        ttnn_V,
        attention_mask=ttnn_mask,
        scale=scale,
    )
    actual = ttnn.to_torch(ttnn_out).to(torch.bfloat16)

    assert list(actual.shape) == list(expected.shape)
    assert_with_pcc(expected.float(), actual.float(), PCC[torch.bfloat16])


# (B, H, S, D) — single-tile, multi-tile, non-square (S != D), multi-head, multi-batch.
SELF_ATTN_SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 128, 64),  # multi-tile, non-square
    (1, 1, 256, 64),  # longer sequence
    (1, 4, 128, 64),  # multi-head
    (2, 4, 128, 64),  # multi-head + multi-batch
    (1, 8, 128, 128),  # wider head_dim
]


@pytest.mark.parametrize("shape", SELF_ATTN_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_self_attention_no_mask_auto_scale(shape, device):
    torch.manual_seed(42)
    B, H, S, D = shape
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    _run(Q, K, V, device)


@pytest.mark.parametrize("shape", SELF_ATTN_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_self_attention_explicit_scale(shape, device):
    torch.manual_seed(42)
    B, H, S, D = shape
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    _run(Q, K, V, device, scale=0.125)


@pytest.mark.parametrize("shape", SELF_ATTN_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_self_attention_causal_mask(shape, device):
    torch.manual_seed(42)
    B, H, S, D = shape
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    mask = make_causal_mask(B, S, S)
    _run(Q, K, V, device, attention_mask=mask)


# (B, H, S_q, S_kv, D) — cross attention: S_q != S_kv.
CROSS_ATTN_SHAPES = [
    (1, 4, 64, 128, 64),  # S_q < S_kv
    (1, 4, 128, 64, 64),  # S_q > S_kv
    (2, 4, 64, 256, 64),  # batched, S_q << S_kv
]


@pytest.mark.parametrize("shape", CROSS_ATTN_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_cross_attention_no_mask(shape, device):
    torch.manual_seed(42)
    B, H, S_q, S_kv, D = shape
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    _run(Q, K, V, device)


@pytest.mark.parametrize("shape", CROSS_ATTN_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_cross_attention_explicit_scale(shape, device):
    torch.manual_seed(42)
    B, H, S_q, S_kv, D = shape
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    _run(Q, K, V, device, scale=0.1)
