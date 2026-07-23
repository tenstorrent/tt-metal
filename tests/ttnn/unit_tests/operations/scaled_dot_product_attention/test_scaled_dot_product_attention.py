# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for scaled_dot_product_attention (Flash Attention).

This file is the IMMUTABLE spec — the implementer must not modify it.

Covers the Phase-0 support contract: bfloat16, TILE_LAYOUT, tile-aligned
shapes, self + cross attention, MHA/GQA/MQA head modes, mask_mode ∈
{none, custom}, scale_mode ∈ {auto, explicit}. Reference is
torch.nn.functional.scaled_dot_product_attention computed in fp32.
"""

import math

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# PCC thresholds keyed by dtype — identical to the golden suite.
PCC = {
    torch.float32: 0.999,
    torch.bfloat16: 0.995,
}

EXPLICIT_SCALE = 0.125


def _reference(Q, K, V, *, attn_mask=None, is_causal=False, scale=None):
    """fp32 reference mirroring the op contract (with GQA/MQA head broadcast)."""
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        assert H_q % H_kv == 0
        r = H_q // H_kv
        Kf = Kf.repeat_interleave(r, dim=1)
        Vf = Vf.repeat_interleave(r, dim=1)
    am = attn_mask.float() if attn_mask is not None else None
    return torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=am, is_causal=is_causal, scale=scale)


def _triangular_mask(B, S_q, S_kv, torch_dtype):
    """Additive (B,1,S_q,S_kv) mask: 0 on/below diagonal, -inf above."""
    m = torch.zeros(B, 1, S_q, S_kv, dtype=torch_dtype)
    m.masked_fill_(torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1), float("-inf"))
    return m


# (Q_shape, K_shape, V_shape) — single-tile, multi-tile, non-square,
# multi-batch, multi-head, cross-attention, GQA, MQA.
SHAPES = [
    ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),  # single tile
    ((1, 1, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),  # multi-tile self
    ((1, 8, 256, 64), (1, 8, 256, 64), (1, 8, 256, 64)),  # multi-head
    ((2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64)),  # multi-batch
    ((1, 4, 64, 64), (1, 4, 128, 64), (1, 4, 128, 64)),  # cross-attn S_q < S_kv
    ((1, 4, 128, 128), (1, 4, 128, 128), (1, 4, 128, 128)),  # non-square D
    ((1, 8, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)),  # GQA 4:1
    ((1, 8, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),  # MQA
]


def _run(device, shapes, dtype, mask_mode, scale_mode):
    torch.manual_seed(42)
    q_shape, k_shape, v_shape = shapes
    Q = torch.randn(q_shape, dtype=dtype)
    K = torch.randn(k_shape, dtype=dtype)
    V = torch.randn(v_shape, dtype=dtype)

    B, _H, S_q, D = q_shape
    S_kv = k_shape[-2]

    attn_mask = _triangular_mask(B, S_q, S_kv, dtype) if mask_mode == "custom" else None
    scale = EXPLICIT_SCALE if scale_mode == "explicit" else None

    expected = _reference(Q, K, V, attn_mask=attn_mask, scale=scale)

    tt_dtype = ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32
    to_dev = lambda t: ttnn.from_torch(
        t,
        dtype=tt_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_q, tt_k, tt_v = to_dev(Q), to_dev(K), to_dev(V)
    tt_mask = to_dev(attn_mask) if attn_mask is not None else None

    tt_out = scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        attn_mask=tt_mask,
        scale=scale,
    )

    out = ttnn.to_torch(tt_out).to(torch.float32)
    assert list(out.shape) == list(q_shape)
    assert_with_pcc(expected, out, PCC[dtype])


@pytest.mark.parametrize("shapes", SHAPES)
@pytest.mark.parametrize("mask_mode", ["none", "custom"])
@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
def test_sdpa(device, shapes, mask_mode, scale_mode):
    _run(device, shapes, torch.bfloat16, mask_mode, scale_mode)
