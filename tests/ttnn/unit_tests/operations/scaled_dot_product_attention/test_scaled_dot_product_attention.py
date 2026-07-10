# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the Flash-Attention scaled_dot_product_attention op.

Immutable spec — the implementer must NOT modify this file. It pins the Phase-0
contract: bfloat16, TILE_LAYOUT, self/cross attention, mask_mode in {none, custom},
scale_mode in {auto, explicit}. Reference is torch.nn.functional.scaled_dot_product_
attention computed in fp32.

The `device` fixture comes from the shared conftest (module-scoped via the local
conftest.py marker); do NOT open a device manually.
"""

import math

import pytest
import torch

import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# PCC tolerance keyed by dtype (same thresholds as the golden suite).
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
}


def _pcc(golden: torch.Tensor, computed: torch.Tensor) -> float:
    a = golden.flatten().to(torch.float32)
    b = computed.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    if a.std() == 0 or b.std() == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _torch_reference(Q, K, V, *, attn_mask=None, is_causal=False, scale=None):
    Qf, Kf, Vf = Q.to(torch.float32), K.to(torch.float32), V.to(torch.float32)
    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        repeats = H_q // H_kv
        Kf = Kf.repeat_interleave(repeats, dim=1)
        Vf = Vf.repeat_interleave(repeats, dim=1)
    am = attn_mask.to(torch.float32) if attn_mask is not None else None
    return torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=am, is_causal=is_causal, scale=scale)


def _make_custom_mask(B, S_q, S_kv, torch_dtype):
    """Upper-triangular additive mask (0 keep, -inf drop) broadcast over heads."""
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch_dtype)
    mask.masked_fill_(
        torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask


# (Q_shape, K_shape, V_shape) — K and V share shape. Covers single-tile,
# multi-tile, non-square (S != D), multi-batch/head, GQA/MQA and cross-attention.
SHAPES = [
    ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),  # single tile
    ((1, 1, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),  # multi-tile
    ((1, 2, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)),  # multi-head, non-square
    ((2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64)),  # multi-batch
    ((1, 4, 128, 64), (1, 4, 64, 64), (1, 4, 64, 64)),  # cross-attention S_q > S_kv
    ((1, 8, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)),  # GQA 4:1
    ((1, 8, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),  # MQA
]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("q_shape, k_shape, v_shape", SHAPES)
@pytest.mark.parametrize("mask_mode", ["none", "custom"])
@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
def test_scaled_dot_product_attention(device, dtype, q_shape, k_shape, v_shape, mask_mode, scale_mode):
    torch.manual_seed(42)
    torch_dtype = _TORCH_DTYPE[dtype]

    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(k_shape, dtype=torch_dtype)
    V = torch.randn(v_shape, dtype=torch_dtype)

    if mask_mode == "custom":
        B, _, S_q, _ = q_shape
        S_kv = k_shape[-2]
        torch_mask = _make_custom_mask(B, S_q, S_kv, torch_dtype)
    else:
        torch_mask = None

    scale = 0.125 if scale_mode == "explicit" else None

    expected = _torch_reference(Q, K, V, attn_mask=torch_mask, is_causal=False, scale=scale)

    to_dev = lambda t: ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_Q, ttnn_K, ttnn_V = to_dev(Q), to_dev(K), to_dev(V)
    ttnn_mask = to_dev(torch_mask) if torch_mask is not None else None

    ttnn_out = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, attn_mask=ttnn_mask, is_causal=False, scale=scale)
    out = ttnn.to_torch(ttnn_out).to(torch.float32)

    assert list(out.shape) == list(q_shape), f"shape {tuple(out.shape)} != {q_shape}"
    pcc = _pcc(expected, out)
    assert pcc >= PCC[dtype], f"PCC {pcc:.5f} < {PCC[dtype]} (dtype={dtype}, mask={mask_mode}, scale={scale_mode})"
