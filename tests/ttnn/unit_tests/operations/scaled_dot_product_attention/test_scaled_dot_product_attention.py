# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the Flash Attention scaled_dot_product_attention op.

This file is the immutable specification. The implementer MUST NOT modify it.

Contract under test (Flash Attention SDPA):
    scores  = Q @ K^T * scale            # scale = 1/sqrt(D) when scale is None
    if mask: scores = scores + attention_mask   # additive (0 attend, -inf mask)
    weights = softmax(scores, dim=-1)
    output  = weights @ V

Q:    (B, H_q,  S_q,  D)
K,V:  (B, H_kv, S_kv, D)   # H_q % H_kv == 0 (MHA/GQA/MQA); S_kv may differ from S_q
out:  (B, H_q,  S_q,  D)

The output must match the fp32 PyTorch reference within a per-dtype PCC
threshold. The Flash Attention tiling is an internal implementation detail —
this test only checks mathematical equivalence to standard SDPA.
"""

from __future__ import annotations

import math

import pytest
import torch

import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# --------------------------------------------------------------------------- #
# Reference + helpers
# --------------------------------------------------------------------------- #

# Same thresholds as the shared golden suite — keyed by dtype, not by op-class.
PCC_THRESHOLD = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # no native torch bf8b; build in bf16
}

EXPLICIT_SCALE = 0.125


def pytorch_sdpa_reference(Q, K, V, *, attention_mask=None, scale=None):
    """Reference SDPA computed in fp32, returned in Q's dtype.

    Handles MHA (H_q == H_kv), GQA (1 < H_kv < H_q), and MQA (H_kv == 1) by
    head-broadcasting K/V via repeat_interleave, matching the op contract.
    """
    original_dtype = Q.dtype
    Qf = Q.to(torch.float32)
    Kf = K.to(torch.float32)
    Vf = V.to(torch.float32)

    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        assert H_q % H_kv == 0, f"H_q ({H_q}) must be a multiple of H_kv ({H_kv})"
        repeats = H_q // H_kv
        Kf = Kf.repeat_interleave(repeats, dim=1)
        Vf = Vf.repeat_interleave(repeats, dim=1)

    D = Qf.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(D)

    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    if attention_mask is not None:
        scores = scores + attention_mask.to(torch.float32)
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, Vf)
    return output.to(original_dtype)


def make_causal_mask(B, S_q, S_kv, *, torch_dtype):
    """Broadcast (B, 1, S_q, S_kv) additive causal mask: -inf above the diagonal."""
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch_dtype)
    mask.masked_fill_(
        torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask


def comp_pcc(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """Pearson correlation between reference and computed tensors (fp32)."""
    golden = golden.flatten().to(torch.float32)
    calculated = calculated.flatten().to(torch.float32)

    # Replace any non-finite values consistently so a stray inf/nan does not
    # poison the correlation (the reference never produces them for these inputs).
    if torch.any(~torch.isfinite(golden)) or torch.any(~torch.isfinite(calculated)):
        finite = torch.isfinite(golden) & torch.isfinite(calculated)
        golden = golden[finite]
        calculated = calculated[finite]

    if golden.numel() < 2:
        return 1.0
    if torch.allclose(golden, calculated):
        return 1.0

    cov = torch.stack([golden, calculated])
    pcc = torch.corrcoef(cov)[0, 1]
    if torch.isnan(pcc):
        # Both constant and equal would have hit allclose above; treat as fail.
        return 0.0
    return float(pcc)


# --------------------------------------------------------------------------- #
# Shapes: (Q_shape, K_shape, V_shape). K and V always share shape.
# --------------------------------------------------------------------------- #

SHAPES = [
    # single-tile MHA self-attention (S=32, D=64)
    pytest.param(((1, 1, 32, 64), (1, 1, 32, 64), (1, 1, 32, 64)), id="single_tile"),
    # multi-tile MHA self-attention (S=128, D=64)
    pytest.param(((1, 1, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)), id="multi_tile"),
    # non-square head config (multi-head, S=256, D=128)
    pytest.param(((1, 4, 256, 128), (1, 4, 256, 128), (1, 4, 256, 128)), id="non_square"),
    # multi-batch multi-head
    pytest.param(((2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64)), id="multi_batch"),
    # longer sequence (Flash Attention's home turf)
    pytest.param(((1, 2, 512, 64), (1, 2, 512, 64), (1, 2, 512, 64)), id="long_seq"),
    # cross-attention: S_q != S_kv
    pytest.param(((1, 4, 128, 64), (1, 4, 256, 64), (1, 4, 256, 64)), id="cross_attn"),
    # GQA: H_q=8, H_kv=2
    pytest.param(((1, 8, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)), id="gqa"),
    # MQA: H_kv=1
    pytest.param(((1, 8, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)), id="mqa"),
]


@pytest.mark.parametrize("shapes", SHAPES)
@pytest.mark.parametrize("mask_mode", ["none", "causal"])
@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
def test_scaled_dot_product_attention(device, shapes, mask_mode, scale_mode):
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    torch_dtype = _TORCH_DTYPE[dtype]

    q_shape, k_shape, v_shape = shapes
    B, _H_q, S_q, _D = q_shape
    S_kv = k_shape[-2]

    torch.manual_seed(42)
    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(k_shape, dtype=torch_dtype)
    V = torch.randn(v_shape, dtype=torch_dtype)

    torch_mask = make_causal_mask(B, S_q, S_kv, torch_dtype=torch_dtype) if mask_mode == "causal" else None
    scale = EXPLICIT_SCALE if scale_mode == "explicit" else None

    expected = pytorch_sdpa_reference(Q, K, V, attention_mask=torch_mask, scale=scale)

    def to_dev(t):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    ttnn_Q, ttnn_K, ttnn_V = to_dev(Q), to_dev(K), to_dev(V)
    ttnn_mask = to_dev(torch_mask) if torch_mask is not None else None

    ttnn_out = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, attention_mask=ttnn_mask, scale=scale)

    assert list(ttnn_out.shape) == list(q_shape), f"shape mismatch: {ttnn_out.shape} vs {q_shape}"
    assert ttnn_out.dtype == dtype
    assert ttnn_out.layout == layout

    torch_out = ttnn.to_torch(ttnn_out)
    pcc = comp_pcc(expected, torch_out)
    assert pcc >= PCC_THRESHOLD[dtype], (
        f"PCC {pcc:.5f} below threshold {PCC_THRESHOLD[dtype]} "
        f"(shapes={shapes}, mask={mask_mode}, scale={scale_mode})"
    )
