# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — non-tile-aligned shape support for Flash-Attention SDPA.

Exercises shapes where the last two logical dims of Q/K/V are not multiples of
32, so the physical TILE_LAYOUT tensor carries a partial last tile:

- ``w_non_aligned`` — head_dim D not %32. The QK contraction and the PV matmul
  include a partial last D-tile; from_torch zero-pads the inputs so the dot
  product over the padded D lanes contributes 0 (no extra handling needed).
- ``h_non_aligned`` — S_q (and, for self/GQA/MQA, S_kv) not %32. The padded KV
  score columns score 0 and would pollute the softmax denominator via
  exp(0 - rowmax); the kernel masks them to -inf before the row-max / row-sum.
  The padded S_q output rows are slipped off by ttnn's logical-shape slicing.
- ``both`` — D and S both non-aligned (the alignment tagger labels this
  ``w_non_aligned`` since it checks D first, but S_kv masking still applies).

The masking is keyed on ``S_kv % 32 != 0``, independent of the alignment tag,
so cross-attention with a non-aligned S_kv but aligned S_q is covered too.

Validated across bfloat16 / float32 / bfloat8_b, mask none/causal, scale
auto/explicit, and MHA / GQA / MQA / cross-attention head configs against the
fp32 PyTorch reference.
"""

from __future__ import annotations

import math

import pytest
import torch

import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# Per-dtype (PCC floor, normalized-RMS ceiling) — matches the golden suite's
# TOLERANCES. Normalized RMS = sqrt(mean(err^2)) / std(reference).
TOLERANCES = {
    ttnn.float32: (0.999, 0.02),
    ttnn.bfloat16: (0.995, 0.05),
    ttnn.bfloat8_b: (0.99, 0.12),
}

_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # no native torch bf8b; build in bf16
}

EXPLICIT_SCALE = 0.125


def pytorch_sdpa_reference(Q, K, V, *, attention_mask=None, scale=None):
    """fp32 reference SDPA with GQA/MQA head broadcasting; returned in Q's dtype."""
    original_dtype = Q.dtype
    Qf, Kf, Vf = Q.to(torch.float32), K.to(torch.float32), V.to(torch.float32)

    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        assert H_q % H_kv == 0
        repeats = H_q // H_kv
        Kf = Kf.repeat_interleave(repeats, dim=1)
        Vf = Vf.repeat_interleave(repeats, dim=1)

    D = Qf.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(D)
    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    if attention_mask is not None:
        scores = scores + attention_mask.to(torch.float32)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, Vf).to(original_dtype)


def make_causal_mask(B, S_q, S_kv, *, torch_dtype):
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch_dtype)
    mask.masked_fill_(
        torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask


def comp_pcc(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    golden = golden.flatten().to(torch.float32)
    calculated = calculated.flatten().to(torch.float32)
    if torch.any(~torch.isfinite(golden)) or torch.any(~torch.isfinite(calculated)):
        finite = torch.isfinite(golden) & torch.isfinite(calculated)
        golden, calculated = golden[finite], calculated[finite]
    if golden.numel() < 2:
        return 1.0
    if torch.allclose(golden, calculated):
        return 1.0
    pcc = torch.corrcoef(torch.stack([golden, calculated]))[0, 1]
    return 0.0 if torch.isnan(pcc) else float(pcc)


def norm_rms(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """sqrt(mean(err^2)) / std(reference) — the golden suite's relative RMS."""
    g = golden.flatten().to(torch.float32)
    c = calculated.flatten().to(torch.float32)
    rms = (g - c).pow(2).mean().sqrt()
    return float(rms / (g.std() + 1e-12))


# (Q_shape, K_shape, V_shape) — K and V share shape. Mirrors the non-aligned
# bucket of the golden feature_spec, plus a couple of extra corners.
NON_ALIGNED_SHAPES = [
    # --- D non-aligned (w_non_aligned); S_kv aligned ---
    pytest.param(((1, 1, 32, 50), (1, 1, 32, 50), (1, 1, 32, 50)), id="w_D50"),
    pytest.param(((1, 8, 64, 47), (1, 8, 64, 47), (1, 8, 64, 47)), id="w_D47_mh"),
    # --- S non-aligned (h_non_aligned); D aligned. self/GQA/MQA => S_kv also non-aligned ---
    pytest.param(((1, 1, 47, 64), (1, 1, 47, 64), (1, 1, 47, 64)), id="h_S47"),
    pytest.param(((1, 4, 47, 64), (1, 4, 47, 64), (1, 4, 47, 64)), id="h_S47_mh"),
    pytest.param(((2, 4, 100, 64), (2, 4, 100, 64), (2, 4, 100, 64)), id="h_S100_mb"),
    pytest.param(((1, 8, 47, 64), (1, 2, 47, 64), (1, 2, 47, 64)), id="h_S47_gqa"),
    pytest.param(((1, 8, 47, 64), (1, 1, 47, 64), (1, 1, 47, 64)), id="h_S47_mqa"),
    # --- both D and S non-aligned ---
    pytest.param(((1, 1, 50, 50), (1, 1, 50, 50), (1, 1, 50, 50)), id="both_50_50"),
    pytest.param(((1, 12, 33, 50), (1, 12, 33, 50), (1, 12, 33, 50)), id="both_33_50_mh"),
    # --- cross-attention with non-aligned S_q != S_kv (and non-aligned D) ---
    pytest.param(((1, 4, 100, 50), (1, 4, 47, 50), (1, 4, 47, 50)), id="both_cross_100_47_50"),
    # --- cross-attention, aligned S_q but non-aligned S_kv (masking still required) ---
    pytest.param(((1, 2, 64, 64), (1, 2, 47, 64), (1, 2, 47, 64)), id="cross_Sq64_Skv47"),
]


@pytest.mark.parametrize("shapes", NON_ALIGNED_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
        pytest.param(ttnn.bfloat8_b, id="bf8b"),
    ],
)
@pytest.mark.parametrize("mask_mode", ["none", "causal"])
@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
def test_sdpa_non_aligned(device, shapes, dtype, mask_mode, scale_mode):
    layout = ttnn.TILE_LAYOUT
    torch_dtype = _TORCH_DTYPE[dtype]

    q_shape, k_shape, v_shape = shapes
    B, _H_q, S_q, _D = q_shape
    S_kv = k_shape[-2]

    torch.manual_seed(0)
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

    # Output carries the logical (non-aligned) shape — ttnn slices the padding.
    assert list(ttnn_out.shape) == list(q_shape), f"shape mismatch: {ttnn_out.shape} vs {q_shape}"
    assert ttnn_out.dtype == dtype
    assert ttnn_out.layout == layout

    torch_out = ttnn.to_torch(ttnn_out)
    assert torch.all(torch.isfinite(torch_out.to(torch.float32))), "non-finite values in output"

    pcc = comp_pcc(expected, torch_out)
    rms = norm_rms(expected, torch_out)
    pcc_floor, rms_ceil = TOLERANCES[dtype]
    assert pcc >= pcc_floor and rms <= rms_ceil, (
        f"PCC {pcc:.5f} (floor {pcc_floor}) / RMS {rms:.4f} (ceil {rms_ceil}) "
        f"shapes={shapes} dtype={dtype} mask={mask_mode} scale={scale_mode}"
    )


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
        pytest.param(ttnn.bfloat8_b, id="bf8b"),
    ],
)
def test_sdpa_aligned_no_regression(device, dtype):
    """The kv_partial==0 path must compile out the pad mask — aligned still works."""
    layout = ttnn.TILE_LAYOUT
    torch_dtype = _TORCH_DTYPE[dtype]
    shape = (1, 4, 128, 64)

    torch.manual_seed(0)
    Q = torch.randn(shape, dtype=torch_dtype)
    K = torch.randn(shape, dtype=torch_dtype)
    V = torch.randn(shape, dtype=torch_dtype)
    expected = pytorch_sdpa_reference(Q, K, V)

    def to_dev(t):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    ttnn_out = scaled_dot_product_attention(to_dev(Q), to_dev(K), to_dev(V))
    torch_out = ttnn.to_torch(ttnn_out)
    pcc = comp_pcc(expected, torch_out)
    assert pcc >= TOLERANCES[dtype][0], f"aligned regression: PCC {pcc:.5f} dtype={dtype}"
