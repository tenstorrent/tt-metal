# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision baseline for scaled_dot_product_attention.

Measures PCC, max abs error, mean abs error, and relative RMS error
across a few representative Phase 0 shapes. Captures the numerical
fidelity floor for the current SUPPORTED rectangle (bf16, TILE, MHA,
tile-aligned). Used as the reference for verifier_report.md and as the
yardstick when refinements (numeric configurability, longer context,
sharding, ...) land — a refinement that drops these numbers without
explanation is a regression.

Run with:
    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_precision_baseline.py
"""

from __future__ import annotations

import math

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _torch_sdpa(q, k, v, scale=None):
    """fp32 reference SDPA — same recipe as the acceptance test, no mask."""
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    s = scale if scale is not None else 1.0 / math.sqrt(qf.shape[-1])
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * s
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, vf)


def _measure(actual_f32: torch.Tensor, expected_f32: torch.Tensor):
    """PCC + abs-error scalars + relative RMS error."""
    a = actual_f32.flatten()
    e = expected_f32.flatten()
    pcc = torch.corrcoef(torch.stack([a, e]))[0, 1].item()
    diff = (a - e).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    abs_rms = torch.sqrt(((a - e) ** 2).mean()).item()
    scale = e.std().item()
    rel_rms = abs_rms / scale if scale > 1e-12 else abs_rms
    return pcc, max_abs, mean_abs, rel_rms


# (B, H, S_q, S_kv, D). Spread across:
#   - single-tile baseline
#   - one moderate multi-tile shape
#   - one multi-head shape (transformer-realistic)
#   - one larger shape that stresses the K-loop length
BASELINE_SHAPES = [
    pytest.param(1, 1, 32, 32, 64, id="small_b1_h1_s32_d64"),
    pytest.param(1, 1, 128, 128, 64, id="medium_b1_h1_s128_d64"),
    pytest.param(1, 4, 128, 128, 64, id="multihead_b1_h4_s128_d64"),
    pytest.param(1, 1, 256, 256, 64, id="larger_b1_h1_s256_d64"),
]


@pytest.mark.parametrize("B,H,S_q,S_kv,D", BASELINE_SHAPES)
def test_sdpa_precision_baseline(device, B, H, S_q, S_kv, D):
    """Measure PCC + abs/RMS error for the Phase 0 bf16 path."""
    torch.manual_seed(0)
    q_t = torch.randn(B, H, S_q, D, dtype=torch.bfloat16) * 0.3
    k_t = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16) * 0.3
    v_t = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16) * 0.3

    expected_f32 = _torch_sdpa(q_t, k_t, v_t)

    q = ttnn.from_torch(q_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(k_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(v_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    actual = ttnn.to_torch(scaled_dot_product_attention(q, k, v)).to(torch.float32)

    pcc, max_abs, mean_abs, rel_rms = _measure(actual, expected_f32)

    # comp_allclose returns a string like "Allclose: True/False; max_abs=...; rtol=..."
    # — log it so the numbers land in the test report.
    allclose_summary = comp_allclose(expected_f32, actual)
    print(
        f"[precision-baseline] shape=(B={B},H={H},S_q={S_q},S_kv={S_kv},D={D}) "
        f"pcc={pcc:.6f} max_abs={max_abs:.4g} mean_abs={mean_abs:.4g} rel_rms={rel_rms:.4g} "
        f"{allclose_summary}"
    )

    # Gate at the Phase 0 envelope (matches the per-dtype TOLERANCES for bf16
    # used by the golden harness — pcc>=0.995, rel_rms<=0.05).
    assert_with_pcc(expected_f32, actual, pcc=0.995)
    assert max_abs < 0.5, f"max_abs={max_abs} exceeded sanity floor 0.5"
    assert rel_rms < 0.05, f"rel_rms={rel_rms} exceeded bf16 envelope 0.05"
