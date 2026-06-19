# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for rms_norm (Phase 0 — bf16 / TILE / tile-aligned).

Measures PCC, max/mean absolute error, and relative RMS error across a small
set of shapes that all route through the (correct) row-parallel Regime A.
Wide / few-row shapes that route through Regime B are intentionally excluded
here — Regime B has a known correctness defect tracked as Refinement 1 in
op_requirements.md (it would dominate the baseline with a systematic miss).

Run:
  scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py
"""

import pytest
import torch

import ttnn

from ttnn.operations.rms_norm import rms_norm
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose


EPSILON = 1e-6


def torch_rms_norm(x, gamma=None, eps=EPSILON):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    out = x * torch.rsqrt(variance + eps)
    if gamma is not None:
        out = out * gamma
    return out


# small, medium, larger — all Regime A (fit one core, enough rows / not wide).
SHAPES = [
    ((1, 1, 32, 32), "small-single-tile"),
    ((1, 1, 64, 128), "medium-multi-tile"),
    ((2, 4, 128, 512), "batched"),
    ((1, 1, 2048, 256), "tall"),
]


@pytest.mark.parametrize("shape,desc", SHAPES, ids=[d for _, d in SHAPES])
@pytest.mark.parametrize("with_gamma", [False, True], ids=["no_gamma", "gamma"])
def test_rms_norm_precision_baseline(device, shape, desc, with_gamma):
    torch.manual_seed(0)
    dtype = ttnn.bfloat16

    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_gamma = None
    ttnn_gamma = None
    if with_gamma:
        W = shape[-1]
        torch_gamma = torch.randn((1, 1, 1, W), dtype=torch.float32)
        ttnn_gamma = ttnn.from_torch(torch_gamma, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    expected = torch_rms_norm(torch_input, torch_gamma, eps=EPSILON)

    ttnn_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False

    ttnn_output = rms_norm(ttnn_input, gamma=ttnn_gamma, epsilon=EPSILON, compute_kernel_config=cfg)
    actual = ttnn.to_torch(ttnn_output).to(torch.float32).reshape(expected.shape)

    diff = (actual - expected).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_rms = (diff.pow(2).mean().sqrt() / expected.float().std()).item()
    _, pcc_str = comp_allclose(expected, actual)

    print(
        f"\n[precision-baseline] shape={shape} gamma={with_gamma} desc={desc}\n"
        f"  max_abs={max_abs:.5f} mean_abs={mean_abs:.5f} rel_rms={rel_rms:.5f}\n"
        f"  {pcc_str}"
    )

    # bf16 tolerance band (matches golden TOLERANCES for bf16).
    assert_with_pcc(expected, actual, pcc=0.995)
    assert rel_rms < 0.04, f"relative RMS {rel_rms:.5f} exceeds 0.04 for {desc} (gamma={with_gamma})"
