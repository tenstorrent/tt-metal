# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision baseline for groupnorm_sc_N_1_HW_C (Phase 0).

Measures PCC, max abs error, mean abs error and relative RMS error across a
small set of representative shapes that all live inside SUPPORTED. Used by
the verification report and the refinement queue as the starting precision
state; subsequent refinements must not regress these.

All cells here are inside SUPPORTED:
  dtype       = bfloat16
  layout      = TILE_LAYOUT
  alignment   = tile_aligned or c_non_aligned
  affine      = gamma_beta
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C


# (shape, num_groups, id) — span small, medium, large, and one SDXL-style
# tile_aligned + (C/G)%32!=0 case.
PRECISION_SHAPES = [
    ((1, 1, 32, 32), 1, "small_aligned_single_tile"),
    ((1, 1, 128, 256), 8, "medium_aligned"),
    ((1, 1, 64, 320), 32, "sdxl_C320_G32_Cg10"),
    ((1, 1, 1024, 256), 8, "larger_HW"),
]


def _torch_groupnorm(x_nhwc, num_groups, gamma, beta, eps):
    """fp32 reference, channels-last (N,1,HW,C) -> (N,C,HW) -> back."""
    N, _, HW, C = x_nhwc.shape
    x_ncl = x_nhwc.reshape(N, HW, C).permute(0, 2, 1).contiguous()
    g1d = gamma.reshape(C) if gamma is not None else None
    b1d = beta.reshape(C) if beta is not None else None
    y = F.group_norm(x_ncl, num_groups=num_groups, weight=g1d, bias=b1d, eps=eps)
    return y.permute(0, 2, 1).reshape(N, 1, HW, C).contiguous()


def _stats(y_out: torch.Tensor, y_ref: torch.Tensor) -> dict:
    a = y_out.float()
    b = y_ref.float()
    diff = (a - b).abs()
    rms_err = (diff.pow(2).mean().sqrt() / b.float().std().clamp_min(1e-8)).item()
    # Pearson correlation
    af = a.flatten() - a.flatten().mean()
    bf = b.flatten() - b.flatten().mean()
    denom = (af.norm() * bf.norm()).item()
    pcc = (af @ bf).item() / denom if denom > 0 else 1.0
    return {
        "pcc": pcc,
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
        "rel_rms_err": rms_err,
    }


@pytest.mark.parametrize("shape, num_groups, shape_id", PRECISION_SHAPES, ids=[s[2] for s in PRECISION_SHAPES])
def test_precision_baseline(device, shape, num_groups, shape_id, request):
    """Record PCC + abs/RMS-error stats for the SUPPORTED Phase 0 surface."""
    torch.manual_seed(0)
    N, one, HW, C = shape

    x_torch = torch.randn(shape, dtype=torch.float32)
    gamma_torch = torch.randn((1, 1, 1, C), dtype=torch.float32)
    beta_torch = torch.randn((1, 1, 1, C), dtype=torch.float32)

    y_ref = _torch_groupnorm(x_torch, num_groups, gamma_torch, beta_torch, eps=1e-5)

    x_tt = ttnn.from_torch(
        x_torch.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tt = ttnn.from_torch(
        gamma_torch.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tt = ttnn.from_torch(
        beta_torch.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    y_tt = groupnorm_sc_N_1_HW_C(x_tt, num_groups, gamma=gamma_tt, beta=beta_tt, eps=1e-5)
    y_out = ttnn.to_torch(y_tt).to(torch.float32)

    s = _stats(y_out, y_ref)
    # comp_allclose returns "(passed, info_string)"; we extract for reporting.
    _, allclose_info = comp_allclose(y_ref, y_out, rtol=0.05, atol=0.05)

    print(
        f"\n[precision_baseline] {shape_id}: "
        f"shape={shape} G={num_groups} | "
        f"pcc={s['pcc']:.6f} max_abs={s['max_abs_err']:.4f} "
        f"mean_abs={s['mean_abs_err']:.5f} "
        f"rel_rms={s['rel_rms_err']:.4f} | {allclose_info}"
    )

    # Acceptance: PCC must be >= 0.995 (the bf16 threshold in the golden suite).
    assert_with_pcc(y_ref, y_out, pcc=0.995)
