# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for groupnorm_sc_N_1_HW_C (verifier, Phase 0).

Measures PCC, max / mean abs error, relative RMS error and the got/true ratio
spread (the scale-bug detector: a tight cluster of `actual / expected` around a
non-1.0 constant is a uniform scale bug, a broad spread centred on 1.0 is
ordinary rounding) across small / medium / SDXL / batched shapes in both
layouts. Metrics come from `assert_with_pcc` and `comp_allclose`; nothing is
hand-computed except the ratio statistics.

    scripts/run_safe_pytest.sh --dev tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C/test_groupnorm_sc_N_1_HW_C_precision_baseline.py -s
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.groupnorm_sc_N_1_HW_C.test_groupnorm_sc_N_1_HW_C import (
    _to_device,
    torch_groupnorm_n_1_hw_c,
)
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

PCC_MIN = 0.995  # golden-suite bf16 gate

SHAPES = [
    pytest.param((1, 1, 32, 32), 1, 0.0, id="small_32x32_G1"),
    pytest.param((1, 1, 128, 128), 4, 0.0, id="medium_128x128_G4"),
    pytest.param((1, 1, 1024, 640), 32, 0.0, id="sdxl_1024x640_G32_straddling"),
    pytest.param((2, 1, 256, 1280), 32, 0.0, id="batch2_256x1280_G32"),
    pytest.param((1, 1, 256, 320), 32, 10.0, id="sd_256x320_G32_mean10sigma"),
]


def _ratio_stats(actual: torch.Tensor, expected: torch.Tensor):
    a = actual.flatten().double()
    e = expected.flatten().double()
    mask = torch.isfinite(a) & torch.isfinite(e) & (e.abs() > 0.05)  # skip near-zero references
    r = a[mask] / e[mask]
    if r.numel() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return float(q[1]), float(q[0]), float(q[2]), float(r.std())


@pytest.mark.parametrize("shape,num_groups,mean_offset", SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_precision_baseline(device, shape, num_groups, mean_offset, layout):
    torch.manual_seed(1234)
    C = shape[-1]
    x = (torch.randn(shape, dtype=torch.float32) + mean_offset).to(torch.bfloat16)
    gamma = torch.randn(1, 1, 1, C, dtype=torch.float32).to(torch.bfloat16)
    beta = torch.randn(1, 1, 1, C, dtype=torch.float32).to(torch.bfloat16)

    tt_x = _to_device(x, device, ttnn.bfloat16, layout)
    tt_g = _to_device(gamma, device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    tt_b = _to_device(beta, device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    tt_y = groupnorm_sc_N_1_HW_C(tt_x, num_groups, gamma=tt_g, beta=tt_b)

    actual = ttnn.to_torch(tt_y).float()
    expected = torch_groupnorm_n_1_hw_c(x, num_groups, gamma=gamma, beta=beta).float()

    _, pcc_msg = assert_with_pcc(expected, actual, PCC_MIN)
    pcc = float(str(pcc_msg).split()[-1]) if "PCC" in str(pcc_msg) else float(pcc_msg)

    diff = (actual - expected).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_rms = (torch.sqrt((diff**2).mean()) / torch.sqrt((expected**2).mean())).item()
    _, allclose_msg = comp_allclose(expected, actual, rtol=0.05, atol=0.05)
    r_med, r_p5, r_p95, r_std = _ratio_stats(actual, expected)

    print(
        f"\nPRECISION {shape} G={num_groups} mean_offset={mean_offset} {layout}: "
        f"pcc={pcc:.6f} max_abs={max_abs:.4f} mean_abs={mean_abs:.5f} rel_rms={rel_rms:.5f} "
        f"ratio_median={r_med:.4f} ratio_p5={r_p5:.4f} ratio_p95={r_p95:.4f} ratio_std={r_std:.4f} | {allclose_msg}"
    )
    assert torch.isfinite(actual).all()
    # Scale-bug guard: the ratio must be centred on 1.0 (structural / scale errors show as a
    # tight cluster around a constant != 1).
    assert abs(r_med - 1.0) < 0.02, f"got/true ratio median {r_med:.4f} is off 1.0 — scale/structural bug"
