# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for the 2D dual-multicast matmul (Phase 0).

Measures PCC, max abs error, mean abs error, and relative RMS error of the
fused matmul against torch.matmul (fp32 reference) across a small shape
ladder. Phase 0 supports exactly one numerical corner: float32 activation +
float32 weight, TILE_LAYOUT, tile-aligned M/K/N, shared 2D weight, maxed
precision (HiFi4, fp32_dest_acc_en=True).

This is a BASELINE record, not a regression gate beyond a generous floor —
the absolute numbers (printed per shape) are what the verification report
quotes. The PCC floor (0.999) matches the golden suite's (float32, True)
tolerance band; the relative-RMS floor is the same 0.02.

Uses assert_with_pcc (tests.ttnn.utils_for_testing) and comp_allclose
(models.common.utility_functions) — no hand-rolled metrics.
"""

import pytest
import torch
import ttnn

from ttnn.operations.matmul import matmul
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose, comp_pcc


# Phase 0 golden tolerance for (effective_dtype=float32, fp32_dest_acc_en=True):
#   pcc >= 0.999, relative RMS <= 0.02  (eval/golden_tests/matmul/helpers.py)
PCC_FLOOR = 0.999
RMS_FLOOR = 0.02

# (A_shape, B_shape) — tile-aligned, shared 2D weight. small / medium / larger
# (multi-block per core) / wide-K (deep reduction streamed along the multicast).
SHAPES = [
    pytest.param((32, 64), (64, 32), id="small_32x64x32"),
    pytest.param((256, 512), (512, 1024), id="medium_256x512x1024"),
    pytest.param((1024, 1024), (1024, 1024), id="large_1024sq"),
    pytest.param((512, 4096), (4096, 4096), id="wideK_512x4096x4096"),
]


def _metrics(expected: torch.Tensor, actual: torch.Tensor):
    """PCC + abs/RMS error stats, all in fp64 against the fp32 reference."""
    e = expected.flatten().to(torch.float64)
    a = actual.flatten().to(torch.float64)
    abs_err = (e - a).abs()
    max_abs = float(abs_err.max())
    mean_abs = float(abs_err.mean())
    # Relative RMS error: ||e - a||_2 / ||e||_2 (scale-free, matches the golden rms).
    denom = float(e.pow(2).mean().sqrt()) or 1.0
    rel_rms = float(abs_err.pow(2).mean().sqrt()) / denom
    _, pcc_val = comp_pcc(expected, actual, pcc=PCC_FLOOR)
    return pcc_val, max_abs, mean_abs, rel_rms


@pytest.mark.parametrize("a_shape, b_shape", SHAPES)
def test_matmul_precision_baseline(device, a_shape, b_shape):
    dtype = ttnn.float32
    torch.manual_seed(0)

    torch_a = torch.randn(a_shape, dtype=torch.float32)
    torch_b = torch.randn(b_shape, dtype=torch.float32)
    expected = torch.matmul(torch_a, torch_b)

    ttnn_a = ttnn.from_torch(torch_a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(torch_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
    ttnn_out = matmul(ttnn_a, ttnn_b, compute_kernel_config=config)
    out = ttnn.to_torch(ttnn_out).to(torch.float32)

    pcc_val, max_abs, mean_abs, rel_rms = _metrics(expected, out)
    print(
        f"\n[precision] {a_shape}@{b_shape}  "
        f"PCC={pcc_val} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rel_rms={rel_rms:.6g} "
        f"| allclose={comp_allclose(expected, out)}"
    )

    # Generous floors (the baseline value is the deliverable; this just guards regressions).
    assert_with_pcc(expected, out, pcc=PCC_FLOOR)
    assert rel_rms <= RMS_FLOOR, f"relative RMS {rel_rms:.6g} > {RMS_FLOOR}"
