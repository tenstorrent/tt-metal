# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for clamp_square_practice (verifier-authored).

Measures, per (shape, bounds): PCC, max/mean abs error, relative RMS error, bf16 ULP error
against the bf16-rounded fp32 reference, and the got/true ratio spread (the scale-bug
detector: a tight cluster around a non-1.0 constant means a structural bug, not rounding).

Reference: torch.clamp(x_bf16.float(), min_val, max_val) ** 2 in fp32 (full torch, never a
device-rounded reference).
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.clamp_square_practice import clamp_square_practice

SHAPES = [
    pytest.param((1, 1, 32, 32), id="small_1tile"),
    pytest.param((1, 1, 256, 256), id="medium_64tiles"),
    pytest.param((2, 4, 256, 512), id="batched_1024tiles"),
    pytest.param((1, 1, 1024, 2048), id="large_2048tiles"),
]

BOUNDS = [
    pytest.param(-1.0, 1.0, id="symmetric"),
    pytest.param(-0.3, 0.7, id="non_bf16_exact"),
    pytest.param(-4.0, 4.0, id="mostly_unclamped"),
]

PCC_TARGET = 0.9999
MAX_ULP = 2  # bf16 ULPs vs the bf16-rounded fp32 reference


def _bf16_ordered(t: torch.Tensor) -> torch.Tensor:
    """Map bf16 bit patterns to a monotonically ordered integer line (for ULP distance)."""
    bits = t.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("min_val, max_val", BOUNDS)
def test_precision_baseline(device, shape, min_val, max_val):
    torch.manual_seed(0)
    x = (torch.randn(shape, dtype=torch.float32) * 2.0).to(torch.bfloat16)
    expected = torch.clamp(x.to(torch.float32), min_val, max_val) ** 2

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(clamp_square_practice(tt_x, min_val, max_val)).to(torch.float32)

    diff = (actual - expected).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_rms = (torch.sqrt(((actual - expected) ** 2).mean()) / torch.sqrt((expected**2).mean())).item()
    ulp = (_bf16_ordered(actual) - _bf16_ordered(expected)).abs()
    max_ulp = ulp.max().item()
    frac_exact = (ulp == 0).float().mean().item()

    nz = expected.abs() > 1e-6
    r = actual[nz] / expected[nz]
    r_median = r.median().item()
    r_p5, r_p95 = torch.quantile(r[: min(r.numel(), 1 << 20)].float(), torch.tensor([0.05, 0.95])).tolist()

    _, allclose_msg = comp_allclose(expected, actual)
    print(
        f"\nPRECISION shape={shape} bounds=({min_val},{max_val}) max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
        f"rel_rms={rel_rms:.6g} max_ulp={max_ulp} exact_frac={frac_exact:.4f} "
        f"ratio_median={r_median:.6f} ratio_p5={r_p5:.6f} ratio_p95={r_p95:.6f} | {allclose_msg}"
    )

    _, pcc_msg = assert_with_pcc(expected, actual, PCC_TARGET)
    print(f"PRECISION pcc: {pcc_msg}")
    assert max_ulp <= MAX_ULP, f"max ULP error {max_ulp} > {MAX_ULP}"
    assert abs(r_median - 1.0) < 1e-2, f"got/true ratio median {r_median} != 1 (scale bug?)"
