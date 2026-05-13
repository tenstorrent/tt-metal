# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision baseline for atan_mean.

Measures PCC, max abs error, mean abs error, and relative RMS error across a
standard shape set so that refinement agents can compare against the Phase-0
baseline. Errors here are bounded by:
  - SFPU atan_tile approximation (fp32 input, MathFidelity.HiFi4)
  - bf16 quantisation of the 1/W reduce scaler (bit-exact for power-of-two W)
  - fp32 destination accumulation in the matmul-mode REDUCE_ROW path

The test asserts PCC >= 0.999 and prints the rest into the captured stdout so
the verifier can scrape the numbers into the report.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.atan_mean import atan_mean

from tests.ttnn.utils_for_testing import check_with_pcc
from models.common.utility_functions import comp_allclose


PCC_THRESHOLD = 0.999


def _relative_rms_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    err = actual.float() - expected.float()
    rms = err.pow(2).mean().sqrt().item()
    expected_rms = expected.float().pow(2).mean().sqrt().item()
    if expected_rms < 1e-12:
        return rms
    return rms / expected_rms


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile_32x32"),
        pytest.param((1, 1, 64, 64), id="small_64x64"),
        pytest.param((1, 1, 256, 128), id="medium_256x128"),
        pytest.param((1, 8, 128, 128), id="batched_1x8x128x128"),
    ],
)
def test_atan_mean_precision_baseline(device, shape, capsys):
    """Measure precision metrics across a standard shape set."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = torch.atan(torch_input).mean(dim=-1)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = atan_mean(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()
    expected = torch_expected.float()

    # PCC
    pcc_passed, pcc_msg = check_with_pcc(expected, actual, pcc=PCC_THRESHOLD)

    # Max + mean abs err via comp_allclose (we use loose rtol/atol so we get
    # the diagnostic string, not the boolean — the diagnostic carries the
    # max ATOL/RTOL deltas we want).
    _, allclose_msg = comp_allclose(expected, actual, rtol=1.0, atol=1.0)

    err = (actual - expected).abs()
    max_abs = err.max().item()
    mean_abs = err.mean().item()
    rel_rms = _relative_rms_error(actual, expected)

    # Print the structured baseline so the verifier can scrape it. Use a
    # consistent prefix so it's easy to grep out of pytest's captured stdout.
    print(f"\nBASELINE shape={tuple(shape)} pcc_msg={pcc_msg}")
    print(f"BASELINE shape={tuple(shape)} max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rel_rms={rel_rms:.6e}")
    print(f"BASELINE shape={tuple(shape)} comp_allclose={allclose_msg}")

    assert pcc_passed, f"PCC below threshold {PCC_THRESHOLD} for shape={shape}: {pcc_msg}"
