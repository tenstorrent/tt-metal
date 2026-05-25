# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision baseline measurement for ttnn.operations.softmax.softmax.

Measures PCC, max absolute error, mean absolute error, and relative RMS
error across a small set of 4D shapes (small, medium, larger) for both
reduce dims and both numeric_stable modes. Results are reported to stdout
so the verifier can capture them in the verification report.

Phase-0 expected behaviour: PCC ≥ 0.9999 (one nine tighter than the
acceptance threshold of 0.999 — the kernel runs in fp32 throughout so
the only loss is the Newton-Raphson recip step, which the design notes
claims ≤1 ULP).
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.softmax import softmax


# Three shape sizes × both reduce dims × both numeric_stable modes.
BASELINE_SHAPES = [
    pytest.param((1, 1, 32, 32), id="small_32x32"),
    pytest.param((1, 1, 64, 128), id="medium_64x128"),
    pytest.param((2, 4, 32, 256), id="batched_32x256"),
    pytest.param((1, 1, 128, 512), id="larger_128x512"),
]


def _relative_rms(expected: torch.Tensor, actual: torch.Tensor) -> float:
    diff = (expected - actual).flatten().to(torch.float64)
    ref_std = expected.flatten().to(torch.float64).std().item()
    if ref_std == 0:
        return float("nan")
    return (diff.pow(2).mean().sqrt() / ref_std).item()


@pytest.mark.parametrize("shape", BASELINE_SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_softmax_precision_baseline(device, shape, dim, numeric_stable, capsys):
    """Measure PCC / abs / RMS error on float32 + HiFi4 + fp32_dest_acc_en."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = torch.softmax(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = softmax(ttnn_input, dim=dim, numeric_stable=numeric_stable)
    torch_output = ttnn.to_torch(ttnn_output)

    # Authoritative correctness check (≥ 0.9999 is one nine tighter than
    # the acceptance threshold). assert_with_pcc owns the failure message.
    assert_with_pcc(torch_expected, torch_output, 0.9999)

    # comp_allclose: max ATOL/RTOL delta from the canonical helper.
    _, allclose_msg = comp_allclose(torch_expected, torch_output, rtol=1e-3, atol=1e-5)

    # Manual metrics: max-abs, mean-abs, relative-RMS, PCC value itself.
    abs_err = (torch_expected - torch_output).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    rms_rel = _relative_rms(torch_expected, torch_output)
    _, pcc_val = comp_pcc(torch_expected, torch_output, pcc=0.9999)

    # Emit a single-line summary the verifier can grep / reorganise.
    summary = (
        f"BASELINE shape={tuple(shape)} dim={dim} numeric_stable={numeric_stable} "
        f"pcc={pcc_val:.7f} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
        f"rms_rel={rms_rel:.3e} {allclose_msg}"
    )
    # Print twice — stdout via capsys for pytest -s, and via sys for the log.
    with capsys.disabled():
        print(summary)
