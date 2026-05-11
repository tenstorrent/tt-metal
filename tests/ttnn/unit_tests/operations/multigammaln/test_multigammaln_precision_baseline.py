# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision-baseline test for multigammaln (order p = 4).

Measures, for a small set of representative shapes, the achieved precision
of the TTNN kernel against torch.special.multigammaln(., 4):
  - PCC                  (Pearson correlation)
  - max abs error
  - mean abs error
  - relative RMS error

This file is the source-of-truth for "what precision does the op deliver
TODAY". Future refinements should tighten its tolerances and update the
table in `verification_report.md` / `changelog.md`.

Inputs are restricted to the in-domain region a > 1.6 to keep the
reference finite. The reflection-branch and out-of-domain behavior are
covered separately by the acceptance test.
"""

import math

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import check_with_pcc
from ttnn.operations.multigammaln import multigammaln


PCC_THRESHOLD = 0.999  # Phase-0 baseline; refinements should tighten.


def _torch_reference(a: torch.Tensor) -> torch.Tensor:
    return torch.special.multigammaln(a.float(), 4)


def _rel_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    err = (actual - expected).pow(2).mean().sqrt()
    norm = expected.pow(2).mean().sqrt().clamp(min=1e-12)
    return float((err / norm).item())


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile_32x32"),
        pytest.param((1, 1, 32, 256), id="multi_tile_W"),
        pytest.param((1, 1, 256, 32), id="multi_tile_H"),
        pytest.param((2, 4, 64, 128), id="multi_batch"),
    ],
)
def test_multigammaln_precision_baseline(device, shape):
    torch.manual_seed(2026)
    # Stay clear of the reflection boundary; this baseline measures the
    # bulk-precision path (every lgamma argument >= 0.1).
    torch_input = 1.6 + 5.0 * torch.rand(shape, dtype=torch.float32)
    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = multigammaln(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()

    # Standard metrics — log all of them so the verifier can capture the table.
    pcc_passed, pcc_message = check_with_pcc(torch_expected, actual, pcc=PCC_THRESHOLD)
    allclose_passed, allclose_message = comp_allclose(torch_expected, actual)
    abs_err = (actual - torch_expected).abs()
    max_abs = float(abs_err.max().item())
    mean_abs = float(abs_err.mean().item())
    rms_rel = _rel_rms(actual, torch_expected)

    print(
        f"\n[precision_baseline] shape={shape} "
        f"PCC={pcc_message} max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} rel_rms={rms_rel:.6g} "
        f"allclose={allclose_message}"
    )

    assert pcc_passed, f"PCC below threshold {PCC_THRESHOLD}: {pcc_message}"
