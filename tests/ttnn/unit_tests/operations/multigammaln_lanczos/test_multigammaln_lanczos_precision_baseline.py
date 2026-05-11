# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision baseline for `multigammaln_lanczos`.

For each shape, this test computes the operation on a randomized in-domain
input (a in [2.0, 10.0]) and measures:

  - PCC (`check_with_pcc` from tests.ttnn.utils_for_testing)
  - max-abs error (`comp_allclose` from models.common.utility_functions)
  - mean-abs error (`comp_allclose`)
  - relative-RMS error (computed directly)

These metrics characterise what precision the Lanczos kernel achieves *today*
(Phase 0) so refinement agents know the starting bar before tightening
tolerances. The PCC assertion is intentionally loose (>= 0.999) — Lanczos at
fp32 with the alternating-sign 6-term polynomial does NOT reach the same PCC
as Stirling+reflection. The recorded numbers (printed below and copied into
verification_report.md) are the load-bearing artifacts of this test.
"""

import math
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.common.utility_functions import comp_allclose

from ttnn.operations.multigammaln_lanczos import multigammaln_lanczos


def _torch_reference(a: torch.Tensor) -> torch.Tensor:
    """torch.special.multigammaln(a, 4) — the user-visible contract."""
    return torch.special.multigammaln(a.float(), 4)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="medium"),
        pytest.param((1, 1, 256, 256), id="large"),
        pytest.param((2, 4, 64, 128), id="multi_batch"),
    ],
)
def test_multigammaln_lanczos_precision_baseline(device, shape):
    """
    Measure PCC, max-abs error, mean-abs error, and relative-RMS error on a
    standard set of shapes. Inputs are sampled in the comfortable Lanczos
    domain `a ∈ [2.0, 10.0]` so the polynomial converges well — this isolates
    the *kernel's* numerical precision from the *approximation's* divergence.
    """
    torch.manual_seed(123)

    # In-domain inputs — the safe Lanczos region.
    torch_input = (2.0 + 8.0 * torch.rand(shape, dtype=torch.float32)).clamp(min=2.0, max=10.0)
    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = multigammaln_lanczos(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()

    # --- Metrics ---
    pcc_pass, pcc_msg = check_with_pcc(torch_expected, actual, pcc=0.999)

    # comp_allclose returns (passing, output_msg) where output_msg includes
    # Max ATOL and Max RTOL deltas.
    _allclose_pass, allclose_msg = comp_allclose(torch_expected, actual)

    diff = (actual - torch_expected).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()

    # Relative RMS error per the spec formula.
    expected_rms = torch_expected.pow(2).mean().sqrt()
    diff_rms = (actual - torch_expected).pow(2).mean().sqrt()
    rel_rms_err = (diff_rms / (expected_rms + 1e-12)).item()

    # Single structured line for easy grep'ing into the verification report.
    print(
        f"PRECISION_BASELINE shape={shape} pcc_msg={pcc_msg!r} "
        f"max_abs_err={max_abs_err:.6e} mean_abs_err={mean_abs_err:.6e} "
        f"rel_rms_err={rel_rms_err:.6e} allclose={allclose_msg!r}"
    )

    assert pcc_pass, f"PCC failed: {pcc_msg}"
