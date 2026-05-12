# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision baseline for glu_fused.

Measures PCC, max/mean abs error, and relative-RMS error against the PyTorch
reference (``torch.nn.functional.glu``) across a small shape set. The numbers
this test records into the captured pytest log are the Phase-0 baseline that
the verification report cites. Refinement agents tighten these.

These tests intentionally do NOT assert tight error bounds — only a loose PCC
floor that simply confirms the op runs end-to-end. The headline metrics are
the ones logged for inspection.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.common.utility_functions import comp_allclose
from ttnn.operations.glu_fused import glu_fused


# Loose floor — the precision baseline is informational, not a regression gate.
PCC_FLOOR = 0.999


def _rel_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative RMS error = sqrt(mean((a-b)^2)) / sqrt(mean(b^2))."""
    diff = actual.double() - expected.double()
    num = diff.pow(2).mean().sqrt().item()
    den = expected.double().pow(2).mean().sqrt().item()
    return num / max(den, 1e-12)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 64), id="single_tile"),
        pytest.param((1, 1, 32, 128), id="multi_tile_W"),
        pytest.param((1, 1, 256, 128), id="medium_HW"),
        pytest.param((2, 2, 128, 256), id="multi_batch_large"),
    ],
)
def test_glu_fused_precision_baseline(device, shape, capsys):
    """
    Record PCC, max abs err, mean abs err, and relative RMS for a small spread
    of shapes. The verification report tabulates the printed values.
    """
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = torch.nn.functional.glu(torch_input, dim=-1)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = glu_fused(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()

    # PCC via the standard test utility.
    pcc_passed, pcc_msg = check_with_pcc(torch_expected, actual, pcc=PCC_FLOOR)

    # Max/mean abs via comp_allclose (we don't care if allclose passes — we
    # just want the ATOL delta string).
    _, allclose_msg = comp_allclose(torch_expected, actual)

    diff = (actual.double() - torch_expected.double()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rrms = _rel_rms(actual, torch_expected)

    # Print key metrics into the captured output for the verification report.
    with capsys.disabled():
        print(
            f"\n[precision-baseline] shape={shape} "
            f"PCC_msg={pcc_msg} "
            f"max_abs={max_abs:.6e} "
            f"mean_abs={mean_abs:.6e} "
            f"rel_rms={rrms:.6e} "
            f"allclose_msg={allclose_msg}"
        )

    assert pcc_passed, pcc_msg
