# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision baseline for multigammaln_lanczos.

Measures PCC, max/mean absolute error, and relative RMS error against the
torch.special.multigammaln(x, 4) reference computed in fp64. The output
table is consumed by the verification report and the changelog.

The Lanczos 6-term polynomial evaluated at fp32 with HiFi4 + fp32_dest_acc
is meaningfully less accurate than libm-fp64, so this baseline records the
*current* precision floor — refinements that touch the precision pipeline
(compute_kernel_config exposure, fidelity changes, unpack-to-dest tweaks)
should re-run this test and compare.
"""

import math

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc

from ttnn.operations.multigammaln_lanczos import multigammaln_lanczos


SAFE_LO = 2.0
SAFE_HI = 10.0


def _torch_reference(x: torch.Tensor) -> torch.Tensor:
    """Reference computed in fp64 to keep golden close to true math."""
    return torch.special.multigammaln(x.double(), 4).float()


def _safe_input(shape, seed: int = 7) -> torch.Tensor:
    torch.manual_seed(seed)
    u = torch.rand(shape, dtype=torch.float32)
    return SAFE_LO + (SAFE_HI - SAFE_LO) * u


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile_32x32"),
        pytest.param((1, 1, 64, 64), id="multi_tile_64x64"),
        pytest.param((1, 1, 256, 256), id="multi_tile_256x256"),
        pytest.param((2, 4, 64, 128), id="batched_2x4x64x128"),
    ],
)
def test_multigammaln_lanczos_precision_baseline(device, shape):
    """
    Measure precision metrics. PCC threshold is 0.999 — the Lanczos 6-term
    at fp32 is far more accurate in PCC than in absolute error on this domain.
    """
    torch_input = _safe_input(shape)
    expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = multigammaln_lanczos(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()

    # Per-tensor metrics.
    diff = (actual - expected).float()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    expected_rms = expected.float().pow(2).mean().sqrt().item()
    rms_err = diff.pow(2).mean().sqrt().item()
    rel_rms = rms_err / (expected_rms + 1e-12)

    # Allclose-style summary (uses generous tolerances — we report the deltas
    # rather than gating on tight allclose).
    _, allclose_msg = comp_allclose(expected, actual, rtol=0.1, atol=0.5)

    # PCC gate.
    pcc_passed, pcc_msg = check_with_pcc(expected, actual, pcc=0.999)

    print(
        f"\n[precision_baseline] shape={tuple(shape)} "
        f"pcc_msg={pcc_msg} "
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
        f"rel_rms={rel_rms:.6g} "
        f"({allclose_msg})"
    )

    assert pcc_passed, f"PCC<0.999 for shape={shape}: {pcc_msg}"
    # Phase-0 measured ceiling on the safe domain (a ∈ [2, 10]) with the
    # UnpackToDestFp32 fix in place: max_abs ≤ 0.006, mean_abs ≤ 0.001,
    # rel_rms ≤ 6e-5 across the 4 baseline shapes. Asserting a small headroom
    # over the measured peaks so an accidental regression (e.g., dropping
    # UnpackToDestFp32) trips this test immediately.
    assert max_abs < 0.05, f"max_abs={max_abs} regressed; expected < 0.05 for shape={shape}"
    assert rel_rms < 5e-4, f"rel_rms={rel_rms} regressed; expected < 5e-4 for shape={shape}"
