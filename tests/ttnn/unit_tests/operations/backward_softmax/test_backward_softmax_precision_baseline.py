# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision baseline for backward_softmax.

Measures PCC, max abs error, mean abs error, and relative RMS error across a
small set of representative shapes. The baseline is a reference for refinement
agents — when they tighten precision (e.g. by exposing the compute config or
switching reduction strategies) the new measurements should match or exceed
this table.

Why PCC instead of `torch.allclose`: the spec test in `test_backward_softmax.py`
uses `atol=0.01`. For larger reductions (W or H ≥ 32) the matmul-based
REDUCE_ROW SUM path on Wormhole B0 produces ~0.1-level absolute error in the
running sum, which propagates into ~0.1-level absolute error in the output
near positions where `dy_i ≈ s` (the catastrophic-cancellation site). PCC is
invariant to this absolute-vs-relative scaling and exposes mathematical
correctness independent of accumulator precision.
"""

import pytest
import torch
import ttnn

from ttnn.operations.backward_softmax import backward_softmax
from tests.ttnn.utils_for_testing import check_with_pcc
from models.common.utility_functions import comp_allclose


def _torch_reference(grad_output: torch.Tensor, output: torch.Tensor, dim: int) -> torch.Tensor:
    """grad_input = output * (grad_output - sum(output * grad_output, dim))"""
    s = (output * grad_output).sum(dim=dim, keepdim=True)
    return output * (grad_output - s)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="32x32"),
        pytest.param((1, 1, 32, 256), id="32x256"),
        pytest.param((1, 1, 64, 128), id="64x128"),
        pytest.param((2, 4, 64, 128), id="2x4x64x128"),
    ],
)
@pytest.mark.parametrize("dim", [-1, -2], ids=["dim=-1", "dim=-2"])
def test_backward_softmax_precision_baseline(device, shape, dim, capsys):
    """
    Run the op on a representative shape and emit a one-line summary the
    verifier scrapes into the precision-baseline table.
    """
    torch.manual_seed(42)
    torch_grad_output = torch.randn(shape, dtype=torch.float32)
    torch_output = torch.randn(shape, dtype=torch.float32)
    expected = _torch_reference(torch_grad_output, torch_output, dim=dim)

    ttnn_grad_output = ttnn.from_torch(
        torch_grad_output,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_grad_input = backward_softmax(ttnn_grad_output, ttnn_output, dim=dim)
    actual = ttnn.to_torch(ttnn_grad_input).float()

    # ---- Metrics ----
    pcc_pass, pcc_msg = check_with_pcc(expected, actual, pcc=0.999)
    # `check_with_pcc` returns the raw float as a string in its message.
    try:
        pcc_value = float(pcc_msg.strip())
    except ValueError:
        pcc_value = float("nan")
    max_abs = (actual - expected).abs().max().item()
    mean_abs = (actual - expected).abs().mean().item()
    expected_rms = expected.pow(2).mean().sqrt().item()
    rel_rms = (actual - expected).pow(2).mean().sqrt().item() / max(expected_rms, 1e-12)
    _, allclose_msg = comp_allclose(expected, actual, rtol=1e-2, atol=1e-2)

    # Use print so pytest -s shows it; also goes to capsys for any CI scraping.
    print(
        f"PRECISION_BASELINE shape={tuple(shape)} dim={dim} "
        f"pcc={pcc_value:.7f} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
        f"rel_rms={rel_rms:.6f} | comp_allclose: {allclose_msg}"
    )

    # Assert PCC ≥ 0.999. This is the bar for "mathematically correct"; the
    # absolute / relative tolerances vary with shape (see the verification
    # report's precision table for the empirical values).
    assert pcc_pass, f"PCC < 0.999 for shape={shape}, dim={dim}: {pcc_msg}"
