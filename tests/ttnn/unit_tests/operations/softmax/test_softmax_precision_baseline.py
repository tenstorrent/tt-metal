# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for softmax.

Measures PCC, max abs error, mean abs error, and relative RMS error
across a representative set of shapes (small, medium, one larger).
float32, TILE_LAYOUT, dim=-1 and dim=-2.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose


def pytorch_softmax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(input_tensor.float(), dim=dim)


SHAPES = [
    pytest.param((1, 1, 32, 32), id="1x1x32x32"),
    pytest.param((1, 1, 64, 128), id="1x1x64x128"),
    pytest.param((2, 4, 64, 64), id="2x4x64x64"),
    pytest.param((4, 8, 32, 256), id="4x8x32x256"),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", [-1, -2], ids=["dim_W", "dim_H"])
def test_softmax_precision_baseline(device, shape, dim):
    """Measure PCC, max abs error, mean abs error, relative RMS error."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=dim)
    torch_output = ttnn.to_torch(ttnn_output)

    torch_expected = pytorch_softmax(torch_input, dim=dim)

    # PCC check (standard assert)
    assert_with_pcc(torch_expected, torch_output, pcc=0.999)

    # Detailed metrics
    passes, msg = comp_allclose(torch_expected, torch_output, rtol=1e-5, atol=1e-8)

    abs_diff = (torch_expected.float() - torch_output.float()).abs()
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()

    # Relative RMS error: RMS(diff) / RMS(expected)
    rms_diff = torch.sqrt((abs_diff**2).mean()).item()
    rms_expected = torch.sqrt((torch_expected.float() ** 2).mean()).item()
    rel_rms_err = rms_diff / rms_expected if rms_expected > 0 else float("inf")

    print(f"\n  shape={shape}, dim={dim}")
    print(f"    PCC >= 0.999 (asserted)")
    print(f"    max_abs_err  = {max_abs_err:.6f}")
    print(f"    mean_abs_err = {mean_abs_err:.6f}")
    print(f"    rel_rms_err  = {rel_rms_err:.6f}")
    print(f"    allclose     = {passes} ({msg})")

    # Sanity: no NaN or Inf
    assert not torch_output.isnan().any(), "Output contains NaN"
    assert not torch_output.isinf().any(), "Output contains Inf"
