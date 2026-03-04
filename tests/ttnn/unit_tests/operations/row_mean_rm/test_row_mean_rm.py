# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test for row_mean_rm operation.

Verifies that the mean across W is computed correctly.
The output is (..., H, 32) with the mean in column 0 of each tile-row.
"""

import pytest
import torch
import ttnn

from .row_mean_rm import row_mean_rm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal_single_tile"),
        pytest.param((1, 1, 32, 128), id="multi_tile_W"),
        pytest.param((1, 1, 64, 128), id="multi_tile_HW"),
    ],
)
def test_row_mean_rm(device, shape):
    """Verify row_mean_rm computes correct row means."""
    torch.manual_seed(0)

    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = row_mean_rm(ttnn_input)

    # Expected output shape: (..., H, 32)
    expected_shape = list(shape[:-1]) + [32]
    assert (
        list(ttnn_output.shape) == expected_shape
    ), f"Shape mismatch: got {list(ttnn_output.shape)}, expected {expected_shape}"

    # Check numerical correctness
    # The reduce produces mean in column 0 of each 32-row block
    torch_output = ttnn.to_torch(ttnn_output).float()

    # Compute expected mean per row
    torch_expected_mean = torch_input.float().mean(dim=-1)  # (..., H)

    # Compare column 0 of each 32-row block
    H = shape[-2]
    for h in range(H):
        actual = torch_output[..., h, 0].item()
        expected = torch_expected_mean[..., h].item()
        assert abs(actual - expected) < 0.1, f"Mean mismatch at row {h}: got {actual}, expected {expected}"
