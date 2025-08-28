# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("use_precomputed_grid", [False, True])
@pytest.mark.parametrize(
    "input_shape, grid_shape",
    [
        ((1, 256, 12, 40), (1, 7, 25281, 2)),
        ((1, 256, 24, 80), (1, 7, 25281, 2)),
        ((1, 256, 48, 160), (1, 7, 25281, 2)),
        ((16, 32, 100, 100), (16, 10000, 4, 2)),
        ((48, 32, 12, 20), (48, 3567, 8, 2)),
        ((8, 32, 100, 100), (8, 300, 4, 2)),
        ((8, 32, 100, 100), (8, 2000, 4, 2)),
        ((16, 32, 50, 50), (16, 10000, 1, 2)),
        ((48, 32, 80, 45), (48, 4832, 1, 2)),
        ((48, 32, 40, 23), (48, 4832, 1, 2)),
        ((48, 32, 20, 12), (48, 4832, 1, 2)),
        ((48, 32, 10, 6), (48, 4832, 1, 2)),
        ((8, 32, 50, 50), (8, 3604, 1, 2)),
    ],
)
def test_grid_sample_near_uniform_grid(device, input_shape, grid_shape, use_precomputed_grid):
    torch.manual_seed(0)

    batch_size, grid_h, grid_w, _ = grid_shape

    batch_size, channels, height, width = input_shape

    input_shape_nhwc = [batch_size, height, width, channels]

    # PyTorch CPU grid_sample has bad behaviour for bfloat16 inputs
    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Generates a uniform grid using torch affine grid
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    theta_batched = theta.unsqueeze(0).expand(batch_size, -1, -1)
    shape = (batch_size, 1, grid_h, grid_w)
    torch_grid = F.affine_grid(theta_batched, shape, align_corners=False)

    # Add small noise to the grid
    torch_grid += torch.randn(grid_shape) * 0.05

    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    torch_grid_bf16 = torch_grid.to(torch.bfloat16)

    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    if use_precomputed_grid:
        ttnn_grid = ttnn.from_torch(torch_grid_bf16, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
        prepared_grid = ttnn.prepare_grid_sample_grid(
            ttnn_grid, input_shape_nhwc, padding_mode="zeros", output_dtype=ttnn.bfloat16
        )
        prepared_grid = ttnn.to_device(prepared_grid, device)
        ttnn_output = ttnn.grid_sample(ttnn_input, prepared_grid, use_precomputed_grid=True)
    else:
        # Use regular grid
        ttnn_grid = ttnn.from_torch(torch_grid_bf16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
        ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)

    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("use_precomputed_grid", [False, True])
@pytest.mark.parametrize(
    "input_shape_nchw, grid_shape, batch_factor",
    [
        # The special shape you requested with familiar input shapes (h/w swapped from existing)
        ((1, 256, 40, 12), (1, 25281, 7, 2), 1),  # No batching (25281/1 = 25281)
        ((1, 256, 80, 24), (1, 25281, 7, 2), 3),  # batch_factor divides 25281 evenly (25281/3 = 8427)
        ((1, 256, 160, 48), (1, 25281, 7, 2), 9),  # batch_factor divides 25281 evenly (25281/9 = 2809)
        # Nice shapes - powers of 2, small grids
        ((1, 64, 32, 32), (1, 16, 16, 2), 2),  # 16/2 = 8 rows
        ((1, 64, 32, 32), (1, 16, 16, 2), 4),  # 16/4 = 4 rows
        ((1, 64, 32, 32), (1, 16, 16, 2), 8),  # 16/8 = 2 rows
        # Not-so-nice shapes - odd numbers, irregular sizes
        ((2, 32, 47, 23), (2, 15, 11, 2), 3),  # 15/3 = 5 rows
        ((1, 128, 100, 50), (1, 35, 25, 2), 5),  # 35/5 = 7 rows
        # Edge cases
        ((1, 256, 64, 64), (1, 1, 1, 2), 1),  # Minimal grid
        ((4, 32, 16, 8), (4, 12, 6, 2), 2),  # Small batch, small grid
        # Mix of no batching cases
        ((8, 32, 100, 100), (8, 300, 4, 2), None),
        ((16, 32, 50, 50), (16, 1000, 10, 2), None),
    ],
)
def test_grid_sample_with_raw_grids(device, input_shape_nchw, grid_shape, batch_factor, use_precomputed_grid):
    """Test grid sample using raw grids with prepare_grid_sample_grid handling batching internally"""
    torch.manual_seed(0)

    # Skip invalid batch factors
    if batch_factor is not None and grid_shape[1] % batch_factor != 0:
        pytest.skip(f"Grid height {grid_shape[1]} not divisible by batch_factor {batch_factor}")

    batch_size, channels, height, width = input_shape_nchw
    input_shape_nhwc = [batch_size, height, width, channels]

    # Generate PyTorch input
    torch_input_nchw = torch.randn(input_shape_nchw, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Generate raw grid coordinates
    grid_n, grid_h, grid_w, _ = grid_shape
    raw_grid = torch.zeros(grid_shape, dtype=torch.float32)

    for n in range(grid_n):
        for h in range(grid_h):
            for w in range(grid_w):
                # Create normalized grid coordinates [-1, 1] with some noise
                norm_x = (w / max(grid_w - 1, 1)) * 2.0 - 1.0 + 0.05 * torch.randn(1).item()
                norm_y = (h / max(grid_h - 1, 1)) * 2.0 - 1.0 + 0.05 * torch.randn(1).item()
                raw_grid[n, h, w, 0] = norm_x  # x coordinate
                raw_grid[n, h, w, 1] = norm_y  # y coordinate

    # Get reference output using PyTorch with the raw grid
    torch_reference_output = F.grid_sample(
        torch_input_nchw, raw_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_reference_output_nhwc = torch_reference_output.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Convert to TTNN
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    if use_precomputed_grid:
        # Use the new prepare_grid_sample_grid with raw grid and batch_factor
        ttnn_raw_grid = ttnn.from_torch(raw_grid, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
        prepared_grid = ttnn.prepare_grid_sample_grid(
            ttnn_raw_grid, input_shape_nhwc, padding_mode="zeros", output_dtype=ttnn.bfloat16, batch_factor=batch_factor
        )
        prepared_grid = ttnn.to_device(prepared_grid, device)
        ttnn_output = ttnn.grid_sample(ttnn_input, prepared_grid, use_precomputed_grid=True)
    else:
        # For regular grid sample, convert raw grid directly
        raw_grid_bf16 = raw_grid.to(torch.bfloat16)
        ttnn_grid = ttnn.from_torch(raw_grid_bf16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
        ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)

    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Verify output shape matches reference
    assert (
        ttnn_output_torch.shape == torch_reference_output_nhwc.shape
    ), f"Expected {torch_reference_output_nhwc.shape}, got {ttnn_output_torch.shape}"

    # Compare outputs
    pcc_passed, pcc_message = assert_with_pcc(torch_reference_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(f"Raw grid test (batch_factor={batch_factor}, precomputed={use_precomputed_grid}): {pcc_message}")
