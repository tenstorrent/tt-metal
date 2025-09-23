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

    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

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


@pytest.mark.parametrize("use_precomputed_grid", [True, False])
@pytest.mark.parametrize("batch_output_channels", [True, False])
@pytest.mark.parametrize("grid_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "input_shape, grid_shape, grid_batching_factor",
    [
        ((1, 256, 48, 160), (1, 25281, 7, 2), 7),
        ((1, 64, 16, 32), (1, 8, 12, 2), 3),
        ((2, 32, 8, 16), (2, 6, 8, 2), 2),
        ((1, 128, 32, 32), (1, 16, 16, 2), 4),
    ],
)
def test_grid_sample_batch_output_channels_flag(
    device, input_shape, grid_shape, grid_batching_factor, batch_output_channels, use_precomputed_grid, grid_dtype
):
    """Test grid sample with batch_output_channels flag - both true and false behaviors"""

    # Skip float32 grid with precomputed grid - not currently supported
    if grid_dtype == ttnn.float32 and use_precomputed_grid:
        pytest.skip("Precomputed grid with FLOAT32 grid dtype is not currently supported")

    torch.manual_seed(42)

    batch_size, channels, height, width = input_shape
    grid_n, grid_h, grid_w, grid_coords = grid_shape

    # Validate that grid_batching_factor is a divisor of width
    assert (
        grid_w % grid_batching_factor == 0
    ), f"grid_batching_factor {grid_batching_factor} must divide grid width {grid_w}"

    # Create input tensor (NCHW -> NHWC)
    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Generate random grid tensor for PyTorch reference
    grid_tensor = torch.rand(batch_size, grid_h, grid_w, 2, dtype=torch.float32) * 2.0 - 1.0

    # Create PyTorch reference output
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, grid_tensor, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Prepare TTNN grid with batching
    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    if use_precomputed_grid:
        # Create precomputed grid
        ttnn_grid_host = ttnn.from_torch(grid_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
        input_shape_nhwc = [batch_size, height, width, channels]
        ttnn_grid_precomputed = ttnn.prepare_grid_sample_grid(
            ttnn_grid_host, input_shape_nhwc, padding_mode="zeros", output_dtype=ttnn.bfloat16
        )

        # Reshape for grid batching: (N, H, W*K, 6) -> (N, H, W, 6*K)
        new_grid_w = grid_w // grid_batching_factor
        final_last_dim = 6 * grid_batching_factor
        ttnn_grid_reshaped = ttnn.reshape(ttnn_grid_precomputed, (batch_size, grid_h, new_grid_w, final_last_dim))
        ttnn_grid_device = ttnn.to_device(ttnn_grid_reshaped, device)
    else:
        # Reshape for grid batching: (N, H, W*K, 2) -> (N, H, W, 2*K)
        new_grid_w = grid_w // grid_batching_factor
        new_last_dim = 2 * grid_batching_factor
        ttnn_grid_host = ttnn.from_torch(grid_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=grid_dtype)
        ttnn_grid_reshaped = ttnn.reshape(ttnn_grid_host, (batch_size, grid_h, new_grid_w, new_last_dim))
        ttnn_grid_device = ttnn.to_device(ttnn_grid_reshaped, device)

    # Call TTNN grid_sample with batch_output_channels flag
    ttnn_output = ttnn.grid_sample(
        ttnn_input,
        ttnn_grid_device,
        use_precomputed_grid=use_precomputed_grid,
        batch_output_channels=batch_output_channels,
    )
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Prepare expected output based on batch_output_channels flag
    if batch_output_channels:
        # batch_output_channels=True: output shape (N, H, W, C*K) - channels batched
        expected_shape = (batch_size, grid_h, new_grid_w, channels * grid_batching_factor)
        torch_expected_nhwc = (
            torch_output_nhwc.view(batch_size, grid_h, new_grid_w, grid_batching_factor, channels)
            .contiguous()
            .view(batch_size, grid_h, new_grid_w, channels * grid_batching_factor)
        )
    else:
        # batch_output_channels=False: output shape (N, H, W*K, C) - W dimension extended
        expected_shape = (batch_size, grid_h, new_grid_w * grid_batching_factor, channels)
        torch_expected_nhwc = torch_output_nhwc.view(batch_size, grid_h, new_grid_w * grid_batching_factor, channels)

    # Verify output shape
    assert ttnn_output_torch.shape == expected_shape, f"Expected {expected_shape}, got {ttnn_output_torch.shape}"

    # Verify numerical correctness
    pcc_passed, pcc_message = assert_with_pcc(torch_expected_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)
