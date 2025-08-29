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
        # ((1, 256, 12, 40), (1, 7, 25281, 2)),
        # ((1, 256, 24, 80), (1, 7, 25281, 2)),
        ((1, 256, 48, 160), (1, 7, 25281, 2)),
        # ((16, 32, 100, 100), (16, 10000, 4, 2)),
        # ((48, 32, 12, 20), (48, 3567, 8, 2)),
        # ((8, 32, 100, 100), (8, 300, 4, 2)),
        # ((8, 32, 100, 100), (8, 2000, 4, 2)),
        # ((16, 32, 50, 50), (16, 10000, 1, 2)),
        # ((48, 32, 80, 45), (48, 4832, 1, 2)),
        # ((48, 32, 40, 23), (48, 4832, 1, 2)),
        # ((48, 32, 20, 12), (48, 4832, 1, 2)),
        # ((48, 32, 10, 6), (48, 4832, 1, 2)),
        # ((8, 32, 50, 50), (8, 3604, 1, 2)),
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


@pytest.mark.parametrize("use_precomputed_grid", [False, True])
@pytest.mark.parametrize(
    "input_shape, grid_shape, channel_extent_factor",
    [
        ((1, 256, 48, 160), (1, 25281, 7, 2), 1),
        # ((1, 32, 16, 16), (1, 8, 8, 2), 1),
        # ((2, 64, 24, 24), (2, 12, 12, 2), 2),
        # ((2, 64, 24, 24), (2, 12, 12, 2), 3),
        # ((1, 128, 32, 32), (1, 16, 16, 2), 2),
        # ((1, 128, 32, 32), (1, 16, 16, 2), 4),
        # ((4, 64, 20, 20), (4, 10, 10, 2), 2),
        # ((4, 64, 20, 20), (4, 10, 10, 2), 5),
        # ((1, 96, 24, 24), (1, 12, 12, 2), 3),
        # ((1, 96, 24, 24), (1, 12, 12, 2), 6),
    ],
)
def test_grid_sample_channel_extending(device, input_shape, grid_shape, channel_extent_factor, use_precomputed_grid):
    """Test grid sample with channel extending functionality (multiple coordinate sets)"""
    torch.manual_seed(0)

    batch_size, channels, height, width = input_shape
    grid_n, grid_h, grid_w, grid_coords = grid_shape

    # Validate that channel_extent_factor is a divisor of width
    assert (
        grid_w % channel_extent_factor == 0
    ), f"channel_extent_factor {channel_extent_factor} must divide grid width {grid_w}"

    # Create input tensor (NCHW -> NHWC)
    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Generate random grid tensor
    grid_tensor = torch.rand(batch_size, grid_h, grid_w, 2, dtype=torch.float32) * 2.0 - 1.0

    # Create host ttnn tensor
    ttnn_grid_host = ttnn.from_torch(grid_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)

    if use_precomputed_grid:
        input_shape_nhwc = [batch_size, height, width, channels]
        ttnn_grid_precomputed = ttnn.prepare_grid_sample_grid(
            ttnn_grid_host, input_shape_nhwc, padding_mode="zeros", output_dtype=ttnn.bfloat16
        )
        new_grid_w = grid_w // channel_extent_factor
        final_last_dim = 6 * channel_extent_factor
        ttnn_grid_reshaped = ttnn.reshape(ttnn_grid_precomputed, (batch_size, grid_h, new_grid_w, final_last_dim))
        ttnn_grid_device = ttnn.to_device(ttnn_grid_reshaped, device)
    else:
        new_grid_w = grid_w // channel_extent_factor
        new_last_dim = 2 * channel_extent_factor
        ttnn_grid_host = ttnn.from_torch(grid_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        ttnn_grid_reshaped = ttnn.reshape(ttnn_grid_host, (batch_size, grid_h, new_grid_w, new_last_dim))
        ttnn_grid_device = ttnn.to_device(ttnn_grid_reshaped, device)

    # Convert input to device
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Run grid sample
    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid_device, use_precomputed_grid=use_precomputed_grid)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # For torch comparison, just do regular grid sample and reshape the output
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, grid_tensor, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc_ = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Reshape the torch output to match ttnn output: divide width by channel_extent_factor, multiply channels by channel_extent_factor
    torch_expected_nhwc = (
        torch_output_nhwc_.view(batch_size, grid_h, new_grid_w, channel_extent_factor, channels)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
        .view(batch_size, grid_h, new_grid_w, channels * channel_extent_factor)
    )

    # Check output shape
    expected_shape = (batch_size, grid_h, new_grid_w, channels * channel_extent_factor)
    assert ttnn_output_torch.shape == expected_shape, f"Expected {expected_shape}, got {ttnn_output_torch.shape}"

    # pcc, pcc_message = assert_with_pcc(torch_output_nhwc_.contiguous(), torch_expected_nhwc.contiguous())
    # logger.info(pcc_message)
    # print(f"TTNN output shape: {ttnn_output_torch.shape}")
    # print(f"Expected shape: {torch_expected_nhwc.shape}")

    # # Check numerical accuracy
    pcc_passed, pcc_message = assert_with_pcc(torch_expected_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(
        f"Channel extending test (extent_factor={channel_extent_factor}, precomputed={use_precomputed_grid}): {pcc_message}"
    )
