# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# Constants
L1_ALIGNMENT_BYTES = 16
BFLOAT16_BYTES_PER_ELEMENT = 2
PRECOMPUTED_GRID_ELEMENTS_PER_POINT = 6
STANDARD_GRID_ELEMENTS_PER_POINT = 2
BFLOAT16S_PER_L1_ALIGNMENT = L1_ALIGNMENT_BYTES // BFLOAT16_BYTES_PER_ELEMENT  # 8


# Utility functions
def div_up(numerator, denominator):
    """Integer division with ceiling (round up)"""
    return (numerator + denominator - 1) // denominator


def align_to_boundary(value, alignment):
    """Round up value to the nearest alignment boundary"""
    return div_up(value, alignment) * alignment


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("use_precomputed_grid", [True])  # , False])
@pytest.mark.parametrize(
    "input_shape, grid_shape",
    [
        # ((1, 256, 48, 160), (1, 1, 25281, 2)),  # channels=32 (multiple of 32), no batching
        # ((1, 64, 32, 64), (1, 15, 15, 2)),  # channels=64 (multiple of 32), no batching
        # ((1, 96, 8, 16), (1, 6, 7, 2)),  # channels=96 (multiple of 32), no batching
    ],
)
def test_grid_sample_sharded(device, input_shape, grid_shape, use_precomputed_grid):
    """Test grid sample with sharded grid tensor and interleaved input tensor"""
    torch.manual_seed(42)

    batch_size, channels, height, width = input_shape
    grid_n, grid_h, grid_w, grid_coords = grid_shape

    assert batch_size == grid_n, "Batch size mismatch between input and grid"
    assert grid_coords == STANDARD_GRID_ELEMENTS_PER_POINT, "Grid should have 2 coordinates (x, y)"
    assert channels % 32 == 0, f"Channels {channels} must be multiple of 32"

    input_shape_nhwc = [batch_size, height, width, channels]

    # Generate input data
    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Generate grid using affine_grid with small noise
    # theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    # theta_batched = theta.unsqueeze(0).expand(batch_size, -1, -1)
    # shape = (batch_size, 1, grid_h, grid_w)
    # torch_grid = F.affine_grid(theta_batched, shape, align_corners=False)
    # torch_grid += torch.randn(grid_shape) * 0.05

    # Create a grid with incrementing values starting from 0.01
    # grid_size = grid_h * grid_w
    # increments = torch.arange(1, grid_size + 1, dtype=torch.float32) * 0.01
    # torch_grid = increments.view(1, grid_h, grid_w, 1).expand(batch_size, -1, -1, 2)
    torch_grid = torch.rand(grid_shape, dtype=torch.float32) * 2 - 1

    # Calculate expected PyTorch output
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Create sharded memory config for grid tensor only
    compute_grid_size = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(y=min(5, compute_grid_size.y), x=min(4, compute_grid_size.x))

    # Grid sharding: round up shard width to L1 alignment
    grid_total_height = batch_size * grid_h * grid_w
    grid_shard_height = div_up(grid_total_height, core_grid.num_cores)
    grid_logical_width = (
        PRECOMPUTED_GRID_ELEMENTS_PER_POINT if use_precomputed_grid else STANDARD_GRID_ELEMENTS_PER_POINT
    )
    # Round up to L1 alignment: width * BFLOAT16_BYTES_PER_ELEMENT should be multiple of L1_ALIGNMENT_BYTES
    grid_shard_width_aligned = (
        align_to_boundary(grid_logical_width * BFLOAT16_BYTES_PER_ELEMENT, L1_ALIGNMENT_BYTES)
        // BFLOAT16_BYTES_PER_ELEMENT
    )

    grid_memory_config = ttnn.create_sharded_memory_config(
        (grid_shard_height, grid_shard_width_aligned),
        core_grid,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Convert input to TTNN with L1 interleaved memory
    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    if use_precomputed_grid:
        # Create precomputed grid
        ttnn_grid_host = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
        ttnn_grid_precomputed = ttnn.prepare_grid_sample_grid(
            ttnn_grid_host, input_shape_nhwc, padding_mode="zeros", output_dtype=ttnn.bfloat16
        )
        # Create interleaved grid tensor first, then convert to sharded
        ttnn_grid_interleaved = ttnn.to_device(
            ttnn_grid_precomputed, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn_grid_device = ttnn.to_memory_config(ttnn_grid_interleaved, grid_memory_config)
    else:
        # Create grid tensor and convert to sharded
        ttnn_grid_interleaved = ttnn.from_torch(
            torch_grid,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_grid_device = ttnn.to_memory_config(ttnn_grid_interleaved, grid_memory_config)

    # Call TTNN grid_sample - should automatically use sharded implementation
    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid_device, use_precomputed_grid=use_precomputed_grid)
    print(ttnn_output.memory_config())
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Verify output shape
    expected_shape = torch_output_nhwc.shape
    assert ttnn_output_torch.shape == expected_shape, f"Expected {expected_shape}, got {ttnn_output_torch.shape}"

    # Calculate and print maximum absolute difference
    abs_diff = torch.abs(torch_output_nhwc - ttnn_output_torch)
    max_abs_diff = abs_diff.max().item()

    # Verify numerical correctness
    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(f"SHARDED grid_sample (precomputed={use_precomputed_grid}): {pcc_message}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("use_precomputed_grid", [True])  # , False])
@pytest.mark.parametrize("batch_output_channels", [True])  # , False])
@pytest.mark.parametrize(
    "input_shape, grid_shape, grid_batching_factor",
    [
        ((1, 256, 48, 160), (1, 1, 2809 * 7, 2), 7),  # grid_batching_factor=7
        # ((1, 32, 16, 32), (1, 8, 8, 2), 2),  # grid_batching_factor=2, reshape to (1, 8, 4, 4)
        # ((1, 96, 24, 32), (1, 6, 16, 2), 4),  # grid_batching_factor=4, reshape to (1, 6, 4, 8)
    ],
)
def test_grid_sample_sharded_batched(
    device, input_shape, grid_shape, grid_batching_factor, use_precomputed_grid, batch_output_channels
):
    """Test grid sample with sharded grid tensor using grid batching (multiple grid coordinates per spatial position)"""
    torch.manual_seed(42)

    batch_size, channels, height, width = input_shape
    grid_n, grid_h, grid_w, grid_coords = grid_shape

    assert batch_size == grid_n, "Batch size mismatch between input and grid"
    assert grid_coords == STANDARD_GRID_ELEMENTS_PER_POINT, "Grid should have 2 coordinates (x, y)"
    assert channels % 32 == 0, f"Channels {channels} must be multiple of 32"

    # Validate that grid_batching_factor is a divisor of width
    assert (
        grid_w % grid_batching_factor == 0
    ), f"grid_batching_factor {grid_batching_factor} must divide grid width {grid_w}"

    input_shape_nhwc = [batch_size, height, width, channels]

    # Generate input data
    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Generate grid tensor (will be reshaped for batching)
    torch_grid = torch.rand(grid_shape, dtype=torch.float32) * 2 - 1

    # Calculate expected PyTorch output
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Convert input to TTNN with L1 interleaved memory
    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    if use_precomputed_grid:
        # Create precomputed grid
        ttnn_grid_host = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
        ttnn_grid_precomputed = ttnn.prepare_grid_sample_grid(
            ttnn_grid_host, input_shape_nhwc, padding_mode="zeros", output_dtype=ttnn.bfloat16
        )

        # Reshape for grid batching: (N, H, W*K, 6) -> (N, H, W, 6*K)
        new_grid_w = grid_w // grid_batching_factor
        final_last_dim = PRECOMPUTED_GRID_ELEMENTS_PER_POINT * grid_batching_factor
        ttnn_grid_reshaped = ttnn.reshape(ttnn_grid_precomputed, (batch_size, grid_h, new_grid_w, final_last_dim))
    else:
        # Reshape for grid batching: (N, H, W*K, 2) -> (N, H, W, 2*K)
        new_grid_w = grid_w // grid_batching_factor
        new_last_dim = STANDARD_GRID_ELEMENTS_PER_POINT * grid_batching_factor
        ttnn_grid_host = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        ttnn_grid_reshaped = ttnn.reshape(ttnn_grid_host, (batch_size, grid_h, new_grid_w, new_last_dim))

    # Create sharded memory config for the reshaped grid tensor
    compute_grid_size = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(y=min(5, compute_grid_size.y), x=min(4, compute_grid_size.x))

    # Grid sharding: round up shard width to L1 alignment
    grid_total_height = batch_size * grid_h * new_grid_w
    grid_shard_height = div_up(grid_total_height, core_grid.num_cores)
    grid_logical_width = final_last_dim if use_precomputed_grid else new_last_dim
    # Round up to L1 alignment: width * BFLOAT16_BYTES_PER_ELEMENT should be multiple of L1_ALIGNMENT_BYTES
    grid_shard_width_aligned = (
        align_to_boundary(grid_logical_width * BFLOAT16_BYTES_PER_ELEMENT, L1_ALIGNMENT_BYTES)
        // BFLOAT16_BYTES_PER_ELEMENT
    )

    grid_memory_config = ttnn.create_sharded_memory_config(
        (grid_shard_height, grid_shard_width_aligned),
        core_grid,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Create interleaved grid tensor first, then convert to sharded
    ttnn_grid_interleaved = ttnn.to_device(ttnn_grid_reshaped, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn_grid_device = ttnn.to_memory_config(ttnn_grid_interleaved, grid_memory_config)

    # Call TTNN grid_sample with batch_output_channels flag - should automatically use sharded implementation
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

    # Calculate and print maximum absolute difference
    abs_diff = torch.abs(torch_expected_nhwc - ttnn_output_torch)
    max_abs_diff = abs_diff.max().item()
    print(f"Grid batching factor: {grid_batching_factor}, batch_output_channels: {batch_output_channels}")
    print(f"Maximum absolute difference: {max_abs_diff}")

    # Verify numerical correctness
    pcc_passed, pcc_message = assert_with_pcc(torch_expected_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(
        f"SHARDED BATCHED grid_sample (precomputed={use_precomputed_grid}, batch_channels={batch_output_channels}, batching={grid_batching_factor}): {pcc_message}"
    )
