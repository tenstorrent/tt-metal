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
        ((2, 32, 16, 32), (2, 8, 8, 2)),  # channels=32 (multiple of 32)
        ((1, 64, 32, 64), (1, 16, 16, 2)),  # channels=64 (multiple of 32)
        ((4, 96, 8, 16), (4, 4, 4, 2)),  # channels=96 (multiple of 32)
    ],
)
def test_grid_sample_sharded(device, input_shape, grid_shape, use_precomputed_grid):
    """Test grid sample with sharded grid tensor and interleaved input tensor"""
    torch.manual_seed(42)

    batch_size, channels, height, width = input_shape
    grid_n, grid_h, grid_w, grid_coords = grid_shape

    assert batch_size == grid_n, "Batch size mismatch between input and grid"
    assert grid_coords == 2, "Grid should have 2 coordinates (x, y)"
    assert channels % 32 == 0, f"Channels {channels} must be multiple of 32"

    input_shape_nhwc = [batch_size, height, width, channels]

    # Generate input data
    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Generate grid using affine_grid with small noise
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    theta_batched = theta.unsqueeze(0).expand(batch_size, -1, -1)
    shape = (batch_size, 1, grid_h, grid_w)
    torch_grid = F.affine_grid(theta_batched, shape, align_corners=False)
    torch_grid += torch.randn(grid_shape) * 0.05

    # Calculate expected PyTorch output
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # Create sharded memory config for grid tensor only
    compute_grid_size = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(y=min(2, compute_grid_size.y), x=min(2, compute_grid_size.x))

    # Grid sharding: round up shard width to L1 alignment (16 bytes for bfloat16)
    grid_total_height = batch_size * grid_h * grid_w
    grid_shard_height = (grid_total_height + core_grid.num_cores - 1) // core_grid.num_cores
    grid_logical_width = 6 if use_precomputed_grid else 2
    # Round up to 16-byte alignment: width * 2 bytes should be multiple of 16
    grid_shard_width_aligned = ((grid_logical_width * 2 + 15) // 16) * 8  # 8 bfloat16s per 16 bytes

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
        ttnn_grid_device = ttnn.to_device(ttnn_grid_precomputed, device, memory_config=grid_memory_config)
    else:
        # Standard grid
        ttnn_grid_host = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        ttnn_grid_device = ttnn.to_device(ttnn_grid_host, device, memory_config=grid_memory_config)

    # Call TTNN grid_sample - should automatically use sharded implementation
    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid_device, use_precomputed_grid=use_precomputed_grid)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Verify output shape
    expected_shape = torch_output_nhwc.shape
    assert ttnn_output_torch.shape == expected_shape, f"Expected {expected_shape}, got {ttnn_output_torch.shape}"

    # Verify that sharded implementation was actually used
    assert not ttnn_input.is_sharded(), "Input tensor should be interleaved"
    assert ttnn_grid_device.is_sharded(), "Grid tensor should be sharded"
    assert ttnn_output.is_sharded(), "Output tensor should be sharded"

    # Verify numerical correctness
    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.98)
    logger.info(f"SHARDED grid_sample (precomputed={use_precomputed_grid}): {pcc_message}")
