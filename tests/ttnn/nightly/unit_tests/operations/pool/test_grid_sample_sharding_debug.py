# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc


# Constants
TILE_HEIGHT = 32
TILE_WIDTH = 32
L1_ALIGNMENT_BYTES = 16
BFLOAT16_BYTES_PER_ELEMENT = 2
FLOAT32_BYTES_PER_ELEMENT = 4
PRECOMPUTED_GRID_ELEMENTS_PER_POINT = 6
STANDARD_GRID_ELEMENTS_PER_POINT = 2


def div_up(numerator, denominator):
    """Integer division with ceiling (round up)"""
    return (numerator + denominator - 1) // denominator


def align_to_tile_boundary(value, tile_size=32):
    """Round up value to the nearest tile boundary (32)"""
    return div_up(value, tile_size) * tile_size


def _prepare_grid_tensor_host(torch_grid, use_precomputed_grid, grid_dtype, input_shape_nhwc, grid_batching_factor):
    """
    Common grid preparation logic for both interleaved and sharded grids.
    """
    batch_size, grid_h, grid_w, grid_coords = torch_grid.shape

    if use_precomputed_grid:
        if input_shape_nhwc is None:
            raise ValueError("input_shape_nhwc is required for precomputed grid")

        # Create precomputed grid
        ttnn_grid_host = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
        ttnn_grid_precomputed = ttnn.prepare_grid_sample_grid(
            ttnn_grid_host, input_shape_nhwc, padding_mode="zeros", output_dtype=ttnn.bfloat16
        )

        if grid_batching_factor is not None:
            # Reshape for grid batching: (N, H, W*K, 6) -> (N, H, W, 6*K)
            new_grid_w = grid_w // grid_batching_factor
            final_last_dim = PRECOMPUTED_GRID_ELEMENTS_PER_POINT * grid_batching_factor
            return ttnn.reshape(ttnn_grid_precomputed, (batch_size, grid_h, new_grid_w, final_last_dim))
        else:
            return ttnn_grid_precomputed
    else:
        # Create regular grid
        ttnn_grid_host = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=grid_dtype)

        if grid_batching_factor is not None:
            # Reshape for grid batching: (N, H, W*K, 2) -> (N, H, W, 2*K)
            new_grid_w = grid_w // grid_batching_factor
            new_last_dim = STANDARD_GRID_ELEMENTS_PER_POINT * grid_batching_factor
            return ttnn.reshape(ttnn_grid_host, (batch_size, grid_h, new_grid_w, new_last_dim))
        else:
            return ttnn_grid_host


def prepare_sharded_grid_memory_config_tile_aligned(
    device,
    batch_size,
    grid_h,
    grid_w,
    use_precomputed_grid,
    grid_dtype,
    grid_batching_factor=None,
    core_grid_override=None,
):
    """
    Create sharded memory configuration for grid tensor with TILE alignment instead of L1 alignment.
    """
    compute_grid_size = device.compute_with_storage_grid_size()

    if core_grid_override is not None:
        # Check if device has enough cores for the specified core grid
        if core_grid_override.y > compute_grid_size.y or core_grid_override.x > compute_grid_size.x:
            import pytest

            pytest.skip(f"Device grid {compute_grid_size} insufficient for test core grid {core_grid_override}")
        core_grid = core_grid_override
    else:
        # Use full device compute grid by default
        core_grid = ttnn.CoreGrid(y=compute_grid_size.y, x=compute_grid_size.x)

    # Calculate grid dimensions based on batching
    if grid_batching_factor is not None:
        new_grid_w = grid_w // grid_batching_factor
        grid_total_height = batch_size * grid_h * new_grid_w
        grid_logical_width = (
            PRECOMPUTED_GRID_ELEMENTS_PER_POINT * grid_batching_factor
            if use_precomputed_grid
            else STANDARD_GRID_ELEMENTS_PER_POINT * grid_batching_factor
        )
    else:
        grid_total_height = batch_size * grid_h * grid_w
        grid_logical_width = (
            PRECOMPUTED_GRID_ELEMENTS_PER_POINT if use_precomputed_grid else STANDARD_GRID_ELEMENTS_PER_POINT
        )

    grid_shard_height = div_up(grid_total_height, core_grid.num_cores)

    # For sharded memory config with TILE alignment: align to 32 in both dimensions
    grid_shard_height_aligned = align_to_tile_boundary(grid_shard_height, TILE_HEIGHT)
    grid_shard_width_aligned = align_to_tile_boundary(grid_logical_width, TILE_WIDTH)

    return ttnn.create_sharded_memory_config(
        (grid_shard_height_aligned, grid_shard_width_aligned),
        core_grid,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def prepare_grid_batching_expected_output(
    torch_output_nhwc, batch_size, grid_h, grid_w, channels, grid_batching_factor, batch_output_channels
):
    """
    Prepare expected PyTorch output for grid batching scenarios.
    """
    new_grid_w = grid_w // grid_batching_factor

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

    return expected_shape, torch_expected_nhwc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("use_precomputed_grid", [True])  # , False])
@pytest.mark.parametrize("batch_output_channels", [False])  # , False])
@pytest.mark.parametrize("grid_dtype", [ttnn.bfloat16])  # , ttnn.float32])
@pytest.mark.parametrize(
    "input_shape, grid_shape, grid_batching_factor",
    [
        ((1, 256, 48, 160), (1, 1, 200 * 4, 2), 4),
        # ((7, 32, 16, 32), (7, 8, 8, 2), 2),
        # ((1, 96, 24, 32), (1, 6, 16, 2), 4),
        # ((2, 64, 32, 32), (2, 4, 12, 2), 3),
    ],
)
@pytest.mark.parametrize(
    "core_grid",
    [
        None,  # Use full device grid
        # ttnn.CoreGrid(y=4, x=3),  # Custom core grid for testing
    ],
)
def test_grid_sample_sharded_grid_batching(
    device,
    input_shape,
    grid_shape,
    grid_batching_factor,
    use_precomputed_grid,
    batch_output_channels,
    grid_dtype,
    core_grid,
):
    """
    Test sharded grid sample with grid batching using TILE-aligned sharding.
    This test:
    1. Creates a grid in TILED format on host
    2. Sends it to device (DRAM) TILED
    3. Converts interleaved to sharded while TILED (to_memory_config)
    4. Untilizes with to_layout()
    5. Compares the standard row major grid vs untilized grid
    """
    if use_precomputed_grid and grid_dtype == ttnn.float32:
        pytest.skip("Precomputed grid only supports bfloat16")

    torch.manual_seed(0)

    batch_size, channels, height, width = input_shape
    grid_n, grid_h, grid_w, grid_coords = grid_shape

    input_shape_nhwc = [batch_size, height, width, channels]

    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    # torch_grid = torch.rand(grid_shape, dtype=torch.float32) * 2 - 1

    torch_grid = torch.zeros(grid_shape, dtype=torch.float32) - 0.5
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # Step 1: Create grid in TILED format on host
    ttnn_grid_row_major_host = _prepare_grid_tensor_host(
        torch_grid, use_precomputed_grid, grid_dtype, input_shape_nhwc, grid_batching_factor
    )

    # Convert to TILED layout on host
    ttnn_grid_tiled_host = ttnn.to_layout(ttnn_grid_row_major_host, ttnn.TILE_LAYOUT)

    # Step 2: Send to device (DRAM) TILED
    ttnn_grid_tiled_dram = ttnn.to_device(ttnn_grid_tiled_host, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Step 3: Convert interleaved to sharded while TILED (to_memory_config)
    sharded_memory_config = prepare_sharded_grid_memory_config_tile_aligned(
        device, batch_size, grid_h, grid_w, use_precomputed_grid, grid_dtype, grid_batching_factor, core_grid
    )
    ttnn_grid_tiled_sharded = ttnn.to_memory_config(ttnn_grid_tiled_dram, sharded_memory_config)

    # Step 4: Untilize with to_layout()
    ttnn_grid_untilized = ttnn.untilize(ttnn_grid_tiled_sharded, use_pack_untilize=False)

    # Step 5: Compare the grids
    # Create standard row major grid for comparison
    ttnn_grid_standard_host = _prepare_grid_tensor_host(
        torch_grid, use_precomputed_grid, grid_dtype, input_shape_nhwc, grid_batching_factor
    )
    ttnn_grid_standard_device = ttnn.to_device(ttnn_grid_standard_host, device=device)

    # Convert both grids to torch for comparison
    grid_untilized_torch = ttnn.to_torch(ttnn_grid_untilized)
    grid_standard_torch = ttnn.to_torch(ttnn_grid_standard_device)

    # Compare the grids - they should be identical
    pcc_passed, pcc_message = check_with_pcc(
        grid_standard_torch[:, :, :, 2:6], grid_untilized_torch[:, :, :, 2:6], pcc=0.999
    )
    logger.info(f"Grid comparison: {pcc_message}")

    print("Untilized grid slice:")
    print(grid_untilized_torch[:, :, 0:32, 0:2])

    print("Standard grid slice:")
    print(grid_standard_torch[:, :, 0:32, 0:2])

    # Also test the actual grid sample operation with the untilized grid
    ttnn_output = ttnn.grid_sample(
        ttnn_input,
        ttnn_grid_untilized,
        use_precomputed_grid=use_precomputed_grid,
        batch_output_channels=batch_output_channels,
    )
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    expected_shape, torch_expected_nhwc = prepare_grid_batching_expected_output(
        torch_output_nhwc, batch_size, grid_h, grid_w, channels, grid_batching_factor, batch_output_channels
    )

    pcc_passed, pcc_message = check_with_pcc(torch_expected_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(f"Grid sample output: {pcc_message}")
