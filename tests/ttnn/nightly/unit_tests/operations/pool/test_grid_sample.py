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
FLOAT32_BYTES_PER_ELEMENT = 4
PRECOMPUTED_GRID_ELEMENTS_PER_POINT = 6
STANDARD_GRID_ELEMENTS_PER_POINT = 2
BFLOAT16S_PER_L1_ALIGNMENT = L1_ALIGNMENT_BYTES // BFLOAT16_BYTES_PER_ELEMENT  # 8
FLOAT32S_PER_L1_ALIGNMENT = L1_ALIGNMENT_BYTES // FLOAT32_BYTES_PER_ELEMENT  # 4


# Utility functions
def div_up(numerator, denominator):
    """Integer division with ceiling (round up)"""
    return (numerator + denominator - 1) // denominator


def align_to_boundary(value, alignment):
    """Round up value to the nearest alignment boundary"""
    return div_up(value, alignment) * alignment


def prepare_grid_batching_expected_output(
    torch_output_nhwc, batch_size, grid_h, grid_w, channels, grid_batching_factor, batch_output_channels
):
    """
    Prepare expected PyTorch output for grid batching scenarios.

    Args:
        torch_output_nhwc: Original PyTorch output in NHWC format
        batch_size: Batch size
        grid_h: Grid height
        grid_w: Grid width (will be divided by grid_batching_factor)
        channels: Number of channels
        grid_batching_factor: Factor by which grid is batched
        batch_output_channels: Whether to batch channels or width dimension

    Returns:
        tuple: (expected_shape, torch_expected_nhwc)
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


def _prepare_grid_tensor_host(torch_grid, use_precomputed_grid, grid_dtype, input_shape_nhwc, grid_batching_factor):
    """
    Common grid preparation logic for both interleaved and sharded grids.

    Args:
        torch_grid: PyTorch grid tensor
        use_precomputed_grid: Whether to use precomputed grid
        grid_dtype: Grid data type (ttnn.bfloat16 or ttnn.float32)
        input_shape_nhwc: Input shape in NHWC format (required for precomputed grid)
        grid_batching_factor: Optional batching factor for reshaping grid

    Returns:
        ttnn tensor: Prepared grid tensor on host (not yet on device)
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


def prepare_ttnn_grid(
    torch_grid, device, use_precomputed_grid, grid_dtype, input_shape_nhwc=None, grid_batching_factor=None
):
    """
    Prepare TTNN grid tensor from PyTorch grid.

    Args:
        torch_grid: PyTorch grid tensor
        device: TTNN device
        use_precomputed_grid: Whether to use precomputed grid
        grid_dtype: Grid data type (ttnn.bfloat16 or ttnn.float32)
        input_shape_nhwc: Input shape in NHWC format (required for precomputed grid)
        grid_batching_factor: Optional batching factor for reshaping grid

    Returns:
        ttnn tensor: Prepared grid tensor on device
    """
    ttnn_grid_reshaped = _prepare_grid_tensor_host(
        torch_grid, use_precomputed_grid, grid_dtype, input_shape_nhwc, grid_batching_factor
    )
    return ttnn.to_device(ttnn_grid_reshaped, device)


def prepare_sharded_grid_memory_config(
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
    Create sharded memory configuration for grid tensor.

    Args:
        device: TTNN device
        batch_size: Batch size
        grid_h: Grid height
        grid_w: Grid width
        use_precomputed_grid: Whether using precomputed grid
        grid_dtype: Grid data type
        grid_batching_factor: Optional batching factor
        core_grid_override: Optional core grid override (for custom sharding)

    Returns:
        ttnn.MemoryConfig: Sharded memory configuration
    """
    compute_grid_size = device.compute_with_storage_grid_size()

    if core_grid_override is not None:
        # Check if device has enough cores for the specified core grid
        if core_grid_override.y > compute_grid_size.y or core_grid_override.x > compute_grid_size.x:
            import pytest

            pytest.skip(f"Device grid {compute_grid_size} insufficient for test core grid {core_grid_override}")
        core_grid = core_grid_override
    else:
        # Use full device compute grid by default instead of hardcoded values
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

    # Round up to L1 alignment: width * BYTES_PER_ELEMENT should be multiple of L1_ALIGNMENT_BYTES
    bytes_per_element = FLOAT32_BYTES_PER_ELEMENT if grid_dtype == ttnn.float32 else BFLOAT16_BYTES_PER_ELEMENT
    grid_shard_width_aligned = (
        align_to_boundary(grid_logical_width * bytes_per_element, L1_ALIGNMENT_BYTES) // bytes_per_element
    )

    return ttnn.create_sharded_memory_config(
        (grid_shard_height, grid_shard_width_aligned),
        core_grid,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def prepare_sharded_ttnn_grid(
    torch_grid,
    device,
    use_precomputed_grid,
    grid_dtype,
    input_shape_nhwc=None,
    grid_batching_factor=None,
    core_grid_override=None,
):
    """
    Prepare TTNN grid tensor with sharding from PyTorch grid.

    Args:
        torch_grid: PyTorch grid tensor
        device: TTNN device
        use_precomputed_grid: Whether to use precomputed grid
        grid_dtype: Grid data type (ttnn.bfloat16 or ttnn.float32)
        input_shape_nhwc: Input shape in NHWC format (required for precomputed grid)
        grid_batching_factor: Optional batching factor for reshaping grid
        core_grid_override: Optional core grid override (for custom sharding)

    Returns:
        ttnn tensor: Prepared sharded grid tensor on device
    """
    batch_size, grid_h, grid_w, grid_coords = torch_grid.shape

    # Use common grid preparation logic
    ttnn_grid_reshaped = _prepare_grid_tensor_host(
        torch_grid, use_precomputed_grid, grid_dtype, input_shape_nhwc, grid_batching_factor
    )

    grid_memory_config = prepare_sharded_grid_memory_config(
        device, batch_size, grid_h, grid_w, use_precomputed_grid, grid_dtype, grid_batching_factor, core_grid_override
    )

    ttnn_grid_interleaved = ttnn.to_device(ttnn_grid_reshaped, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.to_memory_config(ttnn_grid_interleaved, grid_memory_config)


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

    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # Prepare grid tensor
    ttnn_grid_device = prepare_ttnn_grid(torch_grid, device, use_precomputed_grid, ttnn.bfloat16, input_shape_nhwc)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid_device, use_precomputed_grid=use_precomputed_grid)

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
    if grid_dtype == ttnn.float32 and use_precomputed_grid:
        pytest.skip("Precomputed grid only supports bfloat16")

    torch.manual_seed(42)

    batch_size, channels, height, width = input_shape
    grid_n, grid_h, grid_w, grid_coords = grid_shape

    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    grid_tensor = torch.rand(batch_size, grid_h, grid_w, 2, dtype=torch.float32) * 2.0 - 1.0

    torch_output_nchw = F.grid_sample(
        torch_input_nchw, grid_tensor, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    input_shape_nhwc = [batch_size, height, width, channels]
    ttnn_grid_device = prepare_ttnn_grid(
        grid_tensor, device, use_precomputed_grid, grid_dtype, input_shape_nhwc, grid_batching_factor
    )

    ttnn_output = ttnn.grid_sample(
        ttnn_input,
        ttnn_grid_device,
        use_precomputed_grid=use_precomputed_grid,
        batch_output_channels=batch_output_channels,
    )
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    expected_shape, torch_expected_nhwc = prepare_grid_batching_expected_output(
        torch_output_nhwc, batch_size, grid_h, grid_w, channels, grid_batching_factor, batch_output_channels
    )

    pcc_passed, pcc_message = assert_with_pcc(torch_expected_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("use_precomputed_grid", [True, False])
@pytest.mark.parametrize("grid_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "input_shape, grid_shape",
    [
        ((1, 256, 48, 160), (1, 1, 25281, 2)),
        ((48, 64, 32, 64), (48, 15, 15, 2)),
        ((13, 96, 8, 16), (13, 6, 7, 2)),
    ],
)
@pytest.mark.parametrize(
    "core_grid",
    [
        None,  # Use full device grid
        ttnn.CoreGrid(y=5, x=4),  # 5,4 is the grid size of the BOS N1 device
    ],
)
def test_grid_sample_sharded(device, input_shape, grid_shape, use_precomputed_grid, grid_dtype, core_grid):
    """Test grid sample with sharded grid tensor and interleaved input tensor"""
    # Skip precomputed grid tests for float32 since precomputed grid only supports bfloat16
    if use_precomputed_grid and grid_dtype == ttnn.float32:
        pytest.skip("Precomputed grid only supports bfloat16")

    torch.manual_seed(42)

    batch_size, channels, height, width = input_shape
    grid_n, grid_h, grid_w, grid_coords = grid_shape

    input_shape_nhwc = [batch_size, height, width, channels]

    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    torch_grid = torch.rand(grid_shape, dtype=torch.float32) * 2 - 1

    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    ttnn_grid_device = prepare_sharded_ttnn_grid(
        torch_grid, device, use_precomputed_grid, grid_dtype, input_shape_nhwc, core_grid_override=core_grid
    )

    # Call TTNN grid_sample - should automatically use sharded implementation
    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid_device, use_precomputed_grid=use_precomputed_grid)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)


@pytest.mark.parametrize("use_precomputed_grid", [True, False])
@pytest.mark.parametrize("batch_output_channels", [True, False])
@pytest.mark.parametrize("grid_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "input_shape, grid_shape, grid_batching_factor",
    [
        ((1, 256, 48, 160), (1, 1, 2809 * 7, 2), 7),
        ((7, 32, 16, 32), (7, 8, 8, 2), 2),
        ((1, 96, 24, 32), (1, 6, 16, 2), 4),
    ],
)
@pytest.mark.parametrize(
    "core_grid",
    [
        None,  # Use full device grid
        ttnn.CoreGrid(y=5, x=4),  # Limited core grid (5x4=20 cores)
    ],
)
def test_grid_sample_sharded_batched(
    device,
    input_shape,
    grid_shape,
    grid_batching_factor,
    use_precomputed_grid,
    batch_output_channels,
    grid_dtype,
    core_grid,
):
    if use_precomputed_grid and grid_dtype == ttnn.float32:
        pytest.skip("Precomputed grid only supports bfloat16")

    torch.manual_seed(0)

    batch_size, channels, height, width = input_shape
    grid_n, grid_h, grid_w, grid_coords = grid_shape

    input_shape_nhwc = [batch_size, height, width, channels]

    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    torch_grid = torch.rand(grid_shape, dtype=torch.float32) * 2 - 1

    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    ttnn_grid_device = prepare_sharded_ttnn_grid(
        torch_grid, device, use_precomputed_grid, grid_dtype, input_shape_nhwc, grid_batching_factor, core_grid
    )

    ttnn_output = ttnn.grid_sample(
        ttnn_input,
        ttnn_grid_device,
        use_precomputed_grid=use_precomputed_grid,
        batch_output_channels=batch_output_channels,
    )
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    expected_shape, torch_expected_nhwc = prepare_grid_batching_expected_output(
        torch_output_nhwc, batch_size, grid_h, grid_w, channels, grid_batching_factor, batch_output_channels
    )
    pcc_passed, pcc_message = assert_with_pcc(torch_expected_nhwc, ttnn_output_torch, pcc=0.99)
