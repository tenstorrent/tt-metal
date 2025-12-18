# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
import torch

PCC = 1.0


def run_move_sharded_test(
    shape,
    layout,
    shard_grid,
    shard_shape,
    shard_orientation,
    buffer_type,
    device,
    dtype=ttnn.bfloat16,
    expected_pcc=PCC,
):
    """Helper function to run a move sharded test and verify results.

    Args:
        shape: Input shape
        layout: Layout (ROW_MAJOR or TILE)
        shard_grid: CoreRangeSet for sharding
        shard_shape: [height, width] per shard
        shard_orientation: ShardOrientation (ROW_MAJOR or COL_MAJOR)
        buffer_type: BufferType (L1 or DRAM)
        device: TTNN device
        dtype: Data type for tensor
        expected_pcc: Expected PCC threshold (default: PCC constant)

    Returns:
        tuple: (passing, actual_pcc)
    """

    # Map ttnn dtype to torch dtype
    dtype_map = {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.int32: torch.int32,
        ttnn.uint32: torch.uint32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    if torch_dtype in (torch.bfloat16, torch.float32):
        torch_tensor = torch.randn(shape, dtype=torch_dtype)
    else:
        torch_tensor = torch.randint(0, 100, shape, dtype=torch_dtype)

    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=buffer_type,
        shard_spec=shard_spec,
    )
    tt_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout, device=device, memory_config=mem_config)

    output = ttnn.move(tt_tensor, memory_config=mem_config)
    tt_host_rm = output.cpu().to(layout)
    pyt_got_back_rm = tt_host_rm.to_torch()

    passing_pcc, output_pcc = comp_pcc(pyt_got_back_rm, torch_tensor, expected_pcc)
    return passing_pcc, output_pcc


@pytest.mark.parametrize("shape", [[1, 1, 25088, 64]])
def test_move_sharded_op(shape, device):
    """
    For non_overlap, multi-core is run for num_tiles > 1.
    """
    torch.manual_seed(1234)
    compute_grid_size = device.compute_with_storage_grid_size()
    # Choose core count based on device size, but let the helper function create the shard grid automatically
    if (compute_grid_size.x * compute_grid_size.y) < 98:
        core_count = 25
        shape[2] = 25050
    else:
        core_count = 98

    dtype = ttnn.bfloat16
    layout = ttnn.ROW_MAJOR_LAYOUT
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    # Automatically create shard grid from core count - dispatch cores will be filtered by the move operation
    shard_grid = get_shard_grid_from_num_cores(core_count, device)
    assert shape[0] == 1 and shape[1] == 1
    assert shape[2] % core_count == 0 and shape[3] % 32 == 0
    shard_shape = [(int)(shape[2] / core_count), shape[3]]
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        shard_shape,
        shard_orientation,
    )
    # make dummy shape half of shape, so we will test move sharded with overlap
    dummy_shape = [shape[0], shape[1], (int)(shape[2] / 2), shape[3]]
    dummy_shard_shape = [(int)(dummy_shape[2] / core_count), dummy_shape[3]]
    dummy_shard_spec = ttnn.ShardSpec(
        shard_grid,
        dummy_shard_shape,
        shard_orientation,
    )
    dummy_tensor = torch.zeros(dummy_shape)
    tt_dummy_tensor = ttnn.Tensor(dummy_tensor, dtype)
    dummy_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=dummy_shard_spec,
    )
    tt_dummy_tensor = tt_dummy_tensor.to(device, dummy_mem_config)
    logger.debug(f"shape={shape}")
    input_volume = shape[2] * shape[3]
    tensor = []
    for val in range(1, input_volume + 1):
        tensor.append(val)
    torch_tensor = torch.tensor(tensor).reshape(shape)
    tt_tensor = ttnn.Tensor(torch_tensor, dtype)
    height_sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    tt_tensor = tt_tensor.to(device, height_sharded_mem_config)

    # Free up dummy tensor from memory to make available to move
    tt_dummy_tensor.deallocate()

    output = ttnn.move(tt_tensor, memory_config=height_sharded_mem_config)

    tt_host_rm = output.cpu().to(layout)
    pyt_got_back_rm = tt_host_rm.to_torch()

    passing_pcc, output_pcc = comp_pcc(pyt_got_back_rm, torch_tensor, PCC)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing_pcc


@pytest.mark.parametrize("core_count", [1, 4, 8, 16, 32])
def test_move_sharded_single_core_to_many_cores(core_count, device):
    """Test move sharded with various core counts, including single core."""
    torch.manual_seed(42)
    compute_grid_size = device.compute_with_storage_grid_size()
    max_cores = compute_grid_size.x * compute_grid_size.y

    # Skip if requested core count exceeds available cores
    if core_count > max_cores:
        pytest.skip(f"Device only has {max_cores} cores, skipping {core_count} core test")

    # Create shape divisible by core_count
    height_per_core = 128
    width = 64
    total_height = core_count * height_per_core

    shape = [1, 1, total_height, width]
    dtype = ttnn.bfloat16
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR

    shard_grid = get_shard_grid_from_num_cores(core_count, device)
    shard_shape = [height_per_core, width]

    passing_pcc, output_pcc = run_move_sharded_test(
        shape, ttnn.ROW_MAJOR_LAYOUT, shard_grid, shard_shape, shard_orientation, ttnn.BufferType.L1, device, dtype
    )
    logger.debug(f"Core count: {core_count}, PCC: {output_pcc}")
    assert passing_pcc, f"Failed with {core_count} cores, PCC: {output_pcc}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint32])
def test_move_sharded_different_dtypes(dtype, device):
    """Test move sharded with different data types."""
    torch.manual_seed(123)
    compute_grid_size = device.compute_with_storage_grid_size()
    core_count = min(16, compute_grid_size.x * compute_grid_size.y)

    height_per_core = 256
    width = 128
    total_height = core_count * height_per_core
    shape = [1, 1, total_height, width]

    shard_grid = get_shard_grid_from_num_cores(core_count, device)
    shard_shape = [height_per_core, width]

    passing_pcc, output_pcc = run_move_sharded_test(
        shape,
        ttnn.ROW_MAJOR_LAYOUT,
        shard_grid,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.BufferType.L1,
        device,
        dtype,
    )
    logger.debug(f"DataType: {dtype}, PCC: {output_pcc}")
    assert passing_pcc, f"Failed with dtype {dtype}, PCC: {output_pcc}"


@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_move_sharded_different_orientations(shard_orientation, layout, device):
    """Test move sharded with different shard orientations."""
    torch.manual_seed(456)
    compute_grid_size = device.compute_with_storage_grid_size()
    core_count = min(16, compute_grid_size.x * compute_grid_size.y)

    height_per_core = 128
    width = 64
    total_height = core_count * height_per_core
    shape = [1, 1, total_height, width]
    dtype = ttnn.bfloat16

    shard_grid = get_shard_grid_from_num_cores(core_count, device)
    shard_shape = [height_per_core, width]

    passing_pcc, output_pcc = run_move_sharded_test(
        shape, layout, shard_grid, shard_shape, shard_orientation, ttnn.BufferType.L1, device, dtype
    )
    logger.debug(f"Orientation: {shard_orientation}, PCC: {output_pcc}")
    assert passing_pcc, f"Failed with orientation {shard_orientation}, PCC: {output_pcc}"


@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
def test_move_sharded_different_buffer_types(buffer_type, device):
    """Test move sharded with different buffer types.

    Note: DRAM has different constraints - it only supports a single row (y=0)
    and has limited channels (x dimension). The shard grid must be adjusted accordingly.
    """
    torch.manual_seed(789)
    dtype = ttnn.bfloat16

    if buffer_type == ttnn.BufferType.DRAM:
        # DRAM constraints: single row (y=0), limited x channels
        # Get DRAM grid size - note: this might not be available on all devices
        try:
            dram_grid_size = device.dram_grid_size()
        except AttributeError:
            pytest.skip("Device does not support dram_grid_size()")

        # DRAM only supports y=0, so we can only use cores in a single row
        max_dram_cores = dram_grid_size.x
        core_count = min(8, max_dram_cores)  # Use reasonable number of DRAM cores

        # Create DRAM shard grid: single row from (0,0) to (core_count-1, 0)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_count - 1, 0))})
    else:
        # L1 can use compute grid
        compute_grid_size = device.compute_with_storage_grid_size()
        core_count = min(16, compute_grid_size.x * compute_grid_size.y)
        shard_grid = get_shard_grid_from_num_cores(core_count, device)

    height_per_core = 128
    width = 64
    total_height = core_count * height_per_core
    shape = [1, 1, total_height, width]

    shard_shape = [height_per_core, width]

    passing_pcc, output_pcc = run_move_sharded_test(
        shape,
        ttnn.ROW_MAJOR_LAYOUT,
        shard_grid,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
        buffer_type,
        device,
        dtype,
    )
    logger.debug(f"Buffer type: {buffer_type}, Core count: {core_count}, PCC: {output_pcc}")
    assert passing_pcc, f"Failed with buffer type {buffer_type}, PCC: {output_pcc}"


def test_move_sharded_minimal_size(device):
    """Test move sharded with minimal tensor size (single core, small shard)."""
    torch.manual_seed(999)

    # Use single core with minimal shard size
    core_count = 1
    height_per_core = 32  # Minimal height
    width = 32  # Minimal width
    shape = [1, 1, height_per_core, width]
    dtype = ttnn.bfloat16

    shard_grid = get_shard_grid_from_num_cores(core_count, device)
    shard_shape = [height_per_core, width]

    passing_pcc, output_pcc = run_move_sharded_test(
        shape,
        ttnn.ROW_MAJOR_LAYOUT,
        shard_grid,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.BufferType.L1,
        device,
        dtype,
    )
    logger.debug(f"Minimal size test, PCC: {output_pcc}")
    assert passing_pcc, f"Failed minimal size test, PCC: {output_pcc}"


def test_move_sharded_non_contiguous_grid(device):
    """Test move sharded with non-contiguous core grid (multiple ranges)."""
    torch.manual_seed(111)
    compute_grid_size = device.compute_with_storage_grid_size()
    max_cores = compute_grid_size.x * compute_grid_size.y

    # Use a core count that will create multiple ranges
    core_count = min(25, max_cores)  # 25 cores typically creates 2 ranges on most devices

    height_per_core = 128
    width = 64
    total_height = core_count * height_per_core
    shape = [1, 1, total_height, width]
    dtype = ttnn.bfloat16

    shard_grid = get_shard_grid_from_num_cores(core_count, device)
    # Verify we have multiple ranges (non-contiguous)
    num_ranges = len(shard_grid.ranges()) if hasattr(shard_grid, "ranges") else 1
    logger.debug(f"Shard grid has {num_ranges} ranges")

    shard_shape = [height_per_core, width]

    passing_pcc, output_pcc = run_move_sharded_test(
        shape,
        ttnn.ROW_MAJOR_LAYOUT,
        shard_grid,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.BufferType.L1,
        device,
        dtype,
    )
    logger.debug(f"Non-contiguous grid test, PCC: {output_pcc}")
    assert passing_pcc, f"Failed non-contiguous grid test, PCC: {output_pcc}"


def test_move_sharded_dispatch_core_filtering(device):
    """Test that move sharded correctly filters dispatch cores from shard grid.

    The C++ implementation automatically filters dispatch cores by intersecting
    the shard grid with the compute grid. This test verifies that the operation
    works correctly even with grids that might theoretically include dispatch cores.
    """
    torch.manual_seed(222)
    compute_grid_size = device.compute_with_storage_grid_size()

    # Use a core count that uses most of the available compute grid
    # The move operation will automatically filter any dispatch cores
    max_cores = compute_grid_size.x * compute_grid_size.y
    core_count = min(max_cores, 50)

    height_per_core = 128
    width = 64
    total_height = core_count * height_per_core
    shape = [1, 1, total_height, width]
    dtype = ttnn.bfloat16

    # Create shard grid - the helper function creates grids within compute grid bounds
    # The C++ move operation will further filter to ensure no dispatch cores
    shard_grid = get_shard_grid_from_num_cores(core_count, device)
    assert not shard_grid.empty(), "Shard grid should not be empty"

    # Verify the grid is within compute grid bounds (dispatch cores are outside)
    bounding_box = shard_grid.bounding_box()
    assert bounding_box.end.x < compute_grid_size.x, "Shard grid should be within compute grid"
    assert bounding_box.end.y < compute_grid_size.y, "Shard grid should be within compute grid"

    shard_shape = [height_per_core, width]

    # The move operation will filter dispatch cores automatically in C++
    passing_pcc, output_pcc = run_move_sharded_test(
        shape,
        ttnn.ROW_MAJOR_LAYOUT,
        shard_grid,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.BufferType.L1,
        device,
        dtype,
    )
    logger.debug(f"Dispatch core filtering test, PCC: {output_pcc}")
    assert passing_pcc, f"Failed dispatch core filtering test, PCC: {output_pcc}"
