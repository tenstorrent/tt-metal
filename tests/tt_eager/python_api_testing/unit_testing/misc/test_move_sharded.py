# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import ttnn
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
import torch


def run_move_sharded_test(
    shape,
    layout,
    shard_grid,
    shard_shape,
    shard_orientation,
    buffer_type,
    device,
    dtype=ttnn.bfloat16,
):
    """Run move sharded and check results with torch.equal."""

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
    tt_host = output.cpu().to(layout)
    pyt_got_back = tt_host.to_torch().to(torch_tensor.dtype)

    return torch.equal(pyt_got_back, torch_tensor)


@pytest.mark.parametrize(
    "memory_layout", [ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED]
)
@pytest.mark.parametrize("shape", [[1, 1, 25088, 64]])
def test_move_sharded_op(memory_layout, shape, device):
    """Validate move sharded on a large tensor with explicit sharding."""

    torch.manual_seed(1234)
    compute_grid_size = device.compute_with_storage_grid_size()
    if (compute_grid_size.x * compute_grid_size.y) < 98:
        core_count = 25
        shape[2] = 25050
    else:
        core_count = 98

    dtype = ttnn.bfloat16
    layout = ttnn.ROW_MAJOR_LAYOUT
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    shard_grid = get_shard_grid_from_num_cores(core_count, device)
    assert shape[0] == 1 and shape[1] == 1
    assert shape[2] % core_count == 0 and shape[3] % 32 == 0
    shard_shape = [shape[2] // core_count, shape[3]]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)

    dummy_shape = [shape[0], shape[1], shape[2] // 2, shape[3]]
    dummy_shard_shape = [dummy_shape[2] // core_count, dummy_shape[3]]
    dummy_shard_spec = ttnn.ShardSpec(shard_grid, dummy_shard_shape, shard_orientation)
    dummy_tensor = torch.zeros(dummy_shape)
    tt_dummy_tensor = ttnn.Tensor(dummy_tensor, dtype)
    dummy_mem_config = ttnn.MemoryConfig(
        memory_layout=memory_layout,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=dummy_shard_spec,
    )
    tt_dummy_tensor = tt_dummy_tensor.to(device, dummy_mem_config)

    logger.debug(f"shape={shape}")
    input_volume = shape[2] * shape[3]
    torch_tensor = torch.arange(1, input_volume + 1, dtype=torch.float32).reshape(shape).to(torch.bfloat16)
    tt_tensor = ttnn.Tensor(torch_tensor, dtype)
    height_sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=memory_layout,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    tt_tensor = tt_tensor.to(device, height_sharded_mem_config)

    # Free up dummy tensor from memory to make available to move
    tt_dummy_tensor.deallocate()

    output = ttnn.move(tt_tensor, memory_config=height_sharded_mem_config)
    tt_host_rm = output.cpu().to(layout)
    pyt_got_back_rm = tt_host_rm.to_torch().to(torch_tensor.dtype)

    assert torch.equal(pyt_got_back_rm, torch_tensor)


@pytest.mark.parametrize("core_count", [1, 4, 8, 16, 32])
def test_move_sharded_single_core_to_many_cores(core_count, device):
    """Test move sharded with various core counts, including single core."""

    torch.manual_seed(42)
    compute_grid_size = device.compute_with_storage_grid_size()
    max_cores = compute_grid_size.x * compute_grid_size.y

    if core_count > max_cores:
        pytest.skip(f"Device only has {max_cores} cores, skipping {core_count} core test")

    height_per_core = 128
    width = 64
    total_height = core_count * height_per_core

    shape = [1, 1, total_height, width]
    dtype = ttnn.bfloat16
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR

    shard_grid = get_shard_grid_from_num_cores(core_count, device)
    shard_shape = [height_per_core, width]

    assert run_move_sharded_test(
        shape, ttnn.ROW_MAJOR_LAYOUT, shard_grid, shard_shape, shard_orientation, ttnn.BufferType.L1, device, dtype
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
def test_move_sharded_different_layout_and_orientation(dtype, shard_orientation, layout, buffer_type, device):
    """Test move sharded with different buffer types."""

    torch.manual_seed(789)

    if buffer_type == ttnn.BufferType.DRAM:
        try:
            dram_grid_size = device.dram_grid_size()
        except AttributeError:
            pytest.skip("Device does not support dram_grid_size()")

        max_dram_cores = dram_grid_size.x
        core_count = min(8, max_dram_cores)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_count - 1, 0))})
    else:
        compute_grid_size = device.compute_with_storage_grid_size()
        core_count = min(16, compute_grid_size.x * compute_grid_size.y)
        shard_grid = get_shard_grid_from_num_cores(core_count, device)

    height_per_core = 128
    width = 64
    total_height = core_count * height_per_core
    shape = [1, 1, total_height, width]

    shard_shape = [height_per_core, width]

    assert run_move_sharded_test(
        shape,
        layout,
        shard_grid,
        shard_shape,
        shard_orientation,
        buffer_type,
        device,
        dtype,
    )


def test_move_sharded_custom_grid(device):
    """Test move sharded with custom core grid."""

    torch.manual_seed(111)
    compute_grid_size = device.compute_with_storage_grid_size()

    if compute_grid_size.x >= 3:
        shard_ranges = {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
            ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0)),
        }
    elif compute_grid_size.y >= 3:
        shard_ranges = {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(0, 2)),
        }
    else:
        pytest.skip("Device grid too small to form custom shard grid")

    shard_grid = ttnn.CoreRangeSet(shard_ranges)
    shard_ranges_list = list(shard_grid.ranges())
    assert len(shard_ranges_list) > 1, "Shard grid should span multiple disjoint ranges"
    logger.debug(f"Custom shard ranges: {shard_ranges_list}")

    core_count = sum(
        (core_range.end.x - core_range.start.x + 1) * (core_range.end.y - core_range.start.y + 1)
        for core_range in shard_ranges_list
    )

    height_per_core = 128
    width = 64
    total_height = core_count * height_per_core
    shape = [1, 1, total_height, width]
    dtype = ttnn.bfloat16

    shard_shape = [height_per_core, width]

    assert run_move_sharded_test(
        shape,
        ttnn.ROW_MAJOR_LAYOUT,
        shard_grid,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.BufferType.L1,
        device,
        dtype,
    )


def test_move_sharded_to_interleaved_rejected(device):
    """Verify move rejects sharded-to-interleaved conversion (output must be sharded)."""
    torch.manual_seed(42)
    shape = [1, 1, 128, 64]
    core_count = 4

    # Create sharded input tensor
    shard_grid = get_shard_grid_from_num_cores(core_count, device)
    shard_shape = [shape[2] // core_count, shape[3]]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)

    input_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_mem_config
    )

    # Attempt to move to interleaved layout should fail
    with pytest.raises(RuntimeError, match="Expected output tensor memory config to be sharded"):
        ttnn.move(input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)


def test_move_interleaved_to_sharded(device):
    """Test move from interleaved to sharded layout."""

    torch.manual_seed(42)
    shape = [1, 1, 128, 64]

    compute_grid_size = device.compute_with_storage_grid_size()
    core_count = min(4, compute_grid_size.x * compute_grid_size.y)

    # Create interleaved input tensor
    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Move to sharded layout
    shard_grid = get_shard_grid_from_num_cores(core_count, device)
    shard_shape = [shape[2] // core_count, shape[3]]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)

    output_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    output_tensor = ttnn.move(input_tensor, memory_config=output_mem_config)

    # Verify result
    output_torch = output_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    assert torch.equal(output_torch, input_torch)
