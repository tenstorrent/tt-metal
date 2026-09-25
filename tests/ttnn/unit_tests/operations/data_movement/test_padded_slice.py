# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal


def _bf16_pattern(shape, offset):
    numel = 1
    for dim in shape:
        numel *= dim
    return ((torch.arange(numel) + offset) % 64).to(torch.bfloat16).reshape(shape)


def _height_sharded_memory_config(device, shard_height, shard_width, num_cores):
    core_grid = device.compute_with_storage_grid_size()
    if core_grid.x * core_grid.y < num_cores:
        pytest.skip(f"Need {num_cores} cores, device has {core_grid.x * core_grid.y}")
    grid = ttnn.num_cores_to_corerangeset(num_cores, core_grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(grid, (shard_height, shard_width), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


@pytest.mark.parametrize(
    "layout, shape",
    [
        (ttnn.ROW_MAJOR_LAYOUT, (1, 1, 16, 32)),
        (ttnn.TILE_LAYOUT, (1, 1, 64, 32)),
    ],
    ids=["rm", "tile"],
)
def test_padded_slice_cache_hit(layout, shape, device):
    """A second allocation with the same slice bounds must hit the cached program and read the new buffers.

    Row-major selects PaddedSliceRMProgramFactory, which rewrites the reader address and the output CB.
    Tiled selects PaddedSliceTileProgramFactory, which binds the reader buffer and the output CB.
    """
    num_cores = 2
    shard_height = shape[2] // num_cores
    output_mem_config = _height_sharded_memory_config(device, shard_height, shape[3], num_cores)
    begins = [0, 0, 0, 0]
    ends = list(shape)
    step = [1, 1, 1, 1]

    def allocate_input(offset):
        src = _bf16_pattern(shape, offset)
        tt_in = ttnn.from_torch(
            src, device=device, layout=layout, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return src, tt_in

    src_a, tt_in_a = allocate_input(0)
    tt_out_a = ttnn.experimental.padded_slice(tt_in_a, begins, ends, step, memory_config=output_mem_config)
    assert_equal(src_a, ttnn.to_torch(tt_out_a))

    src_b, tt_in_b = allocate_input(17)
    assert tt_in_b.buffer_address() != tt_in_a.buffer_address(), "second input must land at a new address"
    entries = device.num_program_cache_entries()
    tt_out_b = ttnn.experimental.padded_slice(tt_in_b, begins, ends, step, memory_config=output_mem_config)
    assert tt_out_b.buffer_address() != tt_out_a.buffer_address(), "second output must land at a new address"
    assert device.num_program_cache_entries() == entries
    assert_equal(src_b, ttnn.to_torch(tt_out_b))
