# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal


def core_range_set(lo, hi):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*lo), ttnn.CoreCoord(*hi))})


def sharded_memory_config(memory_layout, grid, shard_shape):
    return ttnn.MemoryConfig(
        memory_layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    )


def make_input(shape, dtype):
    # Values are exact in bf16 (0..250) and in int32, so the comparison is bit exact.
    numel = 1
    for dim in shape:
        numel *= dim
    values = (torch.arange(numel, dtype=torch.int64) % 251).to(torch.float32).reshape(shape)
    if dtype == ttnn.int32:
        return values.to(torch.int32)
    return values.to(torch.bfloat16)


def run_tilize(device, shape, dtype, memory_layout, shard_grid, shard_shape, sub_core_grid, sharded_output):
    torch_input = make_input(shape, dtype)
    input_memory_config = sharded_memory_config(memory_layout, shard_grid, shard_shape)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )
    output_memory_config = input_memory_config if sharded_output else ttnn.L1_MEMORY_CONFIG
    tt_output = ttnn.tilize(
        tt_input, memory_config=output_memory_config, use_multicore=True, sub_core_grids=sub_core_grid
    )
    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert_equal(torch_input, ttnn.to_torch(tt_output))


# A one-core sub_core_grids used to route every call to the single-core factory, which reads one
# full-width row-major stick from a single page. When the input is sharded and a row spans several
# shards (WIDTH or BLOCK sharded), that read crosses shard boundaries and the output is wrong.
@pytest.mark.parametrize(
    "shape, dtype, memory_layout, shard_grid, shard_shape, sharded_output, sub_core_grid",
    [
        pytest.param(
            [1, 1, 32, 256],
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            core_range_set((0, 0), (3, 0)),
            [32, 64],
            True,
            core_range_set((0, 0), (0, 0)),
            id="width_4_cores_bf16",
        ),
        pytest.param(
            [1, 1, 32, 256],
            ttnn.int32,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            core_range_set((0, 0), (3, 0)),
            [32, 64],
            True,
            core_range_set((0, 0), (0, 0)),
            id="width_4_cores_int32",
        ),
        pytest.param(
            [1, 1, 32, 64],
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            core_range_set((0, 0), (1, 0)),
            [32, 32],
            True,
            core_range_set((0, 0), (0, 0)),
            id="width_2_cores_bf16",
        ),
        pytest.param(
            [1, 1, 64, 128],
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            core_range_set((0, 0), (1, 1)),
            [32, 64],
            True,
            core_range_set((0, 0), (0, 0)),
            id="block_2x2_bf16",
        ),
        pytest.param(
            [1, 1, 64, 128],
            ttnn.int32,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            core_range_set((0, 0), (1, 1)),
            [32, 64],
            True,
            core_range_set((0, 0), (0, 0)),
            id="block_2x2_int32",
        ),
        pytest.param(
            [1, 1, 32, 256],
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            core_range_set((0, 0), (3, 0)),
            [32, 64],
            False,
            core_range_set((0, 0), (0, 0)),
            id="width_4_cores_interleaved_output",
        ),
    ],
)
def test_tilize_sharded_row_single_core_sub_grid(
    device, shape, dtype, memory_layout, shard_grid, shard_shape, sharded_output, sub_core_grid
):
    run_tilize(device, shape, dtype, memory_layout, shard_grid, shard_shape, sub_core_grid, sharded_output)


# Controls: these are correct before and after the fix. A row that lives inside one shard
# (HEIGHT_SHARDED), an interleaved input, and a sub_core_grids with two or more cores.
@pytest.mark.parametrize(
    "shape, dtype, memory_layout, shard_grid, shard_shape, sub_core_grid",
    [
        pytest.param(
            [1, 1, 128, 256],
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            core_range_set((0, 0), (0, 3)),
            [32, 256],
            core_range_set((0, 0), (0, 0)),
            id="height_4_cores_one_core_sub_grid",
        ),
        pytest.param(
            [1, 1, 32, 256],
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            core_range_set((0, 0), (3, 0)),
            [32, 64],
            core_range_set((0, 0), (1, 0)),
            id="width_4_cores_two_core_sub_grid",
        ),
        pytest.param(
            [1, 1, 32, 256],
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            core_range_set((0, 0), (3, 0)),
            [32, 64],
            core_range_set((1, 0), (4, 0)),
            id="width_4_cores_offset_sub_grid",
        ),
    ],
)
def test_tilize_sharded_row_multi_core_sub_grid(
    device, shape, dtype, memory_layout, shard_grid, shard_shape, sub_core_grid
):
    run_tilize(device, shape, dtype, memory_layout, shard_grid, shard_shape, sub_core_grid, sharded_output=True)


def test_tilize_interleaved_one_core_sub_grid(device):
    torch_input = make_input([1, 1, 32, 256], ttnn.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = ttnn.tilize(
        tt_input, memory_config=ttnn.L1_MEMORY_CONFIG, use_multicore=True, sub_core_grids=core_range_set((0, 0), (0, 0))
    )
    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert_equal(torch_input, ttnn.to_torch(tt_output))
