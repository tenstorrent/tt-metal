# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Row broadcast of a [.., 1, W] operand B: with nothing sharded the reader walks C in column groups and holds B in L1.
# Each result must match the same op with B expanded to A's shape.

import pytest
import torch
import ttnn

DTYPES = {
    "bf16": (ttnn.bfloat16, torch.bfloat16),
    "fp32": (ttnn.float32, torch.float32),
    "bfp8": (ttnn.bfloat8_b, torch.bfloat16),
    "int32": (ttnn.int32, torch.int32),
}
OPS = {"mul": ttnn.multiply, "add": ttnn.add}

# Small shares walk one column at a time; from 128 tiles per core the reader walks groups of 4 columns (the last ones
# for 95 tiles per row are a group of 3).
SHAPES = {
    "groups_of_4": ([1, 1, 2048, 8192], [1, 1, 1, 8192]),
    "groups_of_4_partial": ([1, 1, 4096, 95 * 32], [1, 1, 1, 95 * 32]),
    "groups_of_4_channels": ([4, 3, 1024, 1024], [4, 3, 1, 1024]),
    "rows_fit": ([1, 1, 2048, 1024], [1, 1, 1, 1024]),
    "wide": ([1, 1, 256, 20480], [1, 1, 1, 20480]),
    "part_row_per_core": ([1, 1, 64, 4096], [1, 1, 1, 4096]),
    "odd_width": ([1, 1, 320, 96 * 32], [1, 1, 1, 96 * 32]),
    "b_shared_by_channels": ([2, 3, 256, 1024], [1, 1, 1, 1024]),
    "b_per_channel": ([2, 3, 256, 1024], [2, 3, 1, 1024]),
}


def random(shape, torch_dtype):
    if torch_dtype == torch.int32:
        return torch.randint(-1000, 1000, shape, dtype=torch.int32)
    return torch.randn(shape).to(torch_dtype)


def to_device(t, dtype, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def check(device, op, a_shape, b_shape, dtype, torch_dtype, a_memory_config=ttnn.DRAM_MEMORY_CONFIG):
    torch.manual_seed(0)
    a = random(a_shape, torch_dtype)
    b = random(b_shape, torch_dtype)
    a_tt = to_device(a, dtype, device, a_memory_config)
    out = op(a_tt, to_device(b, dtype, device))
    expanded = op(a_tt, to_device(b.expand(a_shape).contiguous(), dtype, device))
    assert torch.equal(ttnn.to_torch(out), ttnn.to_torch(expanded))


@pytest.mark.parametrize("shape_name", list(SHAPES))
@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize("dtype_name", list(DTYPES))
def test_row_bcast_matches_expanded(device, dtype_name, op_name, shape_name):
    dtype, torch_dtype = DTYPES[dtype_name]
    a_shape, b_shape = SHAPES[shape_name]
    check(device, OPS[op_name], a_shape, b_shape, dtype, torch_dtype)


# The program hash excludes the shape, so one cached program serves rows narrower and wider than the cache.
@pytest.mark.parametrize("dtype_name", list(DTYPES))
def test_row_bcast_cached_program_across_widths(device, dtype_name):
    dtype, torch_dtype = DTYPES[dtype_name]
    for width in (1024, 20480, 96 * 32, 1024):
        check(device, ttnn.multiply, [1, 1, 128, width], [1, 1, 1, width], dtype, torch_dtype)


# One cached program walks every reader mode on cache hits, in both directions: page order (one tile row), one column
# at a time, and groups of 4 with a partial last group (95 columns) and core ranges starting inside a group. The small
# shapes leave most cores idle, so cores also switch between idle and busy.
WALKS = [
    [1, 1, 32, 1024],
    [1, 1, 128, 1024],
    [1, 1, 8192, 95 * 32],
    [1, 1, 37 * 32, 7 * 32],
    [1, 1, 32, 1024],
    [1, 1, 8192, 95 * 32],
    [1, 1, 128, 1024],
]


@pytest.mark.parametrize("dtype_name", ["bf16", "fp32", "bfp8"])
def test_row_bcast_walk_transitions_on_cache_hits(device, dtype_name):
    dtype, torch_dtype = DTYPES[dtype_name]
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    for i, a_shape in enumerate(WALKS):
        check(device, ttnn.multiply, a_shape, [1, 1, 1, a_shape[-1]], dtype, torch_dtype)
        if i == 0:
            entries = device.num_program_cache_entries()
            assert entries > 0, "The initial calls must populate the program cache"
        assert device.num_program_cache_entries() == entries, "Walk transitions must reuse existing programs"


# Height-sharded A keeps the page-order walk.
@pytest.mark.parametrize("dtype_name", ["bf16", "fp32"])
def test_row_bcast_sharded_a(device, dtype_name):
    dtype, torch_dtype = DTYPES[dtype_name]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    memory_config = ttnn.create_sharded_memory_config(
        [64, 1024], grid, ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
    )
    check(device, ttnn.multiply, [1, 1, 64 * 8, 1024], [1, 1, 1, 1024], dtype, torch_dtype, memory_config)
