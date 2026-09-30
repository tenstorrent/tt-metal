# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Contracts: (1) every case the codegen gate rejects falls back to native; (2) an accepted case
# dispatched twice on the codegen path stays a program-cache hit and rebinds its buffers; (3) a
# perf-demoted case still lands on native. The generated block below is emitted from the port's
# coverage ledger; hand-add off-grid regressions beneath it.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.repeat` takes no implementation argument -- it routes on its own. The forced legs below are
# the verification-only entries in the private module; see repeat_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.repeat_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.repeat_force_codegen


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


_DEMOTED = [
    (
        [1, 1, 1, 1],
        {"repeat_dims": ttnn.Shape([1, 2, 1, 1])},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [1, 2, 10, 20],
        {"repeat_dims": ttnn.Shape([1, 2, 1, 1])},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [1, 2, 12, 24],
        {"repeat_dims": ttnn.Shape([1, 2, 1, 1])},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [1, 2, 14, 28],
        {"repeat_dims": ttnn.Shape([1, 2, 1, 1])},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [1, 2, 16, 32],
        {"repeat_dims": ttnn.Shape([1, 2, 1, 1])},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [1, 2, 18, 36],
        {"repeat_dims": ttnn.Shape([1, 2, 1, 1])},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [1, 2, 20, 40],
        {"repeat_dims": ttnn.Shape([1, 2, 1, 1])},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [1, 2, 22, 44],
        {"repeat_dims": ttnn.Shape([1, 2, 1, 1])},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [1, 2, 256, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.L1_MEMORY_CONFIG},
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 256, 128),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 4, 4],
        {"repeat_dims": ttnn.Shape([1, 2, 1, 1])},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [1, 2, 6, 12],
        {"repeat_dims": ttnn.Shape([1, 2, 1, 1])},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.L1_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.L1_MEMORY_CONFIG},
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 8, 16],
        {"repeat_dims": ttnn.Shape([1, 2, 1, 1])},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
]
_DEMOTED_IDS = [
    "[1, 1, 1, 1]|repeat_dims=[1, 2, 1, 1]|bfloat16|row_major",
    "[1, 2, 10, 20]|repeat_dims=[1, 2, 1, 1]|bfloat16|row_major",
    "[1, 2, 12, 24]|repeat_dims=[1, 2, 1, 1]|bfloat16|row_major",
    "[1, 2, 14, 28]|repeat_dims=[1, 2, 1, 1]|bfloat16|row_major",
    "[1, 2, 16, 32]|repeat_dims=[1, 2, 1, 1]|bfloat16|row_major",
    "[1, 2, 18, 36]|repeat_dims=[1, 2, 1, 1]|bfloat16|row_major",
    "[1, 2, 20, 40]|repeat_dims=[1, 2, 1, 1]|bfloat16|row_major",
    "[1, 2, 22, 44]|repeat_dims=[1, 2, 1, 1]|bfloat16|row_major",
    "[1, 2, 256, 128]@HEIGHT/8x1/ROW_MAJOR/input|memory_config=BufferType.L1&repeat_dims=[2, 1, 1, 1]|float32|row_major",
    "[1, 2, 4, 4]|repeat_dims=[1, 2, 1, 1]|bfloat16|row_major",
    "[1, 2, 6, 12]|repeat_dims=[1, 2, 1, 1]|bfloat16|row_major",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 1, 1, 1]|bfloat16|row_major",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 1, 1, 1]|float32|row_major",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.L1&repeat_dims=[2, 1, 1, 1]|bfloat16|row_major",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.L1&repeat_dims=[2, 1, 1, 1]|float32|row_major",
    "[1, 2, 8, 16]|repeat_dims=[1, 2, 1, 1]|bfloat16|row_major",
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _DEMOTED, ids=_DEMOTED_IDS)
def test_repeat_codegen_demotion(device, shape, kwargs, dtype, layout, placement):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=placement)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    entries_before = device.num_program_cache_entries()
    out = ttnn.repeat(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed a perf-demoted case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


_CACHE_HIT = [
    ([1, 1, 1, 1], {"repeat_dims": ttnn.Shape([1, 2, 1, 1])}, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
    (
        [1, 1, 1, 1],
        {"repeat_dims": ttnn.Shape([1, 3, 10, 20])},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [1, 1, 256, 64],
        {"repeat_dims": ttnn.Shape([1, 1, 2, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 1, 256, 64),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 1, 256, 64],
        {"repeat_dims": ttnn.Shape([1, 1, 2, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 1, 256, 64),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 128, 128],
        {"repeat_dims": ttnn.Shape([2, 2, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 128, 128),
            core_grid=ttnn.CoreGrid(x=2, y=2),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 128, 128],
        {"repeat_dims": ttnn.Shape([2, 2, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 128, 128),
            core_grid=ttnn.CoreGrid(x=2, y=2),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 128, 64],
        {"repeat_dims": ttnn.Shape([1, 1, 1, 2]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 128, 64),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 128, 64],
        {"repeat_dims": ttnn.Shape([1, 1, 1, 2]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 128, 64),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 2, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
]
_CACHE_HIT_IDS = [
    "[1, 1, 1, 1]|repeat_dims=[1, 2, 1, 1]|bfloat16|tile",
    "[1, 1, 1, 1]|repeat_dims=[1, 3, 10, 20]|bfloat16|row_major",
    "[1, 1, 256, 64]@HEIGHT/8x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[1, 1, 2, 1]|bfloat16|row_major",
    "[1, 1, 256, 64]@HEIGHT/8x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[1, 1, 2, 1]|bfloat16|tile",
    "[1, 2, 128, 128]@BLOCK/2x2/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 2, 1, 1]|bfloat16|row_major",
    "[1, 2, 128, 128]@BLOCK/2x2/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 2, 1, 1]|bfloat16|tile",
    "[1, 2, 128, 64]@HEIGHT/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[1, 1, 1, 2]|bfloat16|row_major",
    "[1, 2, 128, 64]@HEIGHT/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[1, 1, 1, 2]|bfloat16|tile",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 1, 1, 1]|bfloat16|tile",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 2, 1, 1]|bfloat16|row_major",
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_repeat_codegen_program_cache_hit(device, shape, kwargs, dtype, layout, placement):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=placement)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    entries_after_miss = device.num_program_cache_entries()
    # Same spec, a distinct allocation: the cached program must rebind its Buffer*s
    # instead of reusing the first dispatch's addresses.
    yt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device, memory_config=placement)
    second_golden = ttnn.to_torch(_force_native(yt, **kwargs))
    assert_equal(second_golden, ttnn.to_torch(_force_codegen(yt, **kwargs)))
    msg = "second forced-codegen dispatch missed the program cache"
    assert device.num_program_cache_entries() == entries_after_miss, msg
