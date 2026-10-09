# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DRAM height- and width-sharded unary must match the same op on DRAM interleaved bit for bit in every flow:
bursts, one page in flight (heavy ops on small tensors) and work queue (large tensors)."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal


def dram_sharded(shape, layout, num_shards, shard_width=None):
    """DRAM sharded over num_shards banks. Height shards are whole rows; width shards are shard_width columns
    (shape[-1] // num_shards by default) of every row."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_shards - 1, 0))})
    if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        shard = [-(-shape[-2] // num_shards // 32) * 32, shape[-1]]
    else:
        shard = [shape[-2], shard_width or shape[-1] // num_shards]
    spec = ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(layout, ttnn.BufferType.DRAM, spec)


def run_against_interleaved(op, a, device, memory_config):
    sharded = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    interleaved = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = op(sharded)
    assert out.memory_config().memory_layout == memory_config.memory_layout
    assert_equal(ttnn.to_torch(op(interleaved)), ttnn.to_torch(out))


def random_input(shape, seed=0):
    torch.manual_seed(seed)
    return (torch.rand(shape) * 0.9 + 0.05).to(torch.bfloat16)


OPS = {
    "identity": ttnn.identity,
    "relu": ttnn.relu,
    "silu": ttnn.silu,
    "softplus": ttnn.softplus,
    "tanh": ttnn.tanh,
    "hardswish": ttnn.hardswish,
    "sigmoid": ttnn.sigmoid,
    "softcap": lambda t: ttnn.softcap(t, 10.0),
    "gelu_fast_lut": lambda t: ttnn.gelu(t, variant=ttnn.GeluVariant.FastLut),
    "gelu_accurate": lambda t: ttnn.gelu(t, variant=ttnn.GeluVariant.Accurate),
    "mish_fast": lambda t: ttnn.mish(t, fast_and_approximate_mode=True),
    "elu": ttnn.elu,
    "logit": lambda t: ttnn.logit(t, eps=1e-6),
    "log_sigmoid": ttnn.log_sigmoid,
}


def large_rows(device, width, step):
    """Rows of a [1, 1, rows, width] tensor with about 600 pages per core, above the work-queue threshold."""
    grid = device.compute_with_storage_grid_size()
    rows = 600 * grid.x * grid.y * 32 * 32 // width
    return -(-rows // step) * step


HEIGHT, WIDTH = ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize("case", ["small", "large", "large_short_last_shard", "large_four_shards"])
def test_height_sharded(device, op_name, case):
    num_shards = 4 if case == "large_four_shards" else 7
    rows = 7 * 1024 if case == "small" else large_rows(device, 1024, 32 * num_shards)
    if case == "large_short_last_shard":
        rows -= 64
    shape = [1, 1, rows, 1024]
    run_against_interleaved(OPS[op_name], random_input(shape), device, dram_sharded(shape, HEIGHT, num_shards))


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize(
    "case, width, num_shards",
    [
        ("small", 7 * 1024, 7),
        ("large", 7 * 1024, 7),
        ("large_one_tile_shards", 7 * 32, 7),
        ("large_four_shards", 4 * 1024, 4),
    ],
)
def test_width_sharded(device, op_name, case, width, num_shards):
    rows = 1024 if case == "small" else large_rows(device, width, 32)
    shape = [1, 1, rows, width]
    run_against_interleaved(OPS[op_name], random_input(shape), device, dram_sharded(shape, WIDTH, num_shards))


@pytest.mark.parametrize("layout", [HEIGHT, WIDTH])
def test_fewer_tiles_than_cores(device, layout):
    """Cores without pages get zeroed args and must leave without touching memory."""
    shape = [1, 1, 64, 1024]
    run_against_interleaved(ttnn.relu, random_input(shape), device, dram_sharded(shape, layout, 2))
    run_against_interleaved(ttnn.elu, random_input(shape), device, dram_sharded(shape, layout, 2))


def test_width_sharded_uneven(device):
    """A short last width shard keeps the interleaved page order."""
    shape = [1, 1, 1024, 7 * 1024]
    memory_config = dram_sharded(shape, WIDTH, 7, shard_width=33 * 32)
    run_against_interleaved(ttnn.silu, random_input(shape), device, memory_config)


def test_work_queue_program_cache_reuse(device):
    """Two queue shapes with the same shard spec share one program: the cache-hit path must rewrite
    the total page count and the last shard's size."""
    device.cache_entries_counter.reset()
    large = large_rows(device, 1024, 32 * 7)
    for rows in (large, large - 64):
        shape = [1, 1, rows, 1024]
        a = random_input(shape, seed=rows)
        sharded = ttnn.from_torch(
            a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_sharded(shape, HEIGHT, 7)
        )
        with device.cache_entries_counter.measure():
            out = ttnn.silu(sharded)
        interleaved = ttnn.from_torch(
            a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        assert_equal(ttnn.to_torch(ttnn.silu(interleaved)), ttnn.to_torch(out))
    assert device.cache_entries_counter.total == 1
