# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DRAM height-sharded unary must match the same op on DRAM interleaved bit for bit in every flow: bursts, one page
in flight (heavy ops on small tensors) and work queue (large tensors)."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal


def dram_height_sharded(device, shape, num_shards):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_shards - 1, 0))})
    shard_h = -(-shape[-2] // num_shards // 32) * 32
    spec = ttnn.ShardSpec(grid, [shard_h, shape[-1]], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, spec)


def run_against_interleaved(device, op, shape, num_shards):
    torch.manual_seed(0)
    a = (torch.rand(shape) * 0.9 + 0.05).to(torch.bfloat16)
    sharded = ttnn.from_torch(
        a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_height_sharded(device, shape, num_shards),
    )
    interleaved = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = op(sharded)
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert_equal(ttnn.to_torch(op(interleaved)), ttnn.to_torch(out))


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


def large_rows(device, num_shards):
    """Rows of a [1, 1, rows, 1024] tensor with about 600 pages per core, above the work-queue threshold."""
    grid = device.compute_with_storage_grid_size()
    step = 32 * num_shards
    return -(-600 * grid.x * grid.y // step) * step


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize("case", ["small", "large", "large_short_last_shard", "large_four_shards"])
def test_matches_interleaved(device, op_name, case):
    num_shards = 4 if case == "large_four_shards" else 7
    rows = 7 * 1024 if case == "small" else large_rows(device, num_shards)
    if case == "large_short_last_shard":
        rows -= 64
    run_against_interleaved(device, OPS[op_name], [1, 1, rows, 1024], num_shards)


def test_work_queue_program_cache_reuse(device):
    """Two queue shapes with the same shard spec share one program: the cache-hit path must rewrite
    the total page count and the last shard's size."""
    device.cache_entries_counter.reset()
    large = large_rows(device, 7)
    for rows in (large, large - 64):
        torch.manual_seed(rows)
        shape = [1, 1, rows, 1024]
        a = (torch.rand(shape) * 0.9 + 0.05).to(torch.bfloat16)
        sharded = ttnn.from_torch(
            a,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram_height_sharded(device, shape, 7),
        )
        with device.cache_entries_counter.measure():
            out = ttnn.silu(sharded)
        interleaved = ttnn.from_torch(
            a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        assert_equal(ttnn.to_torch(ttnn.silu(interleaved)), ttnn.to_torch(out))
    assert device.cache_entries_counter.total == 1
