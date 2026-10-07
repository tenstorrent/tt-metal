# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DRAM height-sharded unary: the work queue (DramHeightFlow::WorkQueue) and the one-page static flow
(DramHeightFlow::StaticOnePage) must match the same op on a DRAM interleaved copy bit for bit."""

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


# Single-op chains on the work-queue list (memory- or tail-bound on Blackhole).
QUEUE_OPS = {
    "identity": ttnn.identity,
    "relu": ttnn.relu,
    "silu": ttnn.silu,
    "softplus": ttnn.softplus,
    "tanh": ttnn.tanh,
    "hardswish": ttnn.hardswish,
    "sigmoid": ttnn.sigmoid,
    "softcap": lambda t: ttnn.softcap(t, 10.0),
    "gelu_fast_lut": lambda t: ttnn.gelu(t, variant=ttnn.GeluVariant.FastLut),
}
# Compute-bound ops: static split, one page per barrier on small tensors.
COMPUTE_BOUND_OPS = {
    "gelu_accurate": lambda t: ttnn.gelu(t, variant=ttnn.GeluVariant.Accurate),
    "logit": lambda t: ttnn.logit(t, eps=1e-6),
    "mish_fast": lambda t: ttnn.mish(t, fast_and_approximate_mode=True),
    "elu": ttnn.elu,
}

# About 594 pages per core on a 110-core grid: above the 512-page threshold for the queue.
QUEUE_SHAPE = [1, 1, 7 * 9344, 1024]


@pytest.mark.parametrize("op_name", list(QUEUE_OPS))
@pytest.mark.parametrize(
    "shape, num_shards",
    [
        (QUEUE_SHAPE, 7),
        ([1, 1, 7 * 9344 - 64, 1024], 7),  # short last shard
        ([1, 1, 4 * 16352, 1024], 4),
    ],
)
def test_work_queue_matches_interleaved(device, op_name, shape, num_shards):
    run_against_interleaved(device, QUEUE_OPS[op_name], shape, num_shards)


@pytest.mark.parametrize("op_name", list(COMPUTE_BOUND_OPS))
@pytest.mark.parametrize("shape", [[1, 1, 7 * 1024, 1024], QUEUE_SHAPE])
def test_compute_bound_static_matches_interleaved(device, op_name, shape):
    run_against_interleaved(device, COMPUTE_BOUND_OPS[op_name], shape, 7)


def test_work_queue_program_cache_reuse(device):
    """Two queue shapes with the same shard spec share one program: the cache-hit path must rewrite
    the total page count and the last shard's size."""
    device.cache_entries_counter.reset()
    for rows in (7 * 9344, 7 * 9344 - 64):
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
