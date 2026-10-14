# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Unary ops on DRAM height-sharded, width-sharded and interleaved TILE tensors must match main's path bit for bit
in every read flow: bursts, one page in flight and the work queue. The reference runs the same op on DRAM interleaved
on a few cores, below the cores-per-bank gate, so it always takes main's path."""

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_equal

HEIGHT, WIDTH = ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED

# 16 cores: fewer than 8 per DRAM bank, so the op takes main's path.
MAIN_PATH_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})

# Every op class of the flow rule: light, mid (silu, softplus, sigmoid, fast erf), heavy and unmeasured (atan).
OPS = {
    "identity": ttnn.identity,
    "relu": ttnn.relu,
    "exp": ttnn.exp,
    "log": ttnn.log,
    "tanh": ttnn.tanh,
    "hardswish": ttnn.hardswish,
    "gelu_fast_lut": lambda t, **kw: ttnn.gelu(t, variant=ttnn.GeluVariant.FastLut, **kw),
    "silu": ttnn.silu,
    "softplus": ttnn.softplus,
    "sigmoid": ttnn.sigmoid,
    "erf_fast": lambda t, **kw: ttnn.erf(t, fast_and_approximate_mode=True, **kw),
    "gelu_accurate": lambda t, **kw: ttnn.gelu(t, variant=ttnn.GeluVariant.Accurate, **kw),
    "mish_fast": lambda t, **kw: ttnn.mish(t, fast_and_approximate_mode=True, **kw),
    "elu": ttnn.elu,
    "logit": lambda t, **kw: ttnn.logit(t, eps=1e-6, **kw),
    "log_sigmoid": ttnn.log_sigmoid,
    "atan": ttnn.atan,
}
if is_blackhole():  # softcap is Blackhole only
    OPS["softcap"] = lambda t, **kw: ttnn.softcap(t, 10.0, **kw)


def dram_sharded(shape, layout, num_shards, shard_width=None):
    """DRAM sharded over num_shards banks. Height shards are whole rows; width shards are shard_width columns
    (shape[-1] // num_shards by default) of every row."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_shards - 1, 0))})
    if layout == HEIGHT:
        shard = [-(-shape[-2] // num_shards // 32) * 32, shape[-1]]
    else:
        shard = [shape[-2], shard_width or shape[-1] // num_shards]
    spec = ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(layout, ttnn.BufferType.DRAM, spec)


def random_input(shape, seed=0):
    torch.manual_seed(seed)
    return (torch.rand(shape) * 0.9 + 0.05).to(torch.bfloat16)


def large_rows(device, width, step, pages_per_core=600):
    """Rows of a [1, 1, rows, width] tensor with about pages_per_core pages per core; 600 is above the work-queue
    threshold."""
    grid = device.compute_with_storage_grid_size()
    rows = pages_per_core * grid.x * grid.y * 32 * 32 // width
    return -(-rows // step) * step


def run_against_main_path(op, a, device, memory_config, dtype=ttnn.bfloat16):
    tested = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    reference = ttnn.from_torch(
        a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = op(tested)
    assert out.memory_config().memory_layout == memory_config.memory_layout
    assert_equal(ttnn.to_torch(op(reference, sub_core_grids=MAIN_PATH_GRID)), ttnn.to_torch(out))


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize(
    "case, num_shards",
    [("small", 7), ("large", 7), ("large_short_last_shard", 7), ("large_four_shards", 4), ("large_eight_shards", 8)],
)
def test_height_sharded(device, op_name, case, num_shards):
    rows = 7 * 1024 if case == "small" else large_rows(device, 1024, 32 * num_shards)
    if case == "large_short_last_shard":
        rows -= 64
    shape = [1, 1, rows, 1024]
    run_against_main_path(OPS[op_name], random_input(shape), device, dram_sharded(shape, HEIGHT, num_shards))


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize(
    "case, width, num_shards",
    [
        ("small", 7 * 1024, 7),
        ("large", 7 * 1024, 7),
        ("large_one_tile_shards", 7 * 32, 7),
        ("large_four_shards", 4 * 1024, 4),
        ("large_eight_shards", 8 * 1024, 8),
    ],
)
def test_width_sharded(device, op_name, case, width, num_shards):
    rows = 1024 if case == "small" else large_rows(device, width, 32)
    shape = [1, 1, rows, width]
    run_against_main_path(OPS[op_name], random_input(shape), device, dram_sharded(shape, WIDTH, num_shards))


def test_width_sharded_uneven(device):
    """A short last width shard keeps the interleaved page order."""
    shape = [1, 1, 1024, 7 * 1024]
    memory_config = dram_sharded(shape, WIDTH, 7, shard_width=33 * 32)
    run_against_main_path(ttnn.silu, random_input(shape), device, memory_config)


@pytest.mark.parametrize("layout", [HEIGHT, WIDTH])
def test_fewer_tiles_than_cores(device, layout):
    """Cores without pages get zeroed args and must leave without touching memory."""
    shape = [1, 1, 64, 1024]
    run_against_main_path(ttnn.relu, random_input(shape), device, dram_sharded(shape, layout, 2))
    run_against_main_path(ttnn.elu, random_input(shape), device, dram_sharded(shape, layout, 2))


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize("case", ["small", "large", "odd_shape"])
def test_interleaved(device, op_name, case):
    """Light ops take bursts on small tensors and the queue on large ones, mid ops bursts on large ones; heavy and
    unmeasured ops keep main's path."""
    if case == "odd_shape":
        shape = [3, 5, 96, 160]
    else:
        shape = [1, 1, 7 * 1024 if case == "small" else large_rows(device, 1024, 32), 1024]
    run_against_main_path(OPS[op_name], random_input(shape), device, ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("op_name", ["identity", "silu", "softplus", "sigmoid"])
def test_interleaved_very_large(device, op_name):
    """From 2048 pages per core mid ops queue too. Compared on device, as host conversion of these tensors is slow."""
    shape = [1, 1, large_rows(device, 1024, 32, pages_per_core=2100), 1024]
    x = ttnn.rand(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, low=0.05, high=0.95)
    out = OPS[op_name](x)
    reference = OPS[op_name](x, sub_core_grids=MAIN_PATH_GRID)
    differing = ttnn.sum(ttnn.typecast(ttnn.ne(out, reference), ttnn.float32))
    assert ttnn.to_torch(differing).item() == 0


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat8_b], ids=["fp32", "bfp8"])
@pytest.mark.parametrize("op_name", ["identity", "relu", "silu", "hardswish", "gelu_accurate", "logit"])
@pytest.mark.parametrize("layout", ["height", "width", "interleaved"])
@pytest.mark.parametrize("large", [False, True], ids=["small", "large"])
def test_other_formats(device, dtype, op_name, layout, large):
    """4 KB pages (fp32) take the flows from 128 pages per core on interleaved; bfp8 pages keep main's path."""
    width = 7 * 1024 if layout == "width" else 1024
    if layout == "width":
        rows = large_rows(device, width, 32) if large else 1024
    else:
        rows = large_rows(device, width, 32 * 7) if large else 7 * 1024
    shape = [1, 1, rows, width]
    memory_config = {
        "height": lambda: dram_sharded(shape, HEIGHT, 7),
        "width": lambda: dram_sharded(shape, WIDTH, 7),
        "interleaved": lambda: ttnn.DRAM_MEMORY_CONFIG,
    }[layout]()
    run_against_main_path(OPS[op_name], random_input(shape).to(torch.float32), device, memory_config, dtype)


@pytest.mark.parametrize("layout", ["height", "width", "interleaved"])
def test_int32_relu(device, layout):
    width = 7 * 1024 if layout == "width" else 1024
    shape = [1, 1, large_rows(device, width, 32 * 7), width]
    memory_config = {
        "height": lambda: dram_sharded(shape, HEIGHT, 7),
        "width": lambda: dram_sharded(shape, WIDTH, 7),
        "interleaved": lambda: ttnn.DRAM_MEMORY_CONFIG,
    }[layout]()
    torch.manual_seed(0)
    a = torch.randint(-1000, 1000, shape, dtype=torch.int32)
    run_against_main_path(ttnn.relu, a, device, memory_config, ttnn.int32)


@pytest.mark.parametrize("layout", ["height", "interleaved"])
def test_work_queue_program_cache_reuse(device, layout):
    """Two queue shapes share one program: the cache-hit path must rewrite the total page count and, on shards, the
    last shard's size."""
    device.cache_entries_counter.reset()
    large = large_rows(device, 1024, 32 * 7)
    for rows in (large, large - 64):
        shape = [1, 1, rows, 1024]
        memory_config = dram_sharded(shape, HEIGHT, 7) if layout == "height" else ttnn.DRAM_MEMORY_CONFIG
        a = random_input(shape, seed=rows)
        tested = ttnn.from_torch(
            a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
        reference = ttnn.from_torch(
            a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        with device.cache_entries_counter.measure():
            out = ttnn.relu(tested)
        assert_equal(ttnn.to_torch(ttnn.relu(reference, sub_core_grids=MAIN_PATH_GRID)), ttnn.to_torch(out))
    assert device.cache_entries_counter.total == 1
