# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Unary ops on DRAM height-sharded, width-sharded and interleaved TILE tensors must give the default read path's output
# bit for bit in every read flow: bursts, one page in flight and the work queue. The reference runs the same op on DRAM
# interleaved on a few cores, below the cores-per-bank gate and off the whole grid, so it always takes the default path.

import contextlib

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_equal

HEIGHT, WIDTH = ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED

# 16 cores: fewer than 8 per DRAM bank and not the whole grid, so the op takes the default path.
DEFAULT_PATH_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})

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


# DRAM sharded over num_shards banks. Height shards are whole rows; width shards are shard_width columns of every row,
# shape[-1] // num_shards by default.
def dram_sharded(shape, layout, num_shards, shard_width=None):
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


# Rows of a [1, 1, rows, width] tensor with about pages_per_core pages per core; 600 is above the work-queue threshold.
def large_rows(device, width, step, pages_per_core=600):
    grid = device.compute_with_storage_grid_size()
    rows = pages_per_core * grid.x * grid.y * 32 * 32 // width
    return -(-rows // step) * step


# layout is "height", "width" or "interleaved".
def memory_config_for(layout, shape, num_shards=7):
    if layout == "interleaved":
        return ttnn.DRAM_MEMORY_CONFIG
    return dram_sharded(shape, HEIGHT if layout == "height" else WIDTH, num_shards)


# measure wraps the tested op's call, e.g. to count the program-cache entries it adds.
def run_against_default_path(op, a, device, memory_config, dtype=ttnn.bfloat16, measure=contextlib.nullcontext):
    tested = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    reference = ttnn.from_torch(
        a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with measure():
        out = op(tested)
    assert out.memory_config().memory_layout == memory_config.memory_layout
    assert_equal(ttnn.to_torch(op(reference, sub_core_grids=DEFAULT_PATH_GRID)), ttnn.to_torch(out))


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize(
    "case, num_shards",
    [
        ("small", 7),
        ("few_pages_per_core", 7),
        ("large", 7),
        ("large_four_shards_short_last", 4),
        ("large_all_banks", None),
    ],
)
def test_height_sharded(device, op_name, case, num_shards):
    num_shards = num_shards or device.dram_grid_size().x  # None: one shard per DRAM bank of this device
    if case == "small":
        rows = 7 * 1024
    elif case == "few_pages_per_core":  # bursts with few cores per bank, the queue with many
        rows = large_rows(device, 1024, 32 * num_shards, pages_per_core=64)
    else:
        rows = large_rows(device, 1024, 32 * num_shards)
    if case == "large_four_shards_short_last":
        rows -= 64
    shape = [1, 1, rows, 1024]
    run_against_default_path(OPS[op_name], random_input(shape), device, dram_sharded(shape, HEIGHT, num_shards))


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize(
    "case, width, num_shards",
    [
        ("small", 7 * 1024, 7),
        ("large", 7 * 1024, 7),
        ("large_one_tile_shards", 7 * 32, 7),
        ("large_four_shards", 4 * 1024, 4),
        ("large_all_banks", None, None),
    ],
)
def test_width_sharded(device, op_name, case, width, num_shards):
    num_shards = num_shards or device.dram_grid_size().x  # None: one shard per DRAM bank of this device
    width = width or num_shards * 1024
    rows = 1024 if case == "small" else large_rows(device, width, 32)
    shape = [1, 1, rows, width]
    run_against_default_path(OPS[op_name], random_input(shape), device, dram_sharded(shape, WIDTH, num_shards))


# Half a row per core on 32-page shards over every DRAM bank: the default split's cores start in only two shards, so on
# chips with few cores per bank every op, up to very heavy, takes the queue.
@pytest.mark.parametrize("op_name", ["relu", "silu", "elu", "gelu_accurate"])
def test_width_sharded_cores_start_in_few_shards(device, op_name):
    grid = device.compute_with_storage_grid_size()
    num_shards = device.dram_grid_size().x
    shape = [1, 1, grid.x * grid.y // 2 * 32, num_shards * 1024]
    run_against_default_path(OPS[op_name], random_input(shape), device, dram_sharded(shape, WIDTH, num_shards))


# A short last width shard: the flows need equal shards, so the op keeps the default path and must still be exact.
def test_width_sharded_uneven(device):
    shape = [1, 1, 1024, 7 * 1024]
    memory_config = dram_sharded(shape, WIDTH, 7, shard_width=33 * 32)
    run_against_default_path(ttnn.silu, random_input(shape), device, memory_config)


# 4 pages on at least 8 cores: the default path's idle cores get zeroed args and must leave without touching memory.
@pytest.mark.parametrize("layout", [HEIGHT, WIDTH])
def test_fewer_tiles_than_cores(device, layout):
    shape = [1, 1, 64, 64]
    run_against_default_path(ttnn.relu, random_input(shape), device, dram_sharded(shape, layout, 2))
    run_against_default_path(ttnn.elu, random_input(shape), device, dram_sharded(shape, layout, 2))


# Light ops take bursts at 100 pages per core with many cores per bank and the queue with few, and the queue at 600 on
# every chip. At 460 the default split's cores start on 3 banks, where mid ops queue too with few cores per bank. Heavy
# and unmeasured ops keep the default path.
@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize("case", ["small", "bursts", "large", "few_start_banks", "odd_shape"])
def test_interleaved(device, op_name, case):
    if case == "odd_shape":
        shape = [3, 5, 96, 160]
    elif case == "small":
        shape = [1, 1, 7 * 1024, 1024]
    else:
        pages_per_core = {"bursts": 100, "large": 600, "few_start_banks": 460}[case]
        shape = [1, 1, large_rows(device, 1024, 32, pages_per_core), 1024]
    run_against_default_path(OPS[op_name], random_input(shape), device, ttnn.DRAM_MEMORY_CONFIG)


# Light ops take the queue on very large tensors too. Compared on device, as host conversion of these tensors is slow.
@pytest.mark.parametrize("op_name", ["identity", "relu", "hardswish"])
def test_interleaved_very_large(device, op_name):
    shape = [1, 1, large_rows(device, 1024, 32, pages_per_core=2100), 1024]
    x = ttnn.rand(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, low=0.05, high=0.95)
    out = OPS[op_name](x)
    reference = OPS[op_name](x, sub_core_grids=DEFAULT_PATH_GRID)
    differing = ttnn.sum(ttnn.typecast(ttnn.ne(out, reference), ttnn.float32))
    assert ttnn.to_torch(differing).item() == 0


# 4 KB pages (fp32) take the flows; bfp8 pages keep the default path, so they skip the one-page size. At 200 pages per
# core heavy fp32 ops on height shards read one page at a time with many cores per bank.
@pytest.mark.parametrize(
    "dtype, size",
    [
        (ttnn.float32, "small"),
        (ttnn.float32, "one_page"),
        (ttnn.float32, "large"),
        (ttnn.bfloat8_b, "small"),
        (ttnn.bfloat8_b, "large"),
    ],
    ids=["fp32-small", "fp32-one_page", "fp32-large", "bfp8-small", "bfp8-large"],
)
@pytest.mark.parametrize("op_name", ["identity", "relu", "silu", "hardswish", "gelu_accurate", "logit"])
@pytest.mark.parametrize("layout", ["height", "width", "interleaved"])
def test_other_formats(device, dtype, op_name, layout, size):
    width = 7 * 1024 if layout == "width" else 1024
    if size == "small":
        rows = 1024 if layout == "width" else 7 * 1024
    else:
        step = 32 if layout == "width" else 32 * 7
        rows = large_rows(device, width, step, pages_per_core=200 if size == "one_page" else 600)
    shape = [1, 1, rows, width]
    memory_config = memory_config_for(layout, shape)
    run_against_default_path(OPS[op_name], random_input(shape).to(torch.float32), device, memory_config, dtype)


@pytest.mark.parametrize("layout", ["height", "width", "interleaved"])
def test_int32_relu(device, layout):
    width = 7 * 1024 if layout == "width" else 1024
    shape = [1, 1, large_rows(device, width, 32 * 7), width]
    torch.manual_seed(0)
    a = torch.randint(-1000, 1000, shape, dtype=torch.int32)
    run_against_default_path(ttnn.relu, a, device, memory_config_for(layout, shape), ttnn.int32)


# Two queue shapes share one program: the cache-hit path must rewrite the total page count and, on shards, the last
# shard's size. Growing on the hit catches a stale smaller count, which would leave the last pages unwritten.
@pytest.mark.parametrize("layout", ["height", "interleaved"])
@pytest.mark.parametrize("grow", [False, True], ids=["shrink", "grow"])
def test_work_queue_program_cache_reuse(device, layout, grow):
    device.cache_entries_counter.reset()
    large = large_rows(device, 1024, 32 * 7)
    for rows in (large - 64, large) if grow else (large, large - 64):
        shape = [1, 1, rows, 1024]
        run_against_default_path(
            ttnn.relu,
            random_input(shape, seed=rows),
            device,
            memory_config_for(layout, shape),
            measure=device.cache_entries_counter.measure,
        )
    assert device.cache_entries_counter.total == 1
