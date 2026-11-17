# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# The large tensor cases of test_unary_dram_reads.py.

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.eltwise.test_unary_dram_reads import (
    DEFAULT_PATH_GRID,
    HEIGHT,
    OPS,
    WIDTH,
    dram_sharded,
    large_rows,
    memory_config_for,
    random_input,
    run_against_default_path,
)


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize(
    "case, num_shards",
    [
        ("large", 7),
        ("large_four_shards_short_last", 4),
        ("large_all_banks", None),
    ],
)
def test_height_sharded(device, op_name, case, num_shards):
    num_shards = num_shards or device.dram_grid_size().x  # None: one shard per DRAM bank of this device
    rows = large_rows(device, 1024, 32 * num_shards)
    if case == "large_four_shards_short_last":
        rows -= 64
    shape = [1, 1, rows, 1024]
    run_against_default_path(OPS[op_name], random_input(shape), device, dram_sharded(shape, HEIGHT, num_shards))


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize(
    "case, width, num_shards",
    [
        ("large", 7 * 1024, 7),
        ("large_one_tile_shards", 7 * 32, 7),
        ("large_four_shards", 4 * 1024, 4),
        ("large_all_banks", None, None),
    ],
)
def test_width_sharded(device, op_name, case, width, num_shards):
    num_shards = num_shards or device.dram_grid_size().x  # None: one shard per DRAM bank of this device
    width = width or num_shards * 1024
    rows = large_rows(device, width, 32)
    shape = [1, 1, rows, width]
    run_against_default_path(OPS[op_name], random_input(shape), device, dram_sharded(shape, WIDTH, num_shards))


# Light ops take the queue on every chip.
@pytest.mark.parametrize("op_name", list(OPS))
def test_interleaved(device, op_name):
    shape = [1, 1, large_rows(device, 1024, 32), 1024]
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


# 4 KB pages (fp32) take the flows; bfp8 pages keep the default path.
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat8_b], ids=["fp32", "bfp8"])
@pytest.mark.parametrize("op_name", ["identity", "relu", "silu", "hardswish", "gelu_accurate", "logit"])
@pytest.mark.parametrize("layout", ["height", "width", "interleaved"])
def test_other_formats(device, dtype, op_name, layout):
    width = 7 * 1024 if layout == "width" else 1024
    step = 32 if layout == "width" else 32 * 7
    shape = [1, 1, large_rows(device, width, step), width]
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
