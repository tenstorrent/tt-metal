# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Unary ops on DRAM tensors must give the default read path's output bit for bit in every read flow (see
# unary_dram_reads_utils.py). These cases are small enough for the simulator legs; the ones sized at 600 or more pages
# per core over the whole grid are in test_unary_dram_reads_large.py.

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.eltwise.unary_dram_reads_utils import (
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
@pytest.mark.parametrize("case, num_shards", [("small", 7), ("few_pages_per_core", 7)])
def test_height_sharded(device, op_name, case, num_shards):
    if case == "small":
        rows = 7 * 1024
    else:  # few_pages_per_core: bursts with few cores per bank, the queue with many
        rows = large_rows(device, 1024, 32 * num_shards, pages_per_core=64)
    shape = [1, 1, rows, 1024]
    run_against_default_path(OPS[op_name], random_input(shape), device, dram_sharded(shape, HEIGHT, num_shards))


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize("case, width, num_shards", [("small", 7 * 1024, 7)])
def test_width_sharded(device, op_name, case, width, num_shards):
    shape = [1, 1, 1024, width]
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


# Light ops take bursts at 100 pages per core with many cores per bank and the queue with few. At 460 the default
# split's cores start on 3 banks, where mid ops queue too with few cores per bank. Heavy and unmeasured ops keep the
# default path.
@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize("case", ["small", "bursts", "few_start_banks", "odd_shape"])
def test_interleaved(device, op_name, case):
    if case == "odd_shape":
        shape = [3, 5, 96, 160]
    elif case == "small":
        shape = [1, 1, 7 * 1024, 1024]
    else:
        pages_per_core = {"bursts": 100, "few_start_banks": 460}[case]
        shape = [1, 1, large_rows(device, 1024, 32, pages_per_core), 1024]
    run_against_default_path(OPS[op_name], random_input(shape), device, ttnn.DRAM_MEMORY_CONFIG)


# 4 KB pages (fp32) take the flows; bfp8 pages keep the default path, so they skip the one-page size. At 200 pages per
# core heavy fp32 ops on height shards read one page at a time with many cores per bank.
@pytest.mark.parametrize(
    "dtype, size",
    [
        (ttnn.float32, "small"),
        (ttnn.float32, "one_page"),
        (ttnn.bfloat8_b, "small"),
    ],
    ids=["fp32-small", "fp32-one_page", "bfp8-small"],
)
@pytest.mark.parametrize("op_name", ["identity", "relu", "silu", "hardswish", "gelu_accurate", "logit"])
@pytest.mark.parametrize("layout", ["height", "width", "interleaved"])
def test_other_formats(device, dtype, op_name, layout, size):
    width = 7 * 1024 if layout == "width" else 1024
    if size == "small":
        rows = 1024 if layout == "width" else 7 * 1024
    else:
        step = 32 if layout == "width" else 32 * 7
        rows = large_rows(device, width, step, pages_per_core=200)
    shape = [1, 1, rows, width]
    memory_config = memory_config_for(layout, shape)
    run_against_default_path(OPS[op_name], random_input(shape).to(torch.float32), device, memory_config, dtype)
