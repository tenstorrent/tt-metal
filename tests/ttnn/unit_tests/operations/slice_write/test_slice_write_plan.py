# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""slice_write beyond the acceptance test: the thread plan, shard configurations the golden suite does not
build (the caller's own single shard, a ragged last shard, a column-major grid), the writer-hart knob, and the
op inside one device session with other programs (target page, ring protocols)."""

import pytest
import torch

import ttnn
from ttnn.operations.slice_write import slice_write
from ttnn.operations.slice_write.slice_write_program_artifacts import THREAD_PLAN, create_program_artifacts


def _sharded_source(device, block, shard_rows, num_cores, orientation=ttnn.ShardOrientation.ROW_MAJOR, seed=0):
    torch.manual_seed(seed)
    src = torch.randn(*block, dtype=torch.bfloat16)
    grid = device.compute_with_storage_grid_size()
    if orientation == ttnn.ShardOrientation.ROW_MAJOR:
        cores = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    else:  # a rectangle of whole columns, read column by column
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                                  ttnn.CoreCoord(num_cores // grid.y - 1, grid.y - 1))])
    config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(cores, [shard_rows, block[-1]], orientation))
    return src, ttnn.from_torch(src, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                memory_config=config)


def _destination(device, shape, seed=1):
    torch.manual_seed(seed)
    dst = torch.randn(*shape, dtype=torch.bfloat16)
    return dst, ttnn.from_torch(dst, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _expected(dst, src, start, end):
    out = dst.clone()
    out[tuple(slice(a, b) for a, b in zip(start, end))] = src
    return out


def test_thread_plan_is_built(device):
    """The plan read off the built ProgramSpec: a reader and a writer with the plan's thread counts, no compute."""
    _, src_dev = _sharded_source(device, (1, 32, 32, 32), 32, 32)
    _, dst_dev = _destination(device, (1, 64, 32, 32))
    spec, _, _ = create_program_artifacts(src_dev, dst_dev, (0, 0, 0, 0), (1, 32, 32, 32))
    threads = {k.unique_id: k.num_threads for k in spec.kernels}
    assert threads == {"reader": THREAD_PLAN[0], "writer": THREAD_PLAN[2]}
    assert THREAD_PLAN[1] == 0


CASES = [
    # (block, destination, start, shard_rows, cores, orientation)
    # ResNet-50's traced call as the caller makes it: one shard of 6785 rows on one core.
    ((1, 59, 115, 32), (1, 115, 115, 32), (0, 0, 0, 0), 6785, 1, ttnn.ShardOrientation.ROW_MAJOR),
    # The model's second write, 32 cores with a ragged last shard (202 rows each, the last 178).
    ((1, 56, 115, 32), (1, 115, 115, 32), (0, 59, 0, 0), 202, 32, ttnn.ShardOrientation.ROW_MAJOR),
    # A grid provisioned wider than the data: shards past the last row hold nothing.
    ((1, 3, 17, 32), (1, 6, 17, 32), (0, 2, 0, 0), 8, 8, ttnn.ShardOrientation.ROW_MAJOR),
    # A column-major shard grid: shard i lives on the i-th core in column order.
    ((2, 3, 32, 64), (4, 6, 32, 64), (1, 2, 0, 0), 24, 8, ttnn.ShardOrientation.COL_MAJOR),
]


@pytest.mark.parametrize("block,shape,start,shard_rows,cores,orientation", CASES,
                         ids=["traced_one_shard", "ragged_last_shard", "grid_wider_than_data", "col_major_grid"])
def test_shard_configurations(device, block, shape, start, shard_rows, cores, orientation):
    end = tuple(s + b for s, b in zip(start, block))
    src, src_dev = _sharded_source(device, block, shard_rows, cores, orientation)
    dst, dst_dev = _destination(device, shape)
    out = slice_write(src_dev, dst_dev, list(start), list(end), [1, 1, 1, 1])
    assert out is dst_dev
    assert torch.equal(ttnn.to_torch(out), _expected(dst, src, start, end))
    assert torch.equal(ttnn.to_torch(src_dev), src)


@pytest.mark.parametrize("writers", [1, 2, 4])
def test_writer_harts_knob(device, writers):
    """THREAD_PLAN's W is a live knob: the shard is cut into W ring entries, writer hart t stores entry t.
    A hart that did not run leaves its rows stale against the random destination."""
    block, shape, start = (1, 4, 32, 32), (1, 8, 32, 32), (0, 3, 0, 0)
    end = tuple(s + b for s, b in zip(start, block))
    src, src_dev = _sharded_source(device, block, 40, 4)  # 128 rows, 4 shards of 40 (the last 8): ragged entries
    dst, dst_dev = _destination(device, shape)
    spec, run_args, tensor_args = create_program_artifacts(src_dev, dst_dev, start, end, threads=(1, 0, writers))
    ttnn.generic_op([src_dev, dst_dev], spec, run_args, tensor_args)
    assert torch.equal(ttnn.to_torch(dst_dev), _expected(dst, src, start, end))


def test_programs_before_and_after_in_one_session(device):
    """One device session: a tt-metal program first (tilize), then slice_write twice (the model's two writes),
    then a tt-metal program on its result (to_layout to tile and back). Each step is checked."""
    torch.manual_seed(7)
    x = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    x_dev = ttnn.to_layout(ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device), ttnn.TILE_LAYOUT)
    assert torch.equal(ttnn.to_torch(x_dev), x)

    shape = (1, 64, 32, 32)
    dst, dst_dev = _destination(device, shape)
    expected = dst
    for seed, (start, end) in enumerate([((0, 0, 0, 0), (1, 32, 32, 32)), ((0, 32, 0, 0), (1, 64, 32, 32))]):
        block = tuple(b - a for a, b in zip(start, end))
        src, src_dev = _sharded_source(device, block, 32, 32, seed=seed + 10)
        slice_write(src_dev, dst_dev, start, end, (1, 1, 1, 1))
        expected = _expected(expected, src, start, end)
    assert torch.equal(ttnn.to_torch(dst_dev), expected)

    tiled = ttnn.to_layout(dst_dev, ttnn.TILE_LAYOUT)
    assert torch.equal(ttnn.to_torch(tiled), expected)
    back = ttnn.to_layout(tiled, ttnn.ROW_MAJOR_LAYOUT)
    assert torch.equal(ttnn.to_torch(back), expected)


@pytest.mark.parametrize("writers", [2, 4])
def test_writer_harts_fall_back_when_rows_do_not_divide(device, writers):
    """A plan whose W does not divide the shard's rows runs at W = 1 (op_design.md, Perf lamps) instead of
    refusing a valid call: 51 rows in one shard."""
    block, shape, start = (1, 3, 17, 32), (1, 6, 17, 32), (0, 2, 0, 0)
    end = tuple(s + b for s, b in zip(start, block))
    src, src_dev = _sharded_source(device, block, 51, 1)
    dst, dst_dev = _destination(device, shape)
    spec, run_args, tensor_args = create_program_artifacts(src_dev, dst_dev, start, end, threads=(1, 0, writers))
    assert {k.unique_id: k.num_threads for k in spec.kernels}["writer"] == 1
    ttnn.generic_op([src_dev, dst_dev], spec, run_args, tensor_args)
    assert torch.equal(ttnn.to_torch(dst_dev), _expected(dst, src, start, end))
