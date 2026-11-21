# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests the design requires beyond the acceptance test (op_design.md, Work Distribution):

- the thread plan read off the host program: one reader thread, one writer thread, no compute kernel,
  over a multi-cluster split (with one thread per kind, a correct result over many clusters shows every
  thread ran);
- the walk's hard cases: a cluster whose range starts and ends mid tile row, a skipped (-1) block inside a
  multi-cluster split, wide heads split over the whole grid;
- one device session: another program before the op, and tt-metal's own `to_layout` after it (the
  sticky NoC transaction tag; target page, Ring protocols).
"""

import pytest
import torch

import ttnn
from eval.specs.paged_fill_cache.spec import reference
from ttnn.operations.paged_fill_cache import paged_fill_cache
from ttnn.operations.paged_fill_cache.paged_fill_cache_program_artifacts import (
    THREAD_PLAN,
    TILE_RING_ENTRIES,
    create_program_artifacts,
    split_tiles,
)


def _put(device, cache, x, table):
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return (
        ttnn.from_torch(cache, **tile),
        ttnn.from_torch(x, **tile),
        ttnn.from_torch(
            table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        ),
    )


def _case(device, cache_shape, input_shape, table, batch_idx, seed=0):
    torch.manual_seed(seed)
    cache = torch.randn(cache_shape).bfloat16()
    x = torch.randn(input_shape).bfloat16()
    table = torch.tensor(table, dtype=torch.int32)
    return cache, x, table, _put(device, cache, x, table)


def test_plan_read_off_the_host_program(device):
    """(R, C, W) = (1, 0, 1): one reader, one writer, no compute kernel, on every engaged cluster."""
    cache, x, table, (tc, tx, tp) = _case(device, (128, 8, 32, 64), (1, 8, 512, 64), [list(range(16))], 0)
    spec, _, _ = create_program_artifacts(tc, tx, tp, 0)
    kernels = {k.unique_id: k for k in spec.kernels}
    assert set(kernels) == {"reader", "writer"}
    assert not any(k.is_compute_kernel() for k in spec.kernels)
    assert (kernels["reader"].num_threads, 0, kernels["writer"].num_threads) == THREAD_PLAN == (1, 0, 1)
    rings = {d.unique_id: d for d in spec.dataflow_buffers}
    assert rings["input_tiles"].num_entries == TILE_RING_ENTRIES * THREAD_PLAN[0]
    # 256 tiles fill the 32-cluster grid, 8 tiles each.
    _, counts, _ = split_tiles(8 * 16 * 2, device.compute_with_storage_grid_size())
    grid = device.compute_with_storage_grid_size()
    assert len(counts) == min(grid.x * grid.y, 256)


@pytest.mark.parametrize(
    "cache_shape,input_shape,table,batch_idx",
    [
        # 320 tiles, 10 per cluster, tile rows of 32: ranges start and end mid row; block 1 skipped.
        ((8, 2, 64, 1024), (1, 2, 160, 1024), [[0, 0, 0], [5, -1, 2]], 1),
        # 40 tiles over 32 clusters (2 or 1 each), the third block skipped.
        ((16, 1, 32, 256), (1, 1, 160, 256), [[3, 9, -1, 1, 0]], 0),
        # One tile row 256 tiles wide over the whole grid.
        ((2, 1, 32, 8192), (1, 1, 32, 8192), [[1], [0]], 0),
        # Every block skipped: nothing is written.
        ((4, 2, 32, 64), (1, 2, 64, 64), [[-1, -1]], 0),
    ],
    ids=["mid_row_ranges_skip", "uneven_split_skip", "wide_8192", "all_skipped"],
)
def test_walk_cases(device, cache_shape, input_shape, table, batch_idx):
    cache, x, pt, (tc, tx, tp) = _case(device, cache_shape, input_shape, table, batch_idx)
    out = paged_fill_cache(tc, tx, tp, batch_idx=batch_idx)
    expected = reference(cache, x, pt, batch_idx)
    assert torch.equal(ttnn.to_torch(tc).double(), expected)
    assert torch.equal(ttnn.to_torch(out).double(), expected)


def test_one_device_session(device):
    """Another program, then the op, then a program of tt-metal's own (to_layout: untagged transfers),
    then the op again: each result checked, in one device session."""
    cache, x, pt, (tc, tx, tp) = _case(device, (8, 2, 64, 1024), (1, 2, 160, 1024), [[0, 0, 0], [5, -1, 2]], 1, seed=1)
    # Another program first.
    before = ttnn.to_torch(ttnn.to_layout(tx, ttnn.ROW_MAJOR_LAYOUT))
    assert torch.equal(before, x)
    out = paged_fill_cache(tc, tx, tp, batch_idx=1)
    expected = reference(cache, x, pt, 1)
    # tt-metal's own program right after ours.
    row_major = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
    assert torch.equal(ttnn.to_torch(ttnn.from_device(row_major)).double(), expected)
    # And the op once more after it, through the tensor it returned.
    x2 = (x.float() * -1).bfloat16()
    tx2 = ttnn.from_torch(x2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    paged_fill_cache(out, tx2, tp, batch_idx=1)
    assert torch.equal(ttnn.to_torch(tc).double(), reference(expected, x2, pt, 1))
