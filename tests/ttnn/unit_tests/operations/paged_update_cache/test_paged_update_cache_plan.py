# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The tests op_design.md's Work Distribution asks for, beyond the acceptance test and the golden suite.

(2) the op in one device session after another program and before one of tt-metal's own (trid hand-back);
(3) the shard on a core other than (0, 0): the op follows the shard;
(4) the index-scratch predicate refuses by name when the budget is lowered;
(5) the writer-thread plan: every thread of the plan does its share (W = 2, 3 with the constant turned);
plus the general path with more than one user and shard core, built below validate() (users = 1 is all
SUPPORTED claims; the kernel and host split are written for any number).
"""

import pytest
import torch
import ttnn

from eval.specs.paged_update_cache.spec import reference
from ttnn.operations.paged_update_cache import paged_update_cache
from ttnn.operations.paged_update_cache import paged_update_cache_program_artifacts as artifacts

TILE = 32


def _sharded_input(device, torch_input, cores):
    """Height-shard [1, users, H, D] with one user (one padded tile row) per core of `cores`."""
    users, D = torch_input.shape[1], torch_input.shape[3]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    assert len(cores) == users
    config = ttnn.create_sharded_memory_config(
        (TILE, D), core_grid=grid, strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
    )
    return ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config)


def _case(device, cache_shape, users, rows, positions, table, cores):
    blocks, heads, block_size, D = cache_shape
    torch.manual_seed(0)
    cache = torch.randn(cache_shape).to(torch.bfloat16)
    x = torch.randn(1, users, rows, D).to(torch.bfloat16)
    idx = torch.tensor(positions, dtype=torch.int32)
    pt = torch.tensor(table, dtype=torch.int32)
    dev = dict(
        cache_tensor=ttnn.from_torch(cache, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG),
        input_tensor=_sharded_input(device, x, cores),
        update_idxs_tensor=ttnn.from_torch(idx, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG),
        page_table=ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG),
    )
    expected = reference(cache, x, idx, pt)
    return dev, expected


def _call(dev):
    return paged_update_cache(dev["cache_tensor"], dev["input_tensor"],
                              update_idxs_tensor=dev["update_idxs_tensor"], page_table=dev["page_table"])


def _assert_cache(tensor, expected):
    got = ttnn.to_torch(tensor).to(torch.float64)
    assert torch.equal(got, expected), f"max diff {(got - expected).abs().max()}"


def test_session_order_before_and_after_other_programs(device):
    """(2) A tt-metal program, then the op (twice), then a tt-metal program of untagged transfers, one session."""
    dev, expected = _case(device, (8, 8, 32, 64), 1, 8, [37], [[3, 1, 4, 0, 5, 2, 7, 6]], [ttnn.CoreCoord(0, 0)])
    # A program before: a tilize of an unrelated tensor.
    warm = ttnn.from_torch(torch.randn(64, 64), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn.to_layout(warm, ttnn.TILE_LAYOUT)
    out = _call(dev)
    out = _call({**dev, "cache_tensor": out})  # the op after itself: same write, same result
    # A program after: tt-metal's untilize of the written cache.
    rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
    assert torch.equal(ttnn.to_torch(rm).to(torch.float64), expected)
    _assert_cache(dev["cache_tensor"], expected)


@pytest.mark.parametrize("core", [(5, 2), (7, 3)])
def test_shard_off_origin(device, core):
    """(3) The input's shard on a core other than (0, 0)."""
    dev, expected = _case(device, (6, 12, 96, 96), 1, 12, [150], [[2, 5, 0, 1, 4, 3]], [ttnn.CoreCoord(*core)])
    out = _call(dev)
    _assert_cache(out, expected)
    _assert_cache(dev["cache_tensor"], expected)


def test_index_scratch_cap_refuses(device, monkeypatch):
    """(4) A page-table row that does not fit the index budget is refused by name, before any program runs."""
    dev, _ = _case(device, (8, 8, 32, 64), 1, 8, [0], [list(range(8))], [ttnn.CoreCoord(0, 0)])
    monkeypatch.setattr(artifacts, "L1_RESERVE_BYTES", ttnn.get_max_worker_l1_unreserved_size() - 16)
    with pytest.raises(NotImplementedError, match="index scratch"):
        _call(dev)


@pytest.mark.parametrize("threads", [2, 3])
def test_writer_thread_plan(device, monkeypatch, threads):
    """(5) The plan turned to W writer threads: units (h, j) dealt round-robin; every unit must be written, so a
    thread that did not run (or dealt wrongly) leaves its share of the rows unwritten."""
    monkeypatch.setattr(artifacts, "ROW_WRITER_THREADS", threads)
    dev, expected = _case(device, (8, 8, 64, 128), 1, 8, [77], [[6, 2, 7, 1, 0, 3, 5, 4]], [ttnn.CoreCoord(0, 0)])
    out = _call(dev)
    _assert_cache(out, expected)


def test_several_users_on_several_shard_cores(device):
    """The general path below validate(): 3 users, one per shard core, one of them skipped (R4)."""
    cores = [ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0)]
    dev, expected = _case(device, (12, 4, 32, 64), 3, 4, [5, -1, 70],
                          [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], cores)
    spec, run_args, indices = artifacts.create_program_artifacts(
        dev["cache_tensor"], dev["input_tensor"], dev["update_idxs_tensor"], dev["page_table"])
    ttnn.generic_op([dev["input_tensor"], dev["update_idxs_tensor"], dev["page_table"], dev["cache_tensor"]],
                    spec, run_args, indices)
    _assert_cache(dev["cache_tensor"], expected)
