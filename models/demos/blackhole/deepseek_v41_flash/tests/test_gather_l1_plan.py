# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.gather`` (codegen) program hash vs the live L1 frontier: the "vals" gather of the column-sharded prefill indexer (tt/prefill_sparse.py ``select_dyn_sharded``) at S_pad 524288.

The score row is Lc + 32 = 65568 columns = 2049 tiles = 3 * 683. A row that does not fit L1 is streamed in blocks of ceil(2049 / n) tiles, n = ceil(2049 / affordable), affordable = (usable L1 - 3 pages) /
2048 B with usable L1 = lowest occupied L1 address - allocator base; both the block depth and n are part of the program hash. A 683-tile budget (1,404,928 B usable per core) is the boundary of n = 3 / 4, so one
128 B persistent L1 buffer (what the tail of the compile pass of the model allocates) turns a cached program into a miss: a miss inside a trace capture is "Cannot load new binaries during trace capture".

The test sweeps the L1 bytes held per bank in 128 B steps around that boundary and prints the program cache entry count after every gather: it grows by exactly one at the boundary and nowhere else. It then
captures a trace at the plan it warmed last (a hit)."""

import pytest
import torch

import ttnn

DP = {"l1_small_size": 16384, "trace_region_size": 20_000_000}
C, WT, TOPK = 512, 2049, 512
PAGE = 2048  # bf16 tile


def usable_after_hold(md, hold):
    mv = ttnn.get_memory_view(md, ttnn.BufferType.L1)
    return mv.total_bytes_free_per_bank - hold, mv.num_banks


def hold_l1(md, nbytes, nb):
    """``nbytes`` of L1 on every bank (one page per bank), allocated top-down: lowers the frontier by ``nbytes`` (when nothing else is allocated)."""
    if nbytes == 0:
        return None
    return ttnn.empty(
        [nb, nbytes], dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=md, memory_config=ttnn.L1_MEMORY_CONFIG
    )


def plan(usable, wt=WT, page=PAGE, index_page=4096):
    affordable = max((usable - (index_page + page)) // page, 2)
    return -(-wt // min(affordable, wt))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="l1small")], indirect=True)
@torch.no_grad()
def test_gather_plan_follows_l1_frontier(mesh_device):
    md = mesh_device
    scm = ttnn.from_torch(
        torch.randn(1, 1, C, WT * 32).bfloat16(),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    idc = ttnn.from_torch(
        torch.randint(0, WT * 32, (1, 1, C, TOPK), dtype=torch.int32),
        device=md,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    free0, nb = usable_after_hold(md, 0)
    boundary = free0 - (683 * PAGE + 4096 + PAGE)  # held bytes per bank above which the budget is < 683 tiles
    print(f"GP free L1 per bank {free0} B, {nb} banks; n=3 / n=4 boundary: {boundary} B held", flush=True)
    rows, prev = [], None
    for hold in range(max(boundary - 512, 0), boundary + 640, 128):
        h = hold_l1(md, hold, nb)
        out = ttnn.gather(scm, 3, index=idc)
        ttnn.synchronize_device(md)
        n_entries = md.num_program_cache_entries()
        free = ttnn.get_memory_view(md, ttnn.BufferType.L1).total_bytes_free_per_bank
        rows.append((hold, free, plan(free), n_entries))
        print(
            f"GP held {hold:6d} B  free {free}  predicted n {plan(free)}  program cache entries {n_entries}", flush=True
        )
        ttnn.deallocate(out)
        if h is not None:
            ttnn.deallocate(h)
    steps = [(a, b) for a, b in zip(rows, rows[1:]) if b[3] != a[3]]
    plan_steps = [(a, b) for a, b in zip(rows, rows[1:]) if b[2] != a[2]]
    print(
        f"GP cache-entry steps at {[b[0] for _, b in steps]}, predicted plan steps at {[b[0] for _, b in plan_steps]}",
        flush=True,
    )
    assert len(plan_steps) == 1, "the sweep must cross the n=3/4 boundary once"
    assert [b[0] for _, b in steps][-1:] == [
        b[0] for _, b in plan_steps
    ], "a new program exactly where the plan changes"

    # the trace capture finds its program when the compile pass ran at the same plan (the model's fix: warm at the capture-time frontier)
    h = hold_l1(md, rows[-1][0], nb)
    before = md.num_program_cache_entries()
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    out = ttnn.gather(scm, 3, index=idc)
    ttnn.end_trace_capture(md, tid, cq_id=0)
    assert md.num_program_cache_entries() == before, "the capture compiled a program"
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    print("GP trace capture + replay at the warmed plan OK", flush=True)
    ttnn.release_trace(md, tid)
