# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device time of ``fused_lightning_select_kv`` for a B=4 batch step's selects: four one-row calls (one per session,
what ``DeepSeekV41Indexer.select_kv_batch`` runs today) against one call for all four sessions.

The one call uses the candidate-pool consumer's inputs: a page table that lists the four sessions' key blocks one
after the other, a ``row_table`` with each key's ``kv_cache`` row, and a ``row_mask_table`` that lets query row ``j``
select only session ``j``'s blocks (``num_keys_tensor`` is the total for every row). Every key is then scored for all
four rows and masked, so this is an upper bound on a select that takes a page-table row per query row. ``sq4_same``
is four rows over one session's keys (DSpark verify's multi-row select), the cost without the extra key reads.

V4.1 shapes: 32 index heads of 128, top-512, 64-key blocks, interleaved TILE keys, ROW_MAJOR kv rows of 512. Run::

    python -m tracy -r -p -v --no-web-server -t 8160 -m pytest <this file>
    python <this file> <ops_perf_results csv>
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
from collections import defaultdict

import pytest
import torch

import ttnn

try:
    from tracy import signpost
except ImportError:  # pragma: no cover

    def signpost(*args, **kwargs):
        pass


HEADS, DIM, TOPK, BLOCK, KV_DIM, RATE = 32, 128, 512, 64, 512, 4
ROWS = 4
KEY_COUNTS = tuple(int(k) for k in os.environ.get("SELECT_PERF_KEYS", "512,1024,2048,8192,32768").split(","))
ITERATIONS = 5


def _full_grid(device) -> ttnn.CoreRangeSet:
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})


def _replicated(x: torch.Tensor, device) -> ttnn.Tensor:
    """``[1, H, S, W]`` -> ROW_MAJOR HEIGHT_SHARDED L1, one full replica per core (the select's query layout)."""
    grid = _full_grid(device)
    n = grid.num_cores()
    _, heads, rows, width = x.shape
    mem = ttnn.create_sharded_memory_config(
        (heads * rows, width),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(
        x.repeat(n, 1, 1, 1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem,
    )


def _int32(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="fused_lightning_select_kv is Blackhole-only")
def test_select_sessions_profile(device) -> None:
    torch.manual_seed(0)
    max_keys = max(KEY_COUNTS)
    blocks_per_session = max_keys // BLOCK
    num_blocks = ROWS * blocks_per_session
    # One shared pool of blocks, each session's in shuffled order (as the paged caches are).
    perm = torch.randperm(num_blocks).reshape(ROWS, blocks_per_session)
    key_tt = ttnn.from_torch(
        torch.randn(num_blocks, 1, BLOCK, DIM).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    kv_tt = ttnn.from_torch(
        torch.randn(num_blocks, 1, BLOCK, KV_DIM).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    session_table = _int32(perm, device)  # [ROWS, blocks_per_session]
    page_rows = [_int32(torch.tensor([j]), device) for j in range(ROWS)]

    def run(name, fn):
        fn()
        ttnn.synchronize_device(device)
        signpost(name)
        for _ in range(ITERATIONS):
            fn()
            ttnn.synchronize_device(device)
        signpost(f"{name}-end")

    for keys in KEY_COUNTS:
        n_blocks = keys // BLOCK
        last_pos = keys * RATE - 1
        q1 = _replicated(torch.randn(1, HEADS, 1, DIM), device)
        w1 = _replicated(torch.randn(1, 1, 1, HEADS) * DIM**-0.5, device)
        pos1 = _int32(torch.tensor([last_pos]), device)

        def one_row(j=0):
            outs = ttnn.experimental.deepseek.fused_lightning_select_kv(
                q1, key_tt, w1, kv_tt, session_table, pos1, TOPK, page_table_row_tensor=page_rows[j]
            )
            for t in outs:
                ttnn.deallocate(t)

        def four_one_row():
            for j in range(ROWS):
                one_row(j)

        q4 = _replicated(torch.randn(1, ROWS * HEADS, 1, DIM), device)
        w4 = _replicated(torch.randn(1, 1, ROWS, HEADS) * DIM**-0.5, device)
        pos4 = _int32(torch.full((ROWS,), last_pos), device)

        def sq4_same():
            outs = ttnn.experimental.deepseek.fused_lightning_select_kv(
                q4, key_tt, w4, kv_tt, session_table, pos4, TOPK, page_table_row_tensor=page_rows[0]
            )
            for t in outs:
                ttnn.deallocate(t)

        # The four sessions' first n_blocks blocks back to back; key i of the list maps to its block's kv rows.
        cat = perm[:, :n_blocks].reshape(1, -1)
        cat_table = _int32(cat, device)
        rows = (cat.reshape(-1, 1) * BLOCK + torch.arange(BLOCK)).reshape(1, -1)
        row_table = _int32(rows, device)
        masks = (1 << torch.arange(ROWS)).repeat_interleave(n_blocks).reshape(1, -1)
        mask_table = _int32(masks, device)
        total = _int32(torch.full((ROWS,), ROWS * keys), device)

        def sessions4():
            outs = ttnn.experimental.deepseek.fused_lightning_select_kv(
                q4,
                key_tt,
                w4,
                kv_tt,
                cat_table,
                pos4,
                TOPK,
                num_keys_tensor=total,
                row_table=row_table,
                row_mask_table=mask_table,
            )
            for t in outs:
                ttnn.deallocate(t)

        run(f"{keys}-one_row", one_row)
        run(f"{keys}-four_one_row", four_one_row)
        run(f"{keys}-sq4_same", sq4_same)
        run(f"{keys}-sessions4", sessions4)
        for t in (q1, w1, pos1, q4, w4, pos4, cat_table, row_table, mask_table, total):
            ttnn.deallocate(t)


def _summarize(path: str) -> None:
    """Median device kernel time of the select ops per signposted configuration, per call and per configuration."""
    current, per = None, defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["OP TYPE"] == "signpost":
                name = row["OP CODE"]
                current = None if name.endswith("-end") else name
                continue
            if current and "LightningSelect" in row["OP CODE"]:
                per[current].append(float(row["DEVICE KERNEL DURATION [ns]"]) / 1e3)
    print(f"{'config':<24}{'calls':>6}{'us/call':>9}{'us/config':>11}")
    for name, times in per.items():
        calls = len(times) // ITERATIONS
        groups = [sum(times[i : i + calls]) for i in range(0, len(times), calls)]
        print(f"{name:<24}{calls:6d}{statistics.median(times):9.1f}{statistics.median(groups):11.1f}")


if __name__ == "__main__":
    _summarize(sys.argv[1])
