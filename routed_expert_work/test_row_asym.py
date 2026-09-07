# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Minimal reproduction of the per-grid-row DRAM read asymmetry, using only stock ops.

`ttnn.clone` on a DRAM-interleaved tile tensor gives every participating core the
same number of pages to read and write, so the per-core NCRISC (reader) kernel
duration is a direct, work-equalised measurement of "how fast does DRAM serve
THIS core".  Sweeping the core grid separates two hypotheses:

  positional  - a row is slow because of where it sits relative to the DRAM
                endpoints; a row read in isolation is as slow as it is in a crowd.
  contention  - a row is only slow when every other core is reading too; alone
                it is as fast as any other row.

Run with the device profiler on, e.g.
  TT_METAL_DEVICE_PROFILER=1 pytest routed_expert_work/test_row_asym.py -s
then aggregate with rowprobe.py --zone NCRISC-KERNEL.
"""
import csv
import os
from collections import defaultdict

import pytest
import torch
import ttnn

FREQ_MHZ = 1350.0
CSV = "generated/profiler/.logs/profile_log_device.csv"


def _per_core_kernel_us(zone="NCRISC-KERNEL", risc="NCRISC"):
    """Longest single occurrence of `zone` per core, in us (one per program run).

    The profiler only writes the CSV out on a flush, so a test that runs before any
    flush has happened has nothing to read; skip rather than fail.
    """
    if not os.path.exists(CSV):
        pytest.skip(f"no profiler CSV at {CSV} -- run with TT_METAL_DEVICE_PROFILER=1")
    with open(CSV) as f:
        lines = f.read().splitlines()
    hdr = [h.strip() for h in lines[1].split(",")]
    open_, best = {}, defaultdict(float)
    for rec in csv.DictReader(lines[2:], fieldnames=hdr):
        if (rec.get("zone name") or "").strip() != zone:
            continue
        if (rec.get("RISC processor type") or "").strip() != risc:
            continue
        key = (int(rec["core_x"]), int(rec["core_y"]))
        t = int(rec["time[cycles since reset]"])
        if rec["type"].strip() == "ZONE_START":
            open_[key] = t
        elif key in open_:
            best[key] = max(best[key], (t - open_.pop(key)) / FREQ_MHZ)
    return best


def _report(label, per_core):
    by_row = defaultdict(list)
    for (x, y), v in per_core.items():
        by_row[y].append(v)
    print(f"\nROWASYM {label}  ({len(per_core)} cores)")
    print("  phys y   cores   mean us   min us   max us")
    means = {}
    for y in sorted(by_row):
        v = by_row[y]
        means[y] = sum(v) / len(v)
        print(f"    {y:4d}   {len(v):5d}   {means[y]:7.1f}   {min(v):6.1f}   {max(v):6.1f}")
    if len(means) > 1:
        lo, hi = min(means.values()), max(means.values())
        order = sorted(means, key=means.get, reverse=True)
        print(f"  row spread {lo:.1f}..{hi:.1f} us = {hi/lo:.2f}x   slowest {order[:3]} fastest {order[-3:]}")
    return means


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("mb", [int(v) for v in os.environ.get("ROWASYM_MB", "16,64").split(",")])
def test_clone_row_asymmetry(device, dtype, mb):
    """All-88-core read: every core reads the same page count. Shows the crowd profile."""
    tiles = mb * 1024 * 1024 // 2048
    rows = max(1, tiles // 64)
    t = ttnn.from_torch(torch.randn(rows * 32, 64 * 32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    for _ in range(3):
        ttnn.clone(t)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    _report(f"clone {mb}MB {dtype}", _per_core_kernel_us())


@pytest.mark.parametrize("grid", [(1, 1), (2, 2), (4, 4), (7, 8)])
def test_row_isolation(device, grid):
    """Same total bytes, spread over a shrinking core grid.

    A (1,1)/(2,2) grid reads with almost no competition. If the row ordering
    survives at low core counts the effect is positional; if it collapses the
    effect is arbitration under contention.
    """
    gx, gy = grid
    ncores = gx * gy
    per_core_tiles = 16
    tiles = ncores * per_core_tiles
    t = ttnn.from_torch(
        torch.randn(tiles // 8 * 32, 8 * 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    cfg = ttnn.create_sharded_memory_config(
        t.shape,
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    for _ in range(3):
        ttnn.to_memory_config(t, cfg)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    _report(f"gather grid={gx}x{gy} ({ncores} cores, {per_core_tiles} tiles/core)", _per_core_kernel_us())
