# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multicast probe, L1 -> L1 (no DRAM): how fast can the Blackhole NoC broadcast? Patterns:
  grid1   one sender -> the whole worker grid (2 rectangles: the physical column runs)
  grid16  16 senders at once -> the whole grid each
  row10   one sender per grid row -> its own row (disjoint, concurrent)
  col11   one sender per grid column -> its own column (disjoint, concurrent)
Every sender multicasts TOTAL bytes in 16 KB pieces. Tag ``mc_{pattern}_noc{n}``; "weight_bytes" = bytes per sender,
so the analyzer's GB/s is the per-sender rate (multiply by senders for the aggregate injected)."""

import json
import os
from pathlib import Path

import pytest

import ttnn
from models.demos.mimo_v2_d_p.tests.perf.test_stream_matmul import _crs

STATS_PATH = Path(os.environ.get("MIMO_SE_STATS", "generated/mimo_stream_expert/cases.jsonl"))
TOTAL, PIECE, SLOTS = 4 << 20, 16384, 16


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "posted", [int(v) for v in os.environ.get("MIMO_MC_POSTED", "0,1").split(",")], ids=lambda p: f"posted{p}"
)
@pytest.mark.parametrize("noc", [int(v) for v in os.environ.get("MIMO_MC_NOC", "0,1").split(",")])
@pytest.mark.parametrize("pattern", os.environ.get("MIMO_MC_PATTERNS", "grid1,grid16,row10,col11").split(","))
def test_mcast_probe(device, pattern, noc, posted):
    grid = device.compute_with_storage_grid_size()
    phys = lambda c: device.worker_core_from_logical_core(c)
    pk = lambda c: (phys(c).x << 16) | phys(c).y
    px = [phys(ttnn.CoreCoord(x, 0)).x for x in range(grid.x)]
    runs, start = [], 0
    for x in range(1, grid.x + 1):
        if x == grid.x or px[x] != px[x - 1] + 1:
            runs.append((start, x - 1))
            start = x

    def rect(x0, x1, y0, y1, sender):
        lo, hi = ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)
        a, b = (lo, hi) if noc == 0 else (hi, lo)
        inside = x0 <= sender.x <= x1 and y0 <= sender.y <= y1
        return [pk(a), pk(b), (x1 - x0 + 1) * (y1 - y0 + 1) - int(inside)]

    senders = []  # (core, rect args)
    if pattern in ("grid1", "grid16"):
        n = 1 if pattern == "grid1" else 16
        cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][:: max(1, grid.x * grid.y // n)][
            :n
        ]
        for c in cores:
            senders.append((c, [a for x0, x1 in runs for a in rect(x0, x1, 0, grid.y - 1, c)]))
    elif pattern == "half1":  # one sender -> the larger column run only (one rectangle)
        c = ttnn.CoreCoord(0, 0)
        x0, x1 = runs[0]
        senders.append((c, rect(x0, x1, 0, grid.y - 1, c)))
    elif pattern in ("same2", "same4"):  # 2 / 4 senders at once -> the same (larger) column run rectangle
        x0, x1 = runs[0]
        for i in range(int(pattern[-1])):
            c = ttnn.CoreCoord(x0 + i, 0)
            senders.append((c, rect(x0, x1, 0, grid.y - 1, c)))
    elif pattern == "halves4":  # 2 senders per column run, each -> its run
        for x0, x1 in runs:
            for i in range(2):
                c = ttnn.CoreCoord(x0 + i, 0)
                senders.append((c, rect(x0, x1, 0, grid.y - 1, c)))
    elif pattern == "halves2":  # one sender per column run, each -> its own run (disjoint, concurrent)
        for x0, x1 in runs:
            c = ttnn.CoreCoord(x0, 0)
            senders.append((c, rect(x0, x1, 0, grid.y - 1, c)))
    elif pattern == "row10":
        for y in range(grid.y):
            c = ttnn.CoreCoord(y % grid.x, y)
            senders.append((c, [a for x0, x1 in runs for a in rect(x0, x1, y, y, c)]))
    else:  # col11
        for x in range(grid.x):
            c = ttnn.CoreCoord(x, x % grid.y)
            senders.append((c, rect(x, x, 0, grid.y - 1, c)))
    all_crs = _crs([ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)])
    buf = ttnn.allocate_tensor_on_device(
        ttnn.Shape([grid.x * grid.y * 2 * SLOTS * PIECE // 64, 32]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(all_crs, (2 * SLOTS * PIECE // 64, 32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    src, ring = buf.buffer_address(), buf.buffer_address() + SLOTS * PIECE
    rt = ttnn.RuntimeArgs()
    for c, rects in senders:
        rt[c.x][c.y] = [src, ring, len(rects) // 3] + rects
    k = ttnn.KernelDescriptor(
        kernel_source="models/demos/mimo_v2_d_p/tests/perf/kernels/stream_mm/mc_send.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=_crs([c for c, _ in senders]),
        compile_time_args=[TOTAL, PIECE, SLOTS, posted],
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0 if noc == 0 else ttnn.DataMovementProcessor.RISCV_1,
            noc=ttnn.NOC.NOC_0 if noc == 0 else ttnn.NOC.NOC_1,
        ),
    )
    program = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
    tag = f"mc_{pattern}_noc{noc}_p{posted}"
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a") as f:
        f.write(json.dumps({"tag": tag, "E": 1, "weight_bytes": TOTAL, "flops": 0, "senders": len(senders)}) + "\n")
    try:
        from tracy import signpost
    except ImportError:
        signpost = lambda *a, **k: None
    for it in range(4):
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_start")
        ttnn.generic_op([buf, buf], program)
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_end")
