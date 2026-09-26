# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multicast bandwidth probe (kernels/mcast_bw.cpp): one sender core, 8 MB in PIECE-byte multicasts to the full
worker grid (2 rectangles, split at the physical column gap) or to one rectangle, on NOC0 or NOC1. Tag
``mcast_{target}_noc{n}_p{piece}_f{flush}``; GB/s = bytes injected per rectangle / time."""

import json
import os
from pathlib import Path

import pytest

import ttnn
from models.demos.mimo_v2_d_p.tests.perf.test_stream_matmul import _crs

TOTAL = 8 << 20
STATS_PATH = Path(os.environ.get("MIMO_SE_STATS", "generated/mimo_stream_expert/cases.jsonl"))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("flush", [0, 3])
@pytest.mark.parametrize("piece", [8192, 16384])
@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("target", ["grid", "left", "col"])
def test_mcast_bw(device, target, noc, piece, flush):
    grid = device.compute_with_storage_grid_size()
    phys = lambda c: device.worker_core_from_logical_core(c)
    pk = lambda c: (phys(c).x << 16) | phys(c).y
    src = ttnn.CoreCoord(5, 5)
    px = [phys(ttnn.CoreCoord(x, 0)).x for x in range(grid.x)]
    runs, start = [], 0
    for x in range(1, grid.x + 1):
        if x == grid.x or px[x] != px[x - 1] + 1:
            runs.append((start, x - 1))
            start = x
    if target == "left":
        runs = runs[:1]
    rects = []
    for x0, x1 in runs:
        y0, y1 = 0, grid.y - 1
        if target == "col":
            x0 = x1 = src.x
        lo, hi = ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)
        if noc == 1:
            lo, hi = hi, lo
        n = (x1 - x0 + 1) * (y1 - y0 + 1) - int(x0 <= src.x <= x1)
        rects += [pk(lo), pk(hi), n]
        if target == "col":
            break
    buf = ttnn.allocate_tensor_on_device(
        ttnn.Shape([256 * 32, 32]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_crs([src]), (256 * 32, 32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    rt = ttnn.RuntimeArgs()
    rt[src.x][src.y] = [buf.buffer_address(), len(rects) // 3] + rects
    k = ttnn.KernelDescriptor(
        kernel_source="models/demos/mimo_v2_d_p/tests/perf/kernels/mcast_bw.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=_crs([src]),
        compile_time_args=[TOTAL, piece, flush],
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0 if noc == 0 else ttnn.NOC.NOC_1
        ),
    )
    program = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
    tag = f"mcast_{target}_noc{noc}_p{piece}_f{flush}"
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a") as f:
        f.write(json.dumps({"tag": tag, "E": 1, "weight_bytes": TOTAL, "flops": 0}) + "\n")
    for _ in range(3):
        ttnn.generic_op([buf], program)
    ttnn.synchronize_device(device)
