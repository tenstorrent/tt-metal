# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolate MESH per-program trace-replay dispatch overhead: 1x1 vs 2x1.

Question (Route B diagnostic): is the ~0.91ms/program trace-replay overhead a
MESH-COORDINATION cost (per-program go-signal/completion across both chips), or
is it per-program regardless of mesh size?

Method (confound-free): capture a trace of N chained TINY ops (32x32 add) so
kernel time ~= 0 and the trace-replay wall is dominated by per-program dispatch.
Sweep N, fit wall(N) = intercept + slope*N. slope = per-program dispatch cost on
that mesh. Compare 1x1 slope vs 2x1 slope.

  - 2x1 slope >> 1x1 slope  => overhead is MESH coordination. Two independent
        single-chip processes (2-process DP) would avoid it => real unlock.
  - 2x1 slope ~= 1x1 slope  => per-program cost is intrinsic per chip; splitting
        the mesh won't help. Go to Route A (op fusion) instead.

No submeshes (avoids the ETH-heartbeat crash), no model, tiny tensors (board-safe).

Run:
  source /localdev/gtobar/bge_optimization/local_env.sh
  export TT_VISIBLE_DEVICES=0
  pytest .../mesh_dispatch_microbench.py -s -q
"""

import time

import pytest
import torch
from loguru import logger

import ttnn

N_SWEEP = [50, 100, 200, 400]


def _measure_chain(mesh_device, n, iters=30):
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 32) * 0.01,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def build():
        y = ttnn.add(x, x)
        for _ in range(n - 1):
            y = ttnn.add(y, x)
        return y

    # compile forward
    out = build()
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = build()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    for _ in range(5):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)

    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
        ts.append((time.perf_counter() - t0) * 1e3)
    ttnn.release_trace(mesh_device, tid)
    ttnn.deallocate(out)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def _fit(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    return slope, intercept


@pytest.mark.parametrize(
    "mesh_device", [(1, 1), (2, 1)], indirect=True, ids=["mesh_1x1", "mesh_2x1"]
)
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 90_000_000, "num_command_queues": 1}], indirect=True
)
def test_mesh_dispatch(mesh_device):
    ndev = mesh_device.get_num_devices()
    logger.info(f"[microbench] mesh has {ndev} device(s); sweeping N={N_SWEEP} chained 32x32 adds")
    xs, ys = [], []
    for n in N_SWEEP:
        mn, md = _measure_chain(mesh_device, n)
        xs.append(n)
        ys.append(mn)
        logger.info(f"[microbench] N={n:4d} programs: trace wall min={mn:8.3f}  med={md:8.3f} ms  "
                    f"({mn/n*1000:.2f} us/program)")
    slope, intercept = _fit(xs, ys)
    logger.info("=" * 74)
    logger.info(f"  MESH {ndev}-device  wall(N) = {intercept:.3f} + {slope*1000:.2f} us/program * N")
    logger.info(f"  => per-program trace-replay dispatch = {slope*1000:.2f} us/program  "
                f"(fixed intercept {intercept:.2f} ms)")
    logger.info("=" * 74)
    logger.info(f"METRIC mesh{ndev}_us_per_program {slope*1000:.3f}")
