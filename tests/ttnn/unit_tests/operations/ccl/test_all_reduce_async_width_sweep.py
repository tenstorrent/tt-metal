# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 decode all-reduce width sweep (ccl-overlap P0(b)): zero-code upper bound for row-fold packing.

Times all_reduce_async (persistent-buffer overload, RES40-style 40-core width sharding) on [32, W] for
W in {5120, 2560, 1280}: [32,1280] carries exactly the bytes and fabric packets a fold=4 row-fold of [<=8, 5120]
would send, so T(5120) - T(1280) is the most row-fold can save per all-reduce, and a fit of T against W splits the
per-op floor from the per-byte cost. Also times live_rows_packing at 8 rows, use_noc1_only {False, True} and
num_links {1, 2}. Each point is a traced loop of ITERS all-reduces replayed REPLAYS times (no host dispatch in
the timing); the per-op time is wall time / (ITERS * REPLAYS).

Gates (design ccl-overlap): row-fold (P1) only if T(5120) - T(1280) >= 8 us; the fused matmul+AR (P3) only if the
post-fold byte-proportional part is >= 6 us.

Run (8-chip ring, e.g. a24u08 row 0, drained):
  MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=0,1,2,3,27,26,25,24 \
  pytest tests/ttnn/unit_tests/operations/ccl/test_all_reduce_async_width_sweep.py -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn

GRID = (10, 4)  # RES40
ITERS = 64
REPLAYS = 20
RESULTS = {}


def _width_mem(shard_w):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID[0] - 1, GRID[1] - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [ttnn.TILE_SIZE, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


@pytest.mark.parametrize("use_noc1_only", [False, True], ids=["noc0w", "noc1w"])
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2link"])
@pytest.mark.parametrize(
    "width, rows, live_rows_packing",
    [(5120, 32, False), (2560, 32, False), (1280, 32, False), (5120, 8, True)],
    ids=["w5120", "w2560", "w1280", "w5120-liverows8"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 4 << 20}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 8), id="1x8")], indirect=True)
def test_all_reduce_async_width_sweep(mesh_device, width, rows, live_rows_packing, num_links, use_noc1_only):
    tp = list(mesh_device.shape)[1]
    ncores = GRID[0] * GRID[1]
    mem = _width_mem(width // ncores)
    buf_mem = _width_mem(width * tp // ncores)
    sg = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(sg.x - 1, sg.y - 1))})
    # One semaphore per traced op (the model cycles semaphores the same way; reuse inside a trace would race).
    sems = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(ITERS)]

    torch.manual_seed(width + rows)
    full = torch.randn(1, tp, rows, width).to(torch.bfloat16)
    inp = ttnn.from_torch(
        full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=mem,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )
    buf = ttnn.from_torch(
        torch.zeros([1, 1, ttnn.TILE_SIZE, width * tp]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=buf_mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    extra = {}
    if live_rows_packing:
        extra["live_rows_packing"] = True
    if use_noc1_only:
        extra["use_noc1_only"] = True

    def step(i):
        return ttnn.experimental.all_reduce_async(
            inp,
            buf,
            cluster_axis=1,
            mesh_device=mesh_device,
            multi_device_global_semaphore=sems[i],
            topology=ttnn.Topology.Ring,
            num_links=num_links,
            memory_config=mem,
            **extra,
        )

    # Compile + correctness on one eager call.
    out = step(0)
    ttnn.synchronize_device(mesh_device)
    ref = full.float().sum(dim=1, keepdim=True)
    for dev, d in enumerate(ttnn.get_device_tensors(out)):
        got = ttnn.to_torch(d).float()[..., :rows, :]
        err = (got - ref).abs().max().item()
        assert err <= 0.05 * ref.abs().max().item(), f"device {dev}: |all_reduce - torch sum| = {err:.3e}"
    ttnn.deallocate(out)
    for s in sems:
        ttnn.reset_global_semaphore_value(s, 0)

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outs = [step(i) for i in range(ITERS)]
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)  # warm
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) * 1e6 / (ITERS * REPLAYS)
    ttnn.release_trace(mesh_device, tid)
    for t in outs + [inp, buf]:
        ttnn.deallocate(t)

    key = (num_links, use_noc1_only)
    RESULTS.setdefault(key, {})[(width, rows, live_rows_packing)] = us
    logger.info(
        f"AR_SWEEP width={width} rows={rows} live_rows={live_rows_packing} links={num_links} "
        f"noc1_only={use_noc1_only}: {us:.2f} us/op"
    )
    pts = RESULTS[key]
    if (5120, 32, False) in pts and (1280, 32, False) in pts:
        gain = pts[(5120, 32, False)] - pts[(1280, 32, False)]
        logger.info(
            f"AR_SWEEP links={num_links} noc1_only={use_noc1_only}: T(5120)-T(1280) = {gain:.2f} us "
            f"(row-fold P1 gate >= 8 us: {'PASS' if gain >= 8 else 'FAIL'})"
        )
        if (2560, 32, False) in pts:
            # Least-squares T = floor + slope * W over the three widths.
            ws = [5120, 2560, 1280]
            ts = [pts[(w, 32, False)] for w in ws]
            mw, mt = sum(ws) / 3, sum(ts) / 3
            slope = sum((w - mw) * (t - mt) for w, t in zip(ws, ts)) / sum((w - mw) ** 2 for w in ws)
            floor = mt - slope * mw
            logger.info(
                f"AR_SWEEP links={num_links} noc1_only={use_noc1_only}: floor={floor:.2f} us, "
                f"byte-proportional at 1280={slope * 1280:.2f} us (P3 gate >= 6 us), at 5120={slope * 5120:.2f} us"
            )
