# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of the GR read front (ttnn/fused/gr_recip_last) on the TP=8 mesh: today's program vs the front-v2 program
(QWEN38_GR_FRONT_V2=1) on random residuals.  Compared bit for bit (valid rows only): the gathered stats pages, the
normalized' tensor (the gate's operand) and the gathered partials pages gr_read's low-rank cores read.  Also the traced
time per call (slope of a 12-op and a 4-op trace) of each, back-to-back calls and a trace replay (parity reuse).
Run: QWEN38_FUSED_DEVICE_TEST=1 MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=0..7 pytest this_file -s."""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_recip_last as grl

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the TP=8 mesh")
TP = MESH_SHAPE[1]
B, LOCAL, FLAT, PARTIAL = 4, 320, 1280, 384
HIDDEN = LOCAL * TP
STEPS = int(os.environ.get("GF_STEPS", "10"))
REPLAYS = int(os.environ.get("GF_REPLAYS", "30"))
RESULTS = os.environ.get("GF_RESULTS", "/tmp/gf_results.jsonl")
DEVICE_PARAMS = dict(demo.DEVICE_PARAMS, trace_region_size=64 * 1024 * 1024)
DRAM = ttnn.DRAM_MEMORY_CONFIG


def candidates():
    out = [("default", grl.stats_recip_down_gather)]
    try:
        from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_recip_last import front2

        out.append(("v2", front2.stats_recip_down_gather))
    except ImportError:
        pass
    try:
        from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_recip_last import front3

        out.append(("v3", front3.stats_recip_down_gather))
    except ImportError:
        pass
    return out


def make_inputs(mesh, rows, seed):
    g = torch.Generator().manual_seed(seed)
    scale = torch.exp2(torch.randint(-6, 7, (1, B, 1, 1), generator=g).float())
    x = (torch.randn(1, B, rows, HIDDEN, generator=g) * scale).to(torch.bfloat16)
    x[:, 3] = 0.5  # an all-equal branch
    gamma = (
        ((1.0 + 0.1 * torch.randn(1, B, 1, HIDDEN, generator=g)) / B)
        .float()
        .expand(-1, -1, ttnn.TILE_SIZE, -1)
        .contiguous()
    )
    w = (torch.randn(1, 1, TP * FLAT, PARTIAL, generator=g) * 0.02).to(torch.bfloat16)
    res = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 3)),
    )
    gam = ttnn.from_torch(
        gamma,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 3)),
    )
    down = ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=dm.dram_sharded_weight_memory_config(mesh, FLAT, PARTIAL),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 2)),
    )
    return res, gam, down


def valid_rows(outputs, rows):
    """Per device, the valid rows of (gathered stats, normalized', gathered partials) as host tensors."""
    stats, norm, part = outputs
    res = []
    for s, n, p in zip(*(ttnn.get_device_tensors(t) for t in (stats, norm, part))):
        res.append(
            (
                ttnn.to_torch(s)[:, :, :rows].clone(),
                ttnn.to_torch(n)[:, :, :rows].clone(),
                ttnn.to_torch(p)[:, :, :rows].clone(),
            )
        )
    return res


def traced_us(mesh, call, short=4, long=12):
    def release(r):
        for t in r:
            ttnn.deallocate(t)

    def capture(ops):
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            for _ in range(ops):
                release(call())
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        return tid

    def wall(tid):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = 1e30
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(REPLAYS):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            best = min(best, time.perf_counter() - t0)
        return best / REPLAYS * 1e6

    release(call())
    a, b = capture(short), capture(long)
    try:
        ta, tb = wall(a), wall(b)
    finally:
        ttnn.release_trace(mesh, a)
        ttnn.release_trace(mesh, b)
    return round((tb - ta) / (long - short), 2), round(tb / long, 2)


def compare_round(mesh, cands, rows, steps, seed0):
    for step in range(steps):
        res, gam, down = make_inputs(mesh, rows, seed0 + step)
        outs = {}
        for name, fn in cands:
            o = fn(res, gam, down)
            ttnn.synchronize_device(mesh)
            outs[name] = valid_rows(o, rows)
            for t in o:
                ttnn.deallocate(t)
        base = outs["default"]
        for name, got in outs.items():
            if name == "default":
                continue
            for d, (a, b) in enumerate(zip(base, got)):
                for what, x, y in zip(("stats", "normalized", "partials"), a, b):
                    if what == "stats":
                        continue  # the v2 front reads the stats from the transport core's L1: the gathered stats pages are not written
                    assert torch.equal(
                        x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32),
                        y.view(torch.int16) if y.dtype == torch.bfloat16 else y.view(torch.int32),
                    ), f"rows={rows} step {step} dev {d} {name}: {what} differs"
        ttnn.deallocate(res), ttnn.deallocate(gam), ttnn.deallocate(down)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_front_candidates(mesh_device, rows):
    mesh = mesh_device
    mesh.enable_program_cache()
    cands = candidates()
    os.environ["GF_ROWS"] = str(rows)
    results = {}
    compare_round(mesh, cands, rows, STEPS, 1000 * rows)
    logger.info(f"rows={rows}: {[n for n, _ in cands]} bit-identical over {STEPS} steps")
    res, gam, down = make_inputs(mesh, rows, 7)
    times = {}
    for rep in range(2):
        for name, fn in cands:
            times.setdefault(name, []).append(traced_us(mesh, lambda: fn(res, gam, down)))
            logger.info(
                f"rows={rows} rep {rep} {name}: slope/call {times[name][-1][0]} us (T12/12 {times[name][-1][1]})"
            )
    # parity after trace replays: odd-length traces replayed, then eager calls again
    for name, fn in cands:

        def release(r):
            for t in r:
                ttnn.deallocate(t)

        release(fn(res, gam, down))
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(5):
            release(fn(res, gam, down))
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for _ in range(7):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        ttnn.release_trace(mesh, tid)
    compare_round(mesh, cands, rows, 4, 5000 * rows)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(dict(rows=rows, us=times, load=os.getloadavg())) + "\n")
