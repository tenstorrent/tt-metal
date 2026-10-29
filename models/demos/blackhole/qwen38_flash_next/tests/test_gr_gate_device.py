# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of the GR gate program (gr_read.low_rank_gate) on the TP=8 mesh: today's program vs the gate v3
(QWEN38_GR_GATE_V3=1), bit for bit on the valid rows of the block output and the injection, on random gathered partials,
rows 1/2/4 (+ fallbacks), plus the traced time per call.  The v3 gate may read the partials from the front v2's landing slots,
so the test also runs the pipeline front v2 -> gate on random residuals (``test_pipeline``).
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
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_read

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the TP=8 mesh")
TP = MESH_SHAPE[1]
FLAT, PARTIAL = 1280, 384
STEPS = int(os.environ.get("GG_STEPS", "10"))
REPLAYS = int(os.environ.get("GG_REPLAYS", "30"))
RESULTS = os.environ.get("GG_RESULTS", "/tmp/gg_results.jsonl")
DEVICE_PARAMS = dict(demo.DEVICE_PARAMS, trace_region_size=64 * 1024 * 1024)
DRAM = ttnn.DRAM_MEMORY_CONFIG
V3 = "QWEN38_GR_GATE_V3"


def make_inputs(mesh, rows, seed):
    g = torch.Generator().manual_seed(seed)
    parts = torch.randn(TP, 1, rows, PARTIAL, generator=g) * torch.exp2(
        torch.randint(-3, 4, (TP, 1, rows, 1), generator=g).float()
    )
    norm = (torch.randn(1, 1, rows, TP * FLAT, generator=g)).to(torch.bfloat16)
    up = (torch.randn(1, 1, PARTIAL, TP * FLAT, generator=g) * 0.05).to(torch.bfloat16)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    shard3 = ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 3))
    p = ttnn.from_torch(
        parts, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=rep
    )
    n = ttnn.from_torch(
        norm, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=shard3
    )
    u = ttnn.from_torch(
        up,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=dm.dram_sharded_weight_memory_config(mesh, PARTIAL, FLAT),
        mesh_mapper=shard3,
    )
    return p, n, u


def set_flag(on):
    if on:
        os.environ[V3] = "1"
    else:
        os.environ.pop(V3, None)


def gate(variant, p, n, u):
    if variant == "v3":
        from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_read import gate3

        if gate3.MAX_ROWS >= p.shape[2]:
            return gate3.low_rank_gate3(p, n, u, matmul="chain")
    return gr_read.low_rank_gate(p, n, u, matmul="chain")


def host_valid(outs, rows):
    res = []
    for b, i in zip(*(ttnn.get_device_tensors(t) for t in outs)):
        res.append((ttnn.to_torch(b)[:, :, :rows].clone(), ttnn.to_torch(i)[:, :, :rows].clone()))
    return res


def traced_us(mesh, call, short=4, long=12):
    def release(r):
        for t in r:
            ttnn.deallocate(t)

    def capture(ops):
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(ops):
            release(call())
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


def bits(x):
    return x.view(torch.int16)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_gate_candidates(mesh_device, rows):
    mesh = mesh_device
    mesh.enable_program_cache()
    variants = ["default"] + (["v3"] if os.environ.get("GG_V3", "1") == "1" else [])
    for step in range(STEPS):
        p, n, u = make_inputs(mesh, rows, 100 * rows + step)
        got = {}
        for v in variants:
            o = gate(v, p, n, u)
            ttnn.synchronize_device(mesh)
            got[v] = host_valid(o, rows)
            for t in o:
                ttnn.deallocate(t)
        for v in variants[1:]:
            for d, (a, b) in enumerate(zip(got["default"], got[v])):
                for what, x, y in zip(("block", "injection"), a, b):
                    assert torch.equal(bits(x), bits(y)), f"rows={rows} step {step} dev {d} {v}: {what} differs"
        for t in (p, n, u):
            ttnn.deallocate(t)
    logger.info(f"rows={rows}: {variants} bit-identical over {STEPS} steps")
    p, n, u = make_inputs(mesh, rows, 7)
    times = {}
    for rep in range(2):
        for v in variants:
            times.setdefault(v, []).append(traced_us(mesh, lambda: gate(v, p, n, u)))
            logger.info(f"rows={rows} rep {rep} {v}: slope/call {times[v][-1][0]} us (T12/12 {times[v][-1][1]})")
    with open(RESULTS, "a") as f:
        f.write(json.dumps(dict(rows=rows, us=times, load=os.getloadavg())) + "\n")


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("rows", [1, 2, 16, 32])
def test_pipeline_front_to_gate(mesh_device, rows):
    """Random residuals through the front (default and v2) into the gate (default and v3): the block output and the injection
    of every front x gate combination are bit-identical on the valid rows (the front v2 leaves the pad rows of the gathered pages
    unwritten; the gate must not care)."""

    from models.demos.blackhole.qwen38_flash_next.tests import test_gr_front_device as F
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_recip_last as grl
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_recip_last import front2

    mesh = mesh_device
    mesh.enable_program_cache()
    g = torch.Generator().manual_seed(11)
    up_host = (torch.randn(1, 1, PARTIAL, TP * FLAT, generator=g) * 0.05).to(torch.bfloat16)
    up = ttnn.from_torch(
        up_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=dm.dram_sharded_weight_memory_config(mesh, PARTIAL, FLAT),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 3)),
    )
    for step in range(4):
        res, gam, down = F.make_inputs(mesh, rows, 300 + step)
        got = {}
        for fname, front in (("front", grl.stats_recip_down_gather), ("front2", front2.stats_recip_down_gather)):
            stats, norm, parts = front(res, gam, down)
            ttnn.deallocate(stats)
            for gname in ("default", "v3"):
                o = gate(gname, parts, norm, up)
                ttnn.synchronize_device(mesh)
                got[(fname, gname)] = host_valid(o, rows)
                for t in o:
                    ttnn.deallocate(t)
            ttnn.deallocate(norm), ttnn.deallocate(parts)
        base = got[("front", "default")]
        for key, val in got.items():
            for d, (a, b) in enumerate(zip(base, val)):
                for what, x, y in zip(("block", "injection"), a, b):
                    assert torch.equal(bits(x), bits(y)), f"rows={rows} step {step} dev {d} {key}: {what} differs"
        for t in (res, gam, down):
            ttnn.deallocate(t)
    logger.info(f"rows={rows}: front x gate combinations bit-identical")
