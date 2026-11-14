# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""all_reduce_async one_shot (reduce-scatter + all-gather in one op) on the Qwen3.8 decode all-reduce shape.

The decode / spec-verify all-reduce is [1,1,32,5120] bf16, width-sharded over the 40-core RES40 grid (4 tiles per
core), with a persistent [32, 5120*tp] buffer on the same grid. The default all_reduce_async line-multicasts every
device's whole partial to every peer (8 KB packets, up to 80 packets on the busiest directed link of an 8-ring).
one_shot unicasts each peer's 1/tp slice straight into its owner's buffer, the owner reduces it with the same kernel and
slot order, and the owner multicasts the reduced slice back (at most 32 packets on the busiest link).

test_all_reduce_async_one_shot_correctness: one_shot output is byte-identical to the default output on every device
(and close to the torch sum) for 32, 16 and 8 logical rows, plain and with a fused residual; a second call hits the
program cache and stays identical.

test_all_reduce_async_one_shot_bench: traced latency (ITERS back-to-back all-reduces per trace, REPLAYS replays, two
alternating semaphores as the model cycles them) of the default and one_shot modes, interleaved twice. Logs one
"ONESHOT_BENCH" line per arm and one "ONESHOT_VERDICT" line: GO when the default is >= ONESHOT_GATE_US (8) us slower
per call.

Run (8-chip ring, as served):
  MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=0,1,2,3,27,26,25,24 \\
  pytest tests/ttnn/unit_tests/operations/ccl/test_all_reduce_async_one_shot.py -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn

GRID = (10, 4)  # RES40
DIM = 5120
ITERS = int(os.environ.get("ONESHOT_BENCH_ITERS", "64"))
REPLAYS = int(os.environ.get("ONESHOT_BENCH_REPLAYS", "20"))
GATE_US = float(os.environ.get("ONESHOT_GATE_US", "8"))


def _device_params(trace_bytes):
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 8192  # the model's QWEN36_FABRIC_PAYLOAD_BYTES default
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        "fabric_router_config": router,
        "trace_region_size": trace_bytes,
    }


def _width_mem(shard_w):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID[0] - 1, GRID[1] - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [ttnn.TILE_SIZE, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _setup(mesh_device):
    tp = list(mesh_device.shape)[1]
    ncores = GRID[0] * GRID[1]
    mem = _width_mem(DIM // ncores)
    buf_mem = _width_mem(DIM * tp // ncores)
    sg = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(sg.x - 1, sg.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(2)]
    buf = ttnn.from_torch(
        torch.zeros([1, 1, ttnn.TILE_SIZE, DIM * tp]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=buf_mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return tp, mem, sems, buf


def _partials(mesh_device, tp, rows, mem, seed):
    g = torch.Generator().manual_seed(seed)
    full = torch.randn(1, tp, rows, DIM, generator=g).to(torch.bfloat16)
    t = ttnn.from_torch(
        full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=mem,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )
    return full, t


def _ar(mesh_device, inp, buf, sem, mem, one_shot, residual=None):
    extra = {}
    if one_shot:
        extra["one_shot"] = True
    if residual is not None:
        extra["residual_tensor"] = residual
    return ttnn.experimental.all_reduce_async(
        inp,
        buf,
        cluster_axis=1,
        mesh_device=mesh_device,
        multi_device_global_semaphore=sem,
        topology=ttnn.Topology.Ring,
        num_links=2,
        memory_config=mem,
        **extra,
    )


def _per_device(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


@pytest.mark.parametrize("device_params", [_device_params(8 << 20)], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 8), id="1x8")], indirect=True)
def test_all_reduce_async_one_shot_correctness(mesh_device):
    tp, mem, sems, buf = _setup(mesh_device)
    n_checked = 0
    for rows in (32, 16, 8):
        for fused in (False, True):
            full, inp = _partials(mesh_device, tp, rows, mem, seed=rows + 100 * fused)
            residual = None
            if fused:
                res_t = torch.randn(1, 1, rows, DIM).to(torch.bfloat16)
                residual = ttnn.from_torch(
                    res_t,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=mem,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
            outs = {}
            for i, (name, one_shot) in enumerate((("default", False), ("one_shot", True), ("one_shot#2", True))):
                out = _ar(mesh_device, inp, buf, sems[i % 2], mem, one_shot, residual)
                ttnn.synchronize_device(mesh_device)
                outs[name] = _per_device(out)
                ttnn.deallocate(out)
            ref = full.float().sum(dim=1, keepdim=True)
            if fused:
                ref = ref + res_t.float()
            for dev in range(tp):
                d, o, o2 = outs["default"][dev], outs["one_shot"][dev], outs["one_shot#2"][dev]
                assert o.shape == d.shape, f"rows={rows} fused={fused} dev {dev}: shape {o.shape} vs {d.shape}"
                assert torch.equal(o, d), (
                    f"rows={rows} fused={fused} dev {dev}: one_shot != default "
                    f"(max |diff| {(o.float() - d.float()).abs().max().item():.3e})"
                )
                assert torch.equal(o2, o), f"rows={rows} fused={fused} dev {dev}: cached one_shot call differs"
                assert torch.equal(o, outs["one_shot"][0]), f"rows={rows} fused={fused}: dev {dev} != dev 0"
                err = (o.float() - ref).abs().max().item()
                assert err <= 0.05 * ref.abs().max().item(), f"rows={rows} dev {dev}: |out - torch| = {err:.3e}"
                n_checked += 1
            logger.info(f"ONESHOT_CHECK rows={rows} fused_residual={fused}: byte-identical to default on {tp} devices")
            ttnn.deallocate(inp)
            if residual is not None:
                ttnn.deallocate(residual)
    assert n_checked == 6 * tp


@pytest.mark.parametrize("device_params", [_device_params(16 << 20)], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 8), id="1x8")], indirect=True)
def test_all_reduce_async_one_shot_bench(mesh_device):
    tp, mem, sems, buf = _setup(mesh_device)
    _, inp = _partials(mesh_device, tp, 32, mem, seed=7)

    def trace(one_shot):
        # Compile eagerly, then capture ITERS back-to-back all-reduces (two live outputs at a time).
        out = _ar(mesh_device, inp, buf, sems[0], mem, one_shot)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(out)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        prev = None
        for i in range(ITERS):
            out = _ar(mesh_device, inp, buf, sems[i % 2], mem, one_shot)
            if prev is not None:
                ttnn.deallocate(prev)
            prev = out
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        return tid, prev

    def time_trace(tid):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)  # warm
        t0 = time.perf_counter()
        for _ in range(REPLAYS):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) * 1e6 / (ITERS * REPLAYS)

    traces = {name: trace(one_shot) for name, one_shot in (("default", False), ("one_shot", True))}
    results = {"default": [], "one_shot": []}
    for _ in range(2):
        for name in ("default", "one_shot"):
            us = time_trace(traces[name][0])
            results[name].append(us)
            logger.info(f"ONESHOT_BENCH {name}: {us:.2f} us/all-reduce ([1,1,32,{DIM}] bf16, RES40, 2 links, ring)")
    for tid, last in traces.values():
        ttnn.release_trace(mesh_device, tid)
        ttnn.deallocate(last)
    d = min(results["default"])
    o = min(results["one_shot"])
    verdict = "GO" if d - o >= GATE_US else "STOP"
    logger.info(
        f"ONESHOT_VERDICT default={d:.2f} us one_shot={o:.2f} us saving={d - o:.2f} us/call "
        f"(gate >= {GATE_US:.1f} us): {verdict}"
    )
