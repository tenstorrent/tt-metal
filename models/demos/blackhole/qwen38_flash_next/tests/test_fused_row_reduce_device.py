# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of ``row_reduce`` (ttnn/fused/row_reduce): the row-packed output reduce-scatter against the stock line
reduce-scatters, bit for bit, and its traced time.

The 1x8 TP=8 mesh with the model's fabric config.  Per case (bf16 rows 1 and 2, fp32 row 1) every device holds a random
partial ``[1, 1, rows, 2560]`` with mixed magnitudes (so the rounding of every add matters).  Compared on the rows <
``rows`` of the ``[1, 1, rows, 320]`` outputs:
  * ``ttnn.reduce_scatter(dim=3, cluster_axis=1, topology=Linear)`` as gdn.py / qsa.py call it,
  * ``ttnn.experimental.reduce_scatter_minimal_async`` as ``tt_all_reduce`` (moe.py) calls it (bf16),
  * a host emulation of the summation tree in both line directions (which neighbour is "forward" is established here).
``row_reduce`` must equal the stock reduce-scatter.  The time of each is the slope of a 12-op and a 4-op trace.

Run:  flock -w 10800 /home/ttuser/atupe/device.lock timeout 1800 python -m pytest \
          models/demos/blackhole/qwen38_flash_next/tests/test_fused_row_reduce_device.py -s -p no:cacheprovider -o addopts=
Env:  MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 (as demo/text_demo.py).
"""

import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401 (the demo's fixture)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_reduce as rr

TP = MESH_SHAPE[1]
WIDTH = 2560
SHARD = WIDTH // TP
TRACE_SHORT, TRACE_LONG, REPLAYS, REPEATS = 4, 12, 20, 3
RESULTS = os.environ.get("ROW_REDUCE_RESULTS")
CASES = [
    ("bf16", 1, False),
    ("bf16", 2, False),
    ("fp32", 1, False),
    ("bf16", 1, True),
    ("bf16", 2, True),
]  # (dtype, rows, L1 width sharded)
DEVICE_PARAMS = dict(demo.DEVICE_PARAMS, trace_region_size=64 * 1024 * 1024)
PAYLOAD = int(os.environ.get("ROW_REDUCE_PAYLOAD", "0"))  # fabric max packet payload bytes (0: the fabric default 4352)
if PAYLOAD:
    _router = ttnn.FabricRouterConfig()
    _router.max_packet_payload_size_bytes = PAYLOAD
    DEVICE_PARAMS["fabric_router_config"] = _router


def _partials(dtype, rows, seed):
    """``[TP, 1, rows, WIDTH]`` float32 values that are exact in the input dtype, magnitudes spread over 2^-12 .. 2^12."""

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(TP, 1, rows, WIDTH, generator=g)
    x = x * torch.exp2(torch.randint(-12, 13, (TP, 1, rows, WIDTH), generator=g).float())
    return x.to(torch.bfloat16).float() if dtype == "bf16" else x


def _to_mesh(mesh, x, dtype, sharded=False):
    """The partial on the mesh: DRAM interleaved, or (``sharded``) L1 width sharded over 16 cores as the GDN out-projection
    leaves it."""

    t = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16 if dtype == "bf16" else ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if sharded:
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))])
        spec = ttnn.ShardSpec(grid, [ttnn.TILE_SIZE, WIDTH // 16], ttnn.ShardOrientation.ROW_MAJOR)
        t = ttnn.to_memory_config(t, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec))
    return t


def _from_mesh(mesh, t, rows):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:, :, :rows, :]


def _bits(x, dtype):
    return x.to(torch.bfloat16).view(torch.int16) if dtype == "bf16" else x.float().view(torch.int32)


def _mismatches(a, b, dtype):
    return int((_bits(a, dtype) != _bits(b, dtype)).sum())


def _trunc(v):
    """bf16 truncation (round toward zero) of float32 values."""

    return (v.contiguous().view(torch.int32) & -65536).view(torch.float32)


def _tree(p, d, dtype, mirror, round_mode="rne"):
    """Host emulation of the line reduce-scatter's tree at device ``d`` for ``p`` = [TP, ..., cols] (the slice's
    partials): out = (p_d + F) + B, F = p_{d-1} + (... + (p_1 + p_0)), B = p_{d+1} + (... + (p_6 + p_7)); mirror flips the line.
    """

    rnd = (
        (lambda v: v)
        if dtype != "bf16"
        else (_trunc if round_mode == "trunc" else (lambda v: v.to(torch.bfloat16).float()))
    )
    if mirror:
        p, d = p.flip(0), TP - 1 - d
    f = None if d == 0 else p[0]
    for k in range(1, d):
        f = rnd(p[k] + f)
    g = p[d] if f is None else rnd(p[d] + f)
    if d == TP - 1:
        return g
    b = p[TP - 1]
    for k in range(TP - 2, d, -1):
        b = rnd(p[k] + b)
    return rnd(g + b)


def _host(x, dtype, mirror, round_mode="rne"):
    """The emulated output [TP, 1, rows, SHARD]."""

    return (
        torch.stack([_tree(x[:, :, :, d * SHARD : (d + 1) * SHARD], d, dtype, mirror, round_mode) for d in range(TP)])
        .squeeze(1)
        .unsqueeze(1)
    )


def _traced_us(mesh, call):
    def capture(ops):
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(ops):
            ttnn.deallocate(call())
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        return tid

    def wall(tid):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = 1e30
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            for _ in range(REPLAYS):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            best = min(best, time.perf_counter() - t0)
        return best / REPLAYS * 1e6

    ttnn.deallocate(call())  # eager warm-up: the program cache, the fused op's semaphores
    short, long_ = capture(TRACE_SHORT), capture(TRACE_LONG)
    try:
        ts, tl = wall(short), wall(long_)
    finally:
        ttnn.release_trace(mesh, short)
        ttnn.release_trace(mesh, long_)
    return round((tl - ts) / (TRACE_LONG - TRACE_SHORT), 2), round(tl / TRACE_LONG, 2)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("dtype,rows,sharded", CASES, ids=[f"{d}_r{r}{'_l1' if l1 else ''}" for d, r, l1 in CASES])
def test_row_reduce_is_bitwise_the_stock_reduce_scatter(mesh_device, dtype, rows, sharded, monkeypatch):
    from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce

    mesh = mesh_device
    mesh.enable_program_cache()
    monkeypatch.setenv(rr.ENV, "1")
    x = _partials(dtype, rows, seed=1234 + rows)
    exact = x.sum(0)  # float32 reference only to tell the rounding apart in the log
    tt_ccl = TT_CCL(mesh)
    report = {
        "dtype": dtype,
        "rows": rows,
        "l1_sharded": sharded,
        "fabric_payload": int(ttnn.get_tt_fabric_max_payload_size_bytes()),
        "l1_direct": rr.L1_DIRECT,
    }

    partial = _to_mesh(mesh, x, dtype, sharded)
    assert rr.admits(partial), "the contract must admit this partial"

    stock = _from_mesh(mesh, rr.composed(partial), rows)  # [TP, 1, rows, SHARD]
    report["stock_vs_host_forward"] = _mismatches(stock, _host(x, dtype, False), dtype)
    report["stock_vs_host_mirror"] = _mismatches(stock, _host(x, dtype, True), dtype)
    report["stock_vs_host_forward_trunc"] = _mismatches(stock, _host(x, dtype, False, "trunc"), dtype)
    report["stock_vs_host_mirror_trunc"] = _mismatches(stock, _host(x, dtype, True, "trunc"), dtype)
    report["stock_vs_exact_elements_differing"] = _mismatches(
        stock, torch.stack(exact.split(SHARD, dim=-1)).to(stock.dtype), dtype
    )
    if dtype == "bf16":

        def minimal(src):
            return ttnn.experimental.reduce_scatter_minimal_async(
                src,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
                barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        via_helper = _from_mesh(
            mesh,
            tt_all_reduce(
                _to_mesh(mesh, x, dtype),
                mesh,
                tt_ccl,
                cluster_axis=0,
                dim=3,
                num_reduce_scatter_links=1,
                num_all_gather_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            ),
            rows,
        )
        report["moe_helper_vs_stock_scatter"] = _mismatches(via_helper, stock, dtype)
        report["moe_helper_vs_host_forward"] = _mismatches(via_helper, _host(x, dtype, False), dtype)

    for mirror in (0, 1):
        rr.MIRROR = mirror
        fused = _from_mesh(mesh, rr.row_reduce(partial), rows)
        report[f"row_reduce_mirror{mirror}_vs_stock"] = _mismatches(fused, stock, dtype)
        report[f"row_reduce_mirror{mirror}_vs_host_forward"] = _mismatches(fused, _host(x, dtype, False), dtype)
        if dtype == "bf16":
            report[f"row_reduce_mirror{mirror}_vs_moe_helper"] = _mismatches(fused, via_helper, dtype)
    best = min((0, 1), key=lambda m: report[f"row_reduce_mirror{m}_vs_stock"])
    rr.MIRROR = int(os.environ.get("QWEN38_ROW_REDUCE_MIRROR", "0"))
    report["best_mirror"] = best

    # repeated calls: the barrier / semaphore protocol over many back-to-back launches with the default direction
    again = [_from_mesh(mesh, rr.row_reduce(partial), rows) for _ in range(8)]
    report["repeat_mismatches_vs_stock"] = sum(_mismatches(a, stock, dtype) for a in again)

    stock_us = _traced_us(mesh, lambda: rr.composed(partial))
    fused_us = _traced_us(mesh, lambda: rr.row_reduce(partial))
    report["us_per_op_slope_and_mean12"] = {"stock_reduce_scatter": stock_us, "row_reduce": fused_us}
    if dtype == "bf16":
        report["us_per_op_slope_and_mean12"]["moe_reduce_scatter_minimal_async"] = _traced_us(
            mesh, lambda: minimal(partial)
        )
    print("ROW_REDUCE_RESULT " + json.dumps(report))
    if RESULTS:
        with open(RESULTS, "a") as f:
            f.write(json.dumps(report) + "\n")
    assert report[f"row_reduce_mirror{rr.MIRROR}_vs_stock"] == 0, report
    assert report["repeat_mismatches_vs_stock"] == 0, report
