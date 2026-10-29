# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of the block-input all-gather: stock ``ttnn.all_gather`` vs ``row_gather`` (the earlier packed program) vs
``row_gather2`` (the dense-unpack version), bit for bit and traced time, on the model's calls.

The model's call (gdn.py / qsa.py ``_all_gather_hidden``): the GR read's hidden shard ``[1, 1, rows, 320]`` bf16 TILE in
DRAM (``fp.allocate`` default) -> ``ttnn.all_gather(dim=3, cluster_axis=1, memory_config=<the in-projection's 8-core L1
width-sharded activation, shard 32 x 320>)`` (moe: the 5-core shard 32 x 512).  Compared on rows < ``rows`` of the gathered
``[1, 1, rows, 2560]`` tensor.  Time = slope of a 12-op and a 4-op trace.  Also times ``row_reduce`` with the default
fabric payload vs ROW_GATHER_PAYLOAD (``ROW_GATHER_PAYLOAD=8192``): run the file twice.
"""

import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_gather as rg
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_reduce as rr

TP = MESH_SHAPE[1]
SHARD = 320
WIDTH = SHARD * TP
TRACE_SHORT, TRACE_LONG, REPLAYS, REPEATS = 4, 12, 20, 3
RESULTS = os.environ.get("ROW_GATHER_RESULTS")
PAYLOAD = int(os.environ.get("ROW_GATHER_PAYLOAD", "0"))
DEVICE_PARAMS = dict(demo.DEVICE_PARAMS, trace_region_size=64 * 1024 * 1024)
if PAYLOAD:
    _router = ttnn.FabricRouterConfig()
    _router.max_packet_payload_size_bytes = PAYLOAD
    DEVICE_PARAMS["fabric_router_config"] = _router
CASES = [(1, 8), (2, 8), (1, 5), (2, 5)]  # (rows, consumer storage cores: 8 = gdn / qsa, 5 = moe)


def _v1(x, cfg):
    """The earlier barrier + unpack program (QWEN38_ROW_GATHER=v1)."""

    previous = os.environ.get(rg.ENV)
    os.environ[rg.ENV] = "v1"
    try:
        return rg.row_gather(x, cluster_axis=1, memory_config=cfg)
    finally:
        if previous is None:
            del os.environ[rg.ENV]
        else:
            os.environ[rg.ENV] = previous


CANDIDATES = [("row_gather", _v1)]
try:
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_gather2 as rg2

    CANDIDATES.append(("row_gather2", lambda x, cfg: rg2.row_gather(x, cluster_axis=1, memory_config=cfg)))
except ImportError:
    rg2 = None


def _act_config(cores):
    return ttnn.create_sharded_memory_config(
        (ttnn.TILE_SIZE, WIDTH // cores),
        ttnn.CoreGrid(x=cores, y=1),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _stock(x, cfg):
    return ttnn.all_gather(x, dim=3, cluster_axis=1, memory_config=cfg)


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

    ttnn.deallocate(call())
    short, long_ = capture(TRACE_SHORT), capture(TRACE_LONG)
    try:
        ts, tl = wall(short), wall(long_)
    finally:
        ttnn.release_trace(mesh, short)
        ttnn.release_trace(mesh, long_)
    return round((tl - ts) / (TRACE_LONG - TRACE_SHORT), 2), round(tl / TRACE_LONG, 2)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("rows,cores", CASES, ids=[f"r{r}_c{c}" for r, c in CASES])
def test_gather_candidates_are_bitwise_stock_and_timed(mesh_device, rows, cores):
    mesh = mesh_device
    mesh.enable_program_cache()
    g = torch.Generator().manual_seed(77 + rows)
    x = (
        torch.randn(TP, 1, rows, SHARD, generator=g)
        * torch.exp2(torch.randint(-8, 9, (TP, 1, rows, SHARD), generator=g).float())
    ).to(torch.bfloat16)
    src = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cfg = _act_config(cores)
    expect = torch.cat([x[d] for d in range(TP)], dim=-1)  # [1, rows, WIDTH] on every device

    def host(t):
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:, :, :rows, :]

    report = {"rows": rows, "consumer_cores": cores, "fabric_payload": int(ttnn.get_tt_fabric_max_payload_size_bytes())}
    stock = host(_stock(src, cfg))
    report["stock_vs_host_concat_mismatch"] = int((stock != expect.unsqueeze(0).expand(TP, -1, -1, -1)).sum())
    funcs = [("stock", _stock)] + CANDIDATES
    for name, fn in CANDIDATES:
        out = host(fn(src, cfg))
        report[f"{name}_vs_stock_mismatch"] = int((out.view(torch.int16) != stock.view(torch.int16)).sum())
        again = [host(fn(src, cfg)) for _ in range(6)]
        report[f"{name}_repeat_mismatch"] = sum(
            int((a.view(torch.int16) != stock.view(torch.int16)).sum()) for a in again
        )
    # trace replay with an odd number of gathers kept alive: the parity of the barrier-free landing buffer must survive replays
    for name, fn in CANDIDATES:
        fn(src, cfg)  # eager warm-up
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        outs = [fn(src, cfg) for _ in range(5)]
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        bad = 0
        for _ in range(7):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            bad += sum(int((host(o).view(torch.int16) != stock.view(torch.int16)).sum()) for o in outs)
        ttnn.release_trace(mesh, tid)
        report[f"{name}_trace_replay_mismatch"] = bad
        assert bad == 0, report
    report["us_slope_and_mean12"] = {name: _traced_us(mesh, lambda fn=fn: fn(src, cfg)) for name, fn in funcs}
    # row_reduce at the device's fabric payload (the model's reduce: bf16 partial, DRAM, rows <= 2)
    os.environ[rr.ENV] = "1"
    part = ttnn.from_torch(
        torch.randn(TP, 1, rows, WIDTH).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    report["row_reduce_us_slope_and_mean12"] = _traced_us(mesh, lambda: rr.row_reduce(part, cluster_axis=1))
    print("ROW_GATHER_RESULT " + json.dumps(report))
    if RESULTS:
        with open(RESULTS, "a") as f:
            f.write(json.dumps(report) + "\n")
    for name, _ in CANDIDATES:
        assert report[f"{name}_vs_stock_mismatch"] == 0 and report[f"{name}_repeat_mismatch"] == 0, report
