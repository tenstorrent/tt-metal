# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Task 1 baselines for REGIME_A_AGMM_EXECUTION_PLAN.md (existing ops only; no new op).
#   D=1  : T_mm  = single-chip regime_a_matmul(full in0 @ in1), config=None (Picker v3), unit submesh.
#   D=4/8: T_ag  = standalone all_gather_async of the K-sharded in0 (dim=3) on the cluster axis.
# T_unfused = T_mm + T_ag (sequential); fused ideal = max(T_mm, T_ag). All timings are trace host-wall
# (median of 10 timed iters, 2 warmup dropped) => same measurement basis across T_mm and T_ag.
#
# NOTE: regime_a_matmul's default PARETO ring optimizer uses the unit-mesh-only overload of
# get_worker_noc_hop_distance, so it cannot build on a D>1 mesh. The multi-device-safe overload
# get_worker_noc_hop_distance(MeshDevice*, MeshCoordinate, ...) exists and unblocks the NEW fused op
# (Task 2); we do NOT modify regime_a here. T_mm is therefore measured on a unit mesh.
import os
import time
import json

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

RESULTS = os.environ.get("AGMM_BASELINE_JSON", "/data/cglagovich/agmm_baseline.json")


def _sems(mesh, n):
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    return [[ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)] for _ in range(n)]


def _time_trace(device, build_fn, iters=10, warmup=2):
    build_fn()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    build_fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(warmup + iters):
        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e3)
    ttnn.release_trace(device, tid)
    timed = sorted(ts[warmup:])
    return timed[len(timed) // 2], (max(timed) - min(timed)) / min(timed) * 100.0


def _append(rec):
    data = []
    if os.path.exists(RESULTS):
        try:
            data = json.load(open(RESULTS))
        except Exception:
            data = []
    data.append(rec)
    json.dump(data, open(RESULTS, "w"), indent=2)


def _run_tmm(mesh, M, K, N):
    sub = mesh.create_submesh(ttnn.MeshShape((1, 1)))
    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    b = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    golden = (a.float() @ b.float())[0, 0]
    a_t = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=sub)
    wcfg = ttnn.create_regime_a_weight_memory_config(list(b.shape), ttnn.bfloat16, sub)
    b_t = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=sub, memory_config=wcfg)
    out = ttnn.experimental.regime_a_matmul(a_t, b_t)  # config=None -> Picker v3
    got = ttnn.to_torch(ttnn.from_device(out))[0, 0].float()
    ok, pcc = comp_pcc(golden, got, 0.999)
    t_mm, spread = _time_trace(sub, lambda: ttnn.experimental.regime_a_matmul(a_t, b_t))
    rec = {"shape": f"{M}x{K}x{N}", "M": M, "K": K, "N": N, "D": 1, "metric": "T_mm",
           "pcc": float(pcc), "t_ms": t_mm, "spread_pct": spread}
    logger.info("AGMM_BASELINE " + json.dumps(rec))
    _append(rec)
    assert ok, f"T_mm PCC {pcc} < 0.999"


def _run_tag(mesh, M, K, N, D, cluster_axis, num_links, workers):
    submesh_shape = [1, 1]
    submesh_shape[cluster_axis] = D
    sub = mesh.create_submesh(ttnn.MeshShape(tuple(submesh_shape)))
    assert sub.get_num_devices() == D
    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    shard_dims = [None, None]
    shard_dims[cluster_axis] = 3
    a_sh = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=sub,
                           mesh_mapper=ttnn.ShardTensor2dMesh(sub, mesh_shape=tuple(submesh_shape), dims=shard_dims))
    sems = _sems(sub, 2)
    bufs = [ttnn.from_torch(torch.zeros(1, 1, M, K, dtype=torch.float32), device=sub, layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16,
                            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
                            mesh_mapper=ttnn.ShardTensor2dMesh(sub, mesh_shape=tuple(submesh_shape), dims=[None, None]))
            for _ in range(2)]

    def ag(i):
        return ttnn.experimental.all_gather_async(
            a_sh, persistent_output_buffer=bufs[i], dim=3, multi_device_global_semaphore=sems[i],
            num_links=num_links, topology=ttnn.Topology.Ring, cluster_axis=cluster_axis,
            chunks_per_sync=16, num_workers_per_link=workers, num_buffers_per_channel=2)

    gathered = ag(0)
    got = ttnn.to_torch(gathered, mesh_composer=ttnn.ConcatMesh2dToTensor(sub, mesh_shape=tuple(submesh_shape), dims=[0, 1]))
    g0 = got.float().reshape(-1, M, K)[0]
    ok, pcc = comp_pcc(a[0, 0].float(), g0[:M, :K], 0.999)
    t_ag, spread = _time_trace(sub, lambda: ag(1))
    rec = {"shape": f"{M}x{K}x{N}", "M": M, "K": K, "N": N, "D": D, "metric": "T_ag",
           "cluster_axis": cluster_axis, "num_links": num_links, "workers": workers,
           "pcc": float(pcc), "t_ms": t_ag, "spread_pct": spread}
    logger.info("AGMM_BASELINE " + json.dumps(rec))
    _append(rec)
    assert ok, f"gather PCC {pcc} < 0.999"


def _router(payload):
    c = ttnn._ttnn.fabric.FabricRouterConfig()
    c.max_packet_payload_size_bytes = payload
    return c


_DP = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "fabric_router_config": _router(4096), "trace_region_size": 200000}
_DP_NOFAB = {"trace_region_size": 200000}  # D=1 T_mm: regime_a uses no fabric

# categories: flagship, wide-N (in1-bound), narrow-N (fabric-sensitive), shallow-K (early stall), deep-K
SHAPES = [
    ("flagship_32x6144x3072", 32, 6144, 3072),
    ("wideN_32x6144x6144", 32, 6144, 6144),
    ("narrowN_256x6144x768", 256, 6144, 768),
    ("shallowK_128x2048x512", 128, 2048, 512),
    ("deepK_32x15360x768", 32, 15360, 768),
]


@pytest.mark.parametrize(
    "mesh_device, device_params, D, cluster_axis, num_links, workers",
    [
        [(1, 1), _DP_NOFAB, 1, 0, 1, 1],
        [(4, 8), _DP, 4, 0, 2, 6],
        [(4, 8), _DP, 8, 1, 2, 6],
    ],
    ids=["D1_tmm", "D4_tag", "D8_tag"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("label,M,K,N", SHAPES, ids=[s[0] for s in SHAPES])
def test_agmm_baseline(mesh_device, device_params, D, cluster_axis, num_links, workers, label, M, K, N):
    if D == 1:
        _run_tmm(mesh_device, M, K, N)
    else:
        _run_tag(mesh_device, M, K, N, D, cluster_axis, num_links, workers)
