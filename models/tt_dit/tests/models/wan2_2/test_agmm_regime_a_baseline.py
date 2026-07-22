# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Task 1 baselines for the fused all-gather + regime_a_matmul plan (REGIME_A_AGMM_EXECUTION_PLAN.md):
# measure standalone all-gather (T_ag), unfused all_gather_async -> regime_a_matmul (T_unfused), and
# standalone regime_a on the full-K shape (T_mm), plus PCC, using ONLY existing ops. No new op yet.
#
# in0[M, K] is K-sharded across D devices on the cluster axis; in1[K, N] is the full regime_a weight,
# replicated. all_gather(dim=3) gathers K -> full in0 on every device, then regime_a_matmul(full_in0, in1).
#
# Timing: trace capture + host wall per traced execute (median of N timed iters, warmup dropped).
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
    """Compile once, capture a trace of build_fn(), execute iters+warmup times, return median host ms."""
    build_fn()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = build_fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    ts = []
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e3)
    ttnn.release_trace(device, tid)
    timed = sorted(ts[warmup:])
    return timed[len(timed) // 2], out


def _run(mesh, M, K, N, D, cluster_axis, num_links, num_workers_per_link, cfg):
    submesh_shape = [1, 1]
    submesh_shape[cluster_axis] = D
    sub = mesh.create_submesh(ttnn.MeshShape(tuple(submesh_shape)))
    assert sub.get_num_devices() == D, (sub.get_num_devices(), D)

    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    b = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    golden = (a.float() @ b.float())[0, 0]

    shard_dims = [None, None]
    shard_dims[cluster_axis] = 3  # shard K (tensor dim 3) across the cluster axis
    a_sharded = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=sub,
        mesh_mapper=ttnn.ShardTensor2dMesh(sub, mesh_shape=tuple(submesh_shape), dims=shard_dims),
    )
    wcfg = ttnn.create_regime_a_weight_memory_config(list(b.shape), ttnn.bfloat16, sub)
    b_full = ttnn.from_torch(
        b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=sub, memory_config=wcfg,
        mesh_mapper=ttnn.ShardTensor2dMesh(sub, mesh_shape=tuple(submesh_shape), dims=[None, None]),
    )
    # also a full (already-gathered) in0 for the standalone T_mm measurement
    a_full = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=sub,
        mesh_mapper=ttnn.ShardTensor2dMesh(sub, mesh_shape=tuple(submesh_shape), dims=[None, None]),
    )
    Ns, Pk, Sm, kb, nsb = cfg
    racfg = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)

    per_dev_M = M  # M replicated (not sharded)
    ag_sems = _sems(sub, 3)
    ag_buf = [
        ttnn.from_torch(
            torch.zeros(1, 1, per_dev_M, K, dtype=torch.float32), device=sub, layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            mesh_mapper=ttnn.ShardTensor2dMesh(sub, mesh_shape=tuple(submesh_shape), dims=[None, None]),
        )
        for _ in range(3)
    ]

    def ag(i):
        return ttnn.experimental.all_gather_async(
            a_sharded, persistent_output_buffer=ag_buf[i], dim=3,
            multi_device_global_semaphore=ag_sems[i], num_links=num_links, topology=ttnn.Topology.Ring,
            cluster_axis=cluster_axis, chunks_per_sync=16, num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=2,
        )

    # NOTE: regime_a_matmul cannot build on a D>1 mesh (its PARETO ring optimizer calls
    # get_worker_noc_hop_distance(), which is unit-mesh-only), and the bank-order diagnostic that
    # skips that query is not python-exposed. So T_mm and the literal unfused sequence are NOT
    # measurable multi-device here. We measure standalone all-gather (T_ag) + gather correctness.
    gathered = ag(0)
    got = ttnn.to_torch(gathered, mesh_composer=ttnn.ConcatMesh2dToTensor(sub, mesh_shape=tuple(submesh_shape), dims=[0, 1]))
    # gathered should equal the full in0 [M,K] on every device; compare device 0's copy.
    g = got.float()
    g0 = g.reshape(-1, M, K)[0] if g.numel() >= M * K else g
    ok, pcc = comp_pcc(a[0, 0].float(), g0[:M, :K], 0.999)
    logger.info(f"AG D={D} {M}x{K}x{N} gather-PCC={pcc}")

    t_ag, _ = _time_trace(sub, lambda: ag(1))
    rec = {
        "shape": f"{M}x{K}x{N}", "M": M, "K": K, "N": N, "D": D, "cluster_axis": cluster_axis,
        "num_links": num_links, "workers": num_workers_per_link, "cfg": list(cfg),
        "ag_pcc": float(pcc), "t_ag_ms": t_ag,
    }
    logger.info("AGMM_BASELINE " + json.dumps(rec))
    _append(rec)
    assert ok, f"gather PCC {pcc} < 0.999"


def _append(rec):
    data = []
    if os.path.exists(RESULTS):
        try:
            data = json.load(open(RESULTS))
        except Exception:
            data = []
    data.append(rec)
    json.dump(data, open(RESULTS, "w"), indent=2)


def _dp(fabric):
    return {"fabric_config": fabric, "fabric_router_config": _router(4096), "trace_region_size": 200000}


def _router(payload):
    c = ttnn._ttnn.fabric.FabricRouterConfig()
    c.max_packet_payload_size_bytes = payload
    return c


# (label, M, K, N, cfg(Ns,Pk,Sm,kb,nsb) from the GLX corpus report)
SHAPES = [
    ("narrowN_32x6144x3072", 32, 6144, 3072, (1, 3, 1, 4, 6)),
]


@pytest.mark.parametrize(
    "mesh_device, device_params, cluster_axis, num_links, num_workers_per_link, D",
    [
        [(4, 8), {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "fabric_router_config": _router(4096), "trace_region_size": 200000}, 0, 2, 6, 4],
        [(4, 8), {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "fabric_router_config": _router(4096), "trace_region_size": 200000}, 1, 2, 6, 8],
    ],
    ids=["D4_bh4x8", "D8_bh4x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("label,M,K,N,cfg", SHAPES, ids=[s[0] for s in SHAPES])
def test_agmm_baseline(mesh_device, device_params, cluster_axis, num_links, num_workers_per_link, D, label, M, K, N, cfg):
    _run(mesh_device, M, K, N, D, cluster_axis, num_links, num_workers_per_link, cfg)
