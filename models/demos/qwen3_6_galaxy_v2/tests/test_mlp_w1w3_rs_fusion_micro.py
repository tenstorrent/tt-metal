# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MLP w1/w3 reduce-scatter fusion: 2x separate RS vs concat -> 1 RS -> split.

The decode MLP does TWO separate line_reduce_scatter (w1, w3) on cluster_axis=1. This tests whether
concatenating w1_out|w3_out along dim=3 and doing ONE reduce_scatter (then splitting) is faster
(fewer CCL launches, same data) and numerically identical.

Run:
    python -m pytest --noconftest models/demos/qwen3_6_galaxy_v2/tests/test_mlp_w1w3_rs_fusion_micro.py -v -s
"""
from __future__ import annotations

import statistics
import time

import pytest
import torch

import ttnn

_MESH_SHAPE = (8, 4)
_M = 32
_N = 2176  # intermediate_per_tp (w1/w3 matmul output width)
_DIM = 3
_CA = 1  # cols, ring-4
_RING = _MESH_SHAPE[_CA]
_NW, _NT, _NI = 2, 15, 8


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*_MESH_SHAPE), trace_region_size=90_000_000)
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _time_traced(run_one, mesh):
    run_one()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    run_one()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(_NW):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(_NT):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        ts.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(mesh, tid)
    return statistics.mean(ts) / _NI


def _rs(x, mesh):
    return ttnn.reduce_scatter(
        x, _DIM, cluster_axis=_CA, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=ttnn.Topology.Linear, num_links=1
    )


@pytest.mark.hardware
def test_w1w3_rs_fusion(bh_glx_mesh):
    from models.common.utility_functions import comp_pcc

    mesh = bh_glx_mesh
    cs = _MESH_SHAPE
    torch.manual_seed(1)
    w1 = torch.randn(*cs, _M, _N, dtype=torch.bfloat16) * 0.05
    w3 = torch.randn(*cs, _M, _N, dtype=torch.bfloat16) * 0.05
    Nsc = _N // _RING

    def golden(w):
        s = torch.sum(w, dim=_CA)  # [8,M,N]
        g = torch.zeros(*cs, _M, Nsc, dtype=torch.bfloat16)
        for c in range(_RING):
            g[:, c] = s[:, :, c * Nsc : (c + 1) * Nsc]
        return g

    g1, g3 = golden(w1), golden(w3)

    def up(w):
        return ttnn.from_torch(
            w,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=cs),
        )

    w1t, w3t = up(w1), up(w3)
    gather = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=cs))

    # --- baseline: 2 separate RS ---
    def run2():
        for _ in range(_NI):
            a = _rs(w1t, mesh)
            b = _rs(w3t, mesh)
            ttnn.deallocate(a)
            ttnn.deallocate(b)

    o1, o3 = _rs(w1t, mesh), _rs(w3t, mesh)
    ok1, m1 = comp_pcc(g1.float(), gather(o1).float(), 0.99)
    ok3, m3 = comp_pcc(g3.float(), gather(o3).float(), 0.99)
    ttnn.deallocate(o1)
    ttnn.deallocate(o3)
    print(f"  2x-RS  PCC w1={m1} w3={m3}")
    assert ok1 and ok3
    t2 = _time_traced(run2, mesh)

    # --- fused: concat dim=3 -> 1 RS -> split ---
    def fused_once():
        cat = ttnn.concat([w1t, w3t], dim=_DIM, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [.,.,M,2N]
        red = _rs(cat, mesh)  # [.,.,M,2N/ring]
        ttnn.deallocate(cat)
        half = red.shape[-1] // 2
        a = ttnn.slice(red, [0, 0, 0, 0], [1, red.shape[1], _M, half], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.slice(red, [0, 0, 0, half], [1, red.shape[1], _M, 2 * half], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(red)
        return a, b

    def runf():
        for _ in range(_NI):
            a, b = fused_once()
            ttnn.deallocate(a)
            ttnn.deallocate(b)

    a, b = fused_once()
    okf1, mf1 = comp_pcc(g1.float(), gather(a).float(), 0.99)
    okf3, mf3 = comp_pcc(g3.float(), gather(b).float(), 0.99)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    print(f"  fused  PCC w1={mf1} w3={mf3}")
    assert okf1 and okf3, "fused RS PCC fail"
    tf = _time_traced(runf, mesh)

    print(f"\n  RESULT: 2x-RS={t2:.2f}us  fused(concat+1RS+split)={tf:.2f}us  speedup={t2/tf:.2f}x")
