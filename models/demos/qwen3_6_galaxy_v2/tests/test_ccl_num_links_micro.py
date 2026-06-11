# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""BH_GLX CCL num_links sweep: how many fabric links does Blackhole Galaxy actually support?

GALAXY_NUM_LINKS is hardcoded `1 if is_blackhole() else 4` (a bring-up default, NOT measured).
The decode CCL stack is the #1 decode cost (RMSAllGather 29% + ReduceScatter 21% per GDN layer) and
is written for up to 3 links (`min(3, GALAXY_NUM_LINKS)`) but pinned to 1 on BH. This sweeps num_links
on the decode reduce_scatter + all_gather (cluster_axis=1 cols, Linear topology = BH default), checking
PCC + device-kernel. Tells us: does num_links=2 work + how much faster + does 3 fail (BH max=2)?

Run:
    python -m pytest --noconftest models/demos/qwen3_6_galaxy_v2/tests/test_ccl_num_links_micro.py -v -s
"""
from __future__ import annotations

import statistics
import time

import pytest
import torch

import ttnn

_MESH_SHAPE = (8, 4)
_M = 32
_N = 2176  # decode MLP w1/w3 RS output width (intermediate_per_tp)
_DIM = 3
_CA = 1  # cols, ring-4
_RING = _MESH_SHAPE[_CA]
_TOPO = ttnn.Topology.Linear  # BH_GLX default (CCL_TOPOLOGY)
_LINKS = [1, 2, 3]
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


@pytest.mark.hardware
def test_ccl_num_links(bh_glx_mesh):
    from models.common.utility_functions import comp_pcc

    mesh = bh_glx_mesh
    cs = _MESH_SHAPE
    torch.manual_seed(0)

    up = lambda w: ttnn.from_torch(
        w,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=cs),
    )
    gather = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=cs))

    # ---------- reduce_scatter [.,.,M,N] -> [.,.,M,N/ring] (sum over cols, scatter dim=3) ----------
    rs_in = torch.randn(*cs, _M, _N, dtype=torch.bfloat16) * 0.05
    Nsc = _N // _RING
    s = torch.sum(rs_in, dim=_CA)
    g_rs = torch.zeros(*cs, _M, Nsc, dtype=torch.bfloat16)
    for c in range(_RING):
        g_rs[:, c] = s[:, :, c * Nsc : (c + 1) * Nsc]
    rs_t = up(rs_in)

    print(f"\n  REDUCE_SCATTER  [{_M},{_N}] cluster_axis={_CA} ring-{_RING} {_TOPO}:")
    base = None
    for L in _LINKS:
        try:
            o = ttnn.reduce_scatter(
                rs_t, _DIM, cluster_axis=_CA, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=_TOPO, num_links=L
            )
        except Exception as e:
            print(f"    links={L}: BUILD/RUN FAIL -> {str(e)[:90]}")
            continue
        ok, msg = comp_pcc(g_rs.float(), gather(o).float(), 0.99)
        ttnn.deallocate(o)
        us = _time_traced(
            lambda: [
                ttnn.deallocate(
                    ttnn.reduce_scatter(
                        rs_t, _DIM, cluster_axis=_CA, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=_TOPO, num_links=L
                    )
                )
                for _ in range(_NI)
            ],
            mesh,
        )
        if base is None:
            base = us
        print(f"    links={L}: {us:7.2f} us  speedup={base/us:.2f}x  PCC={'OK' if ok else 'FAIL'} ({msg})")

    # ---------- all_gather [.,.,M,N/ring] -> [.,.,M,N] (gather cols) ----------
    ag_in = torch.randn(*cs, _M, Nsc, dtype=torch.bfloat16) * 0.05
    g_ag = torch.zeros(*cs, _M, _N, dtype=torch.bfloat16)
    for c in range(_RING):
        for cc in range(_RING):
            g_ag[:, c, :, cc * Nsc : (cc + 1) * Nsc] = ag_in[:, cc]
    ag_t = up(ag_in)

    print(f"\n  ALL_GATHER  [{_M},{Nsc}]->[{_M},{_N}] cluster_axis={_CA} ring-{_RING} {_TOPO}:")
    base = None
    for L in _LINKS:
        try:
            o = ttnn.all_gather(
                ag_t,
                _DIM,
                cluster_axis=_CA,
                mesh_device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=_TOPO,
                num_links=L,
            )
        except Exception as e:
            print(f"    links={L}: BUILD/RUN FAIL -> {str(e)[:90]}")
            continue
        ok, msg = comp_pcc(g_ag.float(), gather(o).float(), 0.99)
        ttnn.deallocate(o)
        us = _time_traced(
            lambda: [
                ttnn.deallocate(
                    ttnn.all_gather(
                        ag_t,
                        _DIM,
                        cluster_axis=_CA,
                        mesh_device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        topology=_TOPO,
                        num_links=L,
                    )
                )
                for _ in range(_NI)
            ],
            mesh,
        )
        if base is None:
            base = us
        print(f"    links={L}: {us:7.2f} us  speedup={base/us:.2f}x  PCC={'OK' if ok else 'FAIL'} ({msg})")
