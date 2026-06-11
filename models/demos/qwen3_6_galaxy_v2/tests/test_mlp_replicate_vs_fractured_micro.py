# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode MLP: 2D-fractured (current, 5 CCL) vs replicated-hidden (1-2 CCL). PCC + time.

Current decode MLP (2D-TP, fractured residual): per layer
  ff_norm RMSAllGather(cols) + w1 RS(cols) + w3 RS(cols) + AG(cols) + w2 AR(rows) = 5 CCL.
Replicated hidden (norm local, weights N-split, single reduce):
  REP_B: int split 32-way (rows x cols) -> w2 AR(rows)+AR(cols) = 2 CCL, no redundant compute.
  REP_A: int split 8-way (rows), cols redundant -> w2 AR(rows) = 1 CCL, 4x redundant matmul.
Decode is latency-bound (links A/B), so fewer CCL launches should win even at more bytes/redundant FLOPs.

Run:
    python -m pytest --noconftest models/demos/qwen3_6_galaxy_v2/tests/test_mlp_replicate_vs_fractured_micro.py -v -s
"""
from __future__ import annotations

import statistics
import time

import pytest
import torch

import ttnn

_MS = (8, 4)
_ROWS, _COLS = _MS
_M, _H, _INT = 32, 5120, 17408
_Nr = _INT // _ROWS  # 2176  (int per row)
_Kc = _H // _COLS  # 1280  (H per col)
_N32 = _INT // (_ROWS * _COLS)  # 544 (int per 32)
_Hc = _H // _COLS  # 1280  (H per col)
_NW, _NT, _NI = 2, 12, 4
_CK = None  # set in test


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*_MS), trace_region_size=120_000_000)
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
def test_mlp_replicate_vs_fractured(bh_glx_mesh):
    from models.common.utility_functions import comp_pcc

    mesh = bh_glx_mesh
    global _CK
    _CK = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    DR = ttnn.DRAM_MEMORY_CONFIG
    torch.manual_seed(0)

    x = torch.randn(_M, _H, dtype=torch.bfloat16) * 0.05
    W1 = torch.randn(_H, _INT, dtype=torch.bfloat16) * 0.03
    W3 = torch.randn(_H, _INT, dtype=torch.bfloat16) * 0.03
    W2 = torch.randn(_INT, _H, dtype=torch.bfloat16) * 0.03
    xf, W1f, W3f, W2f = x.float(), W1.float(), W3.float(), W2.float()
    golden = (torch.nn.functional.silu(xf @ W1f) * (xf @ W3f)) @ W2f  # [32,5120]

    def shard(w, dims):
        return ttnn.from_torch(
            w.reshape(1, 1, *w.shape),
            device=mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=DR,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=dims, mesh_shape=_MS),
        )

    rep = lambda w: ttnn.from_torch(
        w.reshape(1, 1, *w.shape),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=DR,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    cat = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=_MS))
    mm = lambda a, b: ttnn.linear(a, b, compute_kernel_config=_CK, dtype=ttnn.bfloat16, memory_config=DR)
    _BG = ttnn.CoreGrid(x=12, y=10)  # big BH grid for the larger replicated matmuls
    mmg = lambda a, b: ttnn.linear(
        a, b, compute_kernel_config=_CK, dtype=ttnn.bfloat16, memory_config=DR, core_grid=_BG
    )
    silu_mul = lambda a, b: ttnn.mul(
        a, b, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat16, memory_config=DR
    )

    # ============ CURRENT: 2D-fractured (5 CCL) ============
    # x col-sharded H/4, replicated rows: dims=(None,-1); W1/W3 dims=(-1,-2); W2 dims=(-2,-1)
    xc = ttnn.from_torch(
        x.reshape(1, 1, _M, _H),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=DR,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=_MS),
    )
    W1c, W3c, W2c = shard(W1, (-1, -2)), shard(W3, (-1, -2)), shard(W2, (-2, -1))

    def cur_once():
        w1 = mm(xc, W1c)
        w3 = mm(xc, W3c)  # [32,2176] partial
        w1 = ttnn.reduce_scatter(w1, 3, cluster_axis=1, memory_config=DR, topology=ttnn.Topology.Linear, num_links=1)
        w3 = ttnn.reduce_scatter(w3, 3, cluster_axis=1, memory_config=DR, topology=ttnn.Topology.Linear, num_links=1)
        ff = silu_mul(w1, w3)
        ttnn.deallocate(w1)
        ttnn.deallocate(w3)  # [32,544]
        ff = ttnn.all_gather(
            ff, 3, cluster_axis=1, memory_config=DR, topology=ttnn.Topology.Linear, num_links=1
        )  # [32,2176]
        o = mm(ff, W2c)
        ttnn.deallocate(ff)  # [32,1280] partial(rows)
        o = ttnn.all_reduce(o, cluster_axis=0, num_links=1, memory_config=DR)  # [32,1280] col-sharded H/4
        return o

    o = cur_once()
    got = cat(o)  # [8*32? ] -> need col-concat of H/4. dims=(0,1): rows replicate(dup), cols give H slices
    got = got[0:_M, :] if got.shape[0] == _M else got.reshape(_ROWS, _COLS, _M, _Hc)[0].transpose(0, 1).reshape(_M, _H)
    ok, msg = comp_pcc(golden, got.float(), 0.98)
    ttnn.deallocate(o)
    print(f"\n  CURRENT 2D-fractured (5 CCL): PCC={'OK' if ok else 'FAIL'} ({msg})")
    t_cur = _time_traced(lambda: [ttnn.deallocate(cur_once()) for _ in range(_NI)], mesh)

    xr = rep(x)  # [32,5120] everywhere

    # ============ REP_B: replicated hidden, int 32-split via 4D reshape, 2 CCL ============
    # N split [ROWS,COLS] blocks: reshape W -> [H, ROWS, COLS, N32], shard dims=(1,2) (rows-dim on
    # mesh-rows, cols-dim on mesh-cols); per chip [H,1,1,N32] -> squeeze to [H,N32]. Memory-neutral.
    def shard4d_n(w):  # [H,INT] -> per chip [H, N32]
        w4 = w.reshape(_H, _ROWS, _COLS, _N32)
        t = ttnn.from_torch(
            w4,
            device=mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=DR,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(1, 2), mesh_shape=_MS),
        )
        return ttnn.reshape(t, (1, 1, _H, _N32))

    def shard4d_k(w):  # [INT,H] -> per chip [N32, H]
        w4 = w.reshape(_ROWS, _COLS, _N32, _H)
        t = ttnn.from_torch(
            w4,
            device=mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=DR,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=_MS),
        )
        return ttnn.reshape(t, (1, 1, _N32, _H))

    try:
        W1b, W3b, W2b = shard4d_n(W1), shard4d_n(W3), shard4d_k(W2)

        def repB_once():
            w1 = mmg(xr, W1b)
            w3 = mmg(xr, W3b)  # [32,544] COMPLETE (big grid)
            ff = silu_mul(w1, w3)
            ttnn.deallocate(w1)
            ttnn.deallocate(w3)
            o = mmg(ff, W2b)
            ttnn.deallocate(ff)  # [32,5120] partial over all 32
            o = ttnn.all_reduce(o, cluster_axis=0, num_links=2, memory_config=DR)
            o = ttnn.all_reduce(o, cluster_axis=1, num_links=2, memory_config=DR)
            return o

        o = repB_once()
        got = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float().reshape(_M, _H)
        okb, msgb = comp_pcc(golden, got, 0.98)
        ttnn.deallocate(o)
        print(f"  REP_B replicated int32-split (2 CCL, mem-neutral): PCC={'OK' if okb else 'FAIL'} ({msgb})")
        t_repB = _time_traced(lambda: [ttnn.deallocate(repB_once()) for _ in range(_NI)], mesh)

        # breakdown: matmuls-only vs all_reduce-only
        def repB_mmonly():
            w1 = mmg(xr, W1b)
            w3 = mmg(xr, W3b)
            ff = silu_mul(w1, w3)
            ttnn.deallocate(w1)
            ttnn.deallocate(w3)
            o = mmg(ff, W2b)
            ttnn.deallocate(ff)
            return o

        t_mm = _time_traced(lambda: [ttnn.deallocate(repB_mmonly()) for _ in range(_NI)], mesh)
        print(f"    [breakdown] REP_B matmuls-only={t_mm:.1f}us  AR-portion~={t_repB - t_mm:.1f}us")
    except Exception as e:
        print(f"  REP_B: FAIL -> {str(e)[:140]}")
        t_repB = None

    # ============ REP_A: replicated hidden, int 8-split on rows, cols redundant, 1 CCL (4x weight mem) ============
    W1a, W3a = shard(W1, (-1, None)), shard(W3, (-1, None))  # N split rows, replicate cols
    W2a = shard(W2, (-2, None))  # K split rows, replicate cols

    def repA_once():
        w1 = mmg(xr, W1a)
        w3 = mmg(xr, W3a)  # [32,2176] COMPLETE (cols redundant, big grid)
        ff = silu_mul(w1, w3)
        ttnn.deallocate(w1)
        ttnn.deallocate(w3)
        o = mmg(ff, W2a)
        ttnn.deallocate(ff)  # [32,5120] partial over rows
        o = ttnn.all_reduce(o, cluster_axis=0, num_links=2, memory_config=DR)  # -> replicated
        return o

    try:
        o = repA_once()
        got = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float().reshape(_M, _H)
        oka, msga = comp_pcc(golden, got, 0.98)
        ttnn.deallocate(o)
        print(f"  REP_A replicated int8-split (1 CCL): PCC={'OK' if oka else 'FAIL'} ({msga})")
        t_repA = _time_traced(lambda: [ttnn.deallocate(repA_once()) for _ in range(_NI)], mesh)
    except Exception as e:
        print(f"  REP_A: FAIL -> {str(e)[:120]}")
        t_repA = None

    print(f"\n  ===== DECODE MLP TIME (full forward, traced, M={_M}) =====")
    print(f"  CURRENT 2D-fractured (5 CCL):          {t_cur:8.2f} us  (1.00x)")
    if t_repB:
        print(f"  REP_B replicated (2 CCL, mem-neutral): {t_repB:8.2f} us  ({t_cur/t_repB:.2f}x)")
    if t_repA:
        print(f"  REP_A replicated (1 CCL, 4x wt mem):   {t_repA:8.2f} us  ({t_cur/t_repA:.2f}x)")
