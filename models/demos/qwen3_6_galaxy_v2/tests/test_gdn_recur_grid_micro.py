# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode GDN recurrent batched matmul: BH core_grid sweep.

The decode GDN recurrent step matmuls are batched over B*H (=32 padded rows x 6 heads = 192) with
small per-batch dims (~[128,128] fp32). The profile shows them on 16 cores @627us. Unlike the
single-batch MLP (auto-grid optimal), this has 192 INDEPENDENT batches -> more cores should add
batch-parallelism. This sweeps core_grid on the batched fp32 matmul, checking PCC + device-kernel time.

Run:
    python -m pytest --noconftest models/demos/qwen3_6_galaxy_v2/tests/test_gdn_recur_grid_micro.py -v -s
"""
from __future__ import annotations

import statistics
import time

import pytest
import torch

import ttnn

_MESH_SHAPE = (8, 4)
_BATCH = 6  # DECODE: B(1) x H(6 v-heads/chip)
_D = 128  # GDN head_dim
_N_WARMUP = 2
_N_TIMED = 12
_N_INNER = 4
_GRIDS = [None, (8, 8), (10, 10), (12, 10)]


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
    for _ in range(_N_WARMUP):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(_N_TIMED):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        ts.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(mesh, tid)
    return statistics.mean(ts) / _N_INNER


@pytest.mark.hardware
def test_gdn_recur_grid_sweep(bh_glx_mesh):
    from models.common.utility_functions import comp_pcc

    mesh = bh_glx_mesh
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    torch.manual_seed(0)
    a = torch.randn(1, _BATCH, _D, _D, dtype=torch.float32) * 0.05
    b = torch.randn(1, _BATCH, _D, _D, dtype=torch.float32) * 0.05
    golden = torch.matmul(a.reshape(_BATCH, _D, _D), b.reshape(_BATCH, _D, _D))

    a_tt = ttnn.from_torch(
        a,
        device=mesh,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    b_tt = ttnn.from_torch(
        b,
        device=mesh,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    print(f"\n  GDN recurrent batched matmul  batch={_BATCH} [{_D},{_D}]x[{_D},{_D}] fp32:")
    base = None
    for g in _GRIDS:
        cg = None if g is None else ttnn.CoreGrid(x=g[0], y=g[1])
        kw = {} if cg is None else {"core_grid": cg}
        try:
            o = ttnn.matmul(a_tt, b_tt, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
        except Exception as e:
            print(f"    grid {str(g):9s}: build FAIL -> {str(e)[:80]}")
            continue
        got = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float().reshape(_BATCH, _D, _D)
        ok, msg = comp_pcc(golden, got, 0.99)
        ttnn.deallocate(o)
        us = _time_traced(
            lambda: [
                ttnn.deallocate(
                    ttnn.matmul(a_tt, b_tt, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
                )
                for _ in range(_N_INNER)
            ],
            mesh,
        )
        if base is None:
            base = us
        gl = "auto" if g is None else f"{g[0]}x{g[1]}={g[0]*g[1]}"
        print(f"    grid {gl:9s}: {us:7.2f} us  speedup={base/us:.2f}x  PCC={'OK' if ok else 'FAIL'} ({msg})")
