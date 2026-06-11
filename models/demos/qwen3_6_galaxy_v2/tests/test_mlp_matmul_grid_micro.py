# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode MLP matmul (w1/w3, w2): BH core_grid sweep.

The decode MLP (_mlp_decode_qwen36) uses plain ttnn.linear with NO core_grid, so ttnn auto-picks
40 (w2) / 68 (w1/w3) of BH's 120 cores. This sweeps core_grid at the decode shapes, checking PCC
(unchanged) and device-kernel time per grid — w2 is the most under-gridded.

Run:
    python -m pytest --noconftest models/demos/qwen3_6_galaxy_v2/tests/test_mlp_matmul_grid_micro.py -v -s
"""
from __future__ import annotations

import statistics
import time

import pytest
import torch

import ttnn

_MESH_SHAPE = (8, 4)
_N_WARMUP = 2
_N_TIMED = 12
_N_INNER = 4
# (label, M, K, N): w1/w3 = 1280->2176 ; w2 = 2176->1280
_CASES = [("w2 (down 2176->1280)", 32, 2176, 1280), ("w1/w3 (gate/up 1280->2176)", 32, 1280, 2176)]
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
def test_mlp_matmul_grid_sweep(bh_glx_mesh):
    from models.common.utility_functions import comp_pcc

    mesh = bh_glx_mesh
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    torch.manual_seed(0)
    for label, M, K, N in _CASES:
        a = torch.randn(1, 1, M, K, dtype=torch.bfloat16) * 0.05
        w = torch.randn(1, 1, K, N, dtype=torch.bfloat16) * 0.05
        golden = a.float().reshape(M, K) @ w.float().reshape(K, N)
        a_tt = ttnn.from_torch(
            a,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        w_tt = ttnn.from_torch(
            w,
            device=mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        print(f"\n  {label}  [{M},{K}]x[{K},{N}]:")
        base = None
        for g in _GRIDS:
            cg = None if g is None else ttnn.CoreGrid(x=g[0], y=g[1])
            kw = {} if cg is None else {"core_grid": cg}
            try:
                o = ttnn.linear(
                    a_tt,
                    w_tt,
                    compute_kernel_config=ckc,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    **kw,
                )
            except Exception as e:
                print(f"    grid {str(g):8s}: build FAIL -> {str(e)[:80]}")
                continue
            got = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float().reshape(M, N)
            ok, msg = comp_pcc(golden, got, 0.99)
            ttnn.deallocate(o)
            us = _time_traced(
                lambda: [
                    ttnn.deallocate(
                        ttnn.linear(
                            a_tt,
                            w_tt,
                            compute_kernel_config=ckc,
                            dtype=ttnn.bfloat16,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            **kw,
                        )
                    )
                    for _ in range(_N_INNER)
                ],
                mesh,
            )
            if base is None:
                base = us
            gl = "auto" if g is None else f"{g[0]}x{g[1]}={g[0]*g[1]}"
            print(f"    grid {gl:9s}: {us:7.2f} us  speedup={base/us:.2f}x  PCC={'OK' if ok else 'FAIL'} ({msg})")
