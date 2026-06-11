# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LM-head decode minimal_matmul: BH core-grid sweep (7x7=49 WH-legacy vs 12x10=120 BH).

The decode LM head (QWEN36_LM_HEAD_PLAIN_DECODE) uses ttnn.experimental.minimal_matmul with
LM_HEAD_PREFILL_PROGCFG, hardcoded to a 7x7 grid (WH-sized). BH has 120 cores. This sweeps the
compute grid at the per-col decode shape [32,1280] x [1280,62208], checking PCC (unchanged — DRAM
output, grid only affects compute distribution) and device-kernel time per grid.

Run:
    python -m pytest --noconftest models/demos/qwen3_6_galaxy_v2/tests/test_lm_head_grid_micro.py -v -s
"""
from __future__ import annotations

import statistics
import time

import pytest
import torch

import ttnn

_MESH_SHAPE = (8, 4)
_M = 32  # tile-padded decode rows
_K = 1280  # dim_per_tp = 5120/4
_N = 62208  # per-col padded vocab = 248832/4
_N_WARMUP = 2
_N_TIMED = 12
_N_INNER = 4
_GRIDS = [(7, 7), (8, 8), (10, 10), (12, 10)]


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
    times = []
    for _ in range(_N_TIMED):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        times.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(mesh, tid)
    return statistics.mean(times) / _N_INNER


@pytest.mark.hardware
def test_lm_head_grid_sweep(bh_glx_mesh):
    from models.common.utility_functions import comp_pcc

    mesh = bh_glx_mesh
    torch.manual_seed(0)
    a = torch.randn(1, 1, _M, _K, dtype=torch.bfloat16) * 0.05
    w = torch.randn(1, 1, _K, _N, dtype=torch.bfloat16) * 0.05
    golden = a.float().reshape(_M, _K) @ w.float().reshape(_K, _N)

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
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    print(f"\n  LM-head minimal_matmul [{_M},{_K}]x[{_K},{_N}]  grid sweep:")
    base = None
    for gx, gy in _GRIDS:
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=1,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=2,
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        )
        try:
            o = ttnn.experimental.minimal_matmul(
                input_tensor=a_tt,
                weight_tensor=w_tt,
                config=cfg,
                compute_kernel_config=ckc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        except Exception as e:
            print(f"    grid {gx}x{gy}={gx*gy:3d}: FAILED to build -> {str(e)[:90]}")
            continue
        got = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float().reshape(_M, _N)
        ok, msg = comp_pcc(golden, got, 0.99)
        ttnn.deallocate(o)
        us = _time_traced(
            lambda: [
                ttnn.deallocate(
                    ttnn.experimental.minimal_matmul(
                        input_tensor=a_tt,
                        weight_tensor=w_tt,
                        config=cfg,
                        compute_kernel_config=ckc,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
                for _ in range(_N_INNER)
            ],
            mesh,
        )
        if base is None:
            base = us
        print(
            f"    grid {gx}x{gy}={gx*gy:3d}: {us:7.1f} us/call  speedup={base/us:.2f}x  PCC={'OK' if ok else 'FAIL'} ({msg})"
        )
