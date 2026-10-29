# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark: the all-gather of the per-column (max, index) pairs of the drafter sampler (U rows, 2 fp32 per row and column) for several layouts / widths, traced (10 calls per trace).
Prints PAIR_GATHER_BENCH <variant> <us per call>."""

import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import ccl_lean
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 50_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("U", [2, 4])
@torch.no_grad()
def test_pair_gather_bench(mesh_device, U):
    md = mesh_device
    rows, cols = tuple(md.shape)
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    lean = ccl_lean.make_lean(mesh_4x8())
    ccl_lean.make_lean_sems(ccl)
    mk = lambda w, lay=ttnn.TILE_LAYOUT, dt=ttnn.float32: ttnn.from_torch(
        torch.randn(1, 1, U, w),
        device=md,
        dtype=dt,
        layout=lay,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    variants = {
        "w2_tile": (lambda t: mc.allgather(t, ccl, axis=1, dim=3), mk(2)),
        "w32_tile": (lambda t: mc.allgather(t, ccl, axis=1, dim=3), mk(32)),
        "w2_tile_lean": (lambda t: lean.allgather(t, ccl, axis=1, dim=3, lean_ag=True), mk(2)),
        "w32_tile_lean": (lambda t: lean.allgather(t, ccl, axis=1, dim=3, lean_ag=True), mk(32)),
        "w32_bf16_tile": (lambda t: mc.allgather(t, ccl, axis=1, dim=3), mk(32, dt=ttnn.bfloat16)),
    }
    for name, (fn, t) in variants.items():
        try:
            fn(t)
            ttnn.synchronize_device(md)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            for _ in range(10):
                fn(t)
            ttnn.end_trace_capture(md, tid, cq_id=0)
            ttnn.synchronize_device(md)
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(10):
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            us = (time.perf_counter() - t0) / 100 * 1e6
            ttnn.release_trace(md, tid)
            print(f"PAIR_GATHER_BENCH U={U} {name} {us:.1f} us", flush=True)
        except Exception as e:  # noqa
            print(f"PAIR_GATHER_BENCH U={U} {name} FAILED {type(e).__name__} {str(e)[:100]}", flush=True)
