# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced cost of the per-round top-2 logits (SpecDecoder.forward: ttnn.topk k=2 on the column-shard logits + all_gather over the 8 columns), U=16 users x n=6 verify rows.
Prints T2 lines (us per call, 20 calls in one trace)."""

import time

import pytest
import torch

import ttnn
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

REP = 20


def traced_us(mesh, fn, n=30):
    def body():
        for _ in range(REP):
            fn()

    body()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(3):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t) / n / REP
    ttnn.release_trace(mesh, tid)
    return dt * 1e6


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_top2_cost(mesh_device):
    md = mesh_device
    cfg = mesh_4x8()
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    for T in (96, 16):
        logits = ttnn.from_torch(
            torch.randn(1, 1, T, 16160),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        tk = lambda: ttnn.topk(logits, k=2, dim=-1, largest=True, sorted=True)[0]
        t2 = tk()
        a = traced_us(md, tk)
        b = traced_us(md, lambda: cfg.allgather(t2, ccl, axis=1, dim=3))
        print(f"T2 T={T}: topk {a:.1f} us + all_gather {b:.1f} us = {a + b:.1f} us per round", flush=True)
