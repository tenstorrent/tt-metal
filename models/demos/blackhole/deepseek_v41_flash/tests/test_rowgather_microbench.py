# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cost of the masked gather + sum that replaces an (o, max, sum) merge when a user's compressed KV pool is sharded over mesh rows: each chip looks the TOP-512 selected rows of every token up in its local
shard (rows it does not own are masked to zero), then ONE all-gather over the mesh rows + a sum rebuilds the 512 rows on every row (exact: one non-zero partial + zeros). Measured over the 4 mesh rows
(cluster_axis 0 = a user spanning all rows, B = 1; B = 2 would use row pairs) for T = 1 and T = 6 tokens per user, 512 selected rows of 512 bf16 each. Prints RG lines: ms per call (traced, 20 calls)."""

import time

import pytest
import torch

import ttnn
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

REP = 20
ROWS_PER = 512  # selected rows per token
DIM = 512
SHARD_ROWS = 65536  # rows per shard (any)


def traced_us(mesh, fn, n=10):
    def body():
        for _ in range(REP):
            fn()

    body()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(2):
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
def test_rowgather(mesh_device):
    md = mesh_device
    cfg, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    rep = ttnn.ReplicateTensorToMesh(md)
    tab = ttnn.from_torch(
        torch.randn(SHARD_ROWS, DIM).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    for T in (1, 6):
        n = T * ROWS_PER
        idx = ttnn.from_torch(
            torch.randint(0, SHARD_ROWS, (n, 1), dtype=torch.int32),
            device=md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        mask = ttnn.from_torch(
            (torch.rand(1, 1, n, 1) < 0.25).to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

        def local():
            e = ttnn.embedding(idx, tab, layout=ttnn.ROW_MAJOR_LAYOUT)  # [n, 1, 512]
            return ttnn.multiply(ttnn.to_layout(ttnn.reshape(e, [1, 1, n, DIM]), ttnn.TILE_LAYOUT), mask)

        def full():
            g = cfg.allgather(local(), ccl, axis=0, dim=0)  # [4, 1, n, 512] over the mesh rows
            return ttnn.sum(g, dim=0, keepdim=True)

        try:
            a = traced_us(md, local)
            b = traced_us(md, full)
            print(
                f"RG T={T} ({n} rows, {n * DIM * 2 / 2**20:.1f} MiB): local lookup+mask {a:.1f} us; + all-gather over 4 mesh rows + sum {b:.1f} us per layer",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"RG T={T}: ERR {str(e)[:300]}", flush=True)
