# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""X.10 follow-up: the serving path's token-id readback (tt/runners/adapter._host_ids: one blocking to_torch per mesh
row of the engine's uint32 [SP, 1, chunk / SP] input) timed on the mesh alone (no model): idle device, and behind a
queued device op of known length (the read waits for the queue)."""

import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
BATCH_SHARED = True
BATCH_LAYERS = "executed"


@mesh_parametrize(timeout_s=600)
def test_ids_readback(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn

    rows, cols = tuple(mesh_device.shape)
    chunk = int(S.get("target.chunk"))
    w = chunk // rows
    src = torch.randint(0, 120000, (rows, 1, w), dtype=torch.int64).to(torch.uint32)
    inp = ttnn.from_torch(
        src,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])
        ),
    )

    def host_ids():  # exactly adapter._host_ids' device reads
        dev = ttnn.get_device_tensors(inp)
        return [ttnn.to_torch(dev[r * cols]).reshape(-1) for r in range(rows)]

    for _ in range(3):
        host_ids()
    ttnn.synchronize_device(mesh_device)
    idle = []
    for _ in range(20):
        t0 = time.perf_counter()
        host_ids()
        idle.append((time.perf_counter() - t0) * 1e3)
    one = []
    for _ in range(20):
        dev = ttnn.get_device_tensors(inp)
        t0 = time.perf_counter()
        ttnn.to_torch(dev[0])
        one.append((time.perf_counter() - t0) * 1e3)
    # behind queued work: a big elementwise op on every chip, then the reads (they wait for the queue)
    x = ttnn.from_torch(
        torch.randn(1, 1, 8192, 8192),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    t0 = time.perf_counter()
    y = ttnn.exp(x)
    ttnn.synchronize_device(mesh_device)
    op_ms = (time.perf_counter() - t0) * 1e3
    behind = []
    for _ in range(10):
        y = ttnn.exp(x)
        t0 = time.perf_counter()
        host_ids()
        behind.append((time.perf_counter() - t0) * 1e3)
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    print(
        f"IDS_READBACK idle: 8 reads median {med(idle):.3f} ms (runs {[round(v, 3) for v in idle]}); one read {med(one):.3f} ms; "
        f"queued op alone {op_ms:.2f} ms; reads issued right after it: median {med(behind):.3f} ms",
        flush=True,
    )
    metrics.record("ids_readback_ms_idle", round(med(idle), 3))
    metrics.record("ids_readback_ms_one_read", round(med(one), 3))
    metrics.record("ids_readback_ms_behind_op", round(med(behind), 3), op_ms=round(op_ms, 2))
