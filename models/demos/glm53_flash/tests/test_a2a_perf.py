# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cost of an expert-parallel MoE exchange at GLM-5.3-Flash decode shapes: ttnn.all_to_all_dispatch +
all_to_all_combine (the DeepSeek-V3 Galaxy decode scheme) vs the hybrid layout's all_gather_rows +
reduce_scatter_rows + all_reduce_cols, traced replay on a (2,4) block (8 devices; 288 experts = 36 per
device) at 4 and 32 tokens per device (= 32 and 256 users on the 8x4 at 1 and 8 per chip).

    GLM53_MESH_SHAPE=2,4 pytest models/demos/glm53_flash/tests/test_a2a_perf.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
H, K, E = 4096, 8, 288
N_REP = 30


def _timed(mesh_device, fn, label):
    out = fn()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(N_REP):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / N_REP * 1e6
    logger.info(f"A2A {label}: {us:.1f} us per call (traced replay)")
    ttnn.release_trace(mesh_device, tid)
    return out, us


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=80_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("B", [4, 32])
def test_expert_exchange(mesh_device, B):
    rows, cols = MESH
    D = rows * cols
    E_loc = E // D
    mesh = GLMMesh.from_device(mesh_device)
    torch.manual_seed(0)
    # tokens [B*D, 1, 1, H] sharded over all devices (B per device); indices [B*D, 1, 1, K]; mapping expert -> device
    tokens = torch.randn(B * D, 1, 1, H, dtype=torch.bfloat16)
    idx = torch.stack([torch.randperm(E)[:K] for _ in range(B * D)]).reshape(B * D, 1, 1, K).to(torch.int16)
    mapping = torch.zeros(1, 1, E, D, dtype=torch.int16)
    for e in range(E):
        mapping[0, 0, e, e // E_loc] = 1
    shard_all = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH, dims=(0, None))  # rows get B*cols, then cols split
    # row-major flattening of the mesh: shard dim 0 over rows, then over cols -> use a two-level split
    tok_rows = tokens.reshape(rows, cols * B, 1, 1, H)
    tokens_dev = ttnn.from_torch(
        tokens,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    idx_dev = ttnn.from_torch(
        idx,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    map_dev = ttnn.from_torch(
        mapping,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    results = {}
    for axis in (0, 1):  # cluster_axis=None (full-mesh dispatch) is not exercised on Blackhole and hung on a 2x4 block
        try:
            out, us = _timed(
                mesh_device,
                lambda: ttnn.all_to_all_dispatch(tokens_dev, idx_dev, map_dev, cluster_axis=axis, num_links=1),
                f"B={B}/dev dispatch cluster_axis={axis} [{B * D} tokens, H={H}, K={K}, E={E}]",
            )
            out_tokens, out_meta = out
            logger.info(f"   dispatch out shapes {out_tokens.shape} {out_meta.shape}")
            results[("dispatch", axis)] = us
            # combine: expert contributions [E_loc(per device), T, 1, H]: one row per (expert, token) -> back to owners
            T = out_tokens.shape[0]
            contrib = ttnn.from_torch(
                torch.randn(E, T, 1, H, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            )
            _, us2 = _timed(
                mesh_device,
                lambda: ttnn.all_to_all_combine(contrib, out_meta, map_dev, cluster_axis=axis, num_links=1),
                f"B={B}/dev combine cluster_axis={axis}",
            )
            results[("combine", axis)] = us2
            ttnn.deallocate(contrib)
        except Exception as e:  # noqa: BLE001
            logger.info(f"A2A B={B} cluster_axis={axis} failed: {str(e).splitlines()[0][:200]}")
    # the hybrid layout's three collectives at the same token count per row
    x = ttnn.from_torch(
        torch.randn(1, 1, rows * B * cols, H, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH, dims=(2, None)),
    )
    _, g = _timed(mesh_device, lambda: mesh.all_gather_rows(x, dim=2), f"B={B}/dev hybrid all_gather_rows")
    big = ttnn.from_torch(
        torch.randn(1, 1, rows * rows * B * cols, H, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH, dims=(2, None)),
    )
    _, rs = _timed(mesh_device, lambda: mesh.reduce_scatter_rows(big, dim=2), f"B={B}/dev hybrid reduce_scatter_rows")
    _, ar = _timed(mesh_device, lambda: mesh.all_reduce_cols(x), f"B={B}/dev hybrid all_reduce_cols")
    logger.info(
        f"SUMMARY B={B}/dev: hybrid 3 collectives {g + rs + ar:.0f} us; a2a "
        + ", ".join(f"{k[0]}@{k[1]}={v:.0f}" for k, v in results.items())
    )
