# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-measurement driver for distributed_rms_norm (not a correctness gate).

Run with --no-precompile: the up-front precompile pass leaves this driver's from_torch inputs zero in the real pass
(the op itself is unaffected: the acceptance suite passes with precompile on).
Runs the LOOSE_CASES perf geometries on the box's 4-chip snake ring (1 x 4, Ring, 2 links, bf16 x / gamma, HiFi4 +
fp32 acc, eps 1e-5) DRMS_PERF_CALLS times each, back to back, then checks the last result. Run under
`scripts/run_safe_pytest.sh --profile` and read DEVICE KERNEL DURATION per device from the CSV.
Also prints the selected plan (regime / split) for each case.
"""

from __future__ import annotations

import os

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.distributed_rms_norm.topology import ring_mock_meshes
from ttnn.operations.distributed_rms_norm import distributed_rms_norm, plan_summary

_RINGS = list(ring_mock_meshes())
CASES = [(1, 1, 640, 1792), (1, 1, 32, 1792), (1, 1, 2048, 1280)]
CALLS = int(os.environ.get("DRMS_PERF_CALLS", "5"))


@pytest.fixture(scope="module")
def mesh_cache():
    cache = _FabricMeshCache()
    yield cache
    cache.close()


@pytest.mark.skipif(not _RINGS, reason="no snake ring on this cluster")
@pytest.mark.parametrize("shape", CASES, ids=lambda s: "x".join(map(str, s)))
def test_perf_ring(mesh_cache, shape):
    mesh_shape, ax = _RINGS[0]
    mesh = mesh_cache.get(mesh_shape)
    G = mesh_shape[ax]
    x = torch.randn((G, *shape), dtype=torch.float32).to(torch.bfloat16)
    g = torch.randn((G, 1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)
    tx = ttnn.from_torch(
        torch.cat(list(x), dim=1) if ax == 1 else torch.cat(list(x), dim=0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape)),
    )
    tg = ttnn.from_torch(
        torch.cat(list(g), dim=1) if ax == 1 else torch.cat(list(g), dim=0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape)),
    )
    print(
        "PLAN",
        shape,
        plan_summary(tx, cluster_axis=ax, gamma=tg, topology=ttnn.Topology.Ring, num_links=2),
    )
    out = None
    for _ in range(CALLS):
        out = distributed_rms_norm(
            tx, cluster_axis=ax, gamma=tg, epsilon=1e-5, topology=ttnn.Topology.Ring, num_links=2
        )
    ttnn.synchronize_device(mesh)
    full = torch.cat(list(x.float()), dim=-1)
    gf = torch.cat(list(g.float().reshape(G, -1)), dim=-1)
    ref = torch.nn.functional.rms_norm(full, (full.shape[-1],), gf, 1e-5)
    got = torch.cat([ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(out)], dim=-1)
    a, b = got.flatten().double(), ref.flatten().double()
    pcc = float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))
    assert pcc > 0.995, pcc
