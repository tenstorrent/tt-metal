# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""R4 sub-block sends (Refinement 4): the wave path, pinned on, and the transport window walk.

`waves_per_block` is parked at 1 by default (measured: it does not win on this board's grid), so the default suites
never run it. This file keeps the knob live: it pins waves=2 through the planner knob (column-waves for
scatter_dim=-1, row-waves for -2), asserts the plan really took it, and checks the result against torch. It also pins
the unwaved ragged-segment case (a block row whose last transport segment is short), which the window walk must
cover completely.

Mesh: the system mesh under FABRIC_2D with the production router payload, as the perf harness opens it.
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter_program_descriptor as pd

# the op module (the package re-exports the function under the module's name)
mmrs_mod = importlib.import_module("ttnn.operations.matmul_reduce_scatter.matmul_reduce_scatter")


def _system_mesh_shape():
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return tuple(shape[i] for i in range(shape.dims()))


def _router_params():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "fabric_router_config": cfg}


def _to_mesh(stacked, mesh_device, dtype):
    rows = stacked.shape[0]
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        glob,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )


def _check(mesh_device, M, K, N, cluster_axis, scatter_dim, expect_waves):
    rows, cols = tuple(mesh_device.shape)
    G = (rows, cols)[cluster_axis]
    torch.manual_seed(0)
    a = torch.randn((rows, cols, M, K)).to(torch.bfloat16)
    w = (torch.randn((rows, cols, K, N)) * K**-0.5).to(torch.bfloat16)
    out = matmul_reduce_scatter(
        _to_mesh(a, mesh_device, ttnn.bfloat16),
        _to_mesh(w, mesh_device, ttnn.bfloat16),
        cluster_axis=cluster_axis,
        scatter_dim=scatter_dim,
        num_links=1,
    )
    # plans of this shape built under the current knob setting (the cache key ends with WAVES_PIN, WAVES_MAX,
    # K_BLOCKS_MIN; other tests in the session may have cached the same shape unpinned)
    plans = [
        p
        for key, (_, p) in mmrs_mod._PLAN_CACHE.items()
        if key[-3:] == (pd.WAVES_PIN, pd.WAVES_MAX, pd.K_BLOCKS_MIN)
        and p["blk"].Mt == M // 32
        and p["blk"].Nt == N // 32
        and p["blk"].scatter_dim == scatter_dim
    ]
    assert plans and all(p["blk"].waves == expect_waves for p in plans), [p["blk"].waves for p in plans]

    total = (a.float() @ w.float()).sum(dim=cluster_axis, keepdim=True)
    host = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=(rows, cols))
    )
    om, on = (M // G, N) if scatter_dim == -2 else (M, N // G)
    for r in range(rows):
        for c in range(cols):
            p = (r, c)[cluster_axis]
            ref = torch.chunk(total[0 if cluster_axis == 0 else r, 0 if cluster_axis == 1 else c], G, dim=scatter_dim)[
                p
            ]
            got = host[r * om : (r + 1) * om, c * on : (c + 1) * on].float()
            pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
            assert pcc > 0.999, f"device ({r},{c}) pcc {pcc}"


@pytest.mark.parametrize("mesh_device", [_system_mesh_shape()], indirect=True)
@pytest.mark.parametrize("device_params", [_router_params()], indirect=True)
@pytest.mark.parametrize(
    "M,K,N,scatter_dim",
    [
        (128, 512, 3584, -1),  # column-waves: block 4 x 28 tiles, waves of 14 columns (whole 7-tile segments)
        (512, 512, 1024, -2),  # row-waves: block 4 x 32 tiles, waves of 2 rows
    ],
    ids=["cols", "rows"],
)
def test_waves_pinned(mesh_device, monkeypatch, M, K, N, scatter_dim):
    monkeypatch.setattr(pd, "WAVES_PIN", 2)
    _check(mesh_device, M, K, N, 1, scatter_dim, expect_waves=2)


@pytest.mark.parametrize("mesh_device", [_system_mesh_shape()], indirect=True)
@pytest.mark.parametrize("device_params", [_router_params()], indirect=True)
def test_ragged_segment_unwaved(mesh_device):
    # block 2 x 16 tiles at G = 4: with 7-tile segments a block row is 7 + 7 + 2 tiles (ragged last segment)
    _check(mesh_device, 64, 256, 2048, 1, -1, expect_waves=1)
