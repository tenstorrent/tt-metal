# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debug tests for matmul_reduce_scatter (DO NOT DELETE — documents the debugging process).

Two bugs found with these:
1. One shared sem_block_ack: acks of different consumer kinds (fwd ports, bwd ports, finals) arrive out of order, so
   a bwd-port ack released a hand-off slot the fwd port had not read yet -> relays added a later block's partial.
   Fix: one ack counter per consumer kind.
2. Cumulative ready counting: a fast compute core's next-block ready signal stood in for a slow core's missing
   current-block signal, so a line-end port gathered a hand-off slot before it was packed (stale = the previous
   call's own block, or zeros on the first call). Only shows without the watcher (--dev hides it). Fix: a core signals
   its next block of a kind only after that kind's consumers acked every earlier block of the kind.

`test_far_block_repeated` is the repro of (2): the line-end port's second block (compute index 1) on a 4-chip line,
repeated calls, every element compared and the wrong tiles decomposed onto per-device partials.
"""

import pytest
import torch
import ttnn

from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter

FABRIC = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}]


def _system_mesh_shape():
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return tuple(shape[i] for i in range(shape.dims()))


def _to_mesh(stacked, mesh_device, dtype=ttnn.bfloat16):
    rows, cols = stacked.shape[0], stacked.shape[1]
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        glob,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )


def _partials(a, w):
    return torch.matmul(a.float(), w.float().reshape(*w.shape[:2], *([1] * (a.dim() - 4)), *w.shape[2:]))


def _expected(a, w, cluster_axis, scatter_dim):
    part = _partials(a, w)
    total = part.sum(dim=cluster_axis, keepdim=True).expand_as(part)
    rows, cols = a.shape[0], a.shape[1]
    g = (rows, cols)[cluster_axis]
    return [
        [torch.chunk(total[r, c], g, dim=scatter_dim)[(r, c)[cluster_axis]] for c in range(cols)] for r in range(rows)
    ]


def _bad_tiles(actual, ref, tol):
    H, W = ref.shape[-2] // 32, ref.shape[-1] // 32
    err = (actual - ref).abs().reshape(H, 32, W, 32).amax(dim=(1, 3))
    return [(i, j) for i in range(H) for j in range(W) if err[i, j] > tol]


@pytest.mark.parametrize("mesh_device", [_system_mesh_shape()], indirect=True)
@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("scatter_dim", [-1, -2])
def test_constant_partials_exact(mesh_device, scatter_dim):
    """Every partial tile is 32 * (rank + 1): checks routing / block ownership / device placement exactly."""
    rows, cols = tuple(mesh_device.shape)
    a_shape, w_shape = (1, 1, 256, 256), (256, 256)
    a = torch.zeros((rows, cols, *a_shape), dtype=torch.bfloat16)
    a[..., :32] = 1
    w = torch.zeros((rows, cols, *w_shape))
    w[:, :, :32, :] = torch.arange(1, rows * cols + 1).float().reshape(rows, cols, 1, 1)
    w = w.to(torch.bfloat16)
    exp = _expected(a, w, 1, scatter_dim)
    out = matmul_reduce_scatter(
        _to_mesh(a, mesh_device), _to_mesh(w, mesh_device), cluster_axis=1, scatter_dim=scatter_dim, num_links=1
    )
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        assert torch.equal(ttnn.to_torch(t).float(), exp[r][c].float()), f"device ({r},{c})"


@pytest.mark.parametrize("mesh_device", [_system_mesh_shape()], indirect=True)
@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
def test_far_block_repeated(mesh_device):
    """Repro of the cumulative-ready race: random partials, HiFi2 + fp32 DEST, 6 back-to-back calls. A wrong tile is
    reported with the per-device partial it is missing."""
    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(0)
    a_shape, w_shape = (1, 1, 256, 1024), (1024, 1024)
    a = torch.randn((rows, cols, *a_shape)).to(torch.bfloat16)
    w = (torch.randn((rows, cols, *w_shape)) * w_shape[0] ** -0.5).to(torch.bfloat16)
    exp = _expected(a, w, 1, -1)
    ta, tw = _to_mesh(a, mesh_device), _to_mesh(w, mesh_device)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)
    for rep in range(6):
        out = matmul_reduce_scatter(ta, tw, cluster_axis=1, scatter_dim=-1, num_links=1, compute_kernel_config=cfg)
        for idx, t in enumerate(ttnn.get_device_tensors(out)):
            r, c = divmod(idx, cols)
            bad = _bad_tiles(ttnn.to_torch(t).float(), exp[r][c].float(), 0.25)
            assert not bad, f"call {rep}, device ({r},{c}): wrong tiles {bad}"
