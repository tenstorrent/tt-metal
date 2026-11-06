# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared setup for the distributed_rms_norm perf tests: the QuietBox 2 ring stand-in (1x4 mesh under
FABRIC_2D_TORUS_XY, 14 KiB + 64 B router payload, the production router config) and small tensor helpers."""

import torch
import ttnn

TP = 4
FABRIC = ttnn.FabricConfig.FABRIC_2D_TORUS_XY


def open_ring_mesh(trace_region_size):
    from tests.scripts.common import get_updated_device_params

    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = 14 * 1024 + 64
    ttnn.set_fabric_config(
        FABRIC,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        cfg,
    )
    params = get_updated_device_params({"fabric_config": FABRIC})
    params.pop("fabric_config")
    params["trace_region_size"] = trace_region_size
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, TP), **params)


def close_ring_mesh(mesh):
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def width_sharded(t, mesh, dtype, layout):
    """Torch (..., W) -> per-device (..., W / TP) slices along the ring axis."""
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(1, TP), dims=(None, t.dim() - 1)),
    )


def replicated(t, mesh, dtype, layout):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def rms_norm_ref(x, gamma, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * gamma.float()


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()
