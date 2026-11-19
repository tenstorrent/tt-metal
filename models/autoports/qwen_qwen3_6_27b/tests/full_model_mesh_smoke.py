# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded-command mesh health check after a failed full-model job."""

import torch

import ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
try:
    x = ttnn.from_torch(
        torch.ones(32, 32),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    out = ttnn.add(x, 1)
    assert torch.equal(ttnn.to_torch(ttnn.get_device_tensors(out)[0]), torch.full((32, 32), 2.0))
    print("MESH_SMOKE_OK", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
