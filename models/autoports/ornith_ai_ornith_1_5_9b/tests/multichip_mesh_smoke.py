# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded externally: verify every chip's dispatch path after failure recovery."""
import torch

import ttnn


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        a = ttnn.from_torch(
            torch.ones(32, 32, dtype=torch.bfloat16),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        b = ttnn.add(a, a)
        ttnn.synchronize_device(mesh)
        assert len(ttnn.get_device_tensors(b)) == 4
        assert all(
            torch.equal(ttnn.to_torch(v), torch.full((32, 32), 2, dtype=torch.bfloat16))
            for v in ttnn.get_device_tensors(b)
        )
        print("MESH_SMOKE_PASS", mesh.get_device_ids(), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
