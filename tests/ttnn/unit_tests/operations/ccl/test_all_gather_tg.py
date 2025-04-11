# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.mark.parametrize("mesh_device", [(1, 32)], indirect=True)
def test_all_gather_submeshes(mesh_device):
    submesh_devices = mesh_device.create_submeshes(ttnn.MeshShape(1, 8))

    for device in submesh_devices:
        torch_input_tensor = torch.rand((1, 1, 32, 256), dtype=torch.bfloat16)

        mesh_tensor = ttnn.from_torch(
            torch_input_tensor,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3),
        )

        ttnn.visualize_mesh_device(device, tensor=mesh_tensor)

        _ = ttnn.all_gather(
            mesh_tensor,
            dim=3,
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
