# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch

import ttnn


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("shape", [(30, 60), (10, 10, 30, 60), (16, 64)])
def test_copy_output_as_point_to_point_preallocated(mesh_device, shape):
    coord0 = ttnn.MeshCoordinate(0, 0)
    coord1 = ttnn.MeshCoordinate(0, 1)

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    clone_output = ttnn.clone(input_tensor)

    copy_output = ttnn.assign(clone_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    ttnn.point_to_point(
        clone_output,
        coord0,
        coord1,
        topology=ttnn.Topology.Linear,
        output_tensor=copy_output,
    )
