# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn


@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize("output_shape", [[2, 32]])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("mem_config", [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_broadcast_op(t3k_mesh_device, num_devices, output_shape, layout, input_dtype, mem_config, cluster_axis):
    mesh_device = t3k_mesh_device
    mesh_shape = tuple(mesh_device.shape)
    print(mesh_shape)
    sender_coord_tuple = tuple(0 for _ in range(len(mesh_shape)))
    sender_coord = ttnn.MeshCoordinate(sender_coord_tuple)

    # Create input tensor for sender
    sender_tensor_torch = torch.arange(1, 1 + torch.prod(torch.tensor(output_shape)), dtype=torch.bfloat16).reshape(
        output_shape
    )
    # Create mesh tensor with sender's tensor at sender_coord, zeros elsewhere
    device_tensors = []
    for i in range(num_devices):
        if i == sender_coord_tuple[cluster_axis]:
            device_tensors.append(sender_tensor_torch)
        else:
            device_tensors.append(torch.zeros(output_shape, dtype=torch.bfloat16))
    mesh_tensor_torch = torch.cat(device_tensors, dim=cluster_axis)
    input_tensor_mesh = ttnn.from_torch(
        mesh_tensor_torch,
        device=mesh_device,
        layout=layout,
        dtype=input_dtype,
        memory_config=mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=cluster_axis),
    )

    # Run broadcast
    output_tensor = ttnn.experimental.broadcast(
        input_tensor_mesh,
        sender_coord=sender_coord,
        num_links=1,
        memory_config=mem_config,
        topology=ttnn.Topology.Linear,
        cluster_axis=cluster_axis,
    )

    # Convert output to torch and check all devices received sender's tensor
    output_tensor_torch = ttnn.to_torch(
        output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=cluster_axis)
    )
    for i in range(num_devices):
        received = output_tensor_torch.narrow(cluster_axis, i, 1).squeeze(cluster_axis)
        assert torch.allclose(received, sender_tensor_torch), f"Device {i} did not receive correct tensor"
