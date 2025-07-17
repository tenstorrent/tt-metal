# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
import math
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


def run_send_recv_test(
    send_device,
    recv_device,
    socket_storage_type,
    socket_fifo_size,
    tensor_shape,
    tensor_mem_config,
    tensor_dtype,
    tensor_layout,
):
    mesh_shape = send_device.shape
    sender_logical_coord = ttnn.CoreCoord(0, 0)
    recv_logical_coord = ttnn.CoreCoord(0, 1)

    socket_connections = []
    for coord in ttnn.MeshCoordinateRange(mesh_shape):
        socket_connections.append(
            ttnn.SocketConnection(
                ttnn.MeshCoreCoord(coord, sender_logical_coord), ttnn.MeshCoreCoord(coord, recv_logical_coord)
            )
        )

    socket_mem_config = ttnn.SocketMemoryConfig(socket_storage_type, socket_fifo_size)
    socket_config = ttnn.SocketConfig(socket_connections, socket_mem_config)
    send_socket, recv_socket = ttnn.create_socket_pair(send_device, recv_device, socket_config)
    torch_input = torch.randn(tensor_shape)
    input_tensor = ttnn.from_torch(
        torch_input,
        device=send_device,
        layout=tensor_layout,
        dtype=tensor_dtype,
        memory_config=tensor_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(send_device),
    )
    output_tensor = ttnn.allocate_tensor_on_device(input_tensor.spec, recv_device)
    ttnn.experimental.send_async(input_tensor, send_socket)
    ttnn.experimental.recv_async(output_tensor, recv_socket)
    ttnn.synchronize_device(send_device)
    ttnn.synchronize_device(recv_device)
    input_data = ttnn.to_torch(input_tensor, mesh_composer=ttnn.ConcatMeshToTensor(send_device, dim=0))
    output_data = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(recv_device, dim=0))
    eq, output = comp_equal(input_data, output_data)
    assert eq, output


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "per_chip_shape",
    [
        ([1, 1, 32, 4096]),
        ([1, 1, 64, 8192]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize(
    "socket_storage_type",
    [
        ttnn.BufferType.DRAM,
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize(
    "socket_fifo_size",
    [
        10 * 1024,
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC}], indirect=True)
def test_send_recv(
    t3k_mesh_device,
    per_chip_shape,
    layout,
    mem_config,
    dtype,
    socket_storage_type,
    socket_fifo_size,
):
    sender_mesh_device = t3k_mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    receiver_mesh_device = t3k_mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 4))
    run_send_recv_test(
        sender_mesh_device,
        receiver_mesh_device,
        socket_storage_type,
        socket_fifo_size,
        per_chip_shape,
        mem_config,
        dtype,
        layout,
    )
