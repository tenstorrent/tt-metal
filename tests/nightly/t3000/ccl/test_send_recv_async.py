# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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


def _make_socket_pair(send_device, recv_device, socket_storage_type, socket_fifo_size):
    sender_logical_coord = ttnn.CoreCoord(0, 0)
    recv_logical_coord = ttnn.CoreCoord(0, 1)
    socket_connections = [
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(coord, sender_logical_coord), ttnn.MeshCoreCoord(coord, recv_logical_coord)
        )
        for coord in ttnn.MeshCoordinateRange(send_device.shape)
    ]
    socket_mem_config = ttnn.SocketMemoryConfig(socket_storage_type, socket_fifo_size)
    socket_config = ttnn.SocketConfig(socket_connections, socket_mem_config)
    return ttnn.create_socket_pair(send_device, recv_device, socket_config)


def _send_recv_once(send_device, recv_device, send_socket, recv_socket, torch_input, mem_config, dtype, layout):
    input_tensor = ttnn.from_torch(
        torch_input,
        device=send_device,
        layout=layout,
        dtype=dtype,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(send_device),
    )
    output_tensor = ttnn.allocate_tensor_on_device(input_tensor.spec, recv_device)
    ttnn.experimental.send_async(input_tensor, send_socket)
    ttnn.experimental.recv_async(output_tensor, recv_socket)
    ttnn.synchronize_device(send_device)
    ttnn.synchronize_device(recv_device)
    return input_tensor, output_tensor


def _assert_send_recv_equal(send_device, recv_device, input_tensor, output_tensor):
    input_data = ttnn.to_torch(input_tensor, mesh_composer=ttnn.ConcatMeshToTensor(send_device, dim=0))
    output_data = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(recv_device, dim=0))
    eq, output = comp_equal(input_data, output_data)
    assert eq, output


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "socket_storage_type",
    [
        ttnn.BufferType.DRAM,
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_send_recv_program_cache(mesh_device, socket_storage_type):
    """Second send/recv hits the program cache with a new socket pair and new tensors.

    The new socket pair has the same config, so it hashes to the same program but owns new config
    buffers; the new tensors get new addresses because the first ones are kept alive. Both the
    socket config addresses and the tensor addresses must be re-applied on the hit.
    """
    send_device = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    recv_device = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 4))
    for device in (send_device, recv_device):
        device.enable_program_cache()
        device.clear_program_cache()

    tensor_shape = [1, 1, 32, 4096]
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    socket_fifo_size = 10 * 1024

    send_socket_miss, recv_socket_miss = _make_socket_pair(
        send_device, recv_device, socket_storage_type, socket_fifo_size
    )
    input_miss, output_miss = _send_recv_once(
        send_device,
        recv_device,
        send_socket_miss,
        recv_socket_miss,
        torch.randn(tensor_shape),
        mem_config,
        dtype,
        layout,
    )
    _assert_send_recv_equal(send_device, recv_device, input_miss, output_miss)
    send_cache_entries = send_device.num_program_cache_entries()
    recv_cache_entries = recv_device.num_program_cache_entries()

    send_socket_hit, recv_socket_hit = _make_socket_pair(
        send_device, recv_device, socket_storage_type, socket_fifo_size
    )
    input_hit, output_hit = _send_recv_once(
        send_device,
        recv_device,
        send_socket_hit,
        recv_socket_hit,
        torch.randn(tensor_shape),
        mem_config,
        dtype,
        layout,
    )

    assert send_device.num_program_cache_entries() == send_cache_entries
    assert recv_device.num_program_cache_entries() == recv_cache_entries
    _assert_send_recv_equal(send_device, recv_device, input_hit, output_hit)
    # A stale output address on the hit would have overwritten the miss output.
    _assert_send_recv_equal(send_device, recv_device, input_miss, output_miss)


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
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_send_recv(
    mesh_device,
    per_chip_shape,
    layout,
    mem_config,
    dtype,
    socket_storage_type,
    socket_fifo_size,
):
    sender_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    receiver_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 4))
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
