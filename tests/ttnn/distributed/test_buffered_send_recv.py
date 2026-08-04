# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Correctness tests for the ``buffered_send`` / ``buffered_recv`` experimental ops.

``buffered_recv`` takes ``N`` output tensors forming a ring of receive buffers, so these tests send a
different tensor on each iteration and check it lands in the expected slot of the rotation.
"""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


def _build_socket_connections(mesh_shape: ttnn.MeshShape, num_connections: int):
    # Senders in core row 0, receivers in row 1: the socket runtime forbids a core appearing in two
    # connections of the same socket.
    sender_cores = [ttnn.CoreCoord(i, 0) for i in range(num_connections)]
    recv_cores = [ttnn.CoreCoord(i, 1) for i in range(num_connections)]

    connections = []
    for coord in ttnn.MeshCoordinateRange(mesh_shape):
        for sender, receiver in zip(sender_cores, recv_cores):
            connections.append(
                ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(coord, sender),
                    ttnn.MeshCoreCoord(coord, receiver),
                )
            )
    return connections


def _run_buffered_send_recv_case(
    mesh_device,
    num_connections,
    socket_page_size,
    tensor_shape,
    num_buffers,
):
    torch.manual_seed(0)

    sender_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
    receiver_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))

    mesh_shape = sender_mesh_device.shape

    socket_connections = _build_socket_connections(mesh_shape, num_connections)
    # The handshake FIFO must be in L1; only the handshake page flows through it.
    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, socket_page_size * 4)
    socket_config = ttnn.SocketConfig(socket_connections, socket_mem_config)
    send_socket, recv_socket = ttnn.create_socket_pair(sender_mesh_device, receiver_mesh_device, socket_config)

    def _make_input(seed):
        torch_input = torch.randn(tensor_shape, dtype=torch.float32, generator=torch.Generator().manual_seed(seed))
        return torch_input, ttnn.from_torch(
            torch_input,
            device=sender_mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
            mesh_mapper=ttnn.ReplicateTensorToMesh(sender_mesh_device),
        )

    num_iterations = 4

    _, spec_tensor = _make_input(0)
    output_tensors = [
        ttnn.allocate_tensor_on_device(spec_tensor.spec, receiver_mesh_device) for _ in range(num_buffers)
    ]

    for iteration in range(num_iterations):
        # A unique input per iteration, so a stale buffer would show up as a mismatch.
        torch_input, input_tensor = _make_input(iteration)
        ttnn.experimental.buffered_send(input_tensor, send_socket)
        ttnn.experimental.buffered_recv(output_tensors, recv_socket)
        ttnn.synchronize_device(sender_mesh_device)
        ttnn.synchronize_device(receiver_mesh_device)

        input_data = ttnn.to_torch(input_tensor, mesh_composer=ttnn.ConcatMeshToTensor(sender_mesh_device, dim=0))
        output_data = ttnn.to_torch(
            output_tensors[iteration % num_buffers],
            mesh_composer=ttnn.ConcatMeshToTensor(receiver_mesh_device, dim=0),
        )
        eq, msg = comp_equal(input_data, output_data)
        assert eq, f"iteration {iteration}: {msg}"


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": 2048}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "tensor_shape",
    [[968, 2048]],
    ids=lambda v: f"size{v}",
)
@pytest.mark.parametrize(
    "socket_page_size",
    [2048],
    ids=lambda v: f"page{v}",
)
@pytest.mark.parametrize(
    "num_connections",
    [1, 2],
    ids=lambda v: f"conn{v}",
)
@pytest.mark.parametrize(
    "num_buffers",
    [3],
    ids=lambda v: f"buffers{v}",
)
def test_buffered_send_recv(
    mesh_device,
    num_connections,
    socket_page_size,
    tensor_shape,
    num_buffers,
):
    """Send a tensor from the device at (0, 0) to the device at (0, 1) and back through the ring."""
    _run_buffered_send_recv_case(
        mesh_device,
        num_connections,
        socket_page_size,
        tensor_shape,
        num_buffers,
    )
