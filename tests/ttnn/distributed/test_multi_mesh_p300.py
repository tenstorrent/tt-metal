# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Simple TTNN example demonstrating the ability to interface with multiple MeshDevices across multiple processes.
# We also demonstrate how a user can setup a pipeline across MeshDevices using sockets.
# Run this on a P300 system as follows:
# cd /path/to/tt-metal
# source python_env/bin/activate
# TT_METAL_HOME=/path/to/tt-metal PYTHONPATH=/path/to/tt-metal tt-run --rank-binding tests/tt_metal/distributed/config/p300_1x1_multi_mesh_rank_binding.yaml --mpi-args "--tag-output" python3 tests/ttnn/distributed/test_multi_mesh_p300.py
#
# Note: Both TT_METAL_HOME and PYTHONPATH must be set to the root of the tt-metal repository.

# The rank binding file initializes a single physical 1x1 mesh per process (1 chip per process).
# The mesh graph descriptor used for this workload is: tests/tt_metal/tt_fabric/custom_mesh_descriptors/p300_split_1x1_multi_mesh.textproto

import torch
import ttnn


def run_multiprocess_workload():
    torch.manual_seed(0)
    # Initialize TT-Fabric to route data over Ethernet
    # Use Fabric 2D mode in this case, to route data between meshes
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    # Each process interfaces with a single 1x1 MeshDevice (1 chip)
    # Sockets are used to route data between meshes
    mesh_shape = ttnn.MeshShape(1, 1)
    device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    # The distributed context gets initialized once a device is opened.
    # This allows the user to use Multi-Host APIs without having to explicitly initialize the distributed context.
    if not ttnn.distributed_context_is_initialized():
        raise ValueError("Distributed context not initialized")
    if int(ttnn.distributed_context_get_size()) != 2:
        raise ValueError("This test requires 2 processes to run")

    # Setup the sender and receiver sockets
    # Each socket endpoint is bound to a single core (0, 0)
    # The socket connects each physical device in the sender mesh
    # to a corresponding physical device in the receiver mesh (as specified by the socket connections)
    sender_logical_coord = ttnn.CoreCoord(0, 0)
    recv_logical_coord = ttnn.CoreCoord(0, 0)
    socket_connections = []
    for coord in ttnn.MeshCoordinateRange(mesh_shape):
        socket_connections.append(
            ttnn.SocketConnection(
                ttnn.MeshCoreCoord(coord, sender_logical_coord), ttnn.MeshCoreCoord(coord, recv_logical_coord)
            )
        )
    # Setup the socket intermediate buffer in L1 and initialize its size to 4KB
    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 4096)
    # Process 0 is the sender and process 1 is the receiver. Reflect this in the socket config
    sender_rank = 0
    receiver_rank = 1
    socket_config = ttnn.SocketConfig(socket_connections, socket_mem_config, sender_rank, receiver_rank)

    # Initialize random input tensor and move it to the device
    # Using smaller tensor size appropriate for single chip
    torch_input = torch.randn(1, 1, 1024, 1024, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    def torch_op_chain(tensor):
        return torch.exp(torch.nn.functional.relu(tensor))

    if int(ttnn.distributed_context_get_rank()) == 0:
        # Create send socket, run the first op on the input tensor and forward the result to the receiver
        send_socket = ttnn.MeshSocket(device, socket_config)
        ttnn.experimental.send_async(ttnn.relu(ttnn_input), send_socket)
    else:
        # Create the recv socket and allocate a tensor on the device to receive the result from the sender
        recv_socket = ttnn.MeshSocket(device, socket_config)
        upstream_input = ttnn.allocate_tensor_on_device(ttnn_input.spec, device)
        ttnn.experimental.recv_async(upstream_input, recv_socket)
        # Run the second op on the received tensor and convert it to a torch tensor
        torch_tensor = ttnn.to_torch(
            ttnn.from_device(ttnn.exp(upstream_input)),
        )
        # Compare the result with the expected result
        if torch.allclose(torch_tensor, torch_op_chain(torch_input)):
            print("Test passed: Output tensor matches expected tensor")
        else:
            raise ValueError("Test failed: Output tensor does not match expected tensor")
    # Issue a barrier to ensure all processes have completed the test
    ttnn.distributed_context_barrier()
    ttnn.close_device(device)


if __name__ == "__main__":
    run_multiprocess_workload()
