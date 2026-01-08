# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Simple TTNN example demonstrating the ability to interface with two N300 devices across processes.
# We also demonstrate how a user can setup a pipeline across MeshDevices using sockets.
# Run this on a system with 2x N300 devices as follows:
#
# t3k --rank-binding tests/ttnn/distributed/disaggregated_pd_rank_binding.yaml \
#     --mpi-args "--tag-output" \
#     python3 tests/ttnn/distributed/test_n150_dual_mesh.py
#
# The rank binding file initializes a single physical 1x2 mesh per process (one N300 each, 2 chips per N300).
# The mesh graph descriptor used for this workload is:
#   tests/tt_metal/tt_fabric/custom_mesh_descriptors/n300_dual_mesh_graph_descriptor.textproto

import torch
import ttnn


def run_n150_dual_mesh_workload():
    torch.manual_seed(0)

    # Initialize TT-Fabric to route data over Ethernet
    # Use Fabric 1D mode for simple multi-node topology
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    # Each process interfaces with a single 1x2 MeshDevice (one N300, 2 chips per board)
    # Sockets are used to route data between meshes
    # Mesh shape must match the device topology in the mesh graph descriptor (1x2 for N300)
    mesh_shape = ttnn.MeshShape(1, 2)

    # Get the visible device IDs for this rank (set by TT_VISIBLE_DEVICES in rank binding)
    # For N300, setting TT_VISIBLE_DEVICES="0" or "1" exposes both chips (PCIe + remote)
    # So get_device_ids() should return exactly 2 device IDs for the 1x2 mesh
    visible_device_ids = ttnn.get_device_ids()
    if len(visible_device_ids) != 2:
        raise ValueError(
            f"Expected exactly 2 devices for N300 (1x2 mesh), got {len(visible_device_ids)}. "
            f"Visible device IDs: {visible_device_ids}. "
            f"Make sure TT_VISIBLE_DEVICES is set correctly in the rank binding."
        )

    # Use both devices (the 2 chips of the N300 board)
    # Specifying physical_device_ids explicitly bypasses SystemMesh lookup issues
    physical_device_ids = visible_device_ids
    device = ttnn.open_mesh_device(mesh_shape=mesh_shape, physical_device_ids=physical_device_ids)

    # The distributed context gets initialized once a device is opened.
    # This allows the user to use Multi-Host APIs without having to explicitly initialize the distributed context.
    if not ttnn.distributed_context_is_initialized():
        raise ValueError("Distributed context not initialized. Run with t3k or tt-run.")
    num_processes = int(ttnn.distributed_context_get_size())
    if num_processes != 2:
        raise ValueError(f"This test requires 2 processes to run, got {num_processes}")

    rank = int(ttnn.distributed_context_get_rank())
    print(f"Process {rank} started on N300 device")

    # Setup the sender and receiver sockets
    # Each socket endpoint is bound to a single core (0, 0)
    # The socket connects the physical device in the sender mesh (process 0)
    # to the physical device in the receiver mesh (process 1)
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
    # For 1x2 mesh (N300), we use ReplicateTensorToMesh (replicate across the 2 chips)
    torch_input = torch.randn(1, 1, 1024, 1024, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    def torch_op_chain(tensor):
        """Reference operation: exp(relu(tensor))"""
        return torch.exp(torch.nn.functional.relu(tensor))

    if rank == 0:
        # === SENDER NODE (Process 0) ===
        # Create send socket, run the first op on the input tensor and forward the result to the receiver
        send_socket = ttnn.MeshSocket(device, socket_config)
        relu_output = ttnn.relu(ttnn_input)
        print(f"Process {rank}: Computed relu, sending to process 1...")
        ttnn.experimental.send_async(relu_output, send_socket)
        print(f"Process {rank}: Data sent successfully")
    elif rank == 1:
        # === RECEIVER NODE (Process 1) ===
        # Create the recv socket and allocate a tensor on the device to receive the result from the sender
        recv_socket = ttnn.MeshSocket(device, socket_config)
        upstream_input = ttnn.allocate_tensor_on_device(ttnn_input.spec, device)
        print(f"Process {rank}: Waiting to receive data from process 0...")
        ttnn.experimental.recv_async(upstream_input, recv_socket)
        print(f"Process {rank}: Data received, computing exp...")

        # Run the second op on the received tensor and convert it to a torch tensor
        exp_output = ttnn.exp(upstream_input)
        # Since the tensor is distributed on a 1x2 mesh (replicated), we need a mesh composer
        # to concatenate the shards. Since it's replicated, concatenating doubles the size,
        # so we slice to get back the original size (take first half since data is duplicated).
        torch_tensor_full = ttnn.to_torch(
            ttnn.from_device(exp_output), mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1)
        )
        # Slice to get original size (replicated tensor, so both halves are identical)
        torch_tensor = torch_tensor_full[:, :, :, : torch_input.shape[3]]

        # Compare the result with the expected result
        if torch.allclose(torch_tensor, torch_op_chain(torch_input), rtol=1e-2, atol=1e-2):
            print("Test passed: Output tensor matches expected tensor")
        else:
            max_diff = (torch_tensor - torch_op_chain(torch_input)).abs().max()
            print(f"Test failed: Output tensor does not match expected tensor. Max diff: {max_diff}")
            raise ValueError("Test failed: Output tensor does not match expected tensor")

    # Issue a barrier to ensure all processes have completed the test
    ttnn.distributed_context_barrier()
    ttnn.close_device(device)
    print(f"Process {rank}: Finished successfully")


if __name__ == "__main__":
    run_n150_dual_mesh_workload()
