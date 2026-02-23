# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Device-to-Device Exchange Interface.

Provides a SocketInterface class that manages bidirectional data exchange between
two cores (potentially on different devices) via D2D sockets and fabric connections.

The SocketInterface sets up:
- An upstream socket (receiving data from a previous stage)
- A downstream socket (sending data to the next stage)
- An intermediate socket pair for internal sender-receiver communication
- A termination semaphore for clean shutdown

Each stage runs a d2d_exchange kernel that receives data from upstream,
and forwards it downstream, optionally using fabric connections for cross-device transfers.
"""

import ttnn


class MeshWrapper:
    def __init__(self, mesh_device=None, mesh_id=None):
        self.mesh_device = mesh_device

        if self.mesh_device is not None:
            self.mesh_id = self.mesh_device.get_system_mesh_id()
        else:
            assert mesh_id is not None
            self.mesh_id = mesh_id

    def get_mesh_device(self):
        return self.mesh_device

    def get_mesh_id(self):
        return self.mesh_id


class SocketInterface:
    def __init__(
        self,
        page_size,
        socket_fifo_size,
        data_size_per_transfer,
        send_core_coord,
        recv_core_coord,
        upstream_socket=None,
        downstream_socket=None,
        upstream_core_coord=None,
        downstream_core_coord=None,
        sender_mesh=None,
        receiver_mesh=None,
        sender_packet_header_cb_index=None,
        receiver_packet_header_cb_index=None,
    ):
        assert (
            sender_mesh.get_mesh_device() or receiver_mesh.get_mesh_device()
        ), "Either sender or receiver mesh device must be set"

        if sender_mesh.get_mesh_device() and receiver_mesh.get_mesh_device():
            assert (
                sender_mesh.get_mesh_id() == receiver_mesh.get_mesh_id()
            ), "Sender and receiver mesh IDs must be the same when both MeshDevices are provided"
            self.mesh_device = sender_mesh.get_mesh_device()
            self.local_socket = True
        else:
            self.mesh_device = (
                sender_mesh.get_mesh_device() if sender_mesh.get_mesh_device() else receiver_mesh.get_mesh_device()
            )
            self.local_socket = False

        self.upstream_socket = None
        self.downstream_socket = None

        if sender_mesh.get_mesh_device():
            # Initialize the upstream socket on the sender mesh, or reuse an existing socket if provided
            if upstream_socket is not None:
                # If an existing socket is provided, assert that it is on the sender mesh
                assert upstream_socket.get_mesh_device().get_system_mesh_id() == sender_mesh.get_mesh_id()
                self.upstream_socket = upstream_socket
                assert upstream_core_coord is None
            else:
                # Upstream socket not provided, create a new socket, on the sender mesh
                socket_connection = ttnn.SocketConnection(upstream_core_coord, send_core_coord)
                socket_memory_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, socket_fifo_size)
                socket_config = ttnn.SocketConfig([socket_connection], socket_memory_config)
                self.upstream_socket_pair = ttnn.create_socket_pair(self.mesh_device, self.mesh_device, socket_config)
                # Initialize upstream as receiver socket
                self.upstream_socket = self.upstream_socket_pair[1]

        if receiver_mesh.get_mesh_device():
            if downstream_socket is not None:
                # If an existing socket is provided, assert that it is on the receiver mesh
                assert downstream_socket.get_mesh_device().get_system_mesh_id() == receiver_mesh.get_mesh_id()
                self.downstream_socket = downstream_socket
                assert downstream_core_coord is None
            else:
                socket_connection = ttnn.SocketConnection(recv_core_coord, downstream_core_coord)
                socket_memory_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, socket_fifo_size)
                socket_config = ttnn.SocketConfig([socket_connection], socket_memory_config)
                self.downstream_socket_pair = ttnn.create_socket_pair(self.mesh_device, self.mesh_device, socket_config)
                # Initialize downstream as sender socket
                self.downstream_socket = self.downstream_socket_pair[0]

        self.page_size = page_size
        self.send_core_coord = send_core_coord
        self.recv_core_coord = recv_core_coord

        # Create a socket between the sender and receiver cores
        socket_connection = ttnn.SocketConnection(send_core_coord, recv_core_coord)
        socket_memory_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, socket_fifo_size)

        if self.local_socket:
            # If running on a host/process where the sender and receiver meshes are the local mesh, create a local socket pair
            socket_config = ttnn.SocketConfig([socket_connection], socket_memory_config)
            self.internal_socket_pair = ttnn.create_socket_pair(
                sender_mesh.get_mesh_device(), receiver_mesh.get_mesh_device(), socket_config
            )
        else:
            # If running across multiple hosts/processes create a single socket interface
            socket_config = ttnn.SocketConfig(
                connections=[socket_connection],
                memory_config=socket_memory_config,
                sender_mesh_id=sender_mesh.get_mesh_id(),
                receiver_mesh_id=receiver_mesh.get_mesh_id(),
            )
            self.internal_socket = ttnn.MeshSocket(self.mesh_device, socket_config)

        if self.send_core_coord.core_coord == self.recv_core_coord.core_coord:
            termination_semaphore_core_range = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(self.send_core_coord.core_coord, self.send_core_coord.core_coord),
                ]
            )
        else:
            termination_semaphore_core_range = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(self.send_core_coord.core_coord, self.send_core_coord.core_coord),
                    ttnn.CoreRange(self.recv_core_coord.core_coord, self.recv_core_coord.core_coord),
                ]
            )
        self.termination_semaphore = ttnn.create_global_semaphore(
            self.mesh_device,
            termination_semaphore_core_range,
            0,
            ttnn.BufferType.L1,
        )
        self.sender_packet_header_cb_index = (
            0 if sender_packet_header_cb_index is None else sender_packet_header_cb_index
        )
        self.receiver_packet_header_cb_index = (
            1 if receiver_packet_header_cb_index is None else receiver_packet_header_cb_index
        )

    def _create_program(
        self,
        my_mesh_device,
        my_core_coord,
        my_upstream_socket,
        my_downstream_socket,
        packet_header_cb_index,
    ):
        # Upstream Socket (feeding this stage) and Downstream Socket (draining this stage) must be on my_core.
        assert my_upstream_socket.get_active_cores()[0] == my_core_coord
        assert my_downstream_socket.get_active_cores()[0] == my_core_coord

        upstream_socket_config_addr = my_upstream_socket.get_config_buffer_address()
        downstream_socket_config_addr = my_downstream_socket.get_config_buffer_address()

        my_upstream_sender_device_coord = my_upstream_socket.get_connection_config()[0].sender_core.device_coord
        my_downstream_recv_device_coord = my_downstream_socket.get_connection_config()[0].receiver_core.device_coord

        my_fabric_node_id = my_mesh_device.get_fabric_node_id(my_core_coord.device_coord)
        my_upstream_fabric_node_id = my_upstream_socket.get_fabric_node_id(
            ttnn.SocketEndpoint.SENDER, my_upstream_sender_device_coord
        )
        my_downstream_fabric_node_id = my_downstream_socket.get_fabric_node_id(
            ttnn.SocketEndpoint.RECEIVER, my_downstream_recv_device_coord
        )

        use_fabric_on_receiver = my_upstream_fabric_node_id != my_fabric_node_id
        use_fabric_on_sender = my_downstream_fabric_node_id != my_fabric_node_id

        if use_fabric_on_receiver or use_fabric_on_sender:
            fabric_max_payload_size = ttnn.get_tt_fabric_max_payload_size_bytes()
            num_whole_fabric_packets = self.page_size // fabric_max_payload_size
            partial_packet_size = self.page_size % fabric_max_payload_size
        else:
            num_whole_fabric_packets = 0
            partial_packet_size = 0
            fabric_max_payload_size = 0

        num_fwd_links = 2
        num_bwd_links = 1

        if num_whole_fabric_packets > 0:
            num_whole_fabric_packets_link_0 = (num_whole_fabric_packets // num_fwd_links) + int(partial_packet_size > 0)
            num_whole_fabric_packets_link_0 = min(num_whole_fabric_packets_link_0, num_whole_fabric_packets)
            num_whole_fabric_packets_link_1 = num_whole_fabric_packets - num_whole_fabric_packets_link_0
        else:
            num_whole_fabric_packets_link_0 = 0
            num_whole_fabric_packets_link_1 = 0

        if use_fabric_on_receiver or use_fabric_on_sender:
            packet_header_cb_num_pages = num_fwd_links + num_bwd_links
            packet_header_cb_page_size = ttnn.get_tt_fabric_packet_header_size_bytes()

            packet_header_cb_desc = ttnn.CBDescriptor(
                total_size=packet_header_cb_num_pages * packet_header_cb_page_size,
                core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(my_core_coord.core_coord, my_core_coord.core_coord)]),
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=packet_header_cb_index,
                        data_format=ttnn.uint32,
                        page_size=packet_header_cb_page_size,
                    )
                ],
            )

        kernel_ct_args = [
            downstream_socket_config_addr,
            upstream_socket_config_addr,
            ttnn.get_global_semaphore_address(self.termination_semaphore),
            self.page_size,
            num_whole_fabric_packets_link_0,
            num_whole_fabric_packets_link_1,
            fabric_max_payload_size,
            partial_packet_size,
            packet_header_cb_index,
            use_fabric_on_receiver,
            use_fabric_on_sender,
        ]

        exchange_kernel = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/d2d_exchange/kernels/d2d_exchange.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(my_core_coord.core_coord, my_core_coord.core_coord)]),
            compile_time_args=kernel_ct_args,
            config=ttnn.WriterConfigDescriptor(),
        )

        program = ttnn.ProgramDescriptor(
            kernels=[exchange_kernel],
            semaphores=[],
            cbs=[packet_header_cb_desc] if (use_fabric_on_receiver or use_fabric_on_sender) else [],
        )

        program.kernels[0].runtime_args[my_core_coord.core_coord.x][my_core_coord.core_coord.y] = []
        rt_args_ref = program.kernels[0].runtime_args[my_core_coord.core_coord.x][my_core_coord.core_coord.y]

        if use_fabric_on_sender:
            for idx in range(num_fwd_links):
                fwd_fabric_args = ttnn.setup_fabric_connection(
                    my_fabric_node_id,
                    my_downstream_fabric_node_id,
                    idx,
                    program,
                    my_core_coord.core_coord,
                )
                rt_args_ref.extend(fwd_fabric_args)

        if use_fabric_on_receiver:
            for idx in range(num_bwd_links):
                bwd_fabric_args = ttnn.setup_fabric_connection(
                    my_fabric_node_id,
                    my_upstream_fabric_node_id,
                    idx,
                    program,
                    my_core_coord.core_coord,
                )
                rt_args_ref.extend(bwd_fabric_args)

        return program

    def run(self):
        dummy_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device
        )
        if self.local_socket:
            sender_program = self._create_program(
                self.mesh_device,
                self.send_core_coord,
                self.upstream_socket,
                self.internal_socket_pair[0],
                self.sender_packet_header_cb_index,
            )

            receiver_program = self._create_program(
                self.mesh_device,
                self.recv_core_coord,
                self.internal_socket_pair[1],
                self.downstream_socket,
                self.receiver_packet_header_cb_index,
            )
        else:
            if self.upstream_socket:
                # Has an upstream socket - is sender
                program = self._create_program(
                    self.mesh_device,
                    self.send_core_coord,
                    self.upstream_socket,
                    self.internal_socket,
                    self.sender_packet_header_cb_index,
                )
            else:
                assert self.downstream_socket, "Internal Error - Has no upstream or downstream socket"
                program = self._create_program(
                    self.mesh_device,
                    self.recv_core_coord,
                    self.internal_socket,
                    self.downstream_socket,
                    self.receiver_packet_header_cb_index,
                )

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        if self.local_socket:
            same_device = self.send_core_coord.device_coord == self.recv_core_coord.device_coord
            if same_device:
                sender_cb_ids = {fd.buffer_index for cb in sender_program.cbs for fd in cb.format_descriptors}
                combined_cbs = sender_program.cbs + [
                    cb
                    for cb in receiver_program.cbs
                    if not any(fd.buffer_index in sender_cb_ids for fd in cb.format_descriptors)
                ]
                combined_program = ttnn.ProgramDescriptor(
                    kernels=sender_program.kernels + receiver_program.kernels,
                    semaphores=[],
                    cbs=combined_cbs,
                )
                # Preserve per-kernel runtime args from the independently built programs.
                combined_program.kernels[0].runtime_args = sender_program.kernels[0].runtime_args
                combined_program.kernels[1].runtime_args = receiver_program.kernels[0].runtime_args
                mesh_program_descriptor[
                    ttnn.MeshCoordinateRange(self.send_core_coord.device_coord, self.send_core_coord.device_coord)
                ] = combined_program
            else:
                mesh_program_descriptor[
                    ttnn.MeshCoordinateRange(self.send_core_coord.device_coord, self.send_core_coord.device_coord)
                ] = sender_program
                mesh_program_descriptor[
                    ttnn.MeshCoordinateRange(self.recv_core_coord.device_coord, self.recv_core_coord.device_coord)
                ] = receiver_program
        else:
            device_coord = (
                self.send_core_coord.device_coord if self.upstream_socket else self.recv_core_coord.device_coord
            )
            mesh_program_descriptor[ttnn.MeshCoordinateRange(device_coord, device_coord)] = program

        io_tensors = [
            dummy_tensor,
            dummy_tensor,
        ]
        return ttnn.generic_op(io_tensors, mesh_program_descriptor)

    def terminate(self, sync_devices):
        ttnn.reset_global_semaphore_value(self.termination_semaphore, 1)
        if sync_devices:
            if self.local_socket:
                ttnn.synchronize_device(self.upstream_socket.get_mesh_device())
                ttnn.synchronize_device(self.downstream_socket.get_mesh_device())
            else:
                ttnn.synchronize_device(self.mesh_device)

    def get_downstream_socket(self):
        return self.downstream_socket_pair[1]

    def get_upstream_socket(self):
        return self.upstream_socket_pair[0]
