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

Multi-upstream mode:
  When upstream_sockets (list) or upstream_core_coords (list) is provided, the sender
  side uses the d2d_exchange_multiple_upstreams kernel which receives from N upstream
  sockets (one per worker), cycles through them, and assembles a single downstream page.
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
        upstream_sockets=None,
        upstream_core_coords=None,
        upstream_page_size=None,
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

        # Determine multi-upstream mode
        self.multi_upstream = upstream_sockets is not None or upstream_core_coords is not None
        if self.multi_upstream:
            assert upstream_page_size is not None, "upstream_page_size required for multi-upstream mode"
            assert upstream_socket is None, "Cannot mix upstream_socket (singular) with multi-upstream params"
            assert upstream_core_coord is None, "Cannot mix upstream_core_coord (singular) with multi-upstream params"
            assert page_size > 0, "page_size must be positive"
            assert socket_fifo_size >= page_size, "socket_fifo_size must be at least page_size"
            assert (
                socket_fifo_size % page_size == 0
            ), f"socket_fifo_size ({socket_fifo_size}) must be a multiple of page_size ({page_size})"
            self.upstream_page_size = upstream_page_size

        self.upstream_socket = None
        self.upstream_sockets_list = None
        self.downstream_socket = None

        if sender_mesh.get_mesh_device():
            if self.multi_upstream:
                if upstream_sockets is not None:
                    self.upstream_sockets_list = upstream_sockets
                elif upstream_core_coords is not None:
                    self.upstream_socket_pairs = []
                    buffer_depth = socket_fifo_size // page_size
                    upstream_fifo_size = self.upstream_page_size * buffer_depth
                    for uc in upstream_core_coords:
                        socket_connection = ttnn.SocketConnection(uc, send_core_coord)
                        socket_memory_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, upstream_fifo_size)
                        socket_config = ttnn.SocketConfig([socket_connection], socket_memory_config)
                        pair = ttnn.create_socket_pair(self.mesh_device, self.mesh_device, socket_config)
                        self.upstream_socket_pairs.append(pair)
                    self.upstream_sockets_list = [pair[1] for pair in self.upstream_socket_pairs]
            else:
                if upstream_socket is not None:
                    assert upstream_socket.get_mesh_device().get_system_mesh_id() == sender_mesh.get_mesh_id()
                    self.upstream_socket = upstream_socket
                    assert upstream_core_coord is None
                else:
                    socket_connection = ttnn.SocketConnection(upstream_core_coord, send_core_coord)
                    socket_memory_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, socket_fifo_size)
                    socket_config = ttnn.SocketConfig([socket_connection], socket_memory_config)
                    self.upstream_socket_pair = ttnn.create_socket_pair(
                        self.mesh_device, self.mesh_device, socket_config
                    )
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
        if self.multi_upstream and sender_mesh.get_mesh_device():
            sync_core_range = ttnn.CoreRangeSet(
                [ttnn.CoreRange(send_core_coord.core_coord, send_core_coord.core_coord)]
            )
            self._page_ready_sem = ttnn.create_global_semaphore(
                self.mesh_device, sync_core_range, 0, ttnn.BufferType.L1
            )
            self._ncrisc_done_sem = ttnn.create_global_semaphore(
                self.mesh_device, sync_core_range, 0, ttnn.BufferType.L1
            )

        self.sender_packet_header_cb_index = (
            0 if sender_packet_header_cb_index is None else sender_packet_header_cb_index
        )
        self.receiver_packet_header_cb_index = (
            1 if receiver_packet_header_cb_index is None else receiver_packet_header_cb_index
        )

    def _create_single_upstream_program(
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

        num_fwd_links = 2
        num_bwd_links = 1

        fabric_max_payload_size = 0
        num_whole_fabric_packets_per_link = 0
        partial_packet_size_per_link = 0

        if use_fabric_on_receiver or use_fabric_on_sender:
            fabric_max_payload_size = ttnn.get_tt_fabric_max_payload_size_bytes()
            page_size_per_link = self.page_size // num_fwd_links
            num_whole_fabric_packets_per_link = page_size_per_link // fabric_max_payload_size
            partial_packet_size_per_link = page_size_per_link % fabric_max_payload_size
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
            num_whole_fabric_packets_per_link,
            fabric_max_payload_size,
            partial_packet_size_per_link,
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

    def _create_multi_upstream_program(
        self,
        my_mesh_device,
        my_core_coord,
        my_upstream_sockets,
        my_downstream_socket,
        packet_header_cb_index,
    ):
        for s in my_upstream_sockets:
            assert s.get_active_cores()[0] == my_core_coord
        assert my_downstream_socket.get_active_cores()[0] == my_core_coord

        num_upstream = len(my_upstream_sockets)
        assert num_upstream == 8, "Multi-upstream kernel requires exactly 8 upstream sockets"
        assert num_upstream % 2 == 0

        num_sockets_per_risc = num_upstream // 2
        brisc_sockets = my_upstream_sockets[:num_sockets_per_risc]
        ncrisc_sockets = my_upstream_sockets[num_sockets_per_risc:]
        brisc_socket_addrs = [s.get_config_buffer_address() for s in brisc_sockets]
        ncrisc_socket_addrs = [s.get_config_buffer_address() for s in ncrisc_sockets]

        downstream_socket_config_addr = my_downstream_socket.get_config_buffer_address()

        my_downstream_recv_device_coord = my_downstream_socket.get_connection_config()[0].receiver_core.device_coord
        my_fabric_node_id = my_mesh_device.get_fabric_node_id(my_core_coord.device_coord)
        my_downstream_fabric_node_id = my_downstream_socket.get_fabric_node_id(
            ttnn.SocketEndpoint.RECEIVER, my_downstream_recv_device_coord
        )

        use_fabric_on_sender = my_downstream_fabric_node_id != my_fabric_node_id

        # For upstream: check if any upstream socket requires fabric (all must be on the same remote device)
        use_fabric_on_receiver = False
        my_upstream_fabric_node_id = None
        for s in my_upstream_sockets:
            sender_device_coord = s.get_connection_config()[0].sender_core.device_coord
            upstream_fid = s.get_fabric_node_id(ttnn.SocketEndpoint.SENDER, sender_device_coord)
            if upstream_fid != my_fabric_node_id:
                use_fabric_on_receiver = True
                if my_upstream_fabric_node_id is None:
                    my_upstream_fabric_node_id = upstream_fid
                else:
                    assert (
                        my_upstream_fabric_node_id == upstream_fid
                    ), "All upstream sockets must be on the same remote device"

        fabric_max_payload_size = 0
        num_whole_fabric_packets_per_link = 0
        partial_packet_size = 0

        # 2 forward headers (one per RISC) + 2 backward headers (one per RISC, if fabric on receiver)
        num_fwd_headers = 2
        num_bwd_headers = 2 if use_fabric_on_receiver else 0

        packet_header_cb_desc = None
        if use_fabric_on_receiver or use_fabric_on_sender:
            fabric_max_payload_size = ttnn.get_tt_fabric_max_payload_size_bytes()
            num_whole_fabric_packets_per_link = self.upstream_page_size // fabric_max_payload_size
            partial_packet_size = self.upstream_page_size % fabric_max_payload_size
            packet_header_cb_num_pages = num_fwd_headers + num_bwd_headers
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

        page_ready_sem_addr = ttnn.get_global_semaphore_address(self._page_ready_sem)
        ncrisc_done_sem_addr = ttnn.get_global_semaphore_address(self._ncrisc_done_sem)

        def _build_ct_args(socket_addrs, socket_start_idx, pkt_hdr_slot_start):
            ct_args = [
                downstream_socket_config_addr,  # 0: sender_socket_config_addr
                num_sockets_per_risc,  # 1: num_upstream_sockets_per_risc
                self.upstream_page_size,  # 2: upstream_page_size
                ttnn.get_global_semaphore_address(self.termination_semaphore),  # 3
                self.page_size,  # 4: page_size (total sender page)
                num_whole_fabric_packets_per_link,  # 5
                fabric_max_payload_size,  # 6: whole_packet_size
                partial_packet_size,  # 7
                packet_header_cb_index,  # 8
                use_fabric_on_receiver,  # 9
                use_fabric_on_sender,  # 10
                page_ready_sem_addr,  # 11
                ncrisc_done_sem_addr,  # 12
                socket_start_idx,  # 13
                pkt_hdr_slot_start,  # 14
            ]
            ct_args.extend(socket_addrs)  # 15-18
            return ct_args

        core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(my_core_coord.core_coord, my_core_coord.core_coord)])
        kernel_source = "models/demos/deepseek_v3_b1/micro_ops/d2d_exchange/kernels/d2d_exchange_multiple_upstreams.cpp"

        # BRISC kernel: handles sockets 0..N/2-1, packet header slots 0-1, fabric link 0
        brisc_kernel = ttnn.KernelDescriptor(
            kernel_source=kernel_source,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_ranges,
            compile_time_args=_build_ct_args(brisc_socket_addrs, 0, 0),
            config=ttnn.WriterConfigDescriptor(),
        )

        # NCRISC kernel: handles sockets N/2..N-1, packet header slots 2-3, fabric link 1
        if use_fabric_on_sender and not use_fabric_on_receiver:
            ncrisc_pkt_hdr_slot_start = 1
        else:
            ncrisc_pkt_hdr_slot_start = 2
        ncrisc_kernel = ttnn.KernelDescriptor(
            kernel_source=kernel_source,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_ranges,
            compile_time_args=_build_ct_args(ncrisc_socket_addrs, num_sockets_per_risc, ncrisc_pkt_hdr_slot_start),
            config=ttnn.ReaderConfigDescriptor(),
        )

        program = ttnn.ProgramDescriptor(
            kernels=[brisc_kernel, ncrisc_kernel],
            semaphores=[],
            cbs=[packet_header_cb_desc] if packet_header_cb_desc is not None else [],
        )

        cx, cy = my_core_coord.core_coord.x, my_core_coord.core_coord.y

        # BRISC runtime args: 1 forward fabric link (idx 0) + 1 backward link (idx 0)
        program.kernels[0].runtime_args[cx][cy] = []
        brisc_rt = program.kernels[0].runtime_args[cx][cy]

        if use_fabric_on_sender:
            fwd_args = ttnn.setup_fabric_connection(
                my_fabric_node_id,
                my_downstream_fabric_node_id,
                0,
                program,
                my_core_coord.core_coord,
            )
            brisc_rt.extend(fwd_args)

        if use_fabric_on_receiver:
            bwd_args = ttnn.setup_fabric_connection(
                my_fabric_node_id,
                my_upstream_fabric_node_id,
                0,
                program,
                my_core_coord.core_coord,
            )
            brisc_rt.extend(bwd_args)

        # NCRISC runtime args: 1 forward fabric link (idx 1) + 1 backward link (idx 1)
        program.kernels[1].runtime_args[cx][cy] = []
        ncrisc_rt = program.kernels[1].runtime_args[cx][cy]

        if use_fabric_on_sender:
            fwd_args = ttnn.setup_fabric_connection(
                my_fabric_node_id,
                my_downstream_fabric_node_id,
                1,
                program,
                my_core_coord.core_coord,
            )
            ncrisc_rt.extend(fwd_args)

        if use_fabric_on_receiver:
            bwd_args = ttnn.setup_fabric_connection(
                my_fabric_node_id,
                my_upstream_fabric_node_id,
                1,
                program,
                my_core_coord.core_coord,
            )
            ncrisc_rt.extend(bwd_args)

        return program

    def _create_sender_program(self, my_mesh_device, my_core_coord, my_downstream_socket, packet_header_cb_index):
        if self.multi_upstream:
            return self._create_multi_upstream_program(
                my_mesh_device,
                my_core_coord,
                self.upstream_sockets_list,
                my_downstream_socket,
                packet_header_cb_index,
            )
        else:
            return self._create_single_upstream_program(
                my_mesh_device,
                my_core_coord,
                self.upstream_socket,
                my_downstream_socket,
                packet_header_cb_index,
            )

    def run(self):
        dummy_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device
        )
        if self.local_socket:
            sender_program = self._create_sender_program(
                self.mesh_device,
                self.send_core_coord,
                self.internal_socket_pair[0],
                self.sender_packet_header_cb_index,
            )

            receiver_program = self._create_single_upstream_program(
                self.mesh_device,
                self.recv_core_coord,
                self.internal_socket_pair[1],
                self.downstream_socket,
                self.receiver_packet_header_cb_index,
            )
        else:
            if self.upstream_socket or self.upstream_sockets_list:
                program = self._create_sender_program(
                    self.mesh_device,
                    self.send_core_coord,
                    self.internal_socket,
                    self.sender_packet_header_cb_index,
                )
            else:
                assert self.downstream_socket, "Internal Error - Has no upstream or downstream socket"
                program = self._create_single_upstream_program(
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
                for i, k in enumerate(sender_program.kernels):
                    combined_program.kernels[i].runtime_args = k.runtime_args
                for i, k in enumerate(receiver_program.kernels):
                    combined_program.kernels[len(sender_program.kernels) + i].runtime_args = k.runtime_args
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
                self.send_core_coord.device_coord
                if (self.upstream_socket or self.upstream_sockets_list)
                else self.recv_core_coord.device_coord
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
                ttnn.synchronize_device(self.mesh_device)
                if self.downstream_socket:
                    ttnn.synchronize_device(self.downstream_socket.get_mesh_device())
            else:
                ttnn.synchronize_device(self.mesh_device)

    def get_downstream_socket(self):
        return self.downstream_socket_pair[1]

    def get_upstream_socket(self):
        return self.upstream_socket_pair[0]

    def get_upstream_sockets(self):
        """Return sender sides of all upstream socket pairs (for multi-upstream mode)."""
        return [pair[0] for pair in self.upstream_socket_pairs]
