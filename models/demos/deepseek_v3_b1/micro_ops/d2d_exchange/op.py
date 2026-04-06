# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

ParallelSocketInterface:
  Manages N parallel 1-1 D2D socket connections between two pipeline stages.
  Each channel has its own core pair (send_core, recv_core) running an independent
  d2d_exchange kernel, but all channels share a single termination semaphore and
  are dispatched together via a single generic_op call.
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


def _build_exchange_program(
    page_size,
    termination_semaphore,
    my_mesh_device,
    my_core_coord,
    my_upstream_socket,
    my_downstream_socket,
    packet_header_cb_index,
):
    """Build a d2d_exchange program for a single upstream->downstream path on one core."""
    assert my_upstream_socket.get_active_cores()[0] == my_core_coord
    assert my_downstream_socket.get_active_cores()[0] == my_core_coord

    print(
        f"  _build_exchange_program: core={my_core_coord} page_size={page_size} "
        f"upstream_conn={my_upstream_socket.get_connection_config()[0]} "
        f"downstream_conn={my_downstream_socket.get_connection_config()[0]}"
    )

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

    packet_header_cb_desc = None
    if use_fabric_on_receiver or use_fabric_on_sender:
        fabric_max_payload_size = ttnn.get_tt_fabric_max_payload_size_bytes()
        page_size_per_link = page_size // num_fwd_links
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
        ttnn.get_global_semaphore_address(termination_semaphore),
        page_size,
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
        cbs=[packet_header_cb_desc] if packet_header_cb_desc is not None else [],
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
            print(
                f"  _build_exchange: fwd setup_fabric_connection idx={idx} "
                f"src_node={my_fabric_node_id} dst_node={my_downstream_fabric_node_id} "
                f"core={my_core_coord.core_coord} "
                f"returned eth_ch={fwd_fabric_args[0]} args={list(fwd_fabric_args)}"
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
            print(
                f"  _build_exchange: bwd setup_fabric_connection idx={idx} "
                f"src_node={my_fabric_node_id} dst_node={my_upstream_fabric_node_id} "
                f"core={my_core_coord.core_coord} "
                f"returned eth_ch={bwd_fabric_args[0]} args={list(bwd_fabric_args)}"
            )
            rt_args_ref.extend(bwd_fabric_args)

    return program


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
        return _build_exchange_program(
            self.page_size,
            self.termination_semaphore,
            my_mesh_device,
            my_core_coord,
            my_upstream_socket,
            my_downstream_socket,
            packet_header_cb_index,
        )

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
        brisc_count = (num_upstream + 1) // 2
        brisc_sockets = my_upstream_sockets[:brisc_count]
        ncrisc_sockets = my_upstream_sockets[brisc_count:]
        num_sockets_per_risc = brisc_count
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

        page_ready_sem_id = 0
        ncrisc_done_sem_id = 1

        def _build_ct_args(socket_addrs, socket_start_idx, pkt_hdr_slot_start):
            ct_args = [
                downstream_socket_config_addr,  # 0: sender_socket_config_addr
                len(socket_addrs),  # 1: num_sockets_this_risc
                self.upstream_page_size,  # 2: upstream_page_size
                ttnn.get_global_semaphore_address(self.termination_semaphore),  # 3
                self.page_size,  # 4: page_size (total sender page)
                num_whole_fabric_packets_per_link,  # 5
                fabric_max_payload_size,  # 6: whole_packet_size
                partial_packet_size,  # 7
                packet_header_cb_index,  # 8
                use_fabric_on_receiver,  # 9
                use_fabric_on_sender,  # 10
                page_ready_sem_id,  # 11
                ncrisc_done_sem_id,  # 12
                socket_start_idx,  # 13
                pkt_hdr_slot_start,  # 14
            ]
            ct_args.extend(socket_addrs)  # 15..15+len(socket_addrs)-1
            return ct_args

        core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(my_core_coord.core_coord, my_core_coord.core_coord)])
        print("running d2d exchange with multi-upstream support:")
        kernel_source = "models/demos/deepseek_v3_b1/micro_ops/d2d_exchange/kernels/d2d_exchange_multiple_upstreams.cpp"

        # BRISC (Writer/NOC0) gets ceil(N/2) sockets; NCRISC (Reader/NOC1) gets floor(N/2).
        # Giving the odd-one-out to BRISC ensures local NOC writes use NOC0 addresses.
        # BRISC kernel: handles sockets 0..brisc_count-1, packet header slots 0-1, fabric link 0
        brisc_kernel = ttnn.KernelDescriptor(
            kernel_source=kernel_source,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_ranges,
            compile_time_args=_build_ct_args(brisc_socket_addrs, 0, 0),
            config=ttnn.WriterConfigDescriptor(),
        )

        # NCRISC kernel: handles sockets brisc_count..N-1, packet header slots 2-3, fabric link 1
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

        page_ready_sem_desc = ttnn.SemaphoreDescriptor(
            id=page_ready_sem_id,
            core_ranges=core_ranges,
            initial_value=0,
        )
        ncrisc_done_sem_desc = ttnn.SemaphoreDescriptor(
            id=ncrisc_done_sem_id,
            core_ranges=core_ranges,
            initial_value=0,
        )

        program = ttnn.ProgramDescriptor(
            kernels=[brisc_kernel, ncrisc_kernel],
            semaphores=[page_ready_sem_desc, ncrisc_done_sem_desc],
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

    def build_programs(self):
        """Build exchange programs and return (device_coord, program) pairs without dispatching.

        This enables multiple SocketInterface instances to have their programs
        merged and dispatched together in a single generic_op call.
        """
        entries = []
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
                for i, k in enumerate(sender_program.kernels):
                    combined_program.kernels[i].runtime_args = k.runtime_args
                for i, k in enumerate(receiver_program.kernels):
                    combined_program.kernels[len(sender_program.kernels) + i].runtime_args = k.runtime_args
                entries.append((self.send_core_coord.device_coord, combined_program))
            else:
                entries.append((self.send_core_coord.device_coord, sender_program))
                entries.append((self.recv_core_coord.device_coord, receiver_program))
        else:
            if self.upstream_socket or self.upstream_sockets_list:
                program = self._create_sender_program(
                    self.mesh_device,
                    self.send_core_coord,
                    self.internal_socket,
                    self.sender_packet_header_cb_index,
                )
                entries.append((self.send_core_coord.device_coord, program))
            else:
                assert self.downstream_socket, "Internal Error - Has no upstream or downstream socket"
                program = self._create_single_upstream_program(
                    self.mesh_device,
                    self.recv_core_coord,
                    self.internal_socket,
                    self.downstream_socket,
                    self.receiver_packet_header_cb_index,
                )
                entries.append((self.recv_core_coord.device_coord, program))
        return entries

    def run(self):
        entries = self.build_programs()

        dummy_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device
        )

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        for device_coord, program in entries:
            mesh_program_descriptor[ttnn.MeshCoordinateRange(device_coord, device_coord)] = program

        io_tensors = [dummy_tensor, dummy_tensor]
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


def _group_by_device(entries):
    """Group (device_coord, program) entries by device_coord using equality comparison."""
    groups = []
    for device_coord, prog in entries:
        found = False
        for group_coord, group_progs in groups:
            if device_coord == group_coord:
                group_progs.append(prog)
                found = True
                break
        if not found:
            groups.append((device_coord, [prog]))
    return groups


class ParallelSocketInterface:
    """Manages N parallel 1-1 D2D socket connections between two pipeline stages.

    Each channel i has its own (send_core_coords[i], recv_core_coords[i]) pair
    running an independent d2d_exchange kernel. All channels share a single
    termination semaphore and are dispatched together in one generic_op call.
    """

    def __init__(
        self,
        page_size,
        socket_fifo_size,
        send_core_coords,
        recv_core_coords,
        upstream_sockets=None,
        downstream_sockets=None,
        upstream_core_coords=None,
        downstream_core_coords=None,
        sender_mesh=None,
        receiver_mesh=None,
        sender_packet_header_cb_index=None,
        receiver_packet_header_cb_index=None,
    ):
        num_channels = len(send_core_coords)
        assert len(recv_core_coords) == num_channels, "send/recv core coord lists must have equal length"
        assert num_channels > 0, "Must have at least one channel"

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

        self.page_size = page_size
        self.num_channels = num_channels
        self.send_core_coords = list(send_core_coords)
        self.recv_core_coords = list(recv_core_coords)

        self._upstream_sockets = [None] * num_channels
        self._downstream_sockets = [None] * num_channels
        self._upstream_socket_pairs = [None] * num_channels
        self._downstream_socket_pairs = [None] * num_channels
        self._internal_pairs = [None] * num_channels
        self._internal_sockets = [None] * num_channels

        socket_memory_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, socket_fifo_size)

        for i in range(num_channels):
            if sender_mesh.get_mesh_device():
                if upstream_sockets is not None and upstream_sockets[i] is not None:
                    assert upstream_sockets[i].get_mesh_device().get_system_mesh_id() == sender_mesh.get_mesh_id()
                    self._upstream_sockets[i] = upstream_sockets[i]
                elif upstream_core_coords is not None and upstream_core_coords[i] is not None:
                    conn = ttnn.SocketConnection(upstream_core_coords[i], send_core_coords[i])
                    cfg = ttnn.SocketConfig([conn], socket_memory_config)
                    pair = ttnn.create_socket_pair(self.mesh_device, self.mesh_device, cfg)
                    self._upstream_socket_pairs[i] = pair
                    self._upstream_sockets[i] = pair[1]

            if receiver_mesh.get_mesh_device():
                if downstream_sockets is not None and downstream_sockets[i] is not None:
                    assert downstream_sockets[i].get_mesh_device().get_system_mesh_id() == receiver_mesh.get_mesh_id()
                    self._downstream_sockets[i] = downstream_sockets[i]
                elif downstream_core_coords is not None and downstream_core_coords[i] is not None:
                    conn = ttnn.SocketConnection(recv_core_coords[i], downstream_core_coords[i])
                    cfg = ttnn.SocketConfig([conn], socket_memory_config)
                    pair = ttnn.create_socket_pair(self.mesh_device, self.mesh_device, cfg)
                    self._downstream_socket_pairs[i] = pair
                    self._downstream_sockets[i] = pair[0]

            conn = ttnn.SocketConnection(send_core_coords[i], recv_core_coords[i])
            if self.local_socket:
                cfg = ttnn.SocketConfig([conn], socket_memory_config)
                self._internal_pairs[i] = ttnn.create_socket_pair(
                    sender_mesh.get_mesh_device(), receiver_mesh.get_mesh_device(), cfg
                )
            else:
                cfg = ttnn.SocketConfig(
                    connections=[conn],
                    memory_config=socket_memory_config,
                    sender_mesh_id=sender_mesh.get_mesh_id(),
                    receiver_mesh_id=receiver_mesh.get_mesh_id(),
                )
                self._internal_sockets[i] = ttnn.MeshSocket(self.mesh_device, cfg)

        all_core_ranges = []
        seen_cores = set()
        for i in range(num_channels):
            for cc in (send_core_coords[i].core_coord, recv_core_coords[i].core_coord):
                key = (cc.x, cc.y)
                if key not in seen_cores:
                    all_core_ranges.append(ttnn.CoreRange(cc, cc))
                    seen_cores.add(key)

        self.termination_semaphore = ttnn.create_global_semaphore(
            self.mesh_device,
            ttnn.CoreRangeSet(all_core_ranges),
            0,
            ttnn.BufferType.L1,
        )

        self.sender_packet_header_cb_index = (
            0 if sender_packet_header_cb_index is None else sender_packet_header_cb_index
        )
        self.receiver_packet_header_cb_index = (
            1 if receiver_packet_header_cb_index is None else receiver_packet_header_cb_index
        )

    def build_programs(self):
        """Build exchange programs and return (device_coord, program) pairs without dispatching.

        This enables multiple ParallelSocketInterface instances to have their programs
        merged and dispatched together in a single generic_op call, which is required
        when entry and exit d2d_exchange kernels share the same device.
        """
        device_program_entries = []

        if self.local_socket:
            for i in range(self.num_channels):
                sender_prog = _build_exchange_program(
                    self.page_size,
                    self.termination_semaphore,
                    self.mesh_device,
                    self.send_core_coords[i],
                    self._upstream_sockets[i],
                    self._internal_pairs[i][0],
                    self.sender_packet_header_cb_index,
                )
                receiver_prog = _build_exchange_program(
                    self.page_size,
                    self.termination_semaphore,
                    self.mesh_device,
                    self.recv_core_coords[i],
                    self._internal_pairs[i][1],
                    self._downstream_sockets[i],
                    self.receiver_packet_header_cb_index,
                )
                device_program_entries.append((self.send_core_coords[i].device_coord, sender_prog))
                device_program_entries.append((self.recv_core_coords[i].device_coord, receiver_prog))
        else:
            for i in range(self.num_channels):
                if self._upstream_sockets[i] is not None:
                    prog = _build_exchange_program(
                        self.page_size,
                        self.termination_semaphore,
                        self.mesh_device,
                        self.send_core_coords[i],
                        self._upstream_sockets[i],
                        self._internal_sockets[i],
                        self.sender_packet_header_cb_index,
                    )
                    device_program_entries.append((self.send_core_coords[i].device_coord, prog))
                else:
                    assert self._downstream_sockets[i] is not None, f"Channel {i}: no upstream or downstream socket"
                    prog = _build_exchange_program(
                        self.page_size,
                        self.termination_semaphore,
                        self.mesh_device,
                        self.recv_core_coords[i],
                        self._internal_sockets[i],
                        self._downstream_sockets[i],
                        self.receiver_packet_header_cb_index,
                    )
                    device_program_entries.append((self.recv_core_coords[i].device_coord, prog))

        return device_program_entries

    def run(self):
        device_program_entries = self.build_programs()

        dummy_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device
        )

        groups = _group_by_device(device_program_entries)

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        for device_coord, progs in groups:
            if len(progs) == 1:
                merged = progs[0]
            else:
                merged = ttnn.merge_program_descriptors(progs)
            mesh_program_descriptor[ttnn.MeshCoordinateRange(device_coord, device_coord)] = merged

        io_tensors = [dummy_tensor, dummy_tensor]
        return ttnn.generic_op(io_tensors, mesh_program_descriptor)

    def terminate(self, sync_devices):
        ttnn.reset_global_semaphore_value(self.termination_semaphore, 1)
        if sync_devices:
            if self.local_socket:
                ttnn.synchronize_device(self.mesh_device)
                for ds in self._downstream_sockets:
                    if ds is not None:
                        ttnn.synchronize_device(ds.get_mesh_device())
                        break
            else:
                ttnn.synchronize_device(self.mesh_device)

    def get_downstream_sockets(self):
        """Return receiver sockets from all downstream socket pairs (one per channel)."""
        return [pair[1] for pair in self._downstream_socket_pairs if pair is not None]

    def get_upstream_sockets(self):
        """Return sender sockets from all upstream socket pairs (one per channel)."""
        return [pair[0] for pair in self._upstream_socket_pairs if pair is not None]

    def get_downstream_socket(self, channel_idx):
        """Return receiver socket from the downstream socket pair for a specific channel."""
        assert self._downstream_socket_pairs[channel_idx] is not None
        return self._downstream_socket_pairs[channel_idx][1]

    def get_upstream_socket(self, channel_idx):
        """Return sender socket from the upstream socket pair for a specific channel."""
        assert self._upstream_socket_pairs[channel_idx] is not None
        return self._upstream_socket_pairs[channel_idx][0]
