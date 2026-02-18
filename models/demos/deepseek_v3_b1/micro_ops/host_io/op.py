# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Host-Device Communication Interface.

Provides bidirectional communication between host and device using H2D (Host-to-Device)
and D2H (Device-to-Host) sockets with termination support.

Current Limitations:
- Only supports communication with a single core on a single chip
- H2D and D2H sockets must be on the same core
- No multi-chip or multi-core host communication support

Modes:
- Loopback mode: H2D receiver and D2H sender communicate via circular buffers (CBs)
  for testing purposes
- Socket mode: H2D receiver and D2H sender forward data to/from downstream/upstream cores
  via D2D (Device-to-Device) sockets

Components:
- H2D receiver: Pulls data from host (or receives via PCIe push), forwards to downstream
- D2H sender: Receives data from upstream, pushes to host via PCIe

Termination:
- Controlled via global semaphore that both kernels poll during blocking operations
- Ensures clean shutdown within ~1000 device cycles without hanging on socket/CB waits

Kernel implementation uses polling loops with termination checks to avoid indefinite
blocking on socket_wait_for_pages() and cb_wait_front() operations.
"""

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import get_interleaved_tensor_accessor_args
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size


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
    ):
        self.upstream_socket = upstream_socket
        self.downstream_socket = downstream_socket
        self.page_size = page_size
        self.send_core_coord = send_core_coord
        self.recv_core_coord = recv_core_coord

        # Create a socket between the sender and receiver cores
        socket_connection = ttnn.SocketConnection(send_core_coord, recv_core_coord)
        socket_memory_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, socket_fifo_size)

        socket_config = ttnn.SocketConfig([socket_connection], socket_memory_config)

        self.intermed_socket_pair = ttnn.create_socket_pair(
            self.upstream_socket.get_mesh_device(), self.downstream_socket.get_mesh_device(), socket_config
        )

        # if (self.send_core_coord.core_coord == self.recv_core_coord.core_coord):
        termination_semaphore_core_range = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(self.send_core_coord.core_coord, self.send_core_coord.core_coord),
            ]
        )
        # else:
        #     termination_semaphore_core_range = ttnn.CoreRangeSet(
        #         [
        #             ttnn.CoreRange(self.send_core_coord.core_coord, self.send_core_coord.core_coord),
        #             ttnn.CoreRange(self.recv_core_coord.core_coord, self.recv_core_coord.core_coord),
        #         ]
        #     )
        self.termination_semaphore = ttnn.create_global_semaphore(
            self.upstream_socket.get_mesh_device(),
            termination_semaphore_core_range,
            0,
            ttnn.BufferType.L1,
        )

    def _create_program(
        self,
        my_mesh_device,
        my_core_coord,
        my_upstream_socket,
        my_downstream_socket,
    ):
        # Upstream Socket (feeding this stage) and Downstream Socket (draining this stage) must be on my_core.
        assert my_upstream_socket.get_active_cores()[0] == my_core_coord
        assert my_downstream_socket.get_active_cores()[0] == my_core_coord

        upstream_socket_config_addr = my_upstream_socket.get_config_buffer_address()
        downstream_socket_config_addr = my_downstream_socket.get_config_buffer_address()

        fabric_max_payload_size = ttnn.get_tt_fabric_max_payload_size_bytes()
        num_whole_fabric_packets = self.page_size // fabric_max_payload_size
        partial_packet_size = self.page_size % fabric_max_payload_size

        num_fwd_links = 2
        num_bwd_links = 1

        if num_whole_fabric_packets > 0:
            num_whole_fabric_packets_link_0 = (num_whole_fabric_packets // num_fwd_links) + int(partial_packet_size > 0)
            num_whole_fabric_packets_link_0 = min(num_whole_fabric_packets_link_0, num_whole_fabric_packets)
            num_whole_fabric_packets_link_1 = num_whole_fabric_packets - num_whole_fabric_packets_link_0
        else:
            num_whole_fabric_packets_link_0 = 0
            num_whole_fabric_packets_link_1 = 0

        packet_header_cb_num_pages = num_fwd_links + num_bwd_links
        packet_header_cb_page_size = ttnn.get_tt_fabric_packet_header_size_bytes()
        packet_header_cb_index = 0

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
        ]

        exchange_kernel = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/d2d_exchange.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(my_core_coord.core_coord, my_core_coord.core_coord)]),
            compile_time_args=kernel_ct_args,
            config=ttnn.WriterConfigDescriptor(),
        )

        program = ttnn.ProgramDescriptor(
            kernels=[exchange_kernel],
            semaphores=[],
            cbs=[packet_header_cb_desc],
        )

        my_upstream_sender_device_coord = my_upstream_socket.get_connection_config()[0].sender_core.device_coord
        my_downstream_recv_device_coord = my_downstream_socket.get_connection_config()[0].receiver_core.device_coord

        my_fabric_node_id = my_mesh_device.get_fabric_node_id(my_core_coord.device_coord)
        my_upstream_fabric_node_id = my_upstream_socket.get_fabric_node_id(
            ttnn.SocketEndpoint.SENDER, my_upstream_sender_device_coord
        )
        my_downstream_fabric_node_id = my_downstream_socket.get_fabric_node_id(
            ttnn.SocketEndpoint.RECEIVER, my_downstream_recv_device_coord
        )

        program.kernels[0].runtime_args[my_core_coord.core_coord.x][my_core_coord.core_coord.y] = []
        rt_args_ref = program.kernels[0].runtime_args[my_core_coord.core_coord.x][my_core_coord.core_coord.y]

        for idx in range(num_fwd_links):
            fwd_fabric_args = ttnn.setup_fabric_connection(
                my_fabric_node_id,
                my_downstream_fabric_node_id,
                idx,
                program,
                my_core_coord.core_coord,
            )
            rt_args_ref.extend(fwd_fabric_args)

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
            ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.upstream_socket.get_mesh_device()
        )
        sender_program = self._create_program(
            self.upstream_socket.get_mesh_device(),
            self.send_core_coord,
            self.upstream_socket,
            self.intermed_socket_pair[0],
        )

        receiver_program = self._create_program(
            self.downstream_socket.get_mesh_device(),
            self.recv_core_coord,
            self.intermed_socket_pair[1],
            self.downstream_socket,
        )
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        mesh_program_descriptor[
            ttnn.MeshCoordinateRange(self.send_core_coord.device_coord, self.send_core_coord.device_coord)
        ] = sender_program
        mesh_program_descriptor[
            ttnn.MeshCoordinateRange(self.recv_core_coord.device_coord, self.recv_core_coord.device_coord)
        ] = receiver_program

        io_tensors = [
            dummy_tensor,
            dummy_tensor,
        ]
        return ttnn.generic_op(io_tensors, mesh_program_descriptor)

    def terminate(self):
        ttnn.reset_global_semaphore_value(self.termination_semaphore, 1)
        ttnn.synchronize_device(self.upstream_socket.get_mesh_device())
        ttnn.synchronize_device(self.downstream_socket.get_mesh_device())


class HostInterface:
    def __init__(
        self,
        h2d_socket,
        d2h_socket,
        h2d_page_size,
        d2h_page_size,
        core_to_core_socket_buffer_size=1024,
        h2d_downstream_core=ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
        d2h_upstream_core=ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
        embedding_tensor=None,
        loopback_mode=False,
    ):
        self.h2d_socket = h2d_socket
        self.d2h_socket = d2h_socket
        self.h2d_page_size = h2d_page_size
        self.d2h_page_size = d2h_page_size
        self.h2d_socket.set_page_size(self.h2d_page_size)
        self.d2h_socket.set_page_size(self.d2h_page_size)
        self.loopback_mode = loopback_mode
        self.core_to_core_socket_buffer_size = core_to_core_socket_buffer_size
        self.embedding_tensor = embedding_tensor
        self.h2d_downstream_core = h2d_downstream_core
        self.d2h_upstream_core = d2h_upstream_core
        # Validate single-core, single-chip constraint
        # Current implementation only supports host communication with one core on one chip
        if len(h2d_socket.get_active_cores()) != 1 or len(d2h_socket.get_active_cores()) != 1:
            raise ValueError("Host <-> Device Communication for Blitz Decode must be on a single core.")
        if h2d_socket.get_mesh_device().id() != d2h_socket.get_mesh_device().id():
            raise ValueError("Expected Host <-> Device Communication for Blitz Decode to be on the same mesh device.")

        if loopback_mode:
            if h2d_socket.get_active_cores()[0] != d2h_socket.get_active_cores()[0]:
                raise ValueError("Expected Host <-> Device Communication for Blitz Decode to be on the same core.")

        self.mesh_device = h2d_socket.get_mesh_device()
        self.h2d_mesh_core_coord = self.h2d_socket.get_active_cores()[0]
        self.d2h_mesh_core_coord = self.d2h_socket.get_active_cores()[0]

        # if (self.h2d_mesh_core_coord.core_coord == self.d2h_mesh_core_coord.core_coord):
        termination_semaphore_core_range = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(self.h2d_mesh_core_coord.core_coord, self.h2d_mesh_core_coord.core_coord),
            ]
        )
        # else:
        #     termination_semaphore_core_range = ttnn.CoreRangeSet(
        #         [
        #             ttnn.CoreRange(self.h2d_mesh_core_coord.core_coord, self.h2d_mesh_core_coord.core_coord),
        #             ttnn.CoreRange(self.d2h_mesh_core_coord.core_coord, self.d2h_mesh_core_coord.core_coord),
        #         ]
        #     )
        self.termination_semaphore = ttnn.create_global_semaphore(
            self.mesh_device,
            termination_semaphore_core_range,
            0,
            ttnn.BufferType.L1,
        )
        # Loopback mode is for testing purposes only - H2D receiver <-> D2H sender communication over CBs
        # For real workloads, downstream/upstream cores connect to H2D receiver and D2H sender via D2D sockets
        # NOTE: Downstream and upstream cores must be on the same device as the host I/O core (single-chip constraint)
        if not loopback_mode:
            downstream_socket_connection = ttnn.SocketConnection(
                self.h2d_mesh_core_coord,
                self.h2d_downstream_core,
            )
            upstream_socket_connection = ttnn.SocketConnection(
                self.d2h_upstream_core,
                self.d2h_mesh_core_coord,
            )
            socket_memory_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, core_to_core_socket_buffer_size)

            downstream_socket_config = ttnn.SocketConfig(
                [downstream_socket_connection],
                socket_memory_config,
            )

            upstream_socket_config = ttnn.SocketConfig(
                [upstream_socket_connection],
                socket_memory_config,
            )

            self.downstream_socket_pair = ttnn.create_socket_pair(
                self.mesh_device, self.mesh_device, downstream_socket_config
            )
            self.upstream_socket_pair = ttnn.create_socket_pair(
                self.mesh_device, self.mesh_device, upstream_socket_config
            )
        else:
            self.intermed_cb_index = 0

        self.has_embedding = self.embedding_tensor is not None
        if self.has_embedding:
            # For now, we assume that tokens will be passed in as 64 bytes packets to embedding.
            # This allows us to add more information in the input packet as needed.
            assert self.h2d_page_size == 64
            assert (
                self.embedding_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
                and self.embedding_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
                and self.embedding_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
            ), f"Expected embedding tensor to be DRAM interleaved with page size {self.embedding_page_size} bytes for shape {self.embedding_tensor.shape}"
            # Tensor is DRAM interleaved, and row major. Page size is inner dim stride.
            self.embedding_page_size = self.embedding_tensor.shape[3] * dtype_size(self.embedding_tensor.dtype)
            self.embedding_cb_index = 2

        self.fabric_packet_header_cb_index = 1
        self.num_fwd_links = 2
        self.num_bwd_links = 1

    def _create_h2d_kernel(self):
        fabric_max_payload_size = ttnn.get_tt_fabric_max_payload_size_bytes()
        num_whole_fabric_packets = self.h2d_page_size // fabric_max_payload_size
        partial_packet_size = self.h2d_page_size % fabric_max_payload_size

        if num_whole_fabric_packets > 0:
            num_whole_fabric_packets_link_0 = (num_whole_fabric_packets // self.num_fwd_links) + int(
                partial_packet_size > 0
            )
            num_whole_fabric_packets_link_0 = min(num_whole_fabric_packets_link_0, num_whole_fabric_packets)
            num_whole_fabric_packets_link_1 = num_whole_fabric_packets - num_whole_fabric_packets_link_0
        else:
            num_whole_fabric_packets_link_0 = 0
            num_whole_fabric_packets_link_1 = 0

        h2d_socket_kernel_ct_args = [
            self.h2d_socket.get_config_buffer_address(),
            ttnn.get_global_semaphore_address(self.termination_semaphore),
            self.h2d_page_size,
            self.h2d_socket.get_h2d_mode() == ttnn.H2DMode.DEVICE_PULL,
            self.loopback_mode,
            self.intermed_cb_index
            if self.loopback_mode
            else self.downstream_socket_pair[0].get_config_buffer_address(),
            self.fabric_packet_header_cb_index,
            fabric_max_payload_size,
            num_whole_fabric_packets_link_0,
            num_whole_fabric_packets_link_1,
            partial_packet_size,
        ]
        # Add CTAs for fused embedding op if needed
        if self.has_embedding:
            h2d_socket_kernel_ct_args.extend(
                [self.embedding_cb_index, self.embedding_page_size, self.embedding_tensor.buffer_address()]
            )
            h2d_socket_kernel_ct_args.extend(get_interleaved_tensor_accessor_args(self.embedding_tensor))

        kernel_source = (
            "models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/fused_h2d_receiver_embedding.cpp"
            if self.has_embedding
            else "models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/h2d_receiver.cpp"
        )

        h2d_kernel = ttnn.KernelDescriptor(
            kernel_source=kernel_source,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet(
                [ttnn.CoreRange(self.h2d_mesh_core_coord.core_coord, self.h2d_mesh_core_coord.core_coord)]
            ),
            compile_time_args=h2d_socket_kernel_ct_args,
            config=ttnn.WriterConfigDescriptor(),
        )
        return h2d_kernel

    def _create_d2h_kernel(self):
        d2h_socket_kernel_ct_args = [
            self.d2h_socket.get_config_buffer_address(),
            ttnn.get_global_semaphore_address(self.termination_semaphore),
            self.d2h_page_size,
            self.loopback_mode,
            # Use a local CB if doing loopback, otherwise communicate with downstream over sockets
            self.intermed_cb_index if self.loopback_mode else self.upstream_socket_pair[1].get_config_buffer_address(),
            self.fabric_packet_header_cb_index,
        ]
        d2h_kernel = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/d2h_sender.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet(
                [ttnn.CoreRange(self.d2h_mesh_core_coord.core_coord, self.d2h_mesh_core_coord.core_coord)]
            ),
            compile_time_args=d2h_socket_kernel_ct_args,
            config=ttnn.ReaderConfigDescriptor(),
        )
        return d2h_kernel

    def _create_cb_descriptors(self, mesh_core_coord):
        cb_descriptors = []
        if self.loopback_mode:
            intermed_cb_desc = ttnn.CBDescriptor(
                total_size=self.core_to_core_socket_buffer_size,
                core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(mesh_core_coord.core_coord, mesh_core_coord.core_coord)]),
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=self.intermed_cb_index,
                        # Setup CB data format for consistency. Value gets ignored in kernel.
                        data_format=self.embedding_tensor.dtype if self.embedding_tensor else ttnn.uint32,
                        page_size=self.d2h_page_size,
                    )
                ],
            )
            cb_descriptors.append(intermed_cb_desc)

        # CB for embedding DRAM reads
        if self.has_embedding:
            embedding_cb_desc = ttnn.CBDescriptor(
                total_size=self.embedding_page_size,
                core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(mesh_core_coord.core_coord, mesh_core_coord.core_coord)]),
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=self.embedding_cb_index,
                        data_format=ttnn.bfloat16,
                        page_size=self.embedding_page_size,
                    )
                ],
            )
            cb_descriptors.append(embedding_cb_desc)

        packet_header_cb_num_pages = self.num_fwd_links + self.num_bwd_links
        packet_header_cb_page_size = ttnn.get_tt_fabric_packet_header_size_bytes()

        packet_header_cb_desc = ttnn.CBDescriptor(
            total_size=packet_header_cb_num_pages * packet_header_cb_page_size,
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(mesh_core_coord.core_coord, mesh_core_coord.core_coord)]),
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=self.fabric_packet_header_cb_index,
                    data_format=ttnn.uint32,
                    page_size=packet_header_cb_page_size,
                )
            ],
        )
        cb_descriptors.append(packet_header_cb_desc)
        return cb_descriptors

    def run(self):
        dummy_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.h2d_socket.get_mesh_device()
        )

        h2d_fabric_node_id = self.mesh_device.get_fabric_node_id(self.h2d_mesh_core_coord.device_coord)
        d2h_fabric_node_id = self.mesh_device.get_fabric_node_id(self.d2h_mesh_core_coord.device_coord)

        my_downstream_fabric_node_id = self.mesh_device.get_fabric_node_id(self.h2d_downstream_core.device_coord)
        my_upstream_fabric_node_id = self.mesh_device.get_fabric_node_id(self.d2h_upstream_core.device_coord)

        h2d_kernel = self._create_h2d_kernel()
        d2h_kernel = self._create_d2h_kernel()
        h2d_cb_descriptors = self._create_cb_descriptors(self.h2d_mesh_core_coord)
        d2h_cb_descriptors = self._create_cb_descriptors(self.d2h_mesh_core_coord)

        h2d_program = ttnn.ProgramDescriptor(
            kernels=[h2d_kernel],
            semaphores=[],
            cbs=h2d_cb_descriptors,
        )
        h2d_program.kernels[0].runtime_args[self.h2d_mesh_core_coord.core_coord.x][
            self.h2d_mesh_core_coord.core_coord.y
        ] = []
        h2d_rt_args_ref = h2d_program.kernels[0].runtime_args[self.h2d_mesh_core_coord.core_coord.x][
            self.h2d_mesh_core_coord.core_coord.y
        ]

        for idx in range(self.num_fwd_links):
            fwd_fabric_args = ttnn.setup_fabric_connection(
                h2d_fabric_node_id,
                my_downstream_fabric_node_id,
                idx,
                h2d_program,
                self.h2d_mesh_core_coord.core_coord,
            )
            h2d_rt_args_ref.extend(fwd_fabric_args)

        d2h_program = ttnn.ProgramDescriptor(
            kernels=[d2h_kernel],
            semaphores=[],
            cbs=d2h_cb_descriptors,
        )

        d2h_program.kernels[0].runtime_args[self.d2h_mesh_core_coord.core_coord.x][
            self.d2h_mesh_core_coord.core_coord.y
        ] = []
        d2h_rt_args_ref = d2h_program.kernels[0].runtime_args[self.d2h_mesh_core_coord.core_coord.x][
            self.d2h_mesh_core_coord.core_coord.y
        ]

        for idx in range(self.num_bwd_links):
            bwd_fabric_args = ttnn.setup_fabric_connection(
                d2h_fabric_node_id,
                my_upstream_fabric_node_id,
                idx,
                d2h_program,
                self.d2h_mesh_core_coord.core_coord,
            )
            d2h_rt_args_ref.extend(bwd_fabric_args)

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        mesh_program_descriptor[
            ttnn.MeshCoordinateRange(self.h2d_mesh_core_coord.device_coord, self.h2d_mesh_core_coord.device_coord)
        ] = h2d_program

        mesh_program_descriptor[
            ttnn.MeshCoordinateRange(self.d2h_mesh_core_coord.device_coord, self.d2h_mesh_core_coord.device_coord)
        ] = d2h_program

        io_tensors = [
            dummy_tensor,
            dummy_tensor,
        ]

        return ttnn.generic_op(io_tensors, mesh_program_descriptor)

    def get_downstream_socket(self):
        return self.downstream_socket_pair[1]

    def get_upstream_socket(self):
        return self.upstream_socket_pair[0]

    def terminate(self):
        self.h2d_socket.barrier()
        self.d2h_socket.barrier()
        ttnn.reset_global_semaphore_value(self.termination_semaphore, 1)
        # ttnn.synchronize_device(self.mesh_device)
