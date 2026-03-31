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


class HostInterface:
    def __init__(
        self,
        h2d_socket,
        d2h_socket,
        h2d_page_size,
        d2h_page_size,
        core_to_core_socket_buffer_size=1024,
        h2d_downstream_core=None,
        d2h_upstream_core=None,
        embedding_tensor=None,
        loopback_mode=False,
        embedding_cb_index=None,
        fabric_packet_header_cb_index=None,
        downstream_socket_page_size=None,
    ):
        assert h2d_socket is not None or d2h_socket is not None, "Either h2d_socket or d2h_socket must be provided"

        if h2d_socket and d2h_socket:
            assert (
                h2d_socket.get_mesh_device() == d2h_socket.get_mesh_device()
            ), "Expected Host <-> Device Communication for Blitz Decode to be on the same mesh device."

        self.h2d_socket = h2d_socket
        self.d2h_socket = d2h_socket
        self.h2d_page_size = h2d_page_size
        self.d2h_page_size = d2h_page_size
        if self.h2d_socket:
            self.h2d_socket.set_page_size(self.h2d_page_size)
        if self.d2h_socket:
            self.d2h_socket.set_page_size(self.d2h_page_size)
        self.loopback_mode = loopback_mode
        self.core_to_core_socket_buffer_size = core_to_core_socket_buffer_size
        self.embedding_tensor = embedding_tensor
        self.h2d_downstream_core = h2d_downstream_core
        self.d2h_upstream_core = d2h_upstream_core

        if self.h2d_socket:
            if len(self.h2d_socket.get_active_cores()) != 1:
                raise ValueError("Host <-> Device Communication for Blitz Decode must be on a single core.")
        if self.d2h_socket:
            if len(self.d2h_socket.get_active_cores()) != 1:
                raise ValueError("Host <-> Device Communication for Blitz Decode must be on a single core.")

        if loopback_mode:
            assert h2d_socket and d2h_socket, "Loopback mode requires both H2D and D2H sockets"
            if h2d_socket.get_active_cores()[0] != d2h_socket.get_active_cores()[0]:
                raise ValueError("Expected Host <-> Device Communication for Blitz Decode to be on the same core.")
        else:
            if self.h2d_socket:
                assert self.h2d_downstream_core is not None
            if self.d2h_socket:
                assert self.d2h_upstream_core is not None

        self.mesh_device = self.h2d_socket.get_mesh_device() if self.h2d_socket else self.d2h_socket.get_mesh_device()

        self.h2d_mesh_core_coord = self.h2d_socket.get_active_cores()[0] if self.h2d_socket else None
        self.d2h_mesh_core_coord = self.d2h_socket.get_active_cores()[0] if self.d2h_socket else None

        # Build termination semaphore core range from whichever sockets are present
        sem_core_ranges = []
        if self.h2d_mesh_core_coord is not None:
            sem_core_ranges.append(
                ttnn.CoreRange(self.h2d_mesh_core_coord.core_coord, self.h2d_mesh_core_coord.core_coord)
            )
        if self.d2h_mesh_core_coord is not None:
            d2h_range = ttnn.CoreRange(self.d2h_mesh_core_coord.core_coord, self.d2h_mesh_core_coord.core_coord)
            if not sem_core_ranges or self.d2h_mesh_core_coord.core_coord != self.h2d_mesh_core_coord.core_coord:
                sem_core_ranges.append(d2h_range)

        termination_semaphore_core_range = ttnn.CoreRangeSet(sem_core_ranges)
        self.termination_semaphore = ttnn.create_global_semaphore(
            self.mesh_device,
            termination_semaphore_core_range,
            0,
            ttnn.BufferType.L1,
        )

        self.downstream_socket_pair = None
        self.upstream_socket_pair = None

        if loopback_mode:
            self.intermed_cb_index = 0
        else:
            socket_memory_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, core_to_core_socket_buffer_size)

            if self.h2d_socket and self.h2d_downstream_core is not None:
                downstream_socket_connection = ttnn.SocketConnection(
                    self.h2d_mesh_core_coord,
                    self.h2d_downstream_core,
                )
                downstream_socket_config = ttnn.SocketConfig(
                    [downstream_socket_connection],
                    socket_memory_config,
                )
                self.downstream_socket_pair = ttnn.create_socket_pair(
                    self.mesh_device, self.mesh_device, downstream_socket_config
                )

            if self.d2h_socket and self.d2h_upstream_core is not None:
                upstream_socket_connection = ttnn.SocketConnection(
                    self.d2h_upstream_core,
                    self.d2h_mesh_core_coord,
                )
                upstream_socket_config = ttnn.SocketConfig(
                    [upstream_socket_connection],
                    socket_memory_config,
                )
                self.upstream_socket_pair = ttnn.create_socket_pair(
                    self.mesh_device, self.mesh_device, upstream_socket_config
                )

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
            # Tensor is DRAM interleaved, and row major. Page size is inner dim (2D: shape[1], 4D: shape[3]).
            self.embedding_page_size = self.embedding_tensor.shape[-1] * dtype_size(self.embedding_tensor.dtype)
            self.downstream_socket_page_size = downstream_socket_page_size if downstream_socket_page_size is not None else self.embedding_page_size
            self.embedding_cb_index = 2 if embedding_cb_index is None else embedding_cb_index

        self.fabric_packet_header_cb_index = (
            1 if fabric_packet_header_cb_index is None else fabric_packet_header_cb_index
        )
        self.num_fwd_links = 2
        self.num_bwd_links = 1

    def _create_h2d_kernel(self):
        use_fabric = (not self.loopback_mode) and (
            self.h2d_downstream_core.device_coord != self.h2d_mesh_core_coord.device_coord
        )

        fabric_max_payload_size = 0
        num_whole_fabric_packets_per_link = 0
        partial_packet_size_per_link = 0

        if use_fabric:
            fabric_max_payload_size = ttnn.get_tt_fabric_max_payload_size_bytes()
            if self.has_embedding:
                page_size_per_link = self.downstream_socket_page_size // self.num_fwd_links
            else:
                page_size_per_link = self.h2d_page_size // self.num_fwd_links
            num_whole_fabric_packets_per_link = page_size_per_link // fabric_max_payload_size
            partial_packet_size_per_link = page_size_per_link % fabric_max_payload_size

        # H2D Receiver Core will forward data to downstream core via fabric if:
        # 1. Not in loopback mode (i.e. real workload)
        # 2. Downstream core is not on the same device as the H2D receiver core
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
            num_whole_fabric_packets_per_link,
            partial_packet_size_per_link,
            use_fabric,
        ]
        # Add CTAs for fused embedding op if needed
        if self.has_embedding:
            h2d_socket_kernel_ct_args.extend(
                [self.embedding_cb_index, self.embedding_page_size, self.embedding_tensor.buffer_address(), self.downstream_socket_page_size]
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
        # D2H Sender Core will forward data to upstream core via fabric if:
        # 1. Not in loopback mode (i.e. real workload)
        # 2. Upstream core is not on the same device as the D2H sender core
        use_fabric = (not self.loopback_mode) and (
            self.d2h_upstream_core.device_coord != self.d2h_mesh_core_coord.device_coord
        )
        d2h_socket_kernel_ct_args = [
            self.d2h_socket.get_config_buffer_address(),
            ttnn.get_global_semaphore_address(self.termination_semaphore),
            self.d2h_page_size,
            self.loopback_mode,
            # Use a local CB if doing loopback, otherwise communicate with downstream over sockets
            self.intermed_cb_index if self.loopback_mode else self.upstream_socket_pair[1].get_config_buffer_address(),
            self.fabric_packet_header_cb_index,
            use_fabric,
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

    def _create_cb_descriptors(self, mesh_core_coord, use_fabric):
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

        if use_fabric:
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
            ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device
        )

        h2d_fabric_node_id = None
        d2h_fabric_node_id = None
        my_downstream_fabric_node_id = None
        my_upstream_fabric_node_id = None

        if self.h2d_mesh_core_coord is not None:
            h2d_fabric_node_id = self.mesh_device.get_fabric_node_id(self.h2d_mesh_core_coord.device_coord)
        if self.d2h_mesh_core_coord is not None:
            d2h_fabric_node_id = self.mesh_device.get_fabric_node_id(self.d2h_mesh_core_coord.device_coord)
        if self.h2d_downstream_core is not None:
            my_downstream_fabric_node_id = self.mesh_device.get_fabric_node_id(self.h2d_downstream_core.device_coord)
        if self.d2h_upstream_core is not None:
            my_upstream_fabric_node_id = self.mesh_device.get_fabric_node_id(self.d2h_upstream_core.device_coord)

        h2d_kernel = None
        h2d_cb_descriptors = None
        d2h_kernel = None
        d2h_cb_descriptors = None

        if self.h2d_socket:
            h2d_kernel = self._create_h2d_kernel()
            h2d_uses_fabric = (
                h2d_fabric_node_id is not None
                and my_downstream_fabric_node_id is not None
                and h2d_fabric_node_id != my_downstream_fabric_node_id
            )
            h2d_cb_descriptors = self._create_cb_descriptors(self.h2d_mesh_core_coord, h2d_uses_fabric)

        if self.d2h_socket:
            d2h_kernel = self._create_d2h_kernel()
            d2h_uses_fabric = (
                d2h_fabric_node_id is not None
                and my_upstream_fabric_node_id is not None
                and d2h_fabric_node_id != my_upstream_fabric_node_id
            )
            d2h_cb_descriptors = self._create_cb_descriptors(self.d2h_mesh_core_coord, d2h_uses_fabric)

        same_device = (
            self.h2d_socket
            and self.d2h_socket
            and self.h2d_mesh_core_coord.device_coord == self.d2h_mesh_core_coord.device_coord
        )

        h2d_program = None
        d2h_program = None

        if same_device:
            h2d_cb_ids = {fd.buffer_index for cb in h2d_cb_descriptors for fd in cb.format_descriptors}
            combined_cbs = h2d_cb_descriptors + [
                cb
                for cb in d2h_cb_descriptors
                if not any(fd.buffer_index in h2d_cb_ids for fd in cb.format_descriptors)
            ]
            h2d_program = ttnn.ProgramDescriptor(
                kernels=[h2d_kernel, d2h_kernel],
                semaphores=[],
                cbs=combined_cbs,
            )
            d2h_program = h2d_program
        else:
            if self.h2d_socket:
                h2d_program = ttnn.ProgramDescriptor(
                    kernels=[h2d_kernel],
                    semaphores=[],
                    cbs=h2d_cb_descriptors,
                )
            if self.d2h_socket:
                d2h_program = ttnn.ProgramDescriptor(
                    kernels=[d2h_kernel],
                    semaphores=[],
                    cbs=d2h_cb_descriptors,
                )

        if self.h2d_socket and h2d_program is not None:
            h2d_program.kernels[0].runtime_args[self.h2d_mesh_core_coord.core_coord.x][
                self.h2d_mesh_core_coord.core_coord.y
            ] = []
            h2d_rt_args_ref = h2d_program.kernels[0].runtime_args[self.h2d_mesh_core_coord.core_coord.x][
                self.h2d_mesh_core_coord.core_coord.y
            ]
            if h2d_uses_fabric:
                for idx in range(self.num_fwd_links):
                    fwd_fabric_args = ttnn.setup_fabric_connection(
                        h2d_fabric_node_id,
                        my_downstream_fabric_node_id,
                        idx,
                        h2d_program,
                        self.h2d_mesh_core_coord.core_coord,
                    )
                    h2d_rt_args_ref.extend(fwd_fabric_args)

        if self.d2h_socket and d2h_program is not None:
            d2h_kernel_idx = 1 if same_device else 0
            d2h_program.kernels[d2h_kernel_idx].runtime_args[self.d2h_mesh_core_coord.core_coord.x][
                self.d2h_mesh_core_coord.core_coord.y
            ] = []
            d2h_rt_args_ref = d2h_program.kernels[d2h_kernel_idx].runtime_args[self.d2h_mesh_core_coord.core_coord.x][
                self.d2h_mesh_core_coord.core_coord.y
            ]

            if d2h_uses_fabric:
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
        if self.h2d_socket and h2d_program is not None:
            mesh_program_descriptor[
                ttnn.MeshCoordinateRange(self.h2d_mesh_core_coord.device_coord, self.h2d_mesh_core_coord.device_coord)
            ] = h2d_program

        if self.d2h_socket and d2h_program is not None and not same_device:
            mesh_program_descriptor[
                ttnn.MeshCoordinateRange(self.d2h_mesh_core_coord.device_coord, self.d2h_mesh_core_coord.device_coord)
            ] = d2h_program

        io_tensors = [
            dummy_tensor,
            dummy_tensor,
        ]

        return ttnn.generic_op(io_tensors, mesh_program_descriptor)

    def get_downstream_socket(self):
        if self.downstream_socket_pair is not None:
            return self.downstream_socket_pair[1]
        else:
            raise ValueError("Downstream socket not available")

    def get_upstream_socket(self):
        if self.upstream_socket_pair is not None:
            return self.upstream_socket_pair[0]
        else:
            raise ValueError("Upstream socket not available")

    def terminate(self, sync_devices):
        if self.h2d_socket:
            self.h2d_socket.barrier()
        if self.d2h_socket:
            self.d2h_socket.barrier()

        ttnn.reset_global_semaphore_value(self.termination_semaphore, 1)
        if sync_devices:
            ttnn.synchronize_device(self.mesh_device)
