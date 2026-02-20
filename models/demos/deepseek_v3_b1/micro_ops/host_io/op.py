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
        h2d_downstream_core=ttnn.CoreCoord(0, 0),
        d2h_upstream_core=ttnn.CoreCoord(0, 0),
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
        # Validate single-core, single-chip constraint
        # Current implementation only supports host communication with one core on one chip
        if len(h2d_socket.get_active_cores()) != 1 or len(d2h_socket.get_active_cores()) != 1:
            raise ValueError("Host <-> Device Communication for Blitz Decode must be on a single core.")
        if h2d_socket.get_active_cores()[0] != d2h_socket.get_active_cores()[0]:
            raise ValueError("Expected Host <-> Device Communication for Blitz Decode to be on the same core.")
        if h2d_socket.get_mesh_device().id() != d2h_socket.get_mesh_device().id():
            raise ValueError("Expected Host <-> Device Communication for Blitz Decode to be on the same mesh device.")

        self.mesh_device = h2d_socket.get_mesh_device()
        self.mesh_core_coord = self.h2d_socket.get_active_cores()[0]
        self.termination_semaphore = ttnn.create_global_semaphore(
            self.mesh_device,
            ttnn.CoreRangeSet([ttnn.CoreRange(self.mesh_core_coord.core_coord, self.mesh_core_coord.core_coord)]),
            0,
            ttnn.BufferType.L1,
        )
        # Loopback mode is for testing purposes only - H2D receiver <-> D2H sender communication over CBs
        # For real workloads, downstream/upstream cores connect to H2D receiver and D2H sender via D2D sockets
        # NOTE: Downstream and upstream cores must be on the same device as the host I/O core (single-chip constraint)
        if not loopback_mode:
            downstream_core = ttnn.MeshCoreCoord(self.mesh_core_coord.device_coord, h2d_downstream_core)
            upstream_core = ttnn.MeshCoreCoord(self.mesh_core_coord.device_coord, d2h_upstream_core)
            downstream_socket_connection = ttnn.SocketConnection(
                self.mesh_core_coord,
                downstream_core,
            )
            upstream_socket_connection = ttnn.SocketConnection(
                upstream_core,
                self.mesh_core_coord,
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

    def _create_h2d_kernel(self):
        h2d_socket_kernel_ct_args = [
            self.h2d_socket.get_config_buffer_address(),
            ttnn.get_global_semaphore_address(self.termination_semaphore),
            self.h2d_page_size,
            self.h2d_socket.get_h2d_mode() == ttnn.H2DMode.DEVICE_PULL,
            self.loopback_mode,
            self.intermed_cb_index
            if self.loopback_mode
            else self.downstream_socket_pair[0].get_config_buffer_address(),
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
                [ttnn.CoreRange(self.mesh_core_coord.core_coord, self.mesh_core_coord.core_coord)]
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
        ]

        d2h_kernel = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/d2h_sender.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet(
                [ttnn.CoreRange(self.mesh_core_coord.core_coord, self.mesh_core_coord.core_coord)]
            ),
            compile_time_args=d2h_socket_kernel_ct_args,
            config=ttnn.ReaderConfigDescriptor(),
        )
        return d2h_kernel

    def _create_cb_descriptors(self):
        cb_descriptors = []
        if self.loopback_mode:
            intermed_cb_desc = ttnn.CBDescriptor(
                total_size=self.core_to_core_socket_buffer_size,
                core_ranges=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(self.mesh_core_coord.core_coord, self.mesh_core_coord.core_coord)]
                ),
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
                core_ranges=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(self.mesh_core_coord.core_coord, self.mesh_core_coord.core_coord)]
                ),
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=self.embedding_cb_index,
                        data_format=ttnn.bfloat16,
                        page_size=self.embedding_page_size,
                    )
                ],
            )
            cb_descriptors.append(embedding_cb_desc)
        return cb_descriptors

    def run(self):
        dummy_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.h2d_socket.get_mesh_device()
        )

        h2d_kernel = self._create_h2d_kernel()
        d2h_kernel = self._create_d2h_kernel()
        cb_descriptors = self._create_cb_descriptors()

        program = ttnn.ProgramDescriptor(
            kernels=[h2d_kernel, d2h_kernel],
            semaphores=[],
            cbs=cb_descriptors,
        )
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        mesh_program_descriptor[
            ttnn.MeshCoordinateRange(self.mesh_core_coord.device_coord, self.mesh_core_coord.device_coord)
        ] = program

        io_tensors = [
            dummy_tensor,
            dummy_tensor,
        ]

        return ttnn.generic_op(io_tensors, mesh_program_descriptor)

    def terminate(self):
        self.h2d_socket.barrier()
        self.d2h_socket.barrier()
        ttnn.reset_global_semaphore_value(self.termination_semaphore, 1)
        ttnn.synchronize_device(self.mesh_device)
