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


class HostInterface:
    def __init__(
        self,
        h2d_socket,
        d2h_socket,
        page_size,
        core_to_core_socket_buffer_size=1024,
        h2d_downstream_core=ttnn.CoreCoord(0, 0),
        d2h_upstream_core=ttnn.CoreCoord(0, 0),
        loopback_mode=False,
    ):
        self.h2d_socket = h2d_socket
        self.d2h_socket = d2h_socket
        self.page_size = page_size
        self.h2d_socket.set_page_size(self.page_size)
        self.d2h_socket.set_page_size(self.page_size)
        self.loopback_mode = loopback_mode
        self.core_to_core_socket_buffer_size = core_to_core_socket_buffer_size

        # Validate single-core, single-chip constraint
        # Current implementation only supports host communication with one core on one chip
        if len(h2d_socket.get_active_cores()) != 1 or len(d2h_socket.get_active_cores()) != 1:
            raise ValueError("Host <-> Device Communication for Blitz Decode must be on a single core.")
        if h2d_socket.get_active_cores()[0] != d2h_socket.get_active_cores()[0]:
            raise ValueError("Expected Host <-> Device Communication for Blitz Decode to be on the same core.")
        if h2d_socket.get_mesh_device() != d2h_socket.get_mesh_device():
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
            downstream_socket_connection = ttnn.SocketConfig(
                self.mesh_core_coord,
                ttnn.MeshCoreCoord(self.mesh_core_coord.device_coord, h2d_downstream_core),
            )
            upstream_socket_connection = ttnn.SocketConfig(
                ttnn.MeshCoreCoord(self.mesh_core_coord.device_coord, d2h_upstream_core),
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

    def run(self):
        dummy_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.h2d_socket.get_mesh_device()
        )

        intermed_cb_index = 0
        h2d_socket_kernel_ct_args = [
            self.h2d_socket.get_config_buffer_address(),
            ttnn.get_global_semaphore_address(self.termination_semaphore),
            self.page_size,
            self.h2d_socket.get_h2d_mode() == ttnn.H2DMode.DEVICE_PULL,
            self.loopback_mode,
            # Use a local CB if doing loopback, otherwise communicate with downstream over sockets
            intermed_cb_index if self.loopback_mode else self.downstream_socket_pair[0].get_config_buffer_address(),
        ]

        d2h_socket_kernel_ct_args = [
            self.d2h_socket.get_config_buffer_address(),
            ttnn.get_global_semaphore_address(self.termination_semaphore),
            self.page_size,
            self.loopback_mode,
            # Use a local CB if doing loopback, otherwise communicate with downstream over sockets
            intermed_cb_index if self.loopback_mode else self.upstream_socket_pair[0].get_config_buffer_address(),
        ]

        h2d_kernel = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/h2d_receiver.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet(
                [ttnn.CoreRange(self.mesh_core_coord.core_coord, self.mesh_core_coord.core_coord)]
            ),
            compile_time_args=h2d_socket_kernel_ct_args,
            config=ttnn.WriterConfigDescriptor(),
        )
        d2h_kernel = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/d2h_sender.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet(
                [ttnn.CoreRange(self.mesh_core_coord.core_coord, self.mesh_core_coord.core_coord)]
            ),
            compile_time_args=d2h_socket_kernel_ct_args,
            config=ttnn.ReaderConfigDescriptor(),
        )
        cb_descriptors = []
        if self.loopback_mode:
            intermed_cb_desc = ttnn.CBDescriptor(
                total_size=self.core_to_core_socket_buffer_size,
                core_ranges=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(self.mesh_core_coord.core_coord, self.mesh_core_coord.core_coord)]
                ),
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=intermed_cb_index,
                        data_format=ttnn.uint32,
                        page_size=self.page_size,
                    )
                ],
            )
            cb_descriptors.append(intermed_cb_desc)
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
