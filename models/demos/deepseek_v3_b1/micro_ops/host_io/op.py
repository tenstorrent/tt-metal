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
        num_h2d_cores=1,  # Number of cores for H2D senders (supports multi-core)
        h2d_cores=None,  # Optional: specify exact cores for H2D senders
        input_tensor=None,  # Optional: sharded tensor to send from H2D cores
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
        self.input_tensor = input_tensor
        self.num_h2d_cores = num_h2d_cores
        self.multi_core_mode = num_h2d_cores > 1

        # Validate input tensor for multi-core mode
        if self.multi_core_mode and self.input_tensor is not None:
            if not self.input_tensor.is_sharded():
                raise ValueError("Input tensor must be sharded for multi-core mode")
            # Verify tensor is sharded across the expected cores
            shard_spec = self.input_tensor.memory_config().shard_spec
            if shard_spec is None:
                raise ValueError("Input tensor must have shard_spec for multi-core mode")

        # Validate single-core, single-chip constraint for single-core mode
        # Multi-core mode allows multiple H2D sender cores but single D2H receiver core
        if not self.multi_core_mode:
            if len(h2d_socket.get_active_cores()) != 1 or len(d2h_socket.get_active_cores()) != 1:
                raise ValueError("Host <-> Device Communication for Blitz Decode must be on a single core.")
            if h2d_socket.get_active_cores()[0] != d2h_socket.get_active_cores()[0]:
                raise ValueError("Expected Host <-> Device Communication for Blitz Decode to be on the same core.")
        else:
            # For multi-core mode, D2H must still be on a single core
            if len(d2h_socket.get_active_cores()) != 1:
                raise ValueError("D2H socket must be on a single core in multi-core mode.")

        if h2d_socket.get_mesh_device().id() != d2h_socket.get_mesh_device().id():
            raise ValueError("Expected Host <-> Device Communication for Blitz Decode to be on the same mesh device.")

        self.mesh_device = h2d_socket.get_mesh_device()

        if self.multi_core_mode:
            # In multi-core mode, D2H is on a separate core
            self.d2h_mesh_core_coord = self.d2h_socket.get_active_cores()[0]
            self.h2d_mesh_core_coords = h2d_cores if h2d_cores else self._get_default_h2d_cores()
            if len(self.h2d_mesh_core_coords) != num_h2d_cores:
                raise ValueError(f"Expected {num_h2d_cores} H2D cores, got {len(self.h2d_mesh_core_coords)}")
        else:
            # Single-core mode
            self.mesh_core_coord = self.h2d_socket.get_active_cores()[0]
            self.h2d_mesh_core_coords = [self.mesh_core_coord]
            self.d2h_mesh_core_coord = self.mesh_core_coord

        # Create termination semaphore on D2H core and all H2D cores
        if self.multi_core_mode:
            all_cores = [self.d2h_mesh_core_coord] + self.h2d_mesh_core_coords
            core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c.core_coord, c.core_coord) for c in all_cores])
        else:
            core_range_set = ttnn.CoreRangeSet(
                [ttnn.CoreRange(self.mesh_core_coord.core_coord, self.mesh_core_coord.core_coord)]
            )

        self.termination_semaphore = ttnn.create_global_semaphore(
            self.mesh_device,
            core_range_set,
            0,
            ttnn.BufferType.L1,
        )

        # Loopback mode is for testing purposes only
        # Multi-core mode requires socket connections from H2D cores to D2H core
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

                downstream_socket_config = ttnn.SocketConfig([downstream_socket_connection], socket_memory_config)
                upstream_socket_config = ttnn.SocketConfig([upstream_socket_connection], socket_memory_config)

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
            if self.multi_core_mode:
                raise ValueError("Embedding mode not supported with multi-core H2D")
            # For now, we assume that tokens will be passed in as 64 bytes packets to embedding.
            # This allows us to add more information in the input packet as needed.
            assert self.h2d_page_size == 64
            assert (
                self.embedding_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
                and self.embedding_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
                and self.embedding_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
            ), f"Expected embedding tensor to be DRAM interleaved with page size {self.embedding_page_size} bytes for shape {self.embedding_tensor.shape}"
            self.embedding_page_size = self.embedding_tensor.shape[3] * dtype_size(self.embedding_tensor.dtype)
            self.embedding_cb_index = 2

    def _get_default_h2d_cores(self):
        """Get default H2D core coordinates in a 2x4 grid starting from (0, 0)"""
        device_coord = self.d2h_mesh_core_coord.device_coord
        cores = []
        for row in range(2):
            for col in range(4):
                cores.append(ttnn.MeshCoreCoord(device_coord, ttnn.CoreCoord(col, row)))
        return cores

    def _create_tensor_writer_kernels(self):
        """Create kernels to populate CBs from sharded input tensor.

        For sharded tensors with globally allocated CBs, the data is already in L1,
        but we need to manually reserve_back and push_back to make it visible to the CB interface.
        """
        if not self.input_tensor:
            return []

        writer_kernels = []
        # Calculate pages per core based on sharding
        shard_spec = self.input_tensor.memory_config().shard_spec
        shard_shape = shard_spec.shape
        element_size = 4  # uint32
        pages_per_core = (shard_shape[0] * shard_shape[1] * element_size) // self.h2d_page_size

        for h2d_core in self.h2d_mesh_core_coords:
            writer_ct_args = [
                0,  # CB index
                pages_per_core,
            ]

            writer_kernel = ttnn.KernelDescriptor(
                kernel_source="models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/cb_populate_from_sharded_tensor.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(h2d_core.core_coord, h2d_core.core_coord)]),
                compile_time_args=writer_ct_args,
                config=ttnn.ReaderConfigDescriptor(),
            )
            writer_kernels.append(writer_kernel)

        return writer_kernels

    def _create_h2d_kernel(self):
        if self.multi_core_mode:
            # Multi-core mode: create kernel descriptors for each H2D sender core
            h2d_kernels = []
            for core_idx, h2d_core in enumerate(self.h2d_mesh_core_coords):
                # Each H2D core reads from its CB and sends to D2H via socket
                h2d_socket_kernel_ct_args = [
                    ttnn.get_global_semaphore_address(self.termination_semaphore),
                    self.h2d_page_size,
                    0,  # CB index for input data (will be configured per core)
                    self.h2d_to_d2h_socket_pairs[core_idx][0].get_config_buffer_address(),  # Sender socket config
                ]

                h2d_kernel = ttnn.KernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/h2d_multicore_sender.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(h2d_core.core_coord, h2d_core.core_coord)]),
                    compile_time_args=h2d_socket_kernel_ct_args,
                    config=ttnn.WriterConfigDescriptor(),
                )
                h2d_kernels.append(h2d_kernel)
            return h2d_kernels
        else:
            # Original single-core mode
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
            return [h2d_kernel]

    def _create_d2h_kernel(self):
        d2h_socket_kernel_ct_args = [
            self.d2h_socket.get_config_buffer_address(),
            ttnn.get_global_semaphore_address(self.termination_semaphore),
            self.d2h_page_size,
            self.loopback_mode,
            # Use a local CB if doing loopback, otherwise communicate with downstream over sockets
            self.intermed_cb_index if self.loopback_mode else self.upstream_socket_pair[1].get_config_buffer_address(),
        ]

            # Pad with zeros if fewer than 8 sockets
            while len(d2h_socket_kernel_ct_args) < 12:  # 4 base args + 8 socket addresses
                d2h_socket_kernel_ct_args.append(0)

            d2h_kernel = ttnn.KernelDescriptor(
                kernel_source="models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/d2h_multicore_receiver.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(self.d2h_mesh_core_coord.core_coord, self.d2h_mesh_core_coord.core_coord)]
                ),
                compile_time_args=d2h_socket_kernel_ct_args,
                config=ttnn.ReaderConfigDescriptor(),
            )
        else:
            # Original single-core mode
            d2h_socket_kernel_ct_args = [
                self.d2h_socket.get_config_buffer_address(),
                ttnn.get_global_semaphore_address(self.termination_semaphore),
                self.d2h_page_size,
                self.loopback_mode,
                # Use a local CB if doing loopback, otherwise communicate with downstream over sockets
                self.intermed_cb_index
                if self.loopback_mode
                else self.upstream_socket_pair[0].get_config_buffer_address(),
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

        if self.multi_core_mode:
            # Multi-core mode: create CBs on each H2D core for input data
            if self.input_tensor:
                # For sharded tensors, use cb_descriptor_from_sharded_tensor
                # This automatically sets up globally allocated CBs pointing to the sharded tensor data
                input_cb_desc = ttnn.cb_descriptor_from_sharded_tensor(0, self.input_tensor)
                cb_descriptors.append(input_cb_desc)
            else:
                # No input tensor - create normal CBs on each H2D core
                for h2d_core in self.h2d_mesh_core_coords:
                    input_cb_desc = ttnn.CBDescriptor(
                        total_size=self.core_to_core_socket_buffer_size,
                        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(h2d_core.core_coord, h2d_core.core_coord)]),
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=0,  # Input CB index
                                data_format=ttnn.uint32,
                                page_size=self.h2d_page_size,
                            )
                        ],
                    )
                    cb_descriptors.append(input_cb_desc)
        elif self.loopback_mode:
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

        writer_kernels = self._create_tensor_writer_kernels()  # Returns list of writer kernels (or empty)
        h2d_kernels = self._create_h2d_kernel()  # Returns list of kernels
        d2h_kernel = self._create_d2h_kernel()
        cb_descriptors = self._create_cb_descriptors()

        # Combine all kernels: writers first, then h2d senders, then d2h receiver
        all_kernels = writer_kernels + h2d_kernels + [d2h_kernel]

        program = ttnn.ProgramDescriptor(
            kernels=all_kernels,
            semaphores=[],
            cbs=cb_descriptors,
        )
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        if self.multi_core_mode:
            # Multi-core mode: program runs on D2H core device
            mesh_program_descriptor[
                ttnn.MeshCoordinateRange(self.d2h_mesh_core_coord.device_coord, self.d2h_mesh_core_coord.device_coord)
            ] = program
        else:
            # Single-core mode
            mesh_program_descriptor[
                ttnn.MeshCoordinateRange(self.mesh_core_coord.device_coord, self.mesh_core_coord.device_coord)
            ] = program

        io_tensors = [
            dummy_tensor,
            dummy_tensor,
        ]

        # If we have an input tensor, include it in the io_tensors
        if self.input_tensor:
            io_tensors.append(self.input_tensor)

        return ttnn.generic_op(io_tensors, mesh_program_descriptor)

    def terminate(self):
        self.h2d_socket.barrier()
        self.d2h_socket.barrier()
        ttnn.reset_global_semaphore_value(self.termination_semaphore, 1)
        ttnn.synchronize_device(self.mesh_device)
