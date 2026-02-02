# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn


class GatherSingleCore:
    """
    Single-core gather implementation using ttnn.generic_op.

    This class implements gather from multiple cores to a single core.
    """

    @staticmethod
    def golden(input_tensor):
        """
        PyTorch reference implementation of gather for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) sharded across multiple cores

        Returns:
            Output tensor with all input shards concatenated on gather_core
        """
        # Simply return the input tensor
        return input_tensor

    @staticmethod
    def op(input_tensor, output_tensor, noc=None):
        """
        Execute gather operation using generic_op.

        Args:
            input_tensor: Input tensor (must be sharded across multiple cores)
            output_tensor: Pre-allocated output tensor (must be sharded on single core)
            noc: NOC to use for gather (ttnn.NOC.NOC_0 or ttnn.NOC.NOC_1). If None,
                 automatically optimizes NOC routing based on hop distance for each sender core.

        Returns:
            Output tensor with input data gathered from all cores to gather_core
        """
        # Get device
        device = input_tensor.device()

        # Get core grids from tensor memory configs
        input_memory_config = input_tensor.memory_config()
        output_memory_config = output_tensor.memory_config()
        input_core_grid = input_memory_config.shard_spec.grid
        output_core_grid = output_memory_config.shard_spec.grid

        # Gather core (first core from output grid)
        gather_core = output_core_grid.ranges()[0].start

        # Get NOC coordinates for gather destination
        gather_dest_noc_core = device.worker_core_from_logical_core(gather_core)

        # Calculate data size from shard shape and element size
        shard_spec = input_memory_config.shard_spec
        shard_shape = shard_spec.shape
        shard_height = shard_shape[0]
        shard_width = shard_shape[1]

        # Get element size in bytes based on dtype
        dtype = input_tensor.dtype
        total_elements = shard_height * shard_width

        # Calculate total size in bytes based on dtype
        if dtype == ttnn.bfloat16:
            element_size_bytes = 2
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        total_size = total_elements * element_size_bytes

        # All cores (input + output) for semaphore allocation
        all_cores = input_core_grid.merge(output_core_grid)

        # Split cores into NOC0 and NOC1 groups based on hop distance optimization
        noc0_cores = []
        noc1_cores = []

        # Convert CoreRangeSet to list of cores
        input_cores_list = ttnn.corerange_to_cores(input_core_grid, row_wise=True)

        # Assign cores to NOC based on noc parameter or hop distance optimization
        if noc is not None:
            # User specified NOC, use it for all cores
            if noc == ttnn.NOC.NOC_0:
                noc0_cores = input_cores_list
            else:
                noc1_cores = input_cores_list
        else:
            # Optimize NOC routing based on hop distance
            for core in input_cores_list:
                noc0_hop = device.get_worker_noc_hop_distance(core, gather_core, ttnn.NOC.NOC_0)
                noc1_hop = device.get_worker_noc_hop_distance(core, gather_core, ttnn.NOC.NOC_1)
                if noc0_hop <= noc1_hop:
                    noc0_cores.append(core)
                else:
                    noc1_cores.append(core)

        # Create CoreRangeSets for NOC0 and NOC1 cores
        noc0_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in noc0_cores])
        noc1_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in noc1_cores])

        # Get sender grid dimensions for computing per-core offset in kernel
        input_core_ranges = list(input_core_grid.ranges())
        sender_grid_range = input_core_ranges[0]
        sender_grid_start_x = sender_grid_range.start.x
        sender_grid_start_y = sender_grid_range.start.y
        sender_grid_end_x = sender_grid_range.end.x
        sender_grid_end_y = sender_grid_range.end.y

        # Semaphore IDs
        noc0_receiver_semaphore_id = 0
        noc1_receiver_semaphore_id = 1

        # Create semaphores
        noc0_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=noc0_receiver_semaphore_id,
            core_ranges=all_cores,
            initial_value=0,
        )

        noc1_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=noc1_receiver_semaphore_id,
            core_ranges=all_cores,
            initial_value=0,
        )

        # CB indices
        src_cb = 0  # Input CB (from sharded input tensor)
        dst_cb = 1  # Output CB (from sharded output tensor)

        # Number of pages
        num_senders = len(input_cores_list)
        src_num_pages = 1  # Each sender has one page
        dst_num_pages = num_senders  # Receiver gets one page per sender

        kernels = []
        semaphores = [noc0_receiver_semaphore_descriptor, noc1_receiver_semaphore_descriptor]

        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/gather/kernels/gather_kernel.cpp"

        # Get the output tensor's buffer address for receiver data address (runtime arg)
        # The dst CB doesn't exist on sender cores, so we pass the buffer address as runtime arg
        receiver_data_addr = output_tensor.buffer_address()

        # ========================================================================
        # Sender kernels (NCRISC) - separate for NOC0 and NOC1
        # ========================================================================
        def create_sender_named_compile_time_args(receiver_semaphore_id):
            return [
                ("gather_dest_noc_x", gather_dest_noc_core.x),
                ("gather_dest_noc_y", gather_dest_noc_core.y),
                ("gather_data_size_bytes", total_size),
                ("gather_receiver_semaphore_id", receiver_semaphore_id),
                ("gather_src_cb", src_cb),
                ("gather_src_num_pages", src_num_pages),
                ("gather_sender_grid_start_x", sender_grid_start_x),
                ("gather_sender_grid_start_y", sender_grid_start_y),
                ("gather_sender_grid_end_x", sender_grid_end_x),
                ("gather_sender_grid_end_y", sender_grid_end_y),
                ("gather_row_major", 1),  # 1 = row-major linearization
                # Role flags
                ("is_sender_core", 1),
                ("is_receiver_core", 0),
            ]

        # NOC0 sender kernel
        if not noc0_core_range_set.empty():
            noc0_sender_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=noc0_core_range_set,
                named_compile_time_args=create_sender_named_compile_time_args(noc0_receiver_semaphore_id),
                common_runtime_args=[receiver_data_addr],  # receiver_data_addr passed as runtime arg
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1,
                    noc=ttnn.NOC.NOC_0,
                ),
            )
            kernels.append(noc0_sender_kernel_descriptor)

        # NOC1 sender kernel
        if not noc1_core_range_set.empty():
            noc1_sender_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=noc1_core_range_set,
                named_compile_time_args=create_sender_named_compile_time_args(noc1_receiver_semaphore_id),
                common_runtime_args=[receiver_data_addr],  # receiver_data_addr passed as runtime arg
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1,
                    noc=ttnn.NOC.NOC_1,
                ),
            )
            kernels.append(noc1_sender_kernel_descriptor)

        # ========================================================================
        # Receiver kernel (BRISC)
        # ========================================================================
        # Determine receiver NOC - use the NOC that doesn't conflict with senders
        output_in_noc0 = noc0_core_range_set.contains(gather_core)
        if output_in_noc0:
            receiver_noc = ttnn.NOC.NOC_1
        else:
            receiver_noc = ttnn.NOC.NOC_0

        receiver_named_compile_time_args = [
            ("gather_noc0_num_senders", len(noc0_cores)),
            ("gather_noc1_num_senders", len(noc1_cores)),
            ("gather_noc0_receiver_semaphore_id", noc0_receiver_semaphore_id),
            ("gather_noc1_receiver_semaphore_id", noc1_receiver_semaphore_id),
            ("gather_dst_cb", dst_cb),
            ("gather_dst_num_pages", dst_num_pages),
            # Role flags
            ("is_sender_core", 0),
            ("is_receiver_core", 1),
        ]

        receiver_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source=kernel_path,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=output_core_grid,
            named_compile_time_args=receiver_named_compile_time_args,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=receiver_noc,
            ),
        )
        kernels.append(receiver_kernel_descriptor)

        # ========================================================================
        # Compute kernels (TRISC) - no-op for gather, but needed for all cores
        # ========================================================================
        # Sender cores compute kernel (no-op)
        if not input_core_grid.empty():
            sender_compute_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=input_core_grid,
                named_compile_time_args=[
                    ("is_sender_core", 1),
                    ("is_receiver_core", 0),
                ],
                config=ttnn.ComputeConfigDescriptor(),
            )
            kernels.append(sender_compute_kernel_descriptor)

        # Receiver core compute kernel (no-op) - only if not already covered by sender grid
        if not input_core_grid.contains(gather_core):
            receiver_compute_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=output_core_grid,
                named_compile_time_args=[
                    ("is_sender_core", 0),
                    ("is_receiver_core", 1),
                ],
                config=ttnn.ComputeConfigDescriptor(),
            )
            kernels.append(receiver_compute_kernel_descriptor)

        # Create CB descriptors from sharded tensors
        src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(src_cb, input_tensor)
        dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(dst_cb, output_tensor)

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=kernels,
            cbs=[src_cb_descriptor, dst_cb_descriptor],
            semaphores=semaphores,
        )

        # Execute generic op
        io_tensors = [input_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
