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

        # Create semaphores
        noc0_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=0,
            core_ranges=all_cores,
            initial_value=0,
        )

        noc1_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=1,
            core_ranges=all_cores,
            initial_value=0,
        )

        kernels = []
        semaphores = []

        # Create CoreRangeSets for NOC0 and NOC1 cores
        noc0_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in noc0_cores])
        noc1_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in noc1_cores])

        # Create receiver kernel (only if we have cores on that NOC)
        if not noc0_core_range_set.empty() or not noc1_core_range_set.empty():
            # Determine receiver NOC - use the NOC that doesn't conflict with senders
            # If output core is in the NOC0 sender set, use NOC1 for receiver, and vice versa
            output_in_noc0 = noc0_core_range_set.contains(gather_core)
            if output_in_noc0:
                receiver_noc = ttnn.NOC.NOC_1
            else:
                receiver_noc = ttnn.NOC.NOC_0

            receiver_compile_args = [
                len(noc0_cores),
                len(noc1_cores),
                noc0_receiver_semaphore_descriptor.id,
                noc1_receiver_semaphore_descriptor.id,
            ]

            receiver_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source="models/demos/deepseek_v3_b1/micro_ops/gather/kernels/gather_receiver.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=output_core_grid,
                compile_time_args=receiver_compile_args,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,
                    noc=receiver_noc,
                ),
            )
            kernels.append(receiver_kernel_descriptor)
            semaphores.append(noc0_receiver_semaphore_descriptor)
            semaphores.append(noc1_receiver_semaphore_descriptor)

        # Create sender kernels
        sender_kernel_path = "models/demos/deepseek_v3_b1/micro_ops/gather/kernels/gather_sender.cpp"

        # Build runtime args as 2D list: runtime_args[core_x][core_y] = [args]
        sender_noc0_runtime_args = ttnn.RuntimeArgs()
        sender_noc1_runtime_args = ttnn.RuntimeArgs()

        # Set runtime arguments for sender kernels
        for i, core in enumerate(input_cores_list):
            sender_runtime_args = [
                input_tensor.buffer_address(),
                output_tensor.buffer_address(),
                i * total_size,
            ]
            if noc0_core_range_set.contains(core):
                sender_noc0_runtime_args[core.x][core.y] = sender_runtime_args
            else:
                sender_noc1_runtime_args[core.x][core.y] = sender_runtime_args

        sender_compile_args = [
            gather_dest_noc_core.x,
            gather_dest_noc_core.y,
            total_size,
            0,  # semaphore (will be set per NOC)
        ]
        if not noc0_core_range_set.empty():
            sender_compile_args[3] = noc0_receiver_semaphore_descriptor.id
            sender_noc0_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source=sender_kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=noc0_core_range_set,
                compile_time_args=list(sender_compile_args),
                runtime_args=sender_noc0_runtime_args,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1,
                    noc=ttnn.NOC.NOC_0,
                ),
            )
            kernels.append(sender_noc0_kernel_descriptor)

        if not noc1_core_range_set.empty():
            sender_compile_args[3] = noc1_receiver_semaphore_descriptor.id
            sender_noc1_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source=sender_kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=noc1_core_range_set,
                compile_time_args=list(sender_compile_args),
                runtime_args=sender_noc1_runtime_args,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1,
                    noc=ttnn.NOC.NOC_1,
                ),
            )
            kernels.append(sender_noc1_kernel_descriptor)

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=kernels,
            semaphores=semaphores,
        )

        # Execute generic op
        io_tensors = [input_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
