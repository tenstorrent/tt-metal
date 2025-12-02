# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn


def get_noc_hop_distance(device, logical_src, logical_dst, noc):
    """
    Calculate the NOC hop distance between two logical cores.

    This replicates the C++ get_worker_noc_hop_distance function.

    Args:
        device: The device to query
        logical_src: Source logical core coordinate
        logical_dst: Destination logical core coordinate
        noc: NOC index (0 or 1)

    Returns:
        Hop distance for the given NOC
    """
    # Convert logical to physical worker coordinates
    src = device.worker_core_from_logical_core(logical_src)
    dst = device.worker_core_from_logical_core(logical_dst)
    grid_size = device.compute_with_storage_grid_size()

    if noc == 0:
        # NOC0: Preferred +x -> +y direction
        dist_right = dst.x - src.x if src.x <= dst.x else grid_size.x - src.x + dst.x
        dist_bottom = dst.y - src.y if src.y <= dst.y else grid_size.y - src.y + dst.y
        return dist_right + dist_bottom
    else:
        # NOC1: Preferred -y -> -x direction
        dist_left = src.x - dst.x if src.x >= dst.x else grid_size.x - dst.x + src.x
        dist_top = src.y - dst.y if src.y >= dst.y else grid_size.y - dst.y + src.y
        return dist_left + dist_top


class GatherSingleCore:
    """
    Single-core gather implementation using ttnn.generic_op.

    This class implements gather from multiple cores to a single core.
    """

    @staticmethod
    def golden(input_tensor, gather_core):
        """
        PyTorch reference implementation of gather for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) sharded across multiple cores
            gather_core: Core coordinate where output will be gathered

        Returns:
            Output tensor with all input shards concatenated on gather_core
        """
        # Simply concatenate all shards along width dimension
        # This is a simplified reference - actual implementation depends on shard layout
        return input_tensor

    @staticmethod
    def op(input_tensor, output_tensor, gather_core, gather_grid, noc=None):
        """
        Execute gather operation using generic_op.

        Args:
            input_tensor: Input tensor (must be sharded across multiple cores)
            output_tensor: Pre-allocated output tensor (must be sharded on single core)
            gather_core: Core coordinate where output will be gathered
            gather_grid: Core range where input is sharded (multiple cores)
            noc: NOC to use for gather (0 or 1). If None, automatically optimizes
                 NOC routing based on hop distance for each sender core.

        Returns:
            Output tensor with input data gathered from all cores to gather_core
        """
        # Get device
        device = input_tensor.device()

        # Get NOC coordinates for gather destination
        gather_dest_noc_core = device.worker_core_from_logical_core(gather_core)

        # Calculate data size from shard shape and element size
        memory_config = input_tensor.memory_config()
        shard_spec = memory_config.shard_spec
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

        # Get input cores (multiple cores)
        # Convert gather_grid (CoreRange) to CoreRangeSet
        input_core_grid = ttnn.CoreRangeSet([gather_grid])
        num_input_cores = gather_grid.grid_size().x * gather_grid.grid_size().y

        # Get output core (single core)
        output_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

        # All cores (input + output) for semaphore allocation
        all_cores = input_core_grid.merge(output_core_grid)

        # Split cores into NOC0 and NOC1 groups based on hop distance optimization
        noc0_cores = []
        noc1_cores = []

        # Convert CoreRangeSet to list of cores
        # Iterate over ranges and generate cores
        input_cores_list = []
        # gather_grid could be CoreRange or CoreRangeSet
        if hasattr(gather_grid, "ranges"):
            # It's a CoreRangeSet
            for core_range in gather_grid.ranges():
                start = core_range.start
                end = core_range.end
                # Generate cores in row-major order (matching shard orientation)
                for y in range(start.y, end.y + 1):
                    for x in range(start.x, end.x + 1):
                        input_cores_list.append(ttnn.CoreCoord(x, y))
        else:
            # It's a CoreRange
            start = gather_grid.start
            end = gather_grid.end
            for y in range(start.y, end.y + 1):
                for x in range(start.x, end.x + 1):
                    input_cores_list.append(ttnn.CoreCoord(x, y))

        # Build a mapping from core to its original index (for offset calculation)
        core_to_index = {}
        for i, core in enumerate(input_cores_list):
            core_to_index[(core.x, core.y)] = i

        # Assign cores to NOC based on noc parameter or hop distance optimization
        if noc is not None:
            # User specified NOC, use it for all cores
            if noc == 0:
                noc0_cores = input_cores_list
            else:
                noc1_cores = input_cores_list
        else:
            # Optimize NOC routing based on hop distance
            for core in input_cores_list:
                noc0_hop = get_noc_hop_distance(device, core, gather_core, 0)
                noc1_hop = get_noc_hop_distance(device, core, gather_core, 1)
                if noc0_hop <= noc1_hop:
                    noc0_cores.append(core)
                else:
                    noc1_cores.append(core)

        # Create semaphores
        # Note: semaphore_ids 0 and 1 are used for NOC0 and NOC1 receivers respectively
        noc0_receiver_semaphore_id = 0
        noc1_receiver_semaphore_id = 1

        noc0_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            core_ranges=all_cores,
            initial_value=0,
        )

        noc1_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            core_ranges=all_cores,
            initial_value=0,
        )

        kernels = []
        semaphores = []

        # Create receiver kernel (only if we have cores on that NOC)
        if len(noc0_cores) > 0 or len(noc1_cores) > 0:
            # Determine receiver NOC - use the NOC that doesn't conflict with senders
            # If output core is in the NOC0 sender set, use NOC1 for receiver, and vice versa
            output_in_noc0 = any(c.x == gather_core.x and c.y == gather_core.y for c in noc0_cores)
            if output_in_noc0:
                receiver_noc = ttnn.NOC.NOC_1
            else:
                receiver_noc = ttnn.NOC.NOC_0

            receiver_compile_args = [
                len(noc0_cores),
                len(noc1_cores),
                noc0_receiver_semaphore_id,
                noc1_receiver_semaphore_id,
            ]

            receiver_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source="models/demos/deepseek_v3_b1/micro_ops/gather/kernels/gather_receiver.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=output_core_grid,
                compile_time_args=receiver_compile_args,
                runtime_args=[[[]]],  # Single core: [[[]]]
                common_runtime_args=[],
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,
                    noc=receiver_noc,
                ),
            )
            kernels.append(receiver_kernel_descriptor)
            semaphores.append(noc0_receiver_semaphore_descriptor)
            semaphores.append(noc1_receiver_semaphore_descriptor)

        # Create sender kernels for NOC0
        if len(noc0_cores) > 0:
            noc0_ranges = []
            for core in noc0_cores:
                noc0_ranges.append(ttnn.CoreRange(core, core))
            noc0_core_range_set = ttnn.CoreRangeSet(noc0_ranges)

            # Create sender kernel for NOC0 cores
            # Each sender needs its own runtime args with offset
            sender_noc0_compile_args = [
                gather_dest_noc_core.x,
                gather_dest_noc_core.y,
                total_size,
                noc0_receiver_semaphore_id,
            ]

            # Build runtime args as 2D list: runtime_args[core_x][core_y] = [args]
            # Find max x and y to size the 2D structure
            max_x = max(core.x for core in noc0_cores)
            max_y = max(core.y for core in noc0_cores)
            sender_noc0_runtime_args = [[[] for _ in range(max_y + 1)] for _ in range(max_x + 1)]

            # Set runtime arguments for each NOC0 sender core
            # Use the original index from input_cores_list for offset calculation
            for core in noc0_cores:
                original_index = core_to_index[(core.x, core.y)]
                offset = original_index * total_size
                sender_rt_args = [
                    input_tensor.buffer_address(),
                    output_tensor.buffer_address(),
                    offset,
                ]
                sender_noc0_runtime_args[core.x][core.y] = sender_rt_args

            sender_noc0_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source="models/demos/deepseek_v3_b1/micro_ops/gather/kernels/gather_sender.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=noc0_core_range_set,
                compile_time_args=sender_noc0_compile_args,
                runtime_args=sender_noc0_runtime_args,
                common_runtime_args=[],
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1,
                    noc=ttnn.NOC.NOC_0,
                ),
            )
            kernels.append(sender_noc0_kernel_descriptor)

        # Create sender kernels for NOC1
        if len(noc1_cores) > 0:
            noc1_ranges = []
            for core in noc1_cores:
                noc1_ranges.append(ttnn.CoreRange(core, core))
            noc1_core_range_set = ttnn.CoreRangeSet(noc1_ranges)

            sender_noc1_compile_args = [
                gather_dest_noc_core.x,
                gather_dest_noc_core.y,
                total_size,
                noc1_receiver_semaphore_id,
            ]

            # Build runtime args as 2D list: runtime_args[core_x][core_y] = [args]
            max_x = max(core.x for core in noc1_cores)
            max_y = max(core.y for core in noc1_cores)
            sender_noc1_runtime_args = [[[] for _ in range(max_y + 1)] for _ in range(max_x + 1)]

            # Set runtime arguments for each NOC1 sender core
            # Use the original index from input_cores_list for offset calculation
            for core in noc1_cores:
                original_index = core_to_index[(core.x, core.y)]
                offset = original_index * total_size
                sender_rt_args = [
                    input_tensor.buffer_address(),
                    output_tensor.buffer_address(),
                    offset,
                ]
                sender_noc1_runtime_args[core.x][core.y] = sender_rt_args

            sender_noc1_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source="models/demos/deepseek_v3_b1/micro_ops/gather/kernels/gather_sender.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=noc1_core_range_set,
                compile_time_args=sender_noc1_compile_args,
                runtime_args=sender_noc1_runtime_args,
                common_runtime_args=[],
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
            cbs=[],  # No circular buffers needed, using sharded tensors directly
        )

        # Execute generic op
        io_tensors = [input_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
