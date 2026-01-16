# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn


class McastSingleCore:
    """
    Single-core multicast implementation using ttnn.generic_op.

    This class implements multicast from a single core to multiple cores.
    """

    @staticmethod
    def golden(input_tensor, num_output_cores):
        """
        PyTorch reference implementation of mcast for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) on single core
            num_output_cores: Number of cores to multicast to

        Returns:
            Output tensor with input replicated across multiple cores
        """
        # Simply replicate the input across all output cores
        return input_tensor.repeat(num_output_cores, 1)

    @staticmethod
    def op(input_tensor, output_tensor, noc=ttnn.NOC.NOC_1):
        """
        Execute mcast operation using generic_op.

        Args:
            input_tensor: Input tensor (must be sharded on single core)
            output_tensor: Pre-allocated output tensor (must be sharded across multiple cores)
            noc: NOC to use for multicast (ttnn.NOC.NOC_0 or ttnn.NOC.NOC_1)

        Returns:
            Output tensor with input data multicast to all cores
        """
        # Get device
        device = input_tensor.device()

        # Get core grids from tensor memory configs
        input_memory_config = input_tensor.memory_config()
        output_memory_config = output_tensor.memory_config()
        input_core_grid = input_memory_config.shard_spec.grid
        output_core_grid = output_memory_config.shard_spec.grid

        # Extract mcast_core (first core from input grid) and mcast_grid (first range from output grid)
        input_core_ranges = list(input_core_grid.ranges())
        output_core_ranges = list(output_core_grid.ranges())
        mcast_core = input_core_ranges[0].start
        mcast_grid = output_core_ranges[0]

        # Get NOC coordinates for mcast destination
        mcast_dest_noc_start_core = device.worker_core_from_logical_core(mcast_grid.start)
        mcast_dest_noc_end_core = device.worker_core_from_logical_core(mcast_grid.end)

        # Calculate total data size from shard shape and element size
        shard_shape = input_memory_config.shard_spec.shape
        shard_height = shard_shape[0]
        shard_width = shard_shape[1]

        # Get element size in bytes based on dtype
        dtype = input_tensor.dtype
        total_elements = shard_height * shard_width

        # Calculate total size in bytes based on dtype
        if dtype == ttnn.bfloat16:
            total_size = total_elements * 2
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        # Determine if sender is part of receiver grid
        is_part_of_receiver_grid = output_core_grid.contains(mcast_core)
        loopback = is_part_of_receiver_grid

        # All cores (input + output) for semaphore allocation
        all_cores = output_core_grid.merge(input_core_grid)

        # Calculate number of output cores
        num_output_cores = mcast_grid.grid_size().x * mcast_grid.grid_size().y

        # Create semaphores
        sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=0,
            core_ranges=all_cores,
            initial_value=0,
        )

        receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=1,
            core_ranges=all_cores,
            initial_value=0,
        )

        # Sender kernel
        sender_named_compile_args = [
            # MCAST: DEFINE_PERSISTENT_MCAST_SENDER_VARS
            ("mcast_dest_noc_start_x", mcast_dest_noc_start_core.x),
            ("mcast_dest_noc_start_y", mcast_dest_noc_start_core.y),
            ("mcast_dest_noc_end_x", mcast_dest_noc_end_core.x),
            ("mcast_dest_noc_end_y", mcast_dest_noc_end_core.y),
            ("mcast_num_cores", num_output_cores),
            ("mcast_loopback", 1 if loopback else 0),
            ("mcast_is_part_of_receiver_grid", 1 if is_part_of_receiver_grid else 0),
            ("mcast_data_sender_semaphore", sender_semaphore_descriptor.id),
            ("mcast_data_receiver_semaphore", receiver_semaphore_descriptor.id),
            # MCAST0: DEFINE_MCAST_SENDER_VARS
            ("mcast0_num_cores", num_output_cores),
            ("mcast0_data_size_bytes", total_size),
        ]

        # Runtime args: [input_data_addr, mcast_receiver_data_addr]
        sender_rt_args = [
            input_tensor.buffer_address(),
            output_tensor.buffer_address(),
        ]

        sender_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/mcast/kernels/mcast_sender.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=input_core_grid,
            named_compile_time_args=sender_named_compile_args,
            common_runtime_args=sender_rt_args,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=noc,
            ),
        )

        # Receiver kernel - use opposite NOC
        receiver_noc = ttnn.NOC.NOC_0 if noc == ttnn.NOC.NOC_1 else ttnn.NOC.NOC_1
        receiver_compile_time_args = [receiver_semaphore_descriptor.id]

        receiver_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/mcast/kernels/mcast_receiver.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=output_core_grid,
            compile_time_args=receiver_compile_time_args,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1,
                noc=receiver_noc,
            ),
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=[sender_kernel_descriptor, receiver_kernel_descriptor],
            semaphores=[sender_semaphore_descriptor, receiver_semaphore_descriptor],
        )

        # Execute generic op
        io_tensors = [input_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
