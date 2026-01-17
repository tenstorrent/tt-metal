# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class McastSingleCore:
    """
    Single-core multicast implementation using ttnn.generic_op.

    This class implements multicast from a single core to multiple cores.
    Uses the unified kernel pattern with a single kernel file that compiles
    for NCRISC (receiver), BRISC (sender), and TRISC (no-op).
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

        # All cores (input + output) for semaphore allocation and kernel execution
        all_cores = output_core_grid.merge(input_core_grid)

        # Calculate number of output cores
        num_output_cores = mcast_grid.grid_size().x * mcast_grid.grid_size().y

        # Semaphore IDs for mcast synchronization
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1

        # Create semaphores
        sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=mcast_data_sender_semaphore_id,
            core_ranges=all_cores,
            initial_value=0,
        )

        receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=mcast_data_receiver_semaphore_id,
            core_ranges=all_cores,
            initial_value=0,
        )

        # CB indices (using input/output tensors as sharded buffers)
        src_cb = 0
        dst_cb = 1

        # Calculate page counts based on shard shape
        # For sharded tensors, num_pages typically = shard_height (one page per row)
        src_num_pages = shard_height
        dst_num_pages = shard_height

        # ========================================================================
        # Named compile-time args for each RISC processor
        # ========================================================================

        # NCRISC (Receiver) named compile-time args
        mcast_receiver_named_compile_time_args = [
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_dst_cb", dst_cb),
            ("mcast_dst_num_pages", dst_num_pages),
            # Sender core also needs src_cb info for NCRISC to setup sharded buffer
            ("mcast_src_cb", src_cb),
            ("mcast_src_num_pages", src_num_pages),
        ]

        # BRISC (Sender) named compile-time args
        mcast_sender_named_compile_time_args = [
            ("mcast_dest_noc_start_x", mcast_dest_noc_start_core.x),
            ("mcast_dest_noc_start_y", mcast_dest_noc_start_core.y),
            ("mcast_dest_noc_end_x", mcast_dest_noc_end_core.x),
            ("mcast_dest_noc_end_y", mcast_dest_noc_end_core.y),
            ("mcast_num_cores", num_output_cores),
            ("mcast_data_sender_semaphore", mcast_data_sender_semaphore_id),
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_data_size_bytes", total_size),
            ("mcast_src_cb", src_cb),
            ("mcast_src_num_pages", src_num_pages),
        ]

        # Get the output tensor's buffer address for mcast destination (runtime arg)
        # This is the L1 address where receivers will store the data
        # The dst cb may not exist on the sender core if it's not a receiver core, so we can't directly get it from the cb write ptr
        mcast_receiver_data_addr = output_tensor.buffer_address()

        # TRISC has no mcast-specific compile-time args (no-op)
        mcast_trisc_named_compile_time_args = []

        # ========================================================================
        # Circular buffer descriptors
        # ========================================================================

        # CB 0: Source (input tensor, on sender core)
        src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(src_cb, input_tensor)

        # CB 1: Destination (output tensor, on receiver cores)
        dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(dst_cb, output_tensor)

        # ========================================================================
        # Unified kernel descriptor
        # ========================================================================
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/mcast/kernels/mcast_kernel.cpp",
            core_ranges=all_cores,
            # NCRISC named compile-time args: mcast receiver
            ncrisc_named_compile_time_args=mcast_receiver_named_compile_time_args,
            # BRISC named compile-time args: mcast sender
            brisc_named_compile_time_args=mcast_sender_named_compile_time_args,
            # TRISC named compile-time args: empty (no-op)
            trisc_named_compile_time_args=mcast_trisc_named_compile_time_args,
            # BRISC runtime args: mcast_receiver_data_addr (output tensor buffer address)
            brisc_common_runtime_args=[mcast_receiver_data_addr],
            # Per-core compile-time role differentiation
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_sender_core",
                    core_range=mcast_core,  # Sender core is the input core
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_receiver_core",
                    core_range=output_core_grid,  # Receiver cores are the output grid
                    value=1,
                    other_value=0,
                ),
            ],
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors(),
            cbs=[src_cb_descriptor, dst_cb_descriptor],
            semaphores=[sender_semaphore_descriptor, receiver_semaphore_descriptor],
        )

        # Execute generic op
        io_tensors = [input_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
