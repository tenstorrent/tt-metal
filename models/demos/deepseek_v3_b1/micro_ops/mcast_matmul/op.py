# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Multi-core matmul with mcast input distribution.

This implements matmul operations distributed across multiple cores where:
- Input activations: [1, K] on single sender core
- Weights: [K, N] WIDTH_SHARDED across M cores (each core has [K, N/M])
- Output: [1, N] WIDTH_SHARDED across same M cores

Available ops:
  - McastMatmulMultiCore: Standard matmul (output = input @ weights)
  - McastMatmulSiLUMultiCore: Fused matmul+SiLU (output = SiLU(input @ weights))

Data flow:
  1. Sender core mcasts input activations to all matmul cores
  2. Each matmul core computes: output_shard = f(input @ weight_shard)
  3. Outputs remain sharded across cores
"""

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class McastMatmulMultiCore:
    """
    Multi-core matmul with mcast input distribution using ttnn.generic_op.

    Architecture:
      - Sender core: Has input activations, mcasts to all matmul cores
      - Matmul cores: Receive input, multiply with local weight shard, produce output shard
    """

    @staticmethod
    def golden(input_a, input_b):
        """
        PyTorch reference implementation of matmul for validation.

        Args:
            input_a: Input tensor A (torch.Tensor) [M, K]
            input_b: Input tensor B (torch.Tensor) [K, N]

        Returns:
            Output tensor [M, N]
        """
        return input_a @ input_b

    @staticmethod
    def op(
        input_tensor,
        weights_tensor,
        output_tensor,
        fp32_dest_acc_en=False,
    ):
        """
        Execute multi-core matmul with mcast input distribution using generic_op.

        Args:
            input_tensor: Input activations [1, K] HEIGHT_SHARDED on single core
            weights_tensor: Weight matrix [K, N] WIDTH_SHARDED across M cores
            output_tensor: Pre-allocated output [1, N] WIDTH_SHARDED across M cores
            fp32_dest_acc_en: Whether to enable FP32 accumulation

        Returns:
            Output tensor with matmul result
        """
        # Get tensor properties
        device = input_tensor.device()
        data_format = input_tensor.dtype

        # Get tiles
        in0_tile = input_tensor.get_tile()
        in1_tile = weights_tensor.get_tile()
        out_tile = output_tensor.get_tile()

        # Get memory configs and core grids
        input_memory_config = input_tensor.memory_config()
        weights_memory_config = weights_tensor.memory_config()
        output_memory_config = output_tensor.memory_config()

        input_core_grid = input_memory_config.shard_spec.grid
        weights_core_grid = weights_memory_config.shard_spec.grid
        output_core_grid = output_memory_config.shard_spec.grid

        # Extract sender core (first core from input grid)
        input_core_ranges = list(input_core_grid.ranges())
        sender_core = input_core_ranges[0].start
        sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])

        # Matmul cores = weights core grid (same as output core grid)
        matmul_core_grid = weights_core_grid

        # Validate shapes
        input_shard_shape = input_memory_config.shard_spec.shape
        weights_shape = weights_tensor.shape
        weights_shard_shape = weights_memory_config.shard_spec.shape
        output_shard_shape = output_memory_config.shard_spec.shape

        # Input: [1, K] -> shard is [1_tile_height, K]
        assert (
            input_shard_shape[0] // in0_tile.tile_shape[0] == 1
        ), f"Input M ({input_shard_shape[0]}) must be a single tile"
        assert (
            input_shard_shape[1] % in0_tile.tile_shape[1] == 0
        ), f"Input K ({input_shard_shape[1]}) must be divisible by tile width"
        assert (
            input_shard_shape[1] == weights_shape[0]
        ), f"Input K ({input_shard_shape[1]}) must equal weights K ({weights_shape[0]})"

        k_num_tiles = input_shard_shape[1] // in0_tile.tile_shape[1]
        out_w_per_core = output_shard_shape[1] // out_tile.tile_shape[1]

        # Calculate number of matmul cores
        matmul_num_cores = matmul_core_grid.num_cores()

        # Get mcast grid (use the bounding box of ALL matmul core ranges)
        # This is important for non-contiguous CoreRangeSets (e.g., 2-range grids)
        matmul_ranges = list(matmul_core_grid.ranges())

        # Compute bounding box across all ranges
        min_x = min(r.start.x for r in matmul_ranges)
        min_y = min(r.start.y for r in matmul_ranges)
        max_x = max(r.end.x for r in matmul_ranges)
        max_y = max(r.end.y for r in matmul_ranges)

        mcast_grid = ttnn.CoreRange(ttnn.CoreCoord(min_x, min_y), ttnn.CoreCoord(max_x, max_y))

        # Check if sender is part of mcast grid
        is_sender_in_mcast_grid = mcast_grid.contains(sender_core)

        # Get NOC coordinates for mcast destination
        mcast_dest_noc_start = device.worker_core_from_logical_core(mcast_grid.start)
        mcast_dest_noc_end = device.worker_core_from_logical_core(mcast_grid.end)

        # Calculate mcast num cores (bounding box size)
        # For non-contiguous grids, this may include "phantom" cores not in the actual grid
        mcast_num_cores = mcast_grid.grid_size().x * mcast_grid.grid_size().y

        # Build bounding box as CoreRangeSet for phantom core detection
        mcast_grid_set = ttnn.CoreRangeSet([mcast_grid])

        # Compute phantom cores: cores in bounding box but not in matmul grid or sender
        # These cores need to run minimal kernel code to ack the mcast semaphore
        phantom_cores = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                core = ttnn.CoreCoord(x, y)
                core_range = ttnn.CoreRange(core, core)
                # Check if this core is NOT in matmul grid and NOT the sender
                # Note: ttnn.CoreCoord doesn't implement equality correctly, so compare x,y directly
                is_sender = x == sender_core.x and y == sender_core.y
                if not matmul_core_grid.contains(core_range) and not is_sender:
                    phantom_cores.append(core)

        # Build phantom core grid if any exist
        phantom_core_grid = None
        if phantom_cores:
            phantom_core_ranges = [ttnn.CoreRange(c, c) for c in phantom_cores]
            phantom_core_grid = ttnn.CoreRangeSet(phantom_core_ranges)

        # Calculate data sizes
        input_tile_size = in0_tile.get_tile_size(data_format)
        mcast_data_size_bytes = k_num_tiles * input_tile_size

        # Semaphore IDs
        mcast_sender_semaphore_id = 0
        mcast_receiver_semaphore_id = 1

        # Get full device grid for semaphores
        device_grid_size = device.compute_with_storage_grid_size()
        full_device_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )

        # CB indices
        src_cb = 0  # Input on sender core (backed by input tensor)
        dst_cb = 1  # Mcast destination on all matmul cores (dynamically allocated)
        in1_cb = 2  # Weights (backed by weights tensor)
        out_cb = 3  # Output (backed by output tensor)

        # ========================================================================
        # CB descriptors
        # ========================================================================

        # CB 0: Source input (on sender core, backed by input tensor)
        src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(src_cb, input_tensor)

        # CB 0 placeholder on matmul cores + phantom cores to ensure consistent L1 layout
        # This ensures CB 1 (dst_cb) has the same offset on sender and all mcast grid cores
        # Without this, mcast sends data to wrong address because get_write_ptr(dst_cb)
        # returns different offsets on sender vs matmul/phantom cores
        placeholder_core_ranges = matmul_core_grid
        if phantom_core_grid is not None:
            placeholder_core_ranges = placeholder_core_ranges.merge(phantom_core_grid)
        src_cb_placeholder_format = ttnn.CBFormatDescriptor(
            buffer_index=src_cb,
            data_format=data_format,
            page_size=input_tile_size,
            tile=ttnn.TileDescriptor(in0_tile),
        )
        src_cb_placeholder_descriptor = ttnn.CBDescriptor(
            total_size=k_num_tiles * input_tile_size,
            core_ranges=placeholder_core_ranges,
            format_descriptors=[src_cb_placeholder_format],
        )

        # CB 1: Mcast destination (on all matmul cores + sender core + phantom cores)
        # Must be allocated on full bounding box for mcast to write data correctly
        dst_cb_core_ranges = matmul_core_grid.merge(sender_core_grid)
        if phantom_core_grid is not None:
            dst_cb_core_ranges = dst_cb_core_ranges.merge(phantom_core_grid)
        dst_tile_descriptor = ttnn.TileDescriptor(in0_tile)
        dst_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=dst_cb,
            data_format=data_format,
            page_size=input_tile_size,
            tile=dst_tile_descriptor,
        )
        dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=k_num_tiles * input_tile_size,
            core_ranges=dst_cb_core_ranges,
            format_descriptors=[dst_cb_format],
        )

        # CB 2: Weights (backed by weights tensor)
        in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in1_cb, weights_tensor)

        # CB 3: Output (backed by output tensor)
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)

        # ========================================================================
        # Semaphore descriptors
        # ========================================================================

        sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=mcast_sender_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=mcast_receiver_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        # ========================================================================
        # Compile-time args
        # ========================================================================

        # NCRISC compile-time args (mcast receiver + weight buffer setup)
        ncrisc_named_compile_time_args = [
            # Mcast source setup (on sender core)
            ("mcast_src_cb", src_cb),
            ("mcast_src_num_pages", k_num_tiles),
            # Mcast receiver args
            ("mcast_data_receiver_semaphore", mcast_receiver_semaphore_id),
            ("mcast_dst_cb", dst_cb),
            ("mcast_dst_num_pages", k_num_tiles),
            # Weight buffer setup
            ("matmul_in1", in1_cb),
            ("matmul_in1_num_pages", k_num_tiles * out_w_per_core),
        ]

        # BRISC compile-time args (mcast sender)
        brisc_named_compile_time_args = [
            ("mcast_dest_noc_start_x", mcast_dest_noc_start.x),
            ("mcast_dest_noc_start_y", mcast_dest_noc_start.y),
            ("mcast_dest_noc_end_x", mcast_dest_noc_end.x),
            ("mcast_dest_noc_end_y", mcast_dest_noc_end.y),
            ("mcast_num_cores", mcast_num_cores),
            ("mcast_data_sender_semaphore", mcast_sender_semaphore_id),
            ("mcast_data_receiver_semaphore", mcast_receiver_semaphore_id),
            ("mcast_data_size_bytes", mcast_data_size_bytes),
            ("mcast_src_cb", src_cb),
            ("mcast_src_num_pages", k_num_tiles),
            ("mcast_dst_cb", dst_cb),
            ("mcast_is_part_of_receiver_grid", is_sender_in_mcast_grid),
        ]

        # TRISC compile-time args (matmul compute)
        trisc_named_compile_time_args = [
            ("mcast_dst_cb", dst_cb),  # in0 = mcast destination
            ("matmul_in1", in1_cb),  # in1 = weights
            ("matmul_out", out_cb),  # output
            ("matmul_k_num_tiles", k_num_tiles),
            ("matmul_out_w_per_core", out_w_per_core),
        ]

        # ========================================================================
        # Kernel descriptor
        # ========================================================================

        # All cores = union of sender, matmul cores, and phantom cores
        all_cores = matmul_core_grid.merge(sender_core_grid)
        if phantom_core_grid is not None:
            all_cores = all_cores.merge(phantom_core_grid)

        # Mcast grid cores = matmul cores + phantom cores (all cores that receive mcast data)
        # This does NOT include sender unless sender is also a matmul core
        mcast_grid_cores = matmul_core_grid
        if phantom_core_grid is not None:
            mcast_grid_cores = mcast_grid_cores.merge(phantom_core_grid)

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/mcast_matmul/kernels/mcast_matmul_kernel.cpp",
            core_ranges=all_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=brisc_named_compile_time_args,
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=fp32_dest_acc_en,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_sender_core",
                    core_range=sender_core,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_matmul_core",
                    core_range=matmul_core_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_mcast_grid_core",
                    core_range=mcast_grid_cores,
                    value=1,
                    other_value=0,
                ),
            ],
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors(),
            cbs=[
                src_cb_descriptor,
                src_cb_placeholder_descriptor,
                dst_cb_descriptor,
                in1_cb_descriptor,
                out_cb_descriptor,
            ],
            semaphores=[sender_semaphore_descriptor, receiver_semaphore_descriptor],
        )

        # Execute generic op
        io_tensors = [input_tensor, weights_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output


class McastMatmulSiLUMultiCore:
    """
    Multi-core fused matmul+SiLU with mcast input distribution using ttnn.generic_op.

    Computes: output[1, N] = SiLU(input[1, K] @ weights[K, N])

    Fusion benefit: SiLU is applied directly to DST registers after matmul,
    avoiding the L1 round-trip that would occur with separate ops.

    Architecture:
      - Sender core: Has input activations, mcasts to all matmul cores
      - Matmul cores: Receive input, compute fused matmul+SiLU with local weight shard
    """

    @staticmethod
    def golden(input_a, input_b):
        """
        PyTorch reference implementation of fused matmul+SiLU for validation.

        Args:
            input_a: Input tensor A (torch.Tensor) [M, K]
            input_b: Input tensor B (torch.Tensor) [K, N]

        Returns:
            Output tensor [M, N] = SiLU(A @ B)
        """
        matmul_out = input_a @ input_b
        return torch.nn.functional.silu(matmul_out)

    @staticmethod
    def op(
        input_tensor,
        weights_tensor,
        output_tensor,
        fp32_dest_acc_en=False,
    ):
        """
        Execute multi-core fused matmul+SiLU with mcast input distribution.

        Args:
            input_tensor: Input activations [1, K] HEIGHT_SHARDED on single core
            weights_tensor: Weight matrix [K, N] WIDTH_SHARDED across M cores
            output_tensor: Pre-allocated output [1, N] WIDTH_SHARDED across M cores
            fp32_dest_acc_en: Whether to enable FP32 accumulation

        Returns:
            Output tensor with fused matmul+SiLU result
        """
        # Get tensor properties
        device = input_tensor.device()
        data_format = input_tensor.dtype

        # Get tiles
        in0_tile = input_tensor.get_tile()
        in1_tile = weights_tensor.get_tile()
        out_tile = output_tensor.get_tile()

        # Get memory configs and core grids
        input_memory_config = input_tensor.memory_config()
        weights_memory_config = weights_tensor.memory_config()
        output_memory_config = output_tensor.memory_config()

        input_core_grid = input_memory_config.shard_spec.grid
        weights_core_grid = weights_memory_config.shard_spec.grid
        output_core_grid = output_memory_config.shard_spec.grid

        # Extract sender core (first core from input grid)
        input_core_ranges = list(input_core_grid.ranges())
        sender_core = input_core_ranges[0].start
        sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])

        # Matmul cores = weights core grid (same as output core grid)
        matmul_core_grid = weights_core_grid

        # Validate shapes
        input_shard_shape = input_memory_config.shard_spec.shape
        weights_shape = weights_tensor.shape
        weights_shard_shape = weights_memory_config.shard_spec.shape
        output_shard_shape = output_memory_config.shard_spec.shape

        # Input: [1, K] -> shard is [1_tile_height, K]
        assert (
            input_shard_shape[0] // in0_tile.tile_shape[0] == 1
        ), f"Input M ({input_shard_shape[0]}) must be a single tile"
        assert (
            input_shard_shape[1] % in0_tile.tile_shape[1] == 0
        ), f"Input K ({input_shard_shape[1]}) must be divisible by tile width"
        assert (
            input_shard_shape[1] == weights_shape[0]
        ), f"Input K ({input_shard_shape[1]}) must equal weights K ({weights_shape[0]})"

        k_num_tiles = input_shard_shape[1] // in0_tile.tile_shape[1]
        out_w_per_core = output_shard_shape[1] // out_tile.tile_shape[1]

        # Calculate number of matmul cores
        matmul_num_cores = matmul_core_grid.num_cores()

        # Get mcast grid (use the bounding box of ALL matmul core ranges)
        # This is important for non-contiguous CoreRangeSets (e.g., 2-range grids)
        matmul_ranges = list(matmul_core_grid.ranges())

        # Compute bounding box across all ranges
        min_x = min(r.start.x for r in matmul_ranges)
        min_y = min(r.start.y for r in matmul_ranges)
        max_x = max(r.end.x for r in matmul_ranges)
        max_y = max(r.end.y for r in matmul_ranges)

        mcast_grid = ttnn.CoreRange(ttnn.CoreCoord(min_x, min_y), ttnn.CoreCoord(max_x, max_y))

        # Check if sender is part of mcast grid
        is_sender_in_mcast_grid = mcast_grid.contains(sender_core)

        # Get NOC coordinates for mcast destination
        mcast_dest_noc_start = device.worker_core_from_logical_core(mcast_grid.start)
        mcast_dest_noc_end = device.worker_core_from_logical_core(mcast_grid.end)

        # Calculate mcast num cores (bounding box size)
        # For non-contiguous grids, this may include "phantom" cores not in the actual grid
        mcast_num_cores = mcast_grid.grid_size().x * mcast_grid.grid_size().y

        # Build bounding box as CoreRangeSet for phantom core detection
        mcast_grid_set = ttnn.CoreRangeSet([mcast_grid])

        # Compute phantom cores: cores in bounding box but not in matmul grid or sender
        # These cores need to run minimal kernel code to ack the mcast semaphore
        phantom_cores = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                core = ttnn.CoreCoord(x, y)
                core_range = ttnn.CoreRange(core, core)
                # Check if this core is NOT in matmul grid and NOT the sender
                # Note: ttnn.CoreCoord doesn't implement equality correctly, so compare x,y directly
                is_sender = x == sender_core.x and y == sender_core.y
                if not matmul_core_grid.contains(core_range) and not is_sender:
                    phantom_cores.append(core)

        # Build phantom core grid if any exist
        phantom_core_grid = None
        if phantom_cores:
            phantom_core_ranges = [ttnn.CoreRange(c, c) for c in phantom_cores]
            phantom_core_grid = ttnn.CoreRangeSet(phantom_core_ranges)

        # Calculate data sizes
        input_tile_size = in0_tile.get_tile_size(data_format)
        mcast_data_size_bytes = k_num_tiles * input_tile_size

        # Semaphore IDs
        mcast_sender_semaphore_id = 0
        mcast_receiver_semaphore_id = 1

        # Get full device grid for semaphores
        device_grid_size = device.compute_with_storage_grid_size()
        full_device_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )

        # CB indices
        src_cb = 0  # Input on sender core (backed by input tensor)
        dst_cb = 1  # Mcast destination on all matmul cores (dynamically allocated)
        in1_cb = 2  # Weights (backed by weights tensor)
        out_cb = 3  # Output (backed by output tensor)

        # ========================================================================
        # CB descriptors
        # ========================================================================

        # CB 0: Source input (on sender core, backed by input tensor)
        src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(src_cb, input_tensor)

        # CB 0 placeholder on matmul cores + phantom cores to ensure consistent L1 layout
        # This ensures CB 1 (dst_cb) has the same offset on sender and all mcast grid cores
        # Without this, mcast sends data to wrong address because get_write_ptr(dst_cb)
        # returns different offsets on sender vs matmul/phantom cores
        placeholder_core_ranges = matmul_core_grid
        if phantom_core_grid is not None:
            placeholder_core_ranges = placeholder_core_ranges.merge(phantom_core_grid)
        src_cb_placeholder_format = ttnn.CBFormatDescriptor(
            buffer_index=src_cb,
            data_format=data_format,
            page_size=input_tile_size,
            tile=ttnn.TileDescriptor(in0_tile),
        )
        src_cb_placeholder_descriptor = ttnn.CBDescriptor(
            total_size=k_num_tiles * input_tile_size,
            core_ranges=placeholder_core_ranges,
            format_descriptors=[src_cb_placeholder_format],
        )

        # CB 1: Mcast destination (on all matmul cores + sender core + phantom cores)
        # Must be allocated on full bounding box for mcast to write data correctly
        dst_cb_core_ranges = matmul_core_grid.merge(sender_core_grid)
        if phantom_core_grid is not None:
            dst_cb_core_ranges = dst_cb_core_ranges.merge(phantom_core_grid)
        dst_tile_descriptor = ttnn.TileDescriptor(in0_tile)
        dst_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=dst_cb,
            data_format=data_format,
            page_size=input_tile_size,
            tile=dst_tile_descriptor,
        )
        dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=k_num_tiles * input_tile_size,
            core_ranges=dst_cb_core_ranges,
            format_descriptors=[dst_cb_format],
        )

        # CB 2: Weights (backed by weights tensor)
        in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in1_cb, weights_tensor)

        # CB 3: Output (backed by output tensor)
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)

        # ========================================================================
        # Semaphore descriptors
        # ========================================================================

        sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=mcast_sender_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=mcast_receiver_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        # ========================================================================
        # Compile-time args
        # ========================================================================

        # NCRISC compile-time args (mcast receiver + weight buffer setup)
        ncrisc_named_compile_time_args = [
            # Mcast source setup (on sender core)
            ("mcast_src_cb", src_cb),
            ("mcast_src_num_pages", k_num_tiles),
            # Mcast receiver args
            ("mcast_data_receiver_semaphore", mcast_receiver_semaphore_id),
            ("mcast_dst_cb", dst_cb),
            ("mcast_dst_num_pages", k_num_tiles),
            # Weight buffer setup
            ("matmul_in1", in1_cb),
            ("matmul_in1_num_pages", k_num_tiles * out_w_per_core),
        ]

        # BRISC compile-time args (mcast sender)
        brisc_named_compile_time_args = [
            ("mcast_dest_noc_start_x", mcast_dest_noc_start.x),
            ("mcast_dest_noc_start_y", mcast_dest_noc_start.y),
            ("mcast_dest_noc_end_x", mcast_dest_noc_end.x),
            ("mcast_dest_noc_end_y", mcast_dest_noc_end.y),
            ("mcast_num_cores", mcast_num_cores),
            ("mcast_data_sender_semaphore", mcast_sender_semaphore_id),
            ("mcast_data_receiver_semaphore", mcast_receiver_semaphore_id),
            ("mcast_data_size_bytes", mcast_data_size_bytes),
            ("mcast_src_cb", src_cb),
            ("mcast_src_num_pages", k_num_tiles),
            ("mcast_dst_cb", dst_cb),
            ("mcast_is_part_of_receiver_grid", is_sender_in_mcast_grid),
        ]

        # TRISC compile-time args (fused matmul+SiLU compute)
        trisc_named_compile_time_args = [
            ("mcast_dst_cb", dst_cb),  # in0 = mcast destination
            ("matmul_in1", in1_cb),  # in1 = weights
            ("matmul_out", out_cb),  # output
            ("matmul_k_num_tiles", k_num_tiles),
            ("matmul_out_w_per_core", out_w_per_core),
        ]

        # ========================================================================
        # Kernel descriptor
        # ========================================================================

        # All cores = union of sender and matmul cores
        all_cores = matmul_core_grid.merge(sender_core_grid)

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/mcast_matmul/kernels/mcast_matmul_silu_kernel.cpp",
            core_ranges=all_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=brisc_named_compile_time_args,
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=fp32_dest_acc_en,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_sender_core",
                    core_range=sender_core,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_matmul_core",
                    core_range=matmul_core_grid,
                    value=1,
                    other_value=0,
                ),
            ],
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors(),
            cbs=[
                src_cb_descriptor,
                src_cb_placeholder_descriptor,
                dst_cb_descriptor,
                in1_cb_descriptor,
                out_cb_descriptor,
            ],
            semaphores=[sender_semaphore_descriptor, receiver_semaphore_descriptor],
        )

        # Execute generic op
        io_tensors = [input_tensor, weights_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output


class McastSwiGLUMultiCore:
    """
    Multi-core fused SwiGLU with mcast input distribution using ttnn.generic_op.

    Computes: output[1, N] = SiLU(input[1, K] @ W_gate[K, N]) * (input[1, K] @ W_up[K, N])

    Fusion benefit: All three operations (gate matmul+SiLU, up matmul, multiply)
    execute on the same core using local CBs, avoiding any cross-core data movement
    between the operations.

    Architecture:
      - Sender core: Has input activations, mcasts to all matmul cores
      - Matmul cores: Receive input, compute fused SwiGLU with local weight shards
    """

    @staticmethod
    def golden(input_a, gate_weights, up_weights):
        """
        PyTorch reference implementation of fused SwiGLU for validation.

        Args:
            input_a: Input tensor A (torch.Tensor) [M, K]
            gate_weights: W_gate tensor (torch.Tensor) [K, N]
            up_weights: W_up tensor (torch.Tensor) [K, N]

        Returns:
            Output tensor [M, N] = SiLU(A @ W_gate) * (A @ W_up)
        """
        gate = input_a @ gate_weights
        gate = torch.nn.functional.silu(gate)
        up = input_a @ up_weights
        return gate * up

    @staticmethod
    def op(
        input_tensor,
        gate_weights_tensor,
        up_weights_tensor,
        output_tensor,
        fp32_dest_acc_en=False,
    ):
        """
        Execute multi-core fused SwiGLU with mcast input distribution.

        Args:
            input_tensor: Input activations [1, K] HEIGHT_SHARDED on single core
            gate_weights_tensor: W_gate matrix [K, N] WIDTH_SHARDED across M cores
            up_weights_tensor: W_up matrix [K, N] WIDTH_SHARDED across SAME M cores
            output_tensor: Pre-allocated output [1, N] WIDTH_SHARDED across M cores
            fp32_dest_acc_en: Whether to enable FP32 accumulation

        Returns:
            Output tensor with fused SwiGLU result
        """
        # Get tensor properties
        device = input_tensor.device()
        data_format = input_tensor.dtype

        # Get tiles
        in0_tile = input_tensor.get_tile()
        gate_tile = gate_weights_tensor.get_tile()
        up_tile = up_weights_tensor.get_tile()
        out_tile = output_tensor.get_tile()

        # Get memory configs and core grids
        input_memory_config = input_tensor.memory_config()
        gate_memory_config = gate_weights_tensor.memory_config()
        up_memory_config = up_weights_tensor.memory_config()
        output_memory_config = output_tensor.memory_config()

        input_core_grid = input_memory_config.shard_spec.grid
        gate_core_grid = gate_memory_config.shard_spec.grid
        up_core_grid = up_memory_config.shard_spec.grid
        output_core_grid = output_memory_config.shard_spec.grid

        # Validate that W_gate and W_up are on the same core grid
        assert gate_core_grid == up_core_grid, "W_gate and W_up must be sharded on the same core grid"

        # Extract sender core (first core from input grid)
        input_core_ranges = list(input_core_grid.ranges())
        sender_core = input_core_ranges[0].start
        sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])

        # Matmul cores = gate/up weights core grid (same as output core grid)
        matmul_core_grid = gate_core_grid

        # Validate shapes
        input_shard_shape = input_memory_config.shard_spec.shape
        gate_shape = gate_weights_tensor.shape
        gate_shard_shape = gate_memory_config.shard_spec.shape
        up_shape = up_weights_tensor.shape
        up_shard_shape = up_memory_config.shard_spec.shape
        output_shard_shape = output_memory_config.shard_spec.shape

        # Input: [1, K] -> shard is [1_tile_height, K]
        assert (
            input_shard_shape[0] // in0_tile.tile_shape[0] == 1
        ), f"Input M ({input_shard_shape[0]}) must be a single tile"
        assert (
            input_shard_shape[1] % in0_tile.tile_shape[1] == 0
        ), f"Input K ({input_shard_shape[1]}) must be divisible by tile width"
        assert (
            input_shard_shape[1] == gate_shape[0]
        ), f"Input K ({input_shard_shape[1]}) must equal W_gate K ({gate_shape[0]})"
        assert (
            input_shard_shape[1] == up_shape[0]
        ), f"Input K ({input_shard_shape[1]}) must equal W_up K ({up_shape[0]})"
        assert gate_shape == up_shape, f"W_gate shape {gate_shape} must equal W_up shape {up_shape}"
        assert (
            gate_shard_shape == up_shard_shape
        ), f"W_gate shard shape {gate_shard_shape} must equal W_up shard shape {up_shard_shape}"

        k_num_tiles = input_shard_shape[1] // in0_tile.tile_shape[1]
        out_w_per_core = output_shard_shape[1] // out_tile.tile_shape[1]

        # Calculate number of matmul cores
        matmul_num_cores = matmul_core_grid.num_cores()

        # Get mcast grid (use the bounding box of ALL matmul core ranges)
        # This is important for non-contiguous CoreRangeSets (e.g., 2-range grids)
        matmul_ranges = list(matmul_core_grid.ranges())

        # Compute bounding box across all ranges
        min_x = min(r.start.x for r in matmul_ranges)
        min_y = min(r.start.y for r in matmul_ranges)
        max_x = max(r.end.x for r in matmul_ranges)
        max_y = max(r.end.y for r in matmul_ranges)

        mcast_grid = ttnn.CoreRange(ttnn.CoreCoord(min_x, min_y), ttnn.CoreCoord(max_x, max_y))

        # Check if sender is part of mcast grid
        is_sender_in_mcast_grid = mcast_grid.contains(sender_core)

        # Get NOC coordinates for mcast destination
        mcast_dest_noc_start = device.worker_core_from_logical_core(mcast_grid.start)
        mcast_dest_noc_end = device.worker_core_from_logical_core(mcast_grid.end)

        # Calculate mcast num cores (bounding box size)
        # For non-contiguous grids, this may include "phantom" cores not in the actual grid
        mcast_num_cores = mcast_grid.grid_size().x * mcast_grid.grid_size().y

        # Build bounding box as CoreRangeSet for phantom core detection
        mcast_grid_set = ttnn.CoreRangeSet([mcast_grid])

        # Compute phantom cores: cores in bounding box but not in matmul grid or sender
        # These cores need to run minimal kernel code to ack the mcast semaphore
        phantom_cores = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                core = ttnn.CoreCoord(x, y)
                core_range = ttnn.CoreRange(core, core)
                # Check if this core is NOT in matmul grid and NOT the sender
                # Note: ttnn.CoreCoord doesn't implement equality correctly, so compare x,y directly
                is_sender = x == sender_core.x and y == sender_core.y
                if not matmul_core_grid.contains(core_range) and not is_sender:
                    phantom_cores.append(core)

        # Build phantom core grid if any exist
        phantom_core_grid = None
        if phantom_cores:
            phantom_core_ranges = [ttnn.CoreRange(c, c) for c in phantom_cores]
            phantom_core_grid = ttnn.CoreRangeSet(phantom_core_ranges)

        # Calculate data sizes
        input_tile_size = in0_tile.get_tile_size(data_format)
        out_tile_size = out_tile.get_tile_size(ttnn.bfloat16)  # intermediate CBs use bfloat16
        mcast_data_size_bytes = k_num_tiles * input_tile_size

        # Semaphore IDs
        mcast_sender_semaphore_id = 0
        mcast_receiver_semaphore_id = 1

        # Get full device grid for semaphores
        device_grid_size = device.compute_with_storage_grid_size()
        full_device_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )

        # CB indices
        src_cb = 0  # Input on sender core (backed by input tensor)
        dst_cb = 1  # Mcast destination on all matmul cores (dynamically allocated)
        gate_weights_cb = 2  # W_gate (backed by tensor)
        up_weights_cb = 3  # W_up (backed by tensor)
        gate_intermediate_cb = 4  # Gate output (dynamically allocated)
        up_intermediate_cb = 5  # Up output (dynamically allocated)
        out_cb = 6  # Final output (backed by output tensor)

        # ========================================================================
        # CB descriptors
        # ========================================================================

        # CB 0: Source input (on sender core, backed by input tensor)
        src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(src_cb, input_tensor)

        # CB 0 placeholder on matmul cores + phantom cores to ensure consistent L1 layout
        # This ensures CB 1 (dst_cb) has the same offset on sender and all mcast grid cores
        # Without this, mcast sends data to wrong address because get_write_ptr(dst_cb)
        # returns different offsets on sender vs matmul/phantom cores
        placeholder_core_ranges = matmul_core_grid
        if phantom_core_grid is not None:
            placeholder_core_ranges = placeholder_core_ranges.merge(phantom_core_grid)
        src_cb_placeholder_format = ttnn.CBFormatDescriptor(
            buffer_index=src_cb,
            data_format=data_format,
            page_size=input_tile_size,
            tile=ttnn.TileDescriptor(in0_tile),
        )
        src_cb_placeholder_descriptor = ttnn.CBDescriptor(
            total_size=k_num_tiles * input_tile_size,
            core_ranges=placeholder_core_ranges,
            format_descriptors=[src_cb_placeholder_format],
        )

        # CB 1: Mcast destination (on all matmul cores + sender core + phantom cores)
        # Must be allocated on full bounding box for mcast to write data correctly
        dst_cb_core_ranges = matmul_core_grid.merge(sender_core_grid)
        if phantom_core_grid is not None:
            dst_cb_core_ranges = dst_cb_core_ranges.merge(phantom_core_grid)
        dst_tile_descriptor = ttnn.TileDescriptor(in0_tile)
        dst_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=dst_cb,
            data_format=data_format,
            page_size=input_tile_size,
            tile=dst_tile_descriptor,
        )
        dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=k_num_tiles * input_tile_size,
            core_ranges=dst_cb_core_ranges,
            format_descriptors=[dst_cb_format],
        )

        # CB 2: W_gate weights (backed by tensor)
        gate_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gate_weights_cb, gate_weights_tensor)

        # CB 3: W_up weights (backed by tensor)
        up_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(up_weights_cb, up_weights_tensor)

        # CB 4: Gate intermediate (dynamically allocated on matmul cores)
        gate_intermediate_tile_descriptor = ttnn.TileDescriptor(out_tile)
        gate_intermediate_format = ttnn.CBFormatDescriptor(
            buffer_index=gate_intermediate_cb,
            data_format=ttnn.bfloat16,  # intermediate uses bfloat16
            page_size=out_tile_size,
            tile=gate_intermediate_tile_descriptor,
        )
        gate_intermediate_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_w_per_core * out_tile_size,
            core_ranges=matmul_core_grid,
            format_descriptors=[gate_intermediate_format],
        )

        # CB 5: Up intermediate (dynamically allocated on matmul cores)
        up_intermediate_tile_descriptor = ttnn.TileDescriptor(out_tile)
        up_intermediate_format = ttnn.CBFormatDescriptor(
            buffer_index=up_intermediate_cb,
            data_format=ttnn.bfloat16,  # intermediate uses bfloat16
            page_size=out_tile_size,
            tile=up_intermediate_tile_descriptor,
        )
        up_intermediate_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_w_per_core * out_tile_size,
            core_ranges=matmul_core_grid,
            format_descriptors=[up_intermediate_format],
        )

        # CB 6: Output (backed by output tensor)
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)

        # ========================================================================
        # Semaphore descriptors
        # ========================================================================

        sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=mcast_sender_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=mcast_receiver_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        # ========================================================================
        # Compile-time args
        # ========================================================================

        # NCRISC compile-time args (mcast receiver + weight buffer setup)
        ncrisc_named_compile_time_args = [
            # Mcast source setup (on sender core)
            ("mcast_src_cb", src_cb),
            ("mcast_src_num_pages", k_num_tiles),
            # Mcast receiver args
            ("mcast_data_receiver_semaphore", mcast_receiver_semaphore_id),
            ("mcast_dst_cb", dst_cb),
            ("mcast_dst_num_pages", k_num_tiles),
            # Weight buffer setup (both gate and up)
            ("gate_weights_cb", gate_weights_cb),
            ("gate_weights_num_pages", k_num_tiles * out_w_per_core),
            ("up_weights_cb", up_weights_cb),
            ("up_weights_num_pages", k_num_tiles * out_w_per_core),
        ]

        # BRISC compile-time args (mcast sender)
        brisc_named_compile_time_args = [
            ("mcast_dest_noc_start_x", mcast_dest_noc_start.x),
            ("mcast_dest_noc_start_y", mcast_dest_noc_start.y),
            ("mcast_dest_noc_end_x", mcast_dest_noc_end.x),
            ("mcast_dest_noc_end_y", mcast_dest_noc_end.y),
            ("mcast_num_cores", mcast_num_cores),
            ("mcast_data_sender_semaphore", mcast_sender_semaphore_id),
            ("mcast_data_receiver_semaphore", mcast_receiver_semaphore_id),
            ("mcast_data_size_bytes", mcast_data_size_bytes),
            ("mcast_src_cb", src_cb),
            ("mcast_src_num_pages", k_num_tiles),
            ("mcast_dst_cb", dst_cb),
            ("mcast_is_part_of_receiver_grid", is_sender_in_mcast_grid),
        ]

        # TRISC compile-time args (fused SwiGLU compute)
        trisc_named_compile_time_args = [
            ("mcast_dst_cb", dst_cb),  # in0 = mcast destination
            ("gate_weights_cb", gate_weights_cb),  # W_gate
            ("up_weights_cb", up_weights_cb),  # W_up
            ("gate_intermediate_cb", gate_intermediate_cb),
            ("up_intermediate_cb", up_intermediate_cb),
            ("out_cb", out_cb),
            ("k_num_tiles", k_num_tiles),
            ("out_w_per_core", out_w_per_core),
        ]

        # ========================================================================
        # Kernel descriptor
        # ========================================================================

        # All cores = union of sender and matmul cores
        all_cores = matmul_core_grid.merge(sender_core_grid)

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/mcast_matmul/kernels/mcast_swiglu_kernel.cpp",
            core_ranges=all_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=brisc_named_compile_time_args,
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=fp32_dest_acc_en,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_sender_core",
                    core_range=sender_core,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_matmul_core",
                    core_range=matmul_core_grid,
                    value=1,
                    other_value=0,
                ),
            ],
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors(),
            cbs=[
                src_cb_descriptor,
                dst_cb_descriptor,
                gate_weights_cb_descriptor,
                up_weights_cb_descriptor,
                gate_intermediate_cb_descriptor,
                up_intermediate_cb_descriptor,
                out_cb_descriptor,
            ],
            semaphores=[sender_semaphore_descriptor, receiver_semaphore_descriptor],
        )

        # Execute generic op
        io_tensors = [input_tensor, gate_weights_tensor, up_weights_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output


class McastDisjointSwiGLUMultiCore:
    """
    Multi-core fused SwiGLU with DISJOINT gate/up grids using ttnn.generic_op.

    Computes: output[1, N] = SiLU(input[1, K] @ W_gate[K, N]) * (input[1, K] @ W_up[K, N])

    Architecture:
      - Input activations: HEIGHT_SHARDED on single sender core
      - W_gate: WIDTH_SHARDED across gate_grid (e.g., 8x9 = 72 cores)
      - W_up: WIDTH_SHARDED across up_grid (e.g., 4x9 = 36 cores)
      - Gate and Up grids are DISJOINT (different cores)
      - Output: WIDTH_SHARDED on up_grid

    Data flow:
      1. Mcast: sender → all matmul cores (gate_grid ∪ up_grid)
      2. Gate cores: SiLU(input @ W_gate), then send to corresponding up cores
      3. Up cores: Receive gate results, compute input @ W_up, multiply
    """

    @staticmethod
    def golden(input_a, gate_weights, up_weights):
        """
        PyTorch reference implementation of fused SwiGLU for validation.

        Args:
            input_a: Input tensor A (torch.Tensor) [M, K]
            gate_weights: W_gate tensor (torch.Tensor) [K, N]
            up_weights: W_up tensor (torch.Tensor) [K, N]

        Returns:
            Output tensor [M, N] = SiLU(A @ W_gate) * (A @ W_up)
        """
        gate = input_a @ gate_weights
        gate = torch.nn.functional.silu(gate)
        up = input_a @ up_weights
        return gate * up

    @staticmethod
    def op(
        input_tensor,
        gate_weights_tensor,
        up_weights_tensor,
        output_tensor,
        fp32_dest_acc_en=False,
    ):
        """
        Execute multi-core fused SwiGLU with disjoint gate/up grids.

        Args:
            input_tensor: Input activations [1, K] HEIGHT_SHARDED on single core
            gate_weights_tensor: W_gate matrix [K, N] WIDTH_SHARDED across gate_grid
            up_weights_tensor: W_up matrix [K, N] WIDTH_SHARDED across up_grid
            output_tensor: Pre-allocated output [1, N] WIDTH_SHARDED across up_grid
            fp32_dest_acc_en: Whether to enable FP32 accumulation

        Returns:
            Output tensor with fused SwiGLU result
        """
        # Get tensor properties
        device = input_tensor.device()
        data_format = input_tensor.dtype

        # Get tiles
        in0_tile = input_tensor.get_tile()
        gate_tile = gate_weights_tensor.get_tile()
        up_tile = up_weights_tensor.get_tile()
        out_tile = output_tensor.get_tile()

        # Get memory configs and core grids
        input_memory_config = input_tensor.memory_config()
        gate_memory_config = gate_weights_tensor.memory_config()
        up_memory_config = up_weights_tensor.memory_config()
        output_memory_config = output_tensor.memory_config()

        input_core_grid = input_memory_config.shard_spec.grid
        gate_core_grid = gate_memory_config.shard_spec.grid
        up_core_grid = up_memory_config.shard_spec.grid
        output_core_grid = output_memory_config.shard_spec.grid

        # Validate that gate and up grids are DISJOINT
        gate_ranges = list(gate_core_grid.ranges())
        up_ranges = list(up_core_grid.ranges())
        assert len(gate_ranges) == 1, "Gate grid must be a single contiguous range"
        assert len(up_ranges) == 1, "Up grid must be a single contiguous range"

        gate_range = gate_ranges[0]
        up_range = up_ranges[0]

        # Gate and up should have same number of rows
        gate_num_rows = gate_range.end.y - gate_range.start.y + 1
        up_num_rows = up_range.end.y - up_range.start.y + 1
        assert gate_num_rows == up_num_rows, f"Gate rows ({gate_num_rows}) must equal up rows ({up_num_rows})"

        gate_num_cols = gate_range.end.x - gate_range.start.x + 1
        up_num_cols = up_range.end.x - up_range.start.x + 1

        # Validate gate cols is 2x up cols for proper mapping
        assert gate_num_cols == 2 * up_num_cols, f"Gate cols ({gate_num_cols}) must be 2x up cols ({up_num_cols})"

        # Extract sender core (first core from input grid)
        input_core_ranges = list(input_core_grid.ranges())
        sender_core = input_core_ranges[0].start
        sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])

        # Validate shapes
        input_shard_shape = input_memory_config.shard_spec.shape
        gate_shape = gate_weights_tensor.shape
        gate_shard_shape = gate_memory_config.shard_spec.shape
        up_shape = up_weights_tensor.shape
        up_shard_shape = up_memory_config.shard_spec.shape
        output_shard_shape = output_memory_config.shard_spec.shape

        # Input: [1, K]
        assert (
            input_shard_shape[0] // in0_tile.tile_shape[0] == 1
        ), f"Input M ({input_shard_shape[0]}) must be a single tile"
        assert (
            input_shard_shape[1] == gate_shape[0]
        ), f"Input K ({input_shard_shape[1]}) must equal W_gate K ({gate_shape[0]})"
        assert (
            input_shard_shape[1] == up_shape[0]
        ), f"Input K ({input_shard_shape[1]}) must equal W_up K ({up_shape[0]})"
        assert gate_shape[1] == up_shape[1], f"W_gate N ({gate_shape[1]}) must equal W_up N ({up_shape[1]})"

        k_num_tiles = input_shard_shape[1] // in0_tile.tile_shape[1]
        out_w_per_gate_core = gate_shard_shape[1] // out_tile.tile_shape[1]
        out_w_per_up_core = up_shard_shape[1] // out_tile.tile_shape[1]

        # Validate output shard matches up shard
        assert (
            output_shard_shape[1] == up_shard_shape[1]
        ), f"Output shard width ({output_shard_shape[1]}) must equal up shard width ({up_shard_shape[1]})"

        # Calculate number of cores
        gate_num_cores = gate_core_grid.num_cores()
        up_num_cores = up_core_grid.num_cores()

        # Mcast grid = bounding box of gate ∪ up grids
        mcast_start_x = min(gate_range.start.x, up_range.start.x)
        mcast_start_y = min(gate_range.start.y, up_range.start.y)
        mcast_end_x = max(gate_range.end.x, up_range.end.x)
        mcast_end_y = max(gate_range.end.y, up_range.end.y)
        mcast_grid = ttnn.CoreRange(
            ttnn.CoreCoord(mcast_start_x, mcast_start_y), ttnn.CoreCoord(mcast_end_x, mcast_end_y)
        )

        # Check if sender is part of mcast grid
        is_sender_in_mcast_grid = mcast_grid.contains(sender_core)

        # Get NOC coordinates for mcast destination
        mcast_dest_noc_start = device.worker_core_from_logical_core(mcast_grid.start)
        mcast_dest_noc_end = device.worker_core_from_logical_core(mcast_grid.end)

        # Calculate mcast num cores (entire grid, not just gate+up)
        mcast_num_cores = mcast_grid.grid_size().x * mcast_grid.grid_size().y

        # Calculate data sizes
        input_tile_size = in0_tile.get_tile_size(data_format)
        out_tile_size = out_tile.get_tile_size(ttnn.bfloat16)
        mcast_data_size_bytes = k_num_tiles * input_tile_size

        # Semaphore IDs
        mcast_sender_semaphore_id = 0
        mcast_receiver_semaphore_id = 1
        gate_recv_semaphore_id = 2  # For gate→up transfer

        # Get full device grid for semaphores
        device_grid_size = device.compute_with_storage_grid_size()
        full_device_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )

        # CB indices
        src_cb = 0  # Input on sender core (backed by input tensor)
        dst_cb = 1  # Mcast destination on all matmul cores
        gate_weights_cb = 2  # W_gate (backed by tensor, on gate cores)
        up_weights_cb = 3  # W_up (backed by tensor, on up cores)
        gate_output_cb = 4  # Gate output (on gate cores, sent to up)
        gate_recv_cb = 5  # Received gate results (on up cores)
        up_output_cb = 6  # Up matmul output (on up cores)
        out_cb = 7  # Final output (backed by output tensor, on up cores)

        # ========================================================================
        # CB descriptors
        # ========================================================================

        # CB 0: Source input (on sender core, backed by input tensor)
        src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(src_cb, input_tensor)

        # CB 1: Mcast destination (on all gate + up cores + sender)
        all_matmul_cores = gate_core_grid.merge(up_core_grid)
        dst_cb_core_ranges = all_matmul_cores.merge(sender_core_grid)
        dst_tile_descriptor = ttnn.TileDescriptor(in0_tile)
        dst_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=dst_cb,
            data_format=data_format,
            page_size=input_tile_size,
            tile=dst_tile_descriptor,
        )
        dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=k_num_tiles * input_tile_size,
            core_ranges=dst_cb_core_ranges,
            format_descriptors=[dst_cb_format],
        )

        # CB 2: W_gate weights (backed by tensor on gate cores)
        # Also need placeholder on up cores for consistent L1 layout
        gate_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gate_weights_cb, gate_weights_tensor)

        # CB 3: W_up weights (backed by tensor on up cores)
        # Also need placeholder on gate cores for consistent L1 layout
        up_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(up_weights_cb, up_weights_tensor)

        # CRITICAL: For CB 5 (gate_recv_cb) to have the same L1 address on gate and up cores,
        # we need: CB2_gate + CB3_gate = CB2_up + CB3_up
        #
        # CB 2 on gate cores: tensor-backed with gate_weights_size
        # CB 2 on up cores: placeholder - should match gate_weights_size
        # CB 3 on gate cores: placeholder - should match up_weights_size
        # CB 3 on up cores: tensor-backed with up_weights_size
        #
        # This ensures sum(CB0-4) is equal on both core types, so CB5 starts at same offset.
        weights_dtype = gate_weights_tensor.dtype
        gate_weights_size = k_num_tiles * out_w_per_gate_core * gate_tile.get_tile_size(weights_dtype)
        up_weights_size = k_num_tiles * out_w_per_up_core * up_tile.get_tile_size(weights_dtype)

        # CB 2 placeholder on up cores - must match CB 2 tensor size on gate cores
        gate_weights_placeholder_format = ttnn.CBFormatDescriptor(
            buffer_index=gate_weights_cb,
            data_format=weights_dtype,
            page_size=gate_tile.get_tile_size(weights_dtype),
            tile=ttnn.TileDescriptor(gate_tile),
        )
        gate_weights_placeholder_descriptor = ttnn.CBDescriptor(
            total_size=gate_weights_size,  # Match gate_weights tensor size for consistent layout
            core_ranges=up_core_grid,  # Placeholder on up cores
            format_descriptors=[gate_weights_placeholder_format],
        )

        # CB 3 placeholder on gate cores - must match CB 3 tensor size on up cores
        up_weights_placeholder_format = ttnn.CBFormatDescriptor(
            buffer_index=up_weights_cb,
            data_format=weights_dtype,
            page_size=up_tile.get_tile_size(weights_dtype),
            tile=ttnn.TileDescriptor(up_tile),
        )
        up_weights_placeholder_descriptor = ttnn.CBDescriptor(
            total_size=up_weights_size,  # Match up_weights tensor size for consistent layout
            core_ranges=gate_core_grid,  # Placeholder on gate cores
            format_descriptors=[up_weights_placeholder_format],
        )

        # CB 4: Gate output (on ALL matmul cores for consistent L1 layout)
        # Allocated on all cores so that CBs 5-6 have consistent addresses
        gate_output_tile_descriptor = ttnn.TileDescriptor(out_tile)
        gate_output_format = ttnn.CBFormatDescriptor(
            buffer_index=gate_output_cb,
            data_format=ttnn.bfloat16,
            page_size=out_tile_size,
            tile=gate_output_tile_descriptor,
        )
        gate_output_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_w_per_gate_core * out_tile_size,
            core_ranges=all_matmul_cores,  # ALL cores for consistent L1 layout
            format_descriptors=[gate_output_format],
        )

        # CB 5: Gate recv (on ALL matmul cores for consistent L1 layout)
        # Gate cores need get_write_ptr to return same address as on up cores
        gate_recv_tile_descriptor = ttnn.TileDescriptor(out_tile)
        gate_recv_format = ttnn.CBFormatDescriptor(
            buffer_index=gate_recv_cb,
            data_format=ttnn.bfloat16,
            page_size=out_tile_size,
            tile=gate_recv_tile_descriptor,
        )
        gate_recv_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_w_per_up_core * out_tile_size,  # 2 * out_w_per_gate_core
            core_ranges=all_matmul_cores,  # ALL cores for consistent L1 layout
            format_descriptors=[gate_recv_format],
        )

        # CB 6: Up output (on ALL matmul cores for consistent L1 layout)
        up_output_tile_descriptor = ttnn.TileDescriptor(out_tile)
        up_output_format = ttnn.CBFormatDescriptor(
            buffer_index=up_output_cb,
            data_format=ttnn.bfloat16,
            page_size=out_tile_size,
            tile=up_output_tile_descriptor,
        )
        up_output_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_w_per_up_core * out_tile_size,
            core_ranges=all_matmul_cores,  # ALL cores for consistent L1 layout
            format_descriptors=[up_output_format],
        )

        # CB 7: Output (backed by output tensor, on up cores)
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)

        # ========================================================================
        # Semaphore descriptors
        # ========================================================================

        sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=mcast_sender_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=mcast_receiver_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        gate_recv_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=gate_recv_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        # ========================================================================
        # Compile-time args (common to all cores)
        # ========================================================================

        # Calculate gate→up transfer parameters
        # Each gate core needs to know its target up core NOC address
        # Gate core (gx, gy) → Up core (up_start_x + gx // 2, up_start_y + gy - gate_start_y)
        # Transfer offset = (gx % 2) * out_w_per_gate_core * tile_size

        # For simplicity in this implementation, we'll use per-core compile-time args
        # via the unified kernel descriptor

        # NCRISC compile-time args
        ncrisc_named_compile_time_args = [
            # Mcast source setup (on sender core)
            ("mcast_src_cb", src_cb),
            ("mcast_src_num_pages", k_num_tiles),
            # Mcast receiver args
            ("mcast_data_receiver_semaphore", mcast_receiver_semaphore_id),
            ("mcast_dst_cb", dst_cb),
            ("mcast_dst_num_pages", k_num_tiles),
            # Gate weights setup
            ("gate_weights_cb", gate_weights_cb),
            ("gate_weights_num_pages", k_num_tiles * out_w_per_gate_core),
            # Up weights setup
            ("up_weights_cb", up_weights_cb),
            ("up_weights_num_pages", k_num_tiles * out_w_per_up_core),
            # Gate recv setup
            ("gate_recv_cb", gate_recv_cb),
            ("gate_recv_num_pages", out_w_per_up_core),
            ("gate_recv_semaphore", gate_recv_semaphore_id),
            ("gate_cores_per_up_core", 2),  # 2 gate cores send to each up core
        ]

        # Get NOC coordinates for gate and up grid starts (for gate→up transfer)
        gate_grid_start_noc = device.worker_core_from_logical_core(gate_range.start)
        up_grid_start_noc = device.worker_core_from_logical_core(up_range.start)

        # BRISC compile-time args
        brisc_named_compile_time_args = [
            ("mcast_dest_noc_start_x", mcast_dest_noc_start.x),
            ("mcast_dest_noc_start_y", mcast_dest_noc_start.y),
            ("mcast_dest_noc_end_x", mcast_dest_noc_end.x),
            ("mcast_dest_noc_end_y", mcast_dest_noc_end.y),
            ("mcast_num_cores", mcast_num_cores),
            ("mcast_data_sender_semaphore", mcast_sender_semaphore_id),
            ("mcast_data_receiver_semaphore", mcast_receiver_semaphore_id),
            ("mcast_data_size_bytes", mcast_data_size_bytes),
            ("mcast_src_cb", src_cb),
            ("mcast_src_num_pages", k_num_tiles),
            ("mcast_dst_cb", dst_cb),
            ("mcast_is_part_of_receiver_grid", is_sender_in_mcast_grid),
            # Gate output params (for gate cores)
            ("gate_output_cb", gate_output_cb),
            ("gate_output_num_pages", out_w_per_gate_core),
            ("gate_recv_cb", gate_recv_cb),
            ("gate_recv_semaphore", gate_recv_semaphore_id),
            # Grid params for computing target up core at runtime (all in NOC coordinates)
            ("gate_grid_start_noc_x", gate_grid_start_noc.x),
            ("gate_grid_start_noc_y", gate_grid_start_noc.y),
            ("up_grid_start_noc_x", up_grid_start_noc.x),
            ("up_grid_start_noc_y", up_grid_start_noc.y),
            ("out_tile_size_bytes", out_tile_size),
            ("out_w_per_gate_core", out_w_per_gate_core),
            ("gate_cores_per_up_core", 2),
        ]

        # TRISC compile-time args
        trisc_named_compile_time_args = [
            ("mcast_dst_cb", dst_cb),
            ("gate_weights_cb", gate_weights_cb),
            ("gate_output_cb", gate_output_cb),
            ("up_weights_cb", up_weights_cb),
            ("gate_recv_cb", gate_recv_cb),
            ("up_output_cb", up_output_cb),
            ("out_cb", out_cb),
            ("k_num_tiles", k_num_tiles),
            ("out_w_per_gate_core", out_w_per_gate_core),
            ("out_w_per_up_core", out_w_per_up_core),
        ]

        # ========================================================================
        # Kernel descriptor
        # ========================================================================

        # All cores = sender ∪ gate ∪ up
        all_cores = gate_core_grid.merge(up_core_grid).merge(sender_core_grid)

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/mcast_matmul/kernels/mcast_disjoint_swiglu_kernel.cpp",
            core_ranges=all_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=brisc_named_compile_time_args,
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=fp32_dest_acc_en,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_sender_core",
                    core_range=sender_core,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_gate_core",
                    core_range=gate_core_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_up_core",
                    core_range=up_core_grid,
                    value=1,
                    other_value=0,
                ),
            ],
        )

        # Create program descriptor
        # Include placeholder descriptors to ensure consistent CB layout across all cores
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors(),
            cbs=[
                src_cb_descriptor,
                dst_cb_descriptor,
                gate_weights_cb_descriptor,
                gate_weights_placeholder_descriptor,  # CB 2 placeholder on up cores
                up_weights_cb_descriptor,
                up_weights_placeholder_descriptor,  # CB 3 placeholder on gate cores
                gate_output_cb_descriptor,
                gate_recv_cb_descriptor,
                up_output_cb_descriptor,
                out_cb_descriptor,
            ],
            semaphores=[
                sender_semaphore_descriptor,
                receiver_semaphore_descriptor,
                gate_recv_semaphore_descriptor,
            ],
        )

        # Execute generic op
        io_tensors = [input_tensor, gate_weights_tensor, up_weights_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
