# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
MoE Routed Expert fused operation.

This implements the MoE routed expert computation (will be extended with more fusions):
- Input: [1, K] tensor on sender core (outside compute grid)
- Mcast to N compute cores
- Each core computes: matmul + activation
- Output: width-sharded across compute cores
"""

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class MoeRoutedExpert:
    """
    MoE Routed Expert fused operation implementation using ttnn.generic_op.
    """

    # Fused activation enum values (must match matmul.hpp FusedActivation enum)
    ACTIVATION_NONE = 0
    ACTIVATION_SIGMOID = 1
    ACTIVATION_SILU = 2

    @staticmethod
    def golden(input_tensor, matmul_weights_tensor):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) [1, K]
            matmul_weights_tensor: Matmul weights (torch.Tensor) [K, N]

        Returns:
            Output tensor with matmul + sigmoid: [1, N]
        """
        import torch

        result = input_tensor @ matmul_weights_tensor
        return torch.sigmoid(result)

    @staticmethod
    def op(
        input_tensor,
        matmul_weights_tensor,
        output_tensor,
        fp32_dest_acc_en=True,
    ):
        """
        Execute mcast + matmul + sigmoid fused operation using generic_op.

        Args:
            input_tensor: Input tensor [1, K] sharded on single sender core (outside matmul grid)
            matmul_weights_tensor: Matmul weights [K, N] width-sharded on matmul cores
            output_tensor: Pre-allocated output tensor [1, N] width-sharded on matmul cores
            fp32_dest_acc_en: Whether to enable FP32 accumulation (default True for precision)

        Returns:
            Output tensor with mcast + matmul + sigmoid result
        """
        # Get tensor properties
        input_shape = input_tensor.shape
        data_format = input_tensor.dtype

        # Tile definitions
        TILE_1x32 = ttnn.Tile((1, 32))
        tile_1x32_size = TILE_1x32.get_tile_size(data_format)

        # Calculate K dimension in tiles
        K = input_shape[1]
        num_tiles_k = K // TILE_1x32.tile_shape[1]  # K / 32

        # Get input core grid (single core for input)
        input_memory_config = input_tensor.memory_config()
        input_core_grid = input_memory_config.shard_spec.grid
        input_core_ranges = list(input_core_grid.ranges())
        sender_core = input_core_ranges[0].start
        sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])

        # Get matmul weights core grid (8 cores for width sharding)
        matmul_weights_memory_config = matmul_weights_tensor.memory_config()
        matmul_weights_core_grid = matmul_weights_memory_config.shard_spec.grid
        num_matmul_cores = matmul_weights_core_grid.num_cores()

        # Get per-core output width in tiles
        matmul_weights_tile = matmul_weights_tensor.get_tile()
        matmul_weights_shard_shape = matmul_weights_memory_config.shard_spec.shape
        matmul_weights_shard_width = matmul_weights_shard_shape[1]
        matmul_out_w = matmul_weights_shard_width // matmul_weights_tile.tile_shape[1]

        # Get device and compute grid
        device = input_tensor.device()
        device_grid_size = device.compute_with_storage_grid_size()

        # Mcast grid: rectangle that includes both sender and matmul cores
        # Get matmul grid bounds
        matmul_grid_ranges = list(matmul_weights_core_grid.ranges())
        matmul_grid_range = matmul_grid_ranges[0]
        matmul_start = matmul_grid_range.start
        matmul_end = matmul_grid_range.end

        # Compute mcast grid as bounding box of sender and matmul cores
        mcast_grid_start_x = min(sender_core.x, matmul_start.x)
        mcast_grid_start_y = min(sender_core.y, matmul_start.y)
        mcast_grid_end_x = max(sender_core.x, matmul_end.x)
        mcast_grid_end_y = max(sender_core.y, matmul_end.y)

        mcast_grid_start = ttnn.CoreCoord(mcast_grid_start_x, mcast_grid_start_y)
        mcast_grid_end = ttnn.CoreCoord(mcast_grid_end_x, mcast_grid_end_y)
        mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_grid_start, mcast_grid_end)])

        # Get NOC coordinates for mcast destination
        mcast_dest_noc_start_core = device.worker_core_from_logical_core(mcast_grid_start)
        mcast_dest_noc_end_core = device.worker_core_from_logical_core(mcast_grid_end)

        # Calculate mcast parameters
        mcast_num_cores = (mcast_grid_end_x - mcast_grid_start_x + 1) * (mcast_grid_end_y - mcast_grid_start_y + 1)
        mcast_is_part_of_receiver_grid = mcast_grid.contains(sender_core_grid)

        # Semaphore IDs
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1

        # Calculate mcast data size in bytes
        mcast_data_size_bytes = num_tiles_k * tile_1x32_size

        # CB indices
        input_cb = 0  # Input tensor (sharded on sender core)
        matmul_input_cb = 1  # Mcast destination CB (receives input on all cores)
        matmul_weights_cb = 2  # Matmul weights (sharded on all cores)
        matmul_output_cb = 3  # Matmul output (maps to output tensor)

        # CB descriptors
        # CB 0: Input tensor (sharded on sender core)
        input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)

        # CB 1: Mcast destination (receives input, allocated on union of sender and receiver grids)
        tile_1x32_descriptor = ttnn.TileDescriptor(TILE_1x32)
        matmul_input_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=matmul_input_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=tile_1x32_descriptor,
        )
        matmul_input_cb_core_ranges = matmul_weights_core_grid.merge(sender_core_grid)
        matmul_input_cb_descriptor = ttnn.CBDescriptor(
            total_size=num_tiles_k * tile_1x32_size,
            core_ranges=matmul_input_cb_core_ranges,
            format_descriptors=[matmul_input_cb_format],
        )

        # CB 2: Matmul weights (sharded on all matmul cores)
        matmul_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_weights_cb, matmul_weights_tensor)

        # CB 3: Matmul output (maps to output tensor)
        matmul_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_output_cb, output_tensor)

        # Mcast page counts
        mcast_src_num_pages = num_tiles_k
        mcast_dst_num_pages = num_tiles_k

        # Named compile-time args for NCRISC (mcast receiver + matmul reader)
        ncrisc_named_compile_time_args = [
            # Mcast sender sharded buffer setup (for sender core)
            ("mcast_src_cb", input_cb),
            ("mcast_src_num_pages", mcast_src_num_pages),
            # Mcast receiver args
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_dst_cb", matmul_input_cb),
            ("mcast_dst_num_pages", mcast_dst_num_pages),
            # Matmul reader args (for sharded buffer setup)
            ("matmul_in0", matmul_input_cb),
            ("matmul_in1", matmul_weights_cb),
            ("matmul_k_num_tiles", num_tiles_k),
            ("matmul_out_w", matmul_out_w),
        ]

        # Named compile-time args for BRISC (mcast sender)
        brisc_named_compile_time_args = [
            # Mcast sender args
            ("mcast_dest_noc_start_x", mcast_dest_noc_start_core.x),
            ("mcast_dest_noc_start_y", mcast_dest_noc_start_core.y),
            ("mcast_dest_noc_end_x", mcast_dest_noc_end_core.x),
            ("mcast_dest_noc_end_y", mcast_dest_noc_end_core.y),
            ("mcast_num_cores", mcast_num_cores),
            ("mcast_data_sender_semaphore", mcast_data_sender_semaphore_id),
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_data_size_bytes", mcast_data_size_bytes),
            ("mcast_src_cb", input_cb),
            ("mcast_dst_cb", matmul_input_cb),
            ("mcast_src_num_pages", mcast_src_num_pages),
            ("mcast_is_part_of_receiver_grid", mcast_is_part_of_receiver_grid),
        ]

        # Named compile-time args for TRISC (matmul + sigmoid compute)
        trisc_named_compile_time_args = [
            ("matmul_in0", matmul_input_cb),
            ("matmul_in1", matmul_weights_cb),
            ("matmul_out", matmul_output_cb),
            ("matmul_k_num_tiles", num_tiles_k),
            ("matmul_out_w", matmul_out_w),
            ("matmul_fused_activation", MoeRoutedExpert.ACTIVATION_SIGMOID),
        ]

        # Semaphore descriptors
        full_device_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )

        mcast_sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=mcast_data_sender_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        mcast_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=mcast_data_receiver_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        # Unified kernel descriptor
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/fused_ops/moe_routed_expert/moe_routed_expert_kernel.cpp",
            core_ranges=full_device_grid,
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
                    named_compile_time_arg="is_mcast_grid_core",
                    core_range=mcast_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_matmul_core",
                    core_range=matmul_weights_core_grid,
                    value=1,
                    other_value=0,
                ),
            ],
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors(),
            cbs=[
                input_cb_descriptor,
                matmul_input_cb_descriptor,
                matmul_weights_cb_descriptor,
                matmul_output_cb_descriptor,
            ],
            semaphores=[
                mcast_sender_semaphore_descriptor,
                mcast_receiver_semaphore_descriptor,
            ],
        )

        # Execute generic op
        io_tensors = [input_tensor, matmul_weights_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
