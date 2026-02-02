# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
MoE Routed Expert fused operation.

This implements the MoE routed expert computation:
- Input: [1, K] tensor on sender core (outside compute grid)
- Mcast to N compute cores
- Each core computes: matmul + activation
- Gather outputs back to sender core
- Gate: top-8 expert selection with normalized scores
- Output: top8 scores [1, 16] + top8 indices [1, 16] on sender core
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
    def golden(input_tensor, matmul_weights_tensor, bias_tensor):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) [1, K]
            matmul_weights_tensor: Matmul weights (torch.Tensor) [K, N]
            bias_tensor: Gate bias tensor (torch.Tensor) [1, N] or [16, 16]

        Returns:
            Tuple of (top8_scores, top8_indices) tensors
        """
        import torch

        # Matmul + sigmoid
        logits = input_tensor @ matmul_weights_tensor
        scores = torch.sigmoid(logits)

        # Add bias
        scores_with_bias = scores.view(-1) + bias_tensor.view(-1)

        # Top-8 selection
        top8_scores, top8_indices = torch.topk(scores_with_bias, k=8)

        # Normalize scores (softmax over top-8)
        top8_scores = torch.softmax(top8_scores.float(), dim=-1)

        return top8_scores, top8_indices

    @staticmethod
    def op(
        input_tensor,
        matmul_weights_tensor,
        intermediate_tensor,
        gate_input_tensor,
        gate_bias_tensor,
        gate_indices_tensor,
        gate_output_scores_tensor,
        gate_output_indices_tensor,
        fp32_dest_acc_en=True,
    ):
        """
        Execute mcast + matmul + sigmoid + gather + gate fused operation using generic_op.

        Args:
            input_tensor: Input tensor [1, K] sharded on single sender core (outside matmul grid)
            matmul_weights_tensor: Matmul weights [K, N] width-sharded on matmul cores
            intermediate_tensor: Intermediate tensor [1, N] width-sharded on matmul cores (matmul output)
            gate_input_tensor: Gate input tensor [16, 16] on sender core (receives gathered matmul output)
            gate_bias_tensor: Gate bias tensor [16, 16] on sender core
            gate_indices_tensor: Gate indices tensor [16, 16] on sender core
            gate_output_scores_tensor: Gate output scores tensor [1, 16] on sender core
            gate_output_indices_tensor: Gate output indices tensor [1, 16] on sender core
            fp32_dest_acc_en: Whether to enable FP32 accumulation (default True for precision)

        Returns:
            Tuple of (gate_output_scores_tensor, gate_output_indices_tensor)
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
        matmul_output_cb = 3  # Matmul output (intermediate on compute cores)
        gate_input_cb = 4  # Gate input (gathered output, tensor-backed on sender core)
        gate_bias_cb = 5  # Gate bias (tensor-backed on sender core)
        gate_indices_cb = 6  # Gate indices (tensor-backed on sender core)
        gate_output_cb = 7  # Gate output scores (tensor-backed on sender core)
        gate_output_indices_cb = 8  # Gate output indices (tensor-backed on sender core)

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

        # CB 3: Matmul output (intermediate on compute cores, tensor-backed)
        matmul_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_output_cb, intermediate_tensor)

        # CB 4: Gate input (gathered matmul output, tensor-backed on sender core)
        gate_input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gate_input_cb, gate_input_tensor)

        # CB 5: Gate bias (tensor-backed on sender core)
        gate_bias_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gate_bias_cb, gate_bias_tensor)

        # CB 6: Gate indices (tensor-backed on sender core)
        gate_indices_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gate_indices_cb, gate_indices_tensor)

        # CB 7: Gate output scores (tensor-backed on sender core)
        gate_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gate_output_cb, gate_output_scores_tensor)

        # CB 8: Gate output indices (tensor-backed on sender core)
        gate_output_indices_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            gate_output_indices_cb, gate_output_indices_tensor
        )

        # Mcast page counts
        mcast_src_num_pages = num_tiles_k
        mcast_dst_num_pages = num_tiles_k

        # Gather parameters
        # Sender core NOC coordinates (receiver for gather)
        sender_core_noc = device.worker_core_from_logical_core(sender_core)

        # Gather data size: each compute core sends matmul_out_w tiles
        intermediate_tile = intermediate_tensor.get_tile()
        intermediate_tile_size = intermediate_tile.get_tile_size(data_format)
        gather_data_size_bytes = matmul_out_w * intermediate_tile_size

        # Number of senders per NOC (for semaphore counting)
        # All compute cores are senders, split by NOC
        # For simplicity, assume all senders use noc0 (row_major=true)
        gather_noc0_num_senders = num_matmul_cores
        gather_noc1_num_senders = 0

        # Gather semaphore IDs
        gather_noc0_receiver_semaphore_id = 2
        gather_noc1_receiver_semaphore_id = 3

        # Gather output: 1 tile of 16x16 (same 256 elements as 8 tiles of 1x32)
        gather_dst_num_pages = 1

        # Gather receiver data address (L1 address of gate input tensor buffer on sender core)
        gather_receiver_data_addr = gate_input_tensor.buffer_address()

        # Named compile-time args for NCRISC (mcast receiver + matmul reader + gather sender)
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
            # Gather sender args (compute cores send to sender core)
            ("gather_dest_noc_x", sender_core_noc.x),
            ("gather_dest_noc_y", sender_core_noc.y),
            ("gather_data_size_bytes", gather_data_size_bytes),
            ("gather_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("gather_src_cb", matmul_output_cb),
            ("gather_src_num_pages", matmul_out_w),
            ("gather_sender_grid_start_x", matmul_start.x),
            ("gather_sender_grid_start_y", matmul_start.y),
            ("gather_sender_grid_end_x", matmul_end.x),
            ("gather_sender_grid_end_y", matmul_end.y),
            ("gather_row_major", 0),  # Column-major grid
            ("gather_receiver_data_addr", gather_receiver_data_addr),
            # Gate reader args (sender core)
            ("gate_input_cb", gate_input_cb),
            ("gate_bias_cb", gate_bias_cb),
            ("gate_input_indices_cb", gate_indices_cb),
        ]

        # Named compile-time args for BRISC (mcast sender + gather receiver)
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
            # Gather receiver args (sender core receives from compute cores)
            ("gather_noc0_num_senders", gather_noc0_num_senders),
            ("gather_noc1_num_senders", gather_noc1_num_senders),
            ("gather_noc0_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("gather_noc1_receiver_semaphore_id", gather_noc1_receiver_semaphore_id),
            ("gather_dst_cb", gate_input_cb),
            ("gather_dst_num_pages", gather_dst_num_pages),
            # Gate writer args (sender core)
            ("gate_output_cb", gate_output_cb),
            ("gate_output_indices_cb", gate_output_indices_cb),
        ]

        # Gate parameters (eps and scaling_factor as uint32 bit patterns)
        # Use little-endian format to match float_to_uint32 in utils.py
        import struct

        gate_eps = int.from_bytes(struct.pack("f", 1e-20), byteorder="little")
        gate_scaling_factor = int.from_bytes(struct.pack("f", 2.5), byteorder="little")
        gate_enable_sigmoid = 0  # Sigmoid already done in matmul

        # Named compile-time args for TRISC (matmul + sigmoid compute + gate compute)
        trisc_named_compile_time_args = [
            ("matmul_in0", matmul_input_cb),
            ("matmul_in1", matmul_weights_cb),
            ("matmul_out", matmul_output_cb),
            ("matmul_k_num_tiles", num_tiles_k),
            ("matmul_out_w", matmul_out_w),
            ("matmul_fused_activation", MoeRoutedExpert.ACTIVATION_SIGMOID),
            # Gate compute args (sender core)
            ("gate_input_cb", gate_input_cb),
            ("gate_bias_cb", gate_bias_cb),
            ("gate_input_indices_cb", gate_indices_cb),
            ("gate_output_cb", gate_output_cb),
            ("gate_output_indices_cb", gate_output_indices_cb),
            ("gate_eps", gate_eps),
            ("gate_scaling_factor", gate_scaling_factor),
            ("gate_enable_sigmoid", gate_enable_sigmoid),
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

        gather_noc0_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=gather_noc0_receiver_semaphore_id,
            core_ranges=full_device_grid,
            initial_value=0,
        )

        gather_noc1_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=gather_noc1_receiver_semaphore_id,
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
                # fp32_dest_acc_en disabled because gate's transpose doesn't support it
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
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
                gate_input_cb_descriptor,
                gate_bias_cb_descriptor,
                gate_indices_cb_descriptor,
                gate_output_cb_descriptor,
                gate_output_indices_cb_descriptor,
            ],
            semaphores=[
                mcast_sender_semaphore_descriptor,
                mcast_receiver_semaphore_descriptor,
                gather_noc0_semaphore_descriptor,
                gather_noc1_semaphore_descriptor,
            ],
        )

        # Execute generic op
        io_tensors = [
            input_tensor,
            matmul_weights_tensor,
            intermediate_tensor,
            gate_input_tensor,
            gate_bias_tensor,
            gate_indices_tensor,
            gate_output_scores_tensor,
            gate_output_indices_tensor,
        ]
        ttnn.generic_op(io_tensors, program_descriptor)

        return gate_output_scores_tensor, gate_output_indices_tensor
