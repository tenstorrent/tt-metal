# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
MoE Routed Expert fused operation.

This implements the MoE routed expert computation:
1. Input: [1, K] tensor on sender core (outside compute grid)
2. Mcast input to N compute cores
3. Each core computes: routing matmul + sigmoid
4. Gather outputs back to sender core
5. Gate: top-K expert selection with normalized scores
6. Mcast expert indices to compute cores
7. DRAM streaming matmul + SiLU with indexed expert weights
8. Output: expert computation result [1, N] on compute cores
"""

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


def get_max_page_size_and_num_pages(device, num_tiles, tile_size):
    """
    Calculate optimal page size and number of pages for NOC transfers.

    The NOC has a maximum burst size that varies by architecture:
    - Wormhole: 8192 bytes
    - Blackhole: 16384 bytes
    """
    total_size = num_tiles * tile_size

    arch = device.arch()
    if arch == ttnn.device.Arch.WORMHOLE_B0:
        noc_max_page_size = 8192
    elif arch == ttnn.device.Arch.BLACKHOLE:
        noc_max_page_size = 16384
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    page_size = (noc_max_page_size // tile_size) * tile_size
    while total_size % page_size != 0 and page_size >= tile_size:
        page_size -= tile_size

    num_pages = total_size // page_size
    return page_size, num_pages


class MoeRoutedExpert:
    """
    MoE Routed Expert fused operation implementation using ttnn.generic_op.
    """

    # Fused activation enum values (must match matmul.hpp FusedActivation enum)
    ACTIVATION_NONE = 0
    ACTIVATION_SIGMOID = 1
    ACTIVATION_SILU = 2

    @staticmethod
    def golden(
        input_tensor,
        routing_weights_tensor,
        bias_tensor,
        expert_weights_dict=None,
        eps=1e-20,
        scaling_factor=2.5,
    ):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) [1, K]
            routing_weights_tensor: Routing matmul weights (torch.Tensor) [K, N_routing]
            bias_tensor: Gate bias tensor (torch.Tensor) [1, 8, 32] or [16, 16]
            expert_weights_dict: Dict mapping expert_idx -> weights [1, 1, K, N_expert] (optional)
            eps: Epsilon for numerical stability in gate
            scaling_factor: Scaling factor for gate scores

        Returns:
            If expert_weights_dict is None:
                Tuple of (top8_scores, top8_indices) tensors
            Else:
                Tuple of (top8_scores, top8_indices, expert_output) tensors
        """
        import torch

        from models.demos.deepseek_v3_b1.micro_ops.deepseek_moe_gate.op import DeepseekMoeGateSingleCore

        # 1. Routing matmul + sigmoid
        logits = input_tensor.float() @ routing_weights_tensor.float()
        scores = torch.sigmoid(logits)

        # 2. Gate: top-8 selection with normalized scores
        # Reshape to (1, 8, 32) for gate golden
        gate_input = scores.reshape(1, 8, 32)
        top8_scores, top8_indices = DeepseekMoeGateSingleCore.golden(
            gate_input, bias_tensor.float(), eps, scaling_factor, enable_sigmoid=False
        )

        # 3. Expert matmul + SiLU (if expert weights provided)
        if expert_weights_dict is not None:
            # Get the first selected expert index
            selected_expert_idx = int(top8_indices[0, 0].item())
            selected_expert_weights = expert_weights_dict[selected_expert_idx]

            # Compute: input @ expert_weights + SiLU
            input_for_expert = input_tensor.reshape(1, 1, 1, -1).float()
            expert_output = input_for_expert @ selected_expert_weights.float()
            expert_output = torch.nn.functional.silu(expert_output)

            return top8_scores, top8_indices, expert_output

        return top8_scores, top8_indices

    @staticmethod
    def op(
        input_tensor,
        mcast_output_tensor,
        gate_mm_weights_tensor,
        gate_mm_output_tensor,
        gate_input_tensor,
        gate_bias_tensor,
        gate_indices_tensor,
        gate_output_scores_tensor,
        gate_output_indices_tensor,
        dram_mm_silu_index_tensor,
        dram_mm_silu_weights_tensor,
        dram_mm_silu_output_tensor,
    ):
        """
        Execute mcast + matmul + sigmoid + gather + gate + index_mcast + dram_matmul + silu fused operation.

        Args:
            input_tensor: Input tensor [1, K] sharded on single sender core (outside matmul grid)
            mcast_output_tensor: Mcast output tensor [1, K] sharded on all cores that need input
                                 (routing matmul cores + DRAM matmul cores)
            gate_mm_weights_tensor: Gate matmul weights [K, N_routing] width-sharded on matmul cores
            gate_mm_output_tensor: Gate matmul output [1, N_routing] width-sharded (routing matmul output)
            gate_input_tensor: Gate input tensor [16, 16] on sender core (receives gathered output)
            gate_bias_tensor: Gate bias tensor [16, 16] on sender core
            gate_indices_tensor: Gate indices tensor [16, 16] on sender core
            gate_output_scores_tensor: Gate output scores tensor [1, 16] on sender core
            gate_output_indices_tensor: Gate output indices tensor [1, 16] on sender core
            dram_mm_silu_index_tensor: DRAM matmul+SiLU index tensor [1, 16] on mcast grid (receives mcasted indices)
            dram_mm_silu_weights_tensor: Expert weights [K, N_expert] width-sharded in DRAM (first expert)
            dram_mm_silu_output_tensor: Expert matmul+SiLU output [1, N_expert] width-sharded on DRAM matmul cores

        Returns:
            Tuple of (gate_output_scores_tensor, gate_output_indices_tensor, dram_mm_silu_output_tensor)
        """
        # Hardcoded parameters
        fp32_dest_acc_en = False  # Gate transpose doesn't support fp32
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
        gate_mm_weights_memory_config = gate_mm_weights_tensor.memory_config()
        gate_mm_weights_core_grid = gate_mm_weights_memory_config.shard_spec.grid
        num_gate_mm_cores = gate_mm_weights_core_grid.num_cores()

        # Get per-core output width in tiles
        gate_mm_weights_tile = gate_mm_weights_tensor.get_tile()
        gate_mm_weights_shard_shape = gate_mm_weights_memory_config.shard_spec.shape
        gate_mm_weights_shard_width = gate_mm_weights_shard_shape[1]
        gate_mm_out_w = gate_mm_weights_shard_width // gate_mm_weights_tile.tile_shape[1]

        # Get device and compute grid
        device = input_tensor.device()
        device_grid_size = device.compute_with_storage_grid_size()

        # Get DRAM matmul cores from output tensor's shard spec
        dram_mm_silu_output_memory_config = dram_mm_silu_output_tensor.memory_config()
        dram_mm_silu_core_ranges = dram_mm_silu_output_memory_config.shard_spec.grid

        # Get matmul grid bounds (for gather sender grid)
        gate_mm_grid_ranges = list(gate_mm_weights_core_grid.ranges())
        gate_mm_grid_range = gate_mm_grid_ranges[0]
        gate_mm_start = gate_mm_grid_range.start
        gate_mm_end = gate_mm_grid_range.end

        # Mcast grid: use the mcast_output_tensor's rectangular grid directly
        # The tensor is already sharded on a rectangle from (0,0) to sender core
        mcast_output_core_grid = mcast_output_tensor.memory_config().shard_spec.grid
        mcast_output_ranges = list(mcast_output_core_grid.ranges())
        mcast_grid_range = mcast_output_ranges[0]  # Single rectangular range
        mcast_grid_start = mcast_grid_range.start
        mcast_grid_end = mcast_grid_range.end
        mcast_grid = mcast_output_core_grid

        # Get NOC coordinates for mcast destination
        mcast_dest_noc_start_core = device.worker_core_from_logical_core(mcast_grid_start)
        mcast_dest_noc_end_core = device.worker_core_from_logical_core(mcast_grid_end)

        # Calculate mcast parameters
        mcast_num_cores = (mcast_grid_end.x - mcast_grid_start.x + 1) * (mcast_grid_end.y - mcast_grid_start.y + 1)
        mcast_is_part_of_receiver_grid = mcast_grid.contains(sender_core_grid)

        # Semaphore IDs
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1

        # Calculate mcast data size in bytes
        mcast_data_size_bytes = num_tiles_k * tile_1x32_size

        # CB indices
        input_cb = 0  # Input tensor (sharded on sender core)
        gate_mm_input_cb = 1  # Mcast destination CB (receives input on all cores) - also used as dram_mm_silu_cb_in0
        gate_mm_weights_cb = 2  # Matmul weights (sharded on all cores)
        gate_mm_output_cb = 3  # Matmul output (intermediate on compute cores)
        gate_input_cb = 4  # Gate input (gathered output, tensor-backed on sender core)
        gate_bias_cb = 5  # Gate bias (tensor-backed on sender core)
        gate_indices_cb = 6  # Gate indices (tensor-backed on sender core)
        gate_output_cb = 7  # Gate output scores (tensor-backed on sender core)
        gate_output_indices_cb = 8  # Gate output indices (tensor-backed on sender core)
        dram_mm_silu_cb_in1 = 9  # DRAM matmul weights working buffer
        dram_mm_silu_cb_index = 10  # DRAM matmul index (receives mcasted indices)
        dram_mm_silu_cb_out = 11  # DRAM matmul output (tensor-backed on compute cores)

        # CB descriptors
        # CB 0: Input tensor (sharded on sender core)
        input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)

        # CB 1: Mcast destination (tensor-backed, sharded on all cores that need input)
        gate_mm_input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gate_mm_input_cb, mcast_output_tensor)

        # CB 2: Matmul weights (sharded on all matmul cores)
        gate_mm_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            gate_mm_weights_cb, gate_mm_weights_tensor
        )

        # CB 3: Matmul output (intermediate on compute cores, tensor-backed)
        gate_mm_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gate_mm_output_cb, gate_mm_output_tensor)

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

        # ========== DRAM Streaming Matmul CBs ==========
        # Get DRAM matmul parameters from weights tensor
        dram_mm_silu_weights_tile = dram_mm_silu_weights_tensor.get_tile()
        dram_mm_silu_weights_dtype = dram_mm_silu_weights_tensor.dtype
        dram_mm_silu_weights_tile_size = dram_mm_silu_weights_tile.get_tile_size(dram_mm_silu_weights_dtype)
        dram_mm_silu_weights_shard_shape = dram_mm_silu_weights_tensor.memory_config().shard_spec.shape
        dram_mm_silu_K = dram_mm_silu_weights_shard_shape[0]
        dram_mm_silu_per_core_n = dram_mm_silu_weights_shard_shape[1] // dram_mm_silu_weights_tile.tile_shape[1]
        dram_mm_silu_Kt = dram_mm_silu_K // dram_mm_silu_weights_tile.tile_shape[0]

        # Use 4 subblocks for DRAM matmul (num_in1_buffers = 3 * 4 = 12 <= 15)
        dram_mm_silu_num_subblocks_k = 4
        dram_mm_silu_subblock_k = dram_mm_silu_Kt // dram_mm_silu_num_subblocks_k
        assert (
            dram_mm_silu_Kt % dram_mm_silu_num_subblocks_k == 0
        ), f"dram_mm_silu_Kt ({dram_mm_silu_Kt}) must be divisible by num_subblocks ({dram_mm_silu_num_subblocks_k})"

        # Calculate page size for NOC transfers
        dram_mm_silu_in1_page_size, dram_mm_silu_in1_num_pages = get_max_page_size_and_num_pages(
            device, dram_mm_silu_subblock_k, dram_mm_silu_weights_tile_size
        )
        dram_mm_silu_in1_block_size_bytes = dram_mm_silu_subblock_k * dram_mm_silu_weights_tile_size

        # CB 9: DRAM matmul weights working buffer
        num_in1_buffers = 3 * dram_mm_silu_num_subblocks_k
        assert num_in1_buffers <= 15, f"num_in1_buffers ({num_in1_buffers}) exceeds NOC_MAX_TRANSACTION_ID (15)"
        dram_mm_silu_in1_CB_tiles = dram_mm_silu_subblock_k * num_in1_buffers
        dram_mm_silu_in1_CB_size = dram_mm_silu_in1_CB_tiles * dram_mm_silu_weights_tile_size

        dram_mm_silu_weights_tile_desc = ttnn.TileDescriptor(dram_mm_silu_weights_tile)
        dram_mm_silu_cb_in1_format = ttnn.CBFormatDescriptor(
            buffer_index=dram_mm_silu_cb_in1,
            data_format=dram_mm_silu_weights_dtype,
            page_size=dram_mm_silu_weights_tile_size,
            tile=dram_mm_silu_weights_tile_desc,
        )
        dram_mm_silu_cb_in1_descriptor = ttnn.CBDescriptor(
            total_size=dram_mm_silu_in1_CB_size,
            core_ranges=dram_mm_silu_core_ranges,
            format_descriptors=[dram_mm_silu_cb_in1_format],
        )

        # CB 10: DRAM matmul index (receives mcasted indices from gate)
        # Tensor-backed for consistent addresses across all cores in mcast grid
        dram_mm_silu_cb_index_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            dram_mm_silu_cb_index, dram_mm_silu_index_tensor
        )

        # Get index tile size for compile-time args
        index_tile = dram_mm_silu_index_tensor.get_tile()
        index_dtype = dram_mm_silu_index_tensor.dtype
        index_tile_size = index_tile.get_tile_size(index_dtype)

        # CB 11: DRAM matmul output (tensor-backed on compute cores)
        dram_mm_silu_cb_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            dram_mm_silu_cb_out, dram_mm_silu_output_tensor
        )

        # DRAM matmul output tiles
        dram_mm_silu_out_num_tiles = dram_mm_silu_per_core_n

        # DRAM matmul tile dimensions
        dram_mm_silu_tile_r_dim = dram_mm_silu_weights_tile.tile_shape[0]

        # Calculate dram_mm_silu_subblock_w based on fp32_dest_acc_en and per_core_N
        if fp32_dest_acc_en:
            if dram_mm_silu_per_core_n <= 8:
                max_subblock_w = 8
            else:
                max_subblock_w = 4
        else:
            if dram_mm_silu_per_core_n <= 16:
                max_subblock_w = 16
            else:
                max_subblock_w = 8

        dram_mm_silu_subblock_w = max_subblock_w
        while dram_mm_silu_subblock_w > 1 and dram_mm_silu_per_core_n % dram_mm_silu_subblock_w != 0:
            dram_mm_silu_subblock_w -= 1

        # Index mcast parameters - reuse same semaphores as input mcast
        index_mcast_sender_semaphore_id = mcast_data_sender_semaphore_id
        index_mcast_receiver_semaphore_id = mcast_data_receiver_semaphore_id
        index_mcast_num_pages = 1  # Single index tile
        index_mcast_data_size_bytes = index_tile_size

        # DRAM matmul buffer address (first expert tensor)
        dram_mm_silu_in1_tensor_addr = dram_mm_silu_weights_tensor.buffer_address()

        # Mcast page counts
        mcast_src_num_pages = num_tiles_k
        mcast_dst_num_pages = num_tiles_k

        # Gather parameters
        # Sender core NOC coordinates (receiver for gather)
        sender_core_noc = device.worker_core_from_logical_core(sender_core)

        # Gather data size: each compute core sends gate_mm_out_w tiles
        gate_mm_output_tile = gate_mm_output_tensor.get_tile()
        gate_mm_output_tile_size = gate_mm_output_tile.get_tile_size(data_format)
        gather_data_size_bytes = gate_mm_out_w * gate_mm_output_tile_size

        # Number of senders per NOC (for semaphore counting)
        # All compute cores are senders, split by NOC
        # For simplicity, assume all senders use noc0 (row_major=true)
        gather_noc0_num_senders = num_gate_mm_cores
        gather_noc1_num_senders = 0

        # Gather semaphore IDs
        gather_noc0_receiver_semaphore_id = 2
        gather_noc1_receiver_semaphore_id = 3

        # Gather output: 1 tile of 16x16 (same 256 elements as 8 tiles of 1x32)
        gather_dst_num_pages = 1

        # Gather receiver data address (L1 address of gate input tensor buffer on sender core)
        gather_receiver_data_addr = gate_input_tensor.buffer_address()

        # Named compile-time args for NCRISC (mcast receiver + matmul reader + gather sender + index mcast receiver)
        ncrisc_named_compile_time_args = [
            # Mcast sender sharded buffer setup (for sender core)
            ("mcast_src_cb", input_cb),
            ("mcast_src_num_pages", mcast_src_num_pages),
            # Mcast receiver args
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_dst_cb", gate_mm_input_cb),
            ("mcast_dst_num_pages", mcast_dst_num_pages),
            # Matmul reader args (for sharded buffer setup)
            ("gate_mm_in0", gate_mm_input_cb),
            ("gate_mm_in1", gate_mm_weights_cb),
            ("gate_mm_k_num_tiles", num_tiles_k),
            ("gate_mm_out_w", gate_mm_out_w),
            # Gather sender args (compute cores send to sender core)
            ("gather_dest_noc_x", sender_core_noc.x),
            ("gather_dest_noc_y", sender_core_noc.y),
            ("gather_data_size_bytes", gather_data_size_bytes),
            ("gather_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("gather_src_cb", gate_mm_output_cb),
            ("gather_src_num_pages", gate_mm_out_w),
            ("gather_sender_grid_start_x", gate_mm_start.x),
            ("gather_sender_grid_start_y", gate_mm_start.y),
            ("gather_sender_grid_end_x", gate_mm_end.x),
            ("gather_sender_grid_end_y", gate_mm_end.y),
            ("gather_row_major", 0),  # Column-major grid
            ("gather_receiver_data_addr", gather_receiver_data_addr),
            # Gate reader args (sender core)
            ("gate_input_cb", gate_input_cb),
            ("gate_bias_cb", gate_bias_cb),
            ("gate_input_indices_cb", gate_indices_cb),
            # Index mcast receiver args (compute cores)
            ("index_mcast_receiver_semaphore", index_mcast_receiver_semaphore_id),
            ("dram_mm_silu_cb_index", dram_mm_silu_cb_index),
            ("index_mcast_num_pages", index_mcast_num_pages),
        ]

        # Named compile-time args for BRISC (mcast sender + gather receiver + index mcast sender + dram matmul writer)
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
            ("mcast_dst_cb", gate_mm_input_cb),
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
            # Index mcast sender args (sender core broadcasts gate indices to compute cores)
            ("index_mcast_sender_semaphore", index_mcast_sender_semaphore_id),
            ("index_mcast_receiver_semaphore", index_mcast_receiver_semaphore_id),
            ("index_mcast_data_size_bytes", index_mcast_data_size_bytes),
            ("index_mcast_num_pages", index_mcast_num_pages),
            ("dram_mm_silu_cb_index", dram_mm_silu_cb_index),
            # DRAM streaming matmul writer args (compute cores)
            ("dram_mm_silu_cb_in1", dram_mm_silu_cb_in1),
            ("dram_mm_silu_cb_out", dram_mm_silu_cb_out),
            ("dram_mm_silu_in1_tensor_addr", dram_mm_silu_in1_tensor_addr),
            ("dram_mm_silu_in1_page_size", dram_mm_silu_in1_page_size),
            ("dram_mm_silu_in1_num_pages", dram_mm_silu_in1_num_pages),
            ("dram_mm_silu_subblock_k", dram_mm_silu_subblock_k),
            ("dram_mm_silu_per_core_n", dram_mm_silu_per_core_n),
            ("dram_mm_silu_in1_block_size_bytes", dram_mm_silu_in1_block_size_bytes),
            ("dram_mm_silu_out_num_tiles", dram_mm_silu_out_num_tiles),
            ("dram_mm_silu_num_subblocks_k", dram_mm_silu_num_subblocks_k),
            ("dram_mm_silu_index_offset", 0),  # Always use first index from the mcasted index tensor
        ]

        # Gate parameters (eps and scaling_factor as uint32 bit patterns)
        # Use little-endian format to match float_to_uint32 in utils.py
        import struct

        gate_eps = int.from_bytes(struct.pack("f", 1e-20), byteorder="little")
        gate_scaling_factor = int.from_bytes(struct.pack("f", 2.5), byteorder="little")
        gate_enable_sigmoid = 0  # Sigmoid already done in matmul

        # Named compile-time args for TRISC (matmul + sigmoid compute + gate compute + dram matmul compute)
        trisc_named_compile_time_args = [
            ("gate_mm_in0", gate_mm_input_cb),
            ("gate_mm_in1", gate_mm_weights_cb),
            ("gate_mm_out", gate_mm_output_cb),
            ("gate_mm_k_num_tiles", num_tiles_k),
            ("gate_mm_out_w", gate_mm_out_w),
            ("gate_mm_fused_activation", MoeRoutedExpert.ACTIVATION_SIGMOID),
            # Gate compute args (sender core)
            ("gate_input_cb", gate_input_cb),
            ("gate_bias_cb", gate_bias_cb),
            ("gate_input_indices_cb", gate_indices_cb),
            ("gate_output_cb", gate_output_cb),
            ("gate_output_indices_cb", gate_output_indices_cb),
            ("gate_eps", gate_eps),
            ("gate_scaling_factor", gate_scaling_factor),
            ("gate_enable_sigmoid", gate_enable_sigmoid),
            # DRAM streaming matmul compute args (compute cores)
            ("dram_mm_silu_cb_in0", gate_mm_input_cb),  # Reuses mcasted input
            ("dram_mm_silu_cb_in1", dram_mm_silu_cb_in1),
            ("dram_mm_silu_cb_out", dram_mm_silu_cb_out),
            ("dram_mm_silu_subblock_k", dram_mm_silu_subblock_k),
            ("dram_mm_silu_per_core_n", dram_mm_silu_per_core_n),
            ("dram_mm_silu_subblock_w", dram_mm_silu_subblock_w),
            ("dram_mm_silu_num_subblocks_k", dram_mm_silu_num_subblocks_k),
            ("dram_mm_silu_tile_r_dim", dram_mm_silu_tile_r_dim),
            ("dram_mm_silu_fuse_silu", 1),  # Always use SiLU for expert computation
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

        # Note: index mcast reuses the same semaphores as input mcast (IDs 0 and 1)

        # Per-core compile-time args for DRAM matmul bank_id and vc
        # Each DRAM matmul core gets a unique bank_id based on its optimal DRAM bank assignment
        # IMPORTANT: Use get_optimal_dram_bank_to_logical_worker_assignment() to get the correct
        # core-to-bank mapping. Do NOT use corerange_to_cores() iteration order as bank_id!
        dram_mm_silu_optimal_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        # Build mapping from core to bank_id based on optimal assignment
        core_to_bank_id = {}
        for bank_id, core in enumerate(dram_mm_silu_optimal_workers):
            core_to_bank_id[(core.x, core.y)] = bank_id
        bank_id_core_values = []
        vc_core_values = []
        for core in dram_mm_silu_optimal_workers:
            bank_id = core_to_bank_id[(core.x, core.y)]
            vc = bank_id & 0x3
            bank_id_core_values.append((core, bank_id))
            vc_core_values.append((core, vc))

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
                    named_compile_time_arg="is_gate_mm_core",
                    core_range=gate_mm_weights_core_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_dram_mm_silu_core",
                    core_range=dram_mm_silu_core_ranges,
                    value=1,
                    other_value=0,
                ),
            ],
            per_core_compile_time_descriptors=[
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="dram_mm_silu_bank_id",
                    core_values=bank_id_core_values,
                    other_value=0,
                ),
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="dram_mm_silu_vc",
                    core_values=vc_core_values,
                    other_value=0,
                ),
            ],
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors(),
            cbs=[
                input_cb_descriptor,
                gate_mm_input_cb_descriptor,
                gate_mm_weights_cb_descriptor,
                gate_mm_output_cb_descriptor,
                gate_input_cb_descriptor,
                gate_bias_cb_descriptor,
                gate_indices_cb_descriptor,
                gate_output_cb_descriptor,
                gate_output_indices_cb_descriptor,
                dram_mm_silu_cb_in1_descriptor,
                dram_mm_silu_cb_index_descriptor,
                dram_mm_silu_cb_out_descriptor,
            ],
            semaphores=[
                mcast_sender_semaphore_descriptor,
                mcast_receiver_semaphore_descriptor,
                gather_noc0_semaphore_descriptor,
                gather_noc1_semaphore_descriptor,
                # Note: index mcast reuses semaphores 0 and 1
            ],
        )

        # Execute generic op
        io_tensors = [
            input_tensor,
            mcast_output_tensor,
            gate_mm_weights_tensor,
            gate_mm_output_tensor,
            gate_input_tensor,
            gate_bias_tensor,
            gate_indices_tensor,
            gate_output_scores_tensor,
            gate_output_indices_tensor,
            dram_mm_silu_index_tensor,
            dram_mm_silu_weights_tensor,
            dram_mm_silu_output_tensor,
        ]
        ttnn.generic_op(io_tensors, program_descriptor)

        return gate_output_scores_tensor, gate_output_indices_tensor, dram_mm_silu_output_tensor
