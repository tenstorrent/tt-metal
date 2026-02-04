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

import math

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


def setup_eltwise_mul(
    in0_tensor,
    in1_tensor,
    out_tensor,
    core_ranges,
    cb_in0_index,
    cb_in1_index,
    cb_out_index,
    per_core_n,
):
    """
    Set up parameters and CB descriptors for element-wise multiply with CB aliasing.

    The mul operation uses 16x16 tiles while the input tensors use 1x32 tiles.
    This function creates aliased CB descriptors that view the same memory with 16x16 tile format.

    Args:
        in0_tensor: First input tensor (e.g., up_proj matmul output)
        in1_tensor: Second input tensor (e.g., gate_proj matmul output)
        out_tensor: Output tensor for fused result
        core_ranges: CoreRangeSet for compute cores
        cb_in0_index: CB index for first input (aliased)
        cb_in1_index: CB index for second input (aliased)
        cb_out_index: CB index for output
        per_core_n: Number of output tiles per core (in 1x32 format)

    Returns:
        Dictionary with mul_num_tiles and CB descriptors
    """
    # Compute mul_num_tiles: total elements / 256 (16x16 tile)
    M = 1  # Output row dim
    tile_width = 32  # 1x32 tiles from matmul
    total_elements = M * per_core_n * tile_width
    mul_num_tiles = math.ceil(total_elements / 256)

    # 16x16 tile format for mul operation
    TILE_16x16 = ttnn.Tile((16, 16))
    tile_16x16_size = TILE_16x16.get_tile_size(ttnn.bfloat16)
    tile_16x16_desc = ttnn.TileDescriptor(TILE_16x16)

    # CB for in0: alias of in0_tensor with 16x16 tile format
    cb_in0_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_in0_index,
        data_format=ttnn.bfloat16,
        page_size=tile_16x16_size,
        tile=tile_16x16_desc,
    )
    cb_in0_descriptor = ttnn.CBDescriptor(
        total_size=mul_num_tiles * tile_16x16_size,
        core_ranges=core_ranges,
        format_descriptors=[cb_in0_format],
    )
    cb_in0_descriptor.set_buffer_from_tensor(in0_tensor)

    # CB for in1: alias of in1_tensor with 16x16 tile format
    cb_in1_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_in1_index,
        data_format=ttnn.bfloat16,
        page_size=tile_16x16_size,
        tile=tile_16x16_desc,
    )
    cb_in1_descriptor = ttnn.CBDescriptor(
        total_size=mul_num_tiles * tile_16x16_size,
        core_ranges=core_ranges,
        format_descriptors=[cb_in1_format],
    )
    cb_in1_descriptor.set_buffer_from_tensor(in1_tensor)

    # CB for out: output with 16x16 tile format (tensor-backed)
    cb_out_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_out_index,
        data_format=ttnn.bfloat16,
        page_size=tile_16x16_size,
        tile=tile_16x16_desc,
    )
    cb_out_descriptor = ttnn.CBDescriptor(
        total_size=mul_num_tiles * tile_16x16_size,
        core_ranges=core_ranges,
        format_descriptors=[cb_out_format],
    )
    cb_out_descriptor.set_buffer_from_tensor(out_tensor)

    return {
        "mul_num_tiles": mul_num_tiles,
        "cb_in0_descriptor": cb_in0_descriptor,
        "cb_in1_descriptor": cb_in1_descriptor,
        "cb_out_descriptor": cb_out_descriptor,
    }


def setup_dram_matmul(
    device,
    weights_tensor,
    output_tensor,
    core_ranges,
    cb_in1_index,
    cb_out_index,
    fp32_dest_acc_en=False,
    num_subblocks_k=4,
):
    """
    Set up parameters and CB descriptors for a DRAM streaming matmul operation.

    Args:
        device: TT device
        weights_tensor: Weight tensor (WIDTH_SHARDED in DRAM)
        output_tensor: Output tensor (WIDTH_SHARDED in L1)
        core_ranges: CoreRangeSet for compute cores
        cb_in1_index: CB index for weights working buffer
        cb_out_index: CB index for output
        fp32_dest_acc_en: Whether FP32 dest accumulation is enabled
        num_subblocks_k: Number of K subblocks (default 4)

    Returns:
        Dictionary with all computed parameters and CB descriptors
    """
    # Get parameters from weights tensor
    weights_tile = weights_tensor.get_tile()
    weights_dtype = weights_tensor.dtype
    weights_tile_size = weights_tile.get_tile_size(weights_dtype)
    weights_shard_shape = weights_tensor.memory_config().shard_spec.shape
    K = weights_shard_shape[0]
    per_core_n = weights_shard_shape[1] // weights_tile.tile_shape[1]
    Kt = K // weights_tile.tile_shape[0]

    # Calculate subblock_k
    subblock_k = Kt // num_subblocks_k
    assert Kt % num_subblocks_k == 0, f"Kt ({Kt}) must be divisible by num_subblocks ({num_subblocks_k})"

    # Calculate page size for NOC transfers
    in1_page_size, in1_num_pages = get_max_page_size_and_num_pages(device, subblock_k, weights_tile_size)
    in1_block_size_bytes = subblock_k * weights_tile_size

    # CB in1: weights working buffer
    num_in1_buffers = 3 * num_subblocks_k
    assert num_in1_buffers <= 15, f"num_in1_buffers ({num_in1_buffers}) exceeds NOC_MAX_TRANSACTION_ID (15)"
    in1_CB_tiles = subblock_k * num_in1_buffers
    in1_CB_size = in1_CB_tiles * weights_tile_size

    weights_tile_desc = ttnn.TileDescriptor(weights_tile)
    cb_in1_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_in1_index,
        data_format=weights_dtype,
        page_size=weights_tile_size,
        tile=weights_tile_desc,
    )
    cb_in1_descriptor = ttnn.CBDescriptor(
        total_size=in1_CB_size,
        core_ranges=core_ranges,
        format_descriptors=[cb_in1_format],
    )

    # CB out: output (tensor-backed)
    cb_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_out_index, output_tensor)

    # Calculate subblock_w based on fp32_dest_acc_en and per_core_n
    if fp32_dest_acc_en:
        max_subblock_w = 8 if per_core_n <= 8 else 4
    else:
        max_subblock_w = 16 if per_core_n <= 16 else 8

    subblock_w = max_subblock_w
    while subblock_w > 1 and per_core_n % subblock_w != 0:
        subblock_w -= 1

    # Tile dimensions
    tile_r_dim = weights_tile.tile_shape[0]

    return {
        # Dimension parameters
        "per_core_n": per_core_n,
        "Kt": Kt,
        "tile_r_dim": tile_r_dim,
        # Subblock parameters
        "num_subblocks_k": num_subblocks_k,
        "subblock_k": subblock_k,
        "subblock_w": subblock_w,
        # NOC transfer parameters
        "in1_page_size": in1_page_size,
        "in1_num_pages": in1_num_pages,
        "in1_block_size_bytes": in1_block_size_bytes,
        # Output parameters
        "out_num_tiles": per_core_n,
        # Buffer address
        "in1_tensor_addr": weights_tensor.buffer_address(),
        # CB descriptors
        "cb_in1_descriptor": cb_in1_descriptor,
        "cb_out_descriptor": cb_out_descriptor,
    }


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
        gate_proj_weights_dict=None,
        up_proj_weights_dict=None,
        down_proj_weights_dict=None,
        eps=1e-20,
        scaling_factor=2.5,
        use_hardcoded_expert_index=False,
    ):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) [1, K]
            routing_weights_tensor: Routing matmul weights (torch.Tensor) [K, N_routing]
            bias_tensor: Gate bias tensor (torch.Tensor) [1, 8, 32] or [16, 16]
            gate_proj_weights_dict: Dict mapping expert_idx -> gate_proj weights [1, 1, K, N_expert] (optional)
            up_proj_weights_dict: Dict mapping expert_idx -> up_proj weights [1, 1, K, N_expert] (optional)
            down_proj_weights_dict: Dict mapping expert_idx -> down_proj weights [1, 1, N_expert, K] (optional)
            eps: Epsilon for numerical stability in gate
            scaling_factor: Scaling factor for gate scores

        Returns:
            If gate_proj_weights_dict is None:
                Tuple of (top8_scores, top8_indices) tensors
            Else:
                Tuple of (top8_scores, top8_indices, down_proj_output) tensors
                down_proj_output = (silu(gate_proj) * up_proj) @ down_proj_weights
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

        # 3. Expert matmuls (if expert weights provided)
        if gate_proj_weights_dict is not None:
            # Get the first selected expert index
            if use_hardcoded_expert_index:
                selected_expert_idx = 0  # For testing: always use expert 0
            else:
                selected_expert_idx = int(top8_indices[0, 0].item())

            # gate_proj: input @ weights + SiLU
            gate_proj_weights = gate_proj_weights_dict[selected_expert_idx]
            input_for_expert = input_tensor.reshape(1, 1, 1, -1).float()
            gate_proj_output = input_for_expert @ gate_proj_weights.float()
            gate_proj_output = torch.nn.functional.silu(gate_proj_output)

            # up_proj: input @ weights (no activation)
            up_proj_weights = up_proj_weights_dict[selected_expert_idx]
            up_proj_output = input_for_expert @ up_proj_weights.float()

            # Fused output: silu(gate_proj) * up_proj
            fused_output = gate_proj_output * up_proj_output

            # down_proj: fused_output @ weights (no activation)
            if down_proj_weights_dict is not None:
                down_proj_weights = down_proj_weights_dict[selected_expert_idx]
                down_proj_output = fused_output @ down_proj_weights.float()
                return top8_scores, top8_indices, down_proj_output

            return top8_scores, top8_indices, fused_output

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
        gate_proj_index_tensor,
        gate_proj_weights_tensor,
        gate_proj_output_tensor,
        up_proj_weights_tensor,
        up_proj_mm_out_tensor,
        fused_output_tensor,
        down_proj_gather_output_tensor,
        down_proj_mcast_output_tensor,
        down_proj_weights_tensor,
        down_proj_output_tensor,
        use_hardcoded_expert_index=False,  # For testing: always use expert index 0
    ):
        """
        Execute the full MoE routed expert fused operation:
        mcast + matmul + sigmoid + gather + gate + index_mcast + gate_proj + up_proj + mul +
        down_proj_gather + down_proj_mcast + down_proj

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
            gate_proj_index_tensor: DRAM matmul+SiLU index tensor [1, 16] on mcast grid (receives mcasted indices)
            gate_proj_weights_tensor: Expert gate_proj weights [K, N_expert] width-sharded in DRAM (first expert)
            gate_proj_output_tensor: Expert gate_proj matmul+SiLU output [1, N_expert] width-sharded on DRAM matmul cores
            up_proj_weights_tensor: Expert up_proj weights [K, N_expert] width-sharded in DRAM (first expert)
            up_proj_mm_out_tensor: Expert up_proj matmul output (intermediate) [1, N_expert] width-sharded
            fused_output_tensor: Fused output silu(gate_proj) * up_proj [1, N_expert] width-sharded
            down_proj_gather_output_tensor: Gathered fused output [1, N_expert] on sender core
            down_proj_mcast_output_tensor: Mcasted fused output [1, N_expert] on mcast grid
            down_proj_weights_tensor: down_proj weights [N_expert, K] width-sharded in DRAM (first expert)
            down_proj_output_tensor: down_proj output [1, K] width-sharded on DRAM matmul cores

        Returns:
            Tuple of (gate_output_scores_tensor, gate_output_indices_tensor, down_proj_output_tensor)
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
        gate_proj_output_memory_config = gate_proj_output_tensor.memory_config()
        gate_proj_core_ranges = gate_proj_output_memory_config.shard_spec.grid

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
        gate_mm_input_cb = 1  # Mcast destination CB (receives input on all cores) - also used as expert matmul input
        gate_mm_weights_cb = 2  # Matmul weights (sharded on all cores)
        gate_mm_output_cb = 3  # Matmul output (intermediate on compute cores)
        gate_input_cb = 4  # Gate input (gathered output, tensor-backed on sender core)
        gate_bias_cb = 5  # Gate bias (tensor-backed on sender core)
        gate_indices_cb = 6  # Gate indices (tensor-backed on sender core)
        gate_output_cb = 7  # Gate output scores (tensor-backed on sender core)
        gate_output_indices_cb = 8  # Gate output indices (tensor-backed on sender core)
        gate_proj_cb_in1 = 9  # DRAM matmul weights working buffer
        gate_proj_cb_index = 10  # DRAM matmul index (receives mcasted indices)
        gate_proj_cb_out = 11  # DRAM matmul output (tensor-backed on compute cores)
        up_proj_cb_in1 = 12  # up_proj matmul weights working buffer
        up_proj_cb_mm_out = 13  # up_proj matmul output intermediate (before mul)
        # Mul CBs for fusing up_proj * gate_proj
        mul_cb_in0 = 14  # up_proj mm output aliased as 16x16 (same memory as up_proj_cb_mm_out)
        mul_cb_in1 = 15  # gate_proj output aliased as 16x16 (same memory as gate_proj_cb_out)
        mul_cb_out = 16  # fused output (tensor-backed)
        # down_proj CBs (gather reads directly from mul_cb_out, no separate CB needed)
        down_proj_gather_dst_cb = 17  # gathered fused output on sender core (tensor-backed)
        down_proj_mcast_dst_cb = 18  # mcasted fused output on compute cores (tensor-backed)
        down_proj_cb_in1 = 19  # down_proj weights working buffer
        down_proj_cb_out = 20  # down_proj output (tensor-backed)

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

        # ========== DRAM Streaming Matmul Setup ==========
        gate_proj_params = setup_dram_matmul(
            device=device,
            weights_tensor=gate_proj_weights_tensor,
            output_tensor=gate_proj_output_tensor,
            core_ranges=gate_proj_core_ranges,
            cb_in1_index=gate_proj_cb_in1,
            cb_out_index=gate_proj_cb_out,
            fp32_dest_acc_en=fp32_dest_acc_en,
        )
        gate_proj_cb_in1_descriptor = gate_proj_params["cb_in1_descriptor"]
        gate_proj_cb_out_descriptor = gate_proj_params["cb_out_descriptor"]

        # ========== up_proj Matmul Setup ==========
        up_proj_params = setup_dram_matmul(
            device=device,
            weights_tensor=up_proj_weights_tensor,
            output_tensor=up_proj_mm_out_tensor,  # Intermediate output (before mul)
            core_ranges=gate_proj_core_ranges,  # Same cores as gate_proj
            cb_in1_index=up_proj_cb_in1,
            cb_out_index=up_proj_cb_mm_out,  # Write to intermediate CB
            fp32_dest_acc_en=fp32_dest_acc_en,
        )
        up_proj_cb_in1_descriptor = up_proj_params["cb_in1_descriptor"]
        up_proj_cb_mm_out_descriptor = up_proj_params["cb_out_descriptor"]

        # ========== Mul Setup (up_proj * gate_proj -> fused output) ==========
        mul_params = setup_eltwise_mul(
            in0_tensor=up_proj_mm_out_tensor,
            in1_tensor=gate_proj_output_tensor,
            out_tensor=fused_output_tensor,
            core_ranges=gate_proj_core_ranges,
            cb_in0_index=mul_cb_in0,
            cb_in1_index=mul_cb_in1,
            cb_out_index=mul_cb_out,
            per_core_n=gate_proj_params["per_core_n"],
        )
        mul_num_tiles = mul_params["mul_num_tiles"]
        mul_cb_in0_descriptor = mul_params["cb_in0_descriptor"]
        mul_cb_in1_descriptor = mul_params["cb_in1_descriptor"]
        mul_cb_out_descriptor = mul_params["cb_out_descriptor"]

        # ========== down_proj_gather Setup ==========
        # Get gate_proj_core grid bounds for gather sender grid
        gate_proj_core_list = list(gate_proj_core_ranges.ranges())
        # Find bounding box of gate_proj cores
        gate_proj_min_x = min(r.start.x for r in gate_proj_core_list)
        gate_proj_min_y = min(r.start.y for r in gate_proj_core_list)
        gate_proj_max_x = max(r.end.x for r in gate_proj_core_list)
        gate_proj_max_y = max(r.end.y for r in gate_proj_core_list)
        num_gate_proj_cores = gate_proj_core_ranges.num_cores()

        # Gather semaphore IDs (defined early so down_proj_gather can reuse them)
        gather_noc0_receiver_semaphore_id = 2
        gather_noc1_receiver_semaphore_id = 3

        # Tile format for 1x32 tiles (used for gather/mcast data size calculations)
        TILE_1x32 = ttnn.Tile((1, 32))
        tile_1x32_size = TILE_1x32.get_tile_size(data_format)
        tile_1x32_desc = ttnn.TileDescriptor(TILE_1x32)

        # Data size per core for gather: only valid data (per_core_n tiles of 1x32)
        # Note: mul outputs 16x16 tiles with padding, but we only send the valid portion
        # to avoid buffer overflow on the receiver (gather output tensor is sized for valid data only)
        down_proj_gather_data_size_bytes = gate_proj_params["per_core_n"] * tile_1x32_size

        # down_proj_gather semaphore IDs (reuse gather semaphores since they're reset)
        down_proj_gather_noc0_receiver_semaphore_id = gather_noc0_receiver_semaphore_id
        down_proj_gather_noc1_receiver_semaphore_id = gather_noc1_receiver_semaphore_id

        # All gate_proj cores are senders
        down_proj_gather_noc0_num_senders = num_gate_proj_cores
        down_proj_gather_noc1_num_senders = 0

        # Gather reads directly from mul_cb_out (CB 16) where mul pushed data
        # No need for a separate aliased CB - we use mul_cb_out and mul_num_tiles

        # CB for gather destination on sender core (tensor-backed)
        down_proj_gather_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            down_proj_gather_dst_cb, down_proj_gather_output_tensor
        )

        # Gather receiver data address
        down_proj_gather_receiver_data_addr = down_proj_gather_output_tensor.buffer_address()

        # Gather output pages (number of tiles in gathered output)
        down_proj_gather_output_shard = down_proj_gather_output_tensor.memory_config().shard_spec.shape
        down_proj_gather_dst_num_pages = (down_proj_gather_output_shard[0] * down_proj_gather_output_shard[1]) // (
            TILE_1x32.tile_shape[0] * TILE_1x32.tile_shape[1]
        )

        # ========== down_proj_mcast Setup ==========
        # CB for mcast destination on compute cores (tensor-backed)
        down_proj_mcast_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            down_proj_mcast_dst_cb, down_proj_mcast_output_tensor
        )

        # Mcast parameters for down_proj (same grid as input mcast)
        down_proj_mcast_sender_semaphore_id = mcast_data_sender_semaphore_id
        down_proj_mcast_receiver_semaphore_id = mcast_data_receiver_semaphore_id

        # Calculate mcast data size (fused output size)
        fused_output_shard = down_proj_gather_output_tensor.memory_config().shard_spec.shape
        down_proj_mcast_num_tiles = (fused_output_shard[0] * fused_output_shard[1]) // (
            TILE_1x32.tile_shape[0] * TILE_1x32.tile_shape[1]
        )
        down_proj_mcast_data_size_bytes = down_proj_mcast_num_tiles * tile_1x32_size

        # ========== down_proj DRAM Matmul Setup ==========
        down_proj_params = setup_dram_matmul(
            device=device,
            weights_tensor=down_proj_weights_tensor,
            output_tensor=down_proj_output_tensor,
            core_ranges=gate_proj_core_ranges,  # Same cores as gate_proj/up_proj
            cb_in1_index=down_proj_cb_in1,
            cb_out_index=down_proj_cb_out,
            fp32_dest_acc_en=fp32_dest_acc_en,
        )
        down_proj_cb_in1_descriptor = down_proj_params["cb_in1_descriptor"]
        down_proj_cb_out_descriptor = down_proj_params["cb_out_descriptor"]

        # CB 10: DRAM matmul index (receives mcasted indices from gate)
        gate_proj_cb_index_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            gate_proj_cb_index, gate_proj_index_tensor
        )

        # Get index tile size for compile-time args
        index_tile = gate_proj_index_tensor.get_tile()
        index_dtype = gate_proj_index_tensor.dtype
        index_tile_size = index_tile.get_tile_size(index_dtype)

        # Index mcast parameters - reuse same semaphores as input mcast
        index_mcast_sender_semaphore_id = mcast_data_sender_semaphore_id
        index_mcast_receiver_semaphore_id = mcast_data_receiver_semaphore_id
        index_mcast_num_pages = 1  # Single index tile
        index_mcast_data_size_bytes = index_tile_size

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
            ("gate_proj_cb_index", gate_proj_cb_index),
            ("index_mcast_num_pages", index_mcast_num_pages),
            # Mul reader args (setup mul_in1 buffer)
            ("mul_cb_in1", mul_cb_in1),
            ("mul_num_tiles", mul_num_tiles),
            # down_proj_gather sender args (gate_proj cores send fused output to sender core)
            ("down_proj_gather_dest_noc_x", sender_core_noc.x),
            ("down_proj_gather_dest_noc_y", sender_core_noc.y),
            ("down_proj_gather_data_size_bytes", down_proj_gather_data_size_bytes),
            ("down_proj_gather_receiver_semaphore_id", down_proj_gather_noc0_receiver_semaphore_id),
            ("down_proj_gather_src_cb", mul_cb_out),  # Read from mul output CB
            ("down_proj_gather_src_num_pages", mul_num_tiles),  # Wait for mul_num_tiles (16x16)
            ("down_proj_gather_sender_grid_start_x", gate_proj_min_x),
            ("down_proj_gather_sender_grid_start_y", gate_proj_min_y),
            ("down_proj_gather_sender_grid_end_x", gate_proj_max_x),
            ("down_proj_gather_sender_grid_end_y", gate_proj_max_y),
            ("down_proj_gather_row_major", 1),  # Row-major for DRAM bank cores
            ("down_proj_gather_receiver_data_addr", down_proj_gather_receiver_data_addr),
            # down_proj_mcast receiver args (compute cores)
            ("down_proj_mcast_receiver_semaphore", down_proj_mcast_receiver_semaphore_id),
            ("down_proj_mcast_dst_cb", down_proj_mcast_dst_cb),
            ("down_proj_mcast_dst_num_pages", down_proj_mcast_num_tiles),
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
            ("gate_proj_cb_index", gate_proj_cb_index),
            # DRAM streaming matmul writer args (compute cores)
            ("gate_proj_cb_in1", gate_proj_cb_in1),
            ("gate_proj_cb_out", gate_proj_cb_out),
            ("gate_proj_in1_tensor_addr", gate_proj_params["in1_tensor_addr"]),
            ("gate_proj_in1_page_size", gate_proj_params["in1_page_size"]),
            ("gate_proj_in1_num_pages", gate_proj_params["in1_num_pages"]),
            ("gate_proj_subblock_k", gate_proj_params["subblock_k"]),
            ("gate_proj_per_core_n", gate_proj_params["per_core_n"]),
            ("gate_proj_in1_block_size_bytes", gate_proj_params["in1_block_size_bytes"]),
            ("gate_proj_out_num_tiles", gate_proj_params["out_num_tiles"]),
            ("gate_proj_num_subblocks_k", gate_proj_params["num_subblocks_k"]),
            ("gate_proj_index_offset", 0),  # Always use first index from the mcasted index tensor
            # up_proj matmul writer args (compute cores) - writes to intermediate CB
            ("up_proj_cb_in1", up_proj_cb_in1),
            ("up_proj_in1_tensor_addr", up_proj_params["in1_tensor_addr"]),
            ("up_proj_in1_page_size", up_proj_params["in1_page_size"]),
            ("up_proj_in1_num_pages", up_proj_params["in1_num_pages"]),
            ("up_proj_subblock_k", up_proj_params["subblock_k"]),
            ("up_proj_per_core_n", up_proj_params["per_core_n"]),
            ("up_proj_in1_block_size_bytes", up_proj_params["in1_block_size_bytes"]),
            ("up_proj_out_num_tiles", up_proj_params["out_num_tiles"]),
            ("up_proj_num_subblocks_k", up_proj_params["num_subblocks_k"]),
            ("up_proj_cb_index", gate_proj_cb_index),  # Reuses same index CB as gate_proj
            ("up_proj_index_offset", 0),  # Same expert index as gate_proj
            ("up_proj_cb_mm_out", up_proj_cb_mm_out),  # Intermediate output for up_proj (before mul)
            # Mul writer args (waits for final output)
            ("mul_cb_out", mul_cb_out),
            ("mul_num_tiles", mul_num_tiles),
            # down_proj_gather receiver args (sender core receives fused output from gate_proj cores)
            ("down_proj_gather_noc0_num_senders", down_proj_gather_noc0_num_senders),
            ("down_proj_gather_noc1_num_senders", down_proj_gather_noc1_num_senders),
            ("down_proj_gather_noc0_receiver_semaphore_id", down_proj_gather_noc0_receiver_semaphore_id),
            ("down_proj_gather_noc1_receiver_semaphore_id", down_proj_gather_noc1_receiver_semaphore_id),
            ("down_proj_gather_dst_cb", down_proj_gather_dst_cb),
            ("down_proj_gather_dst_num_pages", down_proj_gather_dst_num_pages),
            # down_proj_mcast sender args (sender core broadcasts fused output to compute cores)
            ("down_proj_mcast_sender_semaphore", down_proj_mcast_sender_semaphore_id),
            ("down_proj_mcast_receiver_semaphore", down_proj_mcast_receiver_semaphore_id),
            ("down_proj_mcast_data_size_bytes", down_proj_mcast_data_size_bytes),
            ("down_proj_mcast_src_cb", down_proj_gather_dst_cb),  # Same as gather dst
            ("down_proj_mcast_dst_cb", down_proj_mcast_dst_cb),
            ("down_proj_mcast_src_num_pages", down_proj_mcast_num_tiles),
            # down_proj DRAM matmul writer args (compute cores)
            ("down_proj_cb_in1", down_proj_cb_in1),
            ("down_proj_cb_out", down_proj_cb_out),
            ("down_proj_in1_tensor_addr", down_proj_params["in1_tensor_addr"]),
            ("down_proj_in1_page_size", down_proj_params["in1_page_size"]),
            ("down_proj_in1_num_pages", down_proj_params["in1_num_pages"]),
            ("down_proj_subblock_k", down_proj_params["subblock_k"]),
            ("down_proj_per_core_n", down_proj_params["per_core_n"]),
            ("down_proj_in1_block_size_bytes", down_proj_params["in1_block_size_bytes"]),
            ("down_proj_out_num_tiles", down_proj_params["out_num_tiles"]),
            ("down_proj_num_subblocks_k", down_proj_params["num_subblocks_k"]),
            ("down_proj_cb_index", gate_proj_cb_index),  # Reuses same index CB
            ("down_proj_index_offset", 0),  # Same expert index
            # Testing flag: use hardcoded expert index 0 instead of gate output
            ("use_hardcoded_expert_index", 1 if use_hardcoded_expert_index else 0),
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
            ("gate_proj_cb_in0", gate_mm_input_cb),  # Reuses mcasted input
            ("gate_proj_cb_in1", gate_proj_cb_in1),
            ("gate_proj_cb_out", gate_proj_cb_out),
            ("gate_proj_subblock_k", gate_proj_params["subblock_k"]),
            ("gate_proj_per_core_n", gate_proj_params["per_core_n"]),
            ("gate_proj_subblock_w", gate_proj_params["subblock_w"]),
            ("gate_proj_num_subblocks_k", gate_proj_params["num_subblocks_k"]),
            ("gate_proj_tile_r_dim", gate_proj_params["tile_r_dim"]),
            ("gate_proj_fuse_silu", 1),  # Always use SiLU for expert computation
            # up_proj matmul compute args (compute cores) - writes to intermediate CB
            ("up_proj_cb_in0", gate_mm_input_cb),  # Reuses mcasted input (same as gate_proj)
            ("up_proj_cb_in1", up_proj_cb_in1),
            ("up_proj_subblock_k", up_proj_params["subblock_k"]),
            ("up_proj_per_core_n", up_proj_params["per_core_n"]),
            ("up_proj_subblock_w", up_proj_params["subblock_w"]),
            ("up_proj_num_subblocks_k", up_proj_params["num_subblocks_k"]),
            ("up_proj_tile_r_dim", up_proj_params["tile_r_dim"]),
            ("up_proj_fuse_silu", 0),  # No SiLU for up_proj
            ("up_proj_cb_mm_out", up_proj_cb_mm_out),  # Intermediate output for up_proj (before mul)
            # Mul compute args (up_proj * gate_proj -> fused output)
            ("mul_cb_in0", mul_cb_in0),  # up_proj output aliased as 16x16
            ("mul_cb_in1", mul_cb_in1),  # gate_proj output aliased as 16x16
            ("mul_cb_out", mul_cb_out),  # final fused output
            ("mul_num_tiles", mul_num_tiles),
            ("up_proj_per_core_n", up_proj_params["per_core_n"]),  # tiles in mm_out format for cb_wait
            # down_proj matmul compute args (compute cores)
            ("down_proj_cb_in0", down_proj_mcast_dst_cb),  # Mcasted fused output
            ("down_proj_cb_in1", down_proj_cb_in1),
            ("down_proj_cb_out", down_proj_cb_out),
            ("down_proj_subblock_k", down_proj_params["subblock_k"]),
            ("down_proj_per_core_n", down_proj_params["per_core_n"]),
            ("down_proj_subblock_w", down_proj_params["subblock_w"]),
            ("down_proj_num_subblocks_k", down_proj_params["num_subblocks_k"]),
            ("down_proj_tile_r_dim", down_proj_params["tile_r_dim"]),
            ("down_proj_fuse_silu", 0),  # No SiLU for down_proj
            # Testing flag: use hardcoded expert index 0 instead of gate output
            ("use_hardcoded_expert_index", 1 if use_hardcoded_expert_index else 0),
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
        gate_proj_optimal_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        # Build mapping from core to bank_id based on optimal assignment
        core_to_bank_id = {}
        for bank_id, core in enumerate(gate_proj_optimal_workers):
            core_to_bank_id[(core.x, core.y)] = bank_id
        bank_id_core_values = []
        vc_core_values = []
        sender_idx_core_values = []  # For down_proj_gather sender idx
        for idx, core in enumerate(gate_proj_optimal_workers):
            bank_id = core_to_bank_id[(core.x, core.y)]
            vc = bank_id & 0x3
            bank_id_core_values.append((core, bank_id))
            vc_core_values.append((core, vc))
            sender_idx_core_values.append((core, idx))  # Explicit sender idx

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
                    named_compile_time_arg="is_gate_proj_core",
                    core_range=gate_proj_core_ranges,
                    value=1,
                    other_value=0,
                ),
            ],
            per_core_compile_time_descriptors=[
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="gate_proj_bank_id",
                    core_values=bank_id_core_values,
                    other_value=0,
                ),
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="gate_proj_vc",
                    core_values=vc_core_values,
                    other_value=0,
                ),
                # up_proj uses same cores as gate_proj, so same bank_id and vc
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="up_proj_bank_id",
                    core_values=bank_id_core_values,
                    other_value=0,
                ),
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="up_proj_vc",
                    core_values=vc_core_values,
                    other_value=0,
                ),
                # down_proj uses same cores as gate_proj, so same bank_id and vc
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="down_proj_bank_id",
                    core_values=bank_id_core_values,
                    other_value=0,
                ),
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="down_proj_vc",
                    core_values=vc_core_values,
                    other_value=0,
                ),
                # Explicit sender idx for down_proj_gather (scattered optimal DRAM bank cores)
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="down_proj_gather_sender_idx",
                    core_values=sender_idx_core_values,
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
                gate_proj_cb_in1_descriptor,
                gate_proj_cb_index_descriptor,
                gate_proj_cb_out_descriptor,
                up_proj_cb_in1_descriptor,
                up_proj_cb_mm_out_descriptor,
                mul_cb_in0_descriptor,
                mul_cb_in1_descriptor,
                mul_cb_out_descriptor,
                down_proj_gather_dst_cb_descriptor,
                down_proj_mcast_dst_cb_descriptor,
                down_proj_cb_in1_descriptor,
                down_proj_cb_out_descriptor,
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
            gate_proj_index_tensor,
            gate_proj_weights_tensor,
            gate_proj_output_tensor,
            up_proj_weights_tensor,
            up_proj_mm_out_tensor,
            fused_output_tensor,
            down_proj_gather_output_tensor,
            down_proj_mcast_output_tensor,
            down_proj_weights_tensor,
            down_proj_output_tensor,
        ]
        ttnn.generic_op(io_tensors, program_descriptor)

        return gate_output_scores_tensor, gate_output_indices_tensor, down_proj_output_tensor
