# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Fused MoE operation: Routed Expert + Shared Expert.

Architecture:
  MoeRoutedExpertOp  — context setup & build methods for routed expert
  MoeSharedExpertOp  — context setup & build methods for shared expert
  MoeOp              — top-level orchestrator that composes both

Pipeline (routed expert):
  1. Input mcast → 2. Gate matmul → 3. Gate gather → 4. Gate
  5. Index/Scale mcast → 6. gate_proj → 7. up_proj → 8. Fused mul
  9. down_proj gather → 10. down_proj mcast → 11. down_proj → 12. Eltwise add

Pipeline (shared expert, fused):
  3a. Gate/Up KN-sliced matmul (runs on 128 cores after input mcast)
"""

import math
from dataclasses import dataclass
from typing import Any

import torch

import ttnn
from models.demos.deepseek_v3_b1.fused_ops.face_view_utils import FACE_HEIGHT, FACE_WIDTH, can_use_face_view
from models.demos.deepseek_v3_b1.fused_ops.moe_routed_expert.op import (
    get_max_page_size_and_num_pages,
    setup_eltwise_add,
    setup_gate,
    setup_gather,
    setup_mcast,
    setup_sram_matmul,
)
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


def setup_dram_matmul(
    device,
    weights_tensor,
    output_tensor,
    working_buf_tensor,
    core_ranges,
    cb_in1_index,
    cb_out_index,
    fp32_dest_acc_en,
    num_subblocks_k,
):
    """
    Set up parameters and CB descriptors for a DRAM streaming matmul operation.
    Uses a tensor-backed working buffer for the weights CB.

    Args:
        device: TT device
        weights_tensor: Weight tensor (WIDTH_SHARDED in DRAM)
        output_tensor: Output tensor (WIDTH_SHARDED in L1)
        working_buf_tensor: Tensor-backed working buffer for weights CB (WIDTH_SHARDED in L1)
        core_ranges: CoreRangeSet for compute cores
        cb_in1_index: CB index for weights working buffer
        cb_out_index: CB index for output
        fp32_dest_acc_en: Whether FP32 dest accumulation is enabled
        num_subblocks_k: Number of K subblocks

    Returns:
        Dictionary with all computed parameters and CB descriptors
    """
    weights_tile = weights_tensor.get_tile()
    weights_shard_shape = weights_tensor.memory_config().shard_spec.shape
    K = weights_shard_shape[0]
    per_core_n = weights_shard_shape[1] // weights_tile.tile_shape[1]
    Kt = K // weights_tile.tile_shape[0]

    subblock_k = Kt // num_subblocks_k
    assert Kt % num_subblocks_k == 0, f"Kt ({Kt}) must be divisible by num_subblocks ({num_subblocks_k})"

    weights_tile_size = weights_tile.get_tile_size(weights_tensor.dtype)
    in1_page_size, in1_num_pages = get_max_page_size_and_num_pages(device, subblock_k, weights_tile_size)
    in1_block_size_bytes = subblock_k * weights_tile_size

    # CB in1: tensor-backed weights working buffer
    cb_in1_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_in1_index, working_buf_tensor)

    # CB out: tensor-backed output
    cb_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_out_index, output_tensor)

    # subblock_w
    if fp32_dest_acc_en:
        max_subblock_w = 8 if per_core_n <= 8 else 4
    else:
        max_subblock_w = 16 if per_core_n <= 16 else 8
    subblock_w = max_subblock_w
    while subblock_w > 1 and per_core_n % subblock_w != 0:
        subblock_w -= 1

    tile_r_dim = weights_tile.tile_shape[0]

    return {
        "per_core_n": per_core_n,
        "Kt": Kt,
        "tile_r_dim": tile_r_dim,
        "num_subblocks_k": num_subblocks_k,
        "subblock_k": subblock_k,
        "subblock_w": subblock_w,
        "in1_page_size": in1_page_size,
        "in1_num_pages": in1_num_pages,
        "in1_block_size_bytes": in1_block_size_bytes,
        "out_num_tiles": per_core_n,
        "in1_tensor_addr": weights_tensor.buffer_address(),
        "in1_buf_addr": working_buf_tensor.buffer_address(),
        "cb_in1_descriptor": cb_in1_descriptor,
        "cb_out_descriptor": cb_out_descriptor,
    }


def setup_eltwise_mul(
    in0_tensor,
    in1_tensor,
    out_tensor,
    cb_in0_index,
    cb_in1_index,
    cb_out_index,
    per_core_n,
    cb_scalar_index,
    cb_scalar_src_index,
    scalar_src_tensor,
    scalar_buf_tensor,
):
    """
    Set up parameters and CB descriptors for element-wise multiply with CB aliasing and scalar multiply.
    Uses a tensor-backed working buffer for the scalar CB.

    Args:
        in0_tensor: First input tensor (e.g., up_proj matmul output)
        in1_tensor: Second input tensor (e.g., gate_proj matmul output)
        out_tensor: Output tensor for fused result
        cb_in0_index: CB index for first input (aliased)
        cb_in1_index: CB index for second input (aliased)
        cb_out_index: CB index for output
        per_core_n: Number of output tiles per core (in 1x32 format)
        cb_scalar_index: CB index for scalar working buffer
        cb_scalar_src_index: CB index for scalar source
        scalar_src_tensor: Tensor backing the scalar source CB
        scalar_buf_tensor: Tensor-backed working buffer for scalar CB

    Returns:
        Dictionary with mul_num_tiles and CB descriptors
    """
    M = 1
    tile_width = 32
    total_elements = M * per_core_n * tile_width
    mul_num_tiles = math.ceil(total_elements / 256)

    TILE_16x16 = ttnn.Tile((16, 16))
    tile_16x16_size = TILE_16x16.get_tile_size(ttnn.bfloat16)
    tile_16x16_desc = ttnn.TileDescriptor(TILE_16x16)

    # CB for in0: alias of in0_tensor with 16x16 tile format
    cb_in0_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_in0_index, in0_tensor)
    cb_in0_descriptor.total_size = mul_num_tiles * tile_16x16_size
    cb_in0_descriptor.format_descriptors[0].tile = tile_16x16_desc
    cb_in0_descriptor.format_descriptors[0].page_size = tile_16x16_size

    # CB for in1: alias of in1_tensor with 16x16 tile format
    cb_in1_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_in1_index, in1_tensor)
    cb_in1_descriptor.total_size = mul_num_tiles * tile_16x16_size
    cb_in1_descriptor.format_descriptors[0].tile = tile_16x16_desc
    cb_in1_descriptor.format_descriptors[0].page_size = tile_16x16_size

    # CB for out: output with 16x16 tile format (tensor-backed)
    cb_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_out_index, out_tensor)
    cb_out_descriptor.total_size = mul_num_tiles * tile_16x16_size
    cb_out_descriptor.format_descriptors[0].tile = tile_16x16_desc
    cb_out_descriptor.format_descriptors[0].page_size = tile_16x16_size

    # CB for scalar source: receives mcasted scalar (tensor-backed)
    cb_scalar_src_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_scalar_src_index, scalar_src_tensor)

    # CB for scalar working buffer: tensor-backed
    cb_scalar_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_scalar_index, scalar_buf_tensor)

    return {
        "mul_num_tiles": mul_num_tiles,
        "cb_in0_descriptor": cb_in0_descriptor,
        "cb_in1_descriptor": cb_in1_descriptor,
        "cb_out_descriptor": cb_out_descriptor,
        "cb_scalar_index": cb_scalar_index,
        "cb_scalar_src_index": cb_scalar_src_index,
        "cb_scalar_descriptor": cb_scalar_descriptor,
        "cb_scalar_src_descriptor": cb_scalar_src_descriptor,
    }


@dataclass
class _MoeRoutedExpertContext:
    """Holds all computed values needed by MoeRoutedExpertOp helper methods."""

    # Device & mesh
    device: Any
    full_device_grid: Any
    device_grid_size: Any
    mesh_rows: int
    mesh_cols: int

    # Core grids
    sender_core: Any
    input_core_grid: Any
    mcast_grid: Any
    gate_proj_core_ranges: Any
    num_gate_proj_cores: int

    # Data format & tiles
    data_format: Any
    tile_1x32_size: int
    num_tiles_k: int

    # CB indices (26 total)
    input_cb: int
    gate_mm_input_cb: int
    gate_mm_weights_cb: int
    gate_mm_output_cb: int
    gate_input_cb: int
    gate_bias_cb: int
    gate_indices_cb: int
    gate_output_cb: int
    gate_output_indices_cb: int
    gate_proj_cb_in1: int
    gate_proj_cb_index: int
    gate_proj_cb_out: int
    up_proj_cb_in1: int
    up_proj_cb_mm_out: int
    mul_cb_in0: int
    mul_cb_in1: int
    mul_cb_out: int
    down_proj_gather_dst_cb: int
    down_proj_mcast_dst_cb: int
    down_proj_cb_in1: int
    down_proj_cb_out: int
    mul_cb_scalar_src: int
    mul_cb_scalar: int
    add_cb_in0: int
    add_cb_in1: int
    add_cb_out: int

    # Semaphore IDs
    mcast_data_sender_semaphore_id: int
    mcast_data_receiver_semaphore_id: int
    gather_noc0_receiver_semaphore_id: int
    gather_noc1_receiver_semaphore_id: int
    expert_scale_mcast_sender_semaphore_id: int
    expert_scale_mcast_receiver_semaphore_id: int

    # Setup result dicts (from helper functions)
    input_mcast_params: dict
    gate_mm_params: dict
    gate_mm_gather_params: dict
    gate_params: dict
    gate_proj_params: dict
    up_proj_params: dict
    mul_params: dict
    down_proj_gather_params: dict
    down_proj_mcast_params: dict
    down_proj_params: dict
    add_params: dict

    # Index mcast params
    index_mcast_sender_semaphore_id: int
    index_mcast_receiver_semaphore_id: int
    index_mcast_num_pages: int
    index_mcast_data_size_bytes: int

    # Expert scale mcast params
    expert_scale_mcast_num_pages: int
    expert_scale_mcast_data_size_bytes: int

    # Derived values
    mul_num_tiles: int

    # Pre-built CB descriptors (tensor-backed, built in _setup_dimensions)
    input_cb_descriptor: Any
    gate_mm_input_cb_descriptor: Any
    gate_proj_cb_index_descriptor: Any

    # Per-core values (for core descriptors)
    bank_id_core_values: list
    vc_core_values: list
    sender_idx_core_values: list

    # Testing flag
    use_hardcoded_expert_index: bool


@dataclass
class _MoeSharedExpertContext:
    """Holds shared expert values for the fused MoE kernel."""

    # Gate/Up KN-sliced matmul
    gu_weights_cb: int
    gu_out_cb: int
    gu_k_per_core: int
    gu_act_total_tiles: int
    gu_weights_num_pages: int
    compute_core_grid: Any
    gu_k_offset_core_values: list
    gu_weights_cb_descriptor: Any
    gu_out_cb_descriptor: Any

    # Gate/Up Gather + Gated Reduce
    a_cores_list: list
    b_cores_list: list
    a_compute_grid: Any
    b_compute_grid: Any
    group1_cb: int  # gate gather dst (on sender)
    group2_cb: int  # up gather dst (on sender)
    intermed_cb: int  # gated reduce intermediate (on sender)
    mcast_src_cb: int  # gated reduce output (on sender)
    n_parallel: int
    k_parallel: int
    tiles_per_k: int
    k_num_tiles: int
    num_compute_cores: int  # 64 per branch
    use_face_view: bool
    face_tile_desc: Any
    face_tile_size: Any
    kernel_tiles_per_k: int
    kernel_k_num_tiles: int
    mcast_src_num_pages: int
    reduce_tile_size: int
    input_tile_size: int
    gu_gather_data_size_bytes: int
    total_gather_tiles: int
    gather_dest_noc_core: Any
    mcast_gather_core_grid: Any
    ag_receiver_semaphore_id: int
    bg_receiver_semaphore_id: int
    ag_noc1_receiver_semaphore_id: int
    bg_noc1_receiver_semaphore_id: int
    ag_dummy_tensor: Any
    bg_dummy_tensor: Any
    ag_receiver_data_addr: int
    bg_receiver_data_addr: int
    group1_cb_descriptor: Any
    group2_cb_descriptor: Any
    intermed_cb_descriptor: Any
    mcast_src_cb_descriptor: Any
    ag_sender_idx_core_values: list
    bg_sender_idx_core_values: list

    # Residual Mcast
    residual_mcast_src_cb: int
    residual_mcast_dst_cb: int
    residual_mcast_sender_semaphore_id: int
    residual_mcast_receiver_semaphore_id: int
    residual_mcast_params: dict
    residual_mcast_src_cb_descriptor: Any
    residual_mcast_dst_cb_descriptor: Any
    residual_mcast_dst_dummy_tensor: Any  # keep-alive for dst tensor backing the CB

    # Down Mcast (gated reduce output → all 130 cores)
    down_mcast_dst_cb: int
    down_mcast_sender_semaphore_id: int
    down_mcast_receiver_semaphore_id: int
    down_mcast_data_size_bytes: int
    down_mcast_src_num_pages: int
    down_mcast_dst_cb_descriptor: Any
    down_mcast_dst_dummy_tensor: Any  # keep-alive for dst tensor backing the CB

    # Down Proj Matmul (SRAM matmul on 112 cores)
    down_matmul_in1_cb: int
    down_matmul_out_cb: int
    down_matmul_k_num_tiles: int
    down_matmul_out_w_per_core: int
    down_matmul_in1_cb_descriptor: Any
    down_matmul_out_cb_descriptor: Any
    down_weights_num_pages: int  # for setup_sharded_buffer

    # Residual Add (on 112 matmul cores)
    residual_add_out_cb: int
    residual_add_total_in1_tiles: int
    residual_add_out_cb_descriptor: Any


class MoeRoutedExpertOp:
    """
    MoE Routed Expert fused operation (refactored).

    Follows the SharedExpertOp pattern: context dataclass + decomposed build methods.
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
        fused_add_tensor=None,
        eps=1e-20,
        scaling_factor=2.5,
        use_hardcoded_expert_index=False,
        hardcoded_expert_index=0,
        explicit_expert_scale=None,
    ):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensor: [1, K] torch.Tensor
            routing_weights_tensor: [K, N_routing] torch.Tensor
            bias_tensor: [1, 8, 32] or [16, 16] torch.Tensor
            gate_proj_weights_dict: Dict[int, Tensor] expert_idx → [1,1,K,N_expert]
            up_proj_weights_dict: Dict[int, Tensor] expert_idx → [1,1,K,N_expert]
            down_proj_weights_dict: Dict[int, Tensor] expert_idx → [1,1,N_expert,K]
            fused_add_tensor: [1,1,1,K] torch.Tensor (optional)
            eps: Gate epsilon
            scaling_factor: Gate scaling factor
            use_hardcoded_expert_index: Use fixed expert index
            hardcoded_expert_index: Which expert to hardcode
            explicit_expert_scale: Override expert scale value

        Returns:
            (top8_scores, top8_indices, final_output) tensors
        """
        import torch

        from models.demos.deepseek_v3_b1.micro_ops.deepseek_moe_gate.op import DeepseekMoeGateSingleCore

        # 1. Routing matmul + sigmoid
        logits = input_tensor.float() @ routing_weights_tensor.float()
        scores = torch.sigmoid(logits)

        # 2. Gate: top-8 selection with normalized scores
        gate_input = scores.reshape(1, 8, 32)
        top8_scores, top8_indices = DeepseekMoeGateSingleCore.golden(
            gate_input, bias_tensor.float(), eps, scaling_factor, enable_sigmoid=False
        )

        # 3. Expert matmuls (if expert weights provided)
        if gate_proj_weights_dict is not None:
            if use_hardcoded_expert_index:
                selected_expert_idx = hardcoded_expert_index
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

            # Expert scale
            if explicit_expert_scale is not None:
                expert_scale = explicit_expert_scale
            else:
                expert_scale = top8_scores[0, 0].float()

            # Fused output: silu(gate_proj) * up_proj * expert_scale
            fused_output = gate_proj_output * up_proj_output * expert_scale

            # down_proj
            if down_proj_weights_dict is not None:
                down_proj_weights = down_proj_weights_dict[selected_expert_idx]
                down_proj_output = fused_output @ down_proj_weights.float()

                if fused_add_tensor is not None:
                    final_output = down_proj_output + fused_add_tensor.float()
                    return top8_scores, top8_indices, final_output
                return top8_scores, top8_indices, down_proj_output

            return top8_scores, top8_indices, fused_output

        return top8_scores, top8_indices, None

    # ========================================================================
    # Setup & build methods (SharedExpertOp pattern)
    # ========================================================================

    @staticmethod
    def _setup_dimensions(
        input_tensor,
        mcast_output_tensor,
        gate_mm_weights_tensor,
        gate_mm_output_tensor,
        gate_input_tensor,
        gate_bias_tensor,
        gate_indices_tensor,
        gate_output_scores_tensor,
        gate_output_indices_tensor,
        expert_index_tensor,
        expert_scale_tensor,
        gate_proj_weights_tensor,
        gate_proj_output_tensor,
        up_proj_weights_tensor,
        up_proj_mm_out_tensor,
        fused_output_tensor,
        down_proj_gather_output_tensor,
        down_proj_mcast_output_tensor,
        down_proj_weights_tensor,
        down_proj_output_tensor,
        fused_add_tensor,
        final_output_tensor,
        gate_proj_in1_buf_tensor,
        up_proj_in1_buf_tensor,
        down_proj_in1_buf_tensor,
        mul_scalar_buf_tensor,
        use_hardcoded_expert_index=False,
    ):
        """Compute all dimensions, grids, setup params, CB descriptors, and per-core values."""

        # ==================================================================
        # Tensor properties
        # ==================================================================
        data_format = input_tensor.dtype
        TILE_1x32 = ttnn.Tile((1, 32))
        tile_1x32_size = TILE_1x32.get_tile_size(data_format)
        K = input_tensor.shape[1]
        num_tiles_k = K // TILE_1x32.tile_shape[1]

        # Sender core (from input tensor's shard spec)
        input_core_grid = input_tensor.memory_config().shard_spec.grid
        sender_core = list(input_core_grid.ranges())[0].start

        # Device & mesh
        mesh_device = input_tensor.device()
        device_grid_size = mesh_device.compute_with_storage_grid_size()
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]
        device = ttnn.get_device_tensors(input_tensor)[0].device()

        # Core grids
        gate_proj_core_ranges = gate_proj_output_tensor.memory_config().shard_spec.grid
        mcast_grid = mcast_output_tensor.memory_config().shard_spec.grid
        num_gate_proj_cores = gate_proj_core_ranges.num_cores()

        # Full device grid
        full_device_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )

        # ==================================================================
        # Semaphore IDs
        # ==================================================================
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1
        gather_noc0_receiver_semaphore_id = 2
        gather_noc1_receiver_semaphore_id = 3
        expert_scale_mcast_sender_semaphore_id = 4
        expert_scale_mcast_receiver_semaphore_id = 5

        # ==================================================================
        # CB indices
        # ==================================================================
        input_cb = 0
        gate_mm_input_cb = 1
        gate_mm_weights_cb = 2
        gate_mm_output_cb = 3
        gate_input_cb = 4
        gate_bias_cb = 5
        gate_indices_cb = 6
        gate_output_cb = 7
        gate_output_indices_cb = 8
        gate_proj_cb_in1 = 9
        gate_proj_cb_index = 10
        gate_proj_cb_out = 11
        up_proj_cb_in1 = gate_proj_cb_in1  # Shared CB: same buffer, kernel resets pointers between uses
        up_proj_cb_mm_out = 13
        mul_cb_in0 = 14
        mul_cb_in1 = 15
        mul_cb_out = 16
        down_proj_gather_dst_cb = 17
        down_proj_mcast_dst_cb = 18
        down_proj_cb_in1 = 19
        down_proj_cb_out = 20
        mul_cb_scalar_src = 21
        mul_cb_scalar = 22
        add_cb_in0 = 23
        add_cb_in1 = 24
        add_cb_out = 25

        # ==================================================================
        # Input Mcast
        # ==================================================================
        input_mcast_data_size_bytes = num_tiles_k * tile_1x32_size
        input_mcast_params = setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=mcast_grid,
            src_cb=input_cb,
            src_tensor=input_tensor,
            dst_cb=gate_mm_input_cb,
            dst_tensor=mcast_output_tensor,
            sender_semaphore_id=mcast_data_sender_semaphore_id,
            receiver_semaphore_id=mcast_data_receiver_semaphore_id,
            data_size_bytes=input_mcast_data_size_bytes,
        )

        # ==================================================================
        # Gate MM (SRAM Matmul)
        # ==================================================================
        gate_mm_params = setup_sram_matmul(
            in0_cb=gate_mm_input_cb,
            in1_cb=gate_mm_weights_cb,
            out_cb=gate_mm_output_cb,
            weights_tensor=gate_mm_weights_tensor,
            output_tensor=gate_mm_output_tensor,
            k_num_tiles=num_tiles_k,
            fused_activation=MoeRoutedExpertOp.ACTIVATION_SIGMOID,
        )

        # Pre-built CB descriptors (tensor-backed, not from setup helpers)
        input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)
        gate_mm_input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gate_mm_input_cb, mcast_output_tensor)

        # ==================================================================
        # Gate MM Gather
        # ==================================================================
        gate_mm_output_tile = gate_mm_output_tensor.get_tile()
        gate_mm_output_tile_size = gate_mm_output_tile.get_tile_size(data_format)
        gate_mm_gather_data_size_bytes = gate_mm_params["out_w"] * gate_mm_output_tile_size

        gate_mm_gather_params = setup_gather(
            device=device,
            receiver_core=sender_core,
            sender_core_ranges=gate_mm_params["core_grid"],
            num_senders=gate_mm_params["num_cores"],
            data_size_bytes_per_sender=gate_mm_gather_data_size_bytes,
            src_cb=gate_mm_params["out_cb"],
            src_num_pages=gate_mm_params["out_w"],
            dst_cb=gate_input_cb,
            dst_tensor=gate_input_tensor,
            noc0_receiver_semaphore_id=gather_noc0_receiver_semaphore_id,
            noc1_receiver_semaphore_id=gather_noc1_receiver_semaphore_id,
            row_major=False,
            use_explicit_sender_index=False,
        )

        # ==================================================================
        # Gate
        # ==================================================================
        gate_params = setup_gate(
            input_cb=gate_input_cb,
            bias_cb=gate_bias_cb,
            indices_cb=gate_indices_cb,
            output_cb=gate_output_cb,
            output_indices_cb=gate_output_indices_cb,
            input_tensor=gate_input_tensor,
            bias_tensor=gate_bias_tensor,
            indices_tensor=gate_indices_tensor,
            output_scores_tensor=gate_output_scores_tensor,
            output_indices_tensor=gate_output_indices_tensor,
            eps=1e-20,
            scaling_factor=2.5,
            enable_sigmoid=False,
        )

        # ==================================================================
        # Index Mcast
        # ==================================================================
        gate_proj_cb_index_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gate_proj_cb_index, expert_index_tensor)
        index_tile = expert_index_tensor.get_tile()
        index_tile_size = index_tile.get_tile_size(expert_index_tensor.dtype)
        index_mcast_sender_semaphore_id = mcast_data_sender_semaphore_id
        index_mcast_receiver_semaphore_id = mcast_data_receiver_semaphore_id
        index_mcast_num_pages = 1
        index_mcast_data_size_bytes = index_tile_size

        # ==================================================================
        # Expert Scale Mcast
        # ==================================================================
        expert_scale_tile = expert_scale_tensor.get_tile()
        expert_scale_tile_size = expert_scale_tile.get_tile_size(expert_scale_tensor.dtype)
        expert_scale_mcast_num_pages = 1
        expert_scale_mcast_data_size_bytes = expert_scale_tile_size

        # ==================================================================
        # DRAM Streaming Matmul: gate_proj
        # ==================================================================
        gate_proj_params = setup_dram_matmul(
            device=device,
            weights_tensor=gate_proj_weights_tensor,
            output_tensor=gate_proj_output_tensor,
            working_buf_tensor=gate_proj_in1_buf_tensor,
            core_ranges=gate_proj_core_ranges,
            cb_in1_index=gate_proj_cb_in1,
            cb_out_index=gate_proj_cb_out,
            fp32_dest_acc_en=True,
            num_subblocks_k=4,
        )

        # ==================================================================
        # DRAM Streaming Matmul: up_proj
        # ==================================================================
        up_proj_params = setup_dram_matmul(
            device=device,
            weights_tensor=up_proj_weights_tensor,
            output_tensor=up_proj_mm_out_tensor,
            working_buf_tensor=up_proj_in1_buf_tensor,
            core_ranges=gate_proj_core_ranges,
            cb_in1_index=up_proj_cb_in1,
            cb_out_index=up_proj_cb_mm_out,
            fp32_dest_acc_en=True,
            num_subblocks_k=4,
        )

        # ==================================================================
        # Eltwise Mul: silu(gate_proj) * up_proj * expert_scale
        # ==================================================================
        mul_params = setup_eltwise_mul(
            in0_tensor=up_proj_mm_out_tensor,
            in1_tensor=gate_proj_output_tensor,
            out_tensor=fused_output_tensor,
            cb_in0_index=mul_cb_in0,
            cb_in1_index=mul_cb_in1,
            cb_out_index=mul_cb_out,
            per_core_n=gate_proj_params["per_core_n"],
            cb_scalar_index=mul_cb_scalar,
            cb_scalar_src_index=mul_cb_scalar_src,
            scalar_src_tensor=expert_scale_tensor,
            scalar_buf_tensor=mul_scalar_buf_tensor,
        )
        mul_num_tiles = mul_params["mul_num_tiles"]

        # ==================================================================
        # down_proj Gather
        # ==================================================================
        down_proj_gather_data_size_bytes = gate_proj_params["per_core_n"] * tile_1x32_size
        down_proj_gather_params = setup_gather(
            device=device,
            receiver_core=sender_core,
            sender_core_ranges=gate_proj_core_ranges,
            num_senders=num_gate_proj_cores,
            data_size_bytes_per_sender=down_proj_gather_data_size_bytes,
            src_cb=mul_cb_out,
            src_num_pages=mul_num_tiles,
            dst_cb=down_proj_gather_dst_cb,
            dst_tensor=down_proj_gather_output_tensor,
            noc0_receiver_semaphore_id=gather_noc0_receiver_semaphore_id,
            noc1_receiver_semaphore_id=gather_noc1_receiver_semaphore_id,
            row_major=True,
            use_explicit_sender_index=True,
        )

        # ==================================================================
        # down_proj Mcast
        # ==================================================================
        fused_output_shard = down_proj_gather_output_tensor.memory_config().shard_spec.shape
        down_proj_mcast_num_tiles = (fused_output_shard[0] * fused_output_shard[1]) // (
            TILE_1x32.tile_shape[0] * TILE_1x32.tile_shape[1]
        )
        down_proj_mcast_data_size_bytes = down_proj_mcast_num_tiles * tile_1x32_size
        down_proj_mcast_params = setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=mcast_grid,
            src_cb=down_proj_gather_dst_cb,
            src_tensor=down_proj_gather_output_tensor,
            dst_cb=down_proj_mcast_dst_cb,
            dst_tensor=down_proj_mcast_output_tensor,
            sender_semaphore_id=mcast_data_sender_semaphore_id,
            receiver_semaphore_id=mcast_data_receiver_semaphore_id,
            data_size_bytes=down_proj_mcast_data_size_bytes,
        )

        # ==================================================================
        # DRAM Streaming Matmul: down_proj
        # ==================================================================
        down_proj_params = setup_dram_matmul(
            device=device,
            weights_tensor=down_proj_weights_tensor,
            output_tensor=down_proj_output_tensor,
            working_buf_tensor=down_proj_in1_buf_tensor,
            core_ranges=gate_proj_core_ranges,
            cb_in1_index=down_proj_cb_in1,
            cb_out_index=down_proj_cb_out,
            fp32_dest_acc_en=True,
            num_subblocks_k=2,
        )

        # ==================================================================
        # Eltwise Add: down_proj + fused_add
        # ==================================================================
        add_params = setup_eltwise_add(
            in0_tensor=down_proj_output_tensor,
            in1_tensor=fused_add_tensor,
            out_tensor=final_output_tensor,
            cb_in0_index=add_cb_in0,
            cb_in1_index=add_cb_in1,
            cb_out_index=add_cb_out,
        )

        # ==================================================================
        # Per-core bank_id, vc, sender_idx
        # ==================================================================
        gate_proj_optimal_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        core_to_bank_id = {}
        for bank_id, core in enumerate(gate_proj_optimal_workers):
            core_to_bank_id[(core.x, core.y)] = bank_id

        bank_id_core_values = []
        vc_core_values = []
        sender_idx_core_values = []
        bank_ids = []
        for idx, core in enumerate(gate_proj_optimal_workers):
            bank_id = core_to_bank_id[(core.x, core.y)]
            vc = bank_id & 0x3
            for j in range(idx):
                prev_core = gate_proj_optimal_workers[j]
                if prev_core.y == core.y and (bank_ids[j] & 0x3) == (bank_id & 0x3):
                    vc = (vc + 1) & 0x3
                    break
            bank_ids.append(bank_id)
            bank_id_core_values.append((core, bank_id))
            vc_core_values.append((core, vc))
            sender_idx_core_values.append((core, idx))

        # ==================================================================
        # Return context
        # ==================================================================
        return _MoeRoutedExpertContext(
            # Device & mesh
            device=device,
            full_device_grid=full_device_grid,
            device_grid_size=device_grid_size,
            mesh_rows=mesh_rows,
            mesh_cols=mesh_cols,
            # Core grids
            sender_core=sender_core,
            input_core_grid=input_core_grid,
            mcast_grid=mcast_grid,
            gate_proj_core_ranges=gate_proj_core_ranges,
            num_gate_proj_cores=num_gate_proj_cores,
            # Data format & tiles
            data_format=data_format,
            tile_1x32_size=tile_1x32_size,
            num_tiles_k=num_tiles_k,
            # CB indices
            input_cb=input_cb,
            gate_mm_input_cb=gate_mm_input_cb,
            gate_mm_weights_cb=gate_mm_weights_cb,
            gate_mm_output_cb=gate_mm_output_cb,
            gate_input_cb=gate_input_cb,
            gate_bias_cb=gate_bias_cb,
            gate_indices_cb=gate_indices_cb,
            gate_output_cb=gate_output_cb,
            gate_output_indices_cb=gate_output_indices_cb,
            gate_proj_cb_in1=gate_proj_cb_in1,
            gate_proj_cb_index=gate_proj_cb_index,
            gate_proj_cb_out=gate_proj_cb_out,
            up_proj_cb_in1=up_proj_cb_in1,
            up_proj_cb_mm_out=up_proj_cb_mm_out,
            mul_cb_in0=mul_cb_in0,
            mul_cb_in1=mul_cb_in1,
            mul_cb_out=mul_cb_out,
            down_proj_gather_dst_cb=down_proj_gather_dst_cb,
            down_proj_mcast_dst_cb=down_proj_mcast_dst_cb,
            down_proj_cb_in1=down_proj_cb_in1,
            down_proj_cb_out=down_proj_cb_out,
            mul_cb_scalar_src=mul_cb_scalar_src,
            mul_cb_scalar=mul_cb_scalar,
            add_cb_in0=add_cb_in0,
            add_cb_in1=add_cb_in1,
            add_cb_out=add_cb_out,
            # Semaphore IDs
            mcast_data_sender_semaphore_id=mcast_data_sender_semaphore_id,
            mcast_data_receiver_semaphore_id=mcast_data_receiver_semaphore_id,
            gather_noc0_receiver_semaphore_id=gather_noc0_receiver_semaphore_id,
            gather_noc1_receiver_semaphore_id=gather_noc1_receiver_semaphore_id,
            expert_scale_mcast_sender_semaphore_id=expert_scale_mcast_sender_semaphore_id,
            expert_scale_mcast_receiver_semaphore_id=expert_scale_mcast_receiver_semaphore_id,
            # Setup result dicts
            input_mcast_params=input_mcast_params,
            gate_mm_params=gate_mm_params,
            gate_mm_gather_params=gate_mm_gather_params,
            gate_params=gate_params,
            gate_proj_params=gate_proj_params,
            up_proj_params=up_proj_params,
            mul_params=mul_params,
            down_proj_gather_params=down_proj_gather_params,
            down_proj_mcast_params=down_proj_mcast_params,
            down_proj_params=down_proj_params,
            add_params=add_params,
            # Index mcast
            index_mcast_sender_semaphore_id=index_mcast_sender_semaphore_id,
            index_mcast_receiver_semaphore_id=index_mcast_receiver_semaphore_id,
            index_mcast_num_pages=index_mcast_num_pages,
            index_mcast_data_size_bytes=index_mcast_data_size_bytes,
            # Expert scale mcast
            expert_scale_mcast_num_pages=expert_scale_mcast_num_pages,
            expert_scale_mcast_data_size_bytes=expert_scale_mcast_data_size_bytes,
            # Derived
            mul_num_tiles=mul_num_tiles,
            # Pre-built CB descriptors
            input_cb_descriptor=input_cb_descriptor,
            gate_mm_input_cb_descriptor=gate_mm_input_cb_descriptor,
            gate_proj_cb_index_descriptor=gate_proj_cb_index_descriptor,
            # Per-core values
            bank_id_core_values=bank_id_core_values,
            vc_core_values=vc_core_values,
            sender_idx_core_values=sender_idx_core_values,
            # Testing flag
            use_hardcoded_expert_index=use_hardcoded_expert_index,
        )

    @staticmethod
    def _build_compile_time_args(ctx, mesh_chip_id):
        """Build NCRISC, BRISC, and TRISC compile-time arg lists for routed expert."""

        ncrisc_named_compile_time_args = [
            # Input mcast (sender sharded buffer + receiver)
            ("mcast_src_cb", ctx.input_mcast_params["src_cb"]),
            ("mcast_src_num_pages", ctx.input_mcast_params["src_num_pages"]),
            ("mcast_data_receiver_semaphore", ctx.input_mcast_params["receiver_semaphore_id"]),
            ("mcast_dst_cb", ctx.input_mcast_params["dst_cb"]),
            ("mcast_dst_num_pages", ctx.input_mcast_params["dst_num_pages"]),
            # Gate matmul reader
            ("gate_mm_in0", ctx.gate_mm_params["in0_cb"]),
            ("gate_mm_in1", ctx.gate_mm_params["in1_cb"]),
            ("gate_mm_k_num_tiles", ctx.gate_mm_params["k_num_tiles"]),
            ("gate_mm_out_w", ctx.gate_mm_params["out_w"]),
            # Gate gather sender
            ("gather_dest_noc_x", ctx.gate_mm_gather_params["dest_noc_x"]),
            ("gather_dest_noc_y", ctx.gate_mm_gather_params["dest_noc_y"]),
            ("gather_data_size_bytes", ctx.gate_mm_gather_params["data_size_bytes"]),
            ("gather_receiver_semaphore_id", ctx.gate_mm_gather_params["receiver_semaphore_id"]),
            ("gather_src_cb", ctx.gate_mm_gather_params["src_cb"]),
            ("gather_src_num_pages", ctx.gate_mm_gather_params["src_num_pages"]),
            ("gather_sender_grid_start_x", ctx.gate_mm_gather_params["sender_grid_start_x"]),
            ("gather_sender_grid_start_y", ctx.gate_mm_gather_params["sender_grid_start_y"]),
            ("gather_sender_grid_end_x", ctx.gate_mm_gather_params["sender_grid_end_x"]),
            ("gather_sender_grid_end_y", ctx.gate_mm_gather_params["sender_grid_end_y"]),
            ("gather_row_major", ctx.gate_mm_gather_params["row_major"]),
            ("gather_receiver_data_addr", ctx.gate_mm_gather_params["receiver_data_addr"]),
            # Gate reader
            ("gate_input_cb", ctx.gate_params["input_cb"]),
            ("gate_bias_cb", ctx.gate_params["bias_cb"]),
            ("gate_input_indices_cb", ctx.gate_params["indices_cb"]),
            # Index mcast receiver
            ("index_mcast_receiver_semaphore", ctx.index_mcast_receiver_semaphore_id),
            ("gate_proj_cb_index", ctx.gate_proj_cb_index),
            ("index_mcast_num_pages", ctx.index_mcast_num_pages),
            # Expert scale mcast receiver
            ("expert_scale_mcast_receiver_semaphore", ctx.expert_scale_mcast_receiver_semaphore_id),
            ("mul_cb_scalar_src", ctx.mul_cb_scalar_src),
            ("expert_scale_mcast_num_pages", ctx.expert_scale_mcast_num_pages),
            # Mul reader (setup mul_in1 buffer)
            ("mul_cb_in1", ctx.mul_cb_in1),
            ("mul_num_tiles", ctx.mul_num_tiles),
            # down_proj gather sender
            ("down_proj_gather_dest_noc_x", ctx.down_proj_gather_params["dest_noc_x"]),
            ("down_proj_gather_dest_noc_y", ctx.down_proj_gather_params["dest_noc_y"]),
            ("down_proj_gather_data_size_bytes", ctx.down_proj_gather_params["data_size_bytes"]),
            ("down_proj_gather_receiver_semaphore_id", ctx.down_proj_gather_params["receiver_semaphore_id"]),
            ("down_proj_gather_src_cb", ctx.down_proj_gather_params["src_cb"]),
            ("down_proj_gather_src_num_pages", ctx.down_proj_gather_params["src_num_pages"]),
            ("down_proj_gather_sender_grid_start_x", ctx.down_proj_gather_params["sender_grid_start_x"]),
            ("down_proj_gather_sender_grid_start_y", ctx.down_proj_gather_params["sender_grid_start_y"]),
            ("down_proj_gather_sender_grid_end_x", ctx.down_proj_gather_params["sender_grid_end_x"]),
            ("down_proj_gather_sender_grid_end_y", ctx.down_proj_gather_params["sender_grid_end_y"]),
            ("down_proj_gather_row_major", ctx.down_proj_gather_params["row_major"]),
            ("down_proj_gather_receiver_data_addr", ctx.down_proj_gather_params["receiver_data_addr"]),
            # down_proj mcast receiver
            ("down_proj_mcast_receiver_semaphore", ctx.down_proj_mcast_params["receiver_semaphore_id"]),
            ("down_proj_mcast_dst_cb", ctx.down_proj_mcast_params["dst_cb"]),
            ("down_proj_mcast_dst_num_pages", ctx.down_proj_mcast_params["dst_num_pages"]),
            # Eltwise add
            ("add_cb_in0", ctx.add_cb_in0),
            ("add_cb_in1", ctx.add_cb_in1),
            ("add_cb_in0_wait_tiles", ctx.add_params["cb_in0_wait_tiles"]),
            ("add_cb_in1_wait_tiles", ctx.add_params["cb_in1_wait_tiles"]),
            # gate_proj DRAM matmul reader
            ("gate_proj_cb_in1", ctx.gate_proj_cb_in1),
            ("gate_proj_cb_out", ctx.gate_proj_cb_out),
            ("gate_proj_in1_tensor_addr", ctx.gate_proj_params["in1_tensor_addr"]),
            ("gate_proj_in1_page_size", ctx.gate_proj_params["in1_page_size"]),
            ("gate_proj_in1_num_pages", ctx.gate_proj_params["in1_num_pages"]),
            ("gate_proj_subblock_k", ctx.gate_proj_params["subblock_k"]),
            ("gate_proj_per_core_n", ctx.gate_proj_params["per_core_n"]),
            ("gate_proj_in1_block_size_bytes", ctx.gate_proj_params["in1_block_size_bytes"]),
            ("gate_proj_out_num_tiles", ctx.gate_proj_params["out_num_tiles"]),
            ("gate_proj_num_subblocks_k", ctx.gate_proj_params["num_subblocks_k"]),
            ("gate_proj_index_offset", mesh_chip_id),
            # up_proj DRAM matmul reader
            ("up_proj_cb_in1", ctx.up_proj_cb_in1),
            ("up_proj_in1_tensor_addr", ctx.up_proj_params["in1_tensor_addr"]),
            ("up_proj_in1_page_size", ctx.up_proj_params["in1_page_size"]),
            ("up_proj_in1_num_pages", ctx.up_proj_params["in1_num_pages"]),
            ("up_proj_subblock_k", ctx.up_proj_params["subblock_k"]),
            ("up_proj_per_core_n", ctx.up_proj_params["per_core_n"]),
            ("up_proj_in1_block_size_bytes", ctx.up_proj_params["in1_block_size_bytes"]),
            ("up_proj_out_num_tiles", ctx.up_proj_params["out_num_tiles"]),
            ("up_proj_num_subblocks_k", ctx.up_proj_params["num_subblocks_k"]),
            ("up_proj_cb_index", ctx.gate_proj_cb_index),
            ("gate_proj_in1_buf_addr", ctx.gate_proj_params["in1_buf_addr"]),
            ("up_proj_index_offset", mesh_chip_id),
            ("up_proj_cb_mm_out", ctx.up_proj_cb_mm_out),
            # down_proj DRAM matmul reader
            ("down_proj_cb_in1", ctx.down_proj_cb_in1),
            ("down_proj_cb_out", ctx.down_proj_cb_out),
            ("down_proj_in1_tensor_addr", ctx.down_proj_params["in1_tensor_addr"]),
            ("down_proj_in1_page_size", ctx.down_proj_params["in1_page_size"]),
            ("down_proj_in1_num_pages", ctx.down_proj_params["in1_num_pages"]),
            ("down_proj_subblock_k", ctx.down_proj_params["subblock_k"]),
            ("down_proj_per_core_n", ctx.down_proj_params["per_core_n"]),
            ("down_proj_in1_block_size_bytes", ctx.down_proj_params["in1_block_size_bytes"]),
            ("down_proj_out_num_tiles", ctx.down_proj_params["out_num_tiles"]),
            ("down_proj_num_subblocks_k", ctx.down_proj_params["num_subblocks_k"]),
            ("down_proj_cb_index", ctx.gate_proj_cb_index),
            ("down_proj_in1_buf_addr", ctx.down_proj_params["in1_buf_addr"]),
            ("down_proj_index_offset", mesh_chip_id),
            # Testing flag
            ("use_hardcoded_expert_index", 1 if ctx.use_hardcoded_expert_index else 0),
        ]

        brisc_named_compile_time_args = [
            # Input mcast sender
            ("mcast_dest_noc_start_x", ctx.input_mcast_params["dest_noc_start_x"]),
            ("mcast_dest_noc_start_y", ctx.input_mcast_params["dest_noc_start_y"]),
            ("mcast_dest_noc_end_x", ctx.input_mcast_params["dest_noc_end_x"]),
            ("mcast_dest_noc_end_y", ctx.input_mcast_params["dest_noc_end_y"]),
            ("mcast_num_cores", ctx.input_mcast_params["num_cores"]),
            ("mcast_data_sender_semaphore", ctx.input_mcast_params["sender_semaphore_id"]),
            ("mcast_data_receiver_semaphore", ctx.input_mcast_params["receiver_semaphore_id"]),
            ("mcast_data_size_bytes", ctx.input_mcast_params["data_size_bytes"]),
            ("mcast_src_cb", ctx.input_mcast_params["src_cb"]),
            ("mcast_dst_cb", ctx.input_mcast_params["dst_cb"]),
            ("mcast_src_num_pages", ctx.input_mcast_params["src_num_pages"]),
            ("mcast_is_part_of_receiver_grid", ctx.input_mcast_params["is_sender_part_of_receiver_grid"]),
            # Gate gather receiver
            ("gather_noc0_num_senders", ctx.gate_mm_gather_params["noc0_num_senders"]),
            ("gather_noc1_num_senders", ctx.gate_mm_gather_params["noc1_num_senders"]),
            ("gather_noc0_receiver_semaphore_id", ctx.gate_mm_gather_params["noc0_receiver_semaphore_id"]),
            ("gather_noc1_receiver_semaphore_id", ctx.gate_mm_gather_params["noc1_receiver_semaphore_id"]),
            ("gather_dst_cb", ctx.gate_mm_gather_params["dst_cb"]),
            ("gather_dst_num_pages", ctx.gate_mm_gather_params["dst_num_pages"]),
            # Gate writer
            ("gate_output_cb", ctx.gate_params["output_cb"]),
            ("gate_output_indices_cb", ctx.gate_params["output_indices_cb"]),
            # Index mcast sender
            ("index_mcast_sender_semaphore", ctx.index_mcast_sender_semaphore_id),
            ("index_mcast_receiver_semaphore", ctx.index_mcast_receiver_semaphore_id),
            ("index_mcast_data_size_bytes", ctx.index_mcast_data_size_bytes),
            ("index_mcast_num_pages", ctx.index_mcast_num_pages),
            ("gate_proj_cb_index", ctx.gate_proj_cb_index),
            # Expert scale mcast sender
            ("expert_scale_mcast_sender_semaphore", ctx.expert_scale_mcast_sender_semaphore_id),
            ("expert_scale_mcast_receiver_semaphore", ctx.expert_scale_mcast_receiver_semaphore_id),
            ("expert_scale_mcast_data_size_bytes", ctx.expert_scale_mcast_data_size_bytes),
            ("expert_scale_mcast_num_pages", ctx.expert_scale_mcast_num_pages),
            ("mul_cb_scalar_src", ctx.mul_cb_scalar_src),
            ("mul_cb_scalar", ctx.mul_cb_scalar),
            ("mul_scalar_index_offset", mesh_chip_id),
            # Mul writer
            ("mul_cb_out", ctx.mul_cb_out),
            ("mul_num_tiles", ctx.mul_num_tiles),
            # down_proj gather receiver
            ("down_proj_gather_noc0_num_senders", ctx.down_proj_gather_params["noc0_num_senders"]),
            ("down_proj_gather_noc1_num_senders", ctx.down_proj_gather_params["noc1_num_senders"]),
            ("down_proj_gather_noc0_receiver_semaphore_id", ctx.down_proj_gather_params["noc0_receiver_semaphore_id"]),
            ("down_proj_gather_noc1_receiver_semaphore_id", ctx.down_proj_gather_params["noc1_receiver_semaphore_id"]),
            ("down_proj_gather_dst_cb", ctx.down_proj_gather_params["dst_cb"]),
            ("down_proj_gather_dst_num_pages", ctx.down_proj_gather_params["dst_num_pages"]),
            # down_proj mcast sender
            ("down_proj_mcast_sender_semaphore", ctx.down_proj_mcast_params["sender_semaphore_id"]),
            ("down_proj_mcast_receiver_semaphore", ctx.down_proj_mcast_params["receiver_semaphore_id"]),
            ("down_proj_mcast_data_size_bytes", ctx.down_proj_mcast_params["data_size_bytes"]),
            ("down_proj_mcast_src_cb", ctx.down_proj_mcast_params["src_cb"]),
            ("down_proj_mcast_dst_cb", ctx.down_proj_mcast_params["dst_cb"]),
            ("down_proj_mcast_src_num_pages", ctx.down_proj_mcast_params["src_num_pages"]),
            # CB reset addresses for DRAM matmul working buffers
            ("gate_proj_in1_buf_addr", ctx.gate_proj_params["in1_buf_addr"]),
            ("down_proj_in1_buf_addr", ctx.down_proj_params["in1_buf_addr"]),
        ]

        trisc_named_compile_time_args = [
            # Gate matmul compute
            ("gate_mm_in0", ctx.gate_mm_params["in0_cb"]),
            ("gate_mm_in1", ctx.gate_mm_params["in1_cb"]),
            ("gate_mm_out", ctx.gate_mm_params["out_cb"]),
            ("gate_mm_k_num_tiles", ctx.gate_mm_params["k_num_tiles"]),
            ("gate_mm_out_w", ctx.gate_mm_params["out_w"]),
            ("gate_mm_fused_activation", ctx.gate_mm_params["fused_activation"]),
            # Gate compute
            ("gate_input_cb", ctx.gate_params["input_cb"]),
            ("gate_bias_cb", ctx.gate_params["bias_cb"]),
            ("gate_input_indices_cb", ctx.gate_params["indices_cb"]),
            ("gate_output_cb", ctx.gate_params["output_cb"]),
            ("gate_output_indices_cb", ctx.gate_params["output_indices_cb"]),
            ("gate_eps", ctx.gate_params["eps"]),
            ("gate_scaling_factor", ctx.gate_params["scaling_factor"]),
            ("gate_enable_sigmoid", ctx.gate_params["enable_sigmoid"]),
            # gate_proj compute
            ("gate_proj_cb_in0", ctx.gate_mm_params["in0_cb"]),
            ("gate_proj_cb_in1", ctx.gate_proj_cb_in1),
            ("gate_proj_cb_out", ctx.gate_proj_cb_out),
            ("gate_proj_subblock_k", ctx.gate_proj_params["subblock_k"]),
            ("gate_proj_per_core_n", ctx.gate_proj_params["per_core_n"]),
            ("gate_proj_subblock_w", ctx.gate_proj_params["subblock_w"]),
            ("gate_proj_num_subblocks_k", ctx.gate_proj_params["num_subblocks_k"]),
            ("gate_proj_tile_r_dim", ctx.gate_proj_params["tile_r_dim"]),
            ("gate_proj_fuse_silu", 1),
            ("gate_proj_fp32_dest_acc_en", 1),
            # up_proj compute
            ("up_proj_cb_in0", ctx.gate_mm_params["in0_cb"]),
            ("up_proj_cb_in1", ctx.up_proj_cb_in1),
            ("up_proj_subblock_k", ctx.up_proj_params["subblock_k"]),
            ("up_proj_per_core_n", ctx.up_proj_params["per_core_n"]),
            ("up_proj_subblock_w", ctx.up_proj_params["subblock_w"]),
            ("up_proj_num_subblocks_k", ctx.up_proj_params["num_subblocks_k"]),
            ("up_proj_tile_r_dim", ctx.up_proj_params["tile_r_dim"]),
            ("up_proj_fuse_silu", 0),
            ("up_proj_fp32_dest_acc_en", 1),
            ("up_proj_cb_mm_out", ctx.up_proj_cb_mm_out),
            # Mul compute
            ("mul_cb_in0", ctx.mul_cb_in0),
            ("mul_cb_in1", ctx.mul_cb_in1),
            ("mul_cb_out", ctx.mul_cb_out),
            ("mul_num_tiles", ctx.mul_num_tiles),
            ("mul_cb_scalar", ctx.mul_cb_scalar),
            ("mul_fp32_dest_acc_en", 1),
            ("up_proj_per_core_n", ctx.up_proj_params["per_core_n"]),
            # down_proj compute
            ("down_proj_cb_in0", ctx.down_proj_mcast_dst_cb),
            ("down_proj_cb_in1", ctx.down_proj_cb_in1),
            ("down_proj_cb_out", ctx.down_proj_cb_out),
            ("down_proj_subblock_k", ctx.down_proj_params["subblock_k"]),
            ("down_proj_per_core_n", ctx.down_proj_params["per_core_n"]),
            ("down_proj_subblock_w", ctx.down_proj_params["subblock_w"]),
            ("down_proj_num_subblocks_k", ctx.down_proj_params["num_subblocks_k"]),
            ("down_proj_tile_r_dim", ctx.down_proj_params["tile_r_dim"]),
            ("down_proj_fuse_silu", 0),
            ("down_proj_fp32_dest_acc_en", 1),
            # Testing flag
            ("use_hardcoded_expert_index", 1 if ctx.use_hardcoded_expert_index else 0),
            # Eltwise add compute
            ("add_cb_in0", ctx.add_cb_in0),
            ("add_cb_in1", ctx.add_cb_in1),
            ("add_cb_out", ctx.add_cb_out),
            ("add_num_tiles", ctx.add_params["num_tiles"]),
            ("add_cb_in0_wait_tiles", ctx.add_params["cb_in0_wait_tiles"]),
            ("add_cb_in1_wait_tiles", ctx.add_params["cb_in1_wait_tiles"]),
            ("add_slice_size_bytes", ctx.add_params["slice_size_bytes"]),
            # CB reset addresses for DRAM matmul working buffers
            ("gate_proj_in1_buf_addr", ctx.gate_proj_params["in1_buf_addr"]),
            ("down_proj_in1_buf_addr", ctx.down_proj_params["in1_buf_addr"]),
        ]

        return ncrisc_named_compile_time_args, brisc_named_compile_time_args, trisc_named_compile_time_args

    @staticmethod
    def _build_cb_descriptors(ctx):
        """Build circular buffer descriptors for routed expert."""
        return [
            ctx.input_cb_descriptor,
            ctx.gate_mm_input_cb_descriptor,
            ctx.gate_mm_params["weights_cb_descriptor"],
            ctx.gate_mm_params["output_cb_descriptor"],
            ctx.gate_params["input_cb_descriptor"],
            ctx.gate_params["bias_cb_descriptor"],
            ctx.gate_params["indices_cb_descriptor"],
            ctx.gate_params["output_cb_descriptor"],
            ctx.gate_params["output_indices_cb_descriptor"],
            ctx.gate_proj_params["cb_in1_descriptor"],  # Shared by gate_proj and up_proj
            ctx.gate_proj_cb_index_descriptor,
            ctx.gate_proj_params["cb_out_descriptor"],
            ctx.up_proj_params["cb_out_descriptor"],
            ctx.mul_params["cb_in0_descriptor"],
            ctx.mul_params["cb_in1_descriptor"],
            ctx.mul_params["cb_out_descriptor"],
            ctx.mul_params["cb_scalar_src_descriptor"],
            ctx.mul_params["cb_scalar_descriptor"],
            ctx.down_proj_gather_params["dst_cb_descriptor"],
            ctx.down_proj_mcast_params["dst_cb_descriptor"],
            ctx.down_proj_params["cb_in1_descriptor"],
            ctx.down_proj_params["cb_out_descriptor"],
            ctx.add_params["cb_in0_descriptor"],
            ctx.add_params["cb_in1_descriptor"],
            ctx.add_params["cb_out_descriptor"],
        ]

    @staticmethod
    def _build_core_descriptors(ctx):
        """Build unified and per-core compile-time core descriptors for routed expert."""
        unified_compile_time_core_descriptors = [
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_sender_core",
                core_range=ctx.sender_core,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_mcast_grid_core",
                core_range=ctx.mcast_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_gate_mm_core",
                core_range=ctx.gate_mm_params["core_grid"],
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_gate_proj_core",
                core_range=ctx.gate_proj_core_ranges,
                value=1,
                other_value=0,
            ),
        ]

        per_core_compile_time_descriptors = [
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="gate_proj_bank_id",
                core_values=ctx.bank_id_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="gate_proj_vc",
                core_values=ctx.vc_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="up_proj_bank_id",
                core_values=ctx.bank_id_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="up_proj_vc",
                core_values=ctx.vc_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="down_proj_bank_id",
                core_values=ctx.bank_id_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="down_proj_vc",
                core_values=ctx.vc_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="down_proj_gather_sender_idx",
                core_values=ctx.sender_idx_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="add_sender_index",
                core_values=ctx.add_params["sender_index_core_values"],
                other_value=0,
            ),
        ]

        return unified_compile_time_core_descriptors, per_core_compile_time_descriptors


class MoeSharedExpertOp:
    """
    Shared expert setup for the fused MoE kernel.

    Provides context setup and build methods for shared expert components
    (Gate/Up KN-sliced matmul, etc.). Does not execute on its own;
    used by MoeOp to compose the fused kernel.
    """

    @staticmethod
    def _setup_dimensions(
        device,
        shared_gate_up_weights_tensor,
        shared_bias_tensor,
        shared_residual_mcast_dst_tensor,
        shared_down_mcast_dst_tensor,
        shared_down_weights_tensor,
        num_tiles_k,
        tile_1x32_size,
        data_format,
        input_tile,
        input_tile_size,
        sender_core,
        sender_core_grid,
        mcast_grid,
        k_parallel=8,
        n_parallel=8,
    ):
        """
        Compute shared expert dimensions and build _MoeSharedExpertContext.

        Args:
            device: TT device (single chip)
            shared_gate_up_weights_tensor: Gate/Up weights tensor for shared expert
            shared_bias_tensor: Bias/residual tensor [1, N] on sender core
            shared_residual_mcast_dst_tensor: Destination tensor for residual mcast (on mcast grid)
            num_tiles_k: Number of K tiles (from routed expert context)
            tile_1x32_size: Size of a 1x32 tile in bytes
            data_format: Data format (dtype)
            input_tile: Input tile format (e.g., Tile((1, 32)))
            input_tile_size: Size of input tile in bytes
            sender_core: The mcast/gather core (e.g., CoreCoord(12, 9))
            sender_core_grid: CoreRangeSet for sender core
            mcast_grid: CoreRangeSet for mcast destination grid (same as routed input mcast)
            k_parallel: K parallelism factor
            n_parallel: N parallelism factor

        Returns:
            _MoeSharedExpertContext
        """
        from models.demos.deepseek_v3_b1.fused_ops.shared_expert.op import SharedExpertOp

        # ==================================================================
        # CB indices
        # ==================================================================
        shared_gu_weights_cb = 26
        shared_gu_out_cb = 27
        shared_group1_cb = 28  # gate gather dst on sender
        shared_group2_cb = 29  # up gather dst on sender
        shared_intermed_cb = 30  # gated reduce intermediate on sender
        shared_mcast_src_cb = 31  # gated reduce output on sender
        shared_residual_mcast_src_cb = 32  # residual mcast source on sender
        shared_residual_mcast_dst_cb = 33  # residual mcast destination on 112 matmul cores
        shared_down_mcast_dst_cb = 34  # down mcast destination (gated reduce output → all 130 cores)
        shared_down_matmul_in1_cb = 35  # down proj weights (112 matmul cores, tensor-backed)
        shared_down_matmul_out_cb = 36  # down proj matmul output (112 matmul cores, manual)
        shared_residual_add_out_cb = 37  # residual add output (112 matmul cores, manual)

        # ==================================================================
        # Dimensions
        # ==================================================================
        num_compute_cores = 64  # per branch (gate/up)
        assert k_parallel * n_parallel == num_compute_cores
        gu_k_per_core = num_tiles_k // k_parallel
        gu_act_total_tiles = num_tiles_k
        gu_weights_num_pages = gu_k_per_core  # tiles per core

        tiles_per_k = k_parallel
        k_num_tiles = n_parallel
        total_gather_tiles = num_compute_cores  # 64

        # ==================================================================
        # Face-view parameters (for gated reduce)
        # ==================================================================
        tile_h, tile_w = input_tile.tile_shape
        use_face_view = can_use_face_view(tile_h, tile_w, tiles_per_k, k_num_tiles)

        if use_face_view:
            face_tile = ttnn.Tile([FACE_HEIGHT, FACE_WIDTH])
            face_tile_desc = ttnn.TileDescriptor(FACE_HEIGHT, FACE_WIDTH, False)
            face_tile_size = face_tile.get_tile_size(data_format)
            kernel_tiles_per_k = tiles_per_k
            kernel_k_num_tiles = 1
            mcast_src_num_pages = 1
            reduce_tile_size = face_tile_size
        else:
            face_tile_desc = None
            face_tile_size = None
            kernel_tiles_per_k = tiles_per_k
            kernel_k_num_tiles = k_num_tiles
            mcast_src_num_pages = k_num_tiles
            reduce_tile_size = input_tile_size

        gu_gather_data_size_bytes = input_tile_size  # each compute core sends 1 tile

        # ==================================================================
        # Semaphore IDs (offset from routed expert's semaphores 0-5)
        # ==================================================================
        ag_receiver_semaphore_id = 6
        bg_receiver_semaphore_id = 7
        ag_noc1_receiver_semaphore_id = 8
        bg_noc1_receiver_semaphore_id = 9
        residual_mcast_sender_semaphore_id = 10
        residual_mcast_receiver_semaphore_id = 11
        down_mcast_sender_semaphore_id = 12
        down_mcast_receiver_semaphore_id = 13

        # ==================================================================
        # Core grids
        # ==================================================================
        a_cores_list, b_cores_list = SharedExpertOp.build_ab_grids()
        compute_cores_list = a_cores_list + b_cores_list
        compute_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in compute_cores_list])
        a_compute_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in a_cores_list])
        b_compute_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in b_cores_list])

        # NOC coordinates for gather destination (sender core)
        gather_dest_noc_core = device.worker_core_from_logical_core(sender_core)

        # ==================================================================
        # Dummy tensors for gather destination CBs (on sender core)
        # ==================================================================
        ag_dummy_shape = (total_gather_tiles, 32)
        ag_dummy_shard_spec = ttnn.ShardSpec(sender_core_grid, ag_dummy_shape, ttnn.ShardOrientation.ROW_MAJOR)
        ag_dummy_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ag_dummy_shard_spec
        )
        ag_dummy_tensor = ttnn.from_torch(
            torch.zeros(total_gather_tiles, 32, dtype=torch.bfloat16),
            dtype=data_format,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ag_dummy_mem,
            tile=input_tile,
        )
        bg_dummy_tensor = ttnn.from_torch(
            torch.zeros(total_gather_tiles, 32, dtype=torch.bfloat16),
            dtype=data_format,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ag_dummy_mem,
            tile=input_tile,
        )
        ag_receiver_data_addr = ag_dummy_tensor.buffer_address()
        bg_receiver_data_addr = bg_dummy_tensor.buffer_address()

        # ==================================================================
        # CB descriptors
        # ==================================================================
        # CB 26: Gate/Up weights (tensor-backed on 128 compute cores)
        gu_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            shared_gu_weights_cb, shared_gate_up_weights_tensor
        )

        # CB 27: Gate/Up matmul output (1 tile on 128 compute cores)
        input_tile_desc = ttnn.TileDescriptor(input_tile)
        gu_out_format = ttnn.CBFormatDescriptor(
            buffer_index=shared_gu_out_cb,
            data_format=data_format,
            page_size=input_tile_size,
            tile=input_tile_desc,
        )
        gu_out_cb_descriptor = ttnn.CBDescriptor(
            total_size=input_tile_size,
            core_ranges=compute_core_grid,
            format_descriptors=[gu_out_format],
        )

        # CB 28: Gate gather dst (tensor-backed on sender, face-view aliased)
        group1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(shared_group1_cb, ag_dummy_tensor)
        if use_face_view:
            group1_cb_descriptor.format_descriptors[0].tile = face_tile_desc
            group1_cb_descriptor.format_descriptors[0].page_size = face_tile_size

        # CB 29: Up gather dst (tensor-backed on sender, face-view aliased)
        group2_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(shared_group2_cb, bg_dummy_tensor)
        if use_face_view:
            group2_cb_descriptor.format_descriptors[0].tile = face_tile_desc
            group2_cb_descriptor.format_descriptors[0].page_size = face_tile_size

        # CB 30: Intermediate (2 tiles on sender core)
        tile_desc_for_reduce = face_tile_desc if use_face_view else input_tile_desc
        intermed_format = ttnn.CBFormatDescriptor(
            buffer_index=shared_intermed_cb,
            data_format=data_format,
            page_size=reduce_tile_size,
            tile=tile_desc_for_reduce,
        )
        intermed_cb_descriptor = ttnn.CBDescriptor(
            total_size=2 * reduce_tile_size,
            core_ranges=sender_core_grid,
            format_descriptors=[intermed_format],
        )

        # CB 31: Mcast source / gated reduce output (on sender core)
        mcast_src_format = ttnn.CBFormatDescriptor(
            buffer_index=shared_mcast_src_cb,
            data_format=data_format,
            page_size=reduce_tile_size,
            tile=tile_desc_for_reduce,
        )
        mcast_src_cb_descriptor = ttnn.CBDescriptor(
            total_size=mcast_src_num_pages * reduce_tile_size,
            core_ranges=sender_core_grid,
            format_descriptors=[mcast_src_format],
        )

        # ==================================================================
        # Residual Mcast
        # ==================================================================
        bias_shard_spec = shared_bias_tensor.memory_config().shard_spec
        bias_shard_shape = bias_shard_spec.shape
        bias_tile = shared_bias_tensor.get_tile()
        residual_mcast_dst_num_pages = (bias_shard_shape[0] * bias_shard_shape[1]) // (
            bias_tile.tile_shape[0] * bias_tile.tile_shape[1]
        )
        residual_mcast_data_size_bytes = residual_mcast_dst_num_pages * bias_tile.get_tile_size(data_format)

        residual_mcast_params = setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=shared_residual_mcast_dst_tensor.memory_config().shard_spec.grid,
            src_cb=shared_residual_mcast_src_cb,
            src_tensor=shared_bias_tensor,
            dst_cb=shared_residual_mcast_dst_cb,
            dst_tensor=shared_residual_mcast_dst_tensor,
            sender_semaphore_id=residual_mcast_sender_semaphore_id,
            receiver_semaphore_id=residual_mcast_receiver_semaphore_id,
            data_size_bytes=residual_mcast_data_size_bytes,
        )
        residual_mcast_src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            shared_residual_mcast_src_cb, shared_bias_tensor
        )
        residual_mcast_dst_cb_descriptor = residual_mcast_params["dst_cb_descriptor"]

        # ==================================================================
        # Down Mcast (gated reduce output → all 130 cores for down proj)
        # ==================================================================
        down_mcast_data_size_bytes = mcast_src_num_pages * reduce_tile_size
        down_mcast_grid = shared_down_mcast_dst_tensor.memory_config().shard_spec.grid
        down_mcast_src_num_pages = mcast_src_num_pages
        down_mcast_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            shared_down_mcast_dst_cb, shared_down_mcast_dst_tensor
        )

        # ==================================================================
        # Down Proj Matmul (SRAM matmul on 112 non-DRAM cores)
        # ==================================================================
        from models.demos.deepseek_v3_b1.fused_ops.down_proj.op import DownProj

        TILE_1x32 = ttnn.Tile([1, 32])
        down_weights_shard_spec = shared_down_weights_tensor.memory_config().shard_spec
        down_n_per_core = down_weights_shard_spec.shape[1]
        down_matmul_out_w_per_core = down_n_per_core // TILE_1x32.tile_shape[1]
        down_matmul_k_num_tiles = n_parallel  # K_down_tiles = n_parallel
        down_weights_num_pages = down_matmul_k_num_tiles * down_matmul_out_w_per_core

        down_matmul_in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            shared_down_matmul_in1_cb, shared_down_weights_tensor
        )

        tile_1x32_descriptor = ttnn.TileDescriptor(TILE_1x32)
        down_matmul_out_format = ttnn.CBFormatDescriptor(
            buffer_index=shared_down_matmul_out_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=tile_1x32_descriptor,
        )
        down_matmul_out_cb_descriptor = ttnn.CBDescriptor(
            total_size=down_matmul_out_w_per_core * tile_1x32_size,
            core_ranges=DownProj.build_matmul_core_grid(),
            format_descriptors=[down_matmul_out_format],
        )

        # ==================================================================
        # Residual Add (matmul_out + residual → residual_add_out on 112 cores)
        # ==================================================================
        residual_add_total_in1_tiles = DownProj.NUM_MATMUL_CORES * down_matmul_out_w_per_core
        residual_add_out_format = ttnn.CBFormatDescriptor(
            buffer_index=shared_residual_add_out_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=tile_1x32_descriptor,
        )
        residual_add_out_cb_descriptor = ttnn.CBDescriptor(
            total_size=down_matmul_out_w_per_core * tile_1x32_size,
            core_ranges=DownProj.build_matmul_core_grid(),
            format_descriptors=[residual_add_out_format],
        )

        # ==================================================================
        # Per-core values
        # ==================================================================
        # K-offset for matmul
        gu_k_offset_core_values = [(core, (i // n_parallel) * gu_k_per_core) for i, core in enumerate(a_cores_list)] + [
            (core, (i // n_parallel) * gu_k_per_core) for i, core in enumerate(b_cores_list)
        ]

        # Gather sender indices (determines where each core writes in the gather dst CB)
        def compute_gu_sender_indices(cores_list):
            indices = []
            for linear_id, core in enumerate(cores_list):
                k_idx = linear_id // n_parallel
                n_idx = linear_id % n_parallel
                if use_face_view:
                    sender_idx = k_idx * n_parallel + n_idx
                else:
                    sender_idx = n_idx * tiles_per_k + k_idx
                indices.append((core, sender_idx))
            return indices

        ag_sender_idx_core_values = compute_gu_sender_indices(a_cores_list)
        bg_sender_idx_core_values = compute_gu_sender_indices(b_cores_list)

        return _MoeSharedExpertContext(
            gu_weights_cb=shared_gu_weights_cb,
            gu_out_cb=shared_gu_out_cb,
            gu_k_per_core=gu_k_per_core,
            gu_act_total_tiles=gu_act_total_tiles,
            gu_weights_num_pages=gu_weights_num_pages,
            compute_core_grid=compute_core_grid,
            gu_k_offset_core_values=gu_k_offset_core_values,
            gu_weights_cb_descriptor=gu_weights_cb_descriptor,
            gu_out_cb_descriptor=gu_out_cb_descriptor,
            # Gather + Reduce
            a_cores_list=a_cores_list,
            b_cores_list=b_cores_list,
            a_compute_grid=a_compute_grid,
            b_compute_grid=b_compute_grid,
            group1_cb=shared_group1_cb,
            group2_cb=shared_group2_cb,
            intermed_cb=shared_intermed_cb,
            mcast_src_cb=shared_mcast_src_cb,
            n_parallel=n_parallel,
            k_parallel=k_parallel,
            tiles_per_k=tiles_per_k,
            k_num_tiles=k_num_tiles,
            num_compute_cores=num_compute_cores,
            use_face_view=use_face_view,
            face_tile_desc=face_tile_desc,
            face_tile_size=face_tile_size,
            kernel_tiles_per_k=kernel_tiles_per_k,
            kernel_k_num_tiles=kernel_k_num_tiles,
            mcast_src_num_pages=mcast_src_num_pages,
            reduce_tile_size=reduce_tile_size,
            input_tile_size=input_tile_size,
            gu_gather_data_size_bytes=gu_gather_data_size_bytes,
            total_gather_tiles=total_gather_tiles,
            gather_dest_noc_core=gather_dest_noc_core,
            mcast_gather_core_grid=sender_core_grid,
            ag_receiver_semaphore_id=ag_receiver_semaphore_id,
            bg_receiver_semaphore_id=bg_receiver_semaphore_id,
            ag_noc1_receiver_semaphore_id=ag_noc1_receiver_semaphore_id,
            bg_noc1_receiver_semaphore_id=bg_noc1_receiver_semaphore_id,
            ag_dummy_tensor=ag_dummy_tensor,
            bg_dummy_tensor=bg_dummy_tensor,
            ag_receiver_data_addr=ag_receiver_data_addr,
            bg_receiver_data_addr=bg_receiver_data_addr,
            group1_cb_descriptor=group1_cb_descriptor,
            group2_cb_descriptor=group2_cb_descriptor,
            intermed_cb_descriptor=intermed_cb_descriptor,
            mcast_src_cb_descriptor=mcast_src_cb_descriptor,
            ag_sender_idx_core_values=ag_sender_idx_core_values,
            bg_sender_idx_core_values=bg_sender_idx_core_values,
            # Residual Mcast
            residual_mcast_src_cb=shared_residual_mcast_src_cb,
            residual_mcast_dst_cb=shared_residual_mcast_dst_cb,
            residual_mcast_sender_semaphore_id=residual_mcast_sender_semaphore_id,
            residual_mcast_receiver_semaphore_id=residual_mcast_receiver_semaphore_id,
            residual_mcast_params=residual_mcast_params,
            residual_mcast_src_cb_descriptor=residual_mcast_src_cb_descriptor,
            residual_mcast_dst_cb_descriptor=residual_mcast_dst_cb_descriptor,
            residual_mcast_dst_dummy_tensor=shared_residual_mcast_dst_tensor,
            # Down Mcast
            down_mcast_dst_cb=shared_down_mcast_dst_cb,
            down_mcast_sender_semaphore_id=down_mcast_sender_semaphore_id,
            down_mcast_receiver_semaphore_id=down_mcast_receiver_semaphore_id,
            down_mcast_data_size_bytes=down_mcast_data_size_bytes,
            down_mcast_src_num_pages=down_mcast_src_num_pages,
            down_mcast_dst_cb_descriptor=down_mcast_dst_cb_descriptor,
            down_mcast_dst_dummy_tensor=shared_down_mcast_dst_tensor,
            # Down Proj Matmul
            down_matmul_in1_cb=shared_down_matmul_in1_cb,
            down_matmul_out_cb=shared_down_matmul_out_cb,
            down_matmul_k_num_tiles=down_matmul_k_num_tiles,
            down_matmul_out_w_per_core=down_matmul_out_w_per_core,
            down_matmul_in1_cb_descriptor=down_matmul_in1_cb_descriptor,
            down_matmul_out_cb_descriptor=down_matmul_out_cb_descriptor,
            down_weights_num_pages=down_weights_num_pages,
            # Residual Add
            residual_add_out_cb=shared_residual_add_out_cb,
            residual_add_total_in1_tiles=residual_add_total_in1_tiles,
            residual_add_out_cb_descriptor=residual_add_out_cb_descriptor,
        )

    @staticmethod
    def _build_compile_time_args(shared_ctx, input_mcast_dst_cb, input_mcast_params):
        """
        Build shared expert compile-time args to append to routed expert args.

        Args:
            shared_ctx: _MoeSharedExpertContext
            input_mcast_dst_cb: The CB index for the mcast destination (shared with routed)
            input_mcast_params: Input mcast params (reuse NOC coords for residual mcast)

        Returns:
            (ncrisc_args, brisc_args, trisc_args) - lists of named compile-time arg tuples
        """
        ctx = shared_ctx
        ncrisc_args = [
            # Gate/Up weights setup
            ("shared_gu_weights_cb", ctx.gu_weights_cb),
            ("shared_gu_weights_num_pages", ctx.gu_weights_num_pages),
            # Gate gather (A) sender
            ("shared_ag_dest_noc_x", ctx.gather_dest_noc_core.x),
            ("shared_ag_dest_noc_y", ctx.gather_dest_noc_core.y),
            ("shared_ag_data_size_bytes", ctx.gu_gather_data_size_bytes),
            ("shared_ag_receiver_semaphore_id", ctx.ag_receiver_semaphore_id),
            ("shared_ag_src_cb", ctx.gu_out_cb),
            ("shared_ag_src_num_pages", 1),
            ("shared_ag_receiver_data_addr", ctx.ag_receiver_data_addr),
            # Up gather (B) sender
            ("shared_bg_dest_noc_x", ctx.gather_dest_noc_core.x),
            ("shared_bg_dest_noc_y", ctx.gather_dest_noc_core.y),
            ("shared_bg_data_size_bytes", ctx.gu_gather_data_size_bytes),
            ("shared_bg_receiver_semaphore_id", ctx.bg_receiver_semaphore_id),
            ("shared_bg_src_cb", ctx.gu_out_cb),
            ("shared_bg_src_num_pages", 1),
            ("shared_bg_receiver_data_addr", ctx.bg_receiver_data_addr),
            # Residual mcast receiver
            ("shared_residual_mcast_data_receiver_semaphore", ctx.residual_mcast_receiver_semaphore_id),
            ("shared_residual_mcast_dst_cb", ctx.residual_mcast_dst_cb),
            ("shared_residual_mcast_dst_num_pages", ctx.residual_mcast_params["dst_num_pages"]),
            # Residual mcast src (needed for setup_sharded_buffer on sender)
            ("shared_residual_mcast_src_cb", ctx.residual_mcast_src_cb),
            ("shared_residual_mcast_src_num_pages", ctx.residual_mcast_params["src_num_pages"]),
            # Down mcast receiver
            ("shared_down_mcast_data_receiver_semaphore", ctx.down_mcast_receiver_semaphore_id),
            ("shared_down_mcast_dst_cb", ctx.down_mcast_dst_cb),
            ("shared_down_mcast_dst_num_pages", ctx.down_matmul_k_num_tiles),
            # Down proj weights (setup_sharded_buffer on 112 matmul cores)
            ("shared_down_matmul_in1", ctx.down_matmul_in1_cb),
            ("shared_down_matmul_k_num_tiles", ctx.down_matmul_k_num_tiles),
            ("shared_down_matmul_out_w_per_core", ctx.down_matmul_out_w_per_core),
        ]
        brisc_args = [
            # Gate gather (A) receiver
            ("shared_ag_noc0_num_senders", ctx.num_compute_cores),
            ("shared_ag_noc0_receiver_semaphore_id", ctx.ag_receiver_semaphore_id),
            ("shared_ag_noc1_receiver_semaphore_id", ctx.ag_noc1_receiver_semaphore_id),
            ("shared_ag_dst_cb", ctx.group1_cb),
            ("shared_ag_dst_num_pages", ctx.kernel_tiles_per_k if ctx.use_face_view else ctx.total_gather_tiles),
            # Up gather (B) receiver
            ("shared_bg_noc0_num_senders", ctx.num_compute_cores),
            ("shared_bg_noc0_receiver_semaphore_id", ctx.bg_receiver_semaphore_id),
            ("shared_bg_noc1_receiver_semaphore_id", ctx.bg_noc1_receiver_semaphore_id),
            ("shared_bg_dst_cb", ctx.group2_cb),
            ("shared_bg_dst_num_pages", ctx.kernel_tiles_per_k if ctx.use_face_view else ctx.total_gather_tiles),
            # Residual mcast sender (CTArgs reused from routed mcast; only need semaphores, CBs, sizes)
            ("shared_residual_mcast_data_sender_semaphore", ctx.residual_mcast_sender_semaphore_id),
            ("shared_residual_mcast_data_receiver_semaphore", ctx.residual_mcast_receiver_semaphore_id),
            ("shared_residual_mcast_data_size_bytes", ctx.residual_mcast_params["data_size_bytes"]),
            ("shared_residual_mcast_src_cb", ctx.residual_mcast_src_cb),
            ("shared_residual_mcast_src_num_pages", ctx.residual_mcast_params["src_num_pages"]),
            ("shared_residual_mcast_dst_cb", ctx.residual_mcast_dst_cb),
            # Down mcast sender (CTArgs reused from routed mcast; only need semaphores, CBs, sizes)
            ("shared_down_mcast_data_sender_semaphore", ctx.down_mcast_sender_semaphore_id),
            ("shared_down_mcast_data_receiver_semaphore", ctx.down_mcast_receiver_semaphore_id),
            ("shared_down_mcast_data_size_bytes", ctx.down_mcast_data_size_bytes),
            ("shared_down_mcast_src_cb", ctx.mcast_src_cb),  # gated reduce output (CB 31)
            ("shared_down_mcast_src_num_pages", ctx.down_mcast_src_num_pages),
            ("shared_down_mcast_dst_cb", ctx.down_mcast_dst_cb),
        ]
        trisc_args = [
            # Gate/Up matmul
            ("shared_gu_act_cb", input_mcast_dst_cb),
            ("shared_gu_weights_cb", ctx.gu_weights_cb),
            ("shared_gu_out_cb", ctx.gu_out_cb),
            ("shared_gu_k_per_core", ctx.gu_k_per_core),
            ("shared_gu_act_total_tiles", ctx.gu_act_total_tiles),
            # Gated reduce
            ("shared_gated_reduce_group1_cb", ctx.group1_cb),
            ("shared_gated_reduce_group2_cb", ctx.group2_cb),
            ("shared_gated_reduce_intermed_cb", ctx.intermed_cb),
            ("shared_gated_reduce_mcast_src_cb", ctx.mcast_src_cb),
            ("shared_gated_reduce_tiles_per_k", ctx.kernel_tiles_per_k),
            ("shared_gated_reduce_k_num_tiles", ctx.kernel_k_num_tiles),
            # Down proj matmul
            ("shared_down_matmul_in0", ctx.down_mcast_dst_cb),  # mcast'd activation
            ("shared_down_matmul_in1", ctx.down_matmul_in1_cb),
            ("shared_down_matmul_out", ctx.down_matmul_out_cb),
            ("shared_down_matmul_k_num_tiles", ctx.down_matmul_k_num_tiles),
            ("shared_down_matmul_out_w_per_core", ctx.down_matmul_out_w_per_core),
            # Residual add
            ("shared_residual_add_in0", ctx.down_matmul_out_cb),  # matmul output
            ("shared_residual_add_in1", ctx.residual_mcast_dst_cb),  # residual from mcast
            ("shared_residual_add_out", ctx.residual_add_out_cb),
            ("shared_residual_add_out_w", ctx.down_matmul_out_w_per_core),
            ("shared_residual_add_total_in1_tiles", ctx.residual_add_total_in1_tiles),
        ]
        return ncrisc_args, brisc_args, trisc_args

    @staticmethod
    def _build_cb_descriptors(shared_ctx):
        """Build CB descriptors for shared expert."""
        return [
            shared_ctx.gu_weights_cb_descriptor,
            shared_ctx.gu_out_cb_descriptor,
            shared_ctx.group1_cb_descriptor,
            shared_ctx.group2_cb_descriptor,
            shared_ctx.intermed_cb_descriptor,
            shared_ctx.mcast_src_cb_descriptor,
            shared_ctx.residual_mcast_src_cb_descriptor,
            shared_ctx.residual_mcast_dst_cb_descriptor,
            shared_ctx.down_mcast_dst_cb_descriptor,
            shared_ctx.down_matmul_in1_cb_descriptor,
            shared_ctx.down_matmul_out_cb_descriptor,
            shared_ctx.residual_add_out_cb_descriptor,
        ]

    @staticmethod
    def _build_core_descriptors(shared_ctx, sender_core_grid):
        """
        Build core descriptors for shared expert.

        Args:
            shared_ctx: _MoeSharedExpertContext
            sender_core_grid: CoreRangeSet for sender/gated_reduce core

        Returns:
            (unified_descs, per_core_descs) - lists to append to routed expert descriptors
        """
        from models.demos.deepseek_v3_b1.fused_ops.down_proj.op import DownProj

        unified = [
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_shared_compute_core",
                core_range=shared_ctx.compute_core_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_shared_gate_compute_core",
                core_range=shared_ctx.a_compute_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_shared_up_compute_core",
                core_range=shared_ctx.b_compute_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_shared_gated_reduce_core",
                core_range=sender_core_grid,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_shared_mcast_receiver_core",
                core_range=DownProj.build_matmul_core_grid(),
                value=1,
                other_value=0,
            ),
        ]
        per_core = [
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="shared_gu_k_offset",
                core_values=shared_ctx.gu_k_offset_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="shared_ag_sender_idx",
                core_values=shared_ctx.ag_sender_idx_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="shared_bg_sender_idx",
                core_values=shared_ctx.bg_sender_idx_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="shared_residual_add_core_idx",
                core_values=[
                    (core, idx) for idx, core in enumerate(ttnn.corerange_to_cores(DownProj.build_matmul_core_grid()))
                ],
                other_value=0,
            ),
        ]
        return unified, per_core


class MoeOp:
    """
    Top-level fused MoE operation.

    Composes MoeRoutedExpertOp and MoeSharedExpertOp, merging their
    CB descriptors, compile-time args, and core descriptors into a
    single unified kernel invocation.
    """

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
        expert_index_tensor,
        expert_scale_tensor,
        gate_proj_weights_tensor,
        gate_proj_output_tensor,
        up_proj_weights_tensor,
        up_proj_mm_out_tensor,
        fused_output_tensor,
        down_proj_gather_output_tensor,
        down_proj_mcast_output_tensor,
        down_proj_weights_tensor,
        down_proj_output_tensor,
        fused_add_tensor,
        final_output_tensor,
        gate_proj_in1_buf_tensor,
        up_proj_in1_buf_tensor,
        down_proj_in1_buf_tensor,
        mul_scalar_buf_tensor,
        # Shared expert tensors
        shared_gate_up_weights_tensor,
        shared_bias_tensor,
        shared_residual_mcast_dst_tensor,
        shared_down_mcast_dst_tensor,
        shared_down_weights_tensor,
        shared_k_parallel,
        shared_n_parallel,
        use_hardcoded_expert_index=False,
    ):
        """
        Execute the full fused MoE operation (routed + shared expert).

        Returns:
            (gate_output_scores_tensor, gate_output_indices_tensor, final_output_tensor)
        """
        # ==================================================================
        # Setup routed expert context
        # ==================================================================
        routed_ctx = MoeRoutedExpertOp._setup_dimensions(
            input_tensor,
            mcast_output_tensor,
            gate_mm_weights_tensor,
            gate_mm_output_tensor,
            gate_input_tensor,
            gate_bias_tensor,
            gate_indices_tensor,
            gate_output_scores_tensor,
            gate_output_indices_tensor,
            expert_index_tensor,
            expert_scale_tensor,
            gate_proj_weights_tensor,
            gate_proj_output_tensor,
            up_proj_weights_tensor,
            up_proj_mm_out_tensor,
            fused_output_tensor,
            down_proj_gather_output_tensor,
            down_proj_mcast_output_tensor,
            down_proj_weights_tensor,
            down_proj_output_tensor,
            fused_add_tensor,
            final_output_tensor,
            gate_proj_in1_buf_tensor,
            up_proj_in1_buf_tensor,
            down_proj_in1_buf_tensor,
            mul_scalar_buf_tensor,
            use_hardcoded_expert_index=use_hardcoded_expert_index,
        )

        # ==================================================================
        # Setup shared expert context
        # ==================================================================
        # Get input tile info from the input tensor
        device_tensor = ttnn.get_device_tensors(input_tensor)[0]
        input_tile = device_tensor.get_tile()
        input_tile_size = input_tile.get_tile_size(routed_ctx.data_format)

        shared_ctx = MoeSharedExpertOp._setup_dimensions(
            device=routed_ctx.device,
            shared_gate_up_weights_tensor=shared_gate_up_weights_tensor,
            shared_bias_tensor=shared_bias_tensor,
            shared_residual_mcast_dst_tensor=shared_residual_mcast_dst_tensor,
            shared_down_mcast_dst_tensor=shared_down_mcast_dst_tensor,
            shared_down_weights_tensor=shared_down_weights_tensor,
            num_tiles_k=routed_ctx.num_tiles_k,
            tile_1x32_size=routed_ctx.tile_1x32_size,
            data_format=routed_ctx.data_format,
            input_tile=input_tile,
            input_tile_size=input_tile_size,
            sender_core=routed_ctx.sender_core,
            sender_core_grid=routed_ctx.input_core_grid,
            mcast_grid=routed_ctx.mcast_grid,
            k_parallel=shared_k_parallel,
            n_parallel=shared_n_parallel,
        )

        # ==================================================================
        # Build CB descriptors (routed + shared)
        # ==================================================================
        cb_descriptors = MoeRoutedExpertOp._build_cb_descriptors(routed_ctx)
        cb_descriptors += MoeSharedExpertOp._build_cb_descriptors(shared_ctx)

        # ==================================================================
        # Build core descriptors (routed + shared)
        # ==================================================================
        unified_core_descs, per_core_descs = MoeRoutedExpertOp._build_core_descriptors(routed_ctx)
        shared_unified, shared_per_core = MoeSharedExpertOp._build_core_descriptors(
            shared_ctx, sender_core_grid=routed_ctx.input_core_grid
        )
        unified_core_descs += shared_unified
        per_core_descs += shared_per_core

        # ==================================================================
        # Semaphore descriptors
        # ==================================================================
        semaphore_descriptors = [
            ttnn.SemaphoreDescriptor(
                id=routed_ctx.mcast_data_sender_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=routed_ctx.mcast_data_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=routed_ctx.gather_noc0_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=routed_ctx.gather_noc1_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=routed_ctx.expert_scale_mcast_sender_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=1,  # Sender starts as VALID
            ),
            ttnn.SemaphoreDescriptor(
                id=routed_ctx.expert_scale_mcast_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
        ]
        # Shared expert gather semaphores
        semaphore_descriptors += [
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.ag_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.bg_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.ag_noc1_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.bg_noc1_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            # Shared expert residual mcast semaphores
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.residual_mcast_sender_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=1,  # Sender starts as VALID
            ),
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.residual_mcast_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            # Shared expert down mcast semaphores
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.down_mcast_sender_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=1,  # Sender starts as VALID
            ),
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.down_mcast_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
        ]

        # ==================================================================
        # IO tensors
        # ==================================================================
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
            expert_index_tensor,
            gate_proj_weights_tensor,
            gate_proj_output_tensor,
            up_proj_weights_tensor,
            up_proj_mm_out_tensor,
            fused_output_tensor,
            expert_scale_tensor,
            down_proj_gather_output_tensor,
            down_proj_mcast_output_tensor,
            down_proj_weights_tensor,
            down_proj_output_tensor,
            fused_add_tensor,
            final_output_tensor,
            # Tensor-backed working buffers
            gate_proj_in1_buf_tensor,
            up_proj_in1_buf_tensor,
            down_proj_in1_buf_tensor,
            mul_scalar_buf_tensor,
            # Shared expert tensors
            shared_gate_up_weights_tensor,
            shared_ctx.ag_dummy_tensor,
            shared_ctx.bg_dummy_tensor,
            shared_bias_tensor,
            shared_residual_mcast_dst_tensor,
            shared_down_mcast_dst_tensor,
            shared_down_weights_tensor,
        ]

        # ==================================================================
        # Create per-device programs (mesh loop)
        # ==================================================================
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        input_mcast_dst_cb = routed_ctx.input_mcast_params["dst_cb"]

        for row in range(routed_ctx.mesh_rows):
            for col in range(routed_ctx.mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                chip_id = row * routed_ctx.mesh_cols + col

                # Build compile-time args: routed + shared
                ncrisc_args, brisc_args, trisc_args = MoeRoutedExpertOp._build_compile_time_args(routed_ctx, chip_id)
                shared_ncrisc, shared_brisc, shared_trisc = MoeSharedExpertOp._build_compile_time_args(
                    shared_ctx, input_mcast_dst_cb, routed_ctx.input_mcast_params
                )
                ncrisc_args += shared_ncrisc
                brisc_args += shared_brisc
                trisc_args += shared_trisc

                # Create unified kernel
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/moe/moe_kernel.cpp",
                    core_ranges=routed_ctx.full_device_grid,
                    ncrisc_named_compile_time_args=ncrisc_args,
                    brisc_named_compile_time_args=brisc_args,
                    trisc_named_compile_time_args=trisc_args,
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                        math_approx_mode=False,
                        fp32_dest_acc_en=False,
                        dst_full_sync_en=False,
                    ),
                    unified_compile_time_core_descriptors=unified_core_descs,
                    per_core_compile_time_descriptors=per_core_descs,
                )

                # Create program
                program = ttnn.ProgramDescriptor(
                    kernels=unified_kernel.get_kernel_descriptors().kernels,
                    cbs=cb_descriptors,
                    semaphores=semaphore_descriptors,
                )

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute
        ttnn.generic_op(io_tensors, mesh_program_descriptor)

        return gate_output_scores_tensor, gate_output_indices_tensor, final_output_tensor
