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
from typing import Any, Optional

import ttnn
from models.demos.deepseek_v3_b1.fused_ops.face_view_utils import FACE_HEIGHT, FACE_WIDTH, can_use_face_view
from models.demos.deepseek_v3_b1.fused_ops.moe_routed_expert.op import (
    MESH_LEAF,
    MESH_ROOT1,
    MESH_ROOT2,
    MESH_ROOT3,
    get_reduce_device_role,
)
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.deepseek_v3_b1.utils import float_to_uint32


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
    rmsnorm_output_cb: int
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
    rmsnorm_mcast_params: dict
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
    rmsnorm_output_cb_descriptor: Any
    gate_mm_input_cb_descriptor: Any
    gate_proj_cb_index_descriptor: Any

    # Residual mcast (input → shared expert matmul cores)
    residual_mcast_src_cb: int
    residual_mcast_dst_cb: int
    residual_mcast_receiver_semaphore_id: int
    residual_mcast_src_cb_descriptor: Any
    residual_mcast_params: dict

    # RMSNorm (sender core: raw input → normalized input before input mcast)
    rmsnorm_gamma_cb: int
    rmsnorm_gamma_cb_descriptor: Any
    rmsnorm_epsilon_packed: int
    rmsnorm_scalar_packed: int
    rmsnorm_num_tiles: int
    rmsnorm_gamma_num_pages: int

    # Per-core values (for core descriptors)
    bank_id_core_values: list
    vc_core_values: list
    sender_idx_core_values: list

    # Testing flag
    use_hardcoded_expert_index: bool

    # ReduceToOne
    enable_reduce_to_one: bool = False
    reduce_local_cb: int = 0
    reduce_received_cb_r1: int = 0
    reduce_received_cb_r2: int = 0
    reduce_received_cb_r3: int = 0
    reduce_output_cb: int = 0
    reduce_scratch_cb: int = 0
    reduce_packet_cb: int = 0
    reduce_packet_header_cb: int = 0
    reduce_params: dict = None


@dataclass
class _MoeSharedExpertContext:
    """Holds shared expert values for the fused MoE kernel."""

    # Core grids
    compute_core_grid: Any
    a_compute_grid: Any
    b_compute_grid: Any
    a_cores_list: list
    b_cores_list: list
    matmul_core_grid: Any

    # CB indices (flat for compile-time arg readability)
    gu_weights_cb: int
    gu_out_cb: int
    group1_cb: int  # gate gather dst (on sender)
    group2_cb: int  # up gather dst (on sender)
    intermed_cb: int  # gated reduce intermediate (on sender)
    mcast_src_cb: int  # gated reduce output (on sender)
    residual_cb: int  # residual destination on mcast grid
    down_mcast_dst_cb: int

    # Parallelism
    n_parallel: int
    k_parallel: int
    num_compute_cores: int  # 64 per branch

    # Gather params (shared between A and B gathers)
    gather_dest_noc_core: Any
    gu_gather_data_size_bytes: int
    total_gather_tiles: int

    # Semaphore IDs
    ag_receiver_semaphore_id: int
    bg_receiver_semaphore_id: int
    ag_noc1_receiver_semaphore_id: int
    bg_noc1_receiver_semaphore_id: int
    shared_mcast_sender_semaphore_id: int
    shared_mcast_receiver_semaphore_id: int
    output_gather_noc0_receiver_semaphore_id: int
    output_gather_noc1_receiver_semaphore_id: int
    output_mcast_sender_semaphore_id: int
    output_mcast_receiver_semaphore_id: int

    # Keep-alive tensors and derived addresses
    ag_dummy_tensor: Any
    bg_dummy_tensor: Any
    ag_receiver_data_addr: int
    bg_receiver_data_addr: int
    down_mcast_dst_dummy_tensor: Any
    output_gather_dst_dummy_tensor: Any

    # Setup result dicts (grouped by operation)
    gu_matmul_params: dict  # from setup_kn_sliced_matmul
    gated_reduce_params: dict  # from setup_gated_reduce
    down_mcast_params: dict  # down mcast dimensions + CB descriptor
    down_matmul_params: dict  # down proj matmul dimensions + CB descriptors
    residual_add_params: dict  # residual add dimensions + CB descriptor
    output_gather_params: dict  # output gather dimensions + CB descriptor
    output_mcast_params: dict  # output mcast dimensions

    # Per-core values
    gu_k_offset_core_values: list
    ag_sender_idx_core_values: list
    bg_sender_idx_core_values: list


class MoeRoutedExpertOp:
    """
    MoE Routed Expert fused operation (refactored).

    Follows the SharedExpertOp pattern: context dataclass + decomposed build methods.
    """

    # Fused activation enum values (must match matmul.hpp FusedActivation enum)
    ACTIVATION_NONE = 0
    ACTIVATION_SIGMOID = 1
    ACTIVATION_SILU = 2

    # ------------------------------------------------------------------
    # Setup APIs (routed-expert-specific)
    # ------------------------------------------------------------------

    @staticmethod
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
        """
        weights_tile = weights_tensor.get_tile()
        weights_shard_shape = weights_tensor.memory_config().shard_spec.shape
        K = weights_shard_shape[0]
        per_core_n = weights_shard_shape[1] // weights_tile.tile_shape[1]
        Kt = K // weights_tile.tile_shape[0]

        subblock_k = Kt // num_subblocks_k
        assert Kt % num_subblocks_k == 0, f"Kt ({Kt}) must be divisible by num_subblocks ({num_subblocks_k})"

        weights_tile_size = weights_tile.get_tile_size(weights_tensor.dtype)
        in1_page_size, in1_num_pages = MoeOp.get_max_page_size_and_num_pages(device, subblock_k, weights_tile_size)
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

    @staticmethod
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

    @staticmethod
    def setup_eltwise_add(
        in0_tensor,
        in1_tensor,
        out_tensor,
        cb_in0_index,
        cb_in1_index,
        cb_out_index,
    ):
        """
        Set up parameters and CB descriptors for element-wise add with per-core indexing.

        Used after down_proj to add fused_add tensor. Each core uses sender_index to
        offset into the replicated in1 tensor.

        Args:
            in0_tensor: First input tensor (e.g., down_proj output), WIDTH_SHARDED
            in1_tensor: Second input tensor (e.g., fused_add), HEIGHT_SHARDED (replicated)
            out_tensor: Output tensor, WIDTH_SHARDED (padded to 32x32 tile)
            cb_in0_index: CB index for first input (aliased with 32x32 tile format)
            cb_in1_index: CB index for second input (replicated tensor)
            cb_out_index: CB index for output

        Returns:
            Dictionary with eltwise_add parameters and CB descriptors
        """
        # Get core ranges from in0_tensor (same as previous mm)
        core_ranges = in0_tensor.memory_config().shard_spec.grid
        compute_cores_list = ttnn.corerange_to_cores(core_ranges, row_wise=True)
        # Get tensor info
        in0_dtype = in0_tensor.dtype
        in0_shard_shape = in0_tensor.memory_config().shard_spec.shape
        in1_shard_shape = in1_tensor.memory_config().shard_spec.shape

        # Dimensions
        width_per_core = in0_shard_shape[1]  # per-core width (e.g., 896)
        total_width = in1_shard_shape[1]  # full width of replicated tensor (e.g., 7168)

        # Element size
        if in0_dtype == ttnn.bfloat16:
            element_size_bytes = 2
        else:
            raise ValueError(f"Unsupported dtype: {in0_dtype}")

        slice_size_bytes = width_per_core * element_size_bytes

        # Use 32x32 tile view for CB (not tensor's actual tile format)
        # This is CB aliasing - tensor uses 1x32 tiles but CB views as 32x32
        cb_tile_h = 32
        cb_tile_w = 32
        cb_tile_size_bytes = cb_tile_h * cb_tile_w * element_size_bytes
        cb_tile = ttnn.Tile([cb_tile_h, cb_tile_w])
        cb_tile_desc = ttnn.TileDescriptor(cb_tile)

        # CB sizes
        in0_size_bytes = slice_size_bytes
        in1_size_bytes = total_width * element_size_bytes

        # Number of pages for CB wait (total_size / page_size)
        # cb_in0: 1 page (in0_size_bytes / in0_size_bytes = 1)
        # cb_in1: multiple pages (in1_size_bytes / in0_size_bytes)
        in0_wait_tiles = in0_size_bytes // in0_size_bytes  # 1
        in1_wait_tiles = in1_size_bytes // in0_size_bytes

        # Number of output tiles (based on 32x32 CB view, not tensor tile format)
        out_shard_shape = out_tensor.memory_config().shard_spec.shape
        out_shard_elements = out_shard_shape[0] * out_shard_shape[1]
        cb_tile_elements = cb_tile_h * cb_tile_w
        num_tiles = out_shard_elements // cb_tile_elements
        assert num_tiles == 1, f"Expected 1 tile (32x32 view), got {num_tiles}"

        # CB for in0: down_proj output aliased with 32x32 tile format
        cb_in0_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_in0_index, in0_tensor)
        cb_in0_descriptor.total_size = in0_size_bytes
        cb_in0_descriptor.format_descriptors[0].tile = cb_tile_desc
        cb_in0_descriptor.format_descriptors[0].page_size = in0_size_bytes

        # CB for in1: replicated tensor (tensor-backed)
        cb_in1_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_in1_index, in1_tensor)
        cb_in1_descriptor.total_size = in1_size_bytes
        cb_in1_descriptor.format_descriptors[0].tile = cb_tile_desc
        cb_in1_descriptor.format_descriptors[0].page_size = in0_size_bytes  # page_size = slice size for reading

        # CB for out: output (tensor-backed, uses 32x32 CB view)
        cb_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_out_index, out_tensor)
        cb_out_descriptor.total_size = num_tiles * cb_tile_size_bytes
        cb_out_descriptor.format_descriptors[0].tile = cb_tile_desc
        cb_out_descriptor.format_descriptors[0].page_size = cb_tile_size_bytes

        # Per-core sender_index values
        sender_index_core_values = []
        for idx, core in enumerate(compute_cores_list):
            sender_index_core_values.append((core, idx))

        return {
            # CB indices
            "cb_in0": cb_in0_index,
            "cb_in1": cb_in1_index,
            "cb_out": cb_out_index,
            # Dimensions
            "num_tiles": num_tiles,
            "slice_size_bytes": slice_size_bytes,
            # Wait tiles
            "cb_in0_wait_tiles": in0_wait_tiles,
            "cb_in1_wait_tiles": in1_wait_tiles,
            # CB descriptors
            "cb_in0_descriptor": cb_in0_descriptor,
            "cb_in1_descriptor": cb_in1_descriptor,
            "cb_out_descriptor": cb_out_descriptor,
            # Per-core values
            "sender_index_core_values": sender_index_core_values,
        }

    @staticmethod
    def setup_gate(
        input_cb,
        bias_cb,
        indices_cb,
        output_cb,
        output_indices_cb,
        input_tensor,
        bias_tensor,
        indices_tensor,
        output_scores_tensor,
        output_indices_tensor,
        eps=1e-20,
        scaling_factor=2.5,
        enable_sigmoid=False,
    ):
        """
        Set up parameters for the MoE gate operation.

        The gate computes top-K expert selection with normalized scores.

        Args:
            input_cb: Input CB index (receives gathered matmul output)
            bias_cb: Bias CB index
            indices_cb: Indices CB index
            output_cb: Output scores CB index
            output_indices_cb: Output indices CB index
            input_tensor: Input tensor (gathered matmul output)
            bias_tensor: Bias tensor
            indices_tensor: Indices tensor
            output_scores_tensor: Output scores tensor
            output_indices_tensor: Output indices tensor
            eps: Epsilon for numerical stability (default 1e-20)
            scaling_factor: Scaling factor for gate scores (default 2.5)
            enable_sigmoid: Whether to apply sigmoid (default False, already done in matmul)

        Returns:
            Dictionary with gate parameters and CB descriptors
        """
        import struct

        # Convert float parameters to uint32 bit patterns
        eps_uint32 = int.from_bytes(struct.pack("f", eps), byteorder="little")
        scaling_factor_uint32 = int.from_bytes(struct.pack("f", scaling_factor), byteorder="little")

        # CB descriptors
        input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)
        bias_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(bias_cb, bias_tensor)
        indices_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(indices_cb, indices_tensor)
        output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_scores_tensor)
        output_indices_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_indices_cb, output_indices_tensor)

        return {
            # CB indices
            "input_cb": input_cb,
            "bias_cb": bias_cb,
            "indices_cb": indices_cb,
            "output_cb": output_cb,
            "output_indices_cb": output_indices_cb,
            # Parameters (as uint32 bit patterns)
            "eps": eps_uint32,
            "scaling_factor": scaling_factor_uint32,
            "enable_sigmoid": 1 if enable_sigmoid else 0,
            # CB descriptors
            "input_cb_descriptor": input_cb_descriptor,
            "bias_cb_descriptor": bias_cb_descriptor,
            "indices_cb_descriptor": indices_cb_descriptor,
            "output_cb_descriptor": output_cb_descriptor,
            "output_indices_cb_descriptor": output_indices_cb_descriptor,
        }

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
        down_proj_in1_buf_tensor,
        mul_scalar_buf_tensor,
        shared_residual_mcast_src_tensor,
        shared_residual_mcast_dst_tensor,
        rmsnorm_gamma_tensor,
        epsilon=1e-6,
        use_hardcoded_expert_index=False,
        reduce_intermediate_tensors=None,
        reduce_output_tensor=None,
        reduce_semaphores=None,
        reduce_root_coord=None,
        # Semaphore IDs (caller-provided, see MoeOp for top-level definitions)
        mcast_data_sender_semaphore_id=0,
        mcast_data_receiver_semaphore_id=1,
        gather_noc0_receiver_semaphore_id=1,
        gather_noc1_receiver_semaphore_id=1,
        expert_scale_mcast_receiver_semaphore_id=5,
        residual_mcast_receiver_semaphore_id=2,
        index_mcast_receiver_semaphore_id=6,
        down_proj_mcast_receiver_semaphore_id=7,
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

        expert_scale_mcast_sender_semaphore_id = mcast_data_sender_semaphore_id  # Reuse sender semaphore

        # ==================================================================
        # CB indices
        # ==================================================================
        rmsnorm_output_cb = 0
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
        up_proj_cb_mm_out = 12
        mul_cb_in0 = 13
        mul_cb_in1 = 14
        mul_cb_out = 15
        down_proj_gather_dst_cb = 16
        down_proj_mcast_dst_cb = 17
        down_proj_cb_in1 = 18
        down_proj_cb_out = 19
        mul_cb_scalar_src = 20
        mul_cb_scalar = 21
        add_cb_in0 = 22
        add_cb_in1 = 23
        add_cb_out = 24
        # ReduceToOne CBs (for multi-device reduce)
        # Must be after shared expert CBs (25-38) to avoid conflicts
        reduce_local_cb = add_cb_out  # Local data CB (same as add_cb_out for fusion)
        reduce_received_cb_r1 = 39  # Round 1: receives LEAF data
        reduce_received_cb_r2 = 40  # Round 2: receives ROOT3 data
        reduce_received_cb_r3 = 41  # Round 3: receives ROOT2 data
        reduce_output_cb = 42  # Final reduced output
        reduce_scratch_cb = 43  # Scratch for compute
        reduce_packet_cb = 44  # Scratch for sending packets
        reduce_packet_header_cb = 45  # Packet header (persistent)
        # Shared expert CBs (defined in MoeSharedExpertOp)
        residual_mcast_src_cb = MoeSharedExpertOp.RESIDUAL_MCAST_SRC_CB
        residual_mcast_dst_cb = MoeSharedExpertOp.RESIDUAL_MCAST_DST_CB
        rmsnorm_gamma_cb = MoeSharedExpertOp.RMSNORM_GAMMA_CB

        # ==================================================================
        # RMSNorm tile reinterpretation (compute kernel needs 32x32 or 16x32 tiles)
        # ==================================================================
        # Reinterpret N 1x32 tiles as full 32x32 or half 16x32 tiles for compute kernel
        # (matching standalone RMSNorm op.py behavior — mcast just sends raw bytes)
        FULL_32x32_TILE = ttnn.Tile((32, 32))
        HALF_16x32_TILE = ttnn.Tile((16, 32))
        is_16x32_tile = (K // FULL_32x32_TILE.tile_shape[1]) % FULL_32x32_TILE.tile_shape[0] != 0
        rmsnorm_interpreted_tile = HALF_16x32_TILE if is_16x32_tile else FULL_32x32_TILE
        rmsnorm_tile_descriptor = ttnn.TileDescriptor(rmsnorm_interpreted_tile)
        rmsnorm_cb_page_size = rmsnorm_interpreted_tile.get_tile_size(data_format)
        rmsnorm_num_tiles = K // (rmsnorm_interpreted_tile.tile_shape[0] * rmsnorm_interpreted_tile.tile_shape[1])

        # ==================================================================
        # Residual Mcast (raw input from sender → residual CB on mcast grid)
        # ==================================================================
        residual_mcast_src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            residual_mcast_src_cb, shared_residual_mcast_src_tensor
        )
        # Override tile to 32x32 for RMSNorm compute (mcast just sends raw bytes, doesn't care)
        residual_mcast_src_cb_descriptor.format_descriptors[0].tile = rmsnorm_tile_descriptor
        residual_mcast_src_cb_descriptor.format_descriptors[0].page_size = rmsnorm_cb_page_size

        residual_mcast_data_size_bytes = num_tiles_k * tile_1x32_size
        residual_mcast_params = MoeOp.setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=mcast_grid,
            src_cb=residual_mcast_src_cb,
            src_tensor=shared_residual_mcast_src_tensor,
            dst_cb=residual_mcast_dst_cb,
            dst_tensor=shared_residual_mcast_dst_tensor,
            sender_semaphore_id=mcast_data_sender_semaphore_id,
            receiver_semaphore_id=residual_mcast_receiver_semaphore_id,
            data_size_bytes=residual_mcast_data_size_bytes,
        )
        # Override src_num_pages: CB 32 descriptor is now 32x32 tiles, so setup_sharded_buffer
        # and mcast sender need reinterpreted page count. dst_num_pages stays 224 (CB 33 is 1x32).
        residual_mcast_params["src_num_pages"] = rmsnorm_num_tiles

        # ==================================================================
        # RMSNorm (sender core: residual_mcast_src → rmsnorm_output)
        # ==================================================================
        rmsnorm_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(rmsnorm_output_cb, input_tensor)
        rmsnorm_output_cb_descriptor.format_descriptors[0].tile = rmsnorm_tile_descriptor
        rmsnorm_output_cb_descriptor.format_descriptors[0].page_size = rmsnorm_cb_page_size

        rmsnorm_gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(rmsnorm_gamma_cb, rmsnorm_gamma_tensor)
        rmsnorm_gamma_cb_descriptor.format_descriptors[0].tile = rmsnorm_tile_descriptor
        rmsnorm_gamma_cb_descriptor.format_descriptors[0].page_size = rmsnorm_cb_page_size

        rmsnorm_epsilon_packed = float_to_uint32(epsilon)
        rmsnorm_scalar_packed = float_to_uint32(1.0 / math.sqrt(float(K)))

        # setup_sharded_buffer num_pages for gamma: use reinterpreted 32x32 tile count
        rmsnorm_gamma_num_pages = rmsnorm_num_tiles

        # ==================================================================
        # RMSNorm Mcast (broadcasts normalized input from rmsnorm_output_cb to all cores)
        # ==================================================================
        rmsnorm_mcast_data_size_bytes = num_tiles_k * tile_1x32_size
        rmsnorm_mcast_params = MoeOp.setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=mcast_grid,
            src_cb=rmsnorm_output_cb,
            src_tensor=input_tensor,
            dst_cb=gate_mm_input_cb,
            dst_tensor=mcast_output_tensor,
            sender_semaphore_id=mcast_data_sender_semaphore_id,
            receiver_semaphore_id=mcast_data_receiver_semaphore_id,
            data_size_bytes=rmsnorm_mcast_data_size_bytes,
        )
        # Override src_num_pages: RMSNorm pushes rmsnorm_num_tiles (7) 32x32-tile pages,
        # but receiver (CB 1) still needs num_tiles_k (224) 1x32-tile pages for gate MM
        rmsnorm_mcast_params["src_num_pages"] = rmsnorm_num_tiles

        # ==================================================================
        # Gate MM (SRAM Matmul)
        # ==================================================================
        gate_mm_params = MoeOp.setup_sram_matmul(
            in0_cb=gate_mm_input_cb,
            in1_cb=gate_mm_weights_cb,
            out_cb=gate_mm_output_cb,
            weights_tensor=gate_mm_weights_tensor,
            output_tensor=gate_mm_output_tensor,
            k_num_tiles=num_tiles_k,
            fused_activation=MoeRoutedExpertOp.ACTIVATION_SIGMOID,
        )

        # Pre-built CB descriptors (tensor-backed, not from setup helpers)
        gate_mm_input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gate_mm_input_cb, mcast_output_tensor)

        # ==================================================================
        # Gate MM Gather
        # ==================================================================
        gate_mm_output_tile = gate_mm_output_tensor.get_tile()
        gate_mm_output_tile_size = gate_mm_output_tile.get_tile_size(data_format)
        gate_mm_gather_data_size_bytes = gate_mm_params["out_w"] * gate_mm_output_tile_size

        gate_mm_gather_params = MoeOp.setup_gather(
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
        gate_params = MoeRoutedExpertOp.setup_gate(
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
        index_mcast_sender_semaphore_id = mcast_data_sender_semaphore_id  # Reuse sender semaphore
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
        gate_proj_params = MoeRoutedExpertOp.setup_dram_matmul(
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
        up_proj_params = MoeRoutedExpertOp.setup_dram_matmul(
            device=device,
            weights_tensor=up_proj_weights_tensor,
            output_tensor=up_proj_mm_out_tensor,
            working_buf_tensor=gate_proj_in1_buf_tensor,  # Shares CB with gate_proj
            core_ranges=gate_proj_core_ranges,
            cb_in1_index=up_proj_cb_in1,
            cb_out_index=up_proj_cb_mm_out,
            fp32_dest_acc_en=True,
            num_subblocks_k=4,
        )

        # ==================================================================
        # Eltwise Mul: silu(gate_proj) * up_proj * expert_scale
        # ==================================================================
        mul_params = MoeRoutedExpertOp.setup_eltwise_mul(
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
        down_proj_gather_params = MoeOp.setup_gather(
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
        down_proj_mcast_params = MoeOp.setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=mcast_grid,
            src_cb=down_proj_gather_dst_cb,
            src_tensor=down_proj_gather_output_tensor,
            dst_cb=down_proj_mcast_dst_cb,
            dst_tensor=down_proj_mcast_output_tensor,
            sender_semaphore_id=mcast_data_sender_semaphore_id,
            receiver_semaphore_id=down_proj_mcast_receiver_semaphore_id,
            data_size_bytes=down_proj_mcast_data_size_bytes,
        )

        # ==================================================================
        # DRAM Streaming Matmul: down_proj
        # ==================================================================
        down_proj_params = MoeRoutedExpertOp.setup_dram_matmul(
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
        add_params = MoeRoutedExpertOp.setup_eltwise_add(
            in0_tensor=down_proj_output_tensor,
            in1_tensor=fused_add_tensor,
            out_tensor=final_output_tensor,
            cb_in0_index=add_cb_in0,
            cb_in1_index=add_cb_in1,
            cb_out_index=add_cb_out,
        )

        # ==================================================================
        # ReduceToOne Setup
        # ==================================================================
        enable_reduce_to_one = (
            reduce_intermediate_tensors is not None
            and reduce_output_tensor is not None
            and reduce_semaphores is not None
            and reduce_root_coord is not None
            and mesh_rows == 4
            and mesh_cols == 2
        )

        reduce_params = {}
        if enable_reduce_to_one:
            # Get semaphore addresses
            reduce_sem_round1_addr = ttnn.get_global_semaphore_address(reduce_semaphores[0])
            reduce_sem_round2_addr = ttnn.get_global_semaphore_address(reduce_semaphores[1])
            reduce_sem_round3_addr = ttnn.get_global_semaphore_address(reduce_semaphores[2])
            reduce_sem_exit_addr = ttnn.get_global_semaphore_address(reduce_semaphores[3])

            # Get per-device tensors for reduce
            reduce_intermediate_r1_per_device = ttnn.get_device_tensors(reduce_intermediate_tensors[0])
            reduce_intermediate_r2_per_device = ttnn.get_device_tensors(reduce_intermediate_tensors[1])
            reduce_intermediate_r3_per_device = ttnn.get_device_tensors(reduce_intermediate_tensors[2])
            reduce_output_per_device = ttnn.get_device_tensors(reduce_output_tensor)

            # Calculate reduce tensor properties
            reduce_sample = reduce_intermediate_r1_per_device[0]
            reduce_element_size = 2  # bfloat16

            reduce_shard_spec = reduce_sample.memory_config().shard_spec
            reduce_shard_shape = reduce_shard_spec.shape

            # Compute tiles use 32x32 format
            reduce_compute_tile_h = 32
            reduce_compute_tile_w = 32
            reduce_compute_tile_size = reduce_compute_tile_h * reduce_compute_tile_w * reduce_element_size
            reduce_shard_elements = reduce_shard_shape[0] * reduce_shard_shape[1]
            reduce_num_tiles = (reduce_shard_elements + (reduce_compute_tile_h * reduce_compute_tile_w) - 1) // (
                reduce_compute_tile_h * reduce_compute_tile_w
            )

            reduce_payload_size_bytes = reduce_shard_elements * reduce_element_size
            reduce_packet_header_size = 96
            reduce_slot_size_bytes = reduce_packet_header_size + reduce_payload_size_bytes

            # Worker cores from final_output_tensor shard grid (same as gate_proj cores)
            reduce_worker_grid = final_output_tensor.memory_config().shard_spec.grid
            reduce_worker_cores_list = ttnn.corerange_to_cores(reduce_worker_grid, row_wise=True)
            reduce_num_workers = len(reduce_worker_cores_list)

            # Build core -> column mapping for fabric core assignment
            reduce_column_to_cores = {}
            for core in reduce_worker_cores_list:
                x = core.x
                if x not in reduce_column_to_cores:
                    reduce_column_to_cores[x] = []
                reduce_column_to_cores[x].append(core)

            reduce_sorted_columns = sorted(reduce_column_to_cores.keys())
            for x in reduce_sorted_columns:
                reduce_column_to_cores[x].sort(key=lambda c: c.y)

            reduce_num_columns = len(reduce_sorted_columns)
            reduce_num_workers_per_column = len(reduce_column_to_cores[reduce_sorted_columns[0]])

            # Fabric cores: one per column, placed to the right of bottom core
            reduce_fabric_cores = []
            reduce_column_to_fabric_core = {}
            for x in reduce_sorted_columns:
                bottom_core = max(reduce_column_to_cores[x], key=lambda c: c.y)
                fabric_core = ttnn.CoreCoord(bottom_core.x + 1, bottom_core.y)
                reduce_fabric_cores.append(fabric_core)
                reduce_column_to_fabric_core[x] = fabric_core

            # Core to slot index within column
            reduce_core_to_slot_idx = {}
            for x in reduce_sorted_columns:
                for slot_idx, core in enumerate(reduce_column_to_cores[x]):
                    reduce_core_to_slot_idx[(core.x, core.y)] = slot_idx

            # Core to shard index
            reduce_core_to_shard_idx = {}
            for shard_idx, core in enumerate(reduce_worker_cores_list):
                reduce_core_to_shard_idx[(core.x, core.y)] = shard_idx

            # Get output core from reduce_output_tensor
            reduce_output_sample = reduce_output_per_device[0]
            reduce_output_shard_spec = reduce_output_sample.memory_config().shard_spec
            reduce_output_core = reduce_output_shard_spec.grid.ranges()[0].start

            # Get per-device final_output_tensor for reduce_local_cb
            final_output_per_device = ttnn.get_device_tensors(final_output_tensor)

            reduce_params = {
                "sem_round1_addr": reduce_sem_round1_addr,
                "sem_round2_addr": reduce_sem_round2_addr,
                "sem_round3_addr": reduce_sem_round3_addr,
                "sem_exit_addr": reduce_sem_exit_addr,
                "intermediate_r1_per_device": reduce_intermediate_r1_per_device,
                "intermediate_r2_per_device": reduce_intermediate_r2_per_device,
                "intermediate_r3_per_device": reduce_intermediate_r3_per_device,
                "output_per_device": reduce_output_per_device,
                "final_output_per_device": final_output_per_device,
                "num_tiles": reduce_num_tiles,
                "payload_size_bytes": reduce_payload_size_bytes,
                "slot_size_bytes": reduce_slot_size_bytes,
                "compute_tile_size": reduce_compute_tile_size,
                "worker_cores_list": reduce_worker_cores_list,
                "num_workers": reduce_num_workers,
                "num_workers_per_column": reduce_num_workers_per_column,
                "num_columns": reduce_num_columns,
                "fabric_cores": reduce_fabric_cores,
                "column_to_fabric_core": reduce_column_to_fabric_core,
                "core_to_slot_idx": reduce_core_to_slot_idx,
                "core_to_shard_idx": reduce_core_to_shard_idx,
                "output_core": reduce_output_core,
            }

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
            rmsnorm_output_cb=rmsnorm_output_cb,
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
            rmsnorm_mcast_params=rmsnorm_mcast_params,
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
            rmsnorm_output_cb_descriptor=rmsnorm_output_cb_descriptor,
            gate_mm_input_cb_descriptor=gate_mm_input_cb_descriptor,
            gate_proj_cb_index_descriptor=gate_proj_cb_index_descriptor,
            # Residual mcast
            residual_mcast_src_cb=residual_mcast_src_cb,
            residual_mcast_dst_cb=residual_mcast_dst_cb,
            residual_mcast_receiver_semaphore_id=residual_mcast_receiver_semaphore_id,
            residual_mcast_src_cb_descriptor=residual_mcast_src_cb_descriptor,
            residual_mcast_params=residual_mcast_params,
            # RMSNorm
            rmsnorm_gamma_cb=rmsnorm_gamma_cb,
            rmsnorm_gamma_cb_descriptor=rmsnorm_gamma_cb_descriptor,
            rmsnorm_epsilon_packed=rmsnorm_epsilon_packed,
            rmsnorm_scalar_packed=rmsnorm_scalar_packed,
            rmsnorm_num_tiles=rmsnorm_num_tiles,
            rmsnorm_gamma_num_pages=rmsnorm_gamma_num_pages,
            # Per-core values
            bank_id_core_values=bank_id_core_values,
            vc_core_values=vc_core_values,
            sender_idx_core_values=sender_idx_core_values,
            # Testing flag
            use_hardcoded_expert_index=use_hardcoded_expert_index,
            # ReduceToOne
            enable_reduce_to_one=enable_reduce_to_one,
            reduce_local_cb=reduce_local_cb,
            reduce_received_cb_r1=reduce_received_cb_r1,
            reduce_received_cb_r2=reduce_received_cb_r2,
            reduce_received_cb_r3=reduce_received_cb_r3,
            reduce_output_cb=reduce_output_cb,
            reduce_scratch_cb=reduce_scratch_cb,
            reduce_packet_cb=reduce_packet_cb,
            reduce_packet_header_cb=reduce_packet_header_cb,
            reduce_params=reduce_params if enable_reduce_to_one else None,
        )

    @staticmethod
    def _build_compile_time_args(ctx, mesh_chip_id):
        """Build NCRISC, BRISC, and TRISC compile-time arg lists for routed expert."""

        ncrisc_named_compile_time_args = [
            # Input mcast (sender sharded buffer + receiver)
            ("mcast_src_cb", ctx.rmsnorm_mcast_params["src_cb"]),
            ("mcast_src_num_pages", ctx.rmsnorm_mcast_params["src_num_pages"]),
            ("mcast_data_receiver_semaphore", ctx.rmsnorm_mcast_params["receiver_semaphore_id"]),
            ("mcast_dst_cb", ctx.rmsnorm_mcast_params["dst_cb"]),
            ("mcast_dst_num_pages", ctx.rmsnorm_mcast_params["dst_num_pages"]),
            # Residual mcast source (setup_sharded_buffer on sender core)
            ("shared_residual_mcast_src_cb", ctx.residual_mcast_params["src_cb"]),
            ("shared_residual_mcast_src_num_pages", ctx.residual_mcast_params["src_num_pages"]),
            # Residual mcast receiver (input from sender → residual CB on mcast grid)
            ("shared_residual_mcast_data_receiver_semaphore", ctx.residual_mcast_receiver_semaphore_id),
            ("shared_residual_cb", ctx.residual_mcast_dst_cb),
            ("shared_residual_num_pages", ctx.residual_mcast_params["dst_num_pages"]),
            # RMSNorm (setup_sharded_buffer for gamma on sender core)
            ("rmsnorm_gamma_cb", ctx.rmsnorm_gamma_cb),
            ("rmsnorm_gamma_num_pages", ctx.rmsnorm_gamma_num_pages),
            # Gate matmul reader
            ("gate_mm_in0", ctx.gate_mm_params["in0_cb"]),
            ("gate_mm_in1", ctx.gate_mm_params["in1_cb"]),
            ("gate_mm_k_num_tiles", ctx.gate_mm_params["k_num_tiles"]),
            ("gate_mm_out_w", ctx.gate_mm_params["out_w"]),
            # Gate gather receiver (MoeGather: receiver on NCRISC)
            ("gather_noc0_num_senders", ctx.gate_mm_gather_params["noc0_num_senders"]),
            ("gather_noc1_num_senders", ctx.gate_mm_gather_params["noc1_num_senders"]),
            ("gather_noc0_receiver_semaphore_id", ctx.gate_mm_gather_params["noc0_receiver_semaphore_id"]),
            ("gather_noc1_receiver_semaphore_id", ctx.gate_mm_gather_params["noc1_receiver_semaphore_id"]),
            ("gather_dst_cb", ctx.gate_mm_gather_params["dst_cb"]),
            ("gather_dst_num_pages", ctx.gate_mm_gather_params["dst_num_pages"]),
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
            # down_proj gather receiver (MoeGather: receiver on NCRISC)
            ("down_proj_gather_noc0_num_senders", ctx.down_proj_gather_params["noc0_num_senders"]),
            ("down_proj_gather_noc1_num_senders", ctx.down_proj_gather_params["noc1_num_senders"]),
            ("down_proj_gather_noc0_receiver_semaphore_id", ctx.down_proj_gather_params["noc0_receiver_semaphore_id"]),
            ("down_proj_gather_noc1_receiver_semaphore_id", ctx.down_proj_gather_params["noc1_receiver_semaphore_id"]),
            ("down_proj_gather_dst_cb", ctx.down_proj_gather_params["dst_cb"]),
            ("down_proj_gather_dst_num_pages", ctx.down_proj_gather_params["dst_num_pages"]),
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
            # ReduceToOne reader args (CB indices + common RT arg base)
            ("reduce_local_cb", ctx.reduce_local_cb),
            ("reduce_received_cb_r1", ctx.reduce_received_cb_r1),
            ("reduce_received_cb_r2", ctx.reduce_received_cb_r2),
            ("reduce_received_cb_r3", ctx.reduce_received_cb_r3),
            ("reduce_ncrisc_common_rt_arg_base", 0),
        ]

        brisc_named_compile_time_args = [
            # Input mcast sender
            ("mcast_dest_noc_start_x", ctx.rmsnorm_mcast_params["dest_noc_start_x"]),
            ("mcast_dest_noc_start_y", ctx.rmsnorm_mcast_params["dest_noc_start_y"]),
            ("mcast_dest_noc_end_x", ctx.rmsnorm_mcast_params["dest_noc_end_x"]),
            ("mcast_dest_noc_end_y", ctx.rmsnorm_mcast_params["dest_noc_end_y"]),
            ("mcast_num_cores", ctx.rmsnorm_mcast_params["num_cores"]),
            ("mcast_data_sender_semaphore", ctx.rmsnorm_mcast_params["sender_semaphore_id"]),
            ("mcast_data_receiver_semaphore", ctx.rmsnorm_mcast_params["receiver_semaphore_id"]),
            ("mcast_data_size_bytes", ctx.rmsnorm_mcast_params["data_size_bytes"]),
            ("mcast_src_cb", ctx.rmsnorm_mcast_params["src_cb"]),
            ("mcast_dst_cb", ctx.rmsnorm_mcast_params["dst_cb"]),
            ("mcast_src_num_pages", ctx.rmsnorm_mcast_params["src_num_pages"]),
            ("mcast_is_part_of_receiver_grid", ctx.rmsnorm_mcast_params["is_sender_part_of_receiver_grid"]),
            # Residual mcast sender (input from sender → residual CB on mcast grid)
            ("shared_residual_mcast_data_sender_semaphore", ctx.mcast_data_sender_semaphore_id),
            ("shared_residual_mcast_data_receiver_semaphore", ctx.residual_mcast_receiver_semaphore_id),
            ("shared_residual_mcast_data_size_bytes", ctx.residual_mcast_params["data_size_bytes"]),
            ("shared_residual_mcast_src_cb", ctx.residual_mcast_params["src_cb"]),
            ("shared_residual_mcast_src_num_pages", ctx.residual_mcast_params["src_num_pages"]),
            ("shared_residual_mcast_dst_cb", ctx.residual_mcast_dst_cb),
            # Gate gather sender (MoeGather: sender on BRISC)
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
            # down_proj gather sender (MoeGather: sender on BRISC)
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
            # Eltwise add CB (needed by output mcast sender for get_write_ptr)
            ("add_cb_in1", ctx.add_cb_in1),
            # ReduceToOne writer args (CB indices + RT arg bases)
            ("reduce_local_cb", ctx.reduce_local_cb),
            ("reduce_scratch_cb", ctx.reduce_scratch_cb),
            ("reduce_brisc_rt_arg_base", 0),
            ("reduce_brisc_fabric_rt_arg_base", 0),
        ]

        trisc_named_compile_time_args = [
            # RMSNorm compute (sender core only)
            ("rmsnorm_input_cb", ctx.residual_mcast_src_cb),
            ("rmsnorm_gamma_cb", ctx.rmsnorm_gamma_cb),
            ("rmsnorm_output_cb", ctx.rmsnorm_output_cb),
            ("rmsnorm_fp32_acc", 0),
            ("rmsnorm_num_tiles", ctx.rmsnorm_num_tiles),
            ("rmsnorm_rsqrt_fast_approx", 0),
            ("rmsnorm_trisc_common_rt_arg_base", 0),
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
            # ReduceToOne compute args (CB indices)
            ("reduce_local_cb", ctx.reduce_local_cb),
            ("reduce_received_cb_r1", ctx.reduce_received_cb_r1),
            ("reduce_received_cb_r2", ctx.reduce_received_cb_r2),
            ("reduce_received_cb_r3", ctx.reduce_received_cb_r3),
            ("reduce_output_cb", ctx.reduce_output_cb),
            ("reduce_scratch_cb", ctx.reduce_scratch_cb),
        ]

        return ncrisc_named_compile_time_args, brisc_named_compile_time_args, trisc_named_compile_time_args

    @staticmethod
    def _build_cb_descriptors(ctx):
        """Build circular buffer descriptors for routed expert."""
        return [
            ctx.rmsnorm_output_cb_descriptor,
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
            ctx.residual_mcast_src_cb_descriptor,
            ctx.residual_mcast_params["dst_cb_descriptor"],
            ctx.rmsnorm_gamma_cb_descriptor,
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
            # ReduceToOne core descriptors — will be updated per-device if enabled
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_reduce_worker_core",
                core_range=ctx.gate_proj_core_ranges if ctx.enable_reduce_to_one else ttnn.CoreRangeSet([]),
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_reduce_fabric_core",
                core_range=ttnn.CoreRangeSet([]),  # Set per-device
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

    # CB indices for shared expert CBs referenced by routed expert setup
    RESIDUAL_MCAST_SRC_CB = 25  # raw input on sender (residual mcast source)
    RESIDUAL_MCAST_DST_CB = 26  # residual destination on mcast grid
    RMSNORM_GAMMA_CB = 27  # RMSNorm gamma weights on sender

    # ========================================================================
    # Setup APIs
    # ========================================================================

    @staticmethod
    def setup_kn_sliced_matmul(
        weights_tensor,
        output_tensor,
        cb_weights_index,
        cb_out_index,
        num_tiles_k,
        k_parallel,
        n_parallel,
    ):
        """
        Setup KN-sliced matmul: [1, K] x [K, N] distributed across K*N cores.

        Each core holds a shard of shape [K/k_parallel, N/n_parallel] of the weight matrix.
        Each core computes: activation[k_offset:k_offset+k_per_core] @ weights_shard → 1 output tile.

        Args:
            weights_tensor: Gate/Up weights tensor (HEIGHT_SHARDED on K*N compute cores)
            output_tensor: Matmul output tensor (HEIGHT_SHARDED on K*N compute cores, 1 tile per core)
            cb_weights_index: CB index for weights
            cb_out_index: CB index for output
            num_tiles_k: Total K dimension in tiles
            k_parallel: K parallelism factor (number of K slices)
            n_parallel: N parallelism factor (number of N slices)

        Returns:
            dict with:
            - k_per_core: K tiles per core
            - act_total_tiles: Total activation tiles (= num_tiles_k)
            - weights_num_pages: Pages in weights CB per core (= k_per_core)
            - cb_weights_descriptor: CB descriptor for weights (tensor-backed)
            - cb_out_descriptor: CB descriptor for output (tensor-backed)
        """
        k_per_core = num_tiles_k // k_parallel
        act_total_tiles = num_tiles_k
        weights_num_pages = k_per_core

        cb_weights_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_weights_index, weights_tensor)
        cb_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_out_index, output_tensor)

        return {
            "k_per_core": k_per_core,
            "act_total_tiles": act_total_tiles,
            "weights_num_pages": weights_num_pages,
            "cb_weights_descriptor": cb_weights_descriptor,
            "cb_out_descriptor": cb_out_descriptor,
        }

    @staticmethod
    def setup_gated_reduce(
        group1_tensor,
        group2_tensor,
        intermed_tensor,
        output_tensor,
        cb_group1_index,
        cb_group2_index,
        cb_intermed_index,
        cb_out_index,
        input_tile,
        data_format,
        k_parallel,
        n_parallel,
    ):
        """
        Setup gated reduce: SiLU(reduce(group1)) * reduce(group2).

        Requires face-view optimization (asserted). Creates CB descriptors
        with face-view tile aliasing.

        Args:
            group1_tensor: Gate gather destination tensor (tensor-backed on sender)
            group2_tensor: Up gather destination tensor (tensor-backed on sender)
            intermed_tensor: Intermediate tensor for reduce (tensor-backed on sender)
            output_tensor: Gated reduce output / mcast source tensor (tensor-backed on sender)
            cb_group1_index: CB index for gate gather destination
            cb_group2_index: CB index for up gather destination
            cb_intermed_index: CB index for intermediate
            cb_out_index: CB index for gated reduce output
            input_tile: Original tile format (e.g., Tile([1, 32]))
            data_format: Data format (dtype)
            k_parallel: K parallelism factor (= tiles_per_k)
            n_parallel: N parallelism factor (= k_num_tiles)

        Returns:
            dict with:
            - tiles_per_k, k_num_tiles: Parallelism dimensions
            - kernel_tiles_per_k, kernel_k_num_tiles: Kernel-level tile counts (face-view)
            - mcast_src_num_pages: Pages in mcast source CB (1 with face-view)
            - face_tile_desc, face_tile_size: Face tile descriptor and size
            - input_tile_size, reduce_tile_size: Tile sizes
            - cb_group1_descriptor, cb_group2_descriptor: CB descriptors (face-view aliased)
            - cb_intermed_descriptor, cb_out_descriptor: CB descriptors
        """
        tiles_per_k = k_parallel
        k_num_tiles = n_parallel

        tile_h, tile_w = input_tile.tile_shape
        input_tile_size = input_tile.get_tile_size(data_format)
        assert can_use_face_view(tile_h, tile_w, tiles_per_k, k_num_tiles), (
            f"Face-view optimization is required for shared expert gated reduce "
            f"(tile={tile_h}x{tile_w}, tiles_per_k={tiles_per_k}, k_num_tiles={k_num_tiles})"
        )

        face_tile = ttnn.Tile([FACE_HEIGHT, FACE_WIDTH])
        face_tile_desc = ttnn.TileDescriptor(FACE_HEIGHT, FACE_WIDTH, False)
        face_tile_size = face_tile.get_tile_size(data_format)
        kernel_tiles_per_k = tiles_per_k
        kernel_k_num_tiles = 1
        mcast_src_num_pages = 1
        reduce_tile_size = face_tile_size

        # CB descriptors with face-view aliasing
        cb_group1_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_group1_index, group1_tensor)
        cb_group1_descriptor.format_descriptors[0].tile = face_tile_desc
        cb_group1_descriptor.format_descriptors[0].page_size = face_tile_size

        cb_group2_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_group2_index, group2_tensor)
        cb_group2_descriptor.format_descriptors[0].tile = face_tile_desc
        cb_group2_descriptor.format_descriptors[0].page_size = face_tile_size

        cb_intermed_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_intermed_index, intermed_tensor)
        cb_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_out_index, output_tensor)

        return {
            "tiles_per_k": tiles_per_k,
            "k_num_tiles": k_num_tiles,
            "kernel_tiles_per_k": kernel_tiles_per_k,
            "kernel_k_num_tiles": kernel_k_num_tiles,
            "mcast_src_num_pages": mcast_src_num_pages,
            "face_tile_desc": face_tile_desc,
            "face_tile_size": face_tile_size,
            "input_tile_size": input_tile_size,
            "reduce_tile_size": reduce_tile_size,
            "cb_group1_descriptor": cb_group1_descriptor,
            "cb_group2_descriptor": cb_group2_descriptor,
            "cb_intermed_descriptor": cb_intermed_descriptor,
            "cb_out_descriptor": cb_out_descriptor,
        }

    @staticmethod
    def setup_residual_add(
        output_tensor,
        cb_out_index,
        num_matmul_cores,
        out_w_per_core,
    ):
        """
        Setup residual add: matmul_out + residual → residual_add_out on matmul cores.

        A simple element-wise add where both operands are already present on
        each core (matmul output and mcasted residual).

        Args:
            output_tensor: Output tensor for residual add (WIDTH_SHARDED on matmul cores)
            cb_out_index: CB index for output
            num_matmul_cores: Number of matmul cores performing the add
            out_w_per_core: Output width per core in tiles

        Returns:
            dict with:
            - out_cb: Output CB index
            - total_in1_tiles: Total residual tiles across all cores
            - cb_out_descriptor: CB descriptor for output (tensor-backed)
        """
        cb_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_out_index, output_tensor)

        return {
            "out_cb": cb_out_index,
            "total_in1_tiles": num_matmul_cores * out_w_per_core,
            "cb_out_descriptor": cb_out_descriptor,
        }

    @staticmethod
    def setup_kn_per_core_values(
        a_cores_list,
        b_cores_list,
        k_per_core,
        n_parallel,
    ):
        """
        Compute per-core values for KN-sliced matmul + gather.

        For each compute core, determines:
        - k_offset: Starting K-tile index for the matmul slice
        - sender_idx: Index into the gather destination CB (face-view layout)

        Args:
            a_cores_list: Gate (group A) compute cores
            b_cores_list: Up (group B) compute cores
            k_per_core: K tiles per core (from setup_kn_sliced_matmul)
            n_parallel: N parallelism factor

        Returns:
            dict with:
            - k_offset_core_values: [(core, k_offset), ...] for A and B cores
            - ag_sender_idx_core_values: [(core, sender_idx), ...] for A cores
            - bg_sender_idx_core_values: [(core, sender_idx), ...] for B cores
        """
        # K-offset for matmul: each row of k_parallel cores shares the same k_offset
        k_offset_core_values = [(core, (i // n_parallel) * k_per_core) for i, core in enumerate(a_cores_list)] + [
            (core, (i // n_parallel) * k_per_core) for i, core in enumerate(b_cores_list)
        ]

        # Gather sender indices (face-view layout: row-major in [k_parallel, n_parallel])
        def _sender_indices(cores_list):
            return [
                (core, (lid // n_parallel) * n_parallel + (lid % n_parallel)) for lid, core in enumerate(cores_list)
            ]

        return {
            "k_offset_core_values": k_offset_core_values,
            "ag_sender_idx_core_values": _sender_indices(a_cores_list),
            "bg_sender_idx_core_values": _sender_indices(b_cores_list),
        }

    @staticmethod
    def setup_output_mcast(
        gather_dst_num_pages,
        tile_size,
        dst_num_pages,
    ):
        """
        Setup output mcast: sender → all cores, DRAM cores receive shared expert output.

        The sender multicasts the full gathered output. DRAM cores receive it
        into their eltwise add input CB (add_cb_in1) which has a different
        page size than the source CB.

        src_num_pages and dst_num_pages differ because the source CB uses
        1×32 tile-sized pages while the destination CB (add_cb_in1) uses
        slice-sized pages (width_per_core elements each). The total bytes
        are the same: gather_dst_num_pages * tile_size == dst_num_pages * slice_size.

        Args:
            gather_dst_num_pages: Total pages in the gather destination (= num_matmul_cores * out_w)
            tile_size: Size of one tile in bytes (1×32 tile)
            dst_num_pages: Pages in the destination CB (= total_width / width_per_core)

        Returns:
            dict with:
            - data_size_bytes: Total mcast data size
            - src_num_pages: Pages to read from source CB (tile-sized pages)
            - dst_num_pages: Pages each receiver pushes (slice-sized pages)
        """
        return {
            "data_size_bytes": gather_dst_num_pages * tile_size,
            "src_num_pages": gather_dst_num_pages,
            "dst_num_pages": dst_num_pages,
        }

    @staticmethod
    def _setup_dimensions(
        device,
        shared_gate_weights_overlapped,
        shared_up_weights_overlapped,
        shared_down_mcast_dst_tensor,
        shared_down_weights_tensor,
        shared_output_tensor,
        shared_ag_gather_dst_tensor,
        shared_bg_gather_dst_tensor,
        shared_gu_out_tensor,
        shared_intermed_tensor,
        shared_down_mcast_src_tensor,
        shared_down_matmul_out_tensor,
        shared_residual_add_out_tensor,
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
        # Semaphore IDs (overridable, defaults match MoeOp layout)
        ag_receiver_semaphore_id=2,
        bg_receiver_semaphore_id=3,
        ag_noc1_receiver_semaphore_id=2,
        bg_noc1_receiver_semaphore_id=3,
        shared_mcast_sender_semaphore_id=0,
        shared_mcast_receiver_semaphore_id=3,
        output_gather_noc0_receiver_semaphore_id=4,
        output_gather_noc1_receiver_semaphore_id=4,
        output_mcast_sender_semaphore_id=0,
        output_mcast_receiver_semaphore_id=4,
    ):
        """
        Compute shared expert dimensions and build _MoeSharedExpertContext.

        Args:
            device: TT device (single chip)
            shared_gate_weights_overlapped: Gate proj OverlappedTensor (shares fused backing tensor with up)
            shared_up_weights_overlapped: Up proj OverlappedTensor (shares fused backing tensor with gate)
            shared_down_mcast_dst_tensor: Destination tensor for down mcast (on mcast grid)
            shared_down_weights_tensor: Down projection weights tensor
            shared_output_tensor: Output tensor for shared expert
            shared_ag_gather_dst_tensor: Tensor-backed CB for gate gather destination (sender core)
            shared_bg_gather_dst_tensor: Tensor-backed CB for up gather destination (sender core)
            shared_gu_out_tensor: Tensor-backed CB for gate/up matmul output (128 compute cores)
            shared_intermed_tensor: Tensor-backed CB for gated reduce intermediate (sender core)
            shared_down_mcast_src_tensor: Tensor-backed CB for gated reduce output / down mcast source (sender core)
            shared_down_matmul_out_tensor: Tensor-backed CB for down proj matmul output (112 matmul cores)
            shared_residual_add_out_tensor: Tensor-backed CB for residual add output (112 matmul cores)
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
            ag_receiver_semaphore_id: Gate gather NOC0 receiver sem ID
            bg_receiver_semaphore_id: Up gather NOC0 receiver sem ID
            ag_noc1_receiver_semaphore_id: Gate gather NOC1 receiver sem ID
            bg_noc1_receiver_semaphore_id: Up gather NOC1 receiver sem ID
            shared_mcast_sender_semaphore_id: Shared mcast sender sem ID
            shared_mcast_receiver_semaphore_id: Shared mcast receiver sem ID
            output_gather_noc0_receiver_semaphore_id: Output gather NOC0 receiver sem ID
            output_gather_noc1_receiver_semaphore_id: Output gather NOC1 receiver sem ID
            output_mcast_sender_semaphore_id: Output mcast sender sem ID
            output_mcast_receiver_semaphore_id: Output mcast receiver sem ID

        Returns:
            _MoeSharedExpertContext
        """

        # ==================================================================
        # CB indices
        # ==================================================================
        shared_gu_weights_cb = 28
        shared_gu_out_cb = 29
        shared_group1_cb = 30  # gate gather dst on sender
        shared_group2_cb = 31  # up gather dst on sender
        shared_intermed_cb = 32  # gated reduce intermediate on sender
        shared_mcast_src_cb = 33  # gated reduce output on sender
        shared_residual_cb = MoeSharedExpertOp.RESIDUAL_MCAST_DST_CB  # = residual_mcast_dst_cb
        shared_down_mcast_dst_cb = 34  # down mcast destination (gated reduce output → all 130 cores)
        shared_down_matmul_in1_cb = 35  # down proj weights (112 matmul cores, tensor-backed)
        shared_down_matmul_out_cb = 36  # down proj matmul output (112 matmul cores, tensor-backed)
        shared_residual_add_out_cb = 37  # residual add output (112 matmul cores, tensor-backed)
        shared_output_gather_dst_cb = 38  # output gather destination (sender core, tensor-backed)

        # ==================================================================
        # Dimensions
        # ==================================================================
        num_compute_cores = 64  # per branch (gate/up)
        assert k_parallel * n_parallel == num_compute_cores
        total_gather_tiles = num_compute_cores  # 64
        gu_gather_data_size_bytes = input_tile_size  # each compute core sends 1 tile

        # ==================================================================
        # Core grids — from OverlappedTensors
        # ==================================================================
        shared_gate_up_weights_tensor = shared_gate_weights_overlapped.fused_tensor
        a_compute_grid = shared_gate_weights_overlapped.core_range_set
        b_compute_grid = shared_up_weights_overlapped.core_range_set
        compute_core_grid = shared_gate_up_weights_tensor.memory_config().shard_spec.grid

        # Per-core values (k_offset, sender_idx) depend on row-major ordering
        # that must match the weight data layout produced by BDW's
        # _crs_shard_permutation.  The OverlappedTensor CRS enumeration order
        # differs from row-major, so we keep build_ab_grids() for the ordered
        # lists while using the OverlappedTensor grids for CoreRangeSet membership.
        from models.demos.deepseek_v3_b1.fused_ops.shared_expert.op import SharedExpertOp

        a_cores_list, b_cores_list = SharedExpertOp.build_ab_grids()
        assert len(a_cores_list) == num_compute_cores
        assert len(b_cores_list) == num_compute_cores
        assert set((c.x, c.y) for c in a_cores_list) == set(
            (c.x, c.y) for c in ttnn.corerange_to_cores(a_compute_grid)
        ), "Gate cores from OverlappedTensor don't match build_ab_grids()"
        assert set((c.x, c.y) for c in b_cores_list) == set(
            (c.x, c.y) for c in ttnn.corerange_to_cores(b_compute_grid)
        ), "Up cores from OverlappedTensor don't match build_ab_grids()"

        # NOC coordinates for gather destination (sender core)
        gather_dest_noc_core = device.worker_core_from_logical_core(sender_core)

        # ==================================================================
        # Gather destination addresses (from tensor-backed CBs)
        # ==================================================================
        ag_receiver_data_addr = shared_ag_gather_dst_tensor.buffer_address()
        bg_receiver_data_addr = shared_bg_gather_dst_tensor.buffer_address()

        # ==================================================================
        # Setup APIs — KN-sliced matmul and gated reduce
        # ==================================================================
        gu_matmul_params = MoeSharedExpertOp.setup_kn_sliced_matmul(
            weights_tensor=shared_gate_up_weights_tensor,
            output_tensor=shared_gu_out_tensor,
            cb_weights_index=shared_gu_weights_cb,
            cb_out_index=shared_gu_out_cb,
            num_tiles_k=num_tiles_k,
            k_parallel=k_parallel,
            n_parallel=n_parallel,
        )

        gated_reduce_params = MoeSharedExpertOp.setup_gated_reduce(
            group1_tensor=shared_ag_gather_dst_tensor,
            group2_tensor=shared_bg_gather_dst_tensor,
            intermed_tensor=shared_intermed_tensor,
            output_tensor=shared_down_mcast_src_tensor,
            cb_group1_index=shared_group1_cb,
            cb_group2_index=shared_group2_cb,
            cb_intermed_index=shared_intermed_cb,
            cb_out_index=shared_mcast_src_cb,
            input_tile=input_tile,
            data_format=data_format,
            k_parallel=k_parallel,
            n_parallel=n_parallel,
        )

        # ==================================================================
        # Down Mcast (gated reduce output → all 130 cores for down proj)
        # ==================================================================
        mcast_src_num_pages = gated_reduce_params["mcast_src_num_pages"]
        reduce_tile_size = gated_reduce_params["reduce_tile_size"]
        down_mcast_data_size_bytes = mcast_src_num_pages * reduce_tile_size

        down_mcast_params = MoeOp.setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=shared_down_mcast_dst_tensor.memory_config().shard_spec.grid,
            src_cb=shared_mcast_src_cb,
            src_tensor=shared_down_mcast_src_tensor,
            dst_cb=shared_down_mcast_dst_cb,
            dst_tensor=shared_down_mcast_dst_tensor,
            sender_semaphore_id=shared_mcast_sender_semaphore_id,
            receiver_semaphore_id=shared_mcast_receiver_semaphore_id,
            data_size_bytes=down_mcast_data_size_bytes,
        )

        # ==================================================================
        # Down Proj Matmul (SRAM matmul on non-DRAM cores)
        # ==================================================================
        matmul_core_grid = shared_down_weights_tensor.memory_config().shard_spec.grid
        num_matmul_cores = matmul_core_grid.num_cores()

        from models.demos.deepseek_v3_b1.fused_ops.down_proj.op import DownProj

        assert (
            num_matmul_cores == DownProj.NUM_MATMUL_CORES
        ), f"Down weights tensor has {num_matmul_cores} cores, expected {DownProj.NUM_MATMUL_CORES}"
        assert set((c.x, c.y) for c in ttnn.corerange_to_cores(matmul_core_grid)) == set(
            (c.x, c.y) for c in ttnn.corerange_to_cores(DownProj.build_matmul_core_grid())
        ), "Down weights matmul cores from tensor don't match DownProj.build_matmul_core_grid()"

        down_matmul_params = MoeOp.setup_sram_matmul(
            in0_cb=shared_down_mcast_dst_cb,
            in1_cb=shared_down_matmul_in1_cb,
            out_cb=shared_down_matmul_out_cb,
            weights_tensor=shared_down_weights_tensor,
            output_tensor=shared_down_matmul_out_tensor,
            k_num_tiles=n_parallel,  # K_down_tiles = n_parallel
        )

        # ==================================================================
        # Residual Add (matmul_out + residual → residual_add_out on matmul cores)
        # ==================================================================
        residual_add_params = MoeSharedExpertOp.setup_residual_add(
            output_tensor=shared_residual_add_out_tensor,
            cb_out_index=shared_residual_add_out_cb,
            num_matmul_cores=num_matmul_cores,
            out_w_per_core=down_matmul_params["out_w"],
        )

        # ==================================================================
        # Output Gather (matmul cores → sender)
        # ==================================================================
        output_gather_params = MoeOp.setup_gather(
            device=device,
            receiver_core=sender_core,
            sender_core_ranges=matmul_core_grid,
            num_senders=num_matmul_cores,
            data_size_bytes_per_sender=down_matmul_params["out_w"] * tile_1x32_size,
            src_cb=shared_residual_add_out_cb,
            src_num_pages=down_matmul_params["out_w"],
            dst_cb=shared_output_gather_dst_cb,
            dst_tensor=shared_output_tensor,
            noc0_receiver_semaphore_id=output_gather_noc0_receiver_semaphore_id,
            noc1_receiver_semaphore_id=output_gather_noc1_receiver_semaphore_id,
        )

        # ==================================================================
        # Output Mcast (sender → all cores, DRAM cores receive into add_cb_in1)
        # ==================================================================
        mcast_total_cores = mcast_grid.num_cores()
        num_phantom_cores = sender_core.y  # col 12, rows 0..(sender.y-1)
        num_dram_worker_cores = mcast_total_cores - num_matmul_cores - num_phantom_cores - 1
        output_mcast_params = MoeSharedExpertOp.setup_output_mcast(
            gather_dst_num_pages=output_gather_params["dst_num_pages"],
            tile_size=tile_1x32_size,
            dst_num_pages=num_dram_worker_cores,
        )

        # ==================================================================
        # Per-core values
        # ==================================================================
        per_core_values = MoeSharedExpertOp.setup_kn_per_core_values(
            a_cores_list=a_cores_list,
            b_cores_list=b_cores_list,
            k_per_core=gu_matmul_params["k_per_core"],
            n_parallel=n_parallel,
        )
        gu_k_offset_core_values = per_core_values["k_offset_core_values"]
        ag_sender_idx_core_values = per_core_values["ag_sender_idx_core_values"]
        bg_sender_idx_core_values = per_core_values["bg_sender_idx_core_values"]

        return _MoeSharedExpertContext(
            # Core grids
            compute_core_grid=compute_core_grid,
            a_compute_grid=a_compute_grid,
            b_compute_grid=b_compute_grid,
            a_cores_list=a_cores_list,
            b_cores_list=b_cores_list,
            matmul_core_grid=matmul_core_grid,
            # CB indices
            gu_weights_cb=shared_gu_weights_cb,
            gu_out_cb=shared_gu_out_cb,
            group1_cb=shared_group1_cb,
            group2_cb=shared_group2_cb,
            intermed_cb=shared_intermed_cb,
            mcast_src_cb=shared_mcast_src_cb,
            residual_cb=shared_residual_cb,
            down_mcast_dst_cb=shared_down_mcast_dst_cb,
            # Parallelism
            n_parallel=n_parallel,
            k_parallel=k_parallel,
            num_compute_cores=num_compute_cores,
            # Gather params
            gather_dest_noc_core=gather_dest_noc_core,
            gu_gather_data_size_bytes=gu_gather_data_size_bytes,
            total_gather_tiles=total_gather_tiles,
            # Semaphore IDs
            ag_receiver_semaphore_id=ag_receiver_semaphore_id,
            bg_receiver_semaphore_id=bg_receiver_semaphore_id,
            ag_noc1_receiver_semaphore_id=ag_noc1_receiver_semaphore_id,
            bg_noc1_receiver_semaphore_id=bg_noc1_receiver_semaphore_id,
            shared_mcast_sender_semaphore_id=shared_mcast_sender_semaphore_id,
            shared_mcast_receiver_semaphore_id=shared_mcast_receiver_semaphore_id,
            output_gather_noc0_receiver_semaphore_id=output_gather_noc0_receiver_semaphore_id,
            output_gather_noc1_receiver_semaphore_id=output_gather_noc1_receiver_semaphore_id,
            output_mcast_sender_semaphore_id=output_mcast_sender_semaphore_id,
            output_mcast_receiver_semaphore_id=output_mcast_receiver_semaphore_id,
            # Keep-alive tensors
            ag_dummy_tensor=shared_ag_gather_dst_tensor,
            bg_dummy_tensor=shared_bg_gather_dst_tensor,
            ag_receiver_data_addr=ag_receiver_data_addr,
            bg_receiver_data_addr=bg_receiver_data_addr,
            down_mcast_dst_dummy_tensor=shared_down_mcast_dst_tensor,
            output_gather_dst_dummy_tensor=shared_output_tensor,
            # Setup result dicts
            gu_matmul_params=gu_matmul_params,
            gated_reduce_params=gated_reduce_params,
            down_mcast_params=down_mcast_params,
            down_matmul_params=down_matmul_params,
            residual_add_params=residual_add_params,
            output_gather_params=output_gather_params,
            output_mcast_params=output_mcast_params,
            # Per-core values
            gu_k_offset_core_values=gu_k_offset_core_values,
            ag_sender_idx_core_values=ag_sender_idx_core_values,
            bg_sender_idx_core_values=bg_sender_idx_core_values,
        )

    @staticmethod
    def _build_compile_time_args(shared_ctx, rmsnorm_mcast_dst_cb, rmsnorm_mcast_params):
        """
        Build shared expert compile-time args to append to routed expert args.

        Args:
            shared_ctx: _MoeSharedExpertContext
            rmsnorm_mcast_dst_cb: The CB index for the rmsnorm mcast destination (shared with routed)
            rmsnorm_mcast_params: RMSNorm mcast params (reuse NOC coords for residual mcast)

        Returns:
            (ncrisc_args, brisc_args, trisc_args) - lists of named compile-time arg tuples
        """

        ncrisc_args = [
            # Gate/Up weights setup
            ("shared_gu_weights_cb", shared_ctx.gu_weights_cb),
            ("shared_gu_weights_num_pages", shared_ctx.gu_matmul_params["weights_num_pages"]),
            # Gate gather (A) receiver (MoeGather: receiver on NCRISC)
            ("shared_ag_noc0_num_senders", shared_ctx.num_compute_cores),
            ("shared_ag_noc0_receiver_semaphore_id", shared_ctx.ag_receiver_semaphore_id),
            ("shared_ag_noc1_receiver_semaphore_id", shared_ctx.ag_noc1_receiver_semaphore_id),
            ("shared_ag_dst_cb", shared_ctx.group1_cb),
            ("shared_ag_dst_num_pages", shared_ctx.gated_reduce_params["kernel_tiles_per_k"]),
            # Up gather (B) receiver (MoeGather: receiver on NCRISC)
            ("shared_bg_noc0_num_senders", shared_ctx.num_compute_cores),
            ("shared_bg_noc0_receiver_semaphore_id", shared_ctx.bg_receiver_semaphore_id),
            ("shared_bg_noc1_receiver_semaphore_id", shared_ctx.bg_noc1_receiver_semaphore_id),
            ("shared_bg_dst_cb", shared_ctx.group2_cb),
            ("shared_bg_dst_num_pages", shared_ctx.gated_reduce_params["kernel_tiles_per_k"]),
            # Down mcast receiver
            ("shared_down_mcast_data_receiver_semaphore", shared_ctx.shared_mcast_receiver_semaphore_id),
            ("shared_down_mcast_dst_cb", shared_ctx.down_mcast_dst_cb),
            ("shared_down_mcast_dst_num_pages", shared_ctx.down_matmul_params["k_num_tiles"]),
            # Down proj weights (setup_sharded_buffer on 112 matmul cores)
            ("shared_down_matmul_in1", shared_ctx.down_matmul_params["in1_cb"]),
            ("shared_down_matmul_k_num_tiles", shared_ctx.down_matmul_params["k_num_tiles"]),
            ("shared_down_matmul_out_w_per_core", shared_ctx.down_matmul_params["out_w"]),
            # Output gather receiver (MoeGather: receiver on NCRISC)
            ("shared_og_noc0_num_senders", shared_ctx.output_gather_params["noc0_num_senders"]),
            ("shared_og_noc1_num_senders", shared_ctx.output_gather_params["noc1_num_senders"]),
            ("shared_og_noc0_receiver_semaphore_id", shared_ctx.output_gather_params["noc0_receiver_semaphore_id"]),
            ("shared_og_noc1_receiver_semaphore_id", shared_ctx.output_gather_params["noc1_receiver_semaphore_id"]),
            ("shared_og_dst_cb", shared_ctx.output_gather_params["dst_cb"]),
            ("shared_og_dst_num_pages", shared_ctx.output_gather_params["dst_num_pages"]),
            # Output mcast receiver (DRAM cores receive into add_cb_in1) — separate semaphore
            ("shared_output_mcast_data_receiver_semaphore", shared_ctx.output_mcast_receiver_semaphore_id),
            ("shared_output_mcast_dst_num_pages", shared_ctx.output_mcast_params["dst_num_pages"]),
        ]
        brisc_args = [
            # Gate gather (A) sender (MoeGather: sender on BRISC)
            ("shared_ag_dest_noc_x", shared_ctx.gather_dest_noc_core.x),
            ("shared_ag_dest_noc_y", shared_ctx.gather_dest_noc_core.y),
            ("shared_ag_data_size_bytes", shared_ctx.gu_gather_data_size_bytes),
            ("shared_ag_receiver_semaphore_id", shared_ctx.ag_receiver_semaphore_id),
            ("shared_ag_src_cb", shared_ctx.gu_out_cb),
            ("shared_ag_src_num_pages", 1),
            ("shared_ag_receiver_data_addr", shared_ctx.ag_receiver_data_addr),
            # Up gather (B) sender (MoeGather: sender on BRISC)
            ("shared_bg_dest_noc_x", shared_ctx.gather_dest_noc_core.x),
            ("shared_bg_dest_noc_y", shared_ctx.gather_dest_noc_core.y),
            ("shared_bg_data_size_bytes", shared_ctx.gu_gather_data_size_bytes),
            ("shared_bg_receiver_semaphore_id", shared_ctx.bg_receiver_semaphore_id),
            ("shared_bg_src_cb", shared_ctx.gu_out_cb),
            ("shared_bg_src_num_pages", 1),
            ("shared_bg_receiver_data_addr", shared_ctx.bg_receiver_data_addr),
            # Down mcast sender (CTArgs reused from routed mcast; only need semaphores, CBs, sizes)
            ("shared_down_mcast_data_sender_semaphore", shared_ctx.shared_mcast_sender_semaphore_id),
            ("shared_down_mcast_data_receiver_semaphore", shared_ctx.shared_mcast_receiver_semaphore_id),
            ("shared_down_mcast_data_size_bytes", shared_ctx.down_mcast_params["data_size_bytes"]),
            ("shared_down_mcast_src_cb", shared_ctx.mcast_src_cb),  # gated reduce output (CB 31)
            ("shared_down_mcast_src_num_pages", shared_ctx.down_mcast_params["src_num_pages"]),
            ("shared_down_mcast_dst_cb", shared_ctx.down_mcast_dst_cb),
            # Output gather sender (MoeGather: sender on BRISC)
            ("shared_og_dest_noc_x", shared_ctx.output_gather_params["dest_noc_x"]),
            ("shared_og_dest_noc_y", shared_ctx.output_gather_params["dest_noc_y"]),
            ("shared_og_data_size_bytes", shared_ctx.output_gather_params["data_size_bytes"]),
            ("shared_og_receiver_semaphore_id", shared_ctx.output_gather_params["receiver_semaphore_id"]),
            ("shared_og_src_cb", shared_ctx.output_gather_params["src_cb"]),
            ("shared_og_src_num_pages", shared_ctx.output_gather_params["src_num_pages"]),
            ("shared_og_receiver_data_addr", shared_ctx.output_gather_params["receiver_data_addr"]),
            # Output mcast sender (sender core → 130 cores) — separate semaphores to avoid race
            ("shared_output_mcast_data_sender_semaphore", shared_ctx.output_mcast_sender_semaphore_id),
            ("shared_output_mcast_data_receiver_semaphore", shared_ctx.output_mcast_receiver_semaphore_id),
            ("shared_output_mcast_data_size_bytes", shared_ctx.output_mcast_params["data_size_bytes"]),
            ("shared_output_mcast_src_cb", shared_ctx.output_gather_params["dst_cb"]),  # read from output gather dst
            ("shared_output_mcast_src_num_pages", shared_ctx.output_mcast_params["src_num_pages"]),
        ]
        trisc_args = [
            # Gate/Up matmul
            ("shared_gu_act_cb", rmsnorm_mcast_dst_cb),
            ("shared_gu_weights_cb", shared_ctx.gu_weights_cb),
            ("shared_gu_out_cb", shared_ctx.gu_out_cb),
            ("shared_gu_k_per_core", shared_ctx.gu_matmul_params["k_per_core"]),
            ("shared_gu_act_total_tiles", shared_ctx.gu_matmul_params["act_total_tiles"]),
            # Gated reduce
            ("shared_gated_reduce_group1_cb", shared_ctx.group1_cb),
            ("shared_gated_reduce_group2_cb", shared_ctx.group2_cb),
            ("shared_gated_reduce_intermed_cb", shared_ctx.intermed_cb),
            ("shared_gated_reduce_mcast_src_cb", shared_ctx.mcast_src_cb),
            ("shared_gated_reduce_tiles_per_k", shared_ctx.gated_reduce_params["kernel_tiles_per_k"]),
            ("shared_gated_reduce_k_num_tiles", shared_ctx.gated_reduce_params["kernel_k_num_tiles"]),
            # Down proj matmul
            ("shared_down_matmul_in0", shared_ctx.down_mcast_dst_cb),  # mcast'd activation
            ("shared_down_matmul_in1", shared_ctx.down_matmul_params["in1_cb"]),
            ("shared_down_matmul_out", shared_ctx.down_matmul_params["out_cb"]),
            ("shared_down_matmul_k_num_tiles", shared_ctx.down_matmul_params["k_num_tiles"]),
            ("shared_down_matmul_out_w_per_core", shared_ctx.down_matmul_params["out_w"]),
            # Residual add
            ("shared_residual_add_in0", shared_ctx.down_matmul_params["out_cb"]),  # matmul output
            ("shared_residual_add_in1", shared_ctx.residual_cb),  # residual (pre-loaded bias)
            ("shared_residual_add_out", shared_ctx.residual_add_params["out_cb"]),
            ("shared_residual_add_out_w", shared_ctx.down_matmul_params["out_w"]),
            ("shared_residual_add_total_in1_tiles", shared_ctx.residual_add_params["total_in1_tiles"]),
        ]
        return ncrisc_args, brisc_args, trisc_args

    @staticmethod
    def _build_cb_descriptors(shared_ctx):
        """Build CB descriptors for shared expert."""
        return [
            shared_ctx.gu_matmul_params["cb_weights_descriptor"],
            shared_ctx.gu_matmul_params["cb_out_descriptor"],
            shared_ctx.gated_reduce_params["cb_group1_descriptor"],
            shared_ctx.gated_reduce_params["cb_group2_descriptor"],
            shared_ctx.gated_reduce_params["cb_intermed_descriptor"],
            shared_ctx.gated_reduce_params["cb_out_descriptor"],
            shared_ctx.down_mcast_params["dst_cb_descriptor"],
            shared_ctx.down_matmul_params["weights_cb_descriptor"],
            shared_ctx.down_matmul_params["output_cb_descriptor"],
            shared_ctx.residual_add_params["cb_out_descriptor"],
            shared_ctx.output_gather_params["dst_cb_descriptor"],
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
                core_range=shared_ctx.matmul_core_grid,
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
                    (core, idx) for idx, core in enumerate(ttnn.corerange_to_cores(shared_ctx.matmul_core_grid))
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

    # Semaphore IDs (top-level definition)
    # Gather sems overlap with mcast receiver sems (different physical cores).
    # noc1_num_senders=0 for all gathers, so only noc0 sem is used.
    MCAST_SENDER_SEM = 0
    MCAST_DATA_RECEIVER_SEM = 1  # mcast grid; overlaps with DOWN_PROJ_GATHER_SEM on sender core
    DOWN_PROJ_GATHER_SEM = 1  # sender core
    RESIDUAL_MCAST_RECEIVER_SEM = 2  # mcast grid; overlaps with AG_GATHER_SEM on sender core
    AG_GATHER_SEM = 2  # sender core
    SHARED_DOWN_MCAST_RECEIVER_SEM = 3  # mcast grid; overlaps with BG_GATHER_SEM on sender core
    BG_GATHER_SEM = 3  # sender core
    SHARED_OUTPUT_MCAST_RECEIVER_SEM = 4  # mcast grid; overlaps with OUTPUT_GATHER_SEM on sender core
    OUTPUT_GATHER_SEM = 4  # sender core
    # MoE-only mcast receivers (separate, not overlapped)
    EXPERT_SCALE_MCAST_RECEIVER_SEM = 5
    INDEX_MCAST_RECEIVER_SEM = 6
    DOWN_PROJ_MCAST_RECEIVER_SEM = 7
    REDUCE_WORKER_FABRIC_SEM_BASE = 8  # on fabric cores only (8, 9, 10, 11 per worker slot)

    # ------------------------------------------------------------------
    # Shared utility setup APIs (used by both routed and shared experts)
    # ------------------------------------------------------------------

    @staticmethod
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

    @staticmethod
    def setup_mcast(
        device,
        sender_core,
        mcast_grid,
        src_cb,
        src_tensor,
        dst_cb,
        dst_tensor,
        sender_semaphore_id,
        receiver_semaphore_id,
        data_size_bytes,
    ):
        """
        Set up parameters for a multicast operation.

        Mcast broadcasts data from a sender core to all cores in the mcast grid.

        Args:
            device: TT device
            sender_core: Logical CoreCoord of the sender (single core)
            mcast_grid: CoreRangeSet of destination cores (rectangular grid)
            src_cb: Source CB index on sender core
            src_tensor: Source tensor on sender core (for num_pages calculation)
            dst_cb: Destination CB index on receiver cores
            dst_tensor: Destination tensor on receiver cores (for CB descriptor)
            sender_semaphore_id: Semaphore ID for sender
            receiver_semaphore_id: Semaphore ID for receivers
            data_size_bytes: Total data size to mcast in bytes

        Returns:
            Dictionary with mcast parameters for compile-time args
        """
        # Get mcast grid bounds
        mcast_ranges = list(mcast_grid.ranges())
        mcast_grid_range = mcast_ranges[0]  # Single rectangular range
        mcast_grid_start = mcast_grid_range.start
        mcast_grid_end = mcast_grid_range.end

        # Get NOC coordinates for mcast destination
        dest_noc_start_core = device.worker_core_from_logical_core(mcast_grid_start)
        dest_noc_end_core = device.worker_core_from_logical_core(mcast_grid_end)

        # Calculate number of cores in mcast grid
        num_cores = (mcast_grid_end.x - mcast_grid_start.x + 1) * (mcast_grid_end.y - mcast_grid_start.y + 1)

        # Check if sender is part of receiver grid
        sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])
        is_sender_part_of_receiver_grid = mcast_grid.contains(sender_core_grid)

        # Calculate num_pages from source tensor shard shape
        src_shard_shape = src_tensor.memory_config().shard_spec.shape
        src_tile = src_tensor.get_tile()
        src_num_pages = (src_shard_shape[0] * src_shard_shape[1]) // (src_tile.tile_shape[0] * src_tile.tile_shape[1])

        # CB descriptor for destination
        dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(dst_cb, dst_tensor)

        return {
            # Sender args (BRISC)
            "dest_noc_start_x": dest_noc_start_core.x,
            "dest_noc_start_y": dest_noc_start_core.y,
            "dest_noc_end_x": dest_noc_end_core.x,
            "dest_noc_end_y": dest_noc_end_core.y,
            "num_cores": num_cores,
            "sender_semaphore_id": sender_semaphore_id,
            "receiver_semaphore_id": receiver_semaphore_id,
            "data_size_bytes": data_size_bytes,
            "src_cb": src_cb,
            "src_num_pages": src_num_pages,
            "is_sender_part_of_receiver_grid": is_sender_part_of_receiver_grid,
            # Receiver args (NCRISC)
            "dst_cb": dst_cb,
            "dst_num_pages": src_num_pages,  # Same as src_num_pages
            # CB descriptor
            "dst_cb_descriptor": dst_cb_descriptor,
        }

    @staticmethod
    def setup_gather(
        device,
        receiver_core,
        sender_core_ranges,
        num_senders,
        data_size_bytes_per_sender,
        src_cb,
        src_num_pages,
        dst_cb,
        dst_tensor,
        noc0_receiver_semaphore_id,
        noc1_receiver_semaphore_id,
        row_major=True,
        use_explicit_sender_index=False,
    ):
        """
        Set up parameters for a gather operation.

        Gather collects data from multiple sender cores to a single receiver core.

        Args:
            device: TT device
            receiver_core: Logical CoreCoord of the receiver (single core)
            sender_core_ranges: CoreRangeSet of sender cores
            num_senders: Total number of sender cores
            data_size_bytes_per_sender: Bytes each sender sends
            src_cb: Source CB index on sender cores
            src_num_pages: Number of pages to wait for in source CB
            dst_cb: Destination CB index on receiver core
            dst_tensor: Destination tensor on receiver core (for buffer address and CB descriptor)
            noc0_receiver_semaphore_id: Semaphore ID for NOC0 senders
            noc1_receiver_semaphore_id: Semaphore ID for NOC1 senders
            row_major: Grid traversal order (True=row-major, False=column-major)
            use_explicit_sender_index: If True, use explicit per-core sender index (for scattered cores)

        Returns:
            Dictionary with gather parameters for NCRISC and BRISC compile-time args
        """
        # Get receiver NOC coordinates
        receiver_core_noc = device.worker_core_from_logical_core(receiver_core)

        # Get sender grid bounding box
        sender_ranges = list(sender_core_ranges.ranges())
        sender_min_x = min(r.start.x for r in sender_ranges)
        sender_min_y = min(r.start.y for r in sender_ranges)
        sender_max_x = max(r.end.x for r in sender_ranges)
        sender_max_y = max(r.end.y for r in sender_ranges)

        # All senders use NOC0 for simplicity
        noc0_num_senders = num_senders
        noc1_num_senders = 0

        # Calculate dst_num_pages from destination tensor shard shape
        dst_shard_shape = dst_tensor.memory_config().shard_spec.shape
        dst_tile = dst_tensor.get_tile()
        dst_num_pages = (dst_shard_shape[0] * dst_shard_shape[1]) // (dst_tile.tile_shape[0] * dst_tile.tile_shape[1])

        # CB descriptor for destination
        dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(dst_cb, dst_tensor)

        return {
            # NCRISC (sender) args
            "dest_noc_x": receiver_core_noc.x,
            "dest_noc_y": receiver_core_noc.y,
            "data_size_bytes": data_size_bytes_per_sender,
            "receiver_semaphore_id": noc0_receiver_semaphore_id,
            "src_cb": src_cb,
            "src_num_pages": src_num_pages,
            "sender_grid_start_x": sender_min_x,
            "sender_grid_start_y": sender_min_y,
            "sender_grid_end_x": sender_max_x,
            "sender_grid_end_y": sender_max_y,
            "row_major": 1 if row_major else 0,
            "receiver_data_addr": dst_tensor.buffer_address(),
            # BRISC (receiver) args
            "noc0_num_senders": noc0_num_senders,
            "noc1_num_senders": noc1_num_senders,
            "noc0_receiver_semaphore_id": noc0_receiver_semaphore_id,
            "noc1_receiver_semaphore_id": noc1_receiver_semaphore_id,
            "dst_cb": dst_cb,
            "dst_num_pages": dst_num_pages,
            # CB descriptor
            "dst_cb_descriptor": dst_cb_descriptor,
            # Config
            "use_explicit_sender_index": use_explicit_sender_index,
        }

    @staticmethod
    def setup_sram_matmul(
        in0_cb,
        in1_cb,
        out_cb,
        weights_tensor,
        output_tensor,
        k_num_tiles,
        fused_activation=0,
    ):
        """
        Set up parameters for an SRAM matmul operation.

        SRAM matmul computes: output = input @ weights with optional fused activation.
        Weights and output are sharded in L1 (SRAM).

        Args:
            in0_cb: Input CB index (receives mcasted input)
            in1_cb: Weights CB index
            out_cb: Output CB index
            weights_tensor: Weight tensor (WIDTH_SHARDED in L1)
            output_tensor: Output tensor (WIDTH_SHARDED in L1)
            k_num_tiles: K dimension in tiles
            fused_activation: Activation to fuse (0=none, 1=sigmoid, 2=silu)

        Returns:
            Dictionary with matmul parameters and CB descriptors
        """
        # Get per-core output width in tiles from weights tensor
        weights_tile = weights_tensor.get_tile()
        weights_shard_shape = weights_tensor.memory_config().shard_spec.shape
        weights_shard_width = weights_shard_shape[1]
        out_w = weights_shard_width // weights_tile.tile_shape[1]

        # Get core grid from weights tensor
        core_grid = weights_tensor.memory_config().shard_spec.grid

        # CB descriptors
        weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in1_cb, weights_tensor)
        output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)

        return {
            # CB indices
            "in0_cb": in0_cb,
            "in1_cb": in1_cb,
            "out_cb": out_cb,
            # Matmul parameters
            "k_num_tiles": k_num_tiles,
            "out_w": out_w,
            "fused_activation": fused_activation,
            # Core grid
            "core_grid": core_grid,
            "num_cores": core_grid.num_cores(),
            # CB descriptors
            "weights_cb_descriptor": weights_cb_descriptor,
            "output_cb_descriptor": output_cb_descriptor,
        }

    @staticmethod
    def golden(
        input_tensor,
        routing_weights_tensor,
        bias_tensor,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        gate_proj_weights_dict=None,
        up_proj_weights_dict=None,
        down_proj_weights_dict=None,
        eps=1e-20,
        scaling_factor=2.5,
        use_hardcoded_expert_index=False,
        hardcoded_expert_index=0,
        explicit_expert_scale=None,
        rmsnorm_gamma=None,
        rmsnorm_epsilon=1e-6,
    ):
        """
        PyTorch reference for the full fused MoE (routed + shared expert + eltwise add).

        The shared expert residual is the raw (pre-norm) input tensor itself (mcasted at runtime).
        RMSNorm is applied to the input before feeding it to both shared and routed experts.

        Args:
            input_tensor: [1, K] — raw input (pre-norm)
            routing_weights_tensor: [K, N_routing]
            bias_tensor: [1, 8, 32] gate bias
            shared_gate_weights: [K, K_down] shared expert gate weights
            shared_up_weights: [K, K_down] shared expert up weights
            shared_down_weights: [K_down, N] shared expert down weights
            gate_proj_weights_dict: Dict[int, Tensor] routed expert gate weights
            up_proj_weights_dict: Dict[int, Tensor] routed expert up weights
            down_proj_weights_dict: Dict[int, Tensor] routed expert down weights
            eps, scaling_factor, use_hardcoded_expert_index, hardcoded_expert_index,
            explicit_expert_scale: routed expert gate params
            rmsnorm_gamma: [1, K] RMSNorm gamma weights
            rmsnorm_epsilon: RMSNorm epsilon

        Returns:
            (top8_scores, top8_indices, final_output) tensors
        """
        import torch

        from models.demos.deepseek_v3_b1.fused_ops.shared_expert.op import SharedExpertOp

        # Apply RMSNorm: raw input → normalized input (truncate to bfloat16 to match device)
        x = input_tensor.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        normalized_input = ((x * torch.rsqrt(variance + rmsnorm_epsilon)) * rmsnorm_gamma.float()).bfloat16().float()

        # Shared expert: normalized input for compute, raw input for residual
        shared_output = SharedExpertOp.golden(
            normalized_input.float(),
            shared_gate_weights.float(),
            shared_up_weights.float(),
            shared_down_weights.float(),
            input_tensor.float(),  # residual is the raw (pre-norm) input
        ).bfloat16()

        # Reshape to match routed golden's fused_add_tensor expectation [1,1,1,N]
        shared_for_add = shared_output.float().reshape(1, 1, 1, -1)

        # Routed expert with normalized input and shared output as addend
        return MoeRoutedExpertOp.golden(
            normalized_input,
            routing_weights_tensor,
            bias_tensor,
            gate_proj_weights_dict=gate_proj_weights_dict,
            up_proj_weights_dict=up_proj_weights_dict,
            down_proj_weights_dict=down_proj_weights_dict,
            fused_add_tensor=shared_for_add,
            eps=eps,
            scaling_factor=scaling_factor,
            use_hardcoded_expert_index=use_hardcoded_expert_index,
            hardcoded_expert_index=hardcoded_expert_index,
            explicit_expert_scale=explicit_expert_scale,
        )

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
        down_proj_in1_buf_tensor,
        mul_scalar_buf_tensor,
        # RMSNorm gamma weights (sender core)
        rmsnorm_gamma_tensor,
        # Shared expert tensors
        shared_residual_mcast_src_tensor,
        shared_gate_weights_overlapped,
        shared_up_weights_overlapped,
        shared_residual_mcast_dst_tensor,
        shared_down_mcast_dst_tensor,
        shared_down_weights_tensor,
        shared_output_tensor,
        # Shared expert tensor-backed CB tensors
        shared_ag_gather_dst_tensor,
        shared_bg_gather_dst_tensor,
        shared_gu_out_tensor,
        shared_intermed_tensor,
        shared_down_mcast_src_tensor,
        shared_down_matmul_out_tensor,
        shared_residual_add_out_tensor,
        shared_k_parallel,
        shared_n_parallel,
        epsilon=1e-6,
        use_hardcoded_expert_index=False,
        num_iterations=1,
        # ReduceToOne parameters
        reduce_intermediate_tensors: Optional[list] = None,
        reduce_output_tensor: Optional[ttnn.Tensor] = None,
        reduce_semaphores: Optional[list] = None,
        reduce_root_coord: Optional[ttnn.MeshCoordinate] = None,
    ):
        """
        Execute the full fused MoE operation (routed + shared expert).

        Args:
            shared_gate_weights_overlapped: Gate proj OverlappedTensor.
            shared_up_weights_overlapped: Up proj OverlappedTensor.
            num_iterations: Number of iterations to loop inside the kernel (default 1).
            reduce_intermediate_tensors: (Optional) List of 3 intermediate tensors for reduce rounds.
            reduce_output_tensor: (Optional) Final reduced output tensor on ROOT1 device.
            reduce_semaphores: (Optional) List of 4 global semaphores for reduce synchronization.
            reduce_root_coord: (Optional) MeshCoordinate of ROOT1 device.

        Returns:
            (gate_output_scores_tensor, gate_output_indices_tensor, final_output_tensor or reduce_output_tensor)
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
            down_proj_in1_buf_tensor,
            mul_scalar_buf_tensor,
            shared_residual_mcast_src_tensor=shared_residual_mcast_src_tensor,
            shared_residual_mcast_dst_tensor=shared_residual_mcast_dst_tensor,
            rmsnorm_gamma_tensor=rmsnorm_gamma_tensor,
            epsilon=epsilon,
            use_hardcoded_expert_index=use_hardcoded_expert_index,
            reduce_intermediate_tensors=reduce_intermediate_tensors,
            reduce_output_tensor=reduce_output_tensor,
            reduce_semaphores=reduce_semaphores,
            reduce_root_coord=reduce_root_coord,
            # Semaphore IDs from top-level
            mcast_data_sender_semaphore_id=MoeOp.MCAST_SENDER_SEM,
            mcast_data_receiver_semaphore_id=MoeOp.MCAST_DATA_RECEIVER_SEM,
            gather_noc0_receiver_semaphore_id=MoeOp.DOWN_PROJ_GATHER_SEM,
            gather_noc1_receiver_semaphore_id=MoeOp.DOWN_PROJ_GATHER_SEM,
            expert_scale_mcast_receiver_semaphore_id=MoeOp.EXPERT_SCALE_MCAST_RECEIVER_SEM,
            residual_mcast_receiver_semaphore_id=MoeOp.RESIDUAL_MCAST_RECEIVER_SEM,
            index_mcast_receiver_semaphore_id=MoeOp.INDEX_MCAST_RECEIVER_SEM,
            down_proj_mcast_receiver_semaphore_id=MoeOp.DOWN_PROJ_MCAST_RECEIVER_SEM,
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
            shared_gate_weights_overlapped=shared_gate_weights_overlapped,
            shared_up_weights_overlapped=shared_up_weights_overlapped,
            shared_down_mcast_dst_tensor=shared_down_mcast_dst_tensor,
            shared_down_weights_tensor=shared_down_weights_tensor,
            shared_output_tensor=shared_output_tensor,
            shared_ag_gather_dst_tensor=shared_ag_gather_dst_tensor,
            shared_bg_gather_dst_tensor=shared_bg_gather_dst_tensor,
            shared_gu_out_tensor=shared_gu_out_tensor,
            shared_intermed_tensor=shared_intermed_tensor,
            shared_down_mcast_src_tensor=shared_down_mcast_src_tensor,
            shared_down_matmul_out_tensor=shared_down_matmul_out_tensor,
            shared_residual_add_out_tensor=shared_residual_add_out_tensor,
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
            # Semaphore IDs from top-level
            ag_receiver_semaphore_id=MoeOp.AG_GATHER_SEM,
            bg_receiver_semaphore_id=MoeOp.BG_GATHER_SEM,
            ag_noc1_receiver_semaphore_id=MoeOp.AG_GATHER_SEM,
            bg_noc1_receiver_semaphore_id=MoeOp.BG_GATHER_SEM,
            shared_mcast_sender_semaphore_id=MoeOp.MCAST_SENDER_SEM,
            shared_mcast_receiver_semaphore_id=MoeOp.SHARED_DOWN_MCAST_RECEIVER_SEM,
            output_gather_noc0_receiver_semaphore_id=MoeOp.OUTPUT_GATHER_SEM,
            output_gather_noc1_receiver_semaphore_id=MoeOp.OUTPUT_GATHER_SEM,
            output_mcast_sender_semaphore_id=MoeOp.MCAST_SENDER_SEM,
            output_mcast_receiver_semaphore_id=MoeOp.SHARED_OUTPUT_MCAST_RECEIVER_SEM,
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
        # Semaphore descriptors (8 unique IDs: 0-7)
        # Gather sems overlap with mcast receiver sems (different physical cores).
        # MoE-only mcast receivers (5-7) are separate.
        # Reduce fabric sems (8+) are added later when enable_reduce_to_one.
        # ==================================================================
        semaphore_descriptors = [
            ttnn.SemaphoreDescriptor(
                id=MoeOp.MCAST_SENDER_SEM,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=MoeOp.MCAST_DATA_RECEIVER_SEM,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=MoeOp.RESIDUAL_MCAST_RECEIVER_SEM,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=MoeOp.SHARED_DOWN_MCAST_RECEIVER_SEM,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=MoeOp.SHARED_OUTPUT_MCAST_RECEIVER_SEM,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=MoeOp.EXPERT_SCALE_MCAST_RECEIVER_SEM,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=MoeOp.INDEX_MCAST_RECEIVER_SEM,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=MoeOp.DOWN_PROJ_MCAST_RECEIVER_SEM,
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
            down_proj_in1_buf_tensor,
            mul_scalar_buf_tensor,
            # RMSNorm gamma weights (sender core)
            rmsnorm_gamma_tensor,
            # Shared expert tensors
            shared_residual_mcast_src_tensor,
            shared_gate_weights_overlapped.fused_tensor,
            shared_ag_gather_dst_tensor,
            shared_bg_gather_dst_tensor,
            shared_residual_mcast_dst_tensor,
            shared_down_mcast_dst_tensor,
            shared_down_weights_tensor,
            shared_output_tensor,
            # Shared expert tensor-backed CB tensors
            shared_gu_out_tensor,
            shared_intermed_tensor,
            shared_down_mcast_src_tensor,
            shared_down_matmul_out_tensor,
            shared_residual_add_out_tensor,
        ]

        # ==================================================================
        # Create per-device programs (mesh loop)
        # ==================================================================
        enable_reduce_to_one = routed_ctx.enable_reduce_to_one
        reduce_params = routed_ctx.reduce_params
        mesh_device = input_tensor.device()

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        rmsnorm_mcast_dst_cb = routed_ctx.rmsnorm_mcast_params["dst_cb"]

        for row in range(routed_ctx.mesh_rows):
            for col in range(routed_ctx.mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                chip_id = row * routed_ctx.mesh_cols + col

                # Build compile-time args: routed + shared
                ncrisc_args, brisc_args, trisc_args = MoeRoutedExpertOp._build_compile_time_args(routed_ctx, chip_id)
                shared_ncrisc, shared_brisc, shared_trisc = MoeSharedExpertOp._build_compile_time_args(
                    shared_ctx, rmsnorm_mcast_dst_cb, routed_ctx.rmsnorm_mcast_params
                )
                ncrisc_args += shared_ncrisc
                brisc_args += shared_brisc
                trisc_args += shared_trisc

                # Loop iteration count (available to all RISCs in common section)
                ncrisc_args += [("num_iterations", num_iterations)]
                brisc_args += [("num_iterations", num_iterations)]
                trisc_args += [("num_iterations", num_iterations)]

                # Per-device copies for reduce modification
                device_unified_ct_core_descs = list(unified_core_descs)
                device_per_core_ct_descs = list(per_core_descs)
                device_cb_descriptors = list(cb_descriptors)
                device_semaphore_descriptors = list(semaphore_descriptors)
                device_runtime_args_descriptor = None

                # ReduceToOne per-device setup
                if enable_reduce_to_one:
                    # Get device role
                    device_role = get_reduce_device_role(coord, reduce_root_coord)
                    # Determine destination coordinate based on role
                    if device_role == MESH_LEAF:
                        if row == 0:
                            dest_coord = ttnn.MeshCoordinate(row + 1, col)
                        else:  # row == 3
                            dest_coord = ttnn.MeshCoordinate(row - 1, col)
                    elif device_role == MESH_ROOT3:
                        dest_coord = ttnn.MeshCoordinate(reduce_root_coord[0], col)
                    elif device_role == MESH_ROOT2:
                        dest_coord = reduce_root_coord
                    else:  # MESH_ROOT1
                        dest_coord = reduce_root_coord  # No actual send

                    # Get fabric node IDs
                    dest_fabric_node_id = mesh_device.get_fabric_node_id(dest_coord)

                    # Get per-device tensors for this device
                    r1_tensor = reduce_params["intermediate_r1_per_device"][chip_id]
                    r2_tensor = reduce_params["intermediate_r2_per_device"][chip_id]
                    r3_tensor = reduce_params["intermediate_r3_per_device"][chip_id]
                    out_tensor = reduce_params["output_per_device"][chip_id]

                    # Create CB descriptors for reduce operation
                    reduce_all_cores_set = ttnn.CoreRangeSet(
                        [ttnn.CoreRange(c, c) for c in reduce_params["worker_cores_list"]]
                        + [ttnn.CoreRange(c, c) for c in reduce_params["fabric_cores"]]
                    )
                    reduce_tile_desc = ttnn.TileDescriptor(32, 32)  # 32x32 compute tiles
                    reduce_payload = reduce_params["payload_size_bytes"]
                    reduce_dtype = ttnn.bfloat16

                    # reduce_received_cb_r1: backed by intermediate r1 tensor
                    reduce_cb_r1_desc = ttnn.cb_descriptor_from_sharded_tensor(
                        routed_ctx.reduce_received_cb_r1, r1_tensor
                    )
                    reduce_cb_r1_desc.core_ranges = reduce_all_cores_set
                    reduce_cb_r1_desc.total_size = reduce_payload
                    reduce_cb_r1_desc.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=routed_ctx.reduce_received_cb_r1,
                            data_format=reduce_dtype,
                            page_size=reduce_payload,
                            tile=reduce_tile_desc,
                        )
                    ]
                    device_cb_descriptors.append(reduce_cb_r1_desc)

                    # reduce_received_cb_r2: backed by intermediate r2 tensor
                    reduce_cb_r2_desc = ttnn.cb_descriptor_from_sharded_tensor(
                        routed_ctx.reduce_received_cb_r2, r2_tensor
                    )
                    reduce_cb_r2_desc.core_ranges = reduce_all_cores_set
                    reduce_cb_r2_desc.total_size = reduce_payload
                    reduce_cb_r2_desc.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=routed_ctx.reduce_received_cb_r2,
                            data_format=reduce_dtype,
                            page_size=reduce_payload,
                            tile=reduce_tile_desc,
                        )
                    ]
                    device_cb_descriptors.append(reduce_cb_r2_desc)

                    # reduce_received_cb_r3: backed by intermediate r3 tensor
                    reduce_cb_r3_desc = ttnn.cb_descriptor_from_sharded_tensor(
                        routed_ctx.reduce_received_cb_r3, r3_tensor
                    )
                    reduce_cb_r3_desc.core_ranges = reduce_all_cores_set
                    reduce_cb_r3_desc.total_size = reduce_payload
                    reduce_cb_r3_desc.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=routed_ctx.reduce_received_cb_r3,
                            data_format=reduce_dtype,
                            page_size=reduce_payload,
                            tile=reduce_tile_desc,
                        )
                    ]
                    device_cb_descriptors.append(reduce_cb_r3_desc)

                    # reduce_output_cb: backed by reduce output tensor
                    reduce_cb_out_desc = ttnn.cb_descriptor_from_sharded_tensor(routed_ctx.reduce_output_cb, out_tensor)
                    reduce_cb_out_desc.core_ranges = reduce_all_cores_set
                    reduce_cb_out_desc.total_size = reduce_payload
                    reduce_cb_out_desc.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=routed_ctx.reduce_output_cb,
                            data_format=reduce_dtype,
                            page_size=reduce_payload,
                            tile=reduce_tile_desc,
                        )
                    ]
                    device_cb_descriptors.append(reduce_cb_out_desc)

                    # reduce_scratch_cb: scratch buffer for compute (non-tensor-backed)
                    reduce_scratch_size = reduce_params["compute_tile_size"] * reduce_params["num_tiles"]
                    reduce_cb_scratch_desc = ttnn.CBDescriptor(
                        total_size=reduce_scratch_size,
                        core_ranges=reduce_all_cores_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=routed_ctx.reduce_scratch_cb,
                                data_format=ttnn.bfloat16,
                                page_size=reduce_params["compute_tile_size"],
                                tile=reduce_tile_desc,
                            )
                        ],
                    )
                    device_cb_descriptors.append(reduce_cb_scratch_desc)

                    # reduce_packet_cb
                    reduce_packet_size = reduce_params["slot_size_bytes"] * reduce_params["num_workers_per_column"]
                    reduce_cb_packet_desc = ttnn.CBDescriptor(
                        total_size=reduce_packet_size,
                        core_ranges=reduce_all_cores_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=routed_ctx.reduce_packet_cb,
                                data_format=ttnn.bfloat16,
                                page_size=reduce_params["slot_size_bytes"],
                            )
                        ],
                    )
                    device_cb_descriptors.append(reduce_cb_packet_desc)

                    # reduce_packet_header_cb: persistent packet header storage
                    reduce_packet_header_size = 96  # Standard packet header size
                    reduce_cb_packet_header_desc = ttnn.CBDescriptor(
                        total_size=reduce_packet_header_size,
                        core_ranges=reduce_all_cores_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=routed_ctx.reduce_packet_header_cb,
                                data_format=ttnn.bfloat16,
                                page_size=reduce_packet_header_size,
                            )
                        ],
                    )
                    device_cb_descriptors.append(reduce_cb_packet_header_desc)

                    # Destination L1 address depends on role
                    if device_role == MESH_LEAF:
                        dst_l1_addr = r1_tensor.buffer_address()
                        dst_sem_addr = reduce_params["sem_round1_addr"]
                    elif device_role == MESH_ROOT3:
                        dst_l1_addr = r2_tensor.buffer_address()
                        dst_sem_addr = reduce_params["sem_round2_addr"]
                    elif device_role == MESH_ROOT2:
                        dst_l1_addr = r3_tensor.buffer_address()
                        dst_sem_addr = reduce_params["sem_round3_addr"]
                    else:  # MESH_ROOT1
                        dst_l1_addr = 0
                        dst_sem_addr = reduce_params["sem_exit_addr"]

                    # Get physical coords for output core
                    output_core_phys = routed_ctx.device.worker_core_from_logical_core(reduce_params["output_core"])

                    # Add reduce-specific compile-time args (per-device values)
                    ncrisc_args.extend(
                        [
                            ("reduce_device_role", device_role),
                            ("reduce_num_tiles", reduce_params["num_tiles"]),
                        ]
                    )

                    brisc_args.extend(
                        [
                            ("reduce_device_role", device_role),
                            ("reduce_num_tiles", reduce_params["num_tiles"]),
                            ("reduce_payload_size_bytes", reduce_params["payload_size_bytes"]),
                            ("reduce_num_hops", 1),
                            ("reduce_dst_fabric_node_chip_id", dest_fabric_node_id.chip_id),
                            ("reduce_dst_fabric_node_mesh_id", int(dest_fabric_node_id.mesh_id)),
                            ("reduce_output_core_noc_x", output_core_phys.x),
                            ("reduce_output_core_noc_y", output_core_phys.y),
                            ("reduce_num_workers", reduce_params["num_workers_per_column"]),
                            ("reduce_slot_size_bytes", reduce_params["slot_size_bytes"]),
                            ("reduce_packet_cb", routed_ctx.reduce_packet_cb),
                            ("reduce_packet_header_cb", routed_ctx.reduce_packet_header_cb),
                        ]
                    )

                    trisc_args.extend(
                        [
                            ("reduce_device_role", device_role),
                            ("reduce_num_tiles", reduce_params["num_tiles"]),
                        ]
                    )

                    # Update fabric core descriptor for this device
                    fabric_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in reduce_params["fabric_cores"]])
                    for i, desc in enumerate(device_unified_ct_core_descs):
                        if desc.named_compile_time_arg == "is_reduce_fabric_core":
                            device_unified_ct_core_descs[i] = UnifiedCompileTimeCoreDescriptor(
                                named_compile_time_arg="is_reduce_fabric_core",
                                core_range=fabric_core_set,
                                value=1,
                                other_value=0,
                            )
                            break

                    # Build per-core BRISC args for reduce worker cores
                    reduce_worker_fabric_sem_base = MoeOp.REDUCE_WORKER_FABRIC_SEM_BASE
                    reduce_brisc_per_core_args = []
                    for core in reduce_params["worker_cores_list"]:
                        fabric_core = reduce_params["column_to_fabric_core"][core.x]
                        fabric_core_phys = routed_ctx.device.worker_core_from_logical_core(fabric_core)
                        slot_idx = reduce_params["core_to_slot_idx"][(core.x, core.y)]
                        shard_idx = reduce_params["core_to_shard_idx"][(core.x, core.y)]

                        worker_args = [
                            fabric_core_phys.x,  # fabric_core_noc_x
                            fabric_core_phys.y,  # fabric_core_noc_y
                            slot_idx,  # my_slot_idx
                            reduce_worker_fabric_sem_base + slot_idx,  # worker_sem_id
                            dst_l1_addr,  # dst_l1_addr
                            dst_sem_addr,  # dst_sem_addr
                            out_tensor.buffer_address(),  # output_base_addr
                            shard_idx,  # shard_idx
                        ]
                        reduce_brisc_per_core_args.append((core, worker_args))

                    # Fabric cores BRISC args: worker semaphore IDs
                    for fc in reduce_params["fabric_cores"]:
                        fabric_args = [
                            reduce_worker_fabric_sem_base + i for i in range(reduce_params["num_workers_per_column"])
                        ]
                        reduce_brisc_per_core_args.append((fc, fabric_args))

                    device_runtime_args_descriptor = PerCoreRuntimeArgsDescriptor(
                        brisc_args=reduce_brisc_per_core_args,
                    )

                # Build NCRISC common runtime args for reduce (semaphore addresses)
                ncrisc_common_rt_args = []
                if enable_reduce_to_one:
                    ncrisc_common_rt_args = [
                        reduce_params["sem_round1_addr"],
                        reduce_params["sem_round2_addr"],
                        reduce_params["sem_round3_addr"],
                    ]

                # Build defines list
                kernel_defines = []
                if enable_reduce_to_one:
                    kernel_defines.append(("ENABLE_REDUCE_TO_ONE", "1"))

                # Create unified kernel
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/moe/moe_kernel.cpp",
                    core_ranges=routed_ctx.full_device_grid,
                    ncrisc_named_compile_time_args=ncrisc_args,
                    brisc_named_compile_time_args=brisc_args,
                    trisc_named_compile_time_args=trisc_args,
                    ncrisc_common_runtime_args=ncrisc_common_rt_args,
                    trisc_common_runtime_args=[
                        routed_ctx.rmsnorm_epsilon_packed,
                        routed_ctx.rmsnorm_scalar_packed,
                    ],
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                        math_approx_mode=False,
                        fp32_dest_acc_en=False,
                        dst_full_sync_en=False,
                    ),
                    unified_compile_time_core_descriptors=device_unified_ct_core_descs,
                    per_core_compile_time_descriptors=device_per_core_ct_descs,
                    per_core_runtime_args_descriptor=device_runtime_args_descriptor,
                    defines=kernel_defines,
                )

                kernel_result = unified_kernel.get_kernel_descriptors()

                # Add worker→fabric semaphores on fabric cores for reduce signaling
                if enable_reduce_to_one:
                    fabric_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in reduce_params["fabric_cores"]])
                    for worker_idx in range(reduce_params["num_workers_per_column"]):
                        sem_desc = ttnn.SemaphoreDescriptor(
                            id=reduce_worker_fabric_sem_base + worker_idx,
                            core_type=ttnn.CoreType.WORKER,
                            core_ranges=fabric_core_set,
                            initial_value=0,
                        )
                        device_semaphore_descriptors.append(sem_desc)

                # Create program
                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    cbs=device_cb_descriptors,
                    semaphores=device_semaphore_descriptors,
                )

                # Setup fabric connection for reduce fabric cores
                if enable_reduce_to_one and device_role != MESH_ROOT1:
                    fabric_node_id = mesh_device.get_fabric_node_id(coord)
                    num_columns = reduce_params["num_columns"]
                    for fc_idx, fc in enumerate(reduce_params["fabric_cores"]):
                        col_idx = fc_idx
                        link_idx = 0 if col_idx < num_columns // 2 else 1
                        # Find the kernel group containing this specific fabric core
                        fc_kernel_idx = None
                        for group in kernel_result.groups:
                            if group.compile_time_arg_values.get(
                                "is_reduce_fabric_core"
                            ) == 1 and group.core_range_set.contains(fc):
                                fc_kernel_idx = group.brisc_kernel_index
                                break
                        fabric_rt_args_ref = program.kernels[fc_kernel_idx].runtime_args[fc.x][fc.y]
                        fabric_conn_args = ttnn.setup_fabric_connection(
                            fabric_node_id,
                            dest_fabric_node_id,
                            link_idx,
                            program,
                            fc,
                        )
                        fabric_rt_args_ref.extend(fabric_conn_args)

                # Assign to mesh coordinate
                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Add reduce tensors to io_tensors if enabled
        if enable_reduce_to_one:
            io_tensors.extend(
                [
                    reduce_intermediate_tensors[0],
                    reduce_intermediate_tensors[1],
                    reduce_intermediate_tensors[2],
                    reduce_output_tensor,
                ]
            )

        # Execute
        ttnn.generic_op(io_tensors, mesh_program_descriptor)

        # Return appropriate output based on reduce mode
        if enable_reduce_to_one:
            return gate_output_scores_tensor, gate_output_indices_tensor, reduce_output_tensor
        return gate_output_scores_tensor, gate_output_indices_tensor, final_output_tensor
