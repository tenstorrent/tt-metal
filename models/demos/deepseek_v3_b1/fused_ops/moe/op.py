# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
MoE Routed Expert operation (refactored).

Follows the SharedExpertOp pattern with:
  _MoeRoutedExpertContext dataclass → _setup_dimensions → _build_compile_time_args
  → _build_cb_descriptors → _build_core_descriptors → op()

Pipeline:
  1. Input mcast: [1, K] from sender core → all cores in mcast grid
  2. Gate matmul: [1, K] x [K, N_routing] → [1, N_routing] with sigmoid (8 compute cores)
  3. Gate gather: [1, N_per_core] from 8 cores → [16, 16] on sender core
  4. Gate: top-8 expert selection with normalized scores
  5. Index mcast + Scale mcast: expert index and scale → compute cores
  6. gate_proj: DRAM streaming matmul + SiLU with indexed expert weights
  7. up_proj: DRAM streaming matmul with indexed expert weights
  8. Fused mul: silu(gate_proj) * up_proj * expert_scale
  9. down_proj gather: fused output → sender core
  10. down_proj mcast: fused output → compute cores
  11. down_proj: DRAM streaming matmul with indexed expert weights
  12. Eltwise add: down_proj + fused_add → final output
"""

from dataclasses import dataclass
from typing import Any

import ttnn
from models.demos.deepseek_v3_b1.fused_ops.moe_routed_expert.op import (
    setup_dram_matmul,
    setup_eltwise_add,
    setup_eltwise_mul,
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
        up_proj_cb_in1 = 12
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
            core_ranges=gate_proj_core_ranges,
            cb_in0_index=mul_cb_in0,
            cb_in1_index=mul_cb_in1,
            cb_out_index=mul_cb_out,
            per_core_n=gate_proj_params["per_core_n"],
            cb_scalar_index=mul_cb_scalar,
            cb_scalar_src_index=mul_cb_scalar_src,
            scalar_src_tensor=expert_scale_tensor,
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
        """Build NCRISC, BRISC, and TRISC compile-time arg lists."""

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
        ]

        return ncrisc_named_compile_time_args, brisc_named_compile_time_args, trisc_named_compile_time_args

    @staticmethod
    def _build_cb_descriptors(ctx):
        """Build all circular buffer descriptors."""
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
            ctx.gate_proj_params["cb_in1_descriptor"],
            ctx.gate_proj_cb_index_descriptor,
            ctx.gate_proj_params["cb_out_descriptor"],
            ctx.up_proj_params["cb_in1_descriptor"],
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
        """Build unified and per-core compile-time core descriptors."""
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
        use_hardcoded_expert_index=False,
    ):
        """
        Execute the full MoE routed expert fused operation.

        Args:
            input_tensor: [1, K] sharded on sender core
            mcast_output_tensor: [1, K] sharded on mcast grid
            gate_mm_weights_tensor: [K, N_routing] width-sharded on matmul cores
            gate_mm_output_tensor: [1, N_routing] width-sharded on matmul cores
            gate_input_tensor: [16, 16] on sender core
            gate_bias_tensor: [16, 16] on sender core
            gate_indices_tensor: [16, 16] on sender core
            gate_output_scores_tensor: [1, 16] on sender core
            gate_output_indices_tensor: [1, 16] on sender core
            expert_index_tensor: [1, 16] on mcast grid
            expert_scale_tensor: [1, 16] on mcast grid
            gate_proj_weights_tensor: Expert weights in DRAM
            gate_proj_output_tensor: gate_proj output
            up_proj_weights_tensor: Expert weights in DRAM
            up_proj_mm_out_tensor: up_proj intermediate output
            fused_output_tensor: silu(gate_proj) * up_proj * scale
            down_proj_gather_output_tensor: Gathered fused output on sender
            down_proj_mcast_output_tensor: Mcasted fused output on compute cores
            down_proj_weights_tensor: Expert weights in DRAM
            down_proj_output_tensor: down_proj output
            fused_add_tensor: Tensor to add after down_proj
            final_output_tensor: Final output (down_proj + fused_add)
            use_hardcoded_expert_index: For testing, always use expert index 0

        Returns:
            (gate_output_scores_tensor, gate_output_indices_tensor, final_output_tensor)
        """
        # ==================================================================
        # Setup all dimensions and parameters
        # ==================================================================
        ctx = MoeRoutedExpertOp._setup_dimensions(
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
            use_hardcoded_expert_index,
        )

        # ==================================================================
        # Build CB descriptors and core descriptors (shared across devices)
        # ==================================================================
        cb_descriptors = MoeRoutedExpertOp._build_cb_descriptors(ctx)
        unified_core_descs, per_core_descs = MoeRoutedExpertOp._build_core_descriptors(ctx)

        # ==================================================================
        # Semaphore descriptors
        # ==================================================================
        semaphore_descriptors = [
            ttnn.SemaphoreDescriptor(
                id=ctx.mcast_data_sender_semaphore_id,
                core_ranges=ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=ctx.mcast_data_receiver_semaphore_id,
                core_ranges=ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=ctx.gather_noc0_receiver_semaphore_id,
                core_ranges=ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=ctx.gather_noc1_receiver_semaphore_id,
                core_ranges=ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=ctx.expert_scale_mcast_sender_semaphore_id,
                core_ranges=ctx.full_device_grid,
                initial_value=1,  # Sender starts as VALID
            ),
            ttnn.SemaphoreDescriptor(
                id=ctx.expert_scale_mcast_receiver_semaphore_id,
                core_ranges=ctx.full_device_grid,
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
        ]

        # ==================================================================
        # Create per-device programs (mesh loop)
        # ==================================================================
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        for row in range(ctx.mesh_rows):
            for col in range(ctx.mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                chip_id = row * ctx.mesh_cols + col

                # Build compile-time args for this chip
                ncrisc_args, brisc_args, trisc_args = MoeRoutedExpertOp._build_compile_time_args(ctx, chip_id)

                # Create unified kernel
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/moe_routed_expert/moe_routed_expert_kernel.cpp",
                    core_ranges=ctx.full_device_grid,
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
