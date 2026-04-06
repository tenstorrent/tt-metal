# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of MoE module connecting all MoE components.

This module assembles the full MoE pipeline:
1. Dispatch: Route tokens to expert buffers
2. Routed Experts: Process tokens in expert-specific buffers
3. Shared Expert: Process original input (in parallel with routed path)
4. Combine: Reconstruct outputs to original token positions
5. Split Connection: Apply gate weights and sum expert contributions
6. Final: Add routed output + shared output
"""

from typing import Optional, Union

from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, TtMoEGateConfig, TtMoEGatePrefill
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_intermediates import TtMoEIntermediates
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import TtSharedExpert
from models.demos.deepseek_v3_d_p.utils.test_utils import get_input_mem_config


class TtMoe(LightweightModule):
    """
    TTNN implementation of complete MoE pipeline.

    Architecture:
        x → [Dispatch] → dispatched_buffer → [Routed Experts] → expert_outputs
                                                                      ↓
                                                               [Combine] → combined_output
                                                                      ↓
        x → [Shared Expert] → shared_output           [Split Connection] → routed_output
                                      ↓                        ↓
                                final = routed_output + shared_output

    Layout Flow:
        - Dispatch: ROW_MAJOR → ROW_MAJOR
        - Routed Expert: TILE_LAYOUT → TILE_LAYOUT (convert before/after)
        - Combine: ROW_MAJOR → ROW_MAJOR
        - Shared Expert: TILE_LAYOUT → TILE_LAYOUT
        - Split Connection: ROW_MAJOR (elementwise ops)
        - Final Add: ROW_MAJOR
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dispatch_group_size: int,
        num_dispatch_groups: int,
        experts_per_chip: int,
        num_routed_experts: int,
        num_experts_per_tok: int,
        metadata_len: int,
        max_dispatched_tokens_per_expert: int,
        seq_len_per_chip: int,
        gate_weights: dict,
        emb_dim: int = DeepSeekV3Config.EMB_SIZE,
        hidden_dim: int = DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
        num_links: Union[int, tuple[int, int]] = 1,
        topology: Union[ttnn.Topology, tuple[ttnn.Topology, ttnn.Topology]] = ttnn.Topology.Linear,
        routed_expert_weights: list[dict] = None,
        shared_expert_weights: dict = None,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        gate_fallback_mode: GateComputeMode = GateComputeMode.HOST_ALL,
    ):
        """
        Initialize TtMoe module.

        Args:
            mesh_device: TTNN mesh device
            dispatch_group_size: Number of chips in each dispatch group
            num_dispatch_groups: Number of parallel dispatch groups
            experts_per_chip: Number of experts per chip
            num_routed_experts: Total number of routed experts
            num_experts_per_tok: Number of experts each token routes to
            metadata_len: Length of metadata per token
            max_dispatched_tokens_per_expert: Max tokens per expert buffer
            seq_len_per_chip: Sequence length per chip
            emb_dim: Embedding dimension (default: 7168)
            hidden_dim: Hidden/intermediate dimension (default: 2048)
            num_links: Number of ethernet links for CCL. Int applies to both axes;
                       tuple (row, col) allows separate config per axis.
            topology: CCL topology. Scalar applies to both axes;
                      tuple (row, col) allows separate config per axis.
            routed_expert_weights: Optional list of dicts with gate_proj, up_proj, down_proj
                                   per expert. Length must be experts_per_chip.
            shared_expert_weights: Optional dict with gate_proj, up_proj, down_proj
                                   for shared expert.
            activations_dtype: Data type for activations (default: bfloat8_b)
            weights_dtype: Data type for weights (default: bfloat4_b)
            gate_weights: Dict with "weight" and "e_score_correction_bias" keys for gate
            gate_fallback_mode: Fallback mode for gate (default: HOST_ALL)
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.dispatch_group_size = dispatch_group_size
        self.num_dispatch_groups = num_dispatch_groups
        self.experts_per_chip = experts_per_chip
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.seq_len_per_chip = seq_len_per_chip
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # Unpack row/col CCL config
        if isinstance(num_links, tuple):
            self.row_num_links, self.col_num_links = num_links
        else:
            self.row_num_links = self.col_num_links = num_links

        if isinstance(topology, tuple):
            self.row_topology, self.col_topology = topology
        else:
            self.row_topology = self.col_topology = topology

        # Always create dispatch table at init (static tensor) - needed by gate and dispatch module
        expert_dispatch_table = ExpertMapping.create_dispatch_table(
            num_routed_experts, dispatch_group_size, num_dispatch_groups
        )

        # Build gate internally
        gate_config = TtMoEGateConfig()
        gate_config.dim = emb_dim
        gate_config.sp_dim = seq_len_per_chip
        gate_config.n_routed_experts = num_routed_experts
        gate_config.n_activated_experts = num_experts_per_tok
        gate_config.ccl_config["NUM_LINKS"] = self.col_num_links if isinstance(num_links, tuple) else num_links

        self.gate = TtMoEGatePrefill(
            gate_config,
            mesh_device,
            dispatch_table=expert_dispatch_table,
            weight=gate_weights["weight"],
            bias=gate_weights["e_score_correction_bias"],
            fallback_mode=gate_fallback_mode,
        )
        self.gate_input_mem_config = get_input_mem_config(gate_config, mesh_device.shape)

        logger.debug(f"Initializing TtMoe")
        logger.debug(f"  mesh_device.shape={mesh_device.shape}")
        logger.debug(f"  dispatch_group_size={dispatch_group_size}, num_dispatch_groups={num_dispatch_groups}")
        logger.debug(f"  experts_per_chip={experts_per_chip}, num_routed_experts={num_routed_experts}")
        logger.debug(f"  num_experts_per_tok={num_experts_per_tok}")
        logger.debug(f"  seq_len_per_chip={seq_len_per_chip}, emb_dim={emb_dim}, hidden_dim={hidden_dim}")

        self.tt_expert_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(
            mesh_device, expert_dispatch_table, dispatch_axis=0
        )

        # Initialize dispatch module (row axis: axis 0)
        self.dispatch_module = TtDispatchModule(
            mesh_device=mesh_device,
            dispatch_group_size=dispatch_group_size,
            experts_per_chip=experts_per_chip,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=emb_dim,
            cluster_axis=0,
            num_links=self.row_num_links,
            topology=self.row_topology,
        )

        # Initialize combine module (row axis: axis 0)
        self.combine_module = TtCombineModule(
            mesh_device=mesh_device,
            dispatch_group_size=dispatch_group_size,
            num_dispatch_groups=num_dispatch_groups,
            experts_per_chip=experts_per_chip,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=seq_len_per_chip,
            cluster_axis=0,
            num_links=self.row_num_links,
            topology=self.row_topology,
            init_zeros=True,
        )

        # Initialize routed expert
        self.routed_expert = TtRoutedExpert(
            mesh_device=mesh_device,
            experts_per_chip=experts_per_chip,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            max_tokens=max_dispatched_tokens_per_expert,
            torch_weights=routed_expert_weights,
            activations_dtype=activations_dtype,
            weights_dtype=weights_dtype,
        )

        # Initialize shared expert (col axis: axis 1)
        self.shared_expert = TtSharedExpert(
            mesh_device=mesh_device,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            torch_weights=shared_expert_weights,
            num_links=self.col_num_links,
            topology=self.col_topology,
            activations_dtype=activations_dtype,
            weights_dtype=weights_dtype,
        )

        # Initialize reduce module for post-combine reduction (col axis: axis 1)
        # topk_dim=3 because combine output is (1, dispatch_group_size, seq_len, topk, emb_dim)
        # cluster_axis=1 to reduce-scatter across TP axis (same as shared expert)
        self.reduce_module = TtReduceModule(
            mesh_device=mesh_device,
            topk_dim=3,  # topk is at dim 3 in 5D tensor from combine
            cluster_axis=1,  # TP axis for reduce-scatter
            num_links=self.col_num_links,
            topology=self.col_topology,
        )

        logger.debug("TtMoe initialization complete")

    def forward(
        self,
        x: ttnn.Tensor,
        return_intermediates: bool = False,
    ) -> tuple[ttnn.Tensor, Optional[TtMoEIntermediates]]:
        """
        Forward pass through the full MoE pipeline.

        Args:
            x: Input tensor - ROW_MAJOR, sharded:
               - For 2D mesh: sharded dims=(0, -1) - dim 0 across axis 0, dim -1 across axis 1
               - Shape per device: (dispatch_group_size/axis0, seq_len_per_chip, emb_dim/axis1)
            return_intermediates: If True, return intermediate tensors for debugging

        Returns:
            Tuple of (final_output, intermediates):
            - final_output: MoE output with same sharding as input
            - intermediates: TtMoEIntermediates if return_intermediates=True, else None
        """
        logger.debug(f"[TtMoe.forward] INPUT SHAPES:")
        logger.debug(f"  x.shape={x.shape}")

        # ========================================
        # Gate: compute weights/indices/offsets/counts from x
        # ========================================
        # Reshape 3D -> 2D for gate: (batch, seq, emb) -> (batch*seq, emb)
        x_for_gate = ttnn.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
        x_for_gate = ttnn.to_layout(x_for_gate, ttnn.TILE_LAYOUT)
        if self.gate_input_mem_config is not None:
            x_for_gate = ttnn.to_memory_config(x_for_gate, self.gate_input_mem_config)

        scores, indices_raw, gate_logits, tt_expert_offsets, tt_expert_token_counts = self.gate(x_for_gate)

        # Gate outputs uint16 indices; dispatch requires int32.
        # this should be aligned in the further PR.
        # Typecast in TILE_LAYOUT to avoid alignment issues, then convert to ROW_MAJOR.
        if indices_raw.dtype != ttnn.int32:
            indices_raw = ttnn.to_layout(indices_raw, ttnn.TILE_LAYOUT)
            indices_raw = ttnn.typecast(indices_raw, ttnn.int32)
            indices_raw = ttnn.to_layout(indices_raw, ttnn.ROW_MAJOR_LAYOUT)
        else:
            indices_raw = ttnn.to_layout(indices_raw, ttnn.ROW_MAJOR_LAYOUT)
        #
        # Ensure ROW_MAJOR layout for dispatch compatibility
        scores = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT)

        # Reshape back to 3D: (batch*seq, topk) -> (batch, seq, topk)
        seq_dim = x.shape[1]
        batch_dim = x.shape[0]
        weights = ttnn.reshape(scores, (batch_dim, seq_dim, scores.shape[-1]))
        indices = ttnn.reshape(indices_raw, (batch_dim, seq_dim, indices_raw.shape[-1]))

        logger.debug(f"  weights.shape={weights.shape}")
        logger.debug(f"  indices.shape={indices.shape}")

        # ========================================
        # Step 0: All-gather x to get full emb_dim (replicated across TP axis)
        # ========================================
        # Input x is sharded: (dispatch_group_size/axis0, seq_len_per_chip, emb_dim/axis1)
        # Both shared_expert and dispatch need full emb_dim, so all-gather first
        # Only needed if there are multiple devices in TP axis (axis 1)
        if self.mesh_device.shape[1] > 1:
            x_full = ttnn.all_gather(
                x,
                dim=-1,  # Gather along emb_dim
                cluster_axis=1,  # Gather across axis 1 (TP axis)
                num_links=self.col_num_links,
                topology=self.col_topology,
            )
        else:
            x_full = x  # No TP sharding, x already has full emb_dim
        logger.debug(f"[TtMoe.forward] x_full (after all_gather) shape: {x_full.shape}")

        # ========================================
        # Step 1: Shared expert (enabled)
        # ========================================
        # Shared expert expects replicated input (full emb_dim)
        # Convert x_full to TILE_LAYOUT for shared expert
        x_full_tiled = ttnn.to_layout(x_full, ttnn.TILE_LAYOUT)
        logger.debug(f"[TtMoe.forward] x_full_tiled shape: {x_full_tiled.shape}")

        shared_output = self.shared_expert(x_full_tiled)
        logger.debug(f"[TtMoe.forward] Shared expert output shape: {shared_output.shape}")

        # ========================================
        # Step 2: Dispatch (enabled)
        # ========================================
        # Dispatch expects full emb_dim on each device (x_full already has this)

        dispatched_buffer, metadata = self.dispatch_module(
            x_full,
            weights,
            indices,
            tt_expert_offsets,
            self.tt_expert_dispatch_table,
        )
        logger.debug(f"[TtMoe.forward] Dispatch output: buffer={dispatched_buffer.shape}, metadata={metadata.shape}")

        # ========================================
        # Step 3: Routed experts (enabled)
        # ========================================
        # Dispatch output is (1, dispatch_group_size_per_device, experts_per_chip, max_tokens, emb_dim)
        # Routed expert expects (experts_per_chip, max_tokens, emb_dim)
        # Squeeze the first two dimensions
        dispatched_buffer_squeezed = ttnn.squeeze(dispatched_buffer, dim=0)
        dispatched_buffer_squeezed = ttnn.squeeze(dispatched_buffer_squeezed, dim=0)
        logger.debug(f"[TtMoe.forward] dispatched_buffer_squeezed shape: {dispatched_buffer_squeezed.shape}")

        # Convert dispatched_buffer to TILE_LAYOUT for routed experts
        dispatched_buffer_tiled = ttnn.to_layout(dispatched_buffer_squeezed, ttnn.TILE_LAYOUT)
        logger.debug(f"[TtMoe.forward] dispatched_buffer_tiled shape: {dispatched_buffer_tiled.shape}")

        expert_outputs = self.routed_expert(dispatched_buffer_tiled, tt_expert_token_counts)
        logger.debug(f"[TtMoe.forward] expert_outputs shape: {expert_outputs.shape}")

        # Add back the batch dimensions for combine
        # (experts_per_chip, max_tokens, emb_dim) -> (1, 1, experts_per_chip, max_tokens, emb_dim)
        expert_outputs = ttnn.unsqueeze(expert_outputs, dim=0)
        expert_outputs = ttnn.unsqueeze(expert_outputs, dim=0)
        logger.debug(f"[TtMoe.forward] expert_outputs (unsqueezed) shape: {expert_outputs.shape}")

        # ========================================
        # Step 4: Combine (enabled)
        # ========================================
        # Combine expects ROW_MAJOR input
        expert_outputs_rm = ttnn.to_layout(expert_outputs, ttnn.ROW_MAJOR_LAYOUT)
        logger.debug(f"[TtMoe.forward] expert_outputs_rm shape: {expert_outputs_rm.shape} {expert_outputs_rm.dtype=}")

        combined_output = self.combine_module(
            expert_outputs_rm,
            metadata,
            tt_expert_token_counts,
        )
        logger.debug(f"[TtMoe.forward] combined_output shape: {combined_output.shape}")

        # ========================================
        # Step 5: Reduce (weighted sum over topk + reduce-scatter for TP sharding)
        # ========================================
        # combined_output: (1, dispatch_group_size, seq_len_per_chip, num_experts_per_tok, emb_dim)
        #                  (1, 1, 256, 4, 2048) per device - 5D tensor!
        #
        # TtReduceModule does:
        # 1. Apply weights: weights * combined_output (broadcast multiply)
        # 2. Sum over topk dimension (dim=3): (1, 1, 256, 4, 2048) -> (1, 1, 256, 2048)
        # 3. Reduce-scatter across TP axis: (1, 1, 256, 2048) -> (1, 1, 256, 512) per device
        # combined_output_tiled is too big to fit L1; keep in DRAM for now
        combined_output_tiled = ttnn.to_layout(combined_output, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(f"[TtMoe.forward] combined_output_tiled shape: {combined_output_tiled.shape}")

        routed_output = self.reduce_module(combined_output_tiled, weights=weights)
        logger.debug(f"[TtMoe.forward] routed_output (after reduce) shape: {routed_output.shape}")

        # Remove extra batch dimensions to match shared_output shape
        # (1, 1, 256, 512) -> (1, 256, 512)
        routed_output = ttnn.squeeze(routed_output, dim=0)
        logger.debug(f"[TtMoe.forward] routed_output (squeezed) shape: {routed_output.shape}")

        # ========================================
        # Step 6: Final output
        # ========================================
        # final_output = routed_output + shared_output
        # Both should be in TILE_LAYOUT with shape (dispatch_group_size, seq_len_per_chip, emb_dim)
        final_output = ttnn.add(routed_output, shared_output)
        logger.debug(f"[TtMoe.forward] final_output (tiled) shape: {final_output.shape}")

        # Convert to ROW_MAJOR for output consistency
        final_output = ttnn.to_layout(final_output, ttnn.ROW_MAJOR_LAYOUT)
        logger.debug(f"[TtMoe.forward] Final output shape: {final_output.shape}")

        # Build intermediates if requested
        intermediates = None
        if return_intermediates:
            intermediates = TtMoEIntermediates(
                gate_scores=weights,
                gate_indices=indices,
                gate_logits=gate_logits,
                dispatched_buffer=dispatched_buffer,
                metadata=metadata,
                expert_outputs=expert_outputs,
                shared_output=shared_output,
                combined_output=combined_output,
                routed_output=routed_output,
                expert_token_counts=tt_expert_token_counts,
            )

        return final_output, intermediates
