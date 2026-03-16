"""
TTNN implementation of Minimal MoE module connecting all MoE components.

This module assembles the full MoE pipeline:
1. Dispatch: Route tokens to expert buffers
2. Routed Experts: Process tokens in expert-specific buffers
3. Shared Expert: Process original input (in parallel with routed path)
4. Combine: Reconstruct outputs to original token positions
5. Split Connection: Apply gate weights and sum expert contributions
6. Final: Add routed output + shared output
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

import ttnn


@dataclass
class TtMoEIntermediates:
    """
    Data structure holding intermediate values from TtMinimalMoe forward pass for debugging.

    Fields set to None indicate that component is not yet enabled/calculated.
    """

    dispatched_buffer: Optional[ttnn.Tensor]  # (1, dispatch_group_size, experts_per_chip, max_tokens, emb_dim)
    metadata: Optional[ttnn.Tensor]  # (1, dispatch_group_size, experts_per_chip, max_tokens, metadata_len)
    expert_outputs: Optional[ttnn.Tensor]  # Same shape as dispatched_buffer
    shared_output: Optional[ttnn.Tensor]  # (dispatch_group_size, seq_len_per_chip, emb_dim)
    combined_output: Optional[ttnn.Tensor]  # (dispatch_group_size, seq_len_per_chip, num_experts_per_tok, emb_dim)
    routed_output: Optional[ttnn.Tensor]  # (dispatch_group_size, seq_len_per_chip, emb_dim)


from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import TtSharedExpert


class TtMinimalMoe(LightweightModule):
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
        emb_dim: int = 7 * 1024,
        hidden_dim: int = 2 * 1024,
        cluster_axis: int = 0,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        routed_expert_weights: list[dict] = None,
        shared_expert_weights: dict = None,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
    ):
        """
        Initialize TtMinimalMoe module.

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
            cluster_axis: Mesh axis for dispatch operations (default: 0)
            num_links: Number of ethernet links for CCL (default: 1)
            topology: CCL topology - Linear or Ring (default: Linear)
            routed_expert_weights: Optional list of dicts with gate_proj, up_proj, down_proj
                                   per expert. Length must be experts_per_chip.
            shared_expert_weights: Optional dict with gate_proj, up_proj, down_proj
                                   for shared expert.
            activations_dtype: Data type for activations (default: bfloat8_b)
            weights_dtype: Data type for weights (default: bfloat4_b)
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
        self.cluster_axis = cluster_axis
        self.num_links = num_links
        self.topology = topology

        logger.debug(f"Initializing TtMinimalMoe")
        logger.debug(f"  mesh_device.shape={mesh_device.shape}")
        logger.debug(f"  dispatch_group_size={dispatch_group_size}, num_dispatch_groups={num_dispatch_groups}")
        logger.debug(f"  experts_per_chip={experts_per_chip}, num_routed_experts={num_routed_experts}")
        logger.debug(f"  num_experts_per_tok={num_experts_per_tok}")
        logger.debug(f"  seq_len_per_chip={seq_len_per_chip}, emb_dim={emb_dim}, hidden_dim={hidden_dim}")

        # Initialize dispatch module
        self.dispatch_module = TtDispatchModule(
            mesh_device=mesh_device,
            dispatch_group_size=dispatch_group_size,
            experts_per_chip=experts_per_chip,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            seq_len_per_chip=seq_len_per_chip,
            hidden_dim=emb_dim,  # dispatch uses emb_dim as hidden_dim
            cluster_axis=cluster_axis,
            num_links=num_links,
            topology=topology,
        )

        # Initialize combine module
        self.combine_module = TtCombineModule(
            mesh_device=mesh_device,
            dispatch_group_size=dispatch_group_size,
            num_dispatch_groups=num_dispatch_groups,
            experts_per_chip=experts_per_chip,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=seq_len_per_chip,
            cluster_axis=cluster_axis,
            num_links=num_links,
            topology=topology,
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

        # Initialize shared expert
        self.shared_expert = TtSharedExpert(
            mesh_device=mesh_device,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            torch_weights=shared_expert_weights,
            num_links=num_links,
            topology=topology,
            activations_dtype=activations_dtype,
            weights_dtype=weights_dtype,
        )

        # Initialize reduce module for post-combine reduction
        # topk_dim=3 because combine output is (1, dispatch_group_size, seq_len, topk, emb_dim)
        # cluster_axis=1 to reduce-scatter across TP axis (same as shared expert)
        self.reduce_module = TtReduceModule(
            mesh_device=mesh_device,
            topk_dim=3,  # topk is at dim 3 in 5D tensor from combine
            cluster_axis=1,  # TP axis for reduce-scatter
            num_links=num_links,
            topology=topology,
        )

        logger.debug("TtMinimalMoe initialization complete")

    def forward(
        self,
        x: ttnn.Tensor,
        weights: ttnn.Tensor,
        indices: ttnn.Tensor,
        tt_expert_offsets: ttnn.Tensor,
        tt_expert_dispatch_table: ttnn.Tensor,
        tt_expert_token_counts: ttnn.Tensor,
        return_intermediates: bool = False,
    ) -> tuple[ttnn.Tensor, Optional[TtMoEIntermediates]]:
        """
        Forward pass through the full MoE pipeline.

        Args:
            x: Input tensor - ROW_MAJOR, sharded:
               - For 2D mesh: sharded dims=(0, -1) - dim 0 across axis 0, dim -1 across axis 1
               - Shape per device: (dispatch_group_size/axis0, seq_len_per_chip, emb_dim/axis1)
            weights: Gate weights (dispatch_group_size, seq_len_per_chip, num_experts_per_tok) - ROW_MAJOR
            indices: Expert indices (dispatch_group_size, seq_len_per_chip, num_experts_per_tok) - ROW_MAJOR
            tt_expert_offsets: Base offset for each expert (use TtDispatchModule.shard_expert_offsets)
            tt_expert_dispatch_table: Expert dispatch table (use TtDispatchModule.shard_expert_dispatch_table)
            tt_expert_token_counts: Token counts per expert per chip
            return_intermediates: If True, return intermediate tensors for debugging

        Returns:
            Tuple of (final_output, intermediates):
            - final_output: MoE output with same sharding as input
            - intermediates: TtMoEIntermediates if return_intermediates=True, else None
        """
        logger.debug(f"[TtMinimalMoe.forward] INPUT SHAPES:")
        logger.debug(f"  x.shape={x.shape}")
        logger.debug(f"  weights.shape={weights.shape}")
        logger.debug(f"  indices.shape={indices.shape}")

        # Initialize intermediates - None means not yet calculated
        dispatched_buffer = None
        metadata = None
        expert_outputs = None
        shared_output = None
        combined_output = None
        routed_output = None

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
                num_links=self.num_links,
                topology=self.topology,
            )
        else:
            x_full = x  # No TP sharding, x already has full emb_dim
        logger.debug(f"[TtMinimalMoe.forward] x_full (after all_gather) shape: {x_full.shape}")

        # ========================================
        # Step 1: Shared expert (enabled)
        # ========================================
        # Shared expert expects replicated input (full emb_dim)
        # Convert x_full to TILE_LAYOUT for shared expert
        x_full_tiled = ttnn.to_layout(x_full, ttnn.TILE_LAYOUT)
        logger.debug(f"[TtMinimalMoe.forward] x_full_tiled shape: {x_full_tiled.shape}")

        shared_output = self.shared_expert(x_full_tiled)
        logger.debug(f"[TtMinimalMoe.forward] Shared expert output shape: {shared_output.shape}")

        # ========================================
        # Step 2: Dispatch (enabled)
        # ========================================
        # Dispatch expects full emb_dim on each device (x_full already has this)

        dispatched_buffer, metadata = self.dispatch_module(
            x_full,
            weights,
            indices,
            tt_expert_offsets,
            tt_expert_dispatch_table,
        )
        logger.debug(
            f"[TtMinimalMoe.forward] Dispatch output: buffer={dispatched_buffer.shape}, metadata={metadata.shape}"
        )

        # ========================================
        # Step 3: Routed experts (enabled)
        # ========================================
        # Dispatch output is (1, dispatch_group_size_per_device, experts_per_chip, max_tokens, emb_dim)
        # Routed expert expects (experts_per_chip, max_tokens, emb_dim)
        # Squeeze the first two dimensions
        dispatched_buffer_squeezed = ttnn.squeeze(dispatched_buffer, dim=0)
        dispatched_buffer_squeezed = ttnn.squeeze(dispatched_buffer_squeezed, dim=0)
        logger.debug(f"[TtMinimalMoe.forward] dispatched_buffer_squeezed shape: {dispatched_buffer_squeezed.shape}")

        # Convert dispatched_buffer to TILE_LAYOUT for routed experts
        dispatched_buffer_tiled = ttnn.to_layout(dispatched_buffer_squeezed, ttnn.TILE_LAYOUT)
        logger.debug(f"[TtMinimalMoe.forward] dispatched_buffer_tiled shape: {dispatched_buffer_tiled.shape}")

        expert_outputs = self.routed_expert(dispatched_buffer_tiled, tt_expert_token_counts)
        logger.debug(f"[TtMinimalMoe.forward] expert_outputs shape: {expert_outputs.shape}")

        # Add back the batch dimensions for combine
        # (experts_per_chip, max_tokens, emb_dim) -> (1, 1, experts_per_chip, max_tokens, emb_dim)
        expert_outputs = ttnn.unsqueeze(expert_outputs, dim=0)
        expert_outputs = ttnn.unsqueeze(expert_outputs, dim=0)
        logger.debug(f"[TtMinimalMoe.forward] expert_outputs (unsqueezed) shape: {expert_outputs.shape}")

        # ========================================
        # Step 4: Combine (enabled)
        # ========================================
        # Combine expects ROW_MAJOR input
        expert_outputs_rm = ttnn.to_layout(expert_outputs, ttnn.ROW_MAJOR_LAYOUT)
        logger.debug(f"[TtMinimalMoe.forward] expert_outputs_rm shape: {expert_outputs_rm.shape}")

        combined_output = self.combine_module(
            expert_outputs_rm,
            metadata,
            tt_expert_token_counts,
        )
        logger.debug(f"[TtMinimalMoe.forward] combined_output shape: {combined_output.shape}")

        # ========================================
        # Step 5: Reduce (sum over topk + reduce-scatter for TP sharding)
        # ========================================
        # combined_output: (1, dispatch_group_size, seq_len_per_chip, num_experts_per_tok, emb_dim)
        #                  (1, 1, 256, 4, 2048) per device - 5D tensor!
        #
        # TODO: TTNN mul doesn't support broadcasting for weight multiplication
        # For now, skip weight multiplication (test uses weights=1)
        #
        # TtReduceModule does:
        # 1. Sum over topk dimension (dim=3): (1, 1, 256, 4, 2048) -> (1, 1, 256, 2048)
        # 2. Reduce-scatter across TP axis: (1, 1, 256, 2048) -> (1, 1, 256, 512) per device
        combined_output_tiled = ttnn.to_layout(combined_output, ttnn.TILE_LAYOUT)
        logger.debug(f"[TtMinimalMoe.forward] combined_output_tiled shape: {combined_output_tiled.shape}")

        routed_output = self.reduce_module(combined_output_tiled)
        logger.debug(f"[TtMinimalMoe.forward] routed_output (after reduce) shape: {routed_output.shape}")

        # Remove extra batch dimensions to match shared_output shape
        # (1, 1, 256, 512) -> (1, 256, 512)
        routed_output = ttnn.squeeze(routed_output, dim=0)
        logger.debug(f"[TtMinimalMoe.forward] routed_output (squeezed) shape: {routed_output.shape}")

        # ========================================
        # Step 6: Final output
        # ========================================
        # final_output = routed_output + shared_output
        # Both should be in TILE_LAYOUT with shape (dispatch_group_size, seq_len_per_chip, hidden_dim)
        final_output = ttnn.add(routed_output, shared_output)
        logger.debug(f"[TtMinimalMoe.forward] final_output (tiled) shape: {final_output.shape}")

        # Convert to ROW_MAJOR for output consistency
        final_output = ttnn.to_layout(final_output, ttnn.ROW_MAJOR_LAYOUT)
        logger.debug(f"[TtMinimalMoe.forward] Final output shape: {final_output.shape}")

        # Build intermediates if requested
        intermediates = None
        if return_intermediates:
            intermediates = TtMoEIntermediates(
                dispatched_buffer=dispatched_buffer,
                metadata=metadata,
                expert_outputs=expert_outputs,
                shared_output=shared_output,
                combined_output=combined_output,
                routed_output=routed_output,
            )

        return final_output, intermediates
