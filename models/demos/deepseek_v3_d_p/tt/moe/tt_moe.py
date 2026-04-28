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

from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, TtMoEGateConfig, TtMoEGatePrefill
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_intermediates import TtMoEIntermediates
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import TtSharedExpert


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

    @staticmethod
    def check_cache_complete(cache_path: Path, layer_idx: int, experts_per_chip: int) -> bool:
        """Check if MoE cache is complete (gate + routed experts + shared expert)."""
        prefix = f"layer_{layer_idx}"
        if not TtMoEGatePrefill.check_cache_complete(cache_path, f"{prefix}.gate"):
            return False
        if not TtRoutedExpert.check_cache_complete(cache_path, f"{prefix}.routed_expert", experts_per_chip):
            return False
        if not TtSharedExpert.check_cache_complete(cache_path, f"{prefix}.shared_expert"):
            return False
        return True

    @staticmethod
    def build_ttnn_cache(
        gate_weights: dict | None,
        routed_expert_weights: list[dict] | None,
        shared_expert_weights: dict | None,
        experts_per_chip: int,
        emb_dim: int,
        hidden_dim: int,
        mesh_device: ttnn.MeshDevice,
        routed_expert_weights_dtype: ttnn.DataType,
        shared_expert_weights_dtype: ttnn.DataType,
        cache_path: Path,
        layer_idx: int,
    ):
        """Build TTNN cache for MoE (gate + routed experts + shared expert) without device copy."""
        # Build gate cache (delegate to TtMoEGatePrefill)
        if gate_weights:
            from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import TtMoEGateConfig, TtMoEGatePrefill

            # Create minimal config for caching
            gate_config = TtMoEGateConfig()
            gate_config.dim = emb_dim
            gate_config.n_routed_experts = gate_weights["weight"].shape[0]

            TtMoEGatePrefill.build_ttnn_cache(
                torch_weight=gate_weights["weight"],
                torch_bias=gate_weights["e_score_correction_bias"],
                config=gate_config,
                mesh_device=mesh_device,
                cache_path=cache_path,
                cache_name_prefix=f"layer_{layer_idx}.gate",
            )

        # Build routed expert cache
        if routed_expert_weights:
            TtRoutedExpert.build_ttnn_cache(
                routed_expert_weights,
                experts_per_chip,
                mesh_device,
                routed_expert_weights_dtype,
                cache_path,
                f"layer_{layer_idx}.routed_expert",
            )

        # Build shared expert cache
        if shared_expert_weights:
            TtSharedExpert.build_ttnn_cache(
                shared_expert_weights,
                emb_dim,
                hidden_dim,
                mesh_device,
                shared_expert_weights_dtype,
                cache_path,
                f"layer_{layer_idx}.shared_expert",
            )

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
        max_dispatch_buffer_token_size: int,
        seq_len_per_chip: int,
        gate_weights: dict,
        emb_dim: int = DeepSeekV3Config.EMB_SIZE,
        hidden_dim: int = DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
        num_links: Union[int, tuple[int, int]] = 1,
        topology: Union[ttnn.Topology, tuple[ttnn.Topology, ttnn.Topology]] = ttnn.Topology.Linear,
        routed_expert_weights: list[dict] = None,
        shared_expert_weights: dict = None,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        gate_fallback_mode: GateComputeMode = GateComputeMode.HOST_ALL,
        weight_cache_path: Optional[Path] = None,
        layer_idx: int = 0,
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
            max_dispatched_tokens_per_expert: Per-expert theoretical upper bound on the
                number of tokens any single expert may receive (full sequence length).
            max_dispatch_buffer_token_size: Total token capacity of the flat dispatch
                buffer per chip (shared across all local experts).
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
            routed_expert_activations_dtype: Data type for routed expert activations
            routed_expert_weights_dtype: Data type for routed expert weights
            shared_expert_activations_dtype: Data type for shared expert activations
            shared_expert_weights_dtype: Data type for shared expert weights
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

        # Handle cache-only case (gate_weights=None)
        if gate_weights is not None:
            gate_weight = gate_weights["weight"]
            gate_bias = gate_weights["e_score_correction_bias"]
        else:
            # Dummy tensors for cache load (ignored when cache exists)
            gate_weight = torch.empty(num_routed_experts, emb_dim)
            gate_bias = torch.empty(num_routed_experts)

        self.gate = TtMoEGatePrefill(
            gate_config,
            mesh_device,
            dispatch_table=expert_dispatch_table,
            experts_per_chip=experts_per_chip,
            weight=gate_weight,
            bias=gate_bias,
            fallback_mode=gate_fallback_mode,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.gate",
        )
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
            max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
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

        # Build (group, chip, local_expert) -> global expert id table, sharded
        # across the EP mesh so each device holds (1, 1, experts_per_chip).
        # Then squeeze the two leading singleton dims so each device has a 1D
        # (experts_per_chip,) lookup vector (required by extract/insert validators).
        global_expert_idx_tt = ttnn.from_torch(
            ExpertMapping.create_global_expert_idx_table(
                experts_per_chip=experts_per_chip,
                dispatch_group_size=dispatch_group_size,
                num_dispatch_groups=num_dispatch_groups,
            ),
            mesh_mapper=get_ep_mesh_mapper(mesh_device),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )
        global_expert_idx_tt = ttnn.squeeze(global_expert_idx_tt, 0)
        global_expert_idx_tt = ttnn.squeeze(global_expert_idx_tt, 0)

        # Initialize routed expert
        self.routed_expert = TtRoutedExpert(
            mesh_device=mesh_device,
            experts_per_chip=experts_per_chip,
            global_expert_idx_table=global_expert_idx_tt,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            max_tokens=max_dispatched_tokens_per_expert,
            torch_weights=routed_expert_weights,
            activations_dtype=routed_expert_activations_dtype,
            weights_dtype=routed_expert_weights_dtype,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.routed_expert",
        )

        # Initialize shared expert (col axis: axis 1)
        self.shared_expert = TtSharedExpert(
            mesh_device=mesh_device,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            torch_weights=shared_expert_weights,
            num_links=self.col_num_links,
            topology=self.col_topology,
            activations_dtype=shared_expert_activations_dtype,
            weights_dtype=shared_expert_weights_dtype,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.shared_expert",
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

        scores, indices, gate_logits, tt_expert_offsets, tt_expert_token_counts, tt_expert_region_offsets = self.gate(
            ttnn.view(x, (x.shape[0] * x.shape[1], x.shape[2]))
        )
        gate_logits = (
            ttnn.to_memory_config(gate_logits, ttnn.DRAM_MEMORY_CONFIG)
            if return_intermediates
            else ttnn.deallocate(gate_logits)
        )  # gate_logits is only used for debugging/intermediates, move to DRAM or deallocate immediately

        # DEBUG
        # Print full token counts per expert for monitoring
        _counts_4d = ttnn.unsqueeze_to_4D(tt_expert_token_counts)
        _ep_composer = ttnn.create_mesh_composer(self.mesh_device, ttnn.MeshComposerConfig(dims=[1, 0]))
        _counts_host = ttnn.to_torch(_counts_4d, mesh_composer=_ep_composer).squeeze(2)
        logger.info(f"[TtMoe.forward] expert_token_counts: {_counts_host.flatten().tolist()}")

        # DEBUG
        # Print full region offsets per expert for monitoring
        _offsets_4d = ttnn.unsqueeze_to_4D(tt_expert_region_offsets)
        _offsets_host = ttnn.to_torch(_offsets_4d, mesh_composer=_ep_composer).squeeze(2)
        logger.info(f"[TtMoe.forward] expert_region_offsets: {_offsets_host.flatten().tolist()}")

        # Gate outputs uint16 indices; dispatch requires int32.
        # this should be aligned in the further PR.
        # Typecast in TILE_LAYOUT to avoid alignment issues, then convert to ROW_MAJOR.
        if indices.dtype != ttnn.int32:
            indices = ttnn.to_layout(indices, ttnn.TILE_LAYOUT)
            indices = ttnn.typecast(indices, ttnn.int32)
            indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
        else:
            indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
        #
        # Ensure ROW_MAJOR layout for dispatch compatibility
        scores = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT)

        # Reshape back to 3D: (batch*seq, topk) -> (batch, seq, topk)
        seq_dim = x.shape[1]
        batch_dim = x.shape[0]
        scores = ttnn.reshape(scores, (batch_dim, seq_dim, scores.shape[-1]))
        indices = ttnn.reshape(indices, (batch_dim, seq_dim, indices.shape[-1]))

        logger.debug(f"  {scores.shape=} {scores.memory_config()=}")
        logger.debug(f"  {indices.shape=} {indices.memory_config()=}")

        # ========================================
        # Step 0: All-gather x to get full emb_dim (replicated across TP axis)
        # ========================================
        # Input x is sharded: (dispatch_group_size/axis0, seq_len_per_chip, emb_dim/axis1)
        # Both shared_expert and dispatch need full emb_dim, so all-gather first
        # Only needed if there are multiple devices in TP axis (axis 1)
        if self.mesh_device.shape[1] > 1:
            x = ttnn.all_gather(
                x,
                dim=-1,  # Gather along emb_dim
                cluster_axis=1,  # Gather across axis 1 (TP axis)
                num_links=self.col_num_links,
                topology=self.col_topology,
            )
        logger.debug(f"[TtMoe.forward] x (after all_gather) shape: {x.shape}")

        # ========================================
        # Step 1: Shared expert (enabled)
        # ========================================
        # Shared expert expects replicated input (full emb_dim)
        # Convert x to TILE_LAYOUT for shared expert
        logger.debug(f"[TtMoe.forward] {x.shape=} {x.memory_config()=}")

        shared_output = self.shared_expert(x)
        logger.debug(f"[TtMoe.forward] Shared expert output shape: {shared_output.shape}")

        # ========================================
        # Step 2: Dispatch (enabled)
        # ========================================
        # Dispatch expects full emb_dim on each device (x already has this)
        logger.debug(f"[TtMoe.forward] {x.shape=} {x.memory_config()=}")
        dispatched_buffer, metadata = self.dispatch_module(
            x,
            scores,
            indices,
            tt_expert_offsets,
            self.tt_expert_dispatch_table,
        )
        x = ttnn.deallocate(x)
        scores = ttnn.to_memory_config(scores, ttnn.DRAM_MEMORY_CONFIG)
        indices = ttnn.to_memory_config(indices, ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(f"[TtMoe.forward] Dispatch output: buffer={dispatched_buffer.shape}, metadata={metadata.shape}")

        # ========================================
        # Step 3: Routed experts (enabled)
        # ========================================
        # Dispatch output is (1, dispatch_group_size_per_device, experts_per_chip, max_tokens, emb_dim)
        # Routed expert expects (experts_per_chip, max_tokens, emb_dim)
        # Squeeze the first two dimensions

        # Convert dispatched_buffer to TILE_LAYOUT for routed experts
        dispatched_buffer = ttnn.to_layout(
            ttnn.squeeze(ttnn.squeeze(dispatched_buffer, dim=0), dim=0), ttnn.TILE_LAYOUT
        )
        logger.debug(f"[TtMoe.forward] dispatched_buffer_tiled shape: {dispatched_buffer.shape}")

        expert_outputs = self.routed_expert(dispatched_buffer, tt_expert_token_counts, tt_expert_region_offsets)

        if not return_intermediates:
            dispatched_buffer = ttnn.deallocate(dispatched_buffer)
        else:
            # add squeezed dimenisions back for intermediates to match original dispatch output shape
            dispatched_buffer = ttnn.unsqueeze(dispatched_buffer, dim=0)
            dispatched_buffer = ttnn.unsqueeze(dispatched_buffer, dim=0)

        logger.debug(f"[TtMoe.forward] expert_outputs shape: {expert_outputs.shape}")

        # Add back the batch dimensions for combine
        # (experts_per_chip, max_tokens, emb_dim) -> (1, 1, experts_per_chip, max_tokens, emb_dim)
        expert_outputs = ttnn.unsqueeze(expert_outputs, dim=0)
        expert_outputs = ttnn.unsqueeze(expert_outputs, dim=0)
        logger.debug(f"[TtMoe.forward] expert_outputs (unsqueezed) shape: {expert_outputs.shape}")

        # ========================================
        # Step 4: Combine (enabled)
        # ========================================
        # Combine expects TILE_LAYOUT input
        logger.debug(f"[TtMoe.forward] expert_outputs shape: {expert_outputs.shape} {expert_outputs.dtype=}")

        combined_output = self.combine_module(
            expert_outputs,
            metadata,
            tt_expert_token_counts,
            tt_expert_region_offsets,
        )
        logger.debug(f"[TtMoe.forward] combined_output shape: {combined_output.shape} {combined_output.dtype=}")

        # ========================================
        # Step 5: Reduce (fused weighted sum over topk + reduce-scatter for TP sharding)
        # ========================================
        # combined_output: (1, dispatch_group_size, seq_len_per_chip, num_experts_per_tok, emb_dim)
        #                  (1, 1, 256, 4, 2048) per device - 5D tensor, ROW_MAJOR
        #
        # TtReduceModule uses fused post_combine_reduce kernel:
        # 1. Fused weighted sum over topk (dim=3): reads ROW_MAJOR, outputs TILE_LAYOUT
        # 2. Reduce-scatter across TP axis: (1, 1, 256, 2048) -> (1, 1, 256, 512) per device
        routed_output = self.reduce_module(
            combined_output,
            weights=scores,
            indices=indices,
            expert_dispatch_table=self.tt_expert_dispatch_table,
        )
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
            # Check for buffer overflow (dispatch kernel silently drops overflow tokens).
            # The kernel bounds-check is against max_dispatch_buffer_token_size (total per-chip
            # buffer capacity). Group-sparse counts mean each chip's experts_per_chip-sized
            # chunk of _counts_host holds that chip's nonzero counts; the sum of each chunk is
            # the chip's total dispatched tokens and must fit in the dispatch buffer.
            _counts_4d = ttnn.unsqueeze_to_4D(tt_expert_token_counts)
            _ep_composer = ttnn.create_mesh_composer(self.mesh_device, ttnn.MeshComposerConfig(dims=[1, 0]))
            _counts_host = ttnn.to_torch(_counts_4d, mesh_composer=_ep_composer).squeeze(2)
            _per_chip_sums = _counts_host.to(torch.int64).flatten().view(-1, self.experts_per_chip).sum(dim=1)
            max_per_chip_sum = int(_per_chip_sums.max().item())
            max_capacity = self.dispatch_module.max_dispatch_buffer_token_size
            logger.info(
                f"[TtMoe.forward] max per-chip dispatched token sum: {max_per_chip_sum} "
                f"(max_dispatch_buffer_token_size={max_capacity})"
            )
            if max_per_chip_sum > max_capacity:
                logger.error(
                    f"[TtMoe.forward] per-chip dispatched token sum ({max_per_chip_sum}) exceeds "
                    f"max_dispatch_buffer_token_size ({max_capacity}). "
                    f"Overflow tokens were dropped - output data is corrupted. "
                    f"Reduce sequence length."
                )
                logger.debug(f"[TtMoe.forward] expert_token_counts: {_counts_host.flatten().tolist()}")
                logger.debug(f"[TtMoe.forward] per_chip_sums: {_per_chip_sums.tolist()}")

            # Every per-expert region offset must address a row inside the dispatch buffer
            # (i.e. < max_dispatch_buffer_token_size). An offset >= capacity means the
            # expert's region starts past the end of the buffer and its tokens are dropped.
            _offsets_4d = ttnn.unsqueeze_to_4D(tt_expert_region_offsets)
            _offsets_host = ttnn.to_torch(_offsets_4d, mesh_composer=_ep_composer).squeeze(2)
            _offsets_flat = _offsets_host.to(torch.int64).flatten()
            _argmax_offset = int(_offsets_flat.argmax().item())
            max_region_offset = int(_offsets_flat[_argmax_offset].item())
            max_offset_token_count = int(_counts_host.to(torch.int64).flatten()[_argmax_offset].item())
            logger.info(
                f"[TtMoe.forward] max expert region offset: {max_region_offset} "
                f"(token_count for that expert: {max_offset_token_count}, "
                f"max_dispatch_buffer_token_size={max_capacity})"
            )
            if max_region_offset >= max_capacity:
                logger.error(
                    f"[TtMoe.forward] expert region offset ({max_region_offset}) is not below "
                    f"max_dispatch_buffer_token_size ({max_capacity}). "
                    f"Overflow tokens were dropped - output data is corrupted. "
                    f"Reduce sequence length."
                )
                logger.debug(f"[TtMoe.forward] expert_region_offsets: {_offsets_host.flatten().tolist()}")

            intermediates = TtMoEIntermediates(
                gate_scores=scores,
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
