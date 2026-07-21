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

import os
from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, TtMoEGateConfig, TtMoEGatePrefill
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_intermediates import TtMoEIntermediates
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import TtSharedExpert
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

# L1_SMALL region reserved for the MoE routed-expert/combine overlap global semaphore.
# Required because TtMoe, when built with the overlap enabled, uses a global semaphore in L1_SMALL.
# A single semaphore is shared by all MoE layers (owned by TT_CCL), so this size is independent of
# layer count.
MOE_L1_SMALL_REGION_SIZE = 512


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
        - Routed Expert: ROW_MAJOR → TILE_LAYOUT
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

            # Minimal config for caching
            gate_config = TtMoEGateConfig(
                dim=emb_dim,
                n_routed_experts=gate_weights["weight"].shape[0],
            )

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
        emb_dim: int,
        hidden_dim: int,
        n_expert_groups: int,
        n_limited_groups: int,
        route_scale: float,
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
        overlap_shared_expert_with_dispatch: bool = True,
        routing_use_l1_small_for_semaphores: bool = False,
        is_balanced: bool = False,
        overlap_routed_expert_with_combine: bool = True,
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
            overlap_shared_expert_with_dispatch: If True, run the shared expert and dispatch
                on disjoint sub-devices so they overlap on-chip. If False, skip sub-device
                setup and run them sequentially on the full Tensix grid.
            is_balanced: If True, uses zigzag sequence placement for padding awareness.
                Should match the is_balanced flag used in MLA/transformer.
            overlap_routed_expert_with_combine: If True, overlap the routed expert compute
                with the combine. If False, run them sequentially.
        """
        super().__init__()
        self.mesh_device = mesh_device
        # Shared per-mesh CCL singleton: persistent global semaphores for the TP all-gather of x,
        # so all_gather_async reuses them instead of leaking fresh L1 semaphores every layer.
        self.tt_ccl = get_tt_ccl(mesh_device)
        self.dispatch_group_size = dispatch_group_size
        self.num_dispatch_groups = num_dispatch_groups
        self.experts_per_chip = experts_per_chip
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.seq_len_per_chip = seq_len_per_chip
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.overlap_routed_expert_with_combine = overlap_routed_expert_with_combine

        # The routed-expert/combine overlap relies on the unified_routed_expert_moe op, which
        # only runs on Blackhole (TtRoutedExpert.forward gates on is_blackhole()), so the overlap
        # is unsupported on other archs.
        # See https://github.com/tenstorrent/tt-metal/issues/47553
        assert not (
            overlap_routed_expert_with_combine and not is_blackhole()
        ), "overlap_routed_expert_with_combine is only supported on Blackhole"

        # Unpack row/col CCL config
        if isinstance(num_links, tuple):
            self.row_num_links, self.col_num_links = num_links
        else:
            self.row_num_links = self.col_num_links = num_links

        if isinstance(topology, tuple):
            self.row_topology, self.col_topology = topology
        else:
            self.row_topology = self.col_topology = topology

        self.overlap_shared_expert_with_dispatch = overlap_shared_expert_with_dispatch

        # The shared expert, WHEN OVERLAPPED with dispatch, runs on disjoint Tensix sub-devices that
        # still SHARE the EDM fabric routers. In that case its TP-axis reduce-scatter must stay Linear
        # even when the TP axis is a ring: a *ring* reduce-scatter concurrent with dispatch makes the
        # two ops' wrap-link traffic form a cyclic EDM buffer-credit dependency and deadlocks (the
        # shared-expert reduce_scatter wedges on its batch_ready_sem barrier at
        # ring_reduce_scatter_minimal_async_writer.cpp). This mirrors the proven FABRIC_2D_TORUS_Y
        # path, where the overlapped shared expert is Linear (col axis unwrapped) while dispatch
        # rings on SP. The forcing is gated on the overlap flag: with overlap disabled the reduce-
        # scatter runs alone (no concurrent dispatch on the shared routers), so Ring is safe and kept.
        # Every other TP-axis collective (MLA, dense FFN, gate, pre-dispatch all-gather, post-combine
        # reduce) is never overlapped and keeps col_topology (Ring).
        force_shared_expert_linear = (
            self.overlap_shared_expert_with_dispatch and self.col_topology == ttnn.Topology.Ring
        )
        self.shared_expert_topology = ttnn.Topology.Linear if force_shared_expert_linear else self.col_topology
        if force_shared_expert_linear:
            logger.info(
                "TtMoe: shared-expert reduce-scatter forced to Linear (overlapped with dispatch on a "
                "TP-ring fabric) to avoid an EDM deadlock; other TP collectives keep Ring"
            )

        # Always create dispatch table at init (static tensor) - needed by gate and dispatch module
        expert_dispatch_table = ExpertMapping.create_dispatch_table(
            num_routed_experts, dispatch_group_size, num_dispatch_groups
        )

        # Build gate internally
        gate_config = TtMoEGateConfig(
            dim=emb_dim,
            sp_dim=seq_len_per_chip,
            n_routed_experts=num_routed_experts,
            n_activated_experts=num_experts_per_tok,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            route_scale=route_scale,
        )
        gate_config.ccl_config["NUM_LINKS"] = self.col_num_links if isinstance(num_links, tuple) else num_links
        # The gate all-reduce runs on the TP axis (cluster_axis=TP_AXIS), so it follows col_topology.
        gate_config.ccl_config["TOPOLOGY"] = self.col_topology

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
            weight=gate_weight,
            bias=gate_bias,
            fallback_mode=gate_fallback_mode,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.gate",
            is_balanced=is_balanced,
        )

        self.routing_setup = TtMoERoutingSetup(
            mesh_device,
            expert_dispatch_table,
            num_links=gate_config.ccl_config["NUM_LINKS"],
            experts_per_chip=experts_per_chip,
            use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
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

        # ========================================
        # Sub-devices: when either overlap is enabled, split the Tensix grid into a
        # "data movement" (dm) strip and a "compute" strip so the two overlapped ops run on
        # disjoint cores and the Fast-Dispatch per-sub-device counters let them overlap on-chip.
        # The same split serves both overlaps:
        #   - shared-expert / dispatch overlap: dm = dispatch, compute = shared expert
        #   - routed-expert / combine overlap:  dm = combine,  compute = routed expert
        #   sub-device 0 (dm_sd):       rows [0, dm_sd_rows)
        #   sub-device 1 (compute_sd):  rows [dm_sd_rows, grid_y)
        # When both overlaps are disabled, ops run sequentially on the full grid and no
        # sub-device manager is created.
        # ========================================
        if overlap_shared_expert_with_dispatch or overlap_routed_expert_with_combine:
            dm_sd_rows = 1
            grid = mesh_device.compute_with_storage_grid_size()
            grid_x, grid_y = grid.x, grid.y
            assert 0 < dm_sd_rows < grid_y, f"dm_sd_rows={dm_sd_rows} must be in (0, grid_y={grid_y})"
            dm_cores = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, dm_sd_rows - 1))}
            )
            compute_cores = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, dm_sd_rows), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}
            )
            dm_sd = ttnn.SubDevice([dm_cores])
            compute_sd = ttnn.SubDevice([compute_cores])
            self.sd_manager_id = mesh_device.create_sub_device_manager([dm_sd, compute_sd], 0)
            self.dm_sd_id = ttnn.SubDeviceId(0)
            self.compute_sd_id = ttnn.SubDeviceId(1)
            logger.debug(
                f"Sub-devices: grid={grid_x}x{grid_y}, dm=rows[0,{dm_sd_rows}), " f"compute=rows[{dm_sd_rows},{grid_y})"
            )

            # Global semaphore is only needed for overlapping the routed expert with the combine.
            # See TT_CCL.get_routed_expert_global_semaphore.
            if overlap_routed_expert_with_combine:
                self.routed_expert_global_semaphore = self.tt_ccl.get_routed_expert_global_semaphore(dm_cores)

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
            subdevice_id=self.dm_sd_id if self.overlap_shared_expert_with_dispatch else None,
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
            init_zeros=False,
            subdevice_id=self.dm_sd_id if self.overlap_routed_expert_with_combine else None,
            global_semaphore=self.routed_expert_global_semaphore if self.overlap_routed_expert_with_combine else None,
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

        # Persistent, model-wide output buffer for the fused (ROW_MAJOR) routed-expert path. The
        # routed expert writes each expert's region into it while the overlapped combine reads the
        # same buffer.
        self.routed_expert_output = None
        if self.overlap_routed_expert_with_combine:
            self.routed_expert_output = self.tt_ccl.get_routed_expert_output_buffer(
                shape=(1, 1, max_dispatch_buffer_token_size, emb_dim),
                dtype=routed_expert_activations_dtype,
            )

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
            activation=ttnn.RoutedExpertActivation.Silu,
            subdevice_id=self.compute_sd_id if self.overlap_routed_expert_with_combine else None,
            global_semaphore=self.routed_expert_global_semaphore if self.overlap_routed_expert_with_combine else None,
            output_buffer=self.routed_expert_output,
        )

        # Initialize shared expert (col axis: axis 1)
        self.shared_expert = TtSharedExpert(
            mesh_device=mesh_device,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            torch_weights=shared_expert_weights,
            num_links=self.col_num_links,
            topology=self.shared_expert_topology,
            activations_dtype=shared_expert_activations_dtype,
            weights_dtype=shared_expert_weights_dtype,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.shared_expert",
            subdevice_id=self.compute_sd_id if self.overlap_shared_expert_with_dispatch else None,
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

        # Load debug flags from environment
        self.debug_token_count = os.getenv("TT_DS_PREFILL_DEBUG_TOKEN_COUNT", "0").lower() in ("1", "true", "yes")

        logger.debug("TtMoe initialization complete")

    def forward(
        self,
        x: ttnn.Tensor,
        return_intermediates: bool = False,
        actual_isl: int = None,
        padding_side: str = "right",
    ) -> tuple[ttnn.Tensor, Optional[TtMoEIntermediates]]:
        """
        Forward pass through the full MoE pipeline.

        Args:
            x: Input tensor - ROW_MAJOR, sharded:
               - For 2D mesh: sharded dims=(0, -1) - dim 0 across axis 0, dim -1 across axis 1
               - Shape per device: (dispatch_group_size/axis0, seq_len_per_chip, emb_dim/axis1)
            return_intermediates: If True, return intermediate tensors for debugging
            actual_isl: Actual ISL of the sequence (None = no padding)
            padding_side: Padding side of the sequence

        Returns:
            Tuple of (final_output, intermediates):
            - final_output: MoE output with same sharding as input
            - intermediates: TtMoEIntermediates if return_intermediates=True, else None
        """
        signpost(header="MoE_START")
        logger.debug(f"[TtMoe.forward] INPUT SHAPES:")
        logger.debug(f"  x.shape={x.shape}")

        # ========================================
        # Gate: compute weights/indices/offsets/counts from x
        # ========================================
        # Reshape 3D -> 2D for gate: (batch, seq, emb) -> (batch*seq, emb)

        # Padding awareness is only validated/safe for RIGHT padding. With right padding,
        # real tokens have the lowest indices, so they are packed first in every expert
        # region and stay within the shortened FFN/dispatch bound. For left padding the
        # real tokens land at the tail of each region while padded tokens (in non-sentinel
        # gate modes) are dispatched first, so a shortened bound could drop real tokens.
        # Disable padding awareness for left padding and process the full (always-correct)
        # token range by clearing actual_isl for the rest of this forward.
        if actual_isl is not None and padding_side != "right":
            logger.warning(
                "[TtMoe.forward] padding-aware MoE is only supported for right padding; "
                f"got padding_side={padding_side!r}. Falling back to the full token range."
            )
            actual_isl = None

        # Build the per-device [local_real_tokens, pad_side] config once and share the
        # SAME tensor between the gate topk (sentinel-marks padded rows) and the dispatch
        # op (bounds its token loop). This is only valid in DEVICE_FP32, where the gate
        # actually sentinel-marks padded tokens so routing_setup/combine stay consistent
        # with a shortened dispatch loop. In other gate modes padded tokens keep real
        # expert indices, so dispatch must process the full range -> padding_config=None.
        padding_config = None
        if actual_isl is not None and self.gate.fallback_mode == GateComputeMode.DEVICE_FP32:
            padding_config = self.gate.build_padding_config(actual_isl, padding_side)

        scores, indices, gate_logits = self.gate(
            ttnn.view(x, (x.shape[0] * x.shape[1], x.shape[2])),
            actual_isl=actual_isl,
            padding_side=padding_side,
            padding_config=padding_config,
        )

        signpost(header="moe_gate_calculate_dispatch_offsets")
        tt_expert_offsets, tt_expert_token_counts, tt_expert_region_offsets, _ = self.routing_setup(
            ttnn_top_k_experts_indices=indices,
            num_routed_experts=self.num_routed_experts,
            seq_len_per_chip=self.seq_len_per_chip,
            num_experts_per_tok=self.num_experts_per_tok,
        )
        signpost(header="moe_gate_calculate_dispatch_offsets")
        gate_logits = (
            ttnn.to_memory_config(gate_logits, ttnn.DRAM_MEMORY_CONFIG)
            if return_intermediates
            else ttnn.deallocate(gate_logits)
        )  # gate_logits is only used for debugging/intermediates, move to DRAM or deallocate immediately

        if self.debug_token_count:
            # DEBUG: Print full token counts per expert for monitoring (controlled by env var)
            _counts_4d = ttnn.unsqueeze_to_4D(tt_expert_token_counts)
            _ep_composer = ttnn.create_mesh_composer(self.mesh_device, ttnn.MeshComposerConfig(dims=[1, 0]))
            _counts_host = ttnn.to_torch(_counts_4d, mesh_composer=_ep_composer).squeeze(2)
            logger.info(f"[TtMoe.forward] expert_token_counts: {_counts_host.flatten().tolist()}")

            # DEBUG: Print full region offsets per expert for monitoring
            _offsets_4d = ttnn.unsqueeze_to_4D(tt_expert_region_offsets)
            _offsets_host = ttnn.to_torch(_offsets_4d, mesh_composer=_ep_composer).squeeze(2)
            logger.info(f"[TtMoe.forward] expert_region_offsets: {_offsets_host.flatten().tolist()}")

        # Ensure ROW_MAJOR layout for dispatch compatibility
        indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
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
            x = ttnn.experimental.all_gather_async(
                x,
                dim=-1,  # Gather along emb_dim
                cluster_axis=1,  # Gather across axis 1 (TP axis)
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=1),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1),
                num_links=self.col_num_links,
                topology=self.col_topology,
            )
        logger.debug(f"[TtMoe.forward] x (after all_gather) shape: {x.shape}")

        signpost("shared_expert_and_dispatch_start")
        if self.overlap_shared_expert_with_dispatch:
            self.mesh_device.load_sub_device_manager(self.sd_manager_id)
            self.mesh_device.set_sub_device_stall_group([self.compute_sd_id])

        # ========================================
        # Step 1: Dispatch (enabled)
        # ========================================
        # Dispatch expects full emb_dim on each device. It is the longer op, so the host
        # enqueues it first to maximize overlap with the shared expert.
        dispatched_buffer, metadata = self.dispatch_module(
            x,
            scores,
            indices,
            tt_expert_offsets,
            self.tt_expert_dispatch_table,
            padding_config=padding_config,
        )

        # ========================================
        # Step 2: Shared expert (enabled)
        # ========================================
        # Shared expert expects replicated input (full emb_dim)
        shared_output = self.shared_expert(x)
        logger.debug(f"[TtMoe.forward] Shared expert output shape: {shared_output.shape}")

        if self.overlap_shared_expert_with_dispatch:
            self.mesh_device.reset_sub_device_stall_group()
            self.mesh_device.clear_loaded_sub_device_manager()
        signpost("shared_expert_and_dispatch_end")

        # padding_config was shared with both the gate and dispatch; free it now that
        # dispatch (its last consumer) has been issued.
        if padding_config is not None:
            ttnn.deallocate(padding_config)
        ttnn.deallocate(x)
        scores = ttnn.to_memory_config(scores, ttnn.DRAM_MEMORY_CONFIG)
        indices = ttnn.to_memory_config(indices, ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(f"[TtMoe.forward] Dispatch output: buffer={dispatched_buffer.shape}, metadata={metadata.shape}")

        # ========================================
        # Step 3: Routed experts (enabled)
        # ========================================
        # The routed expert consumes the ROW_MAJOR dispatch buffer directly: on Blackhole it
        # tilizes and bf8-packs x internally (main removed the standalone to_layout in #49744);
        # on Wormhole it tiles it internally for the extract loop.

        # Overlap the routed expert with the combine
        signpost("routed_expert_and_combine_start")
        if self.overlap_routed_expert_with_combine:
            self.mesh_device.load_sub_device_manager(self.sd_manager_id)
            self.mesh_device.set_sub_device_stall_group([self.compute_sd_id])

        # ========================================
        # Steps 3 + 4: Routed experts + Combine
        # ========================================
        if self.overlap_routed_expert_with_combine:
            # Overlap: routed expert (compute sub-device) and combine (dm sub-device) run
            # concurrently, synchronized by the routed-expert global semaphore.
            combined_output = self.combine_module(
                self.routed_expert_output,
                metadata,
                tt_expert_token_counts,
                tt_expert_region_offsets,
            )
            expert_outputs = self.routed_expert(
                dispatched_buffer,
                tt_expert_token_counts,
                tt_expert_region_offsets,
            )
        else:
            # No overlap: combine must read the buffer only AFTER routed_expert has written it.
            expert_outputs = self.routed_expert(
                dispatched_buffer,
                tt_expert_token_counts,
                tt_expert_region_offsets,
            )
            combined_output = self.combine_module(
                expert_outputs,
                metadata,
                tt_expert_token_counts,
                tt_expert_region_offsets,
            )
        logger.debug(f"[TtMoe.forward] expert_outputs shape: {expert_outputs.shape} {expert_outputs.dtype=}")
        logger.debug(f"[TtMoe.forward] combined_output shape: {combined_output.shape} {combined_output.dtype=}")

        # Restore the default full-grid
        if self.overlap_routed_expert_with_combine:
            self.mesh_device.reset_sub_device_stall_group()
            self.mesh_device.clear_loaded_sub_device_manager()
        signpost("routed_expert_and_combine_end")

        # Free the dispatch buffer now that the routed expert and combine have consumed it.
        # When return_intermediates=True, keep it so the PCC check can compare against the
        # bfloat16 torch reference.
        if not return_intermediates:
            dispatched_buffer = ttnn.deallocate(dispatched_buffer)

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

        signpost(header="MoE_END")
        return final_output, intermediates
