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
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_latent_proj import TtLatentMoeProjections
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, TtMoEGateConfig, TtMoEGatePrefill
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_intermediates import TtMoEIntermediates
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import ACTIVATION_SILU, TtSharedExpert
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl
from models.demos.deepseek_v3_d_p.utils.expert_dtypes import DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE

# Four similarly-named dimensions, shown for Kimi-K3, the only variant where all four differ:
#
#   emb_dim            7168  model hidden. Gate input, shared-expert input, block in/out.
#   routed_emb_dim     3584  width the routed experts run at (K3's latent space); defaults to emb_dim.
#   hidden_dim         3072  per-routed-expert FFN intermediate.
#   shared_hidden_dim  6144  shared-expert FFN intermediate; defaults to hidden_dim.


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
    def check_cache_complete(
        cache_path: Path,
        layer_idx: int,
        experts_per_chip: int,
        use_latent_moe: bool = False,
        latent_use_norm: bool = True,
        routed_expert_weights_dtype: ttnn.DataType = DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
    ) -> bool:
        """Check if MoE cache is complete (gate + routed experts + shared expert [+ latent proj]).

        ``use_latent_moe`` must be passed for Kimi-K3, otherwise a cache missing the latent
        projections would be reported complete and __init__ would then fail loading them.

        routed_expert_weights_dtype: dtype the routed experts were/will be BUILT at.
        as_tensor stamps it into the tensorbin filename, so the completeness check must pin the
        same value it will later request -- otherwise a stale cache at another dtype reports
        complete and the empty placeholder is loaded as the weights.
        """
        prefix = f"layer_{layer_idx}"
        if not TtMoEGatePrefill.check_cache_complete(cache_path, f"{prefix}.gate"):
            return False
        if not TtRoutedExpert.check_cache_complete(
            cache_path, f"{prefix}.routed_expert", experts_per_chip, routed_expert_weights_dtype
        ):
            return False
        if not TtSharedExpert.check_cache_complete(cache_path, f"{prefix}.shared_expert"):
            return False
        if use_latent_moe and not TtLatentMoeProjections.check_cache_complete(
            cache_path, f"{prefix}.latent_proj", use_norm=latent_use_norm
        ):
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
        shared_hidden_dim: int | None = None,
        routed_emb_dim: int | None = None,
        latent_weights: dict | None = None,
        latent_use_norm: bool = True,
    ):
        """Build TTNN cache for MoE (gate + routed experts + shared expert) without device copy.

        ``shared_hidden_dim`` defaults to ``hidden_dim``; pass it separately when the shared expert's
        intermediate differs from the routed experts' (Kimi-K3: 6144 vs 3072). Note the routed-expert
        cache needs no dim arguments -- it derives shapes from the weight tensors themselves -- so
        ``routed_emb_dim`` does not appear here.
        """
        if shared_hidden_dim is None:
            shared_hidden_dim = hidden_dim
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
                shared_hidden_dim,
                mesh_device,
                shared_expert_weights_dtype,
                cache_path,
                f"layer_{layer_idx}.shared_expert",
            )

        # Gated on the weights, not the dims alone: without weights the placeholder branch would write
        # uninitialised tensorbins that check_cache_complete then reports as a complete cache.
        if latent_weights and routed_emb_dim is not None and routed_emb_dim != emb_dim:
            TtLatentMoeProjections.build_ttnn_cache(
                torch_weights=latent_weights,
                emb_dim=emb_dim,
                routed_emb_dim=routed_emb_dim,
                mesh_device=mesh_device,
                weights_dtype=shared_expert_weights_dtype,
                cache_path=cache_path,
                cache_name_prefix=f"layer_{layer_idx}.latent_proj",
                use_norm=latent_use_norm,
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
        routed_expert_weights_dtype=DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
        routed_expert_activation=ttnn.RoutedExpertActivation.Silu,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        shared_expert_activation: str = ACTIVATION_SILU,
        shared_expert_situ_beta: float | None = None,
        shared_expert_situ_linear_beta: float | None = None,
        gate_fallback_mode: GateComputeMode = GateComputeMode.HOST_ALL,
        weight_cache_path: Optional[Path] = None,
        layer_idx: int = 0,
        overlap_shared_expert_with_dispatch: bool = True,
        routing_use_l1_small_for_semaphores: bool = False,
        is_balanced: bool = False,
        routed_emb_dim: Optional[int] = None,
        shared_hidden_dim: Optional[int] = None,
        latent_weights: dict = None,
        latent_use_norm: bool = True,
        rms_norm_eps: float = 1e-5,
        max_gate_seq_len_per_chip: Optional[int] = None,
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
            shared_expert_activation: GLU activation the shared expert runs -- "silu" (default) or
                "situ" for Kimi-K3's SiTU-GLU. Independent of routed_expert_activation: the two
                sites run different ops (Python-composed vs fused kernel) at different widths.
            shared_expert_situ_beta / shared_expert_situ_linear_beta: SiTU softcap betas, required
                when shared_expert_activation == "situ".
            gate_weights: Dict with "weight" and "e_score_correction_bias" keys for gate
            gate_fallback_mode: Fallback mode for gate (default: HOST_ALL)
            overlap_shared_expert_with_dispatch: If True, run the shared expert and dispatch
                on disjoint sub-devices so they overlap on-chip. If False, skip sub-device
                setup and run them sequentially on the full Tensix grid.
            is_balanced: If True, uses zigzag sequence placement for padding awareness.
                Should match the is_balanced flag used in MLA/transformer.
            routed_emb_dim: Width the ROUTED side runs at -- dispatch, routed experts, combine and
                reduce. Defaults to emb_dim, which is every model except Kimi-K3. K3 sets it to 3584
                against an emb_dim of 7168 ("LatentMoE"), halving the bytes each dispatched token
                moves over fabric. The gate is unaffected and still reads the full emb_dim, as does
                the shared expert.
            shared_hidden_dim: The SHARED expert's FFN intermediate. Defaults to hidden_dim. K3 needs
                it separate because its shared expert is a single MLP at moe_intermediate_size *
                num_shared_experts (6144), while hidden_dim is the per-routed-expert intermediate
                (3072). Every prior model has num_shared_experts == 1, so the two coincided and one
                parameter sufficed.
            latent_weights: Dict with down_proj / up_proj / norm for the LatentMoE projections.
                Only read when routed_emb_dim implies a latent space; None when the TTNN cache exists.
            latent_use_norm: Whether a latent RMSNorm sits between the reduce and the up-projection
                (K3: latent_moe_use_norm=True).
            rms_norm_eps: eps for that latent norm. Passed explicitly because
                TtDistributedRmsNorm defaults to 1e-6 while K3's config says 1e-5.
            routed_expert_activation: GLU activation the fused routed-expert kernel runs.
                Defaults to SiLU (DeepSeek / K2.6 / GLM). Kimi-K3 passes SituGlu. Routed only --
                the shared expert takes shared_expert_activation, which is a separate knob.
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
        self.routed_emb_dim = emb_dim if routed_emb_dim is None else routed_emb_dim
        self.shared_hidden_dim = hidden_dim if shared_hidden_dim is None else shared_hidden_dim
        self.use_latent_moe = self.routed_emb_dim != emb_dim

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
        # Optional SubDeviceTraceController (capture phase): the shared-expert/dispatch overlap's
        # sub-device load/clear go through it so the trace can be split at those boundaries instead of
        # resetting worker state mid-capture. None => load/clear the mesh device directly.
        self._trace_controller = None

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
            max_sp_dim=max_gate_seq_len_per_chip,
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
        # Sub-devices: when overlap is enabled, split the Tensix grid into a "dispatch"
        # strip and a "shared expert" strip so the two ops run on disjoint cores and the
        # Fast-Dispatch per-sub-device counters let them overlap on-chip.
        #   sub-device 0 (dispatch_sd):     rows [0, dispatch_sd_rows)
        #   sub-device 1 (shared_sd):       rows [dispatch_sd_rows, grid_y)
        # When overlap is disabled, both ops run sequentially on the full grid and no
        # sub-device manager is created.
        # ========================================
        if self.overlap_shared_expert_with_dispatch:
            dispatch_sd_rows = 1
            grid = mesh_device.compute_with_storage_grid_size()
            grid_x, grid_y = grid.x, grid.y
            assert 0 < dispatch_sd_rows < grid_y, f"dispatch_sd_rows={dispatch_sd_rows} must be in (0, grid_y={grid_y})"
            dispatch_cores = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, dispatch_sd_rows - 1))}
            )
            shared_cores = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, dispatch_sd_rows), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}
            )
            dispatch_sd = ttnn.SubDevice([dispatch_cores])
            shared_sd = ttnn.SubDevice([shared_cores])
            self.sd_manager_id = mesh_device.create_sub_device_manager([dispatch_sd, shared_sd], 0)
            self.dispatch_sd_id = ttnn.SubDeviceId(0)
            self.shared_sd_id = ttnn.SubDeviceId(1)
            # Stash the CoreRangeSet of the shared sub-device so TtSharedExpert can build
            # sub-device-confined shard_specs in Python without a C++ worker_cores binding.
            self.shared_sd_cores = shared_cores
            logger.debug(
                f"Sub-devices: grid={grid_x}x{grid_y}, dispatch=rows[0,{dispatch_sd_rows}), "
                f"shared=rows[{dispatch_sd_rows},{grid_y})"
            )
        else:
            self.sd_manager_id = None
            self.dispatch_sd_id = None
            self.shared_sd_id = None
            self.shared_sd_cores = None
            logger.debug("Sub-devices disabled: shared expert and dispatch will run sequentially")

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
            emb_dim=self.routed_emb_dim,
            cluster_axis=0,
            num_links=self.row_num_links,
            topology=self.row_topology,
            subdevice_id=self.dispatch_sd_id,
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
            emb_dim=self.routed_emb_dim,
            hidden_dim=hidden_dim,
            max_tokens=max_dispatched_tokens_per_expert,
            torch_weights=routed_expert_weights,
            activations_dtype=routed_expert_activations_dtype,
            weights_dtype=routed_expert_weights_dtype,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.routed_expert",
            activation=routed_expert_activation,
        )

        # Initialize shared expert (col axis: axis 1)
        self.shared_expert = TtSharedExpert(
            mesh_device=mesh_device,
            # The shared expert reads the pre-projection hidden, so it stays at the full emb_dim.
            emb_dim=emb_dim,
            hidden_dim=self.shared_hidden_dim,
            torch_weights=shared_expert_weights,
            num_links=self.col_num_links,
            topology=self.shared_expert_topology,
            activations_dtype=shared_expert_activations_dtype,
            weights_dtype=shared_expert_weights_dtype,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.shared_expert",
            subdevice_id=self.shared_sd_id,
            subdevice_cores=self.shared_sd_cores,
            activation=shared_expert_activation,
            situ_beta=shared_expert_situ_beta,
            situ_linear_beta=shared_expert_situ_linear_beta,
        )

        self.latent_projections = (
            TtLatentMoeProjections(
                mesh_device=mesh_device,
                emb_dim=emb_dim,
                routed_emb_dim=self.routed_emb_dim,
                torch_weights=latent_weights,
                use_norm=latent_use_norm,
                rms_norm_eps=rms_norm_eps,
                # ~10% of routed-expert FLOPs, so they take the shared expert's precision.
                weights_dtype=shared_expert_weights_dtype,
                num_links=self.col_num_links,
                topology=self.col_topology,
                weight_cache_path=weight_cache_path,
                cache_name_prefix=f"layer_{layer_idx}.latent_proj",
            )
            if self.use_latent_moe
            else None
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

    def set_trace_controller(self, controller):
        """Attach (or clear with None) a SubDeviceTraceController. While set, the shared-expert/
        dispatch overlap routes its sub-device load/clear through the controller so a ttnn trace can
        be split at those boundaries (see utils/sub_device_trace.py)."""
        self._trace_controller = controller

    def release_sub_device_manager(self):
        """Remove the overlap sub-device manager this MoE created (no-op if overlap is off). Call
        before closing the mesh device — leaving managers registered at close has been observed to
        segfault the teardown. Idempotent."""
        if getattr(self, "sd_manager_id", None) is not None:
            self.mesh_device.remove_sub_device_manager(self.sd_manager_id)
            self.sd_manager_id = None

    def forward(
        self,
        x: ttnn.Tensor,
        return_intermediates: bool = False,
        actual_isl: int = None,
        padding_side: str = "right",
        actual_start: Optional[int] = None,
        metadata: Optional[tuple] = None,
    ) -> tuple[ttnn.Tensor, Optional[TtMoEIntermediates]]:
        """
        Forward pass through the full MoE pipeline.

        Args:
            x: Input tensor - ROW_MAJOR, sharded:
               - For 2D mesh: sharded dims=(0, -1) - dim 0 across axis 0, dim -1 across axis 1
               - Shape per device: (dispatch_group_size/axis0, seq_len_per_chip, emb_dim/axis1)
            return_intermediates: If True, return intermediate tensors for debugging
            actual_isl: Actual ISL of the sequence (None = no padding). Doubles as the padding-config
                GUARD: not-None enables padding awareness. On the traced path (`metadata` set) that is
                its ONLY role — pass the full chunk, since the real per-chunk bound is read on-device
                from the metadata tensors and this value is unused.
            padding_side: Padding side of the sequence
            actual_start: chunked-prefill absolute KV position of this chunk's first real token
                (None/0 = single-shot, sequential SP layout). Required for correct per-chip real-token
                counts: chunked prefill feeds the KV-pad-aware ROTATED block-cyclic layout, where a
                chip's real rows are NOT its slice of the natural sequence. See build_padding_config.
            metadata: the traced path's (slot_id, actual_start, actual_end) tuple of 1-element uint32
                device tensors. When given, the padding config is built ON DEVICE from them
                (build_padding_config_device) instead of on host — the host builder's from_torch is
                illegal inside a trace capture, and a config baked in at capture time would be wrong
                for every later chunk. Ignored unless padding awareness is active (actual_isl set and
                a DEVICE_FP32 gate).

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
        # Padding awareness is only validated/safe for RIGHT padding. With right padding,
        # real tokens have the lowest indices, so they are packed first in every expert
        # region and stay within the shortened FFN/dispatch bound. For left padding the
        # real tokens land at the tail of each region while padded tokens (in non-sentinel
        # gate modes) are dispatched first, so a shortened bound could drop real tokens.
        # Disable padding awareness for left padding and process the full (always-correct)
        # token range by clearing actual_isl for the rest of this forward.
        #
        # Under trace this stays ON: the captured (metadata) forward always passes the FULL chunk as
        # actual_isl, so build_padding_config yields a single full-range config that the memoization
        # below builds once during warm-up — the replayed command stream contains no host transfer.
        # Only the eager/scalar paths pass a partial actual_isl, and that is the rotated-padded path
        # #51440 fixed.
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
        #
        # NOTE on `actual_isl` in this guard: it plays two DIFFERENT roles depending on the path.
        #   * eager/scalar: it is BOTH the guard (not-None => padding awareness on) AND the actual
        #     bound handed to the host builder below.
        #   * traced (metadata set): it is ONLY the guard. The real per-chunk bound is read on-device
        #     from the metadata tensors, so the caller passes the FULL chunk here (see
        #     TtPrefillRuntime._forward_traced, actual_isl=chunk_size) purely to say "padding
        #     awareness is ON" — its VALUE is deliberately unused. That is what keeps the guard
        #     itself trace-safe: it is a static, capture-time decision (one captured program either
        #     has the padding-aware ops or it does not), while the values that change per chunk stay
        #     on-device. A caller that wanted padding awareness OFF under trace would pass
        #     actual_isl=None and get a capture with no padding-aware path at all.
        padding_config = None
        if actual_isl is not None and self.gate.fallback_mode == GateComputeMode.DEVICE_FP32:
            if metadata is not None:
                # Traced path: the per-chunk scalars live on-device in the metadata tensors, so build
                # the config with the device op. The host builder's from_torch cannot run inside a
                # capture, and a config baked in at capture time would be wrong for every later chunk.
                # `actual_isl` is NOT forwarded here — only the guard above consumed it; the op derives
                # the real bound from actual_start/actual_end itself.
                # Raises for is_balanced=True (not expressible in the op's closed form).
                padding_config = self.gate.build_padding_config_device(metadata, padding_side)
            else:
                padding_config = self.gate.build_padding_config(actual_isl, padding_side, actual_start or 0)

        scores, indices, gate_logits = self.gate(
            ttnn.view(x, (x.shape[0] * x.shape[1], x.shape[2])),
            actual_isl=actual_isl,
            padding_side=padding_side,
            padding_config=padding_config,
            actual_start=actual_start or 0,
        )

        tt_expert_offsets, tt_expert_token_counts, tt_expert_region_offsets, _ = self.routing_setup(
            ttnn_top_k_experts_indices=indices,
            num_routed_experts=self.num_routed_experts,
            num_experts_per_tok=self.num_experts_per_tok,
        )

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

        # ========================================
        # Step 0b: LatentMoE -- project into the latent space (Kimi-K3 only)
        # ========================================
        # Outside the sub-device window below: the down-projection feeds dispatch, so it cannot
        # overlap it. x stays full-width for the shared expert, which reads the pre-projection hidden.
        routed_x = self.latent_projections.to_latent(x) if self.use_latent_moe else x
        if self.use_latent_moe:
            logger.debug(f"[TtMoe.forward] routed_x (latent) shape: {routed_x.shape}")

        signpost("shared_expert_and_dispatch_start")
        if self.overlap_shared_expert_with_dispatch:
            if self._trace_controller is not None:
                self._trace_controller.sub_device_load(self.sd_manager_id)
            else:
                self.mesh_device.load_sub_device_manager(self.sd_manager_id)

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
        # Dispatch expects complete routed-side rows on each device: full emb_dim normally, or the
        # all-gathered latent under LatentMoE (routed_x is x itself when there is no latent space).
        logger.debug(f"[TtMoe.forward] {routed_x.shape=} {routed_x.memory_config()=}")
        dispatched_buffer, metadata = self.dispatch_module(
            routed_x,
            scores,
            indices,
            tt_expert_offsets,
            self.tt_expert_dispatch_table,
            padding_config=padding_config,
        )
        if self.overlap_shared_expert_with_dispatch:
            if self._trace_controller is not None:
                self._trace_controller.sub_device_clear()
            else:
                self.mesh_device.clear_loaded_sub_device_manager()
        # NOTE: padding_config is memoized + owned by the gate (build_padding_config caches it per
        # actual_isl so a captured trace's replay reuses the same device tensor instead of re-issuing a
        # host from_torch). Do NOT deallocate it here — it is reused across forwards/replays; freeing it
        # would leave the cache holding a deallocated tensor (next forward's cache hit fails is_allocated()).
        # Dispatch has consumed routed_x by here. Under LatentMoE it is the latent buffer, a distinct
        # allocation needing its own free; without a latent space it IS x, so it is only an alias to
        # drop so that the deallocate below sees a single reference to the buffer.
        latent_input = None
        if not self.use_latent_moe:
            routed_x = None
        elif return_intermediates:
            # to_memory_config short-circuits when the config already matches, handing back a tensor
            # that SHARES routed_x's buffer instead of copying it -- and routed_x is interleaved-DRAM
            # on every path run today. Freeing it then would leave the PCC path reading freed memory,
            # so release routed_x only when a real copy was made.
            latent_input = ttnn.to_memory_config(routed_x, ttnn.DRAM_MEMORY_CONFIG)
            if routed_x.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                routed_x = ttnn.deallocate(routed_x, force=True)
        else:
            routed_x = ttnn.deallocate(routed_x, force=True)
        x = ttnn.deallocate(x, force=True)
        scores = ttnn.to_memory_config(scores, ttnn.DRAM_MEMORY_CONFIG)
        indices = ttnn.to_memory_config(indices, ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(f"[TtMoe.forward] Dispatch output: buffer={dispatched_buffer.shape}, metadata={metadata.shape}")

        signpost("shared_expert_and_dispatch_end")

        # ========================================
        # Step 3: Routed experts (enabled)
        # ========================================
        # Dispatch output is (1, dispatch_group_size_per_device, experts_per_chip, max_tokens, emb_dim)
        # Routed expert expects (experts_per_chip, max_tokens, emb_dim)
        # Squeeze the first two dimensions

        # TtRoutedExpert.forward owns the per-arch layout/dtype prep: Blackhole
        # consumes the ROW_MAJOR bf16 buffer and returns a fresh output; Wormhole
        # tiles it internally for the extract loop. Either way the ROW_MAJOR input
        # is independent of the result and can be freed here, unless the PCC check
        # needs it to compare against the bfloat16 torch reference.
        squeezed_dispatch = ttnn.squeeze(ttnn.squeeze(dispatched_buffer, dim=0), dim=0)
        expert_outputs = self.routed_expert(squeezed_dispatch, tt_expert_token_counts, tt_expert_region_offsets)
        if not return_intermediates:
            dispatched_buffer = ttnn.deallocate(dispatched_buffer)
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

        # ========================================
        # Step 5b: LatentMoE -- project back out of the latent space (Kimi-K3 only)
        # ========================================
        # Must stay above the squeeze: the distributed norm inside from_latent() is rank-4 only.
        latent_routed_output = None
        if self.use_latent_moe:
            if return_intermediates:
                # Reshape only; from_latent() reads routed_output without mutating it.
                latent_routed_output = ttnn.squeeze(routed_output, dim=0)
            routed_output = self.latent_projections.from_latent(routed_output)
            logger.debug(f"[TtMoe.forward] routed_output (after latent up_proj) shape: {routed_output.shape}")

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
                latent_routed_output=latent_routed_output,
                latent_input=latent_input,
                expert_token_counts=tt_expert_token_counts,
            )

        signpost(header="MoE_END")
        return final_output, intermediates
