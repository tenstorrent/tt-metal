# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TtMiniMaxMoE — MiniMax-M3 expert-parallel routed-expert MoE block.

Composes the (generic, already-validated) DeepSeek EP sub-modules:
    gate -> routing_setup -> dispatch -> routed_expert -> combine -> reduce
but owns the orchestration so it fits MiniMax-M3:
  - NO shared expert here — M3's always-on shared expert is added by the caller
    (tt/mlp.py); DeepSeek's TtMoe builds a mandatory one, which we drop.
  - NO expert groups (host gate; n_group=1 -> plain top-4)
  - emb=6144, hidden=3072, 128 experts / top-4 -> 4 experts/chip on 32 chips

The EP machinery (deepseek_prefill.{dispatch,routed_expert_ffn,combine,...}) is reused
verbatim, with the fused unified_routed_expert_ffn kernel selected for M3's clamped
swigluoai activation; only the shared-expert step of DeepSeek's TtMoe.forward is dropped.

Reference: models/demos/deepseek_v3_d_p/tt/moe/tt_moe.py (TtMoe.__init__/forward).
"""


import os

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, TtMoEGateConfig, TtMoEGatePrefill
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert


class TtMiniMaxMoE(LightweightModule):
    def __init__(
        self,
        mesh_device,
        dispatch_group_size: int,
        num_dispatch_groups: int,
        experts_per_chip: int,
        num_routed_experts: int,
        num_experts_per_tok: int,
        metadata_len: int,
        max_dispatched_tokens_per_expert: int,
        max_dispatch_buffer_token_size: int,
        seq_len_per_chip: int,
        emb_dim: int,
        hidden_dim: int,
        gate_weights: dict,  # {"weight": [E, emb], "e_score_correction_bias": [E]}
        routed_expert_weights: list,  # per-chip list of {gate_proj, up_proj, down_proj}
        num_links: int = 2,
        topology=ttnn.Topology.Linear,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        gate_fallback_mode: GateComputeMode = GateComputeMode.HOST_ALL,
        weight_cache_path=None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.seq_len_per_chip = seq_len_per_chip
        self.experts_per_chip = experts_per_chip
        self.emb_dim = emb_dim
        # Opt-in (bring-up) flag for the region-aware dispatch_tilize; resolved once here, not per forward().
        self.use_region_tilize = os.getenv("REGION_TILIZE") == "1"

        # MiniMax routing: sigmoid + e_score_correction_bias, no groups -> n_group=1.
        gate_config = TtMoEGateConfig(
            dim=emb_dim,
            sp_dim=seq_len_per_chip,
            n_routed_experts=num_routed_experts,
            n_activated_experts=num_experts_per_tok,
            n_expert_groups=1,
            n_limited_groups=1,
            route_scale=1.0,
        )
        gate_config.ccl_config["NUM_LINKS"] = num_links

        expert_dispatch_table = ExpertMapping.create_dispatch_table(
            num_routed_experts, dispatch_group_size, num_dispatch_groups
        )

        self.gate = TtMoEGatePrefill(
            gate_config,
            mesh_device,
            # .get(): an empty gate_weights dict means cache-only loading -> weight/bias=None makes
            # TtMoEGatePrefill load the tilized gate weight + bias straight from its cache.
            weight=gate_weights.get("weight"),
            bias=gate_weights.get("e_score_correction_bias"),
            fallback_mode=gate_fallback_mode,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.gate",
        )
        self.routing_setup = TtMoERoutingSetup(
            mesh_device, expert_dispatch_table, num_links=num_links, experts_per_chip=experts_per_chip
        )
        self.tt_expert_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(
            mesh_device, expert_dispatch_table, dispatch_axis=0
        )
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
            num_links=num_links,
            topology=topology,
            subdevice_id=None,
        )
        self.combine_module = TtCombineModule(
            mesh_device=mesh_device,
            dispatch_group_size=dispatch_group_size,
            num_dispatch_groups=num_dispatch_groups,
            experts_per_chip=experts_per_chip,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=seq_len_per_chip,
            cluster_axis=0,
            num_links=num_links,
            topology=topology,
            # M3's real routing is heavily skewed -> many empty experts/unwritten combine slots. With
            # init_zeros=False those slots keep STALE DRAM (a weight/old activation under the full-model
            # footprint) which the weighted-sum reads as a ~1e38 garbage value -> residual overflow -> nan
            # -> token-0. Zero-init the combine output so unwritten slots are 0. (DS default is True; the
            # False override was unsafe for skewed routing.) See token-0 debug 2026-06-29.
            init_zeros=True,
        )
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
        global_expert_idx_tt = ttnn.squeeze(ttnn.squeeze(global_expert_idx_tt, 0), 0)
        # M3 routed expert: the fused unified_routed_expert_moe kernel with the clamped swigluoai
        # activation (RoutedExpertActivation.SwiGluOai bakes in M3's alpha=1.702 / limit=7.0). This
        # replaced the earlier host-loop CompositeRoutedExpert once #47825 added swigluoai to the kernel.
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
            activation=ttnn.RoutedExpertActivation.SwiGluOai,
        )
        self.reduce_module = TtReduceModule(
            mesh_device=mesh_device,
            topk_dim=3,
            cluster_axis=1,
            num_links=num_links,
            topology=topology,
        )

    def forward(self, x, topk_indices=None, topk_weights=None):
        """Routed (expert-parallel) MoE output.

        x: (dispatch_group_size, seq_len_per_chip, emb_dim) — emb may be TP-sharded
           (then it's all-gathered to full) or already full (replicated, e.g. from the
           decoder layer) in which case the gather is skipped.
        topk_indices/topk_weights: optional external routing [tokens, topk] (from
           MiniMax's TopKRouter). When given, the internal DeepSeek gate is skipped —
           this is the production path (the layer feeds replicated full emb, which the
           DeepSeek host gate's TP-compose would mishandle). When None, the internal
           gate runs (standalone test path; expects TP-sharded emb).
        """
        if topk_indices is None:
            scores, indices, gate_logits = self.gate(ttnn.view(x, (x.shape[0] * x.shape[1], x.shape[2])))
            ttnn.deallocate(gate_logits)
        else:
            indices, scores = topk_indices, topk_weights
        tt_expert_offsets, tt_expert_token_counts, tt_expert_region_offsets, _ = self.routing_setup(
            ttnn_top_k_experts_indices=indices,
            num_routed_experts=self.num_routed_experts,
            seq_len_per_chip=self.seq_len_per_chip,
            num_experts_per_tok=self.num_experts_per_tok,
        )
        indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
        scores = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT)
        b, s = x.shape[0], x.shape[1]
        scores = ttnn.reshape(scores, (b, s, scores.shape[-1]))
        indices = ttnn.reshape(indices, (b, s, indices.shape[-1]))

        # Dispatch needs full emb per chip. All-gather across TP only if emb is sharded;
        # if the input is already full emb (replicated, e.g. from the decoder layer), skip.
        if self.mesh_device.shape[1] > 1 and x.shape[-1] < self.emb_dim:
            x = ttnn.all_gather(
                x, dim=-1, cluster_axis=1, num_links=self.reduce_module.num_links, topology=ttnn.Topology.Linear
            )

        # Dispatch -> per-expert buffers (NO shared expert)
        dispatched_buffer, metadata = self.dispatch_module(
            x, scores, indices, tt_expert_offsets, self.tt_expert_dispatch_table
        )
        ttnn.deallocate(x)
        scores = ttnn.to_memory_config(scores, ttnn.DRAM_MEMORY_CONFIG)
        indices = ttnn.to_memory_config(indices, ttnn.DRAM_MEMORY_CONFIG)

        _dispatched_rm = ttnn.squeeze(ttnn.squeeze(dispatched_buffer, dim=0), dim=0)
        if self.use_region_tilize:
            # Region-aware tilize: skips the worst-case dispatch padding (mostly empty). Pass the per-expert
            # token counts so the kernel bounds work by the filled prefix; experts_per_chip groups the counts
            # into the per-chip valid_rows (the fullest chip's fill). The padded tail past valid_rows is left
            # undefined — safe only because routed_expert (below) reads just the filled prefix.
            dispatched_buffer_tiled = ttnn.experimental.deepseek_prefill.dispatch_tilize(
                _dispatched_rm,
                tt_expert_token_counts,
                output_dtype=self.routed_expert.activations_dtype,
                experts_per_chip=self.experts_per_chip,
            )
        else:
            dispatched_buffer_tiled = ttnn.to_layout(
                _dispatched_rm,
                ttnn.TILE_LAYOUT,
                dtype=self.routed_expert.activations_dtype,
            )
        ttnn.deallocate(dispatched_buffer)

        expert_outputs = self.routed_expert(dispatched_buffer_tiled, tt_expert_token_counts, tt_expert_region_offsets)
        expert_outputs = ttnn.unsqueeze(ttnn.unsqueeze(expert_outputs, dim=0), dim=0)

        combined_output = self.combine_module(
            expert_outputs, metadata, tt_expert_token_counts, tt_expert_region_offsets
        )
        # Fused weighted-sum over topk + reduce-scatter across TP
        routed_output = self.reduce_module(
            combined_output, weights=scores, indices=indices, expert_dispatch_table=self.tt_expert_dispatch_table
        )
        routed_output = ttnn.squeeze(routed_output, dim=0)
        return routed_output
