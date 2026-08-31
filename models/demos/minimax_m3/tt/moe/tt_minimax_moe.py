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

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, TtMoEGateConfig, TtMoEGatePrefill
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.minimax_m3.tt.moe.tt_reduce import TtMiniMaxReduce
from models.demos.minimax_m3.utils.profiler_utils import FINE, zone


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
        route_scale: float = 1.0,
        reduce_scatter_fn=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.seq_len_per_chip = seq_len_per_chip
        self.experts_per_chip = experts_per_chip
        self.emb_dim = emb_dim

        # MiniMax routing: sigmoid + e_score_correction_bias, no groups -> n_group=1. route_scale must
        # match the model's routed_scaling_factor (2.0 for M3): the internal gate applies it to the
        # returned top-k weights whenever it runs instead of a caller-supplied topk.
        gate_config = TtMoEGateConfig(
            dim=emb_dim,
            sp_dim=seq_len_per_chip,
            n_routed_experts=num_routed_experts,
            n_activated_experts=num_experts_per_tok,
            n_expert_groups=1,
            n_limited_groups=1,
            route_scale=route_scale,
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
            # init_zeros=False saves a ~95 us full-buffer zeroing on every chip's critical path per
            # call. It is safe iff every combine-output slot post_combine_reduce reads is guaranteed
            # written: (a) pad rows carry the gate's sentinel expert id whose dispatch-table entry is
            # -1, so pcr skips them; (b) dispatch drops are impossible because the dispatch buffer is
            # sized for the worst case (dispatch_buffer_capacity_factor=4 in mlp.py — factor 2 could
            # drop rows under extreme skew, which is what caused the token-0 NaN of 2026-06-29 and
            # forced init_zeros=True until now). If the factor is ever lowered, this must go back to
            # True.
            init_zeros=False,
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
        # M3's own reduce module (tt/moe/tt_reduce.py), not DeepSeek's: same shared post_combine_reduce
        # kernel, but the closing collective goes through the caller's reduce_scatter_fn — M3 passes
        # MeshConfig.reduce_scatter (reduce_scatter_minimal_async + ping-pong/barrier semaphores) so the
        # MoE's collective matches every other M3 collective instead of being the one plain prim call.
        self.reduce_module = TtMiniMaxReduce(
            mesh_device=mesh_device,
            topk_dim=3,
            cluster_axis=1,
            num_links=num_links,
            topology=topology,
            reduce_scatter_fn=reduce_scatter_fn,
        )

    def forward(self, x, topk_indices=None, topk_weights=None, padding_config=None):
        """Routed (expert-parallel) MoE output.

        x: (dispatch_group_size, seq_len_per_chip, emb_dim) — emb may be TP-sharded
           (then it's all-gathered to full) or already full (replicated, e.g. from the
           decoder layer) in which case the gather is skipped.
        padding_config: the gate's per-device [num_real_tokens, pad_side] row for this chunk, or None
           for a full chunk. Bounds dispatch's token loop, and MUST be the same tensor the gate used —
           see tt/topk.py build_padding_config.

        topk_indices/topk_weights: optional external routing [tokens, topk] (from
           MiniMax's TopKRouter). When given, the internal DeepSeek gate is skipped —
           this is the production path (the layer feeds replicated full emb, which the
           DeepSeek host gate's TP-compose would mishandle). When None, the internal
           gate runs (standalone test path; expects TP-sharded emb).
        """
        if topk_indices is None:
            with zone("gate", FINE):
                scores, indices, gate_logits = self.gate(ttnn.view(x, (x.shape[0] * x.shape[1], x.shape[2])))
                ttnn.deallocate(gate_logits)
        else:
            indices, scores = topk_indices, topk_weights
        with zone("routing_setup", FINE):
            tt_expert_offsets, tt_expert_token_counts, tt_expert_region_offsets, _ = self.routing_setup(
                ttnn_top_k_experts_indices=indices,
                num_routed_experts=self.num_routed_experts,
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
            with zone("pre_dispatch_allgather"):
                x = ttnn.all_gather(
                    x, dim=-1, cluster_axis=1, num_links=self.reduce_module.num_links, topology=ttnn.Topology.Linear
                )

        # Dispatch -> per-expert buffers (NO shared expert)
        with zone("dispatch"):
            dispatched_buffer, metadata = self.dispatch_module(
                x,
                scores,
                indices,
                tt_expert_offsets,
                self.tt_expert_dispatch_table,
                padding_config=padding_config,
            )
            ttnn.deallocate(x)
            scores = ttnn.to_memory_config(scores, ttnn.DRAM_MEMORY_CONFIG)
            indices = ttnn.to_memory_config(indices, ttnn.DRAM_MEMORY_CONFIG)

        with zone("experts_mm"):
            # Hand the ROW_MAJOR dispatch buffer straight to the composite, as DeepSeek does: it then
            # tilizes only each expert's real token region in-kernel, instead of a standalone to_layout
            # tilizing the whole (mostly empty) dispatch buffer. The input stays live for the duration
            # of the call and the composite returns a fresh output, so free it after, not before.
            expert_outputs = self.routed_expert(
                ttnn.squeeze(ttnn.squeeze(dispatched_buffer, dim=0), dim=0),
                tt_expert_token_counts,
                tt_expert_region_offsets,
            )
            ttnn.deallocate(dispatched_buffer)
            expert_outputs = ttnn.unsqueeze(ttnn.unsqueeze(expert_outputs, dim=0), dim=0)

        with zone("combine"):
            combined_output = self.combine_module(
                expert_outputs, metadata, tt_expert_token_counts, tt_expert_region_offsets
            )
        # Fused weighted-sum over topk, then the TP reduce-scatter (see tt_reduce.py).
        with zone("moe_reduce"):
            routed_output = self.reduce_module(
                combined_output, weights=scores, indices=indices, expert_dispatch_table=self.tt_expert_dispatch_table
            )
            routed_output = ttnn.squeeze(routed_output, dim=0)
        return routed_output
