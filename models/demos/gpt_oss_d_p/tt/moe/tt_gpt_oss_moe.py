# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TtGptOssMoE — GPT-OSS prefill expert-parallel routed-expert MoE block.

Mirrors ``models/demos/minimax_m3/tt/moe/tt_minimax_moe.py`` almost line-for-line, composing the
(generic, already-validated) DeepSeek EP sub-modules:

    routing_setup -> all_gather(x) -> dispatch -> routed_expert -> combine -> reduce

GPT-OSS-specific differences vs the MiniMax-M3 template:
  - NO internal gate. GPT-OSS routing (top-k FIRST, then softmax over the k selected logits, with a
    Linear bias) is different math from DeepSeek/MiniMax and lives in ``TtGptOssRouter``. This module
    REQUIRES external ``topk_indices`` / ``topk_weights`` in ``forward`` (no TtMoEGatePrefill).
  - NO shared expert (GPT-OSS has none; the M3 shared-expert wiring is dropped).
  - Expert FFNs carry gate/up/down biases. The current ``unified_routed_expert_moe`` kernel on this
    branch does NOT accept bias args (#49619), so this module runs the routed experts BIAS-FREE.
    Biases are threaded as far as ``__init__`` behind ``use_expert_bias`` and a ``# TODO(#49619)``
    where they would be passed to ``TtRoutedExpert`` — see below.

Keeps M3's ``combine(init_zeros=True)`` (safe for skewed routing) and
``activation=ttnn.RoutedExpertActivation.SwiGluOai`` (bakes GPT-OSS's alpha=1.702 / limit=7.0).

Reference: models/demos/minimax_m3/tt/moe/tt_minimax_moe.py (TtMiniMaxMoE.__init__/forward).
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert


class TtGptOssMoE(LightweightModule):
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
        routed_expert_weights: list,  # per-expert {gate_proj, up_proj, down_proj}, global id order 0..E-1
        routed_expert_biases: list = None,  # per-expert {gate_bias, up_bias, down_bias}, same order (#49619)
        num_links: int = 2,
        topology=ttnn.Topology.Linear,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        weight_cache_path=None,
        layer_idx: int = 0,
        use_expert_bias: bool = False,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.seq_len_per_chip = seq_len_per_chip
        self.experts_per_chip = experts_per_chip
        self.emb_dim = emb_dim

        # ---- Expert bias (#49619) -----------------------------------------------------------------
        # The current unified_routed_expert_moe kernel on this branch is BIAS-FREE — passing a
        # bias/torch_biases kwarg to TtRoutedExpert WILL CRASH. We keep the prepared biases here so
        # the hookup is a one-line change once #49619 lands; do NOT enable use_expert_bias until then.
        self.routed_expert_biases = routed_expert_biases
        self.use_expert_bias = use_expert_bias
        # #49619 (bias in unified_routed_expert_moe / TtRoutedExpert) is now MERGED, so
        # use_expert_bias=True is supported: biases are passed to TtRoutedExpert below.

        expert_dispatch_table = ExpertMapping.create_dispatch_table(
            num_routed_experts, dispatch_group_size, num_dispatch_groups
        )

        # NOTE: no internal gate (TtMoEGatePrefill) — GPT-OSS routing lives in TtGptOssRouter and is
        # passed into forward() as external topk_indices / topk_weights.
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
            # init_zeros=True (M3 override): skewed routing leaves many empty combine slots; with
            # init_zeros=False those keep stale DRAM -> garbage weighted-sum -> nan. Zero-init them.
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

        # Routed expert: the fused unified_routed_expert_moe kernel with the clamped swigluoai
        # activation (RoutedExpertActivation.SwiGluOai bakes GPT-OSS's alpha=1.702 / limit=7.0).
        # #49619 (merged) added expert-bias support to TtRoutedExpert (torch_biases kwarg; the kernel
        # adds gate/up bias before the clamp and down bias after the down matmul, SwiGluOai only).
        # Passing the biases is required for correctness — without them the MoE output is
        # systematically off every layer and the error accumulates through the residual (V PCC
        # collapses with depth). Gated by use_expert_bias so the bias-free path stays available.
        self.routed_expert = TtRoutedExpert(
            mesh_device=mesh_device,
            experts_per_chip=experts_per_chip,
            global_expert_idx_table=global_expert_idx_tt,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            max_tokens=max_dispatched_tokens_per_expert,
            torch_weights=routed_expert_weights,
            torch_biases=self.routed_expert_biases if self.use_expert_bias else None,
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

    def forward(self, x, topk_indices, topk_weights):
        """Routed (expert-parallel) MoE output.

        x: (dispatch_group_size, seq_len_per_chip, emb_dim) — emb may be TP-sharded (then it is
           all-gathered to full) or already full (replicated, e.g. from the decoder layer), in which
           case the gather is skipped.
        topk_indices / topk_weights: REQUIRED external routing [tokens, k] from TtGptOssRouter
           (indices uint16/ROW_MAJOR, weights bf16). Unlike M3 there is no internal gate fallback.
        """
        assert topk_indices is not None and topk_weights is not None, (
            "TtGptOssMoE requires external topk_indices/topk_weights (from TtGptOssRouter); " "it has no internal gate."
        )
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

        # Dispatch needs full emb per chip. All-gather across TP only if emb is sharded; if the input
        # is already full emb (replicated, e.g. from the decoder layer), skip.
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

        dispatched_buffer_tiled = ttnn.to_layout(
            ttnn.squeeze(ttnn.squeeze(dispatched_buffer, dim=0), dim=0),
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
