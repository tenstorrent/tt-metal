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

import json

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombine2dModule, TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, TtMoEGateConfig, TtMoEGatePrefill
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.minimax_m3.tt.moe.tt_reduce import TtMiniMaxReduce
from models.demos.minimax_m3.utils.fabric_env import MOE_COMBINE_V2_FABRICS
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
        combine_version: str = "v1",
        dispatch_version: str = "v1",
        load_stats: bool = False,
        load_stats_file=None,
        global_layer_idx=None,
    ):
        """topology: v1 dispatch and v1 combine on cluster_axis=0 (the TP collectives on axis 1 stay Linear).
        combine_version: "v1" = deepseek_prefill.combine, "v2" = combine_fabric2d (Ring on a torus fabric).
        dispatch_version: "v1" = deepseek_prefill.dispatch, "v2" = dispatch_fabric2d (Ring on a torus fabric).
        load_stats: read the per-expert token counts back every forward and log one M3_MOE_LOAD line (host
           sync; measurement runs only). load_stats_file: also append the raw counts there as JSON lines.
        global_layer_idx: model layer index for the load-stats line only (layer_idx names the weight cache).
        """
        super().__init__()
        assert combine_version in ("v1", "v2"), f"combine_version must be 'v1' or 'v2', got {combine_version!r}"
        assert dispatch_version in ("v1", "v2"), f"dispatch_version must be 'v1' or 'v2', got {dispatch_version!r}"
        self.combine_version = combine_version
        self.dispatch_version = dispatch_version
        # Checked before any weight of this block loads.
        fabric2d_ops = [n for n, v in (("dispatch", dispatch_version), ("combine", combine_version)) if v == "v2"]
        if fabric2d_ops:
            self._check_fabric2d_v2(mesh_device, emb_dim, fabric2d_ops)
        self.mesh_device = mesh_device
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.seq_len_per_chip = seq_len_per_chip
        self.experts_per_chip = experts_per_chip
        self.emb_dim = emb_dim
        self.metadata_len = metadata_len
        self.max_dispatch_buffer_token_size = max_dispatch_buffer_token_size
        self.num_links = num_links

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
        worst_case_tokens = dispatch_group_size * seq_len_per_chip * num_experts_per_tok
        assert (
            max_dispatch_buffer_token_size >= worst_case_tokens
        ), f"init_zeros=False needs a drop-free dispatch buffer: {max_dispatch_buffer_token_size} < {worst_case_tokens}"
        if combine_version == "v2":
            # No init_zeros knob: the output is never zeroed, the same contract as v1 with init_zeros=False
            # (post_combine_reduce skips or zero-weights every slot combine does not write).
            self.combine_module = TtCombine2dModule(
                mesh_device=mesh_device,
                experts_per_chip=experts_per_chip,
                num_experts_per_tok=num_experts_per_tok,
                seq_len_per_chip=seq_len_per_chip,
                cluster_axis=0,
                num_links=num_links,
                topology=ttnn.Topology.Ring,
            )
        else:
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
                # No zero-init (~95 us per call). Invariant: every slot post_combine_reduce reads with a
                # non-zero weight is written, because the dispatch buffer is drop-free (asserted above).
                # Unwritten slots ARE still read: when none of a token's top-k experts is local to this
                # dispatch group (about (1-1/ndg)^topk of tokens, ~1/3 on the 8x4 mesh, plus every padded
                # row) the kernel's must_zero_init branch forces the last slot through with a writer-
                # forced zero weight, so the result is stale_slot * 0. That is exact for finite stale
                # data; for NaN/Inf it relies on the Blackhole FPU returning 0 for NaN*0 and Inf*0
                # (measured on BH Galaxy, not an IEEE guarantee). deepseek_v3_d_p prefill runs the same
                # path with init_zeros=False. Hard guarantee = kernel packs explicit zeros in the
                # must_zero_init branch instead of reading the slot.
                init_zeros=False,
            )
        global_expert_idx_table = ExpertMapping.create_global_expert_idx_table(
            experts_per_chip=experts_per_chip,
            dispatch_group_size=dispatch_group_size,
            num_dispatch_groups=num_dispatch_groups,
        )
        self.load_stats = load_stats
        if load_stats:
            self._init_load_stats(expert_dispatch_table, global_expert_idx_table, load_stats_file)
            self.load_stats_layer_idx = layer_idx if global_layer_idx is None else global_layer_idx
        global_expert_idx_tt = ttnn.from_torch(
            global_expert_idx_table,
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
            # Fallback reduce-scatter on axis 1 (TP), not the MoE's axis-0 topology.
            topology=ttnn.Topology.Linear,
            reduce_scatter_fn=reduce_scatter_fn,
        )

    @staticmethod
    def _check_fabric2d_v2(mesh_device, emb_dim, ops):
        """combine_fabric2d / dispatch_fabric2d: a torus wrapping axis 0 and a payload that fits a token."""
        what = " + ".join(f"{op} v2" for op in ops)
        fabric = ttnn.get_fabric_config()
        if fabric not in MOE_COMBINE_V2_FABRICS:
            raise RuntimeError(
                f"{what} needs a fabric that wraps axis 0 {[str(f) for f in MOE_COMBINE_V2_FABRICS]}, got {fabric}; "
                "open it with M3_FABRIC=2d_torus_xy via utils/fabric_env.set_fabric_config_from_env"
            )
        # One packet carries a bf16 token plus the op's 64 B routing / forwarding tail.
        needed = emb_dim * 2 + 64
        payload = ttnn.get_tt_fabric_max_payload_size_bytes()
        if payload < needed:
            raise RuntimeError(
                f"{what} needs fabric max payload >= {needed} B, fabric has {payload} B; open the fabric with "
                "utils/fabric_env.set_fabric_config_from_env (passes the router config under M3_MOE_*=v2)"
            )
        if "dispatch" in ops:
            # dispatch_fabric2d splits the diametrically opposite chip across both ring directions.
            extent = mesh_device.shape[0]
            if extent < 4 or extent % 2:
                raise RuntimeError(f"dispatch v2 needs an even axis-0 extent >= 4, mesh is {tuple(mesh_device.shape)}")

    def _init_load_stats(self, expert_dispatch_table, global_expert_idx_table, load_stats_file):
        E = self.num_routed_experts
        # Dispatch group (mesh column) that owns each expert: the one whose table row maps it to a chip.
        owned = expert_dispatch_table[:, :E] >= 0
        assert bool((owned.sum(dim=0) == 1).all()), "every expert must belong to exactly one dispatch group"
        self._load_expert_group = owned.to(torch.int64).argmax(dim=0)
        # (num_dispatch_groups, dispatch_group_size, experts_per_chip): experts on chip (row, col).
        self._load_chip_experts = global_expert_idx_table.to(torch.int64)
        self._load_stats_file = load_stats_file
        self._load_stats_calls = 0

    def _log_expert_load(self, total_counts_per_expert):
        """One M3_MOE_LOAD line from routing_setup's total_counts_per_expert (host sync).

        Per device the counts are (1, E), summed over the dispatch group (mesh column) and masked to that
        group's experts, so expert e is read from any row of the column that owns it.
        """
        counts_4d = ttnn.unsqueeze_to_4D(total_counts_per_expert)
        composer = ttnn.create_mesh_composer(self.mesh_device, ttnn.MeshComposerConfig(dims=[1, 0]))
        # (num_cols, num_rows, E)
        host = ttnn.to_torch(counts_4d, mesh_composer=composer).squeeze(2).to(torch.int64)
        E = self.num_routed_experts
        counts = host[self._load_expert_group, 0, torch.arange(E)]
        per_chip = counts[self._load_chip_experts].sum(dim=-1)  # (cols, rows)
        per_col = per_chip.sum(dim=-1)
        tile_padded = ((counts + 31) // 32 * 32)[self._load_chip_experts].sum(dim=-1)
        hot = int(counts.argmax())
        layer = self.load_stats_layer_idx
        logger.info(
            f"M3_MOE_LOAD layer={layer} tokens={int(counts.sum())} per_chip_max={int(per_chip.max())} "
            f"per_chip_mean={per_chip.double().mean().item():.1f} per_chip_min={int(per_chip.min())} "
            f"per_col_max={int(per_col.max())} per_col_mean={per_col.double().mean().item():.1f} "
            f"hottest_expert={hot}:{int(counts[hot])} tile_padded_max={int(tile_padded.max())}"
        )
        if self._load_stats_file:
            with open(self._load_stats_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "layer": layer,
                            "call": self._load_stats_calls,
                            "mesh": list(self.mesh_device.shape),
                            "experts_per_chip": self.experts_per_chip,
                            "counts": counts.tolist(),
                        }
                    )
                    + "\n"
                )
        self._load_stats_calls += 1

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
            routing = self.routing_setup(
                ttnn_top_k_experts_indices=indices,
                num_routed_experts=self.num_routed_experts,
                num_experts_per_tok=self.num_experts_per_tok,
            )
            tt_expert_offsets = routing.global_dispatch_offsets
            tt_expert_token_counts = routing.total_counts_per_expert
            tt_expert_region_offsets = routing.expert_region_offsets
            # (dispatch_group_size, E), row k = device k's global_dispatch_offsets, replicated along
            # axis 0: the expert_offsets table of combine_fabric2d and dispatch_fabric2d. Only v2 reads it.
            all_expert_offsets = routing.all_global_dispatch_offsets
            if self.combine_version != "v2" and self.dispatch_version != "v2":
                ttnn.deallocate(all_expert_offsets)
            del routing
            indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
            scores = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT)
            b, s = x.shape[0], x.shape[1]
            scores = ttnn.reshape(scores, (b, s, scores.shape[-1]))
            indices = ttnn.reshape(indices, (b, s, indices.shape[-1]))
        if self.load_stats:
            with zone("moe_load_stats", FINE):
                self._log_expert_load(tt_expert_token_counts)

        # Dispatch needs full emb per chip. All-gather across TP only if emb is sharded;
        # if the input is already full emb (replicated, e.g. from the decoder layer), skip.
        if self.mesh_device.shape[1] > 1 and x.shape[-1] < self.emb_dim:
            with zone("pre_dispatch_allgather"):
                x = ttnn.all_gather(
                    x, dim=-1, cluster_axis=1, num_links=self.reduce_module.num_links, topology=ttnn.Topology.Linear
                )

        # Dispatch -> per-expert buffers (NO shared expert)
        with zone("dispatch"):
            if self.dispatch_version == "v2":
                # dispatch_fabric2d takes bf16 x and uint16 ROW_MAJOR indices, every input interleaved in
                # DRAM. The router weights stay out of it, as with v1: moe_reduce applies them.
                with zone("dispatch_v2_prep", FINE):
                    x_in = x
                    if x.dtype != ttnn.bfloat16:
                        x = ttnn.typecast(x, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    else:
                        x = _to_dram_interleaved(x)
                    indices = _to_dram_interleaved(indices)
                dispatched_buffer, metadata = ttnn.experimental.deepseek_prefill.dispatch_fabric2d(
                    x,
                    indices,
                    all_expert_offsets,
                    self.tt_expert_dispatch_table,
                    tt_expert_token_counts,
                    tt_expert_region_offsets,
                    padding_config=padding_config,
                    experts_per_chip=self.experts_per_chip,
                    num_routed_experts=self.num_routed_experts,
                    num_experts_per_tok=self.num_experts_per_tok,
                    metadata_len=self.metadata_len,
                    max_dispatch_buffer_token_size=self.max_dispatch_buffer_token_size,
                    seq_len_per_chip=self.seq_len_per_chip,
                    cluster_axis=0,
                    num_links=self.num_links,
                    topology=ttnn.Topology.Ring,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if x is not x_in:
                    ttnn.deallocate(x_in)
                if self.combine_version != "v2":
                    ttnn.deallocate(all_expert_offsets)
            else:
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
            if self.combine_version == "v2":
                # combine_fabric2d takes BF16 only and the fused expert op emits bfp8 TILE with no dtype
                # option. bfp8 -> bf16 is exact; the op untilizes TILE input itself.
                with zone("combine_v2_prep", FINE):
                    expert_outputs_bf16 = ttnn.typecast(expert_outputs, ttnn.bfloat16)
                    ttnn.deallocate(expert_outputs)
                    expert_outputs = expert_outputs_bf16
                combined_output = self.combine_module(
                    expert_outputs, metadata, tt_expert_token_counts, tt_expert_region_offsets, all_expert_offsets
                )
                ttnn.deallocate(all_expert_offsets)
            else:
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


def _to_dram_interleaved(t):
    mc = t.memory_config()
    if mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return t
    return ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
