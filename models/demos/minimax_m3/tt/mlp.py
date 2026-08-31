# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MoE MLP for MiniMax-M3: host top-k router + always-on shared expert + expert-parallel routed experts.

Expert-parallel only (the deployment path): the routed experts run TtMiniMaxMoE (the DeepSeek EP
dispatch/combine reused verbatim + the fused unified_routed_expert_ffn kernel with M3's clamped
swigluoai activation) across the mesh. Needs multi-device + fabric. The single-device / non-EP
expert backends were removed in the prefill cleanup; this mirrors deepseek_v3_d_p's EP-only MoE.
"""

import ttnn
from models.demos.minimax_m3.utils.general_utils import get_cache_file_name
from models.demos.minimax_m3.utils.profiler_utils import FINE, zone
from models.demos.minimax_m3.utils.substate import substate

from .attention.operations import assert_sharded_residual_unpadded
from .dense_mlp import DenseMLP
from .residual import use_sharded_residual
from .topk import TopKRouter


def _make_cache_subdir(tensor_cache_path, name):
    """Create (if needed) and return a Path subdir of the layer's weight cache for EP / composite
    expert weights (they do `path / cache_name`). Returns None when no cache path is configured.

    Raises a clear, actionable error if the cache dir is not writable — typically the shared weight
    cache is owned by another user (read-only), so building a NOT-yet-cached set of weights (e.g. the
    composite MoE on its first run) fails. The fix is to point TT_CACHE_PATH at a directory you own."""
    if not tensor_cache_path:
        return None
    from pathlib import Path

    d = Path(str(tensor_cache_path)) / name
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise PermissionError(
            f"Cannot create weight-cache dir {d}: {e}. The cache directory is not writable (often "
            f"because it is owned by another user). Set TT_CACHE_PATH to a directory you own to "
            f"populate the cache there — optionally seed it from the existing cache first with "
            f"`cp -a --reflink=auto '<existing tensor_cache_...>' \"$TT_CACHE_PATH\"/` to avoid "
            f"re-tilizing the shared weights."
        ) from e
    return d


def _ep_cache_dir(tensor_cache_path):
    """Cache dir for the DeepSeek-style EP sub-modules (gate / routed_expert)."""
    return _make_cache_subdir(tensor_cache_path, "experts_ep")


class MLP:
    """Router + shared expert + expert-parallel routed experts (EP MoE)."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
        expert_weight_dtype=ttnn.bfloat4_b,
        use_ep_moe=False,
        ep_seq_len_per_chip=1024,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl = ccl_manager
        # Residual-stream layout (tt/residual.py). Sharded => this block CONSUMES full emb (the layer's
        # single pre-MLP all-gather, shared with the router and the shared expert) and RETURNS emb/tp:
        # the routed side stops at its reduce-scatter, the shared expert reduce-scatters instead of
        # all-reducing, and the two are added in emb/tp. That removes two of the three all-gathers this
        # block pays under the replicated layout.
        self.sharded_residual = use_sharded_residual() and mesh_config is not None and mesh_config.tp > 1
        # Split state dict. MiniMax's SparseMoeBlock has `gate.weight` (no bias) plus a sibling
        # `e_score_correction_bias` buffer; experts live under `experts.*`.
        router_state_dict = dict(substate(state_dict, "gate"))
        if state_dict and "e_score_correction_bias" in state_dict:
            router_state_dict["e_score_correction_bias"] = state_dict["e_score_correction_bias"]
        experts_state_dict = substate(state_dict, "experts")

        self.router = TopKRouter(
            mesh_device,
            hf_config,
            router_state_dict,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "router"),
            # Tokens per device per forward — lets the router size the fused gate's wide bias at init.
            num_tokens=ep_seq_len_per_chip,
            mesh_config=mesh_config,
        )

        # Cache-only loading: an empty state_dict means "load every tilized weight from the on-disk
        # cache" (the source bf16 was skipped). Conditional submodules must then be built from the
        # cache rather than skipped, so key their construction off the model config / cache, not off
        # substate presence (which is empty in this mode). See tt/weight_cache.py.
        cache_only = not state_dict

        # M3: always-on shared expert (block_sparse_moe.shared_experts.{gate,up,down}_proj), a plain
        # clamped-swigluoai FFN at shared_intermediate_size. Its output is ADDED to the routed-expert
        # output (the routed side already carries routed_scaling_factor from the router). Reuses DenseMLP.
        # M3 MoE layers always have a shared expert -> build it from cache in cache-only mode.
        shared_state_dict = substate(state_dict, "shared_experts")
        self.shared_expert = (
            DenseMLP(
                mesh_device,
                hf_config,
                shared_state_dict,
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "shared_expert"),
            )
            if (shared_state_dict or cache_only)
            else None
        )

        # Expert-parallel routed experts (TtMiniMaxMoE): expert-parallel, host gate. Bundles its own
        # gate, so we bypass self.router for routing inside it. Needs multi-device + fabric.
        self.use_ep_moe = use_ep_moe and mesh_device.get_num_devices() > 1
        if not self.use_ep_moe:
            raise NotImplementedError(
                "MiniMax-M3 MoE is expert-parallel only (use_ep_moe=True + multi-device). The "
                "single-device / non-EP expert backends were removed in the prefill cleanup."
            )

        from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants, extract_mesh_config

        from .moe.tt_minimax_moe import TtMiniMaxMoE

        mc = extract_mesh_config(mesh_device)
        dgs, ndg = mc.dispatch_group_size, mc.num_dispatch_groups
        E = hf_config.num_local_experts
        # dispatch_buffer_capacity_factor=4 sizes the flat dispatch buffer for the TRUE worst case:
        # all num_experts_per_tok(=4) slots of every column token landing on one chip's experts
        # (4 x dgs x seq rows). With factor 2 the dispatch/combine overflow clamps could DROP rows
        # under extreme routing skew, leaving combine-output slots unwritten-but-readable — the
        # reason init_zeros=True was needed (token-0 NaN hunt 2026-06-29). Factor 4 makes drops
        # structurally impossible, which is what lets TtCombineModule run init_zeros=False (the
        # padding sentinel was always safe: dispatch-table entry 128 is -1). Cost: ~+200 MB DRAM
        # per chip of dispatch-buffer capacity; regions are count-driven so no extra runtime work.
        experts_per_chip, metadata_len, max_buf, max_tok = compute_constants(
            ep_seq_len_per_chip, E, hf_config.num_experts_per_tok, mesh_device.get_num_devices(), dgs, 4
        )
        # MiniMax experts: w1=gate, w3=up, w2=down (direct map, no transpose). None in cache-only mode —
        # TtRoutedExpert then loads the tilized per-expert weights straight from the cache.
        routed_w = (
            None
            if cache_only
            else [
                {
                    "gate_proj": experts_state_dict[f"{e}.w1.weight"],
                    "up_proj": experts_state_dict[f"{e}.w3.weight"],
                    "down_proj": experts_state_dict[f"{e}.w2.weight"],
                }
                for e in range(E)
            ]
        )
        # The MoE's closing TP reduce-scatter. Routed through MeshConfig so it uses
        # reduce_scatter_minimal_async with the same ping-pong + barrier semaphores as every other M3
        # collective, instead of the plain `ttnn.reduce_scatter` prim (which carries no barrier
        # semaphore), so every M3 collective goes through one managed path.
        # Contract: reduce over the TP axis, scatter on the last dim.
        # dim is resolved from the tensor's rank rather than passed as -1: MeshConfig.reduce_scatter
        # forwards straight to reduce_scatter_minimal_async, which (unlike the ttnn.reduce_scatter
        # wrapper it replaces) does not normalize a negative dim.
        moe_reduce_scatter = None
        if mesh_config is not None and ccl_manager is not None and mesh_config.tp > 1:
            # Same guard attention's apply_reduce_scatter runs: a non-tile-aligned hidden/tp would land
            # output-dim padding inside one TP column's residual slice after the scatter.
            assert_sharded_residual_unpadded(mesh_config, hf_config.hidden_size)
            moe_reduce_scatter = lambda t: mesh_config.reduce_scatter(  # noqa: E731
                t, ccl_manager, dim=len(t.shape) - 1, axis=mesh_config.tp_axis
            )

        # Routed experts: DeepSeek EP dispatch/combine + the fused unified_routed_expert_moe kernel with
        # M3's clamped swigluoai activation (baked alpha=1.702 / limit=7.0). See TtMiniMaxMoE.
        self.experts = TtMiniMaxMoE(
            mesh_device=mesh_device,
            dispatch_group_size=dgs,
            num_dispatch_groups=ndg,
            experts_per_chip=experts_per_chip,
            num_routed_experts=E,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_tok,
            max_dispatch_buffer_token_size=max_buf,
            seq_len_per_chip=ep_seq_len_per_chip,
            emb_dim=hf_config.hidden_size,
            hidden_dim=hf_config.intermediate_size,
            gate_weights=router_state_dict,
            routed_expert_weights=routed_w,
            num_links=ccl_manager.num_links,
            routed_expert_weights_dtype=expert_weight_dtype,
            weight_cache_path=_ep_cache_dir(tensor_cache_path),
            # Must match the model's routed_scaling_factor (2.0), not the 1.0 default: the internal gate
            # applies it to the top-k weights, so a stale 1.0 silently halves every routed contribution
            # as soon as gate_fallback_mode selects the internal gate over the caller-supplied topk.
            route_scale=getattr(hf_config, "routed_scaling_factor", 1.0),
            reduce_scatter_fn=moe_reduce_scatter,
        )
        self.ep_num_links = ccl_manager.num_links

    def __call__(self, hidden_states, actual_isl=None):
        """Forward (prefill): shared expert + expert-parallel routed experts.

        actual_isl: real (non-pad) tokens in this chunk across the whole SP axis, or None for a full
        chunk. Drives the padding config below; a wrong value silently drops real tokens, so a caller
        that does not track it must pass None (correct, it just does the padded work).

        hidden_states: per-device [1,1,S,H] at FULL emb (the prompts/seq-shards live in the mesh rows).
        Under a sharded residual that full width comes from the layer's single pre-MLP all-gather, and
        all three consumers here — the router, the shared expert and the EP dispatch — read that one
        tensor. The EP dispatch reads rows via cluster_axis=0; the router runs per-row (each row routes
        its own tokens).

        Returns emb/tp under a sharded residual (routed reduce-scatter + shared reduce-scatter, added),
        or full emb under the replicated one (the routed output is all-gathered back and the shared
        expert all-reduced), matching the layer residual either way.
        """
        with zone("shared_expert"):
            shared_out = self.shared_expert(hidden_states) if self.shared_expert is not None else None

        Hfull = hidden_states.shape[-1]
        # ONE padding config per chunk, shared by the gate and the EP dispatch. Built (and memoized) by
        # the router; None for a full chunk. Both consumers must see the SAME tensor — the gate
        # sentinel-marks the padded rows and dispatch shortens its token loop to match. See tt/topk.py.
        padding_config = self.router.build_padding_config(actual_isl)
        with zone("router_topk"):
            idx, wts = self.router(hidden_states, padding_config=padding_config)  # per-row top-k
        x3d = ttnn.squeeze(hidden_states, dim=0)  # [1,1,S,H] -> [1,S,H] per device
        out = self.experts(
            x3d, topk_indices=idx, topk_weights=wts, padding_config=padding_config
        )  # -> [1,S,H/tp] reduce-scattered
        out = ttnn.unsqueeze(out, dim=0)  # -> [1,1,S,H/tp]
        if not self.sharded_residual and self.mesh_device.shape[1] > 1 and out.shape[-1] < Hfull:
            # TP all-gather (reduce-scattered emb -> full emb). Use the MANAGED all_gather_async
            # (mesh_config.allgather, semaphore/barrier-managed — the path DeepSeek's MoE uses) instead of
            # the raw ttnn.all_gather: the raw op left a stale tile-face on a non-device-0 TP column's
            # slice under the full-model footprint -> ~1e38 garbage -> token-0 (token-0 hunt 2026-06-29).
            with zone("tp_allgather"):
                if self.mesh_config is not None and self.ccl is not None:
                    out = self.mesh_config.allgather(out, self.ccl, axis=1, dim=3)
                else:
                    out = ttnn.all_gather(
                        out, dim=-1, cluster_axis=1, num_links=self.ep_num_links, topology=ttnn.Topology.Linear
                    )
        if shared_out is not None:
            with zone("add_shared", FINE):
                out = ttnn.add(out, shared_out)
            shared_out.deallocate(True)
        return out
