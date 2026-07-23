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
from models.demos.minimax_m3.utils.substate import substate

from .dense_mlp import DenseMLP
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
        experts_per_chip, metadata_len, max_buf, max_tok = compute_constants(
            ep_seq_len_per_chip, E, hf_config.num_experts_per_tok, mesh_device.get_num_devices(), dgs, 2
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
        )
        self.ep_num_links = ccl_manager.num_links

    def __call__(self, hidden_states):
        """Forward (prefill): shared expert + expert-parallel routed experts.

        hidden_states: per-device [1,1,S,H] (the prompts/seq-shards live in the mesh rows). The EP
        dispatch reads rows via cluster_axis=0; the router runs per-row (each row routes its own
        tokens). The MoE returns reduce-scattered emb/tp; we all-gather it back to full emb so it
        matches the layer residual's [1,1,S,H]. The shared expert runs on the same input and is added.
        """
        shared_out = self.shared_expert(hidden_states) if self.shared_expert is not None else None

        Hfull = hidden_states.shape[-1]
        idx, wts = self.router(hidden_states, True)  # per-row top-k on [1,1,S,H]
        x3d = ttnn.squeeze(hidden_states, dim=0)  # [1,1,S,H] -> [1,S,H] per device
        out = self.experts(x3d, topk_indices=idx, topk_weights=wts)  # -> [1,S,H/tp] reduce-scattered
        out = ttnn.unsqueeze(out, dim=0)  # -> [1,1,S,H/tp]
        if self.mesh_device.shape[1] > 1 and out.shape[-1] < Hfull:
            # TP all-gather (reduce-scattered emb -> full emb). Use the MANAGED all_gather_async
            # (mesh_config.allgather, semaphore/barrier-managed — the path DeepSeek's MoE uses) instead of
            # the raw ttnn.all_gather: the raw op left a stale tile-face on a non-device-0 TP column's
            # slice under the full-model footprint -> ~1e38 garbage -> token-0 (token-0 hunt 2026-06-29).
            if self.mesh_config is not None and self.ccl is not None:
                out = self.mesh_config.allgather(out, self.ccl, axis=1, dim=3)
            else:
                out = ttnn.all_gather(
                    out, dim=-1, cluster_axis=1, num_links=self.ep_num_links, topology=ttnn.Topology.Linear
                )
        if shared_out is not None:
            out = ttnn.add(out, shared_out)
            shared_out.deallocate(True)
        return out
