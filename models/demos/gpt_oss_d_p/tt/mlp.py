# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS MoE MLP for prefill: top-k router + expert-parallel routed experts.

Thin wrapper around ``TtGptOssRouter`` + ``TtGptOssMoE``. Unlike MiniMax-M3 there is NO shared
expert and NO dense-layer branch — every GPT-OSS layer is a MoE layer. Expert-parallel only (the
deployment path): the routed experts run across the mesh (DeepSeek EP dispatch/combine reused
verbatim + the fused clamped-swigluoai kernel). Needs multi-device + fabric.

State-dict split (HF GptOssMLP): ``router.{weight,bias}`` and ``experts.{gate_up_proj,
gate_up_proj_bias, down_proj, down_proj_bias}``.
"""

from pathlib import Path

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants, extract_mesh_config
from models.demos.gpt_oss_d_p.tt.moe.router import TtGptOssRouter
from models.demos.gpt_oss_d_p.tt.moe.tt_gpt_oss_moe import TtGptOssMoE
from models.demos.gpt_oss_d_p.tt.moe.weights import prepare_routed_expert_weights
from models.demos.gpt_oss_d_p.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss_d_p.utils.substate import substate


def _ep_cache_dir(tensor_cache_path):
    """Create (if needed) and return the EP routed-expert weight-cache subdir, or None."""
    if not tensor_cache_path:
        return None
    d = Path(str(tensor_cache_path)) / "experts_ep"
    d.mkdir(parents=True, exist_ok=True)
    return d


class MLP:
    """Router + expert-parallel routed experts (EP MoE). No shared expert, no dense branch."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        tensor_cache_path=None,
        mesh_config=None,
        expert_weight_dtype=ttnn.bfloat4_b,
        expert_activation_dtype=ttnn.bfloat8_b,
        use_ep_moe=True,
        ep_seq_len_per_chip=1024,
        layer_idx: int = 0,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl = ccl_manager

        # Split state dict: HF GptOssMLP has `router.{weight,bias}` + `experts.*`.
        router_state_dict = substate(state_dict, "router")
        experts_state_dict = substate(state_dict, "experts")

        self.router = TtGptOssRouter(
            mesh_device,
            hf_config,
            router_state_dict,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "router"),
        )

        # GPT-OSS MoE is expert-parallel only (matches deepseek_v3_d_p / minimax_m3 prefill).
        self.use_ep_moe = use_ep_moe and mesh_device.get_num_devices() > 1
        if not self.use_ep_moe:
            raise NotImplementedError("GPT-OSS MoE is expert-parallel only (use_ep_moe=True + multi-device).")

        E = hf_config.num_local_experts
        H = hf_config.hidden_size
        I = hf_config.intermediate_size

        mc = extract_mesh_config(mesh_device)
        dgs, ndg = mc.dispatch_group_size, mc.num_dispatch_groups
        experts_per_chip, metadata_len, max_buf, max_tok = compute_constants(
            ep_seq_len_per_chip, E, hf_config.num_experts_per_tok, mesh_device.get_num_devices(), dgs, 2
        )

        # De-interleave gate_up_proj, transpose to HF (out,in), global expert order 0..E-1.
        # Biases are kept SEPARATE (not attached to the weight dicts) — the current kernel is
        # bias-free (#49619). None in cache-only mode -> TtRoutedExpert loads tilized weights from cache.
        cache_only = not state_dict
        if cache_only:
            routed_w, routed_b = None, None
        else:
            routed_w, routed_b = prepare_routed_expert_weights(experts_state_dict, E, H, I)

        self.experts = TtGptOssMoE(
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
            emb_dim=H,
            hidden_dim=I,
            routed_expert_weights=routed_w,
            routed_expert_biases=routed_b,  # threaded for #49619; NOT passed to TtRoutedExpert yet
            num_links=ccl_manager.num_links,
            routed_expert_activations_dtype=expert_activation_dtype,
            routed_expert_weights_dtype=expert_weight_dtype,
            weight_cache_path=_ep_cache_dir(tensor_cache_path),
            layer_idx=layer_idx,
            use_expert_bias=False,  # TODO(#49619): flip to True once the biased kernel lands
        )
        self.ep_num_links = ccl_manager.num_links

    def __call__(self, hidden_states):
        """Forward (prefill): top-k route, then expert-parallel routed experts.

        hidden_states: per-device [1,1,S,H] (full emb, replicated across TP cols; seq shards on rows).
        Returns [1,1,S,H] (all-gathered back to full emb to match the layer residual).
        """
        Hfull = hidden_states.shape[-1]
        idx, wts = self.router(hidden_states)  # per-row top-k on [1,1,S,H] -> [tokens, k]
        x3d = ttnn.squeeze(hidden_states, dim=0)  # [1,1,S,H] -> [1,S,H] per device
        out = self.experts(x3d, topk_indices=idx, topk_weights=wts)  # -> [1,S,H/tp] reduce-scattered
        out = ttnn.unsqueeze(out, dim=0)  # -> [1,1,S,H/tp]
        if self.mesh_device.shape[1] > 1 and out.shape[-1] < Hfull:
            # TP all-gather (reduce-scattered emb -> full emb). Prefer the managed all_gather (the
            # path DeepSeek/M3 use) over raw ttnn.all_gather to avoid stale-tile garbage.
            if self.mesh_config is not None and self.ccl is not None:
                out = self.mesh_config.allgather(out, self.ccl, axis=1, dim=3)
            else:
                out = ttnn.all_gather(
                    out, dim=-1, cluster_axis=1, num_links=self.ep_num_links, topology=ttnn.Topology.Linear
                )
        return out
