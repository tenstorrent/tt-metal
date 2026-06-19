# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MoE MLP: Router + Experts with minimal abstraction
"""
import ttnn
from models.demos.minimax_m3.tt.expert_configs import MiniMaxM3ExpertProgramConfig
from models.demos.minimax_m3.utils.general_utils import get_cache_file_name
from models.demos.minimax_m3.utils.substate import substate

from .experts import ExpertConfig, Experts
from .experts_throughput import (
    DeepSeekPrefillConfig,
    ThroughputExpertConfig,
    ThroughputExperts,
    create_fused_moe_gpt_config,
)
from .topk import TopKRouter


def _ep_cache_dir(tensor_cache_path):
    """EP sub-modules (gate/routed_expert) want a Path *directory* for weight caching
    (they do `path / name`). Return a Path dir under the layer's cache, or None."""
    if not tensor_cache_path:
        return None
    from pathlib import Path

    d = Path(str(tensor_cache_path)) / "experts_ep"
    d.mkdir(parents=True, exist_ok=True)
    return d


class MLP:
    """Streamlined MoE MLP combining router and experts"""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
        use_throughput_experts=True,
        tokens_per_device=32,
        expert_weight_dtype=ttnn.bfloat4_b,
        use_ep_moe=False,
        ep_seq_len_per_chip=1024,
    ):
        self.mesh_device = mesh_device
        # Split state dict. MiniMax-M2's SparseMoeBlock has `gate.weight` (no bias) plus
        # a sibling `e_score_correction_bias` buffer; experts live under `experts.*`.
        router_state_dict = dict(substate(state_dict, "gate"))
        if state_dict and "e_score_correction_bias" in state_dict:
            router_state_dict["e_score_correction_bias"] = state_dict["e_score_correction_bias"]
        experts_state_dict = substate(state_dict, "experts")

        # Initialize components with mesh_config
        self.router = TopKRouter(
            mesh_device,
            hf_config,
            router_state_dict,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "router"),
        )

        # Throughput experts rely on all_to_all_dispatch/combine across a mesh axis,
        # which has no meaning on a single device and would require fabric.
        if use_throughput_experts and mesh_device.get_num_devices() == 1:
            use_throughput_experts = False

        # EP MoE path (validated TtMiniMaxMoE): expert-parallel, no shared expert, host gate.
        # Bundles its own gate, so we bypass self.router. Needs multi-device + fabric.
        self.use_ep_moe = use_ep_moe and mesh_device.get_num_devices() > 1
        if self.use_ep_moe:
            from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants, extract_mesh_config

            from .experts_throughput.tt_minimax_moe import TtMiniMaxMoE

            mc = extract_mesh_config(mesh_device)
            dgs, ndg = mc.dispatch_group_size, mc.num_dispatch_groups
            E = hf_config.num_local_experts
            experts_per_chip, metadata_len, max_buf, max_tok = compute_constants(
                ep_seq_len_per_chip, E, hf_config.num_experts_per_tok, mesh_device.get_num_devices(), dgs, 2
            )
            # MiniMax experts: w1=gate, w3=up, w2=down (direct map, no transpose).
            routed_w = [
                {
                    "gate_proj": experts_state_dict[f"{e}.w1.weight"],
                    "up_proj": experts_state_dict[f"{e}.w3.weight"],
                    "down_proj": experts_state_dict[f"{e}.w2.weight"],
                }
                for e in range(E)
            ]
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
            self.use_throughput_experts = False
            self.ep_dgs = dgs
            self.ep_num_links = ccl_manager.num_links
            return
        self.use_throughput_experts = use_throughput_experts
        if self.use_throughput_experts:
            # Create TT config
            throughput_expert_config = ThroughputExpertConfig(
                intermediate_size=hf_config.intermediate_size,
                num_experts=hf_config.num_local_experts,
                hidden_size=hf_config.hidden_size,
                num_experts_per_tok=hf_config.num_experts_per_tok,
                num_devices=mesh_device.get_num_devices(),
            )

            # Create fused MoE config if requested
            fused_config = None
            if use_throughput_experts:
                fused_config = create_fused_moe_gpt_config(
                    mesh_device=mesh_device,
                    config=throughput_expert_config,
                    state_dict=experts_state_dict,
                    tokens_per_device=tokens_per_device,
                    num_links=ccl_manager.num_links,
                    tensor_cache_path=get_cache_file_name(tensor_cache_path, "experts"),
                )

            # DeepSeek prefill config: always created when throughput experts are
            # enabled. The two were previously gated by separate flags but were always
            # set together by every caller; now bundled.
            prefill_config = None
            deepseek_permuted_weights = None
            if use_throughput_experts:
                import torch as _torch

                from .experts_throughput.prefill import _compute_weight_permutation

                prefill_config = DeepSeekPrefillConfig(
                    mesh_device=mesh_device,
                    config=throughput_expert_config,
                    dispatch_group_size=mesh_device.shape[0],
                    num_dispatch_groups=mesh_device.shape[1],
                    capacity_factor=2.0,
                    seq_len_per_chip=1024,
                    num_links=ccl_manager.num_links,
                )
                # Permute expert state_dict to GROUP-BASED ordering before loading
                perm = _compute_weight_permutation(
                    mesh_device.shape[0],
                    mesh_device.shape[1],
                    throughput_expert_config.num_experts // (mesh_device.shape[0] * mesh_device.shape[1]),
                )
                perm_t = _torch.tensor(perm, dtype=_torch.long)
                permuted_sd = {
                    k: v.index_select(0, perm_t) if v.shape[0] == throughput_expert_config.num_experts else v
                    for k, v in experts_state_dict.items()
                }
                from .experts_throughput.weights import load_throughput_expert_weights

                deepseek_permuted_weights = load_throughput_expert_weights(
                    mesh_device=mesh_device,
                    config=throughput_expert_config,
                    state_dict=permuted_sd,
                    weight_dtype=ttnn.bfloat4_b,
                    tensor_cache_path=get_cache_file_name(tensor_cache_path, "experts_ds_perm"),
                )
                prefill_config.permuted_weights = deepseek_permuted_weights

            # Create TT experts module
            self.experts = ThroughputExperts(
                mesh_device=mesh_device,
                config=throughput_expert_config,
                state_dict=experts_state_dict,
                weight_dtype=ttnn.bfloat4_b,
                dispatch_cluster_axis=0,
                decode_memory_config=ttnn.L1_MEMORY_CONFIG,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "experts"),
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                fused_config=fused_config,
                prefill_config=prefill_config,
            )
        else:
            # Create expert config from HF config
            expert_config = ExpertConfig(
                intermediate_size=hf_config.intermediate_size,
                num_experts=hf_config.num_local_experts,
                hidden_size=hf_config.hidden_size,
                num_experts_per_tok=hf_config.num_experts_per_tok,
                swiglu_limit=getattr(hf_config, "swiglu_limit", 7.0),
                alpha=getattr(hf_config, "swiglu_alpha", 1.702),
            )

            # Use MiniMax-M2 specific program config
            program_config = MiniMaxM3ExpertProgramConfig()

            # Create experts with new modular implementation
            self.experts = Experts(
                mesh_device=mesh_device,
                config=expert_config,
                state_dict=experts_state_dict,
                ccl_manager=ccl_manager,
                mesh_config=mesh_config,
                program_config=program_config,
                weight_dtype=expert_weight_dtype,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "experts"),
            )

    def __call__(self, hidden_states):
        """Forward pass: route -> experts (prefill)
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
        Returns:
            Expert output tensor [batch, seq_len, hidden_size]
        """
        if getattr(self, "use_ep_moe", False):
            # On-device EP bridge (DeepSeek `_moe_path` pattern, fully traceable — no host hop):
            # the decoder hands us per-device [1,1,S,H] (the R prompts/seq-shards live in the MESH
            # rows). The EP dispatch reads rows via cluster_axis=0, so we just SQUEEZE the leading
            # dim to per-device [1,S,H]; the router runs per-row (each row routes its own tokens).
            # The MoE returns reduce-scattered emb/tp; all-gather it back to full emb so it matches
            # the layer residual's [1,1,S,H] (the EP=1 path returns full emb too).
            Hfull = hidden_states.shape[-1]
            idx, wts = self.router(hidden_states, True)  # per-row top-k on [1,1,S,H]
            x3d = ttnn.squeeze(hidden_states, dim=0)  # [1,1,S,H] -> [1,S,H] per device
            out = self.experts(x3d, topk_indices=idx, topk_weights=wts)  # -> [1,S,H/tp] reduce-scattered
            out = ttnn.unsqueeze(out, dim=0)  # -> [1,1,S,H/tp]
            if self.mesh_device.shape[1] > 1 and out.shape[-1] < Hfull:
                out = ttnn.all_gather(
                    out, dim=-1, cluster_axis=1, num_links=self.ep_num_links, topology=ttnn.Topology.Linear
                )
            return out

        expert_indices, expert_weights = self.router(hidden_states, self.use_throughput_experts)
        expert_output = self.experts(
            hidden_states, topk_expert_indices=expert_indices, topk_expert_weights=expert_weights
        )
        return expert_output
