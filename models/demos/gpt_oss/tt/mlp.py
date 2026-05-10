# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MoE MLP: Router + Experts with minimal abstraction
"""
import ttnn
from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate

from .experts import ExpertConfig, Experts
from .experts_throughput import (
    DeepSeekPrefillConfig,
    ThroughputExpertConfig,
    ThroughputExperts,
    create_fused_moe_gpt_config,
)
from .topk import TopKRouter


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
    ):
        # Split state dict
        router_state_dict = substate(state_dict, "router")
        experts_state_dict = substate(state_dict, "experts")

        # Initialize components with mesh_config
        self.router = TopKRouter(
            mesh_device,
            hf_config,
            router_state_dict,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "router"),
        )

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
                    num_links=4,
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
                swiglu_limit=hf_config.swiglu_limit,
            )

            # Use GPT-OSS specific program config
            program_config = GPTOSSProgramConfig()

            # Create experts with new modular implementation
            self.experts = Experts(
                mesh_device=mesh_device,
                config=expert_config,
                state_dict=experts_state_dict,
                ccl_manager=ccl_manager,
                mesh_config=mesh_config,
                program_config=program_config,
                weight_dtype=ttnn.bfloat4_b,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "experts"),
            )

    def __call__(self, hidden_states, is_decode):
        """Forward pass: route -> experts
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
        Returns:
            Expert output tensor [batch, seq_len, hidden_size]
        """
        expert_indices, expert_weights = self.router(hidden_states, self.use_throughput_experts)
        expert_output = self.experts(
            hidden_states, topk_expert_indices=expert_indices, topk_expert_weights=expert_weights, is_decode=is_decode
        )
        return expert_output
