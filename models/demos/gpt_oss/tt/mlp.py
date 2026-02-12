# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MoE MLP: Router + Experts with minimal abstraction
"""
import ttnn
from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate

from .experts import ExpertConfig, Experts
from .experts_throughput import ThroughputExpertConfig, ThroughputExperts
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

        # TODO: Replace this with a factory method
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

            # Create TT experts module
            self.experts = ThroughputExperts(
                mesh_device=mesh_device,
                config=throughput_expert_config,
                state_dict=experts_state_dict,
                weight_dtype=ttnn.bfloat4_b,
                dispatch_cluster_axis=0,
                decode_memory_config=ttnn.L1_MEMORY_CONFIG,  # L1 for better decode throughput
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "experts"),
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
            hidden_states: Input tensor [1, batch_size, seq_len, hidden_size]
        Returns:
            Expert output tensor [1, batch_size, seq_len, hidden_size]
        """
        original_shape = hidden_states.shape
        batch_size = original_shape[1]

        # For batched prefill, flatten batch*seq into seq dim for router and experts
        if batch_size > 1 and not is_decode:
            seq_len = original_shape[2]
            hidden_size = original_shape[3]
            hidden_states = ttnn.reshape(hidden_states, [1, 1, batch_size * seq_len, hidden_size])

        expert_indices, expert_weights = self.router(hidden_states, self.use_throughput_experts)
        expert_output = self.experts(
            hidden_states, topk_expert_indices=expert_indices, topk_expert_weights=expert_weights, is_decode=is_decode
        )

        # Reshape back to original batched shape
        if batch_size > 1 and not is_decode:
            expert_output = ttnn.reshape(expert_output, original_shape)

        return expert_output
