# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MoE MLP: Router + Experts with minimal abstraction
"""
import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate

from .expert_configs import GPTOSSProgramConfig
from .experts import ExpertConfig, Experts
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

    def __call__(self, hidden_states):
        """Forward pass: route -> experts"""
        router_scores, router_indices, router_logits = self.router(hidden_states)
        router_logits.deallocate()
        router_indices.deallocate()
        expert_output = self.experts(hidden_states, router_scores)
        return expert_output, router_scores
