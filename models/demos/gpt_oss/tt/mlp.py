# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MoE MLP: Router + Experts with minimal abstraction
"""
import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.experimental.stable_diffusion_35_large.tt.substate import substate

from .experts import Experts
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
        self.experts = Experts(
            mesh_device,
            hf_config,
            experts_state_dict,
            ccl_manager,
            dtype=dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "experts"),
            mesh_config=mesh_config,
        )

    def __call__(self, hidden_states):
        """Forward pass: route -> experts"""
        router_scores, router_indices, router_logits = self.router(hidden_states)
        expert_output = self.experts(hidden_states, router_scores)
        return expert_output, router_scores
