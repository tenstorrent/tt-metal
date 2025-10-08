import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.experimental.stable_diffusion_35_large.tt.substate import substate

from .experts import Experts
from .topk import TopKRouter


class MLP:
    def __init__(self, mesh_device, hf_config, state_dict, ccl_manager, dtype=ttnn.bfloat16, tensor_cache_path=None):
        router_state_dict = substate(state_dict, "router")
        experts_state_dict = substate(state_dict, "experts")

        # Initialize router and experts
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
        )

    def __call__(self, hidden_states):
        # Get router outputs
        router_scores, router_indices, router_logits = self.router(hidden_states)

        # Pass through experts with routing weights
        expert_output = self.experts(hidden_states, router_scores)

        return expert_output, router_scores
