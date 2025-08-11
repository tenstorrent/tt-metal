from .experts import Experts
from .topk import TopKRouter


class MLP:
    def __init__(self, config, state_dict, mesh_device):
        """
        Initialize MLP with TopK router and Experts.

        Args:
            config: Configuration object with MoE parameters
            state_dict: State dictionary containing weights for router and experts
            mesh_device: TTNN device
        """
        # Extract router state dict
        router_state_dict = {"weight": state_dict["router.weight"], "bias": state_dict["router.bias"]}

        # Extract experts state dict
        experts_state_dict = {
            "gate_up_proj": state_dict["experts.gate_up_proj"],
            "gate_up_proj_bias": state_dict["experts.gate_up_proj_bias"],
            "down_proj": state_dict["experts.down_proj"],
            "down_proj_bias": state_dict["experts.down_proj_bias"],
        }

        # Initialize router and experts
        self.router = TopKRouter(config, router_state_dict, mesh_device)
        self.experts = Experts(config, experts_state_dict, mesh_device)

    def __call__(self, hidden_states):
        """
        Forward pass through MLP.

        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)

        Returns:
            tuple: (output_tensor, router_scores)
                - output_tensor: Expert outputs weighted by router scores
                - router_scores: Routing scores for each expert
        """
        # Get router outputs
        router_scores, router_indices, router_logits = self.router(hidden_states)

        # Pass through experts with routing weights
        expert_output = self.experts(hidden_states, router_scores)

        return expert_output, router_scores
