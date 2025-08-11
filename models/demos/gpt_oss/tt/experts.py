import ttnn


class Experts:
    def __init__(self, config, state_dict, mesh_device):
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        gate_proj = state_dict["gate_up_proj"][..., ::2].reshape(self.num_experts, 1, self.hidden_size, self.expert_dim)
        up_proj = state_dict["gate_up_proj"][..., 1::2].reshape(self.num_experts, 1, self.hidden_size, self.expert_dim)
        gate_proj_bias = state_dict["gate_up_proj_bias"][..., ::2].reshape(self.num_experts, 1, 1, self.expert_dim)
        up_proj_bias = state_dict["gate_up_proj_bias"][..., 1::2].reshape(self.num_experts, 1, 1, self.expert_dim)

        self.gate_proj = ttnn.from_torch(gate_proj, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        self.up_proj = ttnn.from_torch(up_proj, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        self.gate_proj_bias = ttnn.from_torch(
            gate_proj_bias, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        self.up_proj_bias = ttnn.from_torch(
            up_proj_bias, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

        down_proj = state_dict["down_proj"].reshape(self.num_experts, 1, self.expert_dim, self.hidden_size)
        down_proj_bias = state_dict["down_proj_bias"].reshape(self.num_experts, 1, 1, self.hidden_size)
        self.down_proj = ttnn.from_torch(down_proj, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        self.down_proj_bias = ttnn.from_torch(
            down_proj_bias, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

        self.alpha = 1.702
        self.limit = 7.0

    def __call__(self, hidden_states, routing_weights):
        """
        When training it is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
        Returns:
            torch.Tensor
        """
        batch_size = hidden_states.shape[0]
        assert batch_size == 1, "batch_size must be 1"
        seq_len = hidden_states.shape[1]
        hidden_states = ttnn.reshape(
            hidden_states, (1, batch_size, seq_len, self.hidden_size)
        )  # unsqueeze a dim for expert broadcast
        hidden_states = ttnn.repeat(hidden_states, repeat_dims=(self.num_experts, 1, 1, 1))

        gate = ttnn.matmul(hidden_states, self.gate_proj) + self.gate_proj_bias
        up = ttnn.matmul(hidden_states, self.up_proj) + self.up_proj_bias

        gate = ttnn.clamp(gate, min=None, max=self.limit)
        up = ttnn.clamp(up, min=-self.limit, max=self.limit)
        glu = gate * ttnn.sigmoid(gate * self.alpha)
        next_states = ttnn.matmul(((up + 1) * glu), self.down_proj) + self.down_proj_bias
        routing_weights = ttnn.permute(routing_weights, (1, 0))
        routing_weights = ttnn.reshape(routing_weights, (self.num_experts, batch_size, seq_len, 1))

        next_states = next_states * routing_weights
        next_states = ttnn.sum(next_states, dim=0)
        return next_states
