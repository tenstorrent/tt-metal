# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Reference implementation of routed experts for DeepSeek-V3 MoE.

This implementation matches the DeepseekV3MoEExperts used in the original TTNN tests.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ACT2FN mapping for activation functions
ACT2FN = {
    "silu": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
}


class SingleExpert(nn.Module):
    """
    Single expert MLP module matching DeepseekV3MLP.

    This is a single expert that performs:
    output = down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.moe_intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act] if hasattr(config, "hidden_act") else F.silu

    def forward(self, x):
        """
        Forward pass of a single expert.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size) or (seq_len, hidden_size)

        Returns:
            Output tensor of same shape as input
        """
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class SimplifiedRoutedExperts(nn.Module):
    """
    Reference implementation matching DeepseekV3MoEExperts from test_moe_experts.py.

    This contains multiple expert MLPs and processes them individually.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        # Create a list of experts like DeepseekV3MoEExperts
        self.experts = nn.ModuleList(
            [
                SingleExpert(config, hidden_size=self.hidden_size, intermediate_size=self.moe_intermediate_size).eval()
                for i in range(self.n_routed_experts)
            ]
        )

        # For backward compatibility, create stacked weight tensors
        # These will be views into the expert weights
        self.w1_weight = None
        self.w2_weight = None
        self.w3_weight = None
        self._setup_stacked_weights()

    def _setup_stacked_weights(self):
        """Setup stacked weight tensors for backward compatibility."""
        # Create stacked weight tensors
        w1_list = []
        w2_list = []
        w3_list = []

        for expert in self.experts:
            w1_list.append(expert.gate_proj.weight.data)
            w2_list.append(expert.down_proj.weight.data.t())  # Transpose to match expected shape
            w3_list.append(expert.up_proj.weight.data)

        self.w1_weight = nn.Parameter(torch.stack(w1_list))
        self.w2_weight = nn.Parameter(torch.stack(w2_list))
        self.w3_weight = nn.Parameter(torch.stack(w3_list))

    def reset_parameters(self):
        """Initialize weights using Kaiming uniform."""
        for expert in self.experts:
            for module in [expert.gate_proj, expert.up_proj, expert.down_proj]:
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        self._setup_stacked_weights()

    def forward(self, hidden_states):
        """
        Forward pass matching DeepseekV3MoEExperts.

        Args:
            hidden_states: Input tensor of shape (batch_size, num_experts, seq_len, hidden_size)
                          or (num_experts, seq_len, hidden_size) or (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor - if input has experts dimension, returns same shape
                           otherwise returns concatenated expert outputs
        """
        # Handle different input shapes
        if hidden_states.dim() == 4:
            # Check if this is actually (batch_size, 1, seq_len, hidden_size) for all experts
            batch_size, dim1, seq_len, hidden_size = hidden_states.shape

            if dim1 == 1:
                # This is the test_moe_experts.py case: input for all experts
                # Process through all experts and concatenate
                outputs = []
                for expert in self.experts:
                    outputs.append(expert(hidden_states.squeeze(1)))

                # Return shape: (num_experts, 1, seq_len, hidden_size)
                output = torch.stack(outputs, dim=0)
                return output
            else:
                # Shape: (batch_size, num_experts, seq_len, hidden_size)
                # Process each expert with its corresponding input
                outputs = []
                for i in range(dim1):
                    expert_input = hidden_states[:, i, :, :]  # (batch_size, seq_len, hidden_size)
                    expert_output = self.experts[i](expert_input)
                    outputs.append(expert_output)

                # Stack outputs back
                output = torch.stack(outputs, dim=1)  # (batch_size, num_experts, seq_len, hidden_size)
                return output

        elif hidden_states.dim() == 3:
            # Could be (num_experts, seq_len, hidden_size) or (batch_size, seq_len, hidden_size)
            if hidden_states.shape[0] == self.n_routed_experts:
                # Shape: (num_experts, seq_len, hidden_size)
                num_experts, seq_len, hidden_size = hidden_states.shape

                outputs = []
                for i in range(num_experts):
                    expert_input = hidden_states[i : i + 1]  # Keep dimension
                    expert_output = self.experts[i](expert_input)
                    outputs.append(expert_output)

                return torch.cat(outputs, dim=0)
            else:
                # Shape: (batch_size, seq_len, hidden_size) - process through all experts
                outputs = []
                for expert in self.experts:
                    outputs.append(expert(hidden_states))

                return torch.cat(outputs, dim=0)
        else:
            # Shape: (seq_len, hidden_size) - process through all experts
            outputs = []
            for expert in self.experts:
                outputs.append(expert(hidden_states))

            return torch.cat(outputs, dim=0)

    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict, supporting both individual expert weights and stacked weights.
        """
        # Filter out the w1_weight, w2_weight, w3_weight keys if present
        # as they are now created dynamically from expert weights
        filtered_dict = {k: v for k, v in state_dict.items() if k not in ["w1_weight", "w2_weight", "w3_weight"]}

        # Load the filtered state dict
        super().load_state_dict(filtered_dict, strict=False)

        # Update stacked weights
        self._setup_stacked_weights()


# Aliases for backward compatibility
RoutedExpertsReference = SimplifiedRoutedExperts
RoutedExperts = SimplifiedRoutedExperts
