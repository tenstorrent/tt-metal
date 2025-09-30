# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MLP (Multi-Layer Perceptron) module - feedforward component of transformer"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

import ttnn


@dataclass
class MLPConfig:
    """Configuration for MLP module"""

    hidden_size: int
    intermediate_size: int
    activation: str = "silu"  # Options: "silu", "gelu", "relu"
    dropout: float = 0.0


class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (Feedforward Network) module.

    Standard transformer MLP with two linear transformations and an activation function.
    Supports various activation functions commonly used in transformer models.
    """

    def __init__(
        self,
        config: MLPConfig,
        device: ttnn.Device,
    ):
        super().__init__()
        self.config = config
        self.device = device

        # Weights will be loaded from state dict
        self.w1 = None  # up projection
        self.w2 = None  # down projection
        self.w3 = None  # gate projection (for gated variants like SwiGLU)

        # Select activation function
        self.activation_fn = self._get_activation_fn(config.activation)

    def setup_weights(self, w1: ttnn.Tensor, w2: ttnn.Tensor, w3: Optional[ttnn.Tensor] = None):
        """Set pre-loaded weights for the MLP module"""
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of MLP module.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            output: MLP output of shape [batch_size, seq_len, hidden_size]
        """
        if self.w3 is not None:
            # Gated variant (e.g., SwiGLU)
            return self._forward_gated(x)
        else:
            # Standard MLP
            return self._forward_standard(x)

    def _forward_standard(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Standard MLP: Linear -> Activation -> Linear"""
        # First linear transformation
        hidden = ttnn.matmul(x, self.w1)

        # Apply activation
        hidden = self.activation_fn(hidden)

        # Second linear transformation
        output = ttnn.matmul(hidden, self.w2)

        return output

    def _forward_gated(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Gated MLP variant (e.g., SwiGLU): (Linear -> Activation) * Linear -> Linear"""
        # Gate path
        gate = ttnn.matmul(x, self.w1)
        gate = self.activation_fn(gate)

        # Linear path
        up = ttnn.matmul(x, self.w3)

        # Element-wise multiplication
        hidden = gate * up

        # Down projection
        output = ttnn.matmul(hidden, self.w2)

        return output

    def _get_activation_fn(self, activation: str) -> Callable:
        """Get activation function by name"""
        activation_map = {
            "silu": ttnn.silu,
            "gelu": ttnn.gelu,
            "relu": ttnn.relu,
        }

        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {list(activation_map.keys())}")

        return activation_map[activation]


class MoEMLP(torch.nn.Module):
    """
    Mixture of Experts MLP module.

    Implements sparse MLP where only a subset of experts are activated per token.
    This is a more advanced component that can be used by models like Mixtral.
    """

    def __init__(
        self,
        config: MLPConfig,
        num_experts: int,
        num_experts_per_tok: int,
        device: ttnn.Device,
    ):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.device = device

        # Gate to select experts
        self.gate = None

        # Expert weights
        self.experts_w1 = None
        self.experts_w2 = None
        self.experts_w3 = None

    def setup_weights(
        self,
        gate: ttnn.Tensor,
        experts_w1: ttnn.Tensor,
        experts_w2: ttnn.Tensor,
        experts_w3: Optional[ttnn.Tensor] = None,
    ):
        """Set pre-loaded weights for the MoE MLP module"""
        self.gate = gate
        self.experts_w1 = experts_w1
        self.experts_w2 = experts_w2
        self.experts_w3 = experts_w3

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of MoE MLP module.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            output: MoE MLP output of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape

        # Compute router logits
        router_logits = ttnn.matmul(x, self.gate)  # [batch_size, seq_len, num_experts]

        # Select top-k experts
        routing_weights, selected_experts = ttnn.topk(router_logits, k=self.num_experts_per_tok, dim=-1)

        # Normalize routing weights
        routing_weights = ttnn.softmax(routing_weights, dim=-1)

        # Process tokens through selected experts
        # This is a simplified implementation - actual implementation would
        # batch tokens by expert for efficiency
        output = ttnn.zeros_like(x)

        for expert_idx in range(self.num_experts):
            # Create mask for tokens assigned to this expert
            expert_mask = ttnn.any(selected_experts == expert_idx, dim=-1, keepdim=True)

            if ttnn.any(expert_mask):
                # Get tokens for this expert
                expert_input = x * expert_mask

                # Process through expert
                if self.experts_w3 is not None:
                    # Gated variant
                    gate = ttnn.matmul(expert_input, self.experts_w1[expert_idx])
                    gate = ttnn.silu(gate)
                    up = ttnn.matmul(expert_input, self.experts_w3[expert_idx])
                    hidden = gate * up
                else:
                    # Standard variant
                    hidden = ttnn.matmul(expert_input, self.experts_w1[expert_idx])
                    hidden = ttnn.silu(hidden)

                expert_output = ttnn.matmul(hidden, self.experts_w2[expert_idx])

                # Add weighted expert output
                # Find routing weights for this expert
                expert_weights = ttnn.where(
                    selected_experts == expert_idx, routing_weights, ttnn.zeros_like(routing_weights)
                )
                expert_weight = ttnn.sum(expert_weights, dim=-1, keepdim=True)

                output = output + expert_output * expert_weight

        return output
