# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Routed Experts module.

128 experts, top-8 routing, GeGLU activation (no bias).
moe_intermediate_size = 704 per expert.

HF weight shapes (fused):
  gate_up_proj: [128, 1408, 2816]  (1408 = 2 * 704, gate and up fused)
  down_proj: [128, 2816, 704]

Forward per expert:
  gate, up = linear(x, gate_up_proj[e]).chunk(2)
  output = linear(gelu(gate) * up, down_proj[e])
"""

import torch
import torch.nn.functional as F

import ttnn
from models.demos.gemma4.config import MeshConfig, Mode


class Gemma4ExpertConfig:
    """Configuration for the routed experts, derived from HF config."""

    def __init__(self, hf_config):
        self.hidden_size = hf_config.hidden_size  # 2816
        self.num_experts = hf_config.num_experts  # 128
        self.top_k = hf_config.top_k_experts  # 8
        self.moe_intermediate_size = hf_config.moe_intermediate_size  # 704


class Gemma4Experts:
    def __init__(
        self,
        mesh_device,
        config,
        state_dict,
        ccl_manager,
        mesh_config,
        program_config,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config

        # Load fused gate_up_proj and down_proj as torch tensors
        # These are 3D: [num_experts, intermediate, hidden] and [num_experts, hidden, intermediate]
        # We keep them on CPU and index per-expert during forward
        if state_dict and "gate_up_proj" in state_dict:
            self.gate_up_proj = state_dict["gate_up_proj"]  # [128, 1408, 2816]
            self.down_proj = state_dict["down_proj"]  # [128, 2816, 704]
        else:
            self.gate_up_proj = None
            self.down_proj = None

    def __call__(self, hidden_states, top_k_indices, top_k_weights):
        """
        Routed expert forward pass (CPU-based for correctness, matching HF reference).

        Args:
            hidden_states: torch.Tensor [seq_len, hidden_size] (flattened, on CPU)
            top_k_indices: torch.Tensor [seq_len, top_k] - selected expert indices
            top_k_weights: torch.Tensor [seq_len, top_k] - routing weights

        Returns:
            output: torch.Tensor [seq_len, hidden_size]
        """
        if self.gate_up_proj is None:
            return torch.zeros_like(hidden_states)

        final_hidden_states = torch.zeros_like(hidden_states)

        # Build expert mask to find which experts have tokens
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_indices, num_classes=self.config.num_experts)
            # expert_mask: [seq_len, top_k, num_experts] -> [num_experts, top_k, seq_len]
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.config.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # Fused gate+up: [1408, 2816] @ [tokens, 2816].T -> [tokens, 1408] -> chunk into [tokens, 704] x 2
            gate, up = F.linear(current_state.float(), self.gate_up_proj[expert_idx].float()).chunk(2, dim=-1)
            current_hidden_states = F.gelu(gate, approximate="tanh") * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx].float())

            # Weight by routing weights
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None].float()
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
