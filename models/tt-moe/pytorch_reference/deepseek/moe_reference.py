# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Reference implementation of DeepseekV3MoE from the original model.
This is used for testing the MoEBlock TTNN implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


class DeepseekV3MLP(nn.Module):
    """MLP module matching DeepSeek-V3's expert structure."""

    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act] if hasattr(config, "hidden_act") else F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoEGate(nn.Module):
    """Gate module for MoE routing."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.topk = config.num_experts_per_tok
        self.n_shared_experts = config.n_shared_experts if hasattr(config, "n_shared_experts") else None
        self.scoring_func = config.scoring_func if hasattr(config, "scoring_func") else "sigmoid"
        self.alpha = config.aux_loss_alpha if hasattr(config, "aux_loss_alpha") else 0.001
        self.seq_aux = config.seq_aux if hasattr(config, "seq_aux") else True
        self.topk_method = config.topk_method if hasattr(config, "topk_method") else "noaux_tc"

        # Routing parameters
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))

        # Add e_score_correction_bias for noaux_tc method
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(torch.empty((self.n_routed_experts)))

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain("linear"))
        if hasattr(self, "e_score_correction_bias"):
            nn.init.zeros_(self.e_score_correction_bias)

    def forward(self, hidden_states):
        """
        Forward pass of the gate.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
                         or (batch_size*seq_len, hidden_size)

        Returns:
            topk_idx: Selected expert indices (num_tokens, num_experts_per_tok)
            topk_weight: Normalized weights for selected experts (num_tokens, num_experts_per_tok)
        """
        # Handle both 3D and 2D inputs
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states_flat = hidden_states.view(-1, hidden_size)
        else:
            hidden_states_flat = hidden_states

        # Compute scores
        scores = F.linear(hidden_states_flat.to(self.weight.dtype), self.weight)

        # Apply softmax normalization
        scores = F.softmax(scores.float(), dim=-1)

        # Select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.topk, dim=-1, sorted=True)

        # Renormalize weights
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        return topk_idx, topk_weight.to(hidden_states_flat.dtype)


class DeepseekV3MoE(nn.Module):
    """
    Reference MoE implementation from DeepSeek-V3.
    This matches the structure expected by MoEBlock.convert_weights().
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.training = False  # Always inference mode for testing

        # Create the gate
        self.gate = MoEGate(config)

        # Create the experts
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                for i in range(self.n_routed_experts)
            ]
        )

        # Optional shared experts
        if hasattr(config, "n_shared_experts") and config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV3MLP(config, intermediate_size=intermediate_size)
        else:
            self.shared_experts = None

    def forward(self, hidden_states):
        """
        Forward pass of the MoE module.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
                          or (batch_size, 1, seq_len, hidden_size)

        Returns:
            Output tensor of same shape as input
        """
        # Handle 4D input
        if hidden_states.dim() == 4:
            batch_size, _, seq_len, hidden_size = hidden_states.shape
            hidden_states = hidden_states.squeeze(1)
            had_4d_input = True
        else:
            batch_size, seq_len, hidden_size = hidden_states.shape
            had_4d_input = False

        identity = hidden_states
        orig_shape = hidden_states.shape

        # Get routing decisions from gate
        topk_idx, topk_weight = self.gate(hidden_states)

        # Flatten for processing
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])

        # Run MoE inference
        if not self.training:
            y = self.moe_infer(hidden_states_flat, topk_idx, topk_weight).view(*orig_shape)
        else:
            raise NotImplementedError("Training mode not supported")

        # Add shared experts if configured
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)

        # Restore 4D shape if needed
        if had_4d_input:
            y = y.unsqueeze(1)

        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        """
        Inference for MoE layer.
        Routes tokens to experts based on topk_ids and combines with topk_weight.

        Args:
            x: Flattened input of shape (num_tokens, hidden_size)
            topk_ids: Expert indices of shape (num_tokens, num_experts_per_tok)
            topk_weight: Expert weights of shape (num_tokens, num_experts_per_tok)

        Returns:
            Output tensor of shape (num_tokens, hidden_size)
        """
        num_tokens, hidden_size = x.shape
        num_experts = len(self.experts)

        # Count tokens per expert
        cnts = topk_ids.new_zeros((topk_ids.shape[0], num_experts))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)

        # Sort tokens by expert assignment
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        # Process tokens through experts
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens_for_expert in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens_for_expert
            if num_tokens_for_expert > 0:
                tokens = sorted_tokens[start_idx:end_idx]
                expert_out = self.experts[i](tokens)
                outputs.append(expert_out)
            else:
                # No tokens for this expert
                outputs.append(torch.zeros(0, hidden_size, device=x.device, dtype=x.dtype))
            start_idx = end_idx

        # Concatenate all expert outputs
        concat_output = torch.cat(outputs, dim=0)

        # Unsort and combine weighted outputs
        sorted_weights = topk_weight.view(-1)[idxs]

        # Scale expert outputs by weights
        scaled_output = concat_output * sorted_weights.unsqueeze(-1)

        # Create inverse indices for unsorting
        _, inv_idxs = idxs.sort()
        unsorted_output = scaled_output[inv_idxs]

        # Reshape and sum across experts dimension
        unsorted_output = unsorted_output.view(topk_ids.shape[0], topk_ids.shape[1], -1)
        output = unsorted_output.sum(dim=1)

        return output
