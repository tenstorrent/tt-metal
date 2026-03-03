# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Reference implementation of the complete MoE block for DeepSeek-V3.
This combines the gate and experts into a single module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_gate import ReferenceMoEGate
from .routed_experts import SimplifiedRoutedExperts, SingleExpert

# ACT2FN mapping for activation functions
ACT2FN = {
    "silu": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
}


class ReferenceMoEBlock(nn.Module):
    """
    Reference implementation of the complete MoE block.
    This is adapted from DeepseekV3MoE but simplified for testing.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.training = False  # Always inference mode for testing

        # Create the gate
        self.gate = ReferenceMoEGate(config)

        # Create the experts
        self.experts = nn.ModuleList(
            [
                SingleExpert(config, hidden_size=self.hidden_size, intermediate_size=self.moe_intermediate_size).eval()
                for i in range(self.n_routed_experts)
            ]
        )

        # Optional shared experts (not used in our tests)
        if hasattr(config, "n_shared_experts") and config.n_shared_experts is not None:
            intermediate_size = self.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = SingleExpert(
                config, hidden_size=self.hidden_size, intermediate_size=intermediate_size
            ).eval()
        else:
            self.shared_experts = None

    def forward(self, hidden_states):
        """
        Forward pass of the MoE block.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
                          or (batch_size, 1, seq_len, hidden_size)

        Returns:
            Output tensor of same shape as input
        """
        # Handle 4D input
        if hidden_states.dim() == 4:
            batch_size, _, seq_len, hidden_size = hidden_states.shape
            hidden_states = hidden_states.squeeze(1)  # Remove the extra dimension
            had_4d_input = True
        else:
            batch_size, seq_len, hidden_size = hidden_states.shape
            had_4d_input = False

        identity = hidden_states
        orig_shape = hidden_states.shape

        # Get routing decisions from the gate
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
        sorted_tokens_shape = sorted_tokens.shape

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

        # Create output tensor
        output = torch.zeros_like(x)

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


class SimplifiedMoEBlock(nn.Module):
    """
    Simplified MoE block for testing that processes all experts uniformly.
    This version is easier to debug and compare with TTNN implementation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        # Create the gate
        self.gate = ReferenceMoEGate(config)

        # Create simplified routed experts (processes all experts)
        self.experts = SimplifiedRoutedExperts(config).eval()

        # No shared experts for simplified version
        self.shared_experts = None

    def forward(self, hidden_states):
        """
        Forward pass of the simplified MoE block.
        This version processes input through all experts and then combines based on routing.

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

        # Get routing decisions from the gate
        topk_idx, topk_weight = self.gate(hidden_states)  # (batch_size*seq_len, num_experts_per_tok)

        # Process all tokens through all experts
        # Shape: (batch_size, seq_len, hidden_size) -> (n_experts, batch_size, seq_len, hidden_size)
        all_expert_outputs = self.experts(hidden_states.unsqueeze(0))

        # Reshape for gathering
        batch_seq_len = batch_size * seq_len
        all_expert_outputs = all_expert_outputs.permute(1, 2, 0, 3)  # (batch, seq, experts, hidden)
        all_expert_outputs = all_expert_outputs.reshape(batch_seq_len, self.n_routed_experts, hidden_size)

        # Gather outputs from selected experts
        gathered = torch.zeros(
            batch_seq_len, self.num_experts_per_tok, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for i in range(batch_seq_len):
            for j in range(self.num_experts_per_tok):
                expert_id = topk_idx[i, j]
                gathered[i, j] = all_expert_outputs[i, expert_id]

        # Apply weights and sum
        weighted = gathered * topk_weight.unsqueeze(-1)
        output = weighted.sum(dim=1)  # (batch_seq_len, hidden_size)

        # Reshape back
        output = output.view(batch_size, seq_len, hidden_size)

        # Restore 4D shape if needed
        if had_4d_input:
            output = output.unsqueeze(1)

        return output
