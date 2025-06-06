# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DeepSeek Mixture of Experts (MoE) modules."""

import math

import torch
import torch.nn.functional as F
from torch import nn

import ttnn

from .mlp import DeepseekV3MLP


class MoEGate(nn.Module):
    """
    Gating module that selects top-k experts for each token using grouped scoring.

    Pseudo-code:
    ```
    def forward(hidden_states):
        # Input shape: [batch_size, seq_len, hidden_size=7168]
        # Output: (topk_idx, topk_weight)
        #   topk_idx: [batch_size * seq_len, num_experts_per_tok=8]
        #   topk_weight: [batch_size * seq_len, num_experts_per_tok=8]

        # 1. Compute expert scores
        x = hidden_states.view(-1, hidden_size)  # [batch*seq, 7168]
        logits = x @ weight.T  # [batch*seq, n_routed_experts=256]
        scores = sigmoid(logits)  # [batch*seq, 256]

        # 2. Apply grouped top-k selection (noaux_tc method)
        # Add correction bias for load balancing
        scores_corrected = scores + e_score_correction_bias  # [batch*seq, 256]

        # Group experts and select top groups
        # n_group=8, so 256/8=32 experts per group
        group_scores = scores_corrected.view(batch*seq, n_group=8, 32)
        group_scores = topk(group_scores, k=2, dim=-1).sum(dim=-1)  # [batch*seq, 8]

        # Select top-k groups (topk_group=4)
        group_idx = topk(group_scores, k=topk_group=4)  # [batch*seq, 4]

        # Create mask for selected groups
        group_mask = zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)  # [batch*seq, 8]

        # Expand mask to all experts
        expert_mask = group_mask.unsqueeze(-1).expand(...).reshape(batch*seq, 256)

        # Select top-k experts from masked scores
        masked_scores = scores_corrected.masked_fill(~expert_mask, -inf)
        topk_idx = topk(masked_scores, k=num_experts_per_tok=8)  # [batch*seq, 8]

        # Get weights from original scores
        topk_weight = gather(scores, topk_idx)  # [batch*seq, 8]

        # 3. Normalize weights to sum to 1
        topk_weight = topk_weight / sum(topk_weight, dim=-1, keepdim=True)

        # 4. Apply scaling factor
        topk_weight = topk_weight * routed_scaling_factor  # 2.5

        return topk_idx, topk_weight
    ```

    Shape Examples:
    - Prefill: [1, 100, 7168] -> idx: [100, 8], weight: [100, 8]
    - Decode:  [1, 1, 7168] -> idx: [1, 8], weight: [1, 8]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))

        # Since config has topk_method="noaux_tc", we always have this bias
        self.e_score_correction_bias = nn.Parameter(torch.empty((self.n_routed_experts)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)

        # Since config has scoring_func="sigmoid", we only use sigmoid
        scores = logits.sigmoid()

        ### select top-k experts using noaux_tc method (only method in config)
        assert not self.training
        scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )  # [n, n_group]
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(bsz * seq_len, -1)
        )  # [n, e]
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)

        ### norm gate to sum 1 (always true since norm_topk_prob=true and top_k > 1)
        if self.top_k > 1:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor  # must multiply the scaling factor

        return topk_idx, topk_weight


class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.

    Pseudo-code:
    ```
    def forward(hidden_states):
        # Input shape: [batch_size, seq_len, hidden_size=7168]
        # Output shape: [batch_size, seq_len, hidden_size=7168]

        identity = hidden_states  # Save for shared experts

        # 1. Gate computes expert selection
        topk_idx, topk_weight = gate(hidden_states)
        # topk_idx: [batch*seq, 8], topk_weight: [batch*seq, 8]

        # 2. Route tokens to selected experts (moe_infer)
        x = hidden_states.view(-1, hidden_size)  # [batch*seq, 7168]

        # Count tokens per expert
        cnts = zeros(batch*seq, n_routed_experts=256)
        cnts.scatter_(1, topk_idx, 1)  # [batch*seq, 256]
        tokens_per_expert = sum(cnts, dim=0)  # [256]

        # Sort tokens by expert assignment
        idxs = topk_idx.view(-1).argsort()  # [batch*seq*8]
        sorted_tokens = x[idxs // 8]  # [batch*seq*8, 7168]

        # Process each expert's tokens
        outputs = []
        for expert_id in range(256):
            if tokens_per_expert[expert_id] > 0:
                # Get tokens for this expert
                expert_tokens = sorted_tokens[start:end]  # [n_tokens, 7168]

                # Apply expert MLP (intermediate_size=2048)
                expert_out = experts[expert_id](expert_tokens)  # [n_tokens, 7168]
                outputs.append(expert_out)

        # Concatenate all expert outputs
        outs = cat(outputs)  # [batch*seq*8, 7168]

        # Unsort to original order
        new_x[idxs] = outs

        # Weighted sum of expert outputs
        final = new_x.view(batch*seq, 8, 7168) * topk_weight.unsqueeze(-1)
        y = sum(final, dim=1)  # [batch*seq, 7168]
        y = y.view(batch, seq, 7168)

        # 3. Add shared expert output (always active)
        # Shared expert uses n_shared_experts=1, so intermediate_size=2048*1=2048
        shared_out = shared_experts(identity)  # [batch, seq, 7168]
        y = y + shared_out

        return y
    ```

    Shape Examples:
    - Prefill: [1, 100, 7168] -> [1, 100, 7168]
    - Decode:  [1, 1, 7168] -> [1, 1, 7168]

    Notes:
    - n_routed_experts = 256 expert MLPs
    - n_shared_experts = 1 always-active expert
    - num_experts_per_tok = 8 experts selected per token
    - Expert MLPs use intermediate_size = 2048
    - Shared expert also uses intermediate_size = 2048
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        # Since ep_size=1 in config, we don't need expert parallelism
        self.ep_size = 1
        self.experts_per_rank = config.n_routed_experts
        self.ep_rank = 0
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                for i in range(config.n_routed_experts)
            ]
        )
        self.gate = MoEGate(config)

        # Since n_shared_experts=1 in config, we always have shared experts
        intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        self.shared_experts = DeepseekV3MLP(config=config, intermediate_size=intermediate_size)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)

        # Always add shared experts since n_shared_experts is not None
        y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape

        # Since ep_size=1, we don't need the distributed code
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


# Proposed MoE dispatch/recall APIs for TTNN
def ttnn_all_to_all_dispatch(input_tensor, expert_indices, expert_mapping, sparse_token_tensor, expert_metadata_tensor):
    """
    MISSING OP - Proposed API for MoE token dispatch

    Dispatch tokens to experts across devices based on routing decisions.
    Combines all-to-some (token dispatch) and all-to-all (metadata exchange).

    Args:
        input_tensor: [BATCH_SIZE, 1, 1, hidden_size] - tokens to dispatch
        expert_indices: [BATCH_SIZE, 1, 1, num_experts_per_tok] - selected expert IDs
        expert_mapping: [num_devices, 1, n_routed_experts, num_devices] - expert to device mapping
        sparse_token_tensor: Output buffer [num_devices, BATCH_SIZE, 1, hidden_size]
        expert_metadata_tensor: Output buffer [num_devices, 1, BATCH_SIZE, num_experts_per_tok]

    This operation performs:
    1. Routes each token to appropriate devices based on expert selection
    2. Writes tokens to sparse_token_tensor on target devices
    3. Exchanges expert selection metadata between all devices
    """


def ttnn_recall(expert_contrib_tensor, expert_metadata_tensor, expert_mapping):
    """
    MISSING OP - Proposed API for MoE token recall/gathering

    Gather expert outputs back to original token positions.
    Reverse operation of dispatch - collects contributions from all experts.

    Args:
        expert_contrib_tensor: [n_routed_experts, BATCH_SIZE, 1, hidden_size] - expert outputs
        expert_metadata_tensor: [num_devices, 1, BATCH_SIZE, num_experts_per_tok] - routing info
        expert_mapping: [num_devices, 1, n_routed_experts, num_devices] - expert locations

    Returns:
        recalled_experts_tensor: [num_experts_per_tok, BATCH_SIZE, 1, hidden_size]
    """


class MoEGateTTNN(nn.Module):
    """
    TTNN implementation of gating module that selects top-k experts using grouped scoring.

    Pseudo-code:
    ```
    def forward(hidden_states):
        # Input shape: [1, 1, BATCH_SIZE*SEQ_LEN, hidden_size=7168]
        # Output: (topk_idx, topk_weight)
        #   topk_idx: [BATCH_SIZE*SEQ_LEN, num_experts_per_tok=8]
        #   topk_weight: [BATCH_SIZE*SEQ_LEN, num_experts_per_tok=8]

        # 1. Compute expert scores
        logits = ttnn.linear(hidden_states, weight)  # [1, 1, batch*seq, n_routed_experts=256]
        scores = ttnn.sigmoid(logits)  # [1, 1, batch*seq, 256]

        # 2. Apply grouped top-k selection (noaux_tc method)
        # Add correction bias for load balancing
        scores_corrected = ttnn.add(scores, e_score_correction_bias)  # [1, 1, batch*seq, 256]

        # Reshape for grouped scoring
        scores_grouped = ttnn.reshape(scores_corrected, [1, 1, batch*seq, n_group=8, 32])

        # Top-2 per group and sum
        group_topk = ttnn.topk(scores_grouped, k=2, dim=-1)  # values, indices
        group_scores = ttnn.sum(group_topk.values, dim=-1)  # [1, 1, batch*seq, 8]

        # Select top-k groups
        group_topk = ttnn.topk(group_scores, k=topk_group=4, dim=-1)
        group_idx = group_topk.indices  # [1, 1, batch*seq, 4]

        # Create mask for selected groups (scatter operation)
        group_mask = ttnn.zeros_like(group_scores)
        group_mask = ttnn.scatter(group_mask, dim=-1, index=group_idx, value=1.0)

        # Expand mask to all experts
        expert_mask = ttnn.reshape(group_mask, [1, 1, batch*seq, 8, 1])
        expert_mask = ttnn.broadcast(expert_mask, [1, 1, batch*seq, 8, 32])
        expert_mask = ttnn.reshape(expert_mask, [1, 1, batch*seq, 256])

        # Select top-k experts from masked scores
        masked_scores = ttnn.where(expert_mask, scores_corrected, float('-inf'))
        topk = ttnn.topk(masked_scores, k=num_experts_per_tok=8, dim=-1)
        topk_idx = topk.indices  # [1, 1, batch*seq, 8]

        # Get weights from original scores
        topk_weight = ttnn.gather(scores, topk_idx, dim=-1)  # [1, 1, batch*seq, 8]

        # 3. Normalize weights to sum to 1
        weight_sum = ttnn.sum(topk_weight, dim=-1, keepdim=True)
        topk_weight = ttnn.multiply(topk_weight, ttnn.reciprocal(weight_sum))

        # 4. Apply scaling factor
        topk_weight = ttnn.multiply(topk_weight, routed_scaling_factor)  # 2.5

        # Convert to expected output format
        topk_idx = ttnn.reshape(topk_idx, [batch*seq, 8])
        topk_weight = ttnn.reshape(topk_weight, [batch*seq, 8])

        return topk_idx, topk_weight
    ```

    Shape Examples:
    - Prefill: [1, 1, SEQ_LEN, 7168] -> idx: [SEQ_LEN, 8], weight: [SEQ_LEN, 8]
    - Decode:  [1, 1, BATCH_SIZE, 7168] -> idx: [BATCH_SIZE, 8], weight: [BATCH_SIZE, 8]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 8
        self.n_routed_experts = config.n_routed_experts  # 256
        self.routed_scaling_factor = config.routed_scaling_factor  # 2.5
        self.n_group = config.n_group  # 8
        self.topk_group = config.topk_group  # 4
        self.norm_topk_prob = config.norm_topk_prob  # True
        self.gating_dim = config.hidden_size  # 7168

        # Weights will be loaded as ttnn tensors
        self.weight = None  # [n_routed_experts=256, hidden_size=7168]
        self.e_score_correction_bias = None  # [n_routed_experts=256]

    def forward(self, hidden_states, memory_config=None, compute_kernel_config=None):
        """
        Args:
            hidden_states: TTNN tensor [1, 1, BATCH_SIZE*SEQ_LEN, 7168]

        Returns:
            topk_idx: TTNN tensor [BATCH_SIZE*SEQ_LEN, 8] - selected expert indices
            topk_weight: TTNN tensor [BATCH_SIZE*SEQ_LEN, 8] - expert weights
        """
        batch_seq = hidden_states.shape[2]

        # Compute gating scores
        logits = ttnn.linear(
            hidden_states,
            self.weight,
            bias=None,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
        )

        # Apply sigmoid activation
        scores = ttnn.sigmoid(logits, memory_config=memory_config)

        # Add correction bias
        scores_corrected = ttnn.add(scores, self.e_score_correction_bias, memory_config=memory_config)

        # Grouped top-k selection
        # Reshape to [1, 1, batch*seq, n_group=8, experts_per_group=32]
        scores_grouped = ttnn.reshape(
            scores_corrected, [1, 1, batch_seq, self.n_group, self.n_routed_experts // self.n_group]
        )

        # Top-2 per group
        group_topk = ttnn.topk(scores_grouped, k=2, dim=-1)
        group_scores = ttnn.sum(group_topk.values, dim=-1)  # [1, 1, batch*seq, 8]

        # Select top groups
        group_topk = ttnn.topk(group_scores, k=self.topk_group, dim=-1)
        group_idx = group_topk.indices  # [1, 1, batch*seq, 4]

        # Create group mask using scatter
        group_mask = ttnn.zeros_like(group_scores)
        group_mask = ttnn.scatter(group_mask, dim=-1, index=group_idx, value=1.0)

        # Expand mask to expert dimension
        expert_mask = ttnn.reshape(group_mask, [1, 1, batch_seq, self.n_group, 1])
        expert_mask = ttnn.broadcast(
            expert_mask, [1, 1, batch_seq, self.n_group, self.n_routed_experts // self.n_group]
        )
        expert_mask = ttnn.reshape(expert_mask, [1, 1, batch_seq, self.n_routed_experts])

        # Mask scores and select top-k
        masked_scores = ttnn.where(expert_mask, scores_corrected, ttnn.full_like(scores_corrected, float("-inf")))

        topk = ttnn.topk(masked_scores, k=self.top_k, dim=-1)
        topk_idx = topk.indices  # [1, 1, batch*seq, 8]

        # Get weights from original scores
        topk_weight = ttnn.gather(scores, topk_idx, dim=-1)

        # Normalize weights
        weight_sum = ttnn.sum(topk_weight, dim=-1, keepdim=True)
        weight_sum = ttnn.add(weight_sum, 1e-20)  # Avoid division by zero
        topk_weight = ttnn.multiply(topk_weight, ttnn.reciprocal(weight_sum))

        # Apply scaling factor
        topk_weight = ttnn.multiply_by_scalar(topk_weight, self.routed_scaling_factor)

        # Reshape to expected format
        topk_idx = ttnn.reshape(topk_idx, [batch_seq, self.top_k])
        topk_weight = ttnn.reshape(topk_weight, [batch_seq, self.top_k])

        return topk_idx, topk_weight


class DeepseekV3MoETTNN(nn.Module):
    """
    TTNN implementation of mixed expert module with shared experts.

    Pseudo-code:
    ```
    def forward(hidden_states):
        # Input shape: [1, 1, BATCH_SIZE*SEQ_LEN, hidden_size=7168]
        # Output shape: [1, 1, BATCH_SIZE*SEQ_LEN, hidden_size=7168]

        identity = hidden_states  # Save for shared expert

        # 1. Gate computes expert selection
        topk_idx, topk_weight = gate(hidden_states)
        # topk_idx: [batch*seq, 8], topk_weight: [batch*seq, 8]

        # 2. Dispatch tokens to experts (MISSING OP)
        # This would use the proposed ttnn_all_to_all_dispatch
        sparse_tokens = ttnn.zeros([n_experts, batch*seq, 1, hidden_size])
        expert_metadata = ttnn.zeros([n_devices, 1, batch*seq, 8])

        ttnn_all_to_all_dispatch(
            hidden_states, topk_idx, expert_mapping,
            sparse_tokens, expert_metadata
        )

        # 3. Apply experts
        expert_outputs = []
        for i in range(n_routed_experts):
            # Each expert processes its assigned tokens
            expert_out = experts[i](sparse_tokens[i])
            expert_outputs.append(expert_out)

        expert_contrib = ttnn.stack(expert_outputs)  # [n_experts, batch*seq, 1, hidden_size]

        # 4. Recall tokens from experts (MISSING OP)
        recalled = ttnn_recall(expert_contrib, expert_metadata, expert_mapping)
        # recalled: [8, batch*seq, 1, hidden_size]

        # 5. Weighted sum of expert outputs
        # Reshape weights for broadcasting
        weights = ttnn.reshape(topk_weight, [batch*seq, 8, 1, 1])
        weights = ttnn.transpose(weights, 0, 1)  # [8, batch*seq, 1, 1]

        # Apply weights and sum
        weighted = ttnn.multiply(recalled, weights)  # [8, batch*seq, 1, hidden_size]
        y = ttnn.sum(weighted, dim=0)  # [batch*seq, 1, hidden_size]
        y = ttnn.reshape(y, [1, 1, batch*seq, hidden_size])

        # 6. Add shared expert output (always active)
        shared_out = shared_experts(identity)  # [1, 1, batch*seq, hidden_size]
        y = ttnn.add(y, shared_out)

        return y
    ```

    Shape Examples:
    - Prefill: [1, 1, SEQ_LEN, 7168] -> [1, 1, SEQ_LEN, 7168]
    - Decode:  [1, 1, BATCH_SIZE, 7168] -> [1, 1, BATCH_SIZE, 7168]

    Notes:
    - n_routed_experts = 256 expert MLPs
    - n_shared_experts = 1 always-active expert
    - num_experts_per_tok = 8 experts selected per token
    - Expert MLPs use intermediate_size = 2048
    - Requires proposed dispatch/recall operations for efficient routing
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok  # 8
        self.n_routed_experts = config.n_routed_experts  # 256
        self.n_shared_experts = config.n_shared_experts  # 1
        self.moe_intermediate_size = config.moe_intermediate_size  # 2048

        # Gate module
        self.gate = MoEGateTTNN(config)

        # Expert MLPs - will be loaded as TTNN modules
        self.experts = nn.ModuleList(
            [None for _ in range(self.n_routed_experts)]
        )  # Will be DeepseekV3MLPTTNN instances

        # Shared expert
        self.shared_experts = None  # Will be DeepseekV3MLPTTNN instance

        # Expert mapping for dispatch/recall
        self.expert_mapping = None  # [num_devices, 1, n_experts, num_devices]

    def forward(self, hidden_states, memory_config=None, compute_kernel_config=None):
        """
        Args:
            hidden_states: TTNN tensor [1, 1, BATCH_SIZE*SEQ_LEN, 7168]

        Returns:
            TTNN tensor [1, 1, BATCH_SIZE*SEQ_LEN, 7168]
        """
        # Save input for shared expert
        identity = hidden_states
        batch_seq = hidden_states.shape[2]

        # Get expert routing
        topk_idx, topk_weight = self.gate(
            hidden_states, memory_config=memory_config, compute_kernel_config=compute_kernel_config
        )

        # For now, simplified version without actual dispatch/recall
        # In practice, would use the proposed MoE dispatch ops

        # Placeholder for expert computation
        # Real implementation would:
        # 1. Dispatch tokens to experts across devices
        # 2. Each expert computes on its assigned tokens
        # 3. Recall and combine expert outputs

        # For mock implementation, just show the intended flow:
        # This would be replaced by actual dispatch/recall ops
        expert_output = hidden_states  # Placeholder

        # Add shared expert (always active)
        shared_output = self.shared_experts(
            identity, memory_config=memory_config, compute_kernel_config=compute_kernel_config
        )

        # Combine routed and shared expert outputs
        output = ttnn.add(expert_output, shared_output, memory_config=memory_config)

        return output

    def forward_with_dispatch(self, hidden_states, memory_config=None, compute_kernel_config=None):
        """
        Full implementation using proposed dispatch/recall ops
        This shows how it would work with the missing operations
        """
        identity = hidden_states
        batch_seq = hidden_states.shape[2]

        # Get expert routing
        topk_idx, topk_weight = self.gate(hidden_states)

        # Prepare buffers for dispatch
        sparse_tokens = ttnn.zeros(
            [self.n_routed_experts, batch_seq, 1, self.config.hidden_size],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        expert_metadata = ttnn.zeros(
            [1, 1, batch_seq, self.num_experts_per_tok],  # Assuming single device
            device=hidden_states.device,
            dtype=ttnn.uint16,
        )

        # Dispatch tokens to experts
        ttnn_all_to_all_dispatch(hidden_states, topk_idx, self.expert_mapping, sparse_tokens, expert_metadata)

        # Apply each expert to its tokens
        expert_outputs = []
        for i in range(self.n_routed_experts):
            if self.experts[i] is not None:
                expert_out = self.experts[i](
                    sparse_tokens[i], memory_config=memory_config, compute_kernel_config=compute_kernel_config
                )
                expert_outputs.append(expert_out)

        # Stack expert outputs
        expert_contrib = ttnn.stack(expert_outputs, dim=0)

        # Recall tokens from experts
        recalled = ttnn_recall(expert_contrib, expert_metadata, self.expert_mapping)  # [8, batch*seq, 1, hidden_size]

        # Apply weights and sum
        weights = ttnn.reshape(topk_weight, [batch_seq, self.num_experts_per_tok, 1, 1])
        weights = ttnn.transpose(weights, 0, 1)  # [8, batch*seq, 1, 1]

        weighted = ttnn.multiply(recalled, weights)
        routed_output = ttnn.sum(weighted, dim=0)
        routed_output = ttnn.reshape(routed_output, [1, 1, batch_seq, self.config.hidden_size])

        # Add shared expert
        shared_output = self.shared_experts(identity)
        output = ttnn.add(routed_output, shared_output)

        return output
