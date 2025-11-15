# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Top-K Router implementation for Mixture of Experts (MoE) in GLM-4.5.

Matches the reference grouped top-k selection behavior:
- Apply sigmoid to router logits to obtain scores
- Partition experts into `n_group` groups; select `topk_group` groups by
  summing the top-2 expert scores in each group (as in reference)
- Mask scores outside the selected groups and take top-k experts globally
- Optionally normalize selected weights and scale by `routed_scaling_factor`
"""

import ttnn
from models.demos.glm_45.utils.general_utils import get_cache_file_name


def _grouped_topk(scores, n_group, topk_group, top_k):
    """Implements grouped top-k selection on device using TTNN ops.

    scores: [N, E]
    Returns: (topk_indices [N, top_k], topk_weights [N, top_k])
    """
    N, E = scores.shape[-2], scores.shape[-1]
    if n_group <= 1 or topk_group >= n_group:
        # No grouping constraint; just take top-k globally
        topk_weights, topk_indices = ttnn.topk(scores, k=top_k, dim=-1, sorted=False)
        return topk_indices, topk_weights

    # Reshape into groups: [N, n_group, E_per_group]
    assert E % n_group == 0, "Number of experts must be divisible by n_group"
    e_per_group = E // n_group
    scores_g = ttnn.reshape(scores, (N, n_group, e_per_group))
    # For each group, sum top2 to form group score
    top2_vals, _ = ttnn.topk(scores_g, k=2, dim=-1, sorted=False)
    group_scores = ttnn.sum(top2_vals, dim=-1)  # [N, n_group]
    # Select top groups
    _, group_idx = ttnn.topk(group_scores, k=topk_group, dim=-1, sorted=False)

    # Build mask over experts: expand group indices to expert indices mask
    # Create a zero mask then scatter 1 at selected group positions
    group_mask = ttnn.zeros_like(group_scores)
    one = ttnn.full_like(group_mask, 1)
    group_mask = ttnn.scatter(group_mask, dim=-1, index=group_idx, src=one)
    # Expand mask to expert dimension and reshape back to [N, E]
    group_mask = ttnn.unsqueeze(group_mask, -1)
    group_mask = ttnn.repeat(group_mask, (1, 1, e_per_group))
    group_mask = ttnn.reshape(group_mask, (N, E))

    masked_scores = ttnn.mul(scores, group_mask)
    topk_weights, topk_indices = ttnn.topk(masked_scores, k=top_k, dim=-1, sorted=False)
    return topk_indices, topk_weights


class TopKRouter:
    """
    Top-K Expert Router for Mixture of Experts (MoE) models.

    This class implements a learnable router that assigns each token to the top-k
    most relevant experts. The router consists of a linear projection from hidden
    dimension to num_experts, followed by top-k selection and softmax normalization.

    The routing mechanism is crucial for MoE performance:
    - It determines which experts process each token
    - It provides mixing weights for combining expert outputs
    - It enables dynamic computation allocation based on input

    Attributes:
        top_k: Number of experts selected per token
        num_experts: Total number of available experts
        hidden_dim: Input hidden dimension size
        weight: Router projection weight tensor
        bias: Router projection bias tensor
        compute_config: Compute kernel configuration (currently None due to quality issues)
    """

    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None):
        """
        Initialize the Top-K router with pretrained weights.

        Args:
            mesh_device: TTNN mesh device for tensor placement
            hf_config: HuggingFace config containing MoE parameters
            state_dict: Dictionary containing router weights and biases
            tensor_cache_path: Optional path for caching tensors on device
        """
        self.top_k = hf_config.num_experts_per_tok
        self.num_experts = hf_config.n_routed_experts
        self.hidden_dim = hf_config.hidden_size
        self.n_group = getattr(hf_config, "n_group", 1)
        self.topk_group = getattr(hf_config, "topk_group", 1)
        self.norm_topk_prob = getattr(hf_config, "norm_topk_prob", True)
        self.routed_scaling_factor = getattr(hf_config, "routed_scaling_factor", 1.0)
        self.weight = ttnn.as_tensor(
            state_dict["weight"].transpose(0, 1),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        bias_t = state_dict.get("bias")
        if bias_t is not None:
            self.bias = ttnn.as_tensor(
                bias_t.unsqueeze(0),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                cache_file_name=get_cache_file_name(tensor_cache_path, "bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.bias = None

        # Router compute config (default OK)
        self.compute_config = None
        # Bias to correct expert scoring if needed (default zeros)
        self.e_score_correction_bias = ttnn.zeros((self.num_experts,), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)

    def __call__(self, hidden_states):
        """
        Route tokens to top-k experts.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Tuple of:
                - router_scores: Sparse expert scores for all tokens
                - router_indices: Selected expert indices for each token
                - router_logits: Raw logits before top-k selection (used for load balancing loss)
        """
        hidden_states = ttnn.reshape(hidden_states, (-1, self.hidden_dim))
        router_logits = ttnn.linear(
            hidden_states, self.weight, bias=self.bias, compute_kernel_config=self.compute_config
        )
        # Use raw logits for robust group selection; use sigmoid scores for mixing weights
        selection_scores = router_logits
        scores = ttnn.sigmoid(router_logits)

        N, E = selection_scores.shape[-2], selection_scores.shape[-1]
        if self.n_group <= 1 or self.topk_group >= self.n_group:
            topk_weights, topk_indices = ttnn.topk(scores, k=self.top_k, dim=-1, sorted=False)
        else:
            assert E % self.n_group == 0, "Number of experts must be divisible by n_group"
            e_per_group = E // self.n_group
            sel_g = ttnn.reshape(selection_scores, (N, self.n_group, e_per_group))
            top2_vals, _ = ttnn.topk(sel_g, k=2, dim=-1, sorted=False)
            group_scores = ttnn.sum(top2_vals, dim=-1)
            _, group_idx = ttnn.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)

            group_mask = ttnn.zeros_like(group_scores)
            one = ttnn.full_like(group_mask, 1)
            group_mask = ttnn.scatter(group_mask, dim=-1, index=group_idx, src=one)
            group_mask = ttnn.unsqueeze(group_mask, -1)
            group_mask = ttnn.repeat(group_mask, (1, 1, e_per_group))
            group_mask = ttnn.reshape(group_mask, (N, E))

            masked_scores = ttnn.mul(scores, group_mask)
            topk_weights, topk_indices = ttnn.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)
        if self.norm_topk_prob:
            denom = ttnn.sum(topk_weights, dim=-1, keepdim=True)
            denom = ttnn.add(denom, ttnn.full_like(denom, 1e-20))
            topk_weights = ttnn.divide(topk_weights, denom)
        if self.routed_scaling_factor != 1.0:
            scale = ttnn.full_like(topk_weights, self.routed_scaling_factor)
            topk_weights = ttnn.mul(topk_weights, scale)

        # Scatter weights back to full expert dimension
        zeros = ttnn.zeros_like(scores)
        router_scores = ttnn.scatter(zeros, dim=-1, index=topk_indices, src=topk_weights)
        return router_scores, topk_indices, router_logits
