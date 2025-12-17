# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Top-K Router implementation for Mixture of Experts (MoE) in GPT-OSS.

This module implements the routing mechanism that selects which experts should
process each token in the MoE architecture. The router uses a learned linear
transformation followed by top-k selection to assign tokens to experts.

"""

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name


def topk_router(g, experts_per_token):
    """
    Select top-k experts for each token based on router logits.

    This function implements the core routing logic:
    1. Select top-k expert indices using TTNN's topk operation
    2. Normalize the selected expert weights using softmax
    3. Scatter the weights back to full expert dimension for downstream use

    Args:
        g: Router logits tensor of shape (batch * seq_len, num_experts)
        experts_per_token: Number of experts to select per token (k value)

    Returns:
        Tuple of:
            - router_scores: Sparse scores tensor with non-zero values only for selected experts
            - expert_weights: Normalized weights for selected experts (used for weighted combination)
            - expert_indices: Indices of selected experts for each token

    Note:
        The softmax normalization uses HiFi4 math fidelity and FP32 accumulation
        for numerical stability, as routing decisions are critical for model quality.
    """
    typecast_needed = False
    if g.dtype != ttnn.bfloat16:
        g_og = g
        typecast_needed = True
        g = ttnn.typecast(g, dtype=ttnn.bfloat16)

    expert_weights, expert_indices = ttnn.topk(g, k=experts_per_token, dim=-1, sorted=True)
    if typecast_needed:
        g.deallocate(True)
        g = g_og
    compute_config = ttnn.init_device_compute_kernel_config(
        g.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    expert_weights = ttnn.softmax(expert_weights, dim=1, numeric_stable=True, compute_kernel_config=compute_config)
    router_scores = ttnn.scatter(ttnn.zeros_like(g), dim=1, index=expert_indices, src=expert_weights)
    return router_scores, expert_weights, expert_indices


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
        self.num_experts = hf_config.num_local_experts
        self.hidden_dim = hf_config.hidden_size
        self.weight = ttnn.as_tensor(
            state_dict["weight"].transpose(0, 1),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bias = ttnn.as_tensor(
            state_dict["bias"].unsqueeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Known Issue: Custom compute configs cause output quality degradation
        # Using None (default config) until root cause is identified
        # TODO: Investigate and fix compute config optimization
        # self.compute_config = ttnn.init_device_compute_kernel_config(
        #     mesh_device.arch(),
        #     math_fidelity=ttnn.MathFidelity.HiFi2,
        #     math_approx_mode=False,
        #     fp32_dest_acc_en=False,
        #     packer_l1_acc=False,
        # )
        self.compute_config = None

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
        # Detect decode mode for L1_WIDTH_SHARDED optimization (like tt-transformers MLP)
        is_decode_mode = hidden_states.shape[1] == 1
        mem_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if is_decode_mode else ttnn.DRAM_MEMORY_CONFIG

        hidden_states = ttnn.reshape(hidden_states, (-1, self.hidden_dim))
        router_logits = ttnn.linear(
            hidden_states,
            self.weight,
            bias=self.bias,
            memory_config=mem_config,
            compute_kernel_config=self.compute_config,
        )

        # TopK doesn't support sharded inputs yet - convert to DRAM if sharded
        if is_decode_mode:
            router_logits = ttnn.to_memory_config(router_logits, ttnn.DRAM_MEMORY_CONFIG)

        router_scores, _expert_weights, router_indices = topk_router(router_logits, self.top_k)
        return router_scores, router_indices, router_logits
