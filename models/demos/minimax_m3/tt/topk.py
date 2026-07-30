# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M3 MoE router (gate).

Routing:
  * sigmoid scoring over all experts (the gate Linear has no bias)
  * a separate ``e_score_correction_bias`` is added to the sigmoid scores FOR
    SELECTION ONLY (picks which experts win, not the returned weights)
  * the returned top-k weights are the UNBIASED sigmoid values gathered at the
    selected indices, then normalized to sum to 1

HF reference (MiniMaxM3SparseMoeBlock.route_tokens_to_experts):
    routing_weights = sigmoid(router_logits.float())
    scores_for_choice = routing_weights + e_score_correction_bias
    _, top_k_index = topk(scores_for_choice, top_k)
    top_k_weights = routing_weights.gather(1, top_k_index)
    top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
"""

import ttnn
from models.demos.minimax_m3.utils.general_utils import cache_file_exists, get_cache_file_name


def route_tokens_to_experts(
    router_logits, experts_per_token, num_experts, score_bias, use_throughput_experts, routed_scaling_factor=1.0
):
    """Apply MiniMax-M3 routing to gate logits [tokens, num_experts]."""
    if router_logits.dtype != ttnn.bfloat16:
        router_logits = ttnn.typecast(router_logits, dtype=ttnn.bfloat16)

    # Sigmoid scores over all experts.
    routing_weights = ttnn.sigmoid(router_logits)

    # Selection scores = sigmoid + correction bias (bias affects WHICH experts are
    # chosen, not the returned weights).
    if score_bias is not None:
        scores_for_choice = ttnn.add(routing_weights, score_bias)
    else:
        scores_for_choice = routing_weights

    _, expert_indices = ttnn.topk(scores_for_choice, k=experts_per_token, dim=-1, sorted=True)
    if scores_for_choice is not routing_weights:
        scores_for_choice.deallocate(True)

    # Gather the UNBIASED sigmoid weights at the selected indices, normalize to sum 1.
    top_k_weights = ttnn.gather(routing_weights, dim=-1, index=expert_indices)
    denom = ttnn.sum(top_k_weights, dim=-1, keepdim=True)
    top_k_weights = ttnn.div(top_k_weights, denom)
    denom.deallocate(True)

    # M3: scale the routed weights by routed_scaling_factor (2.0), applied AFTER normalize
    # (DeepSeek/M3 convention; shared expert is added later, unscaled). A factor of 1.0 -> no-op.
    if routed_scaling_factor != 1.0:
        top_k_weights = ttnn.mul(top_k_weights, routed_scaling_factor)

    if use_throughput_experts:
        routing_weights.deallocate(True)
        return expert_indices, top_k_weights

    # Dense [tokens, num_experts] weights (normalized value at selected experts, 0 elsewhere).
    dense_weights = ttnn.scatter(ttnn.zeros_like(routing_weights), dim=1, index=expert_indices, src=top_k_weights)
    routing_weights.deallocate(True)
    top_k_weights.deallocate(True)
    return expert_indices, dense_weights


class TopKRouter:
    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None):
        self.top_k = hf_config.num_experts_per_tok
        self.num_experts = hf_config.num_local_experts
        self.hidden_dim = hf_config.hidden_size
        # M3: routed-expert output is scaled by routed_scaling_factor (2.0, from config; 1.0 if absent).
        self.routed_scaling_factor = getattr(hf_config, "routed_scaling_factor", 1.0)
        self.tensor_cache_path = tensor_cache_path

        # MiniMax-M3 gate Linear has no bias; weight is [num_experts, hidden] -> [hidden, num_experts].
        torch_weight = state_dict["weight"].transpose(0, 1) if state_dict else None
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # e_score_correction_bias [num_experts] -> [1, num_experts], replicated; added to
        # selection scores only. Absent in some checkpoints -> no correction.
        score_bias_torch = None
        if state_dict and "e_score_correction_bias" in state_dict:
            score_bias_torch = state_dict["e_score_correction_bias"].reshape(1, -1)
        bias_cache_file = get_cache_file_name(tensor_cache_path, "e_score_correction_bias")
        # Build the bias tensor when we have the source weight, OR (cache-only loading, empty
        # state_dict) when it was previously cached — torch=None then loads it straight from disk.
        # Whether the checkpoint HAS a correction bias can't be known without the source, so the
        # cached file's existence is the signal.
        build_bias = score_bias_torch is not None or (not state_dict and cache_file_exists(bias_cache_file))
        self.score_bias = (
            ttnn.as_tensor(
                score_bias_torch,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                cache_file_name=bias_cache_file,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
                if isinstance(mesh_device, ttnn.MeshDevice)
                else None,
            )
            if build_bias
            else None
        )

        # Custom compute configs can degrade routing quality; keep the default.
        self.compute_config = None

    def __call__(self, hidden_states, use_throughput_experts):
        # Actual token count from volume (shape[0] after reshape is tile-padded).
        actual_tokens = hidden_states.volume() // self.hidden_dim
        hidden_states = ttnn.reshape(hidden_states, (-1, self.hidden_dim))

        # L1 for decode (small), DRAM for prefill (large sequences).
        is_decode = actual_tokens <= 128
        mem_config = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        router_logits = ttnn.linear(
            hidden_states,
            self.weight,  # no bias (MiniMax-M3)
            memory_config=mem_config,
            compute_kernel_config=self.compute_config,
        )

        expert_indices, expert_weights = route_tokens_to_experts(
            router_logits,
            self.top_k,
            self.num_experts,
            self.score_bias,
            use_throughput_experts,
            self.routed_scaling_factor,
        )
        ttnn.deallocate(router_logits)
        return expert_indices, expert_weights
