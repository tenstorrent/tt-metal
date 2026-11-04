# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blackhole decode using generic collectives and the existing expert matmuls."""

from math import prod

import ttnn

from .decode import expert_mlp_forward


def gather_decode_forward(
    hidden_states,
    topk_expert_indices,
    topk_expert_weights,
    *,
    weights,
    config,
    local_expert_ids,
    program_config,
    mesh_device,
    cluster_axis,
    memory_config,
):
    """Gather tokens, evaluate local experts, and reduce to each token's owner.

    Tokens are sharded along cluster_axis and replicated along the other axis.
    Expert weights and local_expert_ids use the same linear device ordering.
    This avoids the fused moe_gpt kernel's twelve-DRAM-core weight layout.
    Inputs are consumed. The output is [1, 1, local_tokens, hidden_size].
    """
    local_tokens = prod(hidden_states.shape) // config.hidden_size
    total_tokens = local_tokens * mesh_device.shape[cluster_axis]

    def gather(value, width, dtype=None):
        value = ttnn.reshape(value, (1, 1, local_tokens, width))
        if value.layout != ttnn.TILE_LAYOUT:
            tiled = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
            ttnn.deallocate(value)
            value = tiled
        if dtype is not None and value.dtype != dtype:
            converted = ttnn.typecast(value, dtype)
            ttnn.deallocate(value)
            value = converted
        gathered = ttnn.all_gather(value, dim=2, cluster_axis=cluster_axis, memory_config=memory_config)
        ttnn.deallocate(value)
        return gathered

    gathered_tokens = gather(hidden_states, config.hidden_size)
    # FP32 preserves integer expert IDs beyond BF16's exact integer range.
    gathered_indices = gather(topk_expert_indices, config.num_experts_per_tok, ttnn.float32)
    gathered_scores = gather(topk_expert_weights, config.num_experts_per_tok)

    # [1, local_experts, total_tokens, K] -> [1, local_experts, total_tokens, 1].
    # Experts absent from a token's top-k receive exactly zero weight.
    matches = ttnn.eq(gathered_indices, local_expert_ids)
    ttnn.deallocate(gathered_indices)
    mask = ttnn.typecast(matches, ttnn.bfloat16)
    ttnn.deallocate(matches)
    selected_scores = ttnn.mul(mask, gathered_scores, memory_config=memory_config)
    ttnn.deallocate(mask)
    ttnn.deallocate(gathered_scores)
    local_scores = ttnn.sum(selected_scores, dim=3, keepdim=True)
    ttnn.deallocate(selected_scores)

    expert_inputs = ttnn.repeat(gathered_tokens, ttnn.Shape((1, config.num_experts_per_device, 1, 1)))
    ttnn.deallocate(gathered_tokens)
    expert_outputs_rm = expert_mlp_forward(expert_inputs, weights, config, memory_config, program_config, total_tokens)
    expert_outputs = ttnn.to_layout(expert_outputs_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(expert_outputs_rm)
    expert_outputs = ttnn.permute(expert_outputs, (1, 0, 2, 3))
    weighted = ttnn.mul(expert_outputs, local_scores, memory_config=memory_config)
    ttnn.deallocate(expert_outputs)
    ttnn.deallocate(local_scores)
    local_output = ttnn.sum(weighted, dim=1, keepdim=True)
    ttnn.deallocate(weighted)

    # Sum experts across rows while returning tokens to their original row,
    # then sum the remaining expert contributions across columns.
    scattered = ttnn.reduce_scatter(local_output, dim=2, cluster_axis=cluster_axis, memory_config=memory_config)
    ttnn.deallocate(local_output)
    output = ttnn.all_reduce(scattered, cluster_axis=1 - cluster_axis, memory_config=memory_config)
    ttnn.deallocate(scattered)
    return output
