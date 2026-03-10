# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Fused decode forward pass using all_to_all_dispatch_metadata + moe_gpt + selective_reduce_combine.

This implements the optimized sparse MoE forward pass for decode. Compared to the
existing dense flow (all_to_all_dispatch + batched matmul + all_to_all_combine), the
fused flow is more efficient:

  Old (dense):
    all_to_all_dispatch → repeat input × E experts → dense batched matmul (W1/W3/W2)
    → all_to_all_combine → weighted sum → all_reduce

  New (fused/sparse):
    all_to_all_dispatch_metadata → moe_gpt (tilize + W0/W1/SwiGLU + A2A ring + W2 + combine)
    → selective_reduce_combine → all_reduce

Key differences:
  - moe_gpt only processes tokens actually routed to each expert (sparse, not dense)
  - W0/W1/SwiGLU/A2A ring/W2/combine are fused into a single kernel
  - No need to repeat input across the expert dimension
  - selective_reduce_combine aliases moe_gpt's BLOCK_SHARDED combine output directly
    via set_globally_allocated_address — no TM ops needed between moe_gpt and combine

The existing decode_forward (dense flow) is unchanged; this module adds an alternative
path that the ThroughputExperts class can route to when fused_config is provided.
"""

from math import prod

import ttnn

from .config import FusedMoeGptConfig, ThroughputExpertConfig


def fused_decode_forward(
    hidden_states: ttnn.Tensor,
    topk_expert_indices: ttnn.Tensor,
    topk_expert_scores: ttnn.Tensor,
    config: ThroughputExpertConfig,
    fused_config: FusedMoeGptConfig,
    mesh_device,
) -> ttnn.Tensor:
    """Fused decode: all_to_all_dispatch_metadata + moe_gpt + selective_reduce_combine.

    Args:
        hidden_states: Input token embeddings [B, 1, S, H] in L1 (required by dispatch).
            On a (4,8) mesh with row-sharding, typically [32, 1, 1, H] per device (32 tokens).
        topk_expert_indices: Expert routing indices [B, 1, S, K], HEIGHT_SHARDED L1 uint16.
            Must be pre-formatted for all_to_all_dispatch_metadata (L1 HEIGHT_SHARDED).
        topk_expert_scores: Expert routing scores [B, 1, S, K], HEIGHT_SHARDED L1 bfloat16.
            Same memory config as topk_expert_indices.
        config: ThroughputExpertConfig with model dimensions.
        fused_config: FusedMoeGptConfig with pre-allocated resources and weight tensors.
        mesh_device: TTNN mesh device.
    Returns:
        Tensor [1, 1, tokens_per_device, H] in L1, compatible with decode_forward's
        return format. Routing scores are applied to expert outputs before summing
        (following the DeepSeek pattern: permute scores → mul → sum over K dim).
    """
    # Ensure hidden_states is in ROW_MAJOR L1 (all_to_all_dispatch_metadata requires both).
    # In production the prior layer outputs TILE_LAYOUT; dispatch needs ROW_MAJOR.
    if hidden_states.layout != ttnn.ROW_MAJOR_LAYOUT:
        hidden_states_rm = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(hidden_states)
        hidden_states = hidden_states_rm
    if hidden_states.memory_config().buffer_type != ttnn.BufferType.L1:
        hidden_states_l1 = ttnn.to_memory_config(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(hidden_states)
        hidden_states = hidden_states_l1

    cluster_axis = fused_config.cluster_axis
    num_dispatch_devices = mesh_device.shape[cluster_axis] if cluster_axis is not None else prod(mesh_device.shape)
    # global_experts = config.num_experts (total experts across all devices, e.g., 128 for GPT-OSS).
    # num_clusters = number of independent rings = mesh_cols for cluster_axis=0.
    # selective_reduce_combine needs global_experts to compute experts_per_device = 128/32 = 4 ✓.
    # experts_per_ring = global_experts // num_clusters = 128 // 8 = 16 (experts processed per ring).
    global_experts = config.num_experts  # total global experts (e.g., 128 for GPT-OSS)
    num_clusters = prod(mesh_device.shape) // num_dispatch_devices  # 8 independent rings for (4,8)

    # Reshape hidden_states to 2D for dispatch: [tokens_per_device, H]
    # dispatch expects input with tokens on dim -2 (as [1, 1, tokens, H])
    input_shape = hidden_states.shape
    tokens_per_device = input_shape[0] * input_shape[2]
    total_tokens = tokens_per_device * num_dispatch_devices

    # Reshape to [M, 1, 1, K] format expected by dispatch_metadata (tokens on dim 0)
    hidden_states = ttnn.reshape(hidden_states, (tokens_per_device, 1, 1, config.hidden_size))
    # Indices/scores are HEIGHT_SHARDED L1 - reshape logical shape but shard layout stays same
    topk_expert_indices = ttnn.reshape(topk_expert_indices, (tokens_per_device, 1, 1, config.num_experts_per_tok))
    topk_expert_scores = ttnn.reshape(topk_expert_scores, (tokens_per_device, 1, 1, config.num_experts_per_tok))

    # Keep a reference to pre-dispatch scores for post-combine weighting.
    # These are the original per-device routing scores [M, 1, 1, K] before dispatch
    # shuffles them across devices. Used in the scale_experts step after combine.
    pre_dispatch_scores = topk_expert_scores

    # ------------------------------------------------------------------
    # Step 1: all_to_all_dispatch_metadata
    # Routes token metadata (indices, scores) to expert devices; the sparse
    # buffer holds the actual token data routed by the dispatch kernel.
    # Inputs must be in L1 (token data) and HEIGHT_SHARDED L1 (indices/scores).
    # ------------------------------------------------------------------
    (tt_sparse, tt_indices, tt_scores) = ttnn.experimental.all_to_all_dispatch_metadata(
        hidden_states,
        topk_expert_indices,
        topk_expert_scores,
        fused_config.tt_dispatch_mapping,
        cluster_axis=cluster_axis,
        num_links=fused_config.num_links,
        output_tensors=(
            fused_config.dispatch_sparse,
            fused_config.dispatch_indices,
            fused_config.dispatch_scores,
        ),
        cross_device_semaphore=fused_config.dispatch_semaphore,
        dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_UNICAST,
    )
    ttnn.deallocate(hidden_states)

    # Move sparse buffer from DRAM to L1; reshape from [1, total_tokens, H] to [total_tokens, H]
    # (moe_gpt reads total_tokens from sparse_shape[0])
    tt_sparse_l1 = ttnn.to_memory_config(tt_sparse, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_sparse_l1 = ttnn.reshape(tt_sparse_l1, [total_tokens, config.hidden_size])

    # ------------------------------------------------------------------
    # Step 2: moe_gpt (fused sparse compute)
    # Performs: tilize → W0/W1 matmul → SwiGLU → A2A ring → W2 matmul → combine
    # Indices/scores are HEIGHT_SHARDED L1 from dispatch (no format conversion needed;
    # moe_gpt reads them via TensorAccessorArgs which handles HEIGHT_SHARDED buffers).
    # Output[3] is BLOCK_SHARDED L1 on the combine core rectangle.
    # ------------------------------------------------------------------
    moe_gpt_outputs = ttnn.experimental.moe_gpt(
        tt_sparse_l1,
        expert_indices=tt_indices,
        expert_scores=tt_scores,
        expert_mapping=fused_config.tt_moe_gpt_mapping,
        w0_w1_tensor=fused_config.tt_w0_w1,
        w2_tensor=fused_config.tt_w2,
        cluster_axis=cluster_axis,
    )
    ttnn.deallocate(tt_sparse_l1)

    # moe_gpt outputs (matching moe_compute output convention):
    #   [0] token_counts:   [1, padded_E] uint32, interleaved L1
    #   [1] dense_metadata: expert activation metadata, uint32, interleaved L1
    #   [2] dense_e_t:      token indices per expert, uint32, interleaved L1
    #   [3] combine_output: [E*32, K_hidden] bfloat16, BLOCK_SHARDED L1
    #   [4] same as [3] (alias for compatibility with moe_compute convention)

    # ------------------------------------------------------------------
    # Step 3: selective_reduce_combine
    # Routes combined expert outputs back to tokens' originating devices.
    # Takes moe_gpt_outputs[3] directly (BLOCK_SHARDED L1) — no TM ops needed.
    # selective_reduce_combine uses set_globally_allocated_address to alias
    # its input CB to moe_gpt's combine shard, just like in test_moe_gpt_e2e.py.
    # Output shape per device: [K, M, H] where K = select_experts_k (top-k per token).
    # The preallocated output tensor shape determines the combine output layout.
    # Using [K, M, H] (not [experts_per_ring, M, H]) enables direct score application.
    # ------------------------------------------------------------------
    tt_combine_output = ttnn.experimental.selective_reduce_combine(
        moe_gpt_outputs[3],  # dense_input_tensor: BLOCK_SHARDED combine output from moe_gpt
        moe_gpt_outputs[1],  # dense_metadata_tensor:   expert activation metadata
        moe_gpt_outputs[2],  # dense_token_maps_tensor: token indices per expert
        moe_gpt_outputs[0],  # dense_token_counts_tensor: per-expert token counts
        hidden_size=config.hidden_size,
        batch_size=total_tokens,  # total ring tokens (e.g., 128 = M * ring_devices)
        seq_size=1,
        select_experts_k=config.num_experts_per_tok,
        experts=global_experts,  # total global experts: 128 for GPT-OSS
        cluster_axis=cluster_axis,
        topology=ttnn.Topology.Ring,
        num_links=fused_config.num_links,
        token_parallel_core_dim=fused_config.combine_token_parallel_core_dim,
        data_parallel_core_dim=fused_config.combine_data_parallel_core_dim,
        worker_cores=fused_config.combine_worker_cores,
        mux_core_range_set=fused_config.combine_mux_cores,
        output_tensor=fused_config.combine_preallocated,
        optional_cross_device_semaphore=fused_config.combine_semaphore,
    )

    # Deallocate all moe_gpt intermediate outputs (outputs[3] and [4] alias the same buffer)
    for i in range(4):
        ttnn.deallocate(moe_gpt_outputs[i])

    # ------------------------------------------------------------------
    # Post-processing: apply routing scores and reduce to [1, 1, M, H].
    #
    # Following the DeepSeek pattern (PR #39503):
    #   1. Tilize combine output
    #   2. Unsqueeze to 4D: [K, M, H] -> [K, 1, M, H]
    #   3. Permute pre-dispatch scores: [M, 1, 1, K] -> [K, 1, M, 1]
    #   4. Broadcast multiply: [K, 1, M, H] * [K, 1, M, 1] -> [K, 1, M, H]
    #   5. Sum over K dim -> [1, 1, M, H]
    #   6. All-reduce across columns (cluster_axis=1)
    #
    # This works because selective_reduce_combine with a [K, M, H] preallocated
    # output organizes results by top-k index (not ring-local expert ID), matching
    # the order of the routing scores.
    # ------------------------------------------------------------------
    K_sel = config.num_experts_per_tok

    tt_combine_tile = ttnn.to_layout(tt_combine_output, ttnn.TILE_LAYOUT)
    ttnn.deallocate(tt_combine_output)

    # Unsqueeze: [K, M, H] -> [K, 1, M, H]
    tt_unsqueezed = ttnn.unsqueeze(tt_combine_tile, dim=1)

    # Permute pre-dispatch scores: [M, 1, 1, K] -> [K, 1, M, 1]
    topk_weights = ttnn.permute(pre_dispatch_scores, (3, 1, 0, 2))
    topk_weights = ttnn.to_layout(topk_weights, ttnn.TILE_LAYOUT)

    # Scale expert outputs by routing scores
    tt_scaled = ttnn.mul(tt_unsqueezed, topk_weights)
    ttnn.deallocate(tt_unsqueezed)
    ttnn.deallocate(topk_weights)

    # Sum over K experts: [K, 1, M, H] -> [1, 1, M, H]
    tt_sum = ttnn.sum(tt_scaled, dim=0, keepdim=True)
    ttnn.deallocate(tt_scaled)

    # cluster_axis=1 (row direction) has fewer physical Ethernet links than cluster_axis=0.
    # rms_norm.py uses num_links=1 for cluster_axis=1; fused_config.num_links (=4) is for axis=0.
    tt_output = ttnn.all_reduce(
        tt_sum,
        num_links=1,
        topology=ttnn.Topology.Ring,
        cluster_axis=1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(tt_sum)

    return tt_output  # [1, 1, tokens_per_device, H]
