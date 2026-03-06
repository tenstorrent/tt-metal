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
        return format. Routing weights are NOT applied (unweighted sum across expert dim),
        so PCC vs. a reference with bias+weighting will be off — expected until
        proper routing-weight application is implemented.
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
    # ring_local_experts = global_experts // num_clusters = 128 // 8 = 16 (experts visible per ring).
    global_experts = config.num_experts  # total global experts (e.g., 128 for GPT-OSS)
    num_clusters = prod(mesh_device.shape) // num_dispatch_devices  # 8 independent rings for (4,8)

    # Reshape hidden_states to 2D for dispatch: [tokens_per_device, H]
    # dispatch expects input with tokens on dim -2 (as [1, 1, tokens, H])
    input_shape = hidden_states.shape
    tokens_per_device = input_shape[0] * input_shape[2]
    total_tokens = tokens_per_device * num_dispatch_devices

    hidden_states = ttnn.reshape(hidden_states, (1, 1, tokens_per_device, config.hidden_size))
    topk_expert_indices = ttnn.reshape(topk_expert_indices, (1, 1, tokens_per_device, config.num_experts_per_tok))
    topk_expert_scores = ttnn.reshape(topk_expert_scores, (1, 1, tokens_per_device, config.num_experts_per_tok))

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
    # Output shape per device: [ring_local_experts, M, H]
    # where ring_local_experts = global_experts // num_clusters = 128 // 8 = 16 for GPT-OSS.
    # experts_per_device = global_experts / total_devices = 128 / 32 = 4 ✓
    # ------------------------------------------------------------------
    ring_local_experts = global_experts // num_clusters  # 16 for GPT-OSS
    tt_combine_output = ttnn.experimental.selective_reduce_combine(
        moe_gpt_outputs[3],  # dense_input_tensor:      combine output from moe_gpt
        moe_gpt_outputs[1],  # dense_metadata_tensor:   expert activation metadata
        moe_gpt_outputs[2],  # dense_token_maps_tensor: token indices per expert
        moe_gpt_outputs[0],  # dense_token_counts_tensor: per-expert token counts
        hidden_size=config.hidden_size,
        batch_size=tokens_per_device,  # per-device token count (ring handles full routing)
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

    # Deallocate moe_gpt intermediate outputs (output[3] and [4] alias the same buffer)
    for i in range(3):
        ttnn.deallocate(moe_gpt_outputs[i])

    # ------------------------------------------------------------------
    # Post-processing: reduce combine output to [1, 1, M, H] for compatibility
    # with the dense decode_forward return format.
    #
    # combine_output: [ring_local_experts, M, H] (3D, DRAM ROW_MAJOR)
    # Step 1: Convert to TILE_LAYOUT for ttnn.sum
    # Step 2: Reshape to 4D [1, ring_local_experts, M, H]
    # Step 3: Sum over expert dim → [1, 1, M, H]
    #   Note: routing weights NOT applied here (requires expert→k-slot matching).
    #   PCC vs. reference will be off (expected; caller can add weighting later).
    # Step 4: All-reduce across columns (cluster_axis=1) to aggregate results
    #   from the 8 independent column rings.
    # ------------------------------------------------------------------
    tt_combine_tile = ttnn.to_layout(tt_combine_output, ttnn.TILE_LAYOUT)
    ttnn.deallocate(tt_combine_output)

    tt_4d = ttnn.reshape(tt_combine_tile, [1, ring_local_experts, tokens_per_device, config.hidden_size])
    ttnn.deallocate(tt_combine_tile)

    tt_sum = ttnn.sum(tt_4d, dim=1, keepdim=True)  # [1, 1, M, H]
    ttnn.deallocate(tt_4d)

    tt_output = ttnn.all_reduce(
        tt_sum,
        num_links=fused_config.num_links,
        topology=ttnn.Topology.Ring,
        cluster_axis=1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(tt_sum)

    return tt_output  # [1, 1, tokens_per_device, H]
