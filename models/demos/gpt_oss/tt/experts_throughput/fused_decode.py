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
    → selective_reduce_combine → score weighting → sum → all_reduce

Key differences:
  - moe_gpt only processes tokens actually routed to each expert (sparse, not dense)
  - W0/W1/SwiGLU/A2A ring/W2/combine are fused into a single kernel
  - No need to repeat input across the expert dimension
  - selective_reduce_combine aliases moe_gpt's BLOCK_SHARDED combine output directly
    via set_globally_allocated_address — no TM ops needed between moe_gpt and combine
  - Score weighting: combine outputs [K, M, H], permute scores [M, 1, 1, K] -> [K, 1, M, 1],
    broadcast multiply, sum over K dim. No host round-trip needed.

The existing decode_forward (dense flow) is unchanged; this module adds an alternative
path that the ThroughputExperts class can route to when fused_config is provided.
"""

from math import prod

import torch
from loguru import logger

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
        Tensor [1, 1, tokens_per_device, H] in L1. Routing scores are applied by
        permuting scores to [K, 1, M, 1], broadcast-multiplying with the [K, 1, M, H]
        combine output, and summing over the K dimension. No host round-trip.
    """
    # Ensure hidden_states is in ROW_MAJOR L1 (all_to_all_dispatch_metadata requires both).
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
    global_experts = config.num_experts
    num_clusters = prod(mesh_device.shape) // num_dispatch_devices

    input_shape = hidden_states.shape
    tokens_per_device = input_shape[0] * input_shape[2]
    total_tokens = tokens_per_device * num_dispatch_devices
    K_sel = config.num_experts_per_tok

    # Reshape hidden_states to [M, 1, 1, H] for dispatch
    hidden_states = ttnn.reshape(hidden_states, (tokens_per_device, 1, 1, config.hidden_size))

    # Format conversion: router outputs [M, K] uint16 TILE but dispatch needs
    # [M, 1, 1, K] uint16 HEIGHT_SHARDED L1 ROW_MAJOR. Since ttnn.reshape doesn't
    # support uint16, we read to host and re-upload. This is a temporary workaround
    # until the router outputs the correct format directly.
    # TODO: fix router (topk.py) to output [M, 1, 1, K] HEIGHT_SHARDED L1 ROW_MAJOR
    mesh_rows, mesh_cols = mesh_device.shape
    dev_idx_list = ttnn.get_device_tensors(topk_expert_indices)
    dev_scores_list = ttnn.get_device_tensors(topk_expert_scores)
    host_indices = [ttnn.to_torch(t).reshape(tokens_per_device, K_sel).int() for t in dev_idx_list]
    host_scores = [ttnn.to_torch(t).reshape(tokens_per_device, K_sel) for t in dev_scores_list]

    ttnn.deallocate(topk_expert_indices)
    ttnn.deallocate(topk_expert_scores)

    # Re-upload scores to DRAM interleaved for post-combine weighting.
    # We already have host_scores from the host round-trip above.
    # Shape: [M, 1, 1, K] per row, sharded across rows, replicated across cols.
    scores_for_weight = torch.cat(
        [
            host_scores[row * mesh_cols].reshape(tokens_per_device, 1, 1, K_sel).to(torch.bfloat16)
            for row in range(mesh_rows)
        ]
    )
    tt_scores_copy = ttnn.from_torch(
        scores_for_weight,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(mesh_rows, mesh_cols)),
    )

    num_cores_y = min(8, tokens_per_device)
    num_cores_x = (tokens_per_device + num_cores_y - 1) // num_cores_y
    dispatch_shard_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}),
            [1, K_sel],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    idx_stacked = torch.cat(
        [host_indices[row * mesh_cols].reshape(tokens_per_device, 1, 1, K_sel) for row in range(mesh_rows)]
    )
    scores_stacked = torch.cat(
        [host_scores[row * mesh_cols].reshape(tokens_per_device, 1, 1, K_sel) for row in range(mesh_rows)]
    )

    topk_expert_indices = ttnn.from_torch(
        idx_stacked.to(torch.int16),
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=dispatch_shard_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(mesh_rows, mesh_cols)),
    )
    topk_expert_scores = ttnn.from_torch(
        scores_stacked.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=dispatch_shard_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(mesh_rows, mesh_cols)),
    )

    # ------------------------------------------------------------------
    # Step 1: all_to_all_dispatch_metadata
    # ------------------------------------------------------------------
    logger.info("fused_decode: Step 1 - all_to_all_dispatch_metadata")
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
    ttnn.synchronize_device(mesh_device)
    logger.info("fused_decode: Step 1 sync done")
    ttnn.deallocate(hidden_states)

    tt_sparse_l1 = ttnn.to_memory_config(tt_sparse, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_sparse_l1 = ttnn.reshape(tt_sparse_l1, [total_tokens, config.hidden_size])

    # ------------------------------------------------------------------
    # Step 2: moe_gpt (fused sparse compute)
    # ------------------------------------------------------------------
    logger.info("fused_decode: Step 2 - moe_gpt")
    moe_gpt_outputs = ttnn.experimental.moe_gpt(
        tt_sparse_l1,
        expert_indices=tt_indices,
        expert_scores=tt_scores,
        expert_mapping=fused_config.tt_moe_gpt_mapping,
        w0_w1_tensor=fused_config.tt_w0_w1,
        w2_tensor=fused_config.tt_w2,
        cluster_axis=cluster_axis,
    )
    ttnn.synchronize_device(mesh_device)
    logger.info("fused_decode: Step 2 sync done")
    ttnn.deallocate(tt_sparse_l1)

    # ------------------------------------------------------------------
    logger.info("fused_decode: Step 3 - selective_reduce_combine")
    # Step 3: selective_reduce_combine
    # With the K-indexed fix (upstream #38542), the combine writer uses the
    # token_activations metadata to look up each token's K-index, so the
    # output is [select_experts_k, M, H] instead of [experts_per_ring, M, H].
    # ------------------------------------------------------------------
    tt_combine_output = ttnn.experimental.selective_reduce_combine(
        moe_gpt_outputs[4],
        moe_gpt_outputs[1],
        moe_gpt_outputs[2],
        moe_gpt_outputs[0],
        hidden_size=config.hidden_size,
        batch_size=total_tokens,
        seq_size=1,
        select_experts_k=K_sel,
        experts=global_experts,
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
    ttnn.synchronize_device(mesh_device)
    logger.info("fused_decode: Step 3 sync done")

    for i in range(4):  # output[3] and [4] share a buffer; deallocating [3] frees both
        ttnn.deallocate(moe_gpt_outputs[i])

    # ------------------------------------------------------------------
    # Post-processing: apply routing scores and reduce over K dimension.
    #
    # combine_output: [K, M, H] (3D, DRAM ROW_MAJOR)
    # Following the DeepSeek pattern (PR #39503):
    # 1. Tilize and unsqueeze to [K, 1, M, H]
    # 2. Permute saved scores [M, 1, 1, K] -> [K, 1, M, 1]
    # 3. Broadcast multiply: [K, 1, M, H] * [K, 1, M, 1] -> [K, 1, M, H]
    # 4. Sum over K dim -> [1, 1, M, H]
    # 5. All-reduce across columns (cluster_axis=1)
    # ------------------------------------------------------------------
    logger.info("fused_decode: Step 4 - post-processing (tilize, score weighting, sum, all_reduce)")
    tt_combine_tile = ttnn.to_layout(tt_combine_output, ttnn.TILE_LAYOUT)
    # Note: do NOT deallocate tt_combine_output — it aliases fused_config.combine_preallocated
    # which must persist across layers for program cache reuse.

    tt_4d = ttnn.unsqueeze(tt_combine_tile, dim=1)  # [K, 1, M, H]

    # Permute scores: [M, 1, 1, K] -> [K, 1, M, 1]
    tt_scores_perm = ttnn.permute(tt_scores_copy, (3, 1, 0, 2))
    ttnn.deallocate(tt_scores_copy)
    tt_scores_perm = ttnn.to_layout(tt_scores_perm, ttnn.TILE_LAYOUT)

    tt_weighted = ttnn.mul(tt_4d, tt_scores_perm)  # [K, 1, M, H]
    ttnn.deallocate(tt_4d)
    ttnn.deallocate(tt_scores_perm)

    tt_sum = ttnn.sum(tt_weighted, dim=0, keepdim=True)  # [1, 1, M, H]
    ttnn.deallocate(tt_weighted)

    logger.info("fused_decode: Step 4d - all_reduce")
    tt_output = ttnn.all_reduce(
        tt_sum,
        num_links=1,
        topology=ttnn.Topology.Ring,
        cluster_axis=1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(tt_sum)

    ttnn.synchronize_device(mesh_device)
    logger.info("fused_decode: Done")
    return tt_output  # [1, 1, tokens_per_device, H]
