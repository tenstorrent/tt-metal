# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Prefill forward pass for throughput-optimized MoE experts.

This module implements the prefill path using all_to_all_dispatch and all_to_all_combine
to dynamically batch tokens across devices based on expert routing.
"""

import ttnn
from models.demos.gpt_oss.config import MeshConfig

from .config import AllToAllCombineConfig, AllToAllDispatchConfig, ThroughputExpertConfig, ThroughputProgramConfig
from .decode import decode_forward as forward
from .weights import ThroughputExpertWeights


def prefill_forward_chunked(
    hidden_states: ttnn.Tensor,
    topk_expert_indices: ttnn.Tensor,
    topk_expert_weights: ttnn.Tensor,
    weights: ThroughputExpertWeights,
    config: ThroughputExpertConfig,
    expert_mapping_tensors: ttnn.Tensor,
    remap_topk_mask: ttnn.Tensor,
    dispatch_config: AllToAllDispatchConfig,
    combine_config: AllToAllCombineConfig,
    program_config: ThroughputProgramConfig,
    mesh_device,
    mesh_config: MeshConfig,
    ccl_manager,
    chunk_size: int,
) -> ttnn.Tensor:
    """Chunked prefill forward pass for very long sequences.

    Splits the sequence into chunks to manage memory usage, processes each
    chunk through the MoE layer, and concatenates results.

    Args:
        hidden_states: Input tensor [seq_len_per_device, 1, 1, hidden_size]
        topk_expert_indices: Expert indices [seq_len_per_device, 1, 1, num_experts_per_tok]
        topk_expert_weights: Routing weights [seq_len_per_device, 1, 1, num_experts_per_tok]
        weights: Expert weights
        config: Expert configuration
        expert_mapping_tensors: Device-to-expert mapping
        remap_topk_mask: Mask for expert remapping
        dispatch_config: Configuration for all_to_all_dispatch
        combine_config: Configuration for all_to_all_combine
        program_config: Matmul program configuration
        mesh_device: TTNN mesh device
        chunk_size: Maximum tokens per chunk (default: 2048)

    Returns:
        Output tensor [seq_len_per_device, 1, 1, hidden_size]
    """
    seq_len_per_device = hidden_states.shape[2]

    # If sequence fits in one chunk, use regular forward
    if seq_len_per_device <= chunk_size:
        return forward(
            hidden_states,
            topk_expert_indices,
            topk_expert_weights,
            weights,
            config,
            expert_mapping_tensors,
            remap_topk_mask,
            dispatch_config,
            combine_config,
            program_config,
            mesh_device,
            mesh_config,
            ccl_manager,
        )

    # Split into chunks
    hidden_chunks = ttnn.split(hidden_states, chunk_size, dim=2)
    indices_chunks = ttnn.split(topk_expert_indices, chunk_size, dim=0)
    weights_chunks = ttnn.split(topk_expert_weights, chunk_size, dim=0)

    ttnn.deallocate(hidden_states)
    ttnn.deallocate(topk_expert_indices)
    ttnn.deallocate(topk_expert_weights)

    # Process each chunk
    output_chunks = []
    for h_chunk, i_chunk, w_chunk in zip(hidden_chunks, indices_chunks, weights_chunks):
        chunk_output = forward(
            h_chunk,
            i_chunk,
            w_chunk,
            weights,
            config,
            expert_mapping_tensors,
            remap_topk_mask,
            dispatch_config,
            combine_config,
            program_config,
            mesh_device,
            mesh_config,
            ccl_manager,
        )
        output_chunks.append(chunk_output)

        ttnn.deallocate(h_chunk)
        ttnn.deallocate(i_chunk)
        ttnn.deallocate(w_chunk)

    # Concatenate outputs
    output = ttnn.concat(output_chunks, dim=2)
    for chunk in output_chunks:
        ttnn.deallocate(chunk)

    return output
