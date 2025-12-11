# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Prefill forward pass for throughput-optimized MoE experts.

This module implements the prefill path using all_to_all_dispatch and all_to_all_combine
to dynamically batch tokens across devices based on expert routing.
"""

import ttnn

from .config import AllToAllCombineConfig, AllToAllDispatchConfig, ThroughputExpertConfig, ThroughputProgramConfig
from .weights import ThroughputExpertWeights


def _apply_silu_mul(w1_out: ttnn.Tensor, w3_out: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    """Apply SiLU activation and multiply: silu(w1_out) * w3_out.

    This implements the standard SwiGLU activation: silu(gate) * up

    Args:
        w1_out: Gate projection output
        w3_out: Up projection output
        memory_config: Output memory configuration

    Returns:
        Activated tensor
    """
    # Apply SiLU to gate output
    w1_activated = ttnn.silu(w1_out)
    ttnn.deallocate(w1_out)

    # Multiply with up projection
    result = ttnn.mul(w1_activated, w3_out, memory_config=memory_config)
    ttnn.deallocate(w1_activated)
    ttnn.deallocate(w3_out)

    return result


def _apply_swiglu(
    gate: ttnn.Tensor, up: ttnn.Tensor, alpha: float, limit: float, memory_config: ttnn.MemoryConfig
) -> ttnn.Tensor:
    """Apply SwiGLU activation: (up + 1) * (gate * sigmoid(gate * alpha)).

    This implements the GPT-OSS SwiGLU variant with clamping:
    1. Clamp gate (min=None, max=limit) and up (min=-limit, max=limit)
    2. glu = gate * sigmoid(gate * alpha)
    3. result = (up + 1) * glu

    Args:
        gate: Gate projection output
        up: Up projection output
        alpha: Scaling factor for sigmoid
        limit: Clamping limit for swiglu
        memory_config: Output memory configuration

    Returns:
        Activated tensor
    """
    # Clamp gate (max only)
    gate = ttnn.clamp(gate, min=None, max=limit)

    # Clamp up (both min and max)
    up = ttnn.clamp(up, min=-limit, max=limit)

    # Compute gate_alpha = gate * alpha
    gate_alpha = ttnn.mul(gate, alpha)

    # Compute gate_sigmoid = sigmoid(gate_alpha)
    gate_sigmoid = ttnn.sigmoid(gate_alpha)
    ttnn.deallocate(gate_alpha)

    # Compute glu = gate * gate_sigmoid
    glu = ttnn.mul(gate, gate_sigmoid, memory_config=memory_config)
    ttnn.deallocate(gate)
    ttnn.deallocate(gate_sigmoid)

    # Add 1 to up: up = up + 1
    up = ttnn.add(up, 1.0)

    # Multiply: result = up * glu
    result = ttnn.mul(up, glu, memory_config=memory_config)
    ttnn.deallocate(up)
    ttnn.deallocate(glu)

    return result


def prefill_forward(
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
) -> ttnn.Tensor:
    """Prefill forward pass with all_to_all dispatch and combine.

    This implements the MoE forward pass for prefill (seq_len > 1):
    1. all_to_all_dispatch: Route tokens to devices based on expert assignments
    2. Expert computation: Run gate/up/down projections on local experts
    3. all_to_all_combine: Route expert outputs back to original token positions
    4. Apply routing weights and reduce across experts

    Note: In prefill mode, the batch_size dimension is used for sequence length
    since all_to_all operations require DP=num_dispatch_devices. The seq_len
    dimension is interchanged with batch_size for the operation.

    Args:
        hidden_states: Input tensor [seq_len_per_device, 1, 1, hidden_size]
                       (seq_len treated as batch for dispatch)
        topk_expert_indices: Expert indices [seq_len_per_device, 1, 1, num_experts_per_tok]
        topk_expert_weights: Routing weights [seq_len_per_device, 1, 1, num_experts_per_tok]
        weights: Expert weights (sharded by device)
        config: Expert configuration
        expert_mapping_tensors: Device-to-expert mapping
        remap_topk_mask: Mask for expert remapping
        dispatch_config: Configuration for all_to_all_dispatch
        combine_config: Configuration for all_to_all_combine
        program_config: Matmul program configuration
        mesh_device: TTNN mesh device

    Returns:
        Output tensor [seq_len_per_device, 1, 1, hidden_size]
    """
    # In prefill, we treat sequence tokens as batch elements for dispatch
    # This allows dynamic routing of tokens to experts across all devices
    seq_len = 1  # Treated as 1 for dispatch (tokens are in batch dim)
    batch_size_per_device = hidden_states.shape[0]  # Actually seq_len_per_device
    num_dispatch_devices = mesh_device.shape[0]
    total_tokens = batch_size_per_device * num_dispatch_devices

    # Use DRAM for prefill due to larger tensor sizes
    memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Prepare inputs in row-major layout for all_to_all operations
    hidden_rm = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
    hidden_rm = ttnn.reshape(hidden_rm, shape=(batch_size_per_device, 1, seq_len, config.hidden_size))

    topk_indices_rm = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
    topk_indices_rm = ttnn.reshape(
        topk_indices_rm, shape=(batch_size_per_device, 1, seq_len, config.num_experts_per_tok)
    )

    # ========== 1. ALL_TO_ALL_DISPATCH ==========
    # Dispatch tokens to devices based on expert routing
    dispatch_output, dispatch_metadata = ttnn.all_to_all_dispatch(
        hidden_rm,
        topk_indices_rm,
        expert_mapping_tensors,
        **dispatch_config.as_dict(),
    )

    # Reshape for expert computation
    post_dispatch = ttnn.reshape(dispatch_output, shape=(1, 1, total_tokens * seq_len, config.hidden_size))
    post_dispatch = ttnn.to_layout(post_dispatch, ttnn.TILE_LAYOUT)

    # ========== 2. EXPERT TOKEN REMAP (for sparsity) ==========
    # Repeat remap mask for GLOBAL batch (total_tokens) to match dispatch_metadata
    # dispatch_metadata has shape [D, total_tokens, S, K]
    remap_mask = ttnn.repeat(remap_topk_mask, ttnn.Shape((1, total_tokens, 1, 1)))

    # Get sparsity pattern for sparse matmul
    _, sparsity = ttnn.moe_expert_token_remap(
        remap_mask,
        expert_mapping_tensors,
        dispatch_metadata,
        reduction_size=config.sparsity_block_size,
    )

    # ========== 3. EXPERT COMPUTATION ==========
    # Reshape for sparse expert computation
    num_tokens = total_tokens * seq_len
    num_sparse_blocks = num_tokens // config.sparsity_block_size
    expert_input = ttnn.reshape(
        post_dispatch,
        shape=(1, num_sparse_blocks, config.sparsity_block_size, config.hidden_size),
    )

    # Gate projection (w1)
    w1_out = ttnn.sparse_matmul(
        expert_input,
        weights.w1,
        sparsity=sparsity,
        memory_config=memory_config,
        program_config=program_config.get_gate_up_config(config.intermediate_size),
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        output_tile=ttnn.Tile([config.sparsity_block_size, ttnn.TILE_SIZE]),
    )

    # Add gate bias
    # w1_out shape: [1, num_sparse_blocks, 1, num_experts_per_device, block_size, intermediate]
    # Bias shape: [1, 1, 1, num_experts_per_device, 1, intermediate] - broadcasts correctly
    w1_out = ttnn.add(w1_out, weights.w1_bias)

    # Up projection (w3)
    w3_out = ttnn.sparse_matmul(
        expert_input,
        weights.w3,
        sparsity=sparsity,
        memory_config=memory_config,
        program_config=program_config.get_gate_up_config(config.intermediate_size),
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        output_tile=ttnn.Tile([config.sparsity_block_size, ttnn.TILE_SIZE]),
    )

    # Add up bias
    # w3_out shape: [1, num_sparse_blocks, 1, num_experts_per_device, block_size, intermediate]
    # Bias shape: [1, 1, 1, num_experts_per_device, 1, intermediate] - broadcasts correctly
    w3_out = ttnn.add(w3_out, weights.w3_bias)

    # Apply activation: (up + 1) * (gate * sigmoid(gate * alpha))
    activated = _apply_swiglu(w1_out, w3_out, config.alpha, config.swiglu_limit, memory_config)

    # For testing standard SiLU activation instead of SwiGLU, uncomment below and comment above:
    # activated = _apply_silu_mul(w1_out, w3_out, memory_config)

    # Reshape for down projection
    activated = ttnn.squeeze(activated, 0)
    activated = ttnn.squeeze(activated, 1)

    # Down projection (w2)
    expert_output = ttnn.sparse_matmul(
        activated,
        weights.w2,
        sparsity=sparsity,
        memory_config=memory_config,
        program_config=program_config.get_down_config(config.hidden_size),
        is_input_a_sparse=True,
        is_input_b_sparse=False,
        output_tile=ttnn.Tile([config.sparsity_block_size, ttnn.TILE_SIZE]),
    )
    ttnn.deallocate(activated)

    # Add down projection bias
    # expert_output shape: [num_sparse_blocks, num_experts_per_device, block_size, hidden]
    # Bias shape: [1, 1, 1, num_experts_per_device, 1, hidden] -> reshape to [1, num_experts_per_device, 1, hidden]
    w2_bias_4d = ttnn.reshape(weights.w2_bias, (1, config.num_experts_per_device, 1, config.hidden_size))
    expert_output = ttnn.add(expert_output, w2_bias_4d)

    # Reshape expert output for combine
    expert_output = ttnn.permute(expert_output, (1, 0, 2, 3))
    expert_output = ttnn.reshape(
        expert_output,
        shape=(config.num_experts_per_device, total_tokens, seq_len, config.hidden_size),
    )
    expert_output = ttnn.to_layout(expert_output, ttnn.ROW_MAJOR_LAYOUT)

    # ========== 4. ALL_TO_ALL_COMBINE ==========
    # Route expert outputs back to original token positions
    combine_output = ttnn.all_to_all_combine(
        expert_output,
        dispatch_metadata,
        expert_mapping_tensors,
        **combine_config.as_dict(),
    )

    # Reshape combine output
    post_combine = ttnn.reshape(
        combine_output,
        shape=(config.num_experts_per_tok, 1, batch_size_per_device * seq_len, config.hidden_size),
    )
    post_combine = ttnn.to_layout(post_combine, ttnn.TILE_LAYOUT)

    # ========== 5. APPLY ROUTING WEIGHTS AND REDUCE ==========
    # Prepare routing weights for multiplication
    topk_weights_rm = ttnn.to_layout(topk_expert_weights, ttnn.ROW_MAJOR_LAYOUT)
    topk_weights_rm = ttnn.repeat(topk_weights_rm, ttnn.Shape((config.hidden_size, 1, 1, 1)))
    topk_weights_rm = ttnn.permute(topk_weights_rm, (3, 1, 2, 0))
    topk_weights = ttnn.to_layout(topk_weights_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(topk_weights_rm)

    # Multiply expert outputs by routing weights
    weighted_output = ttnn.mul(post_combine, topk_weights, memory_config=memory_config)
    ttnn.deallocate(post_combine)
    ttnn.deallocate(topk_weights)

    # Sum across experts
    output = ttnn.sum(weighted_output, dim=0, keepdim=True)
    ttnn.deallocate(weighted_output)

    return output


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
    chunk_size: int = 2048,
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
    seq_len_per_device = hidden_states.shape[0]

    # If sequence fits in one chunk, use regular forward
    if seq_len_per_device <= chunk_size:
        return prefill_forward(
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
        )

    # Split into chunks
    hidden_chunks = ttnn.split(hidden_states, chunk_size, dim=0)
    indices_chunks = ttnn.split(topk_expert_indices, chunk_size, dim=0)
    weights_chunks = ttnn.split(topk_expert_weights, chunk_size, dim=0)

    ttnn.deallocate(hidden_states)
    ttnn.deallocate(topk_expert_indices)
    ttnn.deallocate(topk_expert_weights)

    # Process each chunk
    output_chunks = []
    for h_chunk, i_chunk, w_chunk in zip(hidden_chunks, indices_chunks, weights_chunks):
        chunk_output = prefill_forward(
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
        )
        output_chunks.append(chunk_output)

        ttnn.deallocate(h_chunk)
        ttnn.deallocate(i_chunk)
        ttnn.deallocate(w_chunk)

    # Concatenate outputs
    output = ttnn.concat(output_chunks, dim=2)

    return output
