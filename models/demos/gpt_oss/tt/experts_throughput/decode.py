# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Decode forward pass for throughput-optimized MoE experts.

This module implements the decode path using all_to_all_dispatch and all_to_all_combine
to dynamically batch tokens across devices based on expert routing.

The MoE forward pass flow is:
1. all_to_all_dispatch: Route tokens to devices based on expert assignments
2. moe_expert_token_remap: Create sparsity pattern for efficient sparse matmul
3. Expert computation: Run gate/up/down projections using sparse matmul
4. all_to_all_combine: Route expert outputs back to original token positions
5. Apply routing weights and reduce across experts
"""

from math import prod

import ttnn

from .config import AllToAllCombineConfig, AllToAllDispatchConfig, ThroughputExpertConfig, ThroughputProgramConfig
from .weights import ThroughputExpertWeights


def prepare_expert_weights(
    topk_expert_weights: ttnn.Tensor,
    num_experts_per_tok: int,
    hidden_size: int,
) -> ttnn.Tensor:
    """Prepare routing weights for element-wise multiplication with expert outputs.

    Transforms routing weights from [B, 1, S, K] to [K, 1, B*S, H] format for
    broadcasting with post-combine expert outputs.

    Args:
        topk_expert_weights: Routing weights [batch_size, 1, seq_len, num_experts_per_tok]
        num_experts_per_tok: Number of experts selected per token (K)
        hidden_size: Hidden dimension size (H)

    Returns:
        Transformed weights tensor [K, 1, B*S, H] in TILE layout, ready for
        element-wise multiplication with expert outputs
    """
    topk_expert_weights = ttnn.reshape(topk_expert_weights, (-1, 1, 1, num_experts_per_tok))
    topk_weights_rm = ttnn.to_layout(topk_expert_weights, ttnn.ROW_MAJOR_LAYOUT)
    topk_weights_rm = ttnn.repeat(topk_weights_rm, ttnn.Shape((1, 1, hidden_size, 1)))
    topk_weights_rm = ttnn.permute(topk_weights_rm, (3, 1, 0, 2))
    topk_weights_reshaped = ttnn.to_layout(topk_weights_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(topk_weights_rm)
    return topk_weights_reshaped


def _apply_swiglu(
    gate: ttnn.Tensor, up: ttnn.Tensor, alpha: float, limit: float, memory_config: ttnn.MemoryConfig
) -> ttnn.Tensor:
    """Apply SwiGLU activation: (up + 1) * (gate * sigmoid(gate * alpha)).

    This implements the GPT-OSS SwiGLU variant with clamping:
    1. Clamp gate (min=None, max=limit) and up (min=-limit, max=limit)
    2. glu = gate * sigmoid(gate * alpha)
    3. result = (up + 1) * glu

    Args:
        gate: Gate projection output [batch, experts, tokens, intermediate]
        up: Up projection output [batch, experts, tokens, intermediate]
        alpha: Scaling factor for sigmoid
        limit: Clamping limit for swiglu
        memory_config: Output memory configuration

    Returns:
        Activated tensor with same shape as inputs
    """
    # Clamp gate (max only)
    gate_clamped = ttnn.clamp(gate, min=None, max=limit)
    ttnn.deallocate(gate)

    # Clamp up (both min and max)
    up_clamped = ttnn.clamp(up, min=-limit, max=limit)
    ttnn.deallocate(up)

    # Compute gate_alpha = gate * alpha
    gate_alpha = ttnn.mul(gate_clamped, alpha)

    # Compute gate_sigmoid = sigmoid(gate_alpha)
    gate_sigmoid = ttnn.sigmoid(gate_alpha)
    ttnn.deallocate(gate_alpha)

    # Compute glu = gate * gate_sigmoid
    glu = ttnn.mul(gate_clamped, gate_sigmoid, memory_config=memory_config)
    ttnn.deallocate(gate_clamped)
    ttnn.deallocate(gate_sigmoid)

    # Add 1 to up: up = up + 1
    up_clamped = ttnn.add(up_clamped, 1.0, output_tensor=up_clamped)

    # Multiply: result = up * glu
    result = ttnn.mul(up_clamped, glu, memory_config=memory_config)
    ttnn.deallocate(up_clamped)
    ttnn.deallocate(glu)

    return result


def expert_mlp_forward(
    experts_input: ttnn.Tensor,
    sparsity: ttnn.Tensor,
    weights: ThroughputExpertWeights,
    config: ThroughputExpertConfig,
    program_config: ThroughputProgramConfig,
    memory_config: ttnn.MemoryConfig,
    total_tokens: int,
    mesh_device=None,
    save_intermediate: bool = False,
) -> ttnn.Tensor:
    """Compute expert MLP forward pass with sparse matmul.

    This implements the expert computation portion of MoE:
    output = down((up + 1) * (gate * sigmoid(gate * alpha)))

    Using sparse matmul to only compute (token_block, expert) pairs where
    tokens are actually routed, significantly reducing computation.

    Args:
        experts_input: Dispatch output in TILE layout [1, 1, B*S, H]
        sparsity: Sparsity tensor indicating active (token_block, expert) pairs
        weights: Expert weights (w1, w2, w3 and biases)
        config: Expert configuration
        program_config: Matmul program configuration
        memory_config: Output memory configuration
        batch_size: Global batch size (batch_per_device * num_devices)
        seq_len: Sequence length
        mesh_device: TTNN mesh device (optional, for debugging)
        save_intermediate: Whether to save intermediate tensors for debugging

    Returns:
        Expert output tensor [experts_per_device, batch_size, seq_len, hidden_size]
        in ROW_MAJOR layout, ready for all_to_all_combine
    """
    # Reshape to sparse block format for matmul
    # Note: reshape returns a view - don't deallocate post_dispatch separately
    num_sparse_blocks = total_tokens // config.sparsity_block_size
    reshaped_expert_input = ttnn.reshape(
        experts_input,
        shape=(1, num_sparse_blocks, config.sparsity_block_size, config.hidden_size),
    )
    # ttnn.deallocate(experts_input)

    # ==========================================================================
    # Gate/Up/Down projections with sparse matmul
    # ==========================================================================
    # Expert MLP: output = down((up + 1) * (gate * sigmoid(gate * alpha)))
    #
    # sparse_matmul only computes (token_block, expert) pairs where sparsity=1,
    # significantly reducing computation for sparse expert activation patterns.

    # Gate projection (w1): [B*S/block, block, H] x [experts, H, I] -> [B*S/block, experts, block, I]
    w1_out = ttnn.sparse_matmul(
        reshaped_expert_input,
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
    w1_out = ttnn.add(w1_out, weights.w1_bias, output_tensor=w1_out)

    # Up projection (w3): same shape as gate
    w3_out = ttnn.sparse_matmul(
        reshaped_expert_input,
        weights.w3,
        sparsity=sparsity,
        memory_config=memory_config,
        program_config=program_config.get_gate_up_config(config.intermediate_size),
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        output_tile=ttnn.Tile([config.sparsity_block_size, ttnn.TILE_SIZE]),
    )
    ttnn.deallocate(reshaped_expert_input)

    # Add up bias
    # w3_out shape: [1, num_sparse_blocks, 1, num_experts_per_device, block_size, intermediate]
    # Bias shape: [1, 1, 1, num_experts_per_device, 1, intermediate] - broadcasts correctly
    w3_out = ttnn.add(w3_out, weights.w3_bias, output_tensor=w3_out)

    # SwiGLU activation: (up + 1) * (gate * sigmoid(gate * alpha))
    activated = _apply_swiglu(w1_out, w3_out, config.alpha, config.swiglu_limit, memory_config)

    # For testing standard SiLU activation instead of SwiGLU, uncomment below and comment above:
    # activated = _apply_silu_mul(w1_out, w3_out, memory_config)

    # Squeeze batch dimensions for down projection
    # From: [1, B*S/block, experts, block, I]
    # To: [B*S/block, experts, block, I]
    # Note: squeeze likely returns views - don't deallocate originals
    activated = ttnn.squeeze(activated, 0)
    activated = ttnn.squeeze(activated, 1)

    # Down projection (w2): [B*S/block, experts, block, I] x [experts, I, H] -> [B*S/block, experts, block, H]
    expert_output_sparse = ttnn.sparse_matmul(
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
    ttnn.deallocate(sparsity)

    # Add down projection bias
    # expert_output shape: [num_sparse_blocks, num_experts_per_device, block_size, hidden]
    # Bias shape: [1, num_experts_per_device, 1, hidden] - broadcasts correctly after squeeze
    expert_output_sparse = ttnn.add(expert_output_sparse, weights.w2_bias)

    # ==========================================================================
    # STEP 6: PREPARE EXPERT OUTPUT FOR ALL_TO_ALL_COMBINE
    # ==========================================================================
    while len(expert_output_sparse.shape) > 4:
        expert_output_sparse = ttnn.squeeze(expert_output_sparse, 0)

    # Reshape from sparse matmul output to format expected by combine:
    # From: [total_tokens/block, experts, block, H]
    # To: [experts_per_device, 1, total_tokens, H] (ROW_MAJOR, tokens on dim -2)
    #
    # Permute to get experts_per_device as first dimension (what combine expects)
    # permute creates a new tensor - safe to deallocate original
    expert_output = ttnn.permute(expert_output_sparse, (1, 0, 2, 3))
    ttnn.deallocate(expert_output_sparse)
    # Note: reshape returns a view, to_layout creates new tensor
    # With tokens on dim -2: [experts_per_device, 1, total_tokens, H]
    expert_output = ttnn.reshape(
        expert_output,
        shape=(config.num_experts_per_device, 1, total_tokens, config.hidden_size),
    )
    expert_output_tiled = expert_output
    expert_output = ttnn.to_layout(expert_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(expert_output_tiled)  # Deallocates the permute output via its reshape view

    return expert_output


def decode_forward(
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
    """Decode forward pass with all_to_all dispatch and combine.

    This implements the MoE forward pass for decode (seq_len=1 per batch element).

    Tensor shape conventions:
    - Input hidden_states: [batch_per_device, 1, seq_len, hidden_size] - TILE layout
    - Expert indices: [batch_per_device, 1, seq_len, num_experts_per_tok] - ROW_MAJOR, uint16
    - Expert mapping: [1, 1, num_experts, num_devices] - ROW_MAJOR, uint16
    - Remap mask: [1, num_dispatch_rows, 1, num_experts] - ROW_MAJOR, bfloat16

    Args:
        hidden_states: Input tensor [batch_size_per_device, 1, seq_len, hidden_size]
        topk_expert_indices: Expert indices per token [batch_per_device, 1, seq_len, k]
        topk_expert_weights: Routing weights [batch_per_device, 1, seq_len, k]
        weights: Expert weights (sharded across devices)
        config: Expert configuration
        expert_mapping_tensors: Device-to-expert mapping [1, 1, num_experts, num_devices]
        remap_topk_mask: Mask for expert remapping [1, dispatch_rows, 1, num_experts]
        dispatch_config: Configuration for all_to_all_dispatch
        combine_config: Configuration for all_to_all_combine
        program_config: Matmul program configuration
        mesh_device: TTNN mesh device

    Returns:
        Output tensor [batch_size_per_device, 1, seq_len, hidden_size]
    """
    # ==========================================================================
    # STEP 0: RESHAPE TO PUT TOKENS ON DIM -2 (seq_len dimension)
    # ==========================================================================
    # This optimization reduces reshapes by keeping tokens on seq_len dim throughout.
    # Input typically comes as [B, 1, S, H] where B*S = total tokens per device.
    # We reshape to [1, 1, tokens_per_device, H] so tokens are on dim -2.

    # Get total tokens per device
    input_shape = hidden_states.shape
    tokens_per_device = input_shape[0] * input_shape[2]  # B * S

    # Reshape hidden states: put all tokens on dim -2
    hidden_states = ttnn.reshape(hidden_states, (1, 1, tokens_per_device, config.hidden_size))

    # typecast creates new tensors
    topk_expert_indices = ttnn.typecast(topk_expert_indices, dtype=ttnn.uint32)
    topk_expert_indices = ttnn.reshape(topk_expert_indices, (-1, 1, 1, config.num_experts_per_tok))
    topk_expert_indices = ttnn.typecast(topk_expert_indices, dtype=ttnn.uint16)

    topk_expert_weights = ttnn.reshape(topk_expert_weights, (-1, 1, 1, config.num_experts_per_tok))

    num_dispatch_devices = (
        mesh_device.shape[dispatch_config.cluster_axis]
        if dispatch_config.cluster_axis is not None
        else prod(mesh_device.shape)
    )
    total_tokens = tokens_per_device * num_dispatch_devices  # Global tokens across dispatch axis

    # ==========================================================================
    # STEP 1: PREPARE INPUTS FOR ALL_TO_ALL_DISPATCH
    # ==========================================================================
    # all_to_all_dispatch requires ROW_MAJOR layout with shape [B, 1, S, H]
    # With tokens on dim -2: [1, 1, tokens_per_device, H]
    # to_layout creates new tensors - safe to deallocate originals
    hidden_rm = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(hidden_states)
    # Shape is already [1, 1, tokens_per_device, H], just ensure it's correct
    hidden_rm = ttnn.reshape(hidden_rm, shape=(1, 1, tokens_per_device, config.hidden_size))

    # Expert indices: [1, 1, tokens_per_device, K]
    topk_indices_rm = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(topk_expert_indices)
    topk_indices_rm = ttnn.reshape(topk_indices_rm, shape=(1, 1, tokens_per_device, config.num_experts_per_tok))

    # ==========================================================================
    # STEP 2: ALL_TO_ALL_DISPATCH - Route tokens to expert devices
    # ==========================================================================
    # Dispatch sends each token to the device(s) that own its assigned expert(s)
    #
    # Inputs (tokens on dim -2):
    #   - hidden_rm: [1, 1, tokens_per_device, H] - token embeddings
    #   - topk_indices_rm: [1, 1, tokens_per_device, K] - which experts each token routes to
    #   - expert_mapping_tensors: [1, 1, E, D] - one-hot mapping of expert -> device
    #
    # With output_concat_dim=2, outputs have seq_len scaled:
    #   - dispatch_output: [D, 1, total_tokens, H] - tokens scattered to expert devices
    #   - dispatch_metadata: [D, 1, total_tokens, K] - expert indices (for combine routing)
    dispatch_output, dispatch_metadata = ttnn.all_to_all_dispatch(
        hidden_rm,
        topk_indices_rm,
        expert_mapping_tensors,
        **dispatch_config.as_dict(),
    )
    ttnn.deallocate(hidden_rm)
    ttnn.deallocate(topk_indices_rm)

    # ==========================================================================
    # STEP 3: MOE_EXPERT_TOKEN_REMAP - Create sparsity pattern
    # ==========================================================================
    # Converts global expert indices to local (per-device) indices and creates
    # a sparsity mask for efficient sparse matmul.
    #
    # The remap_topk_mask is broadcast across the token dimension (now on dim -2)
    # repeat creates a new tensor - safe to deallocate, but remap_topk_mask is reused externally
    # remap_topk_mask: [1, dispatch_rows, 1, num_experts]
    # -> repeat to [1, dispatch_rows, tokens_per_device, num_experts]
    # -> reshape to [1, 1, total_tokens, num_experts] to match dispatch_metadata batch/seq dims
    remap_mask = ttnn.repeat(remap_topk_mask, ttnn.Shape((1, 1, tokens_per_device, 1)))
    remap_mask = ttnn.reshape(remap_mask, (1, 1, total_tokens, config.num_experts))
    # moe_expert_token_remap returns:
    #   - mapping: [D, tokens, 1, experts_per_device] - local expert activation weights
    #   - sparsity: [D, 1, tokens/reduction_size, experts_per_device] - which blocks are active
    #
    # The sparsity tensor tells sparse_matmul which expert blocks have tokens,
    # avoiding computation on empty slots.
    _, sparsity = ttnn.moe_expert_token_remap(
        remap_mask,
        expert_mapping_tensors,
        dispatch_metadata,
        reduction_size=config.sparsity_block_size,
    )
    ttnn.deallocate(remap_mask)

    # ==========================================================================
    # STEP 4: PREPARE DISPATCH OUTPUT FOR EXPERT COMPUTATION
    # ==========================================================================
    # Reshape dispatch output for sparse matmul:
    # From: [D, 1, total_tokens, H] (ROW_MAJOR from dispatch with tokens on dim -2)
    # To: [1, total_tokens/block_size, block_size, H] (TILE for matmul)
    #
    # The sparse matmul operates on blocks of tokens, with sparsity indicating
    # which (token_block, expert) pairs need computation.
    # Note: reshape returns view, but to_layout creates new tensor
    post_dispatch = ttnn.reshape(dispatch_output, shape=(1, 1, total_tokens, config.hidden_size))
    post_dispatch_rm = post_dispatch
    post_dispatch = ttnn.to_layout(post_dispatch, ttnn.TILE_LAYOUT)
    ttnn.deallocate(post_dispatch_rm)  # This deallocates dispatch_output via the view

    # ==========================================================================
    # STEP 5: EXPERT COMPUTATION - Gate/Up/Down projections with sparse matmul
    # ==========================================================================
    # Expert MLP: output = down((up + 1) * (gate * sigmoid(gate * alpha)))
    #
    # sparse_matmul only computes (token_block, expert) pairs where sparsity=1,
    # significantly reducing computation for sparse expert activation patterns.
    expert_output = expert_mlp_forward(
        experts_input=post_dispatch,
        sparsity=sparsity,
        weights=weights,
        config=config,
        program_config=program_config,
        memory_config=dispatch_config.memory_config,
        total_tokens=total_tokens,
        mesh_device=mesh_device,
        save_intermediate=False,
    )

    # ==========================================================================
    # STEP 7: ALL_TO_ALL_COMBINE - Route expert outputs back to token positions
    # ==========================================================================
    # Combine routes each expert output back to the device that owns the original token.
    # Uses dispatch_metadata to know which token each output corresponds to.
    #
    # With output_shard_dim=2, output has tokens sharded on dim -2:
    # Output shape: [num_experts_per_tok, 1, tokens_per_device, H]
    # (each token gets outputs from k experts stacked in first dimension)
    combine_output = ttnn.all_to_all_combine(
        expert_output,
        dispatch_metadata,
        expert_mapping_tensors,
        **combine_config.as_dict(),
    )
    ttnn.deallocate(expert_output)
    ttnn.deallocate(dispatch_metadata)

    # ==========================================================================
    # STEP 8: APPLY ROUTING WEIGHTS AND REDUCE ACROSS EXPERTS
    # ==========================================================================
    # Combine output already has tokens on dim -2: [K, 1, tokens_per_device, H]
    # No reshape needed! Just convert to TILE layout.
    post_combine = ttnn.to_layout(combine_output, ttnn.TILE_LAYOUT)
    ttnn.deallocate(combine_output)

    # Prepare routing weights for broadcasting:
    # From: [B, 1, S, K] (original topk weights)
    # To: [K, 1, B*S, H] (matches post_combine for element-wise multiply)
    topk_weights_reshaped = prepare_expert_weights(
        topk_expert_weights=topk_expert_weights,
        num_experts_per_tok=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
    )

    # Weighted sum: sum_k(expert_output_k * routing_weight_k)
    memory_config = dispatch_config.memory_config
    weighted_output = ttnn.mul(post_combine, topk_weights_reshaped, memory_config=memory_config)
    ttnn.deallocate(post_combine)
    ttnn.deallocate(topk_weights_reshaped)

    # Sum across K experts (first dimension)
    output = ttnn.sum(weighted_output, dim=0, keepdim=True)
    ttnn.deallocate(weighted_output)

    # All-reduce across columns (cluster_axis=1) to aggregate expert outputs
    # Why this is needed:
    # 1. Experts are sharded across ALL devices in 2D (rows x cols)
    # 2. all_to_all_dispatch/combine on axis=0 handles row-wise redistribution
    # 3. But tokens may route to experts in different COLUMNS
    # 4. After combine, each device has partial results from experts in its column
    # 5. We need to sum these partials across columns to get complete expert outputs
    output_all_reduced = ttnn.all_reduce(
        output,
        num_links=1,
        topology=ttnn.Topology.Ring,
        cluster_axis=1,
        memory_config=memory_config,
    )
    ttnn.deallocate(output)

    # Final shape: [1, 1, tokens_per_device, H] (tokens on dim -2)
    return output_all_reduced
