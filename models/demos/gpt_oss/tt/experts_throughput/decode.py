# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Decode forward pass for throughput-optimized MoE experts.

This module implements the decode path using all_to_all_dispatch and all_to_all_combine
to dynamically batch tokens across devices based on expert routing.

The MoE forward pass flow is:
1. all_to_all_dispatch: Route tokens to devices based on expert assignments
2. Expert computation: Run gate/up/down projections using regular batched matmul
3. all_to_all_combine: Route expert outputs back to original token positions
4. Apply routing weights and reduce across experts
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
    # Prepare routing weights for broadcasting:
    # topk_expert_weights is [1, 1, tokens_per_device, K] (tokens on dim -2)
    # We want [K, 1, tokens_per_device, 1] so it can broadcast across hidden_size.
    # to_layout creates new tensor - safe to deallocate original
    topk_weights_rm = ttnn.to_layout(topk_expert_weights, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(topk_expert_weights)
    # permute to [K, 1, tokens_per_device, 1]
    topk_weights_rm = ttnn.permute(topk_weights_rm, (3, 1, 2, 0))
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
        gate: Gate projection output
        up: Up projection output
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
    ttnn.add(up_clamped, 1.0, output_tensor=up_clamped)

    # Multiply: result = up * glu
    result = ttnn.mul(up_clamped, glu, memory_config=memory_config)
    ttnn.deallocate(up_clamped)
    ttnn.deallocate(glu)

    return result


def expert_mlp_forward(
    experts_input: ttnn.Tensor,
    weights: ThroughputExpertWeights,
    config: ThroughputExpertConfig,
    memory_config: ttnn.MemoryConfig,
    program_config: ThroughputProgramConfig,
    total_tokens: int,
) -> ttnn.Tensor:
    # All dispatched tokens are processed by all local experts. The combine
    # step will select the correct expert output for each token.

    # Build 1D multicast program configs sized for total_tokens (M dimension)
    down_matmul_config = program_config.get_down_config(n=config.hidden_size, m=total_tokens)

    # Choose between fused and unfused gate/up projection
    if weights.w1_w3_fused is not None:
        # ======================================================================
        # FUSED PATH: Single matmul for gate+up projections
        # ======================================================================
        assert weights.w1_w3_bias_fused is not None, "Fused bias must be present when using fused weights"

        # Get fused-specific matmul config (output size is 2*intermediate_size)
        fused_gate_up_matmul_config = program_config.get_fused_gate_up_config(
            n=weights.w1_w3_fused.shape[-1], m=total_tokens
        )

        # Fused projection: [1, E, total_tokens, H] x [1, E, H, 2*I] -> [1, E, total_tokens, 2*I]
        w1_w3_out = ttnn.matmul(experts_input, weights.w1_w3_fused, memory_config=memory_config)
        ttnn.deallocate(experts_input)

        # Add fused bias: [1, num_experts_per_device, 1, 2*I] broadcasts across total_tokens
        ttnn.add(w1_w3_out, weights.w1_w3_bias_fused, output_tensor=w1_w3_out)

        # Split into gate and up projections
        # w1_w3_out: [1, num_experts_per_device, total_tokens, 2*intermediate_size]
        # Split along last dimension: first half is gate, second half is up
        shape = w1_w3_out.shape

        # Extract gate projection (first half of last dimension)
        w1_out = ttnn.slice(
            w1_w3_out, [0, 0, 0, 0], [shape[0], shape[1], shape[2], config.intermediate_size], [1, 1, 1, 1]
        )
        w3_out = ttnn.slice(
            w1_w3_out,
            [0, 0, 0, config.intermediate_size],
            [shape[0], shape[1], shape[2], 2 * config.intermediate_size],
            [1, 1, 1, 1],
        )

        ttnn.deallocate(w1_w3_out)

    else:
        # ======================================================================
        # UNFUSED PATH: Separate matmuls for gate and up projections
        # ======================================================================
        assert weights.w1 is not None, "Unfused weights (w1) must be present when not using fused mode"
        assert weights.w3 is not None, "Unfused weights (w3) must be present when not using fused mode"
        assert weights.w1_bias is not None, "Unfused bias (w1_bias) must be present when not using fused mode"
        assert weights.w3_bias is not None, "Unfused bias (w3_bias) must be present when not using fused mode"

        # Get unfused-specific matmul config (output size is intermediate_size)
        gate_up_matmul_config = program_config.get_gate_up_config(n=config.intermediate_size, m=total_tokens)

        # Gate projection (w1)
        w1_out = ttnn.matmul(experts_input, weights.w1, memory_config=memory_config)
        # Bias: [1, num_experts_per_device, 1, I] broadcasts across total_tokens
        ttnn.add(w1_out, weights.w1_bias, output_tensor=w1_out)

        # Up projection (w3)
        w3_out = ttnn.matmul(experts_input, weights.w3, memory_config=memory_config)
        ttnn.deallocate(experts_input)
        # Bias: [1, num_experts_per_device, 1, I] broadcasts across total_tokens
        ttnn.add(w3_out, weights.w3_bias, output_tensor=w3_out)

    # SwiGLU activation: (up + 1) * (gate * sigmoid(gate * alpha))
    activated = _apply_swiglu(w1_out, w3_out, config.alpha, config.swiglu_limit, memory_config)

    # Down projection (w2): [1, E, total_tokens, I] x [1, E, I, H] -> [1, E, total_tokens, H]
    expert_output = ttnn.matmul(activated, weights.w2, memory_config=memory_config)
    ttnn.deallocate(activated)
    # Bias: [1, num_experts_per_device, 1, H] broadcasts across total_tokens
    ttnn.add(expert_output, weights.w2_bias, output_tensor=expert_output)

    # ==========================================================================
    # STEP 5: PREPARE EXPERT OUTPUT FOR ALL_TO_ALL_COMBINE
    # ==========================================================================
    # expert_output: [1, num_experts_per_device, total_tokens, H] in TILE
    # Combine expects: [num_experts_per_device, 1, total_tokens, H] in ROW_MAJOR
    expert_output = ttnn.reshape(
        expert_output,
        shape=(config.num_experts_per_device, 1, total_tokens, -1),
    )

    expert_output_tiled = expert_output
    expert_output = ttnn.to_layout(expert_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(expert_output_tiled)

    return expert_output


def apply_allreduce(tensor, mesh_config, ccl_manager, hidden_size: int):
    """
    Apply tensor parallel allreduce if needed.

    Args:
        tensor: Input tensor
        mesh_config: Mesh configuration
        ccl_manager: Communication manager
        batch_size: Batch size for final reshape
        seq_len: Sequence length for final reshape
        hidden_size: Hidden size for final reshape

    Returns:
        Tensor after allreduce (if TP > 1) or original tensor
    """
    tensor = mesh_config.allreduce(tensor, ccl_manager, pad_size=0, axis=1)
    # Remove padding added in weights.py for tile-aligned CCL operations.
    # Slice from padded_hidden back to hidden_size on the last dimension.
    # Works for both decode [1, 1, batch, padded_hidden] and prefill [1, batch, seq_len, padded_hidden].
    shape = tensor.shape
    tensor = ttnn.slice(
        tensor,
        starts=[0, 0, 0, 0],
        ends=[shape[0], shape[1], shape[2], hidden_size],
        steps=[1, 1, 1, 1],
    )
    return tensor


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
    mesh_config=None,
    ccl_manager=None,
) -> ttnn.Tensor:
    """Decode forward pass with all_to_all dispatch and combine.

    This implements the MoE forward pass for decode (seq_len=1 per batch element).
    Uses regular batched matmul instead of sparse matmul - all dispatched tokens
    are processed by all local experts, and all_to_all_combine routes the correct
    expert outputs back to their original token positions.

    Args:
        hidden_states: Input tensor [batch_size_per_device, 1, seq_len, hidden_size]
        topk_expert_indices: Expert indices per token [batch_per_device, 1, seq_len, k]
        topk_expert_weights: Routing weights [batch_per_device, 1, seq_len, k]
        weights: Expert weights (sharded across devices)
        config: Expert configuration
        expert_mapping_tensors: Device-to-expert mapping [1, 1, num_experts, num_devices]
        remap_topk_mask: Mask for expert remapping (unused, kept for interface compat)
        dispatch_config: Configuration for all_to_all_dispatch
        combine_config: Configuration for all_to_all_combine
        program_config: Matmul program configuration for expert projections
        mesh_device: TTNN mesh device

    Returns:
        Output tensor [batch_size_per_device, 1, seq_len, hidden_size]
    """
    # ==========================================================================
    # STEP 0: RESHAPE TO PUT TOKENS ON DIM -2 (seq_len dimension)
    # ==========================================================================
    input_shape = hidden_states.shape
    tokens_per_device = input_shape[0] * input_shape[2]  # B * S

    hidden_states = ttnn.reshape(hidden_states, (1, 1, tokens_per_device, config.hidden_size))

    # typecast creates new tensors - safe to deallocate originals
    topk_expert_indices_orig = topk_expert_indices
    topk_expert_indices = ttnn.typecast(topk_expert_indices, dtype=ttnn.uint32)
    ttnn.deallocate(topk_expert_indices_orig)

    topk_expert_indices = ttnn.reshape(topk_expert_indices, (1, 1, tokens_per_device, config.num_experts_per_tok))
    topk_expert_indices_u32 = topk_expert_indices
    topk_expert_indices = ttnn.typecast(topk_expert_indices, dtype=ttnn.uint16)
    ttnn.deallocate(topk_expert_indices_u32)

    topk_expert_weights = ttnn.reshape(topk_expert_weights, (1, 1, tokens_per_device, config.num_experts_per_tok))

    num_dispatch_devices = (
        mesh_device.shape[dispatch_config.cluster_axis]
        if dispatch_config.cluster_axis is not None
        else prod(mesh_device.shape)
    )
    total_tokens = tokens_per_device * num_dispatch_devices  # Global tokens across dispatch axis

    # ==========================================================================
    # STEP 1: PREPARE INPUTS FOR ALL_TO_ALL_DISPATCH
    # ==========================================================================
    hidden_rm = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(hidden_states)
    hidden_rm = ttnn.reshape(hidden_rm, shape=(1, 1, tokens_per_device, config.hidden_size))

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
    # STEP 3: PREPARE DISPATCH OUTPUT FOR EXPERT COMPUTATION
    # ==========================================================================
    # dispatch_output: [1, 1, total_tokens, H] in ROW_MAJOR
    # Convert to TILE layout and repeat across expert dimension for batched matmul.
    # ttnn.matmul requires batch dims to match exactly (no broadcasting).
    post_dispatch = ttnn.reshape(dispatch_output, shape=(1, 1, total_tokens, config.hidden_size))
    post_dispatch_rm = post_dispatch
    post_dispatch = ttnn.to_layout(post_dispatch, ttnn.TILE_LAYOUT)
    ttnn.deallocate(post_dispatch_rm)

    # Repeat input across expert dimension:
    # [1, 1, total_tokens, H] -> [1, num_experts_per_device, total_tokens, H]
    post_dispatch = ttnn.repeat(post_dispatch, ttnn.Shape((1, config.num_experts_per_device, 1, 1)))

    # ==========================================================================
    # STEP 4: EXPERT COMPUTATION - Gate/Up/Down projections with batched matmul
    # ==========================================================================
    # Batched matmul: input repeated across expert dim, weights have expert dim.
    #   Input:   [1, num_experts_per_device, total_tokens, H]
    #   Weights: [1, num_experts_per_device, H, I (or 2*I for fused)]
    #   Output:  [1, num_experts_per_device, total_tokens, I (or 2*I for fused)]

    expert_output = expert_mlp_forward(
        experts_input=post_dispatch,
        weights=weights,
        config=config,
        memory_config=dispatch_config.memory_config,
        program_config=program_config,
        total_tokens=total_tokens,
    )

    # ==========================================================================
    # STEP 6: ALL_TO_ALL_COMBINE - Route expert outputs back to token positions
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
    # STEP 7: APPLY ROUTING WEIGHTS AND REDUCE ACROSS EXPERTS
    # ==========================================================================
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
    if config.use_experimental_all_reduce:
        output_all_reduced = apply_allreduce(output, mesh_config, ccl_manager, config.hidden_size)
    else:
        output_all_reduced = ttnn.all_reduce(
            output,
            num_links=4,
            topology=ttnn.Topology.Ring,
            cluster_axis=1,
            memory_config=memory_config,
        )
    ttnn.deallocate(output)

    # Final shape: [1, 1, tokens_per_device, H] (tokens on dim -2)
    return output_all_reduced
