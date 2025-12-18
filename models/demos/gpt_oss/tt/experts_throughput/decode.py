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

import torch

import ttnn

from .config import AllToAllCombineConfig, AllToAllDispatchConfig, ThroughputExpertConfig, ThroughputProgramConfig
from .weights import ThroughputExpertWeights


def save_intermediate_sparse_matmul(
    tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice, config: ThroughputExpertConfig, num_sparse_blocks: int, name: str
) -> None:
    # Save tensor for debugging - need to carefully combine sparse blocks
    # Current shape: [1, num_sparse_blocks, 1, num_experts_per_device, block_size, intermediate]
    # Tokens are split across dim 1 (num_sparse_blocks) and dim 4 (block_size)
    # Experts are in dim 3 (num_experts_per_device per device)

    # Step 1: Squeeze the extra dim 2
    # [1, num_sparse_blocks, 1, num_experts_per_device, block_size, intermediate]
    # -> [1, num_sparse_blocks, num_experts_per_device, block_size, intermediate]
    squeezed = ttnn.squeeze(tensor, 2)

    # Step 2: Permute to group sparse_blocks and block_size together
    # [1, num_sparse_blocks, num_experts_per_device, block_size, intermediate]
    # -> [1, num_experts_per_device, num_sparse_blocks, block_size, intermediate]
    permuted = ttnn.permute(squeezed, (0, 2, 1, 3, 4))

    # Step 3: Reshape to combine sparse_blocks * block_size into tokens
    # [1, num_experts_per_device, num_sparse_blocks, block_size, intermediate]
    # -> [1, num_experts_per_device, num_tokens_per_device, intermediate]
    num_tokens_per_device = num_sparse_blocks * config.sparsity_block_size
    reshaped = ttnn.reshape(
        permuted, (1, config.num_experts_per_device, num_tokens_per_device, config.intermediate_size)
    )

    # Step 4: Get tensors from all devices and concatenate manually
    # Each device has: [1, num_experts_per_device, num_tokens_per_device, intermediate]
    torch_w1_out_device0 = ttnn.to_torch(
        reshaped,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )
    # We only care about the first device so we can slice with [:1, :, :num_tokens_per_device, :]
    torch_w1_out_device0 = torch_w1_out_device0[:1, :, :num_tokens_per_device, :]
    # Now lets save the experts individually so we can easily compare with torch
    for expert_idx in range(config.num_experts_per_device):
        torch.save(
            torch_w1_out_device0[:, expert_idx : expert_idx + 1, :, :],
            f"gpt_oss_debug/{name}_tt_expert_{expert_idx}.pt",
        )
    ttnn.deallocate(squeezed)
    ttnn.deallocate(permuted)
    ttnn.deallocate(reshaped)


def _apply_silu_mul(w1_out: ttnn.Tensor, w3_out: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    """Apply SiLU activation and multiply: silu(w1_out) * w3_out.

    This implements the standard SwiGLU activation: silu(gate) * up

    Args:
        w1_out: Gate projection output [batch, experts, tokens, intermediate]
        w3_out: Up projection output [batch, experts, tokens, intermediate]
        memory_config: Output memory configuration

    Returns:
        Activated tensor with same shape as inputs
    """
    # Apply SiLU to gate output
    w1_activated = ttnn.silu(w1_out)
    ttnn.deallocate(w1_out)

    # Element-wise multiply with up projection (SwiGLU pattern)
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
        gate: Gate projection output [batch, experts, tokens, intermediate]
        up: Up projection output [batch, experts, tokens, intermediate]
        alpha: Scaling factor for sigmoid
        limit: Clamping limit for swiglu
        memory_config: Output memory configuration

    Returns:
        Activated tensor with same shape as inputs
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
    hidden_states = ttnn.reshape(hidden_states, (-1, 1, 1, config.hidden_size))
    topk_expert_indices = ttnn.typecast(topk_expert_indices, dtype=ttnn.uint32)
    topk_expert_indices = ttnn.reshape(topk_expert_indices, (-1, 1, 1, config.num_experts_per_tok))
    topk_expert_indices = ttnn.typecast(topk_expert_indices, dtype=ttnn.uint16)
    topk_expert_weights = ttnn.reshape(topk_expert_weights, (-1, 1, 1, config.num_experts_per_tok))

    seq_len = 1  # Decode mode always has seq_len=1
    batch_size_per_device = hidden_states.shape[0]
    num_dispatch_devices = (
        mesh_device.shape[dispatch_config.cluster_axis]
        if dispatch_config.cluster_axis is not None
        else prod(mesh_device.shape)
    )
    batch_size = batch_size_per_device * num_dispatch_devices  # Global batch across dispatch axis

    # ==========================================================================
    # STEP 1: PREPARE INPUTS FOR ALL_TO_ALL_DISPATCH
    # ==========================================================================
    # all_to_all_dispatch requires ROW_MAJOR layout with shape [B, 1, S, H]
    # Convert from TILE layout used by transformer layers
    hidden_rm = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
    hidden_rm = ttnn.reshape(hidden_rm, shape=(batch_size_per_device, 1, seq_len, config.hidden_size))

    # Expert indices need to be in ROW_MAJOR with shape [B, 1, S, K]
    # where K = num_experts_per_tok (top-k experts selected per token)
    topk_indices_rm = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
    topk_indices_rm = ttnn.reshape(
        topk_indices_rm, shape=(batch_size_per_device, 1, seq_len, config.num_experts_per_tok)
    )

    # ==========================================================================
    # STEP 2: ALL_TO_ALL_DISPATCH - Route tokens to expert devices
    # ==========================================================================
    # Dispatch sends each token to the device(s) that own its assigned expert(s)
    #
    # Inputs:
    #   - hidden_rm: [B_per_device, 1, S, H] - token embeddings
    #   - topk_indices_rm: [B_per_device, 1, S, K] - which experts each token routes to
    #   - expert_mapping_tensors: [1, 1, E, D] - one-hot mapping of expert -> device
    #
    # Outputs:
    #   - dispatch_output: [D, B_global, S, H] - tokens scattered to expert devices
    #   - dispatch_metadata: [D, B_global, S, K] - expert indices (for combine routing)
    dispatch_output, dispatch_metadata = ttnn.all_to_all_dispatch(
        hidden_rm,
        topk_indices_rm,
        expert_mapping_tensors,
        **dispatch_config.as_dict(),
    )
    # ttnn.deallocate(hidden_rm)
    ttnn.deallocate(topk_indices_rm)

    # ==========================================================================
    # STEP 3: MOE_EXPERT_TOKEN_REMAP - Create sparsity pattern
    # ==========================================================================
    # Converts global expert indices to local (per-device) indices and creates
    # a sparsity mask for efficient sparse matmul.
    #
    # The remap_topk_mask is broadcast across batch dimension
    remap_mask = ttnn.repeat(remap_topk_mask, ttnn.Shape((1, batch_size_per_device, 1, 1)))

    # moe_expert_token_remap returns:
    #   - mapping: [D, B, S, experts_per_device] - local expert activation weights
    #   - sparsity: [D, 1, B*S/reduction_size, experts_per_device] - which blocks are active
    #
    # The sparsity tensor tells sparse_matmul which expert blocks have tokens,
    # avoiding computation on empty slots.
    _, sparsity = ttnn.moe_expert_token_remap(
        remap_mask,
        expert_mapping_tensors,
        dispatch_metadata,
        reduction_size=config.sparsity_block_size,
    )
    # ttnn.deallocate(remap_mask)

    # ==========================================================================
    # STEP 4: PREPARE DISPATCH OUTPUT FOR EXPERT COMPUTATION
    # ==========================================================================
    # Reshape dispatch output for sparse matmul:
    # From: [D, B, S, H] (ROW_MAJOR from dispatch)
    # To: [1, B*S/block_size, block_size, H] (TILE for matmul)
    #
    # The sparse matmul operates on blocks of tokens, with sparsity indicating
    # which (token_block, expert) pairs need computation.
    post_dispatch = ttnn.reshape(dispatch_output, shape=(1, 1, batch_size * seq_len, config.hidden_size))
    post_dispatch = ttnn.to_layout(post_dispatch, ttnn.TILE_LAYOUT)
    # ttnn.deallocate(dispatch_output)

    # Reshape to sparse block format for matmul
    num_tokens = batch_size * seq_len
    num_sparse_blocks = num_tokens // config.sparsity_block_size
    expert_input = ttnn.reshape(
        post_dispatch,
        shape=(1, num_sparse_blocks, config.sparsity_block_size, config.hidden_size),
    )
    # ttnn.deallocate(post_dispatch)

    memory_config = dispatch_config.memory_config

    # ==========================================================================
    # STEP 5: EXPERT COMPUTATION - Gate/Up/Down projections with sparse matmul
    # ==========================================================================
    # Expert MLP: output = down((up + 1) * (gate * sigmoid(gate * alpha)))
    #
    # sparse_matmul only computes (token_block, expert) pairs where sparsity=1,
    # significantly reducing computation for sparse expert activation patterns.

    # Gate projection (w1): [B*S/block, block, H] x [experts, H, I] -> [B*S/block, experts, block, I]
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
    save_intermediate = False
    if save_intermediate:
        for expert_idx in range(4):
            torch.save(
                ttnn.to_torch(weights.w1, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[
                    :, expert_idx : expert_idx + 1, :, :
                ],
                f"gpt_oss_debug/gate_weights_tt_expert_{expert_idx}.pt",
            )
        save_intermediate_sparse_matmul(w1_out, mesh_device, config, num_sparse_blocks, "gate")

    # Up projection (w3): same shape as gate
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
    ttnn.deallocate(expert_input)

    # Add up bias
    # w3_out shape: [1, num_sparse_blocks, 1, num_experts_per_device, block_size, intermediate]
    # Bias shape: [1, 1, 1, num_experts_per_device, 1, intermediate] - broadcasts correctly
    w3_out = ttnn.add(w3_out, weights.w3_bias)
    if save_intermediate:
        for expert_idx in range(4):
            torch.save(
                ttnn.to_torch(weights.w3, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[
                    :, expert_idx : expert_idx + 1, :, :
                ],
                f"gpt_oss_debug/up_weights_tt_expert_{expert_idx}.pt",
            )
        save_intermediate_sparse_matmul(w3_out, mesh_device, config, num_sparse_blocks, "up")
    # SwiGLU activation: (up + 1) * (gate * sigmoid(gate * alpha))
    activated = _apply_swiglu(w1_out, w3_out, config.alpha, config.swiglu_limit, memory_config)

    # For testing standard SiLU activation instead of SwiGLU, uncomment below and comment above:
    # activated = _apply_silu_mul(w1_out, w3_out, memory_config)

    # Squeeze batch dimensions for down projection
    # From: [1, B*S/block, experts, block, I]
    # To: [B*S/block, experts, block, I]
    activated = ttnn.squeeze(activated, 0)
    activated = ttnn.squeeze(activated, 1)

    # Down projection (w2): [B*S/block, experts, block, I] x [experts, I, H] -> [B*S/block, experts, block, H]
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
    ttnn.deallocate(sparsity)

    # Add down projection bias
    # expert_output shape: [num_sparse_blocks, num_experts_per_device, block_size, hidden]
    # Bias shape: [1, num_experts_per_device, 1, hidden] - broadcasts correctly after squeeze
    # expert_output = ttnn.add(expert_output, weights.w2_bias)

    # ==========================================================================
    # STEP 6: PREPARE EXPERT OUTPUT FOR ALL_TO_ALL_COMBINE
    # ==========================================================================
    # Reshape from sparse matmul output to format expected by combine:
    # From: [B*S/block, experts, block, H]
    # To: [experts_per_device, B_global, S, H] (ROW_MAJOR)
    #
    # Permute to get experts_per_device as first dimension (what combine expects)
    expert_output = ttnn.permute(expert_output, (1, 0, 2, 3))
    expert_output = ttnn.reshape(
        expert_output,
        shape=(config.num_experts_per_device, batch_size, seq_len, config.hidden_size),
    )
    expert_output = ttnn.to_layout(expert_output, ttnn.ROW_MAJOR_LAYOUT)

    # ==========================================================================
    # STEP 7: ALL_TO_ALL_COMBINE - Route expert outputs back to token positions
    # ==========================================================================
    # Combine routes each expert output back to the device that owns the original token.
    # Uses dispatch_metadata to know which token each output corresponds to.
    #
    # Output shape: [num_experts_per_tok, B_per_device, S, H]
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
    # Reshape combine output for weighted sum:
    # Shape: [K, 1, B_per_device * S, H] where K = num_experts_per_tok
    post_combine = ttnn.reshape(
        combine_output,
        shape=(config.num_experts_per_tok, 1, batch_size_per_device * seq_len, config.hidden_size),
    )
    post_combine = ttnn.to_layout(post_combine, ttnn.TILE_LAYOUT)
    ttnn.deallocate(combine_output)

    # Prepare routing weights for broadcasting:
    # From: [B, 1, S, K] (original topk weights)
    # To: [K, 1, B*S, H] (matches post_combine for element-wise multiply)
    #
    # Steps:
    # 1. Repeat along hidden_size dimension
    # 2. Permute to [K, 1, B*S, H]
    topk_weights_rm = ttnn.to_layout(topk_expert_weights, ttnn.ROW_MAJOR_LAYOUT)
    topk_weights_rm = ttnn.repeat(topk_weights_rm, ttnn.Shape((1, 1, config.hidden_size, 1)))
    topk_weights_rm = ttnn.permute(topk_weights_rm, (3, 1, 0, 2))
    topk_weights_reshaped = ttnn.to_layout(topk_weights_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(topk_weights_rm)

    # Weighted sum: sum_k(expert_output_k * routing_weight_k)
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
    output = ttnn.all_reduce(
        output,
        num_links=1,
        topology=ttnn.Topology.Linear,
        cluster_axis=1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Final shape: [1, 1, B_per_device * S, H]
    return output
