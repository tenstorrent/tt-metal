# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Decode forward pass for experts (seq_len=1)."""

import ttnn
from models.demos.gpt_oss.config import Mode

from .config import ExpertConfig, ProgramConfig
from .operations import apply_expert_parallel_allreduce, apply_swiglu, apply_tensor_parallel_allreduce
from .weights import ExpertWeights


def decode_forward(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config: ExpertConfig,
    mesh_config,
    mesh_device,
    ccl_manager,
    program_config: ProgramConfig,
):
    """
    Decode forward pass - optimized for single token (seq_len=1).

    Args:
        hidden_states: Input tensor [batch, 1, hidden_size]
        routing_weights: Router output [batch, num_experts]
        weights: Expert weights
        config: Expert configuration
        mesh_config: Mesh parallelization config
        mesh_device: TTNN mesh device
        ccl_manager: Communication manager
        program_config: Model-specific program configs

    Returns:
        Expert output [1, batch, 1, hidden_size]
    """
    activation_dtype = ttnn.bfloat8_b
    batch_dim = 1
    seq_dim = 2
    batch_size = hidden_states.shape[batch_dim]
    seq_len = hidden_states.shape[seq_dim]

    # ✅ Use exceptions instead of assertions
    if seq_len != 1:
        raise ValueError(f"Decode mode requires seq_len=1, got {seq_len}")
    if batch_size != 1:
        raise NotImplementedError(f"Currently only batch_size=1 supported, got {batch_size}")

    # Get parallelization config
    mode_config = mesh_config.get_config(Mode.DECODE)
    ep, tp = mode_config.ep, mode_config.tp
    # Prepare inputs for sparse matmul
    # hidden_states_4D = ttnn.unsqueeze_to_4D(hidden_states)
    sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)

    # EP-specific routing remap for sparsity
    if ep > 1:
        # DEBUG: Convert to ROW_MAJOR before reshape to diagnose potential reshape bugs
        sparsity_was_tiled = sparsity.layout == ttnn.TILE_LAYOUT
        if sparsity_was_tiled:
            sparsity_temp = ttnn.to_layout(sparsity, ttnn.ROW_MAJOR_LAYOUT)
        else:
            sparsity_temp = sparsity
        sparsity_reshaped = ttnn.reshape(sparsity_temp, (1, sparsity_temp.shape[-1]))
        if sparsity_was_tiled:
            sparsity_reshaped = ttnn.to_layout(sparsity_reshaped, ttnn.TILE_LAYOUT)
        sparsity = ttnn.moe_routing_remap(sparsity_reshaped, 4, 4, 0)
        routing_weights = ttnn.tilize_with_zero_padding(sparsity, use_multicore=True)

    num_experts_per_tok = config.num_experts_per_tok // ep
    output_tile = ttnn.Tile([32, 32])

    # Gate projection
    gate = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_proj,
        sparsity=sparsity,
        nnz=num_experts_per_tok,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(hidden_states.shape[2], weights.gate_proj.shape[3]),
        dtype=activation_dtype,
    )
    # Note: reshape/transpose operations return views - do not deallocate originals
    # DEBUG: Convert to ROW_MAJOR before reshape to diagnose potential reshape bugs
    gate_was_tiled = gate.layout == ttnn.TILE_LAYOUT
    if gate_was_tiled:
        gate = ttnn.to_layout(gate, ttnn.ROW_MAJOR_LAYOUT)
    gate = ttnn.reshape(gate, (batch_size, config.num_experts, 1, weights.intermediate_size_per_device))
    if gate_was_tiled:
        gate = ttnn.to_layout(gate, ttnn.TILE_LAYOUT)
    gate = ttnn.transpose(gate, 1, 2)
    # DEBUG: Convert to ROW_MAJOR before reshape to diagnose potential reshape bugs
    gate_was_tiled2 = gate.layout == ttnn.TILE_LAYOUT
    if gate_was_tiled2:
        gate = ttnn.to_layout(gate, ttnn.ROW_MAJOR_LAYOUT)
    gate = ttnn.reshape(gate, (batch_size, config.num_experts, weights.intermediate_size_per_device))
    if gate_was_tiled2:
        gate = ttnn.to_layout(gate, ttnn.TILE_LAYOUT)
    gate = ttnn.add(gate, weights.gate_proj_bias, output_tensor=gate)

    # Up projection
    up = ttnn.sparse_matmul(
        hidden_states,
        weights.up_proj,
        sparsity=sparsity,
        nnz=num_experts_per_tok,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(hidden_states.shape[2], weights.up_proj.shape[3]),
        dtype=activation_dtype,
    )
    hidden_states.deallocate(True)
    # Note: reshape/transpose operations return views - do not deallocate originals
    # DEBUG: Convert to ROW_MAJOR before reshape to diagnose potential reshape bugs
    up_was_tiled = up.layout == ttnn.TILE_LAYOUT
    if up_was_tiled:
        up = ttnn.to_layout(up, ttnn.ROW_MAJOR_LAYOUT)
    up = ttnn.reshape(up, (batch_size, config.num_experts, 1, weights.intermediate_size_per_device))
    if up_was_tiled:
        up = ttnn.to_layout(up, ttnn.TILE_LAYOUT)
    up = ttnn.transpose(up, 1, 2)
    # DEBUG: Convert to ROW_MAJOR before reshape to diagnose potential reshape bugs
    up_was_tiled2 = up.layout == ttnn.TILE_LAYOUT
    if up_was_tiled2:
        up = ttnn.to_layout(up, ttnn.ROW_MAJOR_LAYOUT)
    up = ttnn.reshape(up, (batch_size, config.num_experts, weights.intermediate_size_per_device))
    if up_was_tiled2:
        up = ttnn.to_layout(up, ttnn.TILE_LAYOUT)
    up = ttnn.add(up, weights.up_proj_bias, output_tensor=up)

    # Apply SwiGLU activation (consumes gate and up internally)
    down_input = apply_swiglu(gate, up, config)
    # Note: transpose/reshape operations return views - do not deallocate originals
    down_input = ttnn.transpose(down_input, 1, 0)
    # DEBUG: Convert to ROW_MAJOR before reshape to diagnose potential reshape bugs
    down_input_was_tiled = down_input.layout == ttnn.TILE_LAYOUT
    if down_input_was_tiled:
        down_input = ttnn.to_layout(down_input, ttnn.ROW_MAJOR_LAYOUT)
    down_input = ttnn.reshape(down_input, (1, config.num_experts, seq_len, weights.intermediate_size_per_device))
    if down_input_was_tiled:
        down_input = ttnn.to_layout(down_input, ttnn.TILE_LAYOUT)
    # Down projection
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        nnz=num_experts_per_tok,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        is_input_a_sparse=True,
        program_config=program_config.get_decode_down_config(down_input.shape[2], weights.down_proj.shape[-1]),
        dtype=activation_dtype,
    )

    down_input.deallocate(True)
    sparsity.deallocate(True)
    # Apply bias and routing weights
    # Note: permute/reshape operations return views - do not deallocate originals
    next_states = ttnn.permute(down, (0, 2, 1, 3))
    # DEBUG: Convert to ROW_MAJOR before reshape to diagnose potential reshape bugs
    next_states_was_tiled = next_states.layout == ttnn.TILE_LAYOUT
    if next_states_was_tiled:
        next_states = ttnn.to_layout(next_states, ttnn.ROW_MAJOR_LAYOUT)
    next_states = ttnn.reshape(next_states, (batch_size, config.num_experts, config.hidden_size))
    if next_states_was_tiled:
        next_states = ttnn.to_layout(next_states, ttnn.TILE_LAYOUT)
    next_states = ttnn.add(next_states, weights.down_proj_bias, output_tensor=next_states)
    routing_weights = ttnn.permute(routing_weights, (1, 0))
    # DEBUG: Convert to ROW_MAJOR before reshape to diagnose potential reshape bugs
    routing_weights_was_tiled = routing_weights.layout == ttnn.TILE_LAYOUT
    if routing_weights_was_tiled:
        routing_weights = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
    routing_weights = ttnn.reshape(routing_weights, (batch_size, config.num_experts, 1))
    if routing_weights_was_tiled:
        routing_weights = ttnn.to_layout(routing_weights, ttnn.TILE_LAYOUT)

    next_states = ttnn.mul(next_states, routing_weights, output_tensor=next_states)
    routing_weights.deallocate(True)

    # Reduce across experts
    next_states = ttnn.sum(next_states, dim=1)
    # Note: unsqueeze_to_4D typically returns a view, so we don't deallocate the sum result
    next_states = ttnn.unsqueeze_to_4D(next_states)

    # Expert parallel communication
    if ep > 1:
        next_states = apply_expert_parallel_allreduce(next_states, mesh_config, ccl_manager)

    # Note: unsqueeze_to_4D typically returns a view
    next_states = ttnn.unsqueeze_to_4D(next_states)

    # Tensor parallel communication
    if tp > 1:
        # Note: apply_tensor_parallel_allreduce already handles deallocating the input tensor
        next_states = apply_tensor_parallel_allreduce(
            next_states, mesh_config, mesh_device, ccl_manager, activation_dtype, seq_len, tp
        )

    # Final reshape
    # Note: reshape typically returns a view, so we don't deallocate the original
    # DEBUG: Convert to ROW_MAJOR before reshape to diagnose potential reshape bugs
    next_states_final_was_tiled = next_states.layout == ttnn.TILE_LAYOUT
    if next_states_final_was_tiled:
        next_states = ttnn.to_layout(next_states, ttnn.ROW_MAJOR_LAYOUT)
    next_states = ttnn.reshape(
        next_states,
        (1, batch_size, seq_len, config.hidden_size),
        (1, batch_size, max(32, seq_len), config.hidden_size),
    )
    if next_states_final_was_tiled:
        next_states = ttnn.to_layout(next_states, ttnn.TILE_LAYOUT)

    return next_states
