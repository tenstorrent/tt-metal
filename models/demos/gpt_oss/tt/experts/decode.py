# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Decode forward pass for experts (seq_len=1)."""

import ttnn
from models.demos.gpt_oss.config import Mode

from .config import ExpertConfig, ProgramConfig
from .operations import (
    apply_expert_parallel_allreduce,
    apply_routing_weights,
    apply_swiglu,
    apply_tensor_parallel_allreduce,
    reduce_experts,
)
from .weights import ExpertWeights


def decode_forward(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config: ExpertConfig,
    mesh_config,
    mesh_device,  # ✅ Added mesh_device parameter
    ccl_manager,
    program_config: ProgramConfig,
    activation_dtype,
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
        activation_dtype: Data type for activations

    Returns:
        Expert output [batch, 1, hidden_size]
    """
    batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]

    # ✅ Use exceptions instead of assertions
    if seq_len != 1:
        raise ValueError(f"Decode mode requires seq_len=1, got {seq_len}")
    if batch_size != 1:
        raise NotImplementedError(f"Currently only batch_size=1 supported, got {batch_size}")

    # Get parallelization config
    mode_config = mesh_config.get_config(Mode.DECODE)
    ep, tp = mode_config.ep, mode_config.tp
    print("EP", ep)

    # Prepare inputs for sparse matmul
    hidden_states_4D = ttnn.unsqueeze_to_4D(hidden_states)
    sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)

    # EP-specific routing remap for sparsity
    if ep > 1:
        # sparsity_reshaped = ttnn.reshape(sparsity, (1, sparsity.shape[-1]))
        # sparsity.deallocate(True)  # ✅ Deallocate old sparsity
        # sparsity = ttnn.moe_routing_remap(sparsity_reshaped, 4, 4, 0)
        # sparsity_reshaped.deallocate(True)  # ✅ Deallocate reshaped
        # # Note: sparsity is used for sparse matmuls, routing_weights used for output scaling
        sparsity = ttnn.moe_routing_remap(ttnn.reshape(sparsity, (1, sparsity.shape[-1])), 4, 4, 0)
        routing_weights = ttnn.tilize_with_zero_padding(sparsity, use_multicore=True)

    num_experts_per_tok = config.num_experts_per_tok // ep
    output_tile = ttnn.Tile([32, 32])

    # Gate projection
    gate = ttnn.sparse_matmul(
        hidden_states_4D,
        weights.gate_proj,
        sparsity=sparsity,
        nnz=num_experts_per_tok,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(hidden_states_4D.shape[2], weights.gate_proj.shape[3]),
        dtype=activation_dtype,
    )
    gate = ttnn.reshape(gate, (batch_size, config.num_experts, seq_len, weights.intermediate_size_per_device))
    gate = ttnn.add(gate, weights.gate_proj_bias, output_tensor=gate)

    # Up projection
    up = ttnn.sparse_matmul(
        hidden_states_4D,
        weights.up_proj,
        sparsity=sparsity,
        nnz=num_experts_per_tok,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(hidden_states_4D.shape[2], weights.up_proj.shape[3]),
        dtype=activation_dtype,
    )
    hidden_states_4D.deallocate(True)

    up = ttnn.reshape(up, (batch_size, config.num_experts, seq_len, weights.intermediate_size_per_device))
    up = ttnn.add(up, weights.up_proj_bias, output_tensor=up)

    # Apply SwiGLU activation
    down_input = apply_swiglu(gate, up, config)
    down_input = ttnn.reshape(down_input, (1, config.num_experts, seq_len, weights.intermediate_size_per_device))

    # Prepare routing weights for applying to expert outputs
    # Note: routing_weights_tilized is only set if ep > 1, but we don't use it here
    routing_weights_permuted = ttnn.permute(routing_weights, (1, 0))
    routing_weights.deallocate(True)  # ✅ Deallocate original routing_weights
    routing_weights = ttnn.reshape(routing_weights_permuted, (batch_size, config.num_experts, seq_len, 1))
    routing_weights_permuted.deallocate(True)  # ✅ Deallocate permuted

    # Down projection
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        nnz=num_experts_per_tok,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        is_input_a_sparse=True,
        program_config=program_config.get_decode_down_config(down_input.shape[2], weights.down_proj.shape[-1]),
        dtype=activation_dtype,
    )
    down_input.deallocate(True)

    # Apply bias and routing weights
    next_states = ttnn.reshape(down, (batch_size, config.num_experts, seq_len, config.hidden_size))
    next_states = ttnn.add(next_states, weights.down_proj_bias, output_tensor=next_states)
    next_states = apply_routing_weights(next_states, routing_weights)
    routing_weights.deallocate(True)  # ✅ Deallocate routing_weights after use
    sparsity.deallocate(True)  # ✅ Deallocate sparsity after all sparse matmuls

    # Reduce across experts
    next_states = reduce_experts(next_states)

    # Expert parallel communication
    if ep > 1:
        next_states = apply_expert_parallel_allreduce(next_states, mesh_config, ccl_manager)

    # Tensor parallel communication
    if tp > 1:
        next_states = apply_tensor_parallel_allreduce(
            next_states, mesh_config, mesh_device, ccl_manager, activation_dtype, seq_len, tp  # ✅ Pass mesh_device
        )

    # Final reshape
    next_states = ttnn.reshape(
        next_states,
        (batch_size, seq_len, config.hidden_size),
        (batch_size, max(32, seq_len), config.hidden_size),
    )

    return next_states
