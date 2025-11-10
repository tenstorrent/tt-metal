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
        Expert output [batch, 1, hidden_size]
    """
    activation_dtype = ttnn.bfloat8_b
    batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]

    # ✅ Use exceptions instead of assertions
    if seq_len != 1:
        raise ValueError(f"Decode mode requires seq_len=1, got {seq_len}")
    if batch_size != 1:
        raise NotImplementedError(f"Currently only batch_size=1 supported, got {batch_size}")

    # Get parallelization config
    mode_config = mesh_config.get_config(Mode.DECODE)
    ep, tp = mode_config.ep, mode_config.tp
    # Prepare inputs for sparse matmul
    hidden_states_4D = ttnn.unsqueeze_to_4D(hidden_states)
    sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)

    # EP-specific routing remap for sparsity
    if ep > 1:
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
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(hidden_states_4D.shape[2], weights.gate_proj.shape[3]),
        dtype=activation_dtype,
    )
    gate = ttnn.reshape(gate, (batch_size, config.num_experts, 1, weights.intermediate_size_per_device))
    gate = ttnn.transpose(gate, 1, 2)

    gate = ttnn.reshape(gate, (batch_size, config.num_experts, weights.intermediate_size_per_device))
    gate = ttnn.add(gate, weights.gate_proj_bias, output_tensor=gate)

    # Up projection
    up = ttnn.sparse_matmul(
        hidden_states_4D,
        weights.up_proj,
        sparsity=sparsity,
        nnz=num_experts_per_tok,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(hidden_states_4D.shape[2], weights.up_proj.shape[3]),
        dtype=activation_dtype,
    )
    hidden_states_4D.deallocate(True)
    up = ttnn.reshape(up, (batch_size, config.num_experts, 1, weights.intermediate_size_per_device))
    up = ttnn.transpose(up, 1, 2)
    up = ttnn.reshape(up, (batch_size, config.num_experts, weights.intermediate_size_per_device))
    up = ttnn.add(up, weights.up_proj_bias, output_tensor=up)

    # Apply SwiGLU activation
    down_input = apply_swiglu(gate, up, config)
    # height_sharded_mem_config = ttnn.create_sharded_memory_config(
    #             shape=(32, 384),  # Shape per shard (tile-aligned)
    #             core_grid=ttnn.CoreGrid(x=1, y=1),
    #             strategy=ttnn.ShardStrategy.HEIGHT,
    #             orientation=ttnn.ShardOrientation.ROW_MAJOR,
    #             use_height_and_width_as_shard_shape=True,
    #         )
    # print(down_input.shape)
    # down_input = ttnn.to_memory_config(down_input, height_sharded_mem_config)
    # height_sharded_mem_config_next = ttnn.create_sharded_memory_config(
    #         shape=(32*32, 384),  # Shape per shard (tile-aligned)
    #         core_grid=ttnn.CoreGrid(x=1, y=1),
    #         strategy=ttnn.ShardStrategy.HEIGHT,
    #         orientation=ttnn.ShardOrientation.ROW_MAJOR,
    #         use_height_and_width_as_shard_shape=True,
    #     )
    # print("down input", down_input.memory_config())
    down_input = ttnn.transpose(down_input, 1, 0)  # , memory_config=height_sharded_mem_config_next)
    down_input = ttnn.reshape(down_input, (1, config.num_experts, seq_len, weights.intermediate_size_per_device))
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
    next_states = ttnn.reshape(ttnn.permute(down, (0, 2, 1, 3)), (batch_size, config.num_experts, config.hidden_size))
    next_states = ttnn.add(next_states, weights.down_proj_bias, output_tensor=next_states)
    routing_weights = ttnn.permute(routing_weights, (1, 0))
    routing_weights = ttnn.reshape(routing_weights, (batch_size, config.num_experts, 1))

    next_states = ttnn.mul(next_states, routing_weights, output_tensor=next_states)
    routing_weights.deallocate(True)

    # Reduce across experts
    next_states = ttnn.unsqueeze_to_4D(ttnn.sum(next_states, dim=1))

    # Expert parallel communication
    if ep > 1:
        next_states = apply_expert_parallel_allreduce(next_states, mesh_config, ccl_manager)

    next_states = ttnn.unsqueeze_to_4D(next_states)
    # Tensor parallel communication
    if tp > 1:
        next_states = apply_tensor_parallel_allreduce(
            next_states, mesh_config, mesh_device, ccl_manager, activation_dtype, seq_len, tp
        )

    # Final reshape
    next_states = ttnn.reshape(
        next_states,
        (batch_size, seq_len, config.hidden_size),
        (batch_size, max(32, seq_len), config.hidden_size),
    )

    return next_states
