# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Prefill forward pass for experts (seq_len>1)."""

import ttnn
from models.demos.gpt_oss.config import Mode

from .config import ExpertConfig, ProgramConfig
from .operations import (
    apply_expert_parallel_allreduce,
    apply_routing_weights,
    apply_sequence_parallel_allgather,
    apply_swiglu,
    apply_tensor_parallel_allreduce,
    reduce_experts,
)
from .weights import ExpertWeights


def _reshard_for_sequence_parallel(hidden_states, routing_weights, mesh_device):
    """Reshard tensors for sequence parallel processing."""
    hidden_states_torch = ttnn.to_torch(ttnn.get_device_tensors(hidden_states)[0])
    routing_weights_torch = ttnn.to_torch(ttnn.get_device_tensors(routing_weights)[0])
    hidden_states.deallocate(True)
    routing_weights.deallocate(True)

    routing_weights = ttnn.from_torch(
        routing_weights_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(dims=(-2, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device),
    )
    hidden_states = ttnn.from_torch(
        hidden_states_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(dims=(-2, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device),
    )

    return hidden_states, routing_weights


def _process_prefill_chunk(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config: ExpertConfig,
    prefill_sparsity,
    program_config: ProgramConfig,
    ep,
    tp,
):
    """Process a single chunk of the sequence in prefill mode."""
    activation_dtype = ttnn.bfloat8_b
    TILE_SIZE = 32
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]

    # Reshape for prefill (group tokens into tiles)
    hidden_states_4D = ttnn.unsqueeze_to_4D(hidden_states)
    hidden_states_4D = ttnn.reshape(hidden_states_4D, (1, seq_len // TILE_SIZE, TILE_SIZE, config.hidden_size))
    group_size = seq_len // TILE_SIZE

    # Prepare sparsity
    # Note: prefill_sparsity is cached and reused, don't deallocate it
    sparsity_repeated = ttnn.repeat(prefill_sparsity, (1, 1, group_size, 1))
    routing_weights_4d = ttnn.unsqueeze_to_4D(routing_weights)
    sparsity_layout = sparsity_repeated  # ttnn.to_layout(routing_weights_4d, ttnn.ROW_MAJOR_LAYOUT)
    # routing_weights_4d.deallocate(True)  # ✅ Deallocate temporary 4D tensor

    num_experts_per_tok = (config.num_experts // ep) * group_size
    output_tile = ttnn.Tile([32, 32])
    # Gate projection
    gate = ttnn.sparse_matmul(
        hidden_states_4D,
        weights.gate_proj,
        sparsity=sparsity_layout,
        nnz=num_experts_per_tok,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_prefill_gate_up_config(hidden_states_4D.shape[2], weights.gate_proj.shape[3]),
        dtype=activation_dtype,
    )
    gate = ttnn.transpose(gate, 1, 3)
    gate = ttnn.reshape(gate, (batch_size, config.num_experts, seq_len, weights.intermediate_size_per_device))
    bias_transposed = ttnn.transpose(weights.gate_proj_bias, 1, 0)
    gate = ttnn.add(gate, bias_transposed, output_tensor=gate)

    # Up projection
    up = ttnn.sparse_matmul(
        hidden_states_4D,
        weights.up_proj,
        sparsity=sparsity_layout,
        nnz=num_experts_per_tok,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_prefill_gate_up_config(hidden_states_4D.shape[2], weights.up_proj.shape[3]),
        dtype=activation_dtype,
    )
    hidden_states_4D.deallocate(True)

    up = ttnn.transpose(up, 1, 3)
    up = ttnn.reshape(up, (batch_size, config.num_experts, seq_len, weights.intermediate_size_per_device))
    bias_transposed = ttnn.transpose(weights.up_proj_bias, 1, 0)
    up = ttnn.add(up, bias_transposed, output_tensor=up)

    # Apply SwiGLU
    down_input = apply_swiglu(gate, up, config)
    down_input = ttnn.reshape(down_input, (1, config.num_experts, seq_len, weights.intermediate_size_per_device))

    # Update routing weights and sparsity for down projection
    num_experts_per_tok = config.num_experts // ep
    prefill_sparsity_reshaped = ttnn.reshape(prefill_sparsity, (1, config.num_experts))
    routing_weights = ttnn.mul(
        routing_weights,
        prefill_sparsity_reshaped,
        output_tensor=routing_weights,
    )
    # prefill_sparsity_reshaped.deallocate(True)  # ✅ Deallocate reshaped sparsity
    # sparsity_layout.deallocate(True)  # ✅ Deallocate sparsity_layout after gate/up matmuls
    # sparsity_repeated.deallocate(True)  # ✅ Deallocate repeated sparsity (not used for down)

    routing_weights_permuted = ttnn.permute(routing_weights, (1, 0))
    routing_weights.deallocate(True)  # ✅ Deallocate before reshape
    routing_weights = ttnn.reshape(routing_weights_permuted, (batch_size, config.num_experts, seq_len, 1))
    # routing_weights_permuted.deallocate(True)  # ✅ Deallocate permuted

    # Process down projection in splits if needed
    split_size = program_config.down_split_size
    if seq_len > split_size:
        down_input_list = ttnn.split(down_input, split_size, dim=2)
        down_input.deallocate(True)
        routing_weights_list = ttnn.split(routing_weights, split_size, dim=2)
        routing_weights.deallocate(True)
    else:
        down_input_list = [down_input]
        routing_weights_list = [routing_weights]

    # Process each split
    next_states_reduced_list = []
    for i, down_input_split in enumerate(down_input_list):
        down = ttnn.sparse_matmul(
            down_input_split,
            weights.down_proj,
            sparsity=prefill_sparsity,
            nnz=num_experts_per_tok,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            program_config=program_config.get_prefill_down_config(
                down_input_split.shape[2], weights.down_proj.shape[-1]
            ),
            dtype=activation_dtype,
        )
        down_input_split.deallocate(True)

        # Apply bias and routing weights
        split_seq_len = seq_len if seq_len < split_size else split_size
        next_states = ttnn.reshape(down, (batch_size, config.num_experts, split_seq_len, config.hidden_size))
        bias_transposed = ttnn.transpose(weights.down_proj_bias, 1, 0)
        next_states = ttnn.add(next_states, bias_transposed, output_tensor=next_states)
        next_states = apply_routing_weights(next_states, routing_weights_list[i])

        # Reduce across experts
        next_states_reduced = reduce_experts(next_states)
        next_states_reduced_list.append(next_states_reduced)
        routing_weights_list[i].deallocate(True)

    # Concatenate splits (deallocates list elements internally in some TTNN versions)
    next_states_concat = ttnn.concat(next_states_reduced_list, dim=2)

    # # ✅ Deallocate all split results after concat
    # for tensor in next_states_reduced_list:
    #     tensor.deallocate(True)

    return next_states_concat


def prefill_forward(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config: ExpertConfig,
    mesh_config,
    mesh_device,
    ccl_manager,
    program_config: ProgramConfig,
    prefill_sparsity,
):
    """
    Prefill forward pass - optimized for sequence processing (seq_len>1).

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        routing_weights: Router output [seq_len, num_experts]
        weights: Expert weights
        config: Expert configuration
        mesh_config: Mesh parallelization config
        mesh_device: TTNN mesh device
        ccl_manager: Communication manager
        program_config: Model-specific program configs
        prefill_sparsity: Cached prefill sparsity mask

    Returns:
        Expert output [batch, seq_len, hidden_size]
    """
    activation_dtype = ttnn.bfloat8_b
    batch_size = hidden_states.shape[0]
    seq_len_global = hidden_states.shape[1]

    # ✅ Use exceptions instead of assertions and validate seq_len
    if batch_size != 1:
        raise NotImplementedError(f"Currently only batch_size=1 supported, got {batch_size}")

    if seq_len_global <= 1:
        raise ValueError(
            f"Prefill mode requires seq_len>1, got {seq_len_global}. " f"Use decode mode for single tokens."
        )

    # ✅ Validate seq_len is divisible by TILE_SIZE (32)
    TILE_SIZE = 32
    if seq_len_global % TILE_SIZE != 0:
        raise ValueError(
            f"Prefill seq_len must be divisible by {TILE_SIZE} (TILE_SIZE), "
            f"got {seq_len_global}. Please pad your sequence."
        )

    # Get parallelization config
    mode_config = mesh_config.get_config(Mode.PREFILL)
    ep, sp, tp = mode_config.ep, mode_config.sp, mode_config.tp

    # Reshard for sequence parallelism if needed
    if sp > 1:
        hidden_states, routing_weights = _reshard_for_sequence_parallel(
            hidden_states, routing_weights, mesh_device  # ✅ Use explicit mesh_device
        )

    # Chunk processing for very long sequences
    chunk_size = program_config.sequence_chunk_size
    if hidden_states.shape[1] > chunk_size:
        hidden_states_chunks = ttnn.split(hidden_states, chunk_size, dim=1)
        hidden_states.deallocate(True)
        routing_weights_chunks = ttnn.split(routing_weights, chunk_size, dim=0)
        routing_weights.deallocate(True)
    else:
        hidden_states_chunks = [hidden_states]
        routing_weights_chunks = [routing_weights]

    # Process each chunk
    next_states_list = []
    for hidden_chunk, routing_chunk in zip(hidden_states_chunks, routing_weights_chunks):
        next_states = _process_prefill_chunk(
            hidden_chunk,
            routing_chunk,
            weights,
            config,
            prefill_sparsity,
            program_config,
            ep,
            tp,
        )
        next_states_list.append(next_states)
        hidden_chunk.deallocate(True)  # ✅ Deallocate chunk after processing
        routing_chunk.deallocate(True)  # ✅ Deallocate chunk after processing

    # Concatenate all chunks
    next_states = ttnn.concat(next_states_list, dim=2)

    # ✅ Deallocate chunk results after concat
    # for tensor in next_states_list:
    #     tensor.deallocate(True)

    # Expert parallel communication
    if ep > 1:
        next_states = apply_expert_parallel_allreduce(next_states, mesh_config, ccl_manager)

    # Tensor parallel communication
    if tp > 1:
        next_states = apply_tensor_parallel_allreduce(
            next_states,
            mesh_config,
            mesh_device,
            ccl_manager,
            activation_dtype,
            seq_len_global,
            tp,  # ✅ Pass mesh_device
        )

    # Sequence parallel all-gather
    if sp > 1:
        next_states = apply_sequence_parallel_allgather(next_states, mesh_config, ccl_manager)

    # Final reshape
    next_states = ttnn.reshape(
        next_states,
        (batch_size, seq_len_global, config.hidden_size),
        (batch_size, max(32, seq_len_global), config.hidden_size),
    )

    return next_states
