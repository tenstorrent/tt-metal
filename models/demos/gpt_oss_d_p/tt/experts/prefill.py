# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill forward pass for experts (seq_len>1)."""

import ttnn
from models.demos.gpt_oss.config import Mode

from .config import ExpertConfig, ProgramConfig
from .operations import (
    apply_expert_parallel_allreduce,
    apply_routing_weights,
    apply_sequence_parallel_reduce_scatter,
    apply_tensor_parallel_allreduce,
    reduce_experts,
)
from .weights import ExpertWeights


def _gather_for_sequence_parallel(hidden_states, routing_weights, mesh_config, ccl_manager):
    """All-gather SP-sharded inputs to full sequence length.

    Used when SP > 1 and EP > 1 share the same axis: each row needs all tokens
    so it can route them to its local subset of experts.
    """
    cluster_axis = mesh_config.sp_axis
    hidden_states_gathered = ttnn.all_gather(
        hidden_states,
        dim=2,
        cluster_axis=cluster_axis,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
    )
    routing_weights_gathered = ttnn.all_gather(
        routing_weights,
        dim=0,
        cluster_axis=cluster_axis,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
    )
    hidden_states.deallocate(True)
    routing_weights.deallocate(True)
    return hidden_states_gathered, routing_weights_gathered


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
    _, batch_size, seq_len, hidden_size = hidden_states.shape
    activation_dtype = ttnn.bfloat8_b
    TILE_SIZE = 32

    # Reshape for prefill (group tokens into tiles)
    # Note: unsqueeze_to_4D/reshape operations return views - do not deallocate originals
    hidden_states_4D = ttnn.unsqueeze_to_4D(hidden_states)
    hidden_states_4D = ttnn.reshape(hidden_states_4D, (1, seq_len // TILE_SIZE, TILE_SIZE, config.hidden_size))
    group_size = seq_len // TILE_SIZE

    # Prepare sparsity
    # Note: prefill_sparsity is cached and reused, don't deallocate it
    sparsity_repeated = ttnn.repeat(prefill_sparsity, (1, 1, group_size, 1))
    sparsity_layout = sparsity_repeated

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
        program_config=program_config.get_prefill_gate_up_config(
            hidden_states_4D.shape[2], weights.gate_proj.shape[3], k=hidden_states_4D.shape[-1]
        ),
        dtype=activation_dtype,
    )
    # Note: transpose/reshape operations return views - do not deallocate originals
    gate = ttnn.transpose(gate, 1, 3)
    gate = ttnn.reshape(gate, (batch_size, config.num_experts, seq_len, weights.intermediate_size_per_device))
    bias_transposed = ttnn.transpose(weights.gate_proj_bias, 1, 0)
    gate = ttnn.add(gate, bias_transposed, output_tensor=gate)

    # # Do partial swiglu before up projection to save memory (fused gate projection + swiglu gate activation)
    # Part 1
    gate = ttnn.clamp(gate, min=None, max=config.swiglu_limit, output_tensor=gate)
    gate_alpha = ttnn.mul(gate, config.alpha)
    gate_sigmoid = ttnn.sigmoid(gate_alpha)
    gate_alpha.deallocate(True)
    glu = ttnn.mul(gate, gate_sigmoid, output_tensor=gate)

    gate_sigmoid.deallocate(True)

    # Up projection
    up = ttnn.sparse_matmul(
        hidden_states_4D,
        weights.up_proj,
        sparsity=sparsity_layout,
        nnz=num_experts_per_tok,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_prefill_gate_up_config(
            hidden_states_4D.shape[2], weights.up_proj.shape[3], k=hidden_states_4D.shape[-1]
        ),
        dtype=activation_dtype,
    )
    hidden_states_4D.deallocate(True)
    # Note: sparsity_layout is created from repeat(prefill_sparsity), and prefill_sparsity
    # is reused later (line 123, 151). Don't deallocate as repeat may return a view/alias.

    # Note: transpose/reshape operations return views - do not deallocate originals
    up = ttnn.transpose(up, 1, 3)
    up = ttnn.reshape(up, (batch_size, config.num_experts, seq_len, weights.intermediate_size_per_device))
    bias_transposed = ttnn.transpose(weights.up_proj_bias, 1, 0)
    up = ttnn.add(up, bias_transposed, output_tensor=up)

    # Apply SwiGLU (consumes gate and up internally)

    # partial swiglu part 2
    up = ttnn.clamp(up, min=-config.swiglu_limit, max=config.swiglu_limit, output_tensor=up)
    up = ttnn.add(up, 1, output_tensor=up)
    down_input = ttnn.mul(up, glu, output_tensor=up)
    glu.deallocate(True)

    # Disabled regular swiglu to save memory by deallocating gate early.
    # down_input = apply_swiglu(gate, up, config)

    # Note: reshape returns a view - do not deallocate original
    down_input = ttnn.reshape(down_input, (1, config.num_experts, seq_len, weights.intermediate_size_per_device))

    # Update routing weights and sparsity for down projection
    num_experts_per_tok = config.num_experts // ep
    prefill_sparsity_reshaped = ttnn.reshape(prefill_sparsity, (1, config.num_experts))
    routing_weights = ttnn.mul(
        routing_weights,
        prefill_sparsity_reshaped,
        output_tensor=routing_weights,
    )

    # Note: permute/reshape operations return views - do not deallocate originals
    routing_weights = ttnn.permute(routing_weights, (1, 0))
    routing_weights = ttnn.reshape(routing_weights, (batch_size, config.num_experts, seq_len, 1))

    # Process down projection in splits if needed
    split_size = program_config.get_down_split_size(seq_len)
    if seq_len > split_size:
        down_input_list = ttnn.split(down_input, split_size, dim=2)
        down_input.deallocate(True)
        routing_weights_list = ttnn.split(routing_weights, split_size, dim=2)
        routing_weights.deallocate(True)
    else:
        down_input_list = [down_input]
        routing_weights_list = [routing_weights]

    # Process each split and stream-concatenate to avoid holding all split outputs.
    next_states_reduced_acc = None
    for i, down_input_split in enumerate(down_input_list):
        split_seq_len = down_input_split.shape[2]  # actual chunk size (handles uneven last chunk)
        down = ttnn.sparse_matmul(
            down_input_split,
            weights.down_proj,
            sparsity=prefill_sparsity,
            nnz=num_experts_per_tok,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            program_config=program_config.get_prefill_down_config(
                split_seq_len, weights.down_proj.shape[-1], k=down_input_split.shape[-1]
            ),
            dtype=activation_dtype,
        )
        down_input_split.deallocate(True)

        # Apply bias and routing weights
        # Note: reshape returns a view - do not deallocate original
        next_states = ttnn.reshape(down, (batch_size, config.num_experts, split_seq_len, config.hidden_size))
        bias_transposed = ttnn.transpose(weights.down_proj_bias, 1, 0)
        next_states = ttnn.add(next_states, bias_transposed, output_tensor=next_states)
        next_states = apply_routing_weights(next_states, routing_weights_list[i])

        # Reduce across experts
        next_states_reduced = reduce_experts(next_states)
        down.deallocate(True)
        if next_states_reduced_acc is None:
            next_states_reduced_acc = next_states_reduced
        else:
            # ToDo: Replace with slice_write.
            # Concat re-creates the output_tensor every iteration.
            next_states_concat = ttnn.concat([next_states_reduced_acc, next_states_reduced], dim=2)
            next_states_reduced_acc.deallocate(True)
            next_states_reduced.deallocate(True)
            next_states_reduced_acc = next_states_concat
        routing_weights_list[i].deallocate(True)

    return next_states_reduced_acc


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

    Expects SP-sharded inputs when SP > 1: hidden_states [1, B, S/sp, H] and
    routing_weights [S/sp, E].  Returns SP-sharded output [1, B, S/sp, H].

    Args:
        hidden_states: Input tensor [1, batch, seq_len_local, hidden_size]
        routing_weights: Router output [seq_len_local, num_experts]
        weights: Expert weights
        config: Expert configuration
        mesh_config: Mesh parallelization config
        mesh_device: TTNN mesh device
        ccl_manager: Communication manager
        program_config: Model-specific program configs
        prefill_sparsity: Cached prefill sparsity mask

    Returns:
        Expert output [1, batch, seq_len_local, hidden_size]
    """
    batch_dim = 1
    seq_dim = 2
    batch_size = hidden_states.shape[batch_dim]
    seq_len_local = hidden_states.shape[seq_dim]  # S/sp per device (or S when SP=1)

    if batch_size != 1:
        raise NotImplementedError(f"Currently only batch_size=1 supported, got {batch_size}")

    if seq_len_local <= 1:
        raise ValueError(f"Prefill mode requires seq_len>1, got {seq_len_local}. Use decode mode for single tokens.")

    TILE_SIZE = 32

    mode_config = mesh_config.get_config(Mode.PREFILL)
    ep, sp, tp = mode_config.ep, mode_config.sp, mode_config.tp

    # The per-device sequence length must be tile-aligned.
    if seq_len_local % TILE_SIZE != 0:
        raise ValueError(
            f"Per-device sequence length must be divisible by {TILE_SIZE} "
            f"(got {seq_len_local}, SP={sp}). Pad your sequence to a multiple of {TILE_SIZE * sp}."
        )

    # When SP and EP share the same axis (rows), all-gather the SP-sharded inputs so each row
    # sees all tokens and can route them to its local expert subset.
    # When EP=1 every row holds all experts and processes its shard independently — no gather needed.
    if sp > 1 and ep > 1:
        hidden_states, routing_weights = _gather_for_sequence_parallel(
            hidden_states, routing_weights, mesh_config, ccl_manager
        )

    # Chunk processing for very long sequences
    chunk_size = program_config.sequence_chunk_size
    if hidden_states.shape[seq_dim] > chunk_size:
        hidden_states_chunks = ttnn.split(hidden_states, chunk_size, dim=seq_dim)
        hidden_states.deallocate(True)
        routing_weights_chunks = ttnn.split(routing_weights, chunk_size, dim=0)
        routing_weights.deallocate(True)
    else:
        hidden_states_chunks = [hidden_states]
        routing_weights_chunks = [routing_weights]

    # Process each chunk and stream-concatenate to reduce peak DRAM usage.
    next_states_acc = None
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
        if next_states_acc is None:
            next_states_acc = next_states
        else:
            next_states_concat = ttnn.concat([next_states_acc, next_states], dim=2)
            next_states_acc.deallocate(True)
            next_states.deallocate(True)
            next_states_acc = next_states_concat
        hidden_chunk.deallocate(True)
        routing_chunk.deallocate(True)
    next_states = next_states_acc

    # Post-computation collectives.
    #
    # SP > 1, EP > 1 (same axis): reduce_scatter replaces the old EP all_reduce + SP all_gather
    # pair — it sums each row's partial expert contributions across rows while scattering back
    # to SP-sharded layout in a single fused op.
    #
    # SP > 1, EP = 1: each row computed its full shard independently; no row collective needed.
    #
    # SP = 1, EP > 1: standard EP all_reduce (current behaviour, unchanged).
    if sp > 1 and ep > 1:
        next_states = apply_sequence_parallel_reduce_scatter(next_states, mesh_config, ccl_manager)
    elif ep > 1:
        next_states = apply_expert_parallel_allreduce(next_states, mesh_config, ccl_manager)

    if tp > 1:
        next_states = apply_tensor_parallel_allreduce(
            next_states,
            mesh_config,
            mesh_device,
            seq_len_local,
            ccl_manager,
        )

    # Final reshape to [1, B, seq_len_local, H]
    next_states = ttnn.reshape(
        next_states,
        (1, batch_size, seq_len_local, config.hidden_size),
        (1, batch_size, max(32, seq_len_local), config.hidden_size),
    )

    return next_states
