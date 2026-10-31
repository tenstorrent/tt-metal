# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill forward pass for experts (seq_len>1)."""

import ttnn
from models.demos.gpt_oss.config import Mode

from .. import opt_switches
from .config import ExpertConfig, ProgramConfig
from .operations import (
    apply_expert_parallel_allreduce,
    apply_routing_weights,
    apply_sequence_parallel_allgather,
    apply_tensor_parallel_allreduce,
    reduce_experts,
)
from .weights import ExpertWeights


def _reshard_for_sequence_parallel(hidden_states, routing_weights, mesh_config, ccl_manager):
    """
    Convert replicated prefill inputs to SP row-sharded tensors using device-side CCL.

    This avoids host reads (`to_torch/get_device_tensors`) so it is trace-capture safe.
    The input tensors are replicated across rows, so reduce-scatter sums identical values.
    We rescale by 1/sp to recover the original values after sharding.
    """
    sp = mesh_config.get_config(Mode.PREFILL).sp
    if sp <= 1:
        return hidden_states, routing_weights

    cluster_axis = mesh_config.sp_axis
    scale = 1.0 / sp

    hidden_states_sharded = ttnn.reduce_scatter(
        hidden_states,
        dim=2,  # sequence dimension for hidden states: [1, B, S, H]
        cluster_axis=cluster_axis,
        memory_config=hidden_states.memory_config(),
        topology=ccl_manager.topology,
        num_links=ccl_manager.num_links,
    )
    routing_weights_sharded = ttnn.reduce_scatter(
        routing_weights,
        dim=0,  # sequence dimension for routing weights: [S, E]
        cluster_axis=cluster_axis,
        memory_config=routing_weights.memory_config(),
        topology=ccl_manager.topology,
        num_links=ccl_manager.num_links,
    )

    hidden_states_sharded = ttnn.mul(hidden_states_sharded, scale, output_tensor=hidden_states_sharded)
    routing_weights_sharded = ttnn.mul(routing_weights_sharded, scale, output_tensor=routing_weights_sharded)

    # Inputs are replaced by sharded outputs; release replicated tensors early.
    hidden_states.deallocate(True)
    routing_weights.deallocate(True)

    return hidden_states_sharded, routing_weights_sharded


def _m_blocked(cfg, m_tiles, max_in0_kb=280):
    """GPT_OSS_OPT_PREFILL_MBLOCK: the same 1D program config with per_core_M = all
    token tiles and out_block_h > 1, so each in1 (expert weight) block a core reads from DRAM is reused over
    out_block_h token tiles instead of being re-read for every tile. K blocking (in0_block_w), N split and subblocks are
    unchanged, so every output tile accumulates the same products in the same order. out_block_h is the largest divisor
    of m_tiles whose double-buffered in0 block (BFP8 ~1088 B per tile) stays under max_in0_kb."""
    tile_bytes = 1088
    out_block_h = 1
    for d in range(1, m_tiles + 1):
        if m_tiles % d == 0 and 2 * d * cfg.in0_block_w * tile_bytes <= max_in0_kb * 1024:
            out_block_h = d
    out_subblock_h, out_subblock_w, out_block_w = 1, cfg.out_subblock_w, cfg.out_block_w
    if opt_switches.on("PREFILL_SUBBLOCK"):
        # Larger output subblocks (up to 8 tiles of 16-bit Dest): each unpacked in0 tile feeds out_subblock_w products
        # and each in1 tile out_subblock_h, instead of one. Accumulation order over K per output tile is unchanged.
        out_block_w = cfg.per_core_N
        best = (1, 1)
        for w in range(1, out_block_w + 1):
            for h in range(1, out_block_h + 1):
                if out_block_w % w == 0 and out_block_h % h == 0 and h * w <= 8 and h * w > best[0] * best[1]:
                    best = (h, w)
        out_subblock_h, out_subblock_w = best
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=cfg.compute_with_storage_grid_size,
        in0_block_w=cfg.in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=m_tiles,
        per_core_N=cfg.per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _process_prefill_chunk(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config: ExpertConfig,
    prefill_sparsity,
    program_config: ProgramConfig,
    ep,
    tp,
    indexed_tensors=None,
):
    """Process a single chunk of the sequence in prefill mode."""
    _, batch_size, seq_len, hidden_size = hidden_states.shape
    activation_dtype = ttnn.bfloat8_b
    TILE_SIZE = 32

    # GPT_OSS_OPT_PREFILL_MBLOCK: EP=1 prefill experts without per-token-tile weight re-reads.
    if (opt_switches.on("PREFILL_MBLOCK") or indexed_tensors is not None) and ep == 1:
        return _process_prefill_chunk_mblock(
            hidden_states, routing_weights, weights, config, prefill_sparsity, program_config, indexed_tensors
        )

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
        down = ttnn.sparse_matmul(
            down_input_split,
            weights.down_proj,
            sparsity=prefill_sparsity,
            nnz=num_experts_per_tok,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            program_config=program_config.get_prefill_down_config(
                down_input_split.shape[2], weights.down_proj.shape[-1], k=down_input_split.shape[-1]
            ),
            dtype=activation_dtype,
        )
        down_input_split.deallocate(True)

        # Apply bias and routing weights
        split_seq_len = seq_len if seq_len < split_size else split_size
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


def _process_prefill_chunk_mblock(
    hidden_states, routing_weights, weights, config, prefill_sparsity, program_config, indexed_tensors=None
):
    """GPT_OSS_OPT_PREFILL_MBLOCK (EP=1 only, where the prefill sparsity is all ones).

    Same computation as _process_prefill_chunk, but the gate/up sparse matmuls take the chunk as one [1, 1, S, H]
    input (one sparsity row, nnz = E) instead of S/32 groups of one token tile, and all three expert matmuls use
    per_core_M = S/32 with out_block_h > 1 (_m_blocked). Each core then reads an expert's weight columns from DRAM
    S/32/out_block_h times instead of S/32 times. The gate/up outputs come out as [.., E, S, I] directly, so the two
    [1, S/32, 1, E, 32, I] -> [1, E, S, I] transposes of the original path go away.

    indexed_tensors (GPT_OSS_OPT_PREFILL_FUSEBIAS): run the three matmuls in indexed mode over all E experts
    (indices 0..E-1, so the compact output equals the expanded one) with each expert's bias fused into the matmul,
    instead of the three row-broadcast bias adds over [1, E, S, N] (and the output zero-fill of mask mode)."""
    _, batch_size, seq_len, hidden_size = hidden_states.shape
    activation_dtype = ttnn.bfloat8_b
    TILE_SIZE = 32
    m_tiles = seq_len // TILE_SIZE
    num_experts = config.num_experts
    I = weights.intermediate_size_per_device
    hidden_states_4D = ttnn.unsqueeze_to_4D(hidden_states)
    # The single in0 sender core of each expert matmul re-reads the whole [S, H] input once per expert (E times).
    # PREFILL_IN0L1: keep it L1-interleaved instead of DRAM (same values) while it fits.
    if opt_switches.on("PREFILL_IN0L1") and seq_len <= 2048:
        hidden_states_4D = ttnn.to_memory_config(hidden_states_4D, ttnn.L1_MEMORY_CONFIG)
    output_tile = ttnn.Tile([32, 32])

    fuse = indexed_tensors is not None

    def gate_up(weight, bias=None):
        out = ttnn.sparse_matmul(
            hidden_states_4D,
            weight,
            sparsity=prefill_sparsity,
            nnz=None if fuse else num_experts,
            indices=indexed_tensors.all_indices if fuse else None,
            bias=bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=_m_blocked(
                program_config.get_prefill_gate_up_config(TILE_SIZE, weight.shape[3], k=hidden_states_4D.shape[-1]),
                m_tiles,
            ),
            dtype=activation_dtype,
        )
        return ttnn.reshape(out, (batch_size, num_experts, seq_len, I))

    if fuse:
        gate = gate_up(weights.gate_proj, indexed_tensors.gate_bias)
    else:
        gate = gate_up(weights.gate_proj)
        bias_transposed = ttnn.transpose(weights.gate_proj_bias, 1, 0)
        gate = ttnn.add(gate, bias_transposed, output_tensor=gate)
    gate = ttnn.clamp(gate, min=None, max=config.swiglu_limit, output_tensor=gate)
    gate_alpha = ttnn.mul(gate, config.alpha)
    gate_sigmoid = ttnn.sigmoid(gate_alpha)
    gate_alpha.deallocate(True)
    glu = ttnn.mul(gate, gate_sigmoid, output_tensor=gate)
    gate_sigmoid.deallocate(True)

    if fuse:
        up = gate_up(weights.up_proj, indexed_tensors.up_bias)
        hidden_states_4D.deallocate(True)
    else:
        up = gate_up(weights.up_proj)
        hidden_states_4D.deallocate(True)
        bias_transposed = ttnn.transpose(weights.up_proj_bias, 1, 0)
        up = ttnn.add(up, bias_transposed, output_tensor=up)
    up = ttnn.clamp(up, min=-config.swiglu_limit, max=config.swiglu_limit, output_tensor=up)
    up = ttnn.add(up, 1, output_tensor=up)
    down_input = ttnn.mul(up, glu, output_tensor=up)
    glu.deallocate(True)
    down_input = ttnn.reshape(down_input, (1, num_experts, seq_len, I))

    prefill_sparsity_reshaped = ttnn.reshape(prefill_sparsity, (1, num_experts))
    routing_weights = ttnn.mul(routing_weights, prefill_sparsity_reshaped, output_tensor=routing_weights)
    routing_weights = ttnn.permute(routing_weights, (1, 0))
    routing_weights = ttnn.reshape(routing_weights, (batch_size, num_experts, seq_len, 1))

    split_size = program_config.get_down_split_size(seq_len)
    if seq_len > split_size:
        down_input_list = ttnn.split(down_input, split_size, dim=2)
        down_input.deallocate(True)
        routing_weights_list = ttnn.split(routing_weights, split_size, dim=2)
        routing_weights.deallocate(True)
    else:
        down_input_list = [down_input]
        routing_weights_list = [routing_weights]

    next_states_reduced_acc = None
    for i, down_input_split in enumerate(down_input_list):
        split_seq_len = down_input_split.shape[2]
        down = ttnn.sparse_matmul(
            down_input_split,
            weights.down_proj,
            sparsity=prefill_sparsity,
            nnz=None if fuse else num_experts,
            indices=indexed_tensors.all_indices if fuse else None,
            bias=indexed_tensors.down_bias if fuse else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            program_config=_m_blocked(
                program_config.get_prefill_down_config(
                    split_seq_len, weights.down_proj.shape[-1], k=down_input_split.shape[-1]
                ),
                split_seq_len // TILE_SIZE,
            ),
            dtype=activation_dtype,
        )
        down_input_split.deallocate(True)
        next_states = ttnn.reshape(down, (batch_size, num_experts, split_seq_len, config.hidden_size))
        if not fuse:
            bias_transposed = ttnn.transpose(weights.down_proj_bias, 1, 0)
            next_states = ttnn.add(next_states, bias_transposed, output_tensor=next_states)
        next_states = apply_routing_weights(next_states, routing_weights_list[i])
        next_states_reduced = reduce_experts(next_states)
        down.deallocate(True)
        if next_states_reduced_acc is None:
            next_states_reduced_acc = next_states_reduced
        else:
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
    indexed_tensors=None,
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
        Expert output [1, batch, seq_len, hidden_size]
    """
    activation_dtype = ttnn.bfloat8_b
    batch_dim = 1
    seq_dim = 2
    batch_size = hidden_states.shape[batch_dim]
    seq_len_global = hidden_states.shape[seq_dim]

    if batch_size != 1:
        raise NotImplementedError(f"Currently only batch_size=1 supported, got {batch_size}")

    if seq_len_global <= 1:
        raise ValueError(
            f"Prefill mode requires seq_len>1, got {seq_len_global}. " f"Use decode mode for single tokens."
        )

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
            **({"indexed_tensors": indexed_tensors} if indexed_tensors is not None else {}),
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

    # Expert parallel communication
    if ep > 1:
        next_states = apply_expert_parallel_allreduce(next_states, mesh_config, ccl_manager)

    # Tensor parallel communication
    if tp > 1:
        next_states = apply_tensor_parallel_allreduce(
            next_states,
            mesh_config,
            mesh_device,
            seq_len_global,
            ccl_manager,
        )

    # Sequence parallel all-gather
    if sp > 1:
        next_states = apply_sequence_parallel_allgather(next_states, mesh_config, ccl_manager)

    # Final reshape
    next_states = ttnn.reshape(
        next_states,
        (1, batch_size, seq_len_global, config.hidden_size),
        (1, batch_size, max(32, seq_len_global), config.hidden_size),
    )

    return next_states
