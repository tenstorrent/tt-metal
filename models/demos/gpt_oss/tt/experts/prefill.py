# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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


def _process_prefill_chunk(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config: ExpertConfig,
    prefill_sparsity,
    program_config: ProgramConfig,
    ep,
    tp,
    dense_core_grid=None,
):
    """Process a single chunk of the sequence in prefill mode.

    The chunk is processed in `down_split_size` sub-splits along the sequence. For each split the fused
    gate/up projection runs over the EP group's experts, the result is split into its gate and up halves,
    SwiGLU is applied and the down projection follows; the per-expert outputs are weighted, reduced and
    stream-concatenated. Working per split keeps the peak DRAM footprint at a few split-sized
    [E, split, N] activations rather than chunk-sized ones.
    """
    _, batch_size, seq_len, hidden_size = hidden_states.shape
    activation_dtype = ttnn.bfloat8_b
    TILE_SIZE = 32
    ip = weights.intermediate_padded_per_device
    output_tile = ttnn.Tile([32, 32])
    experts_per_ep = config.num_experts // ep

    # Routing weights: zero the experts owned by other EP groups, then [S, E] -> [B, E, S, 1]
    # Note: prefill_sparsity is cached and reused, don't deallocate it
    prefill_sparsity_reshaped = ttnn.reshape(prefill_sparsity, (1, config.num_experts))
    routing_weights = ttnn.mul(routing_weights, prefill_sparsity_reshaped, output_tensor=routing_weights)
    # Routing-aware sparsity for the fused gate/up projection: a 32-token group only needs the experts routed to
    # at least one of its tokens (for GPT-OSS-120B top-4 that is ~83 of 128 on average, vs all 128 with the dense
    # EP mask), and sparse_matmul's prefill cost is dominated by the per-(group, expert) pair overhead. The down
    # projection keeps the per-expert EP mask: its pairs are few and large, so per-group sparsity would only add
    # pairs. nnz is left to the kernel for the gate/up call -- it must equal count_nonzero exactly when given.
    # EP=1 (single-row meshes, TP only): every device holds all experts, so the MoE runs as dense matmuls --
    # one [split, H] x [H, 2Ip] matmul per expert for gate/up and one batched [E, split, Ip] x [E, Ip, H] matmul for
    # down. Measured on P150 for a 1024-token split of GPT-OSS-120B: gate/up 24.5 -> 6.4 (+1.2 concat) ms, down
    # 23.8 -> 3.7 ms versus the sparse_matmul path, whose 1D-multicast kernel keeps the whole M on <= 24 cores and
    # re-streams every expert's weights once per 32-token tile. EP>1 keeps the sparse path (per-EP-group mask).
    dense_moe = ep == 1 and dense_core_grid is not None
    if dense_moe and weights.gate_up_proj_per_expert is None:
        _cache_dense_weights(weights, config.num_experts)
    group_mask = (
        None if dense_moe else _group_expert_mask(routing_weights, seq_len, config.num_experts)
    )  # [1, S/32, 1, E] row-major
    # Note: permute/reshape operations return views - do not deallocate originals
    routing_weights = ttnn.permute(routing_weights, (1, 0))
    routing_weights = ttnn.reshape(routing_weights, (batch_size, config.num_experts, seq_len, 1))

    # This function consumes hidden_states and routing_weights (the split copies, or the tensors
    # themselves when there is a single split, are released as each split is processed).
    split_size = program_config.get_down_split_size(seq_len)
    if seq_len > split_size:
        hidden_list = ttnn.split(hidden_states, split_size, dim=2)
        hidden_states.deallocate(True)  # the splits are device copies; the chunk is dead from here on
        routing_list = ttnn.split(routing_weights, split_size, dim=2)
        routing_weights.deallocate(True)
    else:
        hidden_list = [hidden_states]
        routing_list = [routing_weights]

    # Process each split and stream-concatenate to avoid holding all split outputs.
    next_states_reduced_acc = None
    group_offset = 0
    for hidden_split, routing_split in zip(hidden_list, routing_list):
        split_len = hidden_split.shape[2]
        group_size = split_len // TILE_SIZE

        if dense_moe:
            # One dense matmul per expert over the whole split: [1, 1, split, H] x [1, 1, H, 2Ip] -> [1, 1, split, 2Ip],
            # concatenated along the expert dim into [1, E, split, 2Ip] (the layout the rest of the pipeline uses).
            hidden_4D = ttnn.unsqueeze_to_4D(hidden_split)
            per_expert = [
                ttnn.matmul(
                    hidden_4D,
                    w_e,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=activation_dtype,
                    core_grid=dense_core_grid,
                    compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
                )
                for w_e in weights.gate_up_proj_per_expert
            ]
            hidden_4D.deallocate(True)  # releases the split (view)
            gate_up = ttnn.concat(per_expert, dim=1)
            for t in per_expert:
                t.deallocate(True)
        else:
            # Group tokens into tiles: [1, B, split, H] -> [1, G, 32, H]. This reshape is a view of
            # hidden_split, so deallocating hidden_4D below releases the split itself (intended).
            hidden_4D = ttnn.unsqueeze_to_4D(hidden_split)
            hidden_4D = ttnn.reshape(hidden_4D, (1, group_size, TILE_SIZE, config.hidden_size))
            split_mask = ttnn.slice(
                group_mask, [0, group_offset, 0, 0], [1, group_offset + group_size, 1, config.num_experts]
            )
            group_offset += group_size

            # Fused gate/up projection: [1, G, 32, H] x [1, E, H, 2 * Ip] -> [1, G, 1, E, 32, 2 * Ip]
            # (skipped (group, expert) pairs are zero-filled by the op)
            gate_up = ttnn.sparse_matmul(
                hidden_4D,
                weights.gate_up_proj,
                sparsity=split_mask,
                nnz=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                program_config=program_config.get_prefill_gate_up_config(
                    hidden_4D.shape[2], weights.gate_up_proj.shape[3], k=hidden_4D.shape[-1]
                ),
                dtype=activation_dtype,
            )
            hidden_4D.deallocate(True)
            split_mask.deallocate(True)
            # Note: transpose/reshape operations return views - do not deallocate originals
            gate_up = ttnn.transpose(gate_up, 1, 3)
            gate_up = ttnn.reshape(gate_up, (batch_size, config.num_experts, split_len, 2 * ip))
        gate_up = ttnn.add(gate_up, weights.gate_up_proj_bias_t, output_tensor=gate_up)
        # Split at the tile-aligned half: gate = [..., :Ip], up = [..., Ip:]
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [batch_size, config.num_experts, split_len, ip])
        up = ttnn.slice(gate_up, [0, 0, 0, ip], [batch_size, config.num_experts, split_len, 2 * ip])
        gate_up.deallocate(True)

        # SwiGLU (consumes gate and up): [B, E, split, Ip]; the zero-padded columns stay exactly 0.
        down_input = apply_swiglu(gate, up, config)

        if dense_moe:
            # Routing weights are applied to the down INPUT ([E, split, Ip], a quarter of the down output) -- the down
            # projection is linear, so this is exact -- and the down bias is folded into a tiny [split, E] x [E, H]
            # matmul added after the expert reduction (as the decode path does). This removes two elementwise passes
            # over the [E, split, H] down output.
            down_input = apply_routing_weights(down_input, routing_split)
            # Batched dense down projection: [1, E, split, Ip] x [1, E, Ip, H] -> [1, E, split, H]
            down = ttnn.matmul(
                down_input,
                weights.down_proj_padded,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=activation_dtype,
                core_grid=dense_core_grid,
                compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
            )
            down_input.deallocate(True)
            next_states_reduced = reduce_experts(down)
            down.deallocate(True)
            routing_tokens = ttnn.permute(routing_split, (0, 3, 2, 1))  # [1, 1, split, E]
            bias_contrib = ttnn.matmul(
                routing_tokens,
                ttnn.reshape(weights.down_proj_bias, (1, 1, config.num_experts, config.hidden_size)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            routing_tokens.deallocate(True)
            next_states_reduced = ttnn.add(next_states_reduced, bias_contrib, output_tensor=next_states_reduced)
            bias_contrib.deallocate(True)
        else:
            down = ttnn.sparse_matmul(
                down_input,
                weights.down_proj,
                sparsity=prefill_sparsity,
                nnz=experts_per_ep,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                is_input_a_sparse=True,
                program_config=program_config.get_prefill_down_config(
                    down_input.shape[2], weights.down_proj.shape[-1], k=down_input.shape[-1]
                ),
                dtype=activation_dtype,
            )
            down_input.deallocate(True)

            # Apply bias and routing weights
            # Note: reshape returns a view - do not deallocate original
            next_states = ttnn.reshape(down, (batch_size, config.num_experts, split_len, config.hidden_size))
            bias_transposed = ttnn.transpose(weights.down_proj_bias, 1, 0)
            next_states = ttnn.add(next_states, bias_transposed, output_tensor=next_states)
            next_states = apply_routing_weights(next_states, routing_split)

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
        routing_split.deallocate(True)
    if group_mask is not None:
        group_mask.deallocate(True)

    return next_states_reduced_acc


# bf16 activations x bfloat8_b weights: HiFi2 keeps full bf8 precision; L1 accumulation in the packer.
_DENSE_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)


def _cache_dense_weights(weights: ExpertWeights, num_experts: int):
    """One-time, device-side preparation for the dense prefill path (kept for the model's lifetime):
    per-expert slices of the fused gate/up weights, and down_proj with K zero-padded to the tile multiple that the
    SwiGLU output carries (dense matmul checks logical K; the padded activation columns are exactly zero)."""
    hidden, n = weights.gate_up_proj.shape[2], weights.gate_up_proj.shape[3]
    per_expert = [ttnn.slice(weights.gate_up_proj, [0, e, 0, 0], [1, e + 1, hidden, n]) for e in range(num_experts)]
    object.__setattr__(weights, "gate_up_proj_per_expert", per_expert)  # ExpertWeights is a frozen dataclass
    pad_k = weights.intermediate_padded_per_device - weights.intermediate_size_per_device
    down_padded = (
        ttnn.pad(weights.down_proj, padding=[(0, 0), (0, 0), (0, pad_k), (0, 0)], value=0.0)
        if pad_k > 0
        else weights.down_proj
    )
    object.__setattr__(weights, "down_proj_padded", down_padded)


def _dense_core_grid(mesh_device):
    """Core grid for the dense prefill matmuls: the full compute grid, at most 12 wide (N = 24 output tiles)."""
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=min(grid.x, 12))


def _group_expert_mask(routing_weights, seq_len, num_experts):
    """[S, E] dense routing weights (0 for unselected experts) -> [1, S/32, 1, E] row-major bf16 mask with 1.0 where
    any token of the 32-token group routes to the expert (the sparse_matmul sparsity layout for a [1, G, 32, K] input).
    """
    groups = seq_len // 32
    grouped = ttnn.reshape(routing_weights, (1, groups, 32, num_experts))  # tile-aligned view
    used = ttnn.sum(grouped, dim=2, keepdim=True)  # [1, G, 1, E], > 0 iff some token in the group uses e
    mask = ttnn.gt(used, 0.0)
    used.deallocate(True)
    mask_rm = ttnn.to_layout(mask, ttnn.ROW_MAJOR_LAYOUT)
    mask.deallocate(True)
    return mask_rm


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
            dense_core_grid=_dense_core_grid(mesh_device),
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
