# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill forward pass for experts (seq_len>1)."""

import os

import torch
from loguru import logger

import ttnn
from models.demos.gpt_oss.config import Mode

from .config import SORTED_MOE_MIN_EXPERTS, ExpertConfig, ProgramConfig
from .operations import (
    apply_expert_parallel_allreduce,
    apply_routing_weights,
    apply_sequence_parallel_allgather,
    apply_swiglu,
    apply_swiglu_fused,
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
    sp,
    tp,
    dense_core_grid=None,
    scratch=None,
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
    # Note: prefill_sparsity is cached and reused, don't deallocate it (with EP=1 it is all ones: skip the pass)
    if ep > 1:
        prefill_sparsity_reshaped = ttnn.reshape(prefill_sparsity, (1, config.num_experts))
        routing_weights = ttnn.mul(routing_weights, prefill_sparsity_reshaped, output_tensor=routing_weights)
    # Single-row meshes (EP=1, SP=1: every device holds all experts and the same tokens; all Blackhole meshes and
    # LoudBox) run the MoE as dense matmuls -- see the comment block ahead of the helpers below for the three
    # sub-paths. Multi-row meshes (Galaxy: sequence-parallel rows, sp > 1) keep the pre-existing sparse_matmul path
    # below, whose per-device work does not depend on what the other rows hold; the expert-sorted sub-path in
    # particular plans on the host from ONE device's routed-token counts, which is only valid when every device
    # holds the same tokens. The sparse path's gate/up uses the per-32-token-group mask (`_group_expert_mask`: a
    # group only needs the experts routed to at least one of its tokens; nnz is left to the kernel, it must equal
    # count_nonzero exactly when given) and, for EP>1, the per-EP-group mask on the down projection.
    dense_moe = ep == 1 and sp == 1 and dense_core_grid is not None
    sorted_moe = dense_moe and config.num_experts >= _SORTED_MOE_MIN_EXPERTS
    group_mask = (
        None if dense_moe else _group_expert_mask(routing_weights, seq_len, config.num_experts)
    )  # [1, S/32, 1, E] row-major
    # Token-major routing weights ([1, 1, S, E], a view) for the dense path's folded down-bias matmul; sliced per split.
    routing_tokens_all = ttnn.reshape(routing_weights, (1, 1, seq_len, config.num_experts)) if dense_moe else None
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
    token_offset = 0
    gate_up_per_expert = None  # per-expert weight copies for the dense loop: made on first use, freed with the chunk
    try:
        for hidden_split, routing_split in zip(hidden_list, routing_list):
            next_states_reduced_acc, group_offset, token_offset, gate_up_per_expert = _process_split(
                hidden_split,
                routing_split,
                next_states_reduced_acc,
                group_offset,
                token_offset,
                gate_up_per_expert,
                weights,
                config,
                prefill_sparsity,
                program_config,
                dense_moe,
                sorted_moe,
                dense_core_grid,
                scratch,
                group_mask,
                routing_tokens_all,
                batch_size,
                ep,
            )
    finally:
        if gate_up_per_expert is not None:
            for w_e in gate_up_per_expert:
                w_e.deallocate(True)
    if group_mask is not None:
        group_mask.deallocate(True)
    if routing_tokens_all is not None:
        routing_tokens_all.deallocate(True)

    return next_states_reduced_acc


def _process_split(
    hidden_split,
    routing_split,
    next_states_reduced_acc,
    group_offset,
    token_offset,
    gate_up_per_expert,
    weights,
    config,
    prefill_sparsity,
    program_config,
    dense_moe,
    sorted_moe,
    dense_core_grid,
    scratch,
    group_mask,
    routing_tokens_all,
    batch_size,
    ep,
):
    """One `down_split_size` split of a chunk (see `_process_prefill_chunk`); returns the updated accumulator,
    offsets and per-expert weight list. Consumes hidden_split and routing_split."""
    activation_dtype = ttnn.bfloat8_b
    TILE_SIZE = 32
    ip = weights.intermediate_padded_per_device
    output_tile = ttnn.Tile([32, 32])
    experts_per_ep = config.num_experts // ep
    split_len = hidden_split.shape[2]
    group_size = split_len // TILE_SIZE

    if dense_moe:
        hidden_4D = ttnn.unsqueeze_to_4D(hidden_split)  # [1, 1, split, H] (view of the split)
        plan = None
        if split_len > _DENSE_BMM_MAX_TOKENS and sorted_moe:
            plan = _sorted_moe_plan(routing_tokens_all, token_offset, split_len, config)
        if plan is not None:
            next_states_reduced = _sorted_moe_forward(
                hidden_4D,
                plan,
                routing_tokens_all,
                token_offset,
                split_len,
                weights,
                config,
                activation_dtype,
                dense_core_grid,
                scratch,
            )
        else:
            if split_len <= _DENSE_BMM_MAX_TOKENS:
                gate_up = _dense_gate_up_batched(hidden_4D, weights, config, activation_dtype, dense_core_grid)
            else:
                if gate_up_per_expert is None:
                    gate_up_per_expert = _slice_experts(weights.gate_up_proj, range(config.num_experts))
                gate_up = _dense_gate_up_loop(hidden_4D, gate_up_per_expert, weights, activation_dtype, dense_core_grid)
            next_states_reduced = _dense_tail(
                gate_up,
                routing_split,
                routing_tokens_all,
                token_offset,
                split_len,
                weights,
                config,
                activation_dtype,
                dense_core_grid,
                ip,
            )
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
        # Apply bias and routing weights, reduce across experts
        # Note: reshape returns a view - do not deallocate original
        next_states = ttnn.reshape(down, (batch_size, config.num_experts, split_len, config.hidden_size))
        bias_transposed = ttnn.transpose(weights.down_proj_bias, 1, 0)
        next_states = ttnn.add(next_states, bias_transposed, output_tensor=next_states)
        next_states = apply_routing_weights(next_states, routing_split)
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
    token_offset += split_len
    return next_states_reduced_acc, group_offset, token_offset, gate_up_per_expert


# ---------------------------------------------------------------------------------------------------------------------
# Single-row (EP=1, SP=1) dense MoE. Measured on P150 (gpt-oss-120b, 1024-token split) against the sparse_matmul path,
# whose 1D-multicast kernel keeps the whole M on <= 24 cores and re-streams every expert's weights once per 32-token
# tile: gate/up 24.5 -> 6.4 ms, down 23.8 -> 3.7 ms. Three sub-paths, chosen per split:
#   * splits of <= _DENSE_BMM_MAX_TOKENS tokens (the traced 128-token prefill): the activations are replicated per
#     expert and ONE batched matmul runs over all experts (`_dense_gate_up_batched`; 128 separate launches cost ~30 us
#     each on device and dominated 128-token prefills);
#   * longer splits, models with >= _SORTED_MOE_MIN_EXPERTS experts: hot/cold expert-sorted MoE over the routed rows
#     only (`_sorted_moe_plan` / `_sorted_moe_forward`);
#   * longer splits otherwise (gpt-oss-20b): one matmul per expert over the whole split, concatenated
#     (`_dense_gate_up_loop`).
# In all of them SwiGLU is one fused binary op, the routing weights are applied to the down INPUT and the down bias is
# folded into a tiny [split, E] x [E, H] matmul (`_dense_tail`). No weight copies persist: the per-expert / hot-expert
# weights the loop and the sorted path need are sliced from the fused tensors per call and freed.
# ---------------------------------------------------------------------------------------------------------------------


def _dense_gate_up_batched(hidden_4D, weights, config, activation_dtype, dense_core_grid):
    """Fused gate/up for a short split, [1, 1, split, H] -> [1, E, split, 2Ip] with the bias added, as ONE batched
    matmul over all experts (the activations are replicated per expert). Consumes hidden_4D."""
    hidden_rep = ttnn.repeat(hidden_4D, ttnn.Shape((1, config.num_experts, 1, 1)))
    hidden_4D.deallocate(True)
    gate_up = _batched_matmul(hidden_rep, weights.gate_up_proj, activation_dtype, dense_core_grid)
    hidden_rep.deallocate(True)
    return ttnn.add(gate_up, weights.gate_up_proj_bias_t, output_tensor=gate_up)


def _dense_gate_up_loop(hidden_4D, gate_up_per_expert, weights, activation_dtype, dense_core_grid):
    """Fused gate/up for a long split, [1, 1, split, H] -> [1, E, split, 2Ip] with the bias added: one matmul per expert
    over the whole split (weights from `_slice_experts`), concatenated. The bias is fused into the matmul when the
    per-expert bias tiles exist (models below SORTED_MOE_MIN_EXPERTS experts, see weights.py), otherwise added once
    after the concat. Consumes hidden_4D."""
    biases = weights.gate_up_proj_bias_per_expert or [None] * len(gate_up_per_expert)
    per_expert = [
        ttnn.linear(
            hidden_4D,
            w_e,
            bias=b_e,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=activation_dtype,
            core_grid=dense_core_grid,
            compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
        )
        for w_e, b_e in zip(gate_up_per_expert, biases)
    ]
    hidden_4D.deallocate(True)
    gate_up = ttnn.concat(per_expert, dim=1)
    for t in per_expert:
        t.deallocate(True)
    if weights.gate_up_proj_bias_per_expert is None:
        gate_up = ttnn.add(gate_up, weights.gate_up_proj_bias_t, output_tensor=gate_up)
    return gate_up


def _dense_tail(
    gate_up,
    routing_split,
    routing_tokens_all,
    token_offset,
    split_len,
    weights,
    config,
    activation_dtype,
    dense_core_grid,
    ip,
):
    """[1, E, split, 2Ip] gate/up (bias included) -> [1, 1, split, H] MoE output for the split. Consumes gate_up.
    The routing weights are applied to the down INPUT (a fraction of the down output; exact since down is linear) and
    the down bias is folded into a tiny [split, E] x [E, H] matmul added after the expert reduction."""
    E = config.num_experts
    gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, E, split_len, ip])
    up = ttnn.slice(gate_up, [0, 0, 0, ip], [1, E, split_len, 2 * ip])
    gate_up.deallocate(True)
    down_input = apply_swiglu_fused(gate, up, config)  # one fused binary op
    gate.deallocate(True)
    up.deallocate(True)
    down_input = apply_routing_weights(down_input, routing_split)
    down = ttnn.matmul(
        down_input,
        weights.down_proj,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=activation_dtype,
        core_grid=dense_core_grid,
        compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
    )
    down_input.deallocate(True)
    reduced = reduce_experts(down)
    down.deallocate(True)
    return _add_folded_down_bias(reduced, routing_tokens_all, token_offset, split_len, weights, config)


def _add_folded_down_bias(reduced, routing_tokens_all, token_offset, split_len, weights, config):
    """reduced [1, 1, split, H] += routing_weights[split, E] @ down_bias[E, H] (in place)."""
    routing_tokens = ttnn.slice(
        routing_tokens_all, [0, 0, token_offset, 0], [1, 1, token_offset + split_len, config.num_experts]
    )
    bias_contrib = ttnn.matmul(
        routing_tokens,
        ttnn.reshape(weights.down_proj_bias, (1, 1, config.num_experts, config.hidden_size)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    if split_len != routing_tokens_all.shape[2]:  # a full-range slice aliases its input
        routing_tokens.deallocate(True)
    reduced = ttnn.add(reduced, bias_contrib, output_tensor=reduced)
    bias_contrib.deallocate(True)
    return reduced


def _slice_experts(w, expert_ids):
    """Independent [1, 1, K, N] device copies of the given experts of a [1, E, K, N] weight tensor (ttnn.slice; the
    caller frees them)."""
    return [ttnn.slice(w, [0, e, 0, 0], [1, e + 1, w.shape[2], w.shape[3]]) for e in expert_ids]


def _gather_experts(w, expert_ids):
    """[1, len(ids), K, N] device copy of the given experts' weights (per-expert slices, concatenated; the caller frees
    the result -- with a single expert the slice itself is returned, ttnn.concat of one tensor would alias it)."""
    parts = _slice_experts(w, expert_ids)
    if len(parts) == 1:
        return parts[0]
    out = ttnn.concat(parts, dim=1)
    for p in parts:
        p.deallocate(True)
    return out


_SORTED_MOE_DEBUG = os.getenv("GPT_OSS_SORTED_MOE_DEBUG", "0") == "1"
# Last plan chosen by _sorted_moe_plan ({"split", "cap", "hot"}); read by tests to assert which path ran.
LAST_SORTED_MOE_PLAN = {}
_SORTED_MOE_MAX_HOT = 16  # more hot experts than this -> dense per-expert loop for the split
_SORTED_CAPS = (32, 64, 96, 128, 160, 192, 256)  # gathered rows per cold expert the plan may choose from
# Cost model (ms per 1024-token split, P150, 120B shapes) used to pick the hot/cold threshold on the host:
_SORTED_FIXED_MS, _SORTED_PER_KROW_MS, _HOT_FIXED_MS, _HOT_PER_EXPERT_MS = 2.5, 0.27, 1.0, 0.25
_DENSE_PER_EXPERT_MS = (
    0.125  # per-expert cost of the dense loop over a 1024-token split (gate/up + concat + down share)
)
_SORTED_MOE_MIN_EXPERTS = SORTED_MOE_MIN_EXPERTS  # see experts/config.py


def _sorted_moe_plan(routing_tokens_all, token_offset, split_len, config):
    """Host-side plan for one split from the per-expert routed-token counts (one small device->host read).

    Real GPT-OSS routing is very skewed (the hottest expert of a 1024-token split often takes 30-90% of the
    tokens), so the experts are partitioned: HOT experts (count > cap) run dense over the whole split as a small
    batched group, COLD experts run expert-sorted with `cap` gathered rows each. `cap` is chosen from a small cost
    model over the count distribution. Returns (routing^T [1, 1, E, split], cap, hot_ids, cold mask) or None (use
    the dense per-expert loop when too many experts are hot)."""
    E = config.num_experts
    # This does a device->host read of the per-expert counts, so it must never run under trace capture (a captured
    # plan would be replayed for other prompts). The sorted path is only taken for splits longer than
    # MAX_TRACEABLE_PREFILL_TOKENS, and model_config.py checks every traced prefill length against that bound.
    assert split_len > MAX_TRACEABLE_PREFILL_TOKENS, "the sorted MoE path is for eager (untraced) long splits only"
    routing_tokens = ttnn.slice(routing_tokens_all, [0, 0, token_offset, 0], [1, 1, token_offset + split_len, E])
    routing_t = ttnn.transpose(routing_tokens, 2, 3)  # [1, 1, E, split]
    if split_len != routing_tokens_all.shape[2]:  # a full-range slice aliases its input
        routing_tokens.deallocate(True)
    active = ttnn.gt(routing_t, 0.0)
    # Count in fp32: a bf16 sum has a spacing of 2 above 256, so 257 routed tokens would read as 256 and the topk
    # below would silently drop one of them.
    active32 = ttnn.typecast(active, ttnn.float32)
    active.deallocate(True)
    counts = ttnn.sum(active32, dim=3, keepdim=True)  # [1, 1, E, 1] fp32
    active32.deallocate(True)
    # the dense path only runs on single-row meshes, where every device holds the same tokens and routing weights, so
    # one device's counts suffice (mesh tensors need a composer for a direct to_torch)
    counts_host = ttnn.to_torch(ttnn.get_device_tensors(counts)[0]).reshape(-1).to(torch.int64)
    best = None
    for cap in _SORTED_CAPS:
        if cap > split_len:
            break
        hot = (counts_host > cap).sum().item()
        if hot > _SORTED_MOE_MAX_HOT:
            continue
        cost = (
            _SORTED_FIXED_MS
            + _SORTED_PER_KROW_MS * (E * cap / 1024)
            + (_HOT_FIXED_MS + _HOT_PER_EXPERT_MS * hot if hot else 0.0)
        )
        if best is None or cost < best[0]:
            best = (cost, cap, hot)
    # The sorted path only pays off when the routed rows are few relative to E x split: for gpt-oss-120b (E=128,
    # ~32 routed tokens per expert per 1024) it is ~3x cheaper; for gpt-oss-20b (E=32, ~128 per expert) the dense
    # per-expert loop is as cheap and has no host round-trip, so it is kept.
    dense_cost = _DENSE_PER_EXPERT_MS * E * split_len / 1024
    if best is None or best[0] >= dense_cost:
        routing_t.deallocate(True)
        counts.deallocate(True)
        return None
    _, cap, n_hot = best
    hot_ids = [int(e) for e in torch.nonzero(counts_host > cap).reshape(-1).tolist()]
    LAST_SORTED_MOE_PLAN.update(split=split_len, cap=cap, hot=n_hot)
    if _SORTED_MOE_DEBUG:
        top = counts_host.topk(min(4, E)).values.tolist()
        logger.info(
            f"SORTED-MOE split={split_len} cap={cap} hot={n_hot} top4={top} zero={(counts_host == 0).sum().item()}"
        )
    # cold mask on device (1.0 for experts handled by the sorted path): no per-split host upload
    cold_mask32 = ttnn.le(counts, float(cap))
    cold_mask_t = ttnn.typecast(cold_mask32, ttnn.bfloat16)
    cold_mask32.deallocate(True)
    counts.deallocate(True)
    return routing_t, cap, hot_ids, cold_mask_t


def _sorted_moe_forward(
    hidden_4D,
    plan,
    routing_tokens_all,
    token_offset,
    split_len,
    weights,
    config,
    activation_dtype,
    dense_core_grid,
    scratch,
):
    """Hot/cold expert-sorted MoE for one split ([1, 1, split, H] -> [1, 1, split, H]); consumes hidden_4D.

    Cold experts: topk over the transposed routing weights gives each expert its `cap` largest-weight tokens (all
    its routed tokens, then zero-weight fillers); ttnn.embedding gathers those rows (and one-hot rows from an
    identity generated on device), gate/up and down run as batched matmuls over the gathered [E, cap, *] rows only,
    each row is scaled by its slot weight (zeroed for hot experts) and scattered back with one-hot^T @ rows. Hot
    experts (their routed-token count exceeds `cap`): the activations are replicated per hot expert and gate/up /
    down run as small batched matmuls over the whole split (weights sliced from the fused tensors for this split),
    weighted by their routing weights and reduced. The math equals the dense path."""
    routing_t, cap, hot_ids, cold_mask_t = plan
    E, H, ip = config.num_experts, config.hidden_size, weights.intermediate_padded_per_device
    device = weights.gate_up_proj.device()
    table = ttnn.reshape(hidden_4D, (split_len, H))
    if table.dtype != ttnn.bfloat16:  # embedding gathers from a bf16 table
        table16 = ttnn.typecast(table, ttnn.bfloat16)
        hidden_4D.deallocate(True)
        table = table16
        hidden_4D = ttnn.reshape(table, (1, 1, split_len, H))

    # ---- cold experts: sorted / gathered rows ----
    vals, idx = ttnn.topk(routing_t, k=cap, dim=3, largest=True)  # [1, 1, E, cap]
    if hot_ids:
        vals = ttnn.mul(vals, cold_mask_t, output_tensor=vals)  # hot experts contribute via the dense group below
        hot_idx_t = ttnn.from_torch(
            torch.tensor([hot_ids], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )  # [1, n_hot]
        routing_t_table = ttnn.reshape(routing_t, (E, split_len))  # gather table for the hot routing rows
    else:
        routing_t.deallocate(True)
    cold_mask_t.deallocate(True)
    idx_flat = ttnn.reshape(ttnn.to_layout(ttnn.typecast(idx, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (1, E * cap))
    idx.deallocate(True)
    rows = ttnn.reshape(ttnn.embedding(idx_flat, table, layout=ttnn.TILE_LAYOUT), (1, E, cap, H))
    eye = _identity_rows(split_len, device, scratch)
    onehot = ttnn.reshape(ttnn.embedding(idx_flat, eye, layout=ttnn.TILE_LAYOUT), (1, 1, E * cap, split_len))
    idx_flat.deallocate(True)
    gate_up = _batched_matmul(rows, weights.gate_up_proj, activation_dtype, dense_core_grid)
    rows.deallocate(True)
    gate_up = ttnn.add(gate_up, weights.gate_up_proj_bias_t, output_tensor=gate_up)
    gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, E, cap, ip])
    up = ttnn.slice(gate_up, [0, 0, 0, ip], [1, E, cap, 2 * ip])
    gate_up.deallocate(True)
    act = apply_swiglu_fused(gate, up, config)
    gate.deallocate(True)
    up.deallocate(True)
    slot_w = ttnn.to_layout(ttnn.reshape(ttnn.to_layout(vals, ttnn.ROW_MAJOR_LAYOUT), (1, E, cap, 1)), ttnn.TILE_LAYOUT)
    vals.deallocate(True)
    act = ttnn.mul(act, slot_w, output_tensor=act)
    slot_w.deallocate(True)
    down = ttnn.matmul(
        act,
        weights.down_proj,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=activation_dtype,
        core_grid=dense_core_grid,
        compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
    )
    act.deallocate(True)
    out = ttnn.matmul(  # scatter back: out[split, H] = onehot^T [split, E*cap] @ down[E*cap, H]
        onehot,
        ttnn.reshape(down, (1, 1, E * cap, H)),
        transpose_a=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=activation_dtype,
        core_grid=dense_core_grid,
        compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
    )
    onehot.deallocate(True)
    down.deallocate(True)

    # ---- hot experts: dense over the whole split, as one small batched group ----
    if hot_ids:
        n_hot = len(hot_ids)
        w_hot = _gather_experts(weights.gate_up_proj, hot_ids)  # [1, n_hot, H, 2Ip]
        hidden_rep = ttnn.repeat(hidden_4D, ttnn.Shape((1, n_hot, 1, 1)))
        gu_hot = ttnn.matmul(
            hidden_rep,
            w_hot,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=activation_dtype,
            core_grid=dense_core_grid,
            compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
        )
        hidden_rep.deallocate(True)
        w_hot.deallocate(True)
        # fused gate/up bias rows of the hot experts, [1, n_hot, 1, 2Ip]: one gather from the [E, 2Ip] bias table
        b_rows = ttnn.embedding(hot_idx_t, _bias_table(weights, E, ip, scratch), layout=ttnn.ROW_MAJOR_LAYOUT)
        b_hot = ttnn.to_layout(ttnn.reshape(b_rows, (1, n_hot, 1, 2 * ip)), ttnn.TILE_LAYOUT)
        b_rows.deallocate(True)
        gu_hot = ttnn.add(gu_hot, b_hot, output_tensor=gu_hot)
        b_hot.deallocate(True)
        gate_h = ttnn.slice(gu_hot, [0, 0, 0, 0], [1, n_hot, split_len, ip])
        up_h = ttnn.slice(gu_hot, [0, 0, 0, ip], [1, n_hot, split_len, 2 * ip])
        gu_hot.deallocate(True)
        act_h = apply_swiglu_fused(gate_h, up_h, config)
        gate_h.deallocate(True)
        up_h.deallocate(True)
        # routing weights of the hot experts, [1, n_hot, split, 1]: gather rows of routing^T [E, split] (one op, no
        # per-expert slice program variants)
        rw_rows = ttnn.embedding(hot_idx_t, routing_t_table, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, n_hot, split]
        rw_hot = ttnn.to_layout(ttnn.reshape(rw_rows, (1, n_hot, split_len, 1)), ttnn.TILE_LAYOUT)
        rw_rows.deallocate(True)
        act_h = ttnn.mul(act_h, rw_hot, output_tensor=act_h)
        rw_hot.deallocate(True)
        wd_hot = _gather_experts(weights.down_proj, hot_ids)  # [1, n_hot, Ip, H]
        down_h = ttnn.matmul(
            act_h,
            wd_hot,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=activation_dtype,
            core_grid=dense_core_grid,
            compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
        )
        act_h.deallocate(True)
        wd_hot.deallocate(True)
        hot_out = reduce_experts(down_h)  # [1, 1, split, H]
        down_h.deallocate(True)
        out = ttnn.add(out, hot_out, output_tensor=out)
        hot_out.deallocate(True)
        hot_idx_t.deallocate(True)
        routing_t.deallocate(True)
    table.deallocate(True)  # releases the split (view) or the bf16 copy
    return _add_folded_down_bias(out, routing_tokens_all, token_offset, split_len, weights, config)


def _identity_rows(n, device, cache):
    """[n, n] bf16 row-major identity generated ON DEVICE (the one-hot table for the sorted path's scatter matmul):
    arange -> broadcast equality -> bf16; compared in fp32 because bf16 integers are only exact up to 256. ~0.13 ms
    for n = 1024 on P150. Kept in the per-call `cache` (prefill_forward frees it) so the splits of one call share it
    while nothing persists per layer."""
    if ("eye", n) in cache:
        return cache[("eye", n)]
    ar = ttnn.arange(0, n, 1, dtype=ttnn.float32, device=device)  # [n] fp32, row-major
    eye32 = ttnn.eq(ttnn.reshape(ar, (n, 1)), ttnn.reshape(ar, (1, n)))  # [n, n] fp32 (reshapes are views)
    ar.deallocate(True)
    eye16 = ttnn.typecast(eye32, ttnn.bfloat16)
    eye32.deallocate(True)
    if eye16.layout == ttnn.ROW_MAJOR_LAYOUT:  # embedding gathers from a row-major table
        eye = eye16
    else:
        eye = ttnn.to_layout(eye16, ttnn.ROW_MAJOR_LAYOUT)
        eye16.deallocate(True)
    cache[("eye", n)] = eye
    return eye


def _bias_table(weights, E, ip, cache):
    """[E, 2Ip] bf16 row-major copy of the fused gate/up bias (ttnn.embedding gathers hot experts' rows from it); one
    per prefill_forward call, kept in the per-call `cache`."""
    if "bias" not in cache:
        cache["bias"] = ttnn.to_layout(ttnn.reshape(weights.gate_up_proj_bias, (E, 2 * ip)), ttnn.ROW_MAJOR_LAYOUT)
    return cache["bias"]


# bf16 activations x bfloat4_b weights: HiFi2 keeps full bf8 precision; L1 accumulation in the packer.
_DENSE_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)

# Upper bound on the tokens per device of a split that the one-launch batched matmul handles (per_core_M <= 8), and
# therefore on any prefill length that may be TRACED: the expert-sorted path above this length reads the routed-token
# counts back to the host to plan, which a trace would bake in (model_config.py checks the traced lengths against it).
_DENSE_BMM_MAX_TOKENS = 256
MAX_TRACEABLE_PREFILL_TOKENS = _DENSE_BMM_MAX_TOKENS

_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.float32: 4096}
# Headroom below ttnn.get_max_worker_l1_unreserved_size(): that figure is measured from the kernel-config base, while
# circular buffers start ~69 KiB higher at the allocator base (111616 B on Blackhole today), and any L1-resident tensor
# alive during prefill lowers the ceiling further.
_L1_RESERVE_BYTES = 192 * 1024
_l1_budget = None


def _bmm_cb_bytes(per_core_m, per_core_n, in0_block_w, kt, in0_tile_bytes, in1_tile_bytes):
    """Static circular-buffer bytes of one core of the MatmulMultiCoreReuse (batched) program with interleaved inputs
    (matmul_multicore_reuse_optimized_program_factory.cpp): double-buffered in0 and in1 blocks, the bf8 output block
    and, when the packer's L1 accumulation engages (more than two K blocks), a separate bf16 intermediate block."""
    out = per_core_m * per_core_n * _TILE_BYTES[ttnn.bfloat8_b]
    interm = per_core_m * per_core_n * _TILE_BYTES[ttnn.bfloat16] if kt // in0_block_w > 2 else 0
    return 2 * per_core_m * in0_block_w * in0_tile_bytes + 2 * per_core_n * in0_block_w * in1_tile_bytes + out + interm


def _bmm_config(core_grid, mt, kt, nt, in0_dtype, in1_dtype):
    """MatmulMultiCoreReuseProgramConfig for a batched matmul with exactly one [M x N] output block per batch entry:
    per_core_M = M and per_core_N = N (the kernel requires the whole N per core, and with several blocks per core it
    advances between them by a whole batch entry -- a per_core_M < M block on a core that owns two experts would read
    the wrong expert), so the only degree of freedom is the K block: the widest divisor of Kt whose circular buffers
    fit the device's worker L1 (queried once, minus a reserve). At TP=8 that is the shipped in0_block_w=6 for the
    [8 x 24]-tile block; at TP=1/2 (180 / 90 output tiles) no K block fits and None is returned, so the caller lets
    ttnn pick (its automatic config is L1-checked and correct for batched inputs)."""
    global _l1_budget
    if _l1_budget is None:
        _l1_budget = ttnn.get_max_worker_l1_unreserved_size() - _L1_RESERVE_BYTES
    in0_tile, in1_tile = _TILE_BYTES[in0_dtype], _TILE_BYTES[in1_dtype]
    for in0_block_w in (d for d in (6, 5, 4, 3, 2, 1) if kt % d == 0):
        if _bmm_cb_bytes(mt, nt, in0_block_w, kt, in0_tile, in1_tile) <= _l1_budget:
            return ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(core_grid.x, core_grid.y),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=next(d for d in (8, 6, 4, 3, 2, 1) if nt % d == 0),
                per_core_M=mt,
                per_core_N=nt,
            )
    return None


def _batched_matmul(a, b, activation_dtype, dense_core_grid):
    """a [1, B, M, K] x b [1, B, K, N] -> [1, B, M, N] (bf8) with the L1-fitted one-block-per-expert config, or ttnn's
    automatic config on the given grid when that block does not fit L1 (TP=1/2 shapes)."""
    program_config = _bmm_config(
        dense_core_grid, a.shape[2] // 32, a.shape[3] // 32, b.shape[3] // 32, a.dtype, b.dtype
    )
    grid_kwargs = {"program_config": program_config} if program_config is not None else {"core_grid": dense_core_grid}
    return ttnn.matmul(
        a,
        b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=activation_dtype,
        compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
        **grid_kwargs,
    )


def warmup_prefill_programs(weights, config, program_config, mesh_config, mesh_device, seq_lens):
    """Compile every program of the single-row dense prefill path whose SHAPE depends on the prompt's data, so that
    none is compiled after a trace has been captured (a program compiled next to a live trace can be overwritten by
    its replays, tenstorrent/tt-metal#55588: garbage or a hang on the first prompt that needs it). The expert-sorted
    path plans per split from the routed-token counts, so `cap` (topk k and the gathered-row counts), the number of
    hot experts (the batched group's shapes) and WHICH experts are hot (one ttnn.slice program per expert id) all
    vary with the prompt; the per-expert loop it falls back to depends on the shapes only but is rarely taken. Each
    variant runs once here with synthetic inputs, for every down-split length the model uses (identical kernels
    dedup, so this is mostly device time: ~1-2 s per split length once the kernels are built). The <= 256-token
    batched path depends on the padded length only and is compiled by the ordinary warm-up of each length."""
    mode = mesh_config.get_config(Mode.PREFILL)
    if not (mode.ep == 1 and mode.sp == 1) or config.num_experts < _SORTED_MOE_MIN_EXPERTS:
        return
    E, H = config.num_experts, config.hidden_size
    grid = _dense_core_grid(mesh_device)
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    def upload(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)

    for t in _slice_experts(weights.gate_up_proj, range(E)) + _slice_experts(weights.down_proj, range(E)):
        t.deallocate(True)  # the hot group gathers weights by expert id: one slice program per id
    for split_len in sorted({program_config.get_down_split_size(s) for s in seq_lens}):
        top = torch.rand(split_len, E).topk(config.num_experts_per_tok, dim=-1)
        routing = torch.zeros(split_len, E).scatter(1, top.indices, torch.softmax(top.values, dim=-1))
        routing_tokens_all = upload(routing.reshape(1, 1, split_len, E))
        scratch = {}
        variants = [(cap, 0) for cap in _SORTED_CAPS if cap <= split_len]
        variants += [(_SORTED_CAPS[1], n_hot) for n_hot in range(1, _SORTED_MOE_MAX_HOT + 1)]
        for cap, n_hot in variants:
            routing_t = ttnn.transpose(routing_tokens_all, 2, 3)  # [1, 1, E, split], a copy (the forward frees it)
            cold = torch.ones(1, 1, E, 1)
            cold[0, 0, :n_hot] = 0.0
            plan = (routing_t, cap, list(range(n_hot)), upload(cold))
            out = _sorted_moe_forward(
                upload(torch.randn(1, 1, split_len, H)),
                plan,
                routing_tokens_all,
                0,
                split_len,
                weights,
                config,
                ttnn.bfloat8_b,
                grid,
                scratch,
            )
            out.deallocate(True)
        per_expert = _slice_experts(weights.gate_up_proj, range(E))
        gate_up = _dense_gate_up_loop(
            upload(torch.randn(1, 1, split_len, H)), per_expert, weights, ttnn.bfloat8_b, grid
        )
        routing_split = upload(routing.T.reshape(1, E, split_len, 1))
        out = _dense_tail(
            gate_up,
            routing_split,
            routing_tokens_all,
            0,
            split_len,
            weights,
            config,
            ttnn.bfloat8_b,
            grid,
            weights.intermediate_padded_per_device,
        )
        out.deallocate(True)
        routing_split.deallocate(True)
        for t in per_expert:
            t.deallocate(True)
        for t in scratch.values():
            t.deallocate(True)
        routing_tokens_all.deallocate(True)
        logger.info(f"pre-compiled the expert-sorted prefill variants for {split_len}-token splits")


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
    scratch = {}  # expert-sorted path tables (one-hot identities, row-major bias) shared by this call's splits
    for hidden_chunk, routing_chunk in zip(hidden_states_chunks, routing_weights_chunks):
        next_states = _process_prefill_chunk(
            hidden_chunk,
            routing_chunk,
            weights,
            config,
            prefill_sparsity,
            program_config,
            ep,
            sp,
            tp,
            dense_core_grid=_dense_core_grid(mesh_device),
            scratch=scratch,
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
    for table in scratch.values():  # created by _identity_rows / _bias_table
        table.deallocate(True)
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
