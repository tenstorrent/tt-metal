# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill forward pass for experts (seq_len>1)."""

import os

import torch
from loguru import logger

import ttnn
from models.demos.gpt_oss.config import Mode

from .config import ExpertConfig, ProgramConfig
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
    for hidden_split, routing_split in zip(hidden_list, routing_list):
        split_len = hidden_split.shape[2]
        group_size = split_len // TILE_SIZE

        if dense_moe:
            hidden_4D = ttnn.unsqueeze_to_4D(hidden_split)  # [1, 1, split, H] (view of the split)
            bmm_config = _dense_bmm_config(dense_core_grid, split_len, weights)
            plan = None
            if bmm_config is None:
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
                )
            else:
                gate_up = _dense_gate_up(hidden_4D, bmm_config, weights, config, activation_dtype, dense_core_grid)
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
    if group_mask is not None:
        group_mask.deallocate(True)
    if routing_tokens_all is not None:
        routing_tokens_all.deallocate(True)

    return next_states_reduced_acc


def _dense_gate_up(hidden_4D, bmm_config, weights, config, activation_dtype, dense_core_grid):
    """Fused gate/up projection for a whole split, [1, 1, split, H] -> [1, E, split, 2Ip] with the bias added.
    Consumes hidden_4D. Short splits (bmm_config given): replicate the activations per expert and run ONE batched
    matmul (one expert per core; 128 separate launches cost ~30 us each on device, which dominates 128-token
    prefills). Otherwise one ttnn.linear (fused bias) per expert over the whole split, concatenated."""
    if bmm_config is not None:
        hidden_rep = ttnn.repeat(hidden_4D, ttnn.Shape((1, config.num_experts, 1, 1)))
        hidden_4D.deallocate(True)
        gate_up = ttnn.matmul(
            hidden_rep,
            weights.gate_up_proj,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=activation_dtype,
            program_config=bmm_config,
            compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
        )
        hidden_rep.deallocate(True)
        return ttnn.add(gate_up, weights.gate_up_proj_bias_t, output_tensor=gate_up)
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
        for w_e, b_e in zip(weights.gate_up_proj_per_expert, weights.gate_up_proj_bias_per_expert)
    ]
    hidden_4D.deallocate(True)
    gate_up = ttnn.concat(per_expert, dim=1)
    for t in per_expert:
        t.deallocate(True)
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
    The routing weights are applied to the down INPUT (a quarter of the down output; exact since down is linear) and
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
        weights.down_proj_padded,
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


_SORTED_MOE_DEBUG = os.getenv("GPT_OSS_SORTED_MOE_DEBUG", "0") == "1"
# Last plan chosen by _sorted_moe_plan ({"split", "cap", "hot"}); read by tests to assert which path ran.
LAST_SORTED_MOE_PLAN = {}
_SORTED_MOE_MAX_HOT = 16  # more hot experts than this -> dense per-expert loop for the split
# Cost model (ms per 1024-token split, P150, 120B shapes) used to pick the hot/cold threshold on the host:
_SORTED_FIXED_MS, _SORTED_PER_KROW_MS, _HOT_FIXED_MS, _HOT_PER_EXPERT_MS = 2.5, 0.27, 1.0, 0.25
_DENSE_PER_EXPERT_MS = (
    0.125  # per-expert cost of the dense loop over a 1024-token split (gate/up + concat + down share)
)
# Measured on P150x8: the sorted path halves gpt-oss-120b (E=128) prefill at ISL >= 1024 but is slower than the dense
# loop for gpt-oss-20b (E=32: ~128 routed tokens per expert per 1024, so the gathered rows are not much fewer and the
# fixed cost + host round-trip dominate).
_SORTED_MOE_MIN_EXPERTS = 64


def _sorted_moe_plan(routing_tokens_all, token_offset, split_len, config):
    """Host-side plan for one split from the per-expert routed-token counts (one small device->host read).

    Real GPT-OSS routing is very skewed (the hottest expert of a 1024-token split often takes 30-90% of the
    tokens), so the experts are partitioned: HOT experts (count > cap) run dense over the whole split as a small
    batched group, COLD experts run expert-sorted with `cap` gathered rows each. `cap` is chosen from a small cost
    model over the count distribution. Returns (routing^T [1, 1, E, split], cap, hot_ids) or None (use the dense
    per-expert loop when too many experts are hot)."""
    E = config.num_experts
    if E < _SORTED_MOE_MIN_EXPERTS:
        return None
    # This does a device->host read of the per-expert counts, so it must never run under trace capture (a captured
    # plan would be replayed for other prompts). It cannot: the sorted path is only taken for splits longer than
    # _DENSE_BMM_MAX_TOKENS (256) and the only traced prefill length is 128 tokens.
    assert split_len > _DENSE_BMM_MAX_TOKENS, "the sorted MoE path is for eager (untraced) long splits only"
    routing_tokens = ttnn.slice(routing_tokens_all, [0, 0, token_offset, 0], [1, 1, token_offset + split_len, E])
    routing_t = ttnn.transpose(routing_tokens, 2, 3)  # [1, 1, E, split]
    if split_len != routing_tokens_all.shape[2]:  # a full-range slice aliases its input
        routing_tokens.deallocate(True)
    active = ttnn.gt(routing_t, 0.0)
    counts = ttnn.sum(active, dim=3, keepdim=True)  # [1, 1, E, 1]
    active.deallocate(True)
    # routing weights are replicated across the TP devices, so one device's counts suffice (mesh tensors need a
    # composer for a direct to_torch)
    counts_host = ttnn.to_torch(ttnn.get_device_tensors(counts)[0]).reshape(-1).to(torch.int64)
    best = None
    for cap in (32, 64, 96, 128, 160, 192, 256):
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
    cold_mask_t = ttnn.le(counts, float(cap))
    counts.deallocate(True)
    return routing_t, cap, hot_ids, cold_mask_t


def _sorted_moe_forward(
    hidden_4D, plan, routing_tokens_all, token_offset, split_len, weights, config, activation_dtype, dense_core_grid
):
    """Hot/cold expert-sorted MoE for one split ([1, 1, split, H] -> [1, 1, split, H]); consumes hidden_4D.

    Cold experts: topk over the transposed routing weights gives each expert its `cap` largest-weight tokens (all
    its routed tokens, then zero-weight fillers); ttnn.embedding gathers those rows (and one-hot rows from a cached
    identity), gate/up and down run as batched matmuls over the gathered [E, cap, *] rows only, each row is scaled by
    its slot weight (zeroed for hot experts) and scattered back with one-hot^T @ rows. Hot experts (their routed-token
    count exceeds `cap`): the activations are replicated per hot expert and gate/up / down run as small batched
    matmuls over the whole split, weighted by their routing weights and reduced. The math equals the dense path."""
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
    onehot = ttnn.reshape(
        ttnn.embedding(idx_flat, _eye(weights, split_len), layout=ttnn.TILE_LAYOUT), (1, 1, E * cap, split_len)
    )
    idx_flat.deallocate(True)
    gate_up = ttnn.matmul(
        rows,
        weights.gate_up_proj,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=activation_dtype,
        program_config=_bmm_config(dense_core_grid, cap // 32, H // 32, (2 * ip) // 32),
        compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
    )
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
        weights.down_proj_padded,
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
        # ttnn.concat of a single tensor returns that tensor, so with one hot expert the cached per-expert weight and
        # bias are used directly and must not be deallocated below.
        w_hot = ttnn.concat([weights.gate_up_proj_per_expert[e] for e in hot_ids], dim=1)  # [1, n_hot, H, 2Ip]
        b_hot = ttnn.concat([weights.gate_up_proj_bias_per_expert[e] for e in hot_ids], dim=1)  # [1, n_hot, 1, 2Ip]
        owns_hot_weights = n_hot > 1
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
        if owns_hot_weights:
            w_hot.deallocate(True)
        gu_hot = ttnn.add(gu_hot, b_hot, output_tensor=gu_hot)
        if owns_hot_weights:
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
        wd_hot = ttnn.concat([weights.down_proj_per_expert[e] for e in hot_ids], dim=1)  # [1, n_hot, Ip, H]
        down_h = ttnn.matmul(
            act_h,
            wd_hot,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=activation_dtype,
            core_grid=dense_core_grid,
            compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
        )
        act_h.deallocate(True)
        if owns_hot_weights:
            wd_hot.deallocate(True)
        hot_out = reduce_experts(down_h)  # [1, 1, split, H]
        down_h.deallocate(True)
        out = ttnn.add(out, hot_out, output_tensor=out)
        hot_out.deallocate(True)
        hot_idx_t.deallocate(True)
        routing_t.deallocate(True)
    table.deallocate(True)  # releases the split (view) or the bf16 copy
    return _add_folded_down_bias(out, routing_tokens_all, token_offset, split_len, weights, config)


def _eye(weights, n, _unused=None):
    """Cached [n, n] bf16 identity on device (one-hot table for the sorted path's scatter matmul)."""
    tables = weights.eye_tables
    if tables is None:
        tables = {}
        object.__setattr__(weights, "eye_tables", tables)
    if n not in tables:
        import torch

        tables[n] = ttnn.from_torch(
            torch.eye(n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=weights.gate_up_proj.device()
        )
    return tables[n]


def _bmm_config(core_grid, mt, kt, nt):
    """MatmulMultiCoreReuseProgramConfig with one [mt x nt]-tile output block per core (batched matmul, one batch
    entry per core)."""
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(core_grid.x, core_grid.y),
        in0_block_w=next(d for d in (6, 5, 4, 3, 2, 1) if kt % d == 0),
        out_subblock_h=1,
        out_subblock_w=next(d for d in (8, 6, 4, 3, 2, 1) if nt % d == 0),
        per_core_M=mt,
        per_core_N=nt,
    )


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
    bias_t = weights.gate_up_proj_bias_t  # [E, 1, n]
    biases = [
        ttnn.typecast(ttnn.reshape(ttnn.slice(bias_t, [e, 0, 0], [e + 1, 1, n]), (1, 1, 1, n)), ttnn.bfloat16)
        for e in range(num_experts)
    ]
    object.__setattr__(weights, "gate_up_proj_bias_per_expert", biases)
    pad_k = weights.intermediate_padded_per_device - weights.intermediate_size_per_device
    down_padded = (
        ttnn.pad(weights.down_proj, padding=[(0, 0), (0, 0), (0, pad_k), (0, 0)], value=0.0)
        if pad_k > 0
        else weights.down_proj
    )
    object.__setattr__(weights, "down_proj_padded", down_padded)
    ip_pad, hidden_out = down_padded.shape[2], down_padded.shape[3]
    down_per_expert = [
        ttnn.slice(down_padded, [0, e, 0, 0], [1, e + 1, ip_pad, hidden_out]) for e in range(num_experts)
    ]
    object.__setattr__(weights, "down_proj_per_expert", down_per_expert)


_DENSE_BMM_MAX_TOKENS = 256  # per_core_M <= 8 keeps the per-core output block (M x 2Ip tiles) within L1


def _dense_bmm_config(core_grid, split_len, weights: ExpertWeights):
    """One-launch batched matmul config for short splits (one expert's whole [split, 2Ip] output per core), or None
    when the split is too long for the per-core block to fit in L1 (then the sorted / per-expert paths are used)."""
    if split_len > _DENSE_BMM_MAX_TOKENS:
        return None
    return _bmm_config(
        core_grid, split_len // 32, weights.gate_up_proj.shape[2] // 32, weights.gate_up_proj.shape[3] // 32
    )


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
