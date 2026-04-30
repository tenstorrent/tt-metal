# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill forward pass using DeepSeek prefill dispatch/combine ops.

Uses GROUP-BASED dispatch table (standard ExpertMapping) with permuted
ThroughputExpertWeights to match the dispatch's expert-to-device mapping.
"""

from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup

from .config import ThroughputExpertConfig
from .decode import _apply_swiglu


def _compute_weight_permutation(mesh_rows, mesh_cols, experts_per_chip):
    """Compute permutation to reorder experts from LINEAR to GROUP-BASED ordering.

    ShardTensorToMesh distributes experts in linear device order (device 0, 1, 2, ...).
    The DeepSeek dispatch table uses group-based ordering (column-major: group=col, chip=row).
    This permutation, applied to the expert dim of the state_dict before loading,
    ensures each device gets the experts that the dispatch table routes to it.

    Returns a list of global expert indices in the order ShardTensorToMesh will distribute them.
    """
    perm = []
    for row in range(mesh_rows):
        for col in range(mesh_cols):
            for i in range(experts_per_chip):
                group_expert = ExpertMapping.get_global_expert_idx(
                    group=col,
                    chip=row,
                    local_expert=i,
                    experts_per_chip=experts_per_chip,
                    dispatch_group_size=mesh_rows,
                    num_dispatch_groups=mesh_cols,
                )
                perm.append(group_expert)
    return perm


class DeepSeekPrefillConfig:
    """Pre-initialized modules for DeepSeek prefill MoE path."""

    def __init__(
        self,
        mesh_device,
        config: ThroughputExpertConfig,
        dispatch_group_size: int = 4,
        num_dispatch_groups: int = 8,
        capacity_factor: float = 2.0,
        seq_len_per_chip: int = 128,
        num_links: int = 4,
        topology=None,
    ):
        if topology is None:
            topology = ttnn.Topology.Linear

        self.mesh_device = mesh_device
        self.config = config
        self.dispatch_group_size = dispatch_group_size
        self.num_dispatch_groups = num_dispatch_groups

        experts_per_chip = config.num_experts // (dispatch_group_size * num_dispatch_groups)
        self.experts_per_chip = experts_per_chip
        self.seq_len_per_chip = seq_len_per_chip
        self.capacity_factor = capacity_factor

        _, metadata_len, max_dispatched = compute_constants(
            seq_len_per_chip=seq_len_per_chip,
            num_routed_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            num_devices=dispatch_group_size * num_dispatch_groups,
            dispatch_group_size=dispatch_group_size,
            capacity_factor=capacity_factor,
        )
        self.metadata_len = metadata_len
        self.max_dispatched_tokens_per_expert = max_dispatched

        # Standard GROUP-BASED dispatch table
        expert_dispatch_table = ExpertMapping.create_dispatch_table(
            config.num_experts, dispatch_group_size, num_dispatch_groups
        )
        self.expert_dispatch_table = expert_dispatch_table
        self.tt_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(
            mesh_device, expert_dispatch_table, dispatch_axis=0
        )

        self.dispatch_module = TtDispatchModule(
            mesh_device=mesh_device,
            dispatch_group_size=dispatch_group_size,
            experts_per_chip=experts_per_chip,
            num_routed_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_dispatched,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=config.hidden_size,
            cluster_axis=0,
            num_links=num_links,
            topology=topology,
        )

        self.combine_module = TtCombineModule(
            mesh_device=mesh_device,
            dispatch_group_size=dispatch_group_size,
            num_dispatch_groups=num_dispatch_groups,
            experts_per_chip=experts_per_chip,
            num_experts_per_tok=config.num_experts_per_tok,
            seq_len_per_chip=seq_len_per_chip,
            cluster_axis=0,
            num_links=num_links,
            topology=topology,
            init_zeros=True,
        )

        self.routing_setup = TtMoERoutingSetup(mesh_device, expert_dispatch_table, num_links=num_links)

        self.permuted_weights = None  # Set by mlp.py after host-side permutation

        logger.info(
            f"DeepSeekPrefillConfig: experts_per_chip={experts_per_chip}, "
            f"max_dispatched={max_dispatched}, dgs={dispatch_group_size}, ndg={num_dispatch_groups}"
        )


# Keep for backward compatibility
def _prepare_expert_weights_for_deepseek(state_dict, config):
    E = config.num_experts
    if "gate_up_proj" in state_dict:
        gate_up = state_dict["gate_up_proj"]
        down = state_dict["down_proj"]
        gate_all = gate_up[..., ::2]
        up_all = gate_up[..., 1::2]
    elif "gate_proj" in state_dict:
        gate_all = state_dict["gate_proj"]
        up_all = state_dict["up_proj"]
        down = state_dict["down_proj"]
    else:
        raise ValueError("Expected gate_up_proj or gate_proj, got: %s" % list(state_dict.keys()))
    return [
        {
            "gate_proj": gate_all[e].T.contiguous(),
            "up_proj": up_all[e].T.contiguous(),
            "down_proj": down[e].T.contiguous(),
        }
        for e in range(E)
    ]


def forward_prefill_deepseek(
    hidden_states,
    topk_expert_indices,
    topk_expert_weights,
    config,
    prefill_config,
    mesh_device,
    mesh_config=None,
    ccl_manager=None,
    weights=None,
    program_config=None,
):
    """DeepSeek prefill MoE with seq-dim chunking.

    Wraps _forward_prefill_deepseek_chunk, splitting inputs along the seq
    dim when seq_per_device > prefill_config.seq_len_per_chip. This lets
    us serve arbitrary contexts (up to model max_context) while keeping
    each MoE call within the post_combine_reduce 72-core hardware limit.

    Input seq MUST be an integer multiple of prefill_config.seq_len_per_chip.
    hidden_states seq dim is axis 2 (shape [1, 1, S, H]);
    topk_expert_indices / topk_expert_weights seq dim is axis 0.
    """
    pc = prefill_config
    chunk = pc.seq_len_per_chip
    seq_per_device = hidden_states.shape[2] if len(hidden_states.shape) >= 3 else hidden_states.shape[0]

    if seq_per_device == chunk:
        return _forward_prefill_deepseek_chunk(
            hidden_states,
            topk_expert_indices,
            topk_expert_weights,
            config,
            prefill_config,
            mesh_device,
            mesh_config,
            ccl_manager,
            weights,
            program_config,
        )

    assert seq_per_device % chunk == 0, (
        f"DeepSeek prefill requires seq ({seq_per_device}) to be a multiple of " f"seq_len_per_chip ({chunk})."
    )
    n_chunks = seq_per_device // chunk
    logger.info(f"DeepSeek prefill: chunking seq={seq_per_device} into {n_chunks} x {chunk}")

    h_chunks = ttnn.split(hidden_states, chunk, dim=2)
    i_chunks = ttnn.split(topk_expert_indices, chunk, dim=0)
    w_chunks = ttnn.split(topk_expert_weights, chunk, dim=0)
    ttnn.deallocate(hidden_states)
    ttnn.deallocate(topk_expert_indices)
    ttnn.deallocate(topk_expert_weights)

    outs = []
    for h, i, w in zip(h_chunks, i_chunks, w_chunks):
        out = _forward_prefill_deepseek_chunk(
            h,
            i,
            w,
            config,
            prefill_config,
            mesh_device,
            mesh_config,
            ccl_manager,
            weights,
            program_config,
        )
        ttnn.deallocate(h)
        ttnn.deallocate(i)
        ttnn.deallocate(w)
        # Compress per-chunk output to BFP8 before accumulating. Halves the
        # peak working set during concat (critical for seq >= 128K contexts).
        out_compact = ttnn.typecast(out, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if out_compact is not out:
            ttnn.deallocate(out)
        outs.append(out_compact)
    acc = ttnn.concat(outs, dim=2)
    for o in outs:
        ttnn.deallocate(o)
    return acc


def _forward_prefill_deepseek_chunk(
    hidden_states,
    topk_expert_indices,
    topk_expert_weights,
    config,
    prefill_config,
    mesh_device,
    mesh_config=None,
    ccl_manager=None,
    weights=None,
    program_config=None,
):
    """One-shot DeepSeek prefill — input seq must equal prefill_config.seq_len_per_chip."""
    pc = prefill_config
    # Use pre-permuted weights (GROUP-BASED ordering matching dispatch table)
    pw = getattr(pc, "permuted_weights", None) or weights

    # Step 1: All-gather x across TP axis
    x_full_is_new = False
    if mesh_device.shape[1] > 1 and hidden_states.shape[-1] < config.hidden_size:
        x_full = ttnn.all_gather(hidden_states, dim=-1, cluster_axis=1, num_links=4, topology=ttnn.Topology.Linear)
        x_full_is_new = True
    else:
        x_full = hidden_states

    if x_full.layout != ttnn.ROW_MAJOR_LAYOUT:
        x_full_prev = x_full
        x_full = ttnn.to_layout(x_full, ttnn.ROW_MAJOR_LAYOUT)
        if x_full_is_new and x_full is not x_full_prev:
            ttnn.deallocate(x_full_prev)
        x_full_is_new = True

    # Step 2: Routing setup
    if topk_expert_indices.dtype == ttnn.uint16:
        idx_for_routing = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
    else:
        idx_tile = ttnn.to_layout(topk_expert_indices, ttnn.TILE_LAYOUT)
        idx_u16 = ttnn.typecast(idx_tile, ttnn.uint16)
        ttnn.deallocate(idx_tile)
        idx_for_routing = ttnn.to_layout(idx_u16, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(idx_u16)

    tt_offsets, tt_counts, _ = pc.routing_setup(
        ttnn_top_k_experts_indices=idx_for_routing,
        num_routed_experts=config.num_experts,
        seq_len_per_chip=pc.seq_len_per_chip,
        num_experts_per_tok=config.num_experts_per_tok,
    )
    ttnn.deallocate(idx_for_routing)

    # Step 3: Format indices/scores for dispatch
    if topk_expert_indices.dtype != ttnn.int32:
        indices_tile = ttnn.to_layout(topk_expert_indices, ttnn.TILE_LAYOUT)
        indices_i32 = ttnn.typecast(indices_tile, ttnn.int32)
        ttnn.deallocate(indices_tile)
        indices_rm = ttnn.to_layout(indices_i32, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(indices_i32)
    else:
        indices_rm = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
    scores_rm = ttnn.to_layout(topk_expert_weights, ttnn.ROW_MAJOR_LAYOUT)
    scores_for_reduce = ttnn.clone(scores_rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Step 4: Reshape to 3D for dispatch (reshape is a view; dispatch consumes)
    x_3d = ttnn.reshape(x_full, (1, pc.seq_len_per_chip, config.hidden_size))
    w_3d = ttnn.reshape(scores_rm, (1, pc.seq_len_per_chip, config.num_experts_per_tok))
    i_3d = ttnn.reshape(indices_rm, (1, pc.seq_len_per_chip, config.num_experts_per_tok))

    # Step 5: Dispatch
    dispatched, metadata = pc.dispatch_module(x_3d, w_3d, i_3d, tt_offsets, pc.tt_dispatch_table)
    # Dispatch done — x/scores/indices no longer needed
    if x_full_is_new:
        ttnn.deallocate(x_full)
    ttnn.deallocate(scores_rm)
    ttnn.deallocate(indices_rm)
    ttnn.deallocate(tt_offsets)

    # Step 6: Expert compute using GROUP-permuted ThroughputExpertWeights
    buf = ttnn.squeeze(dispatched, dim=0)
    if len(buf.shape) == 3:
        buf = ttnn.unsqueeze(buf, dim=0)
    # NaN cleanup skipped — unfilled dispatch slots dont affect valid token outputs
    buf_tiled = ttnn.to_layout(buf, ttnn.TILE_LAYOUT)
    ttnn.deallocate(dispatched)

    memory_config = ttnn.DRAM_MEMORY_CONFIG

    if pw.w1_w3_fused is not None:
        w1_w3_out = ttnn.matmul(buf_tiled, pw.w1_w3_fused, memory_config=memory_config)
        ttnn.deallocate(buf_tiled)
        ttnn.add(w1_w3_out, pw.w1_w3_bias_fused, output_tensor=w1_w3_out)
        shape = w1_w3_out.shape
        w1_out = ttnn.slice(
            w1_w3_out, [0, 0, 0, 0], [shape[0], shape[1], shape[2], config.intermediate_size], [1, 1, 1, 1]
        )
        w3_out = ttnn.slice(
            w1_w3_out,
            [0, 0, 0, config.intermediate_size],
            [shape[0], shape[1], shape[2], 2 * config.intermediate_size],
            [1, 1, 1, 1],
        )
        ttnn.deallocate(w1_w3_out)
    else:
        w1_out = ttnn.matmul(buf_tiled, pw.w1, memory_config=memory_config)
        ttnn.add(w1_out, pw.w1_bias, output_tensor=w1_out)
        w3_out = ttnn.matmul(buf_tiled, pw.w3, memory_config=memory_config)
        ttnn.deallocate(buf_tiled)
        ttnn.add(w3_out, pw.w3_bias, output_tensor=w3_out)

    activated = _apply_swiglu(w1_out, w3_out, config.alpha, config.swiglu_limit, memory_config)

    expert_output = ttnn.matmul(activated, pw.w2, memory_config=memory_config)
    ttnn.deallocate(activated)
    ttnn.add(expert_output, pw.w2_bias, output_tensor=expert_output)

    expert_out_rm = ttnn.to_layout(expert_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(expert_output)
    expert_out_rm = ttnn.unsqueeze(expert_out_rm, dim=0)

    # Step 7: Combine
    combined = pc.combine_module(expert_out_rm, metadata, tt_counts)
    ttnn.deallocate(expert_out_rm)
    ttnn.deallocate(metadata)
    ttnn.deallocate(tt_counts)

    # Step 8: Fused post-combine reduce (w_reduce/w_5d may be views of scores_for_reduce)
    w_reduce = ttnn.reshape(scores_for_reduce, (1, 1, pc.seq_len_per_chip, config.num_experts_per_tok))
    w_5d = ttnn.unsqueeze(w_reduce, dim=-1)
    w_5d_is_new = False
    if w_5d.layout != ttnn.ROW_MAJOR_LAYOUT:
        w_5d_prev = w_5d
        w_5d = ttnn.to_layout(w_5d, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w_5d_is_new = w_5d is not w_5d_prev
    try:
        summed = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            combined, w_5d, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    except Exception as _e:
        from loguru import logger

        logger.warning(f"FUSED FALLBACK: {_e}")
        K = config.num_experts_per_tok
        seq = pc.seq_len_per_chip
        D = config.hidden_size
        acc = None
        for k in range(K):
            expert_k = ttnn.slice(combined, (0, 0, 0, k, 0), (1, 1, seq, k + 1, D))
            weight_k = ttnn.slice(w_5d, (0, 0, 0, k, 0), (1, 1, seq, k + 1, 1))
            weighted = ttnn.mul(expert_k, weight_k)
            if acc is None:
                acc = weighted
            else:
                acc = ttnn.add(acc, weighted)
        summed = ttnn.reshape(acc, (1, 1, seq, D))
        summed = ttnn.to_layout(summed, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # post_combine_reduce done — free combined, weight scratch
    ttnn.deallocate(combined)
    if w_5d_is_new:
        ttnn.deallocate(w_5d)
    ttnn.deallocate(scores_for_reduce)

    if mesh_device.shape[1] > 1:
        output = ttnn.all_reduce(summed, cluster_axis=1, num_links=4, topology=ttnn.Topology.Linear)
        if output is not summed:
            ttnn.deallocate(summed)
    else:
        output = summed

    while len(output.shape) > 4:
        output = ttnn.squeeze(output, dim=0)
    while len(output.shape) < 4:
        output = ttnn.unsqueeze(output, dim=0)

    return output
