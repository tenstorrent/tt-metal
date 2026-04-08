# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill forward pass using DeepSeek prefill dispatch/combine ops.

Replaces the chunked-decode prefill path with native prefill CCL ops for
better performance on long sequences. Device-side routing setup (no host round-trip).
"""

from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

from .config import ThroughputExpertConfig


class DeepSeekPrefillConfig:
    """Pre-initialized modules for DeepSeek prefill MoE path.

    Created once at model init, stores dispatch/combine/expert modules and
    static tensors (dispatch table). Reused across layers and calls.
    """

    def __init__(
        self,
        mesh_device,
        config: ThroughputExpertConfig,
        routed_expert_weights: list,
        dispatch_group_size: int = 4,
        num_dispatch_groups: int = 8,
        capacity_factor: float = 2.0,
        seq_len_per_chip: int = 128,
        num_links: int = 1,
        topology=None,
        activations_dtype=None,
        weights_dtype=None,
    ):
        if topology is None:
            topology = ttnn.Topology.Linear
        if activations_dtype is None:
            activations_dtype = ttnn.bfloat8_b
        if weights_dtype is None:
            weights_dtype = ttnn.bfloat4_b

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

        # Static dispatch table
        expert_dispatch_table = ExpertMapping.create_dispatch_table(
            config.num_experts, dispatch_group_size, num_dispatch_groups
        )
        self.expert_dispatch_table = expert_dispatch_table
        self.tt_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(
            mesh_device, expert_dispatch_table, dispatch_axis=0
        )

        # Dispatch module
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

        # Combine module
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

        # Routed expert (sequential per-expert FFN)
        self.routed_expert = TtRoutedExpert(
            mesh_device=mesh_device,
            experts_per_chip=experts_per_chip,
            emb_dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            max_tokens=max_dispatched,
            torch_weights=routed_expert_weights,
            activations_dtype=activations_dtype,
            weights_dtype=weights_dtype,
        )

        # Reduce module (weighted sum + reduce-scatter across TP axis)
        self.reduce_module = TtReduceModule(
            mesh_device=mesh_device,
            topk_dim=3,
            cluster_axis=1,
            num_links=num_links,
            topology=topology,
        )

        # Device-side routing setup (no host round-trip for offsets/counts)
        self.routing_setup = TtMoERoutingSetup(mesh_device, expert_dispatch_table, num_links=num_links)

        logger.info(
            f"DeepSeekPrefillConfig: experts_per_chip={experts_per_chip}, "
            f"max_dispatched={max_dispatched}, dgs={dispatch_group_size}, ndg={num_dispatch_groups}"
        )


def _prepare_expert_weights_for_deepseek(
    state_dict: dict,
    config: ThroughputExpertConfig,
) -> list:
    """Convert GPT-OSS expert weights to per-expert dicts for TtRoutedExpert.

    GPT-OSS stores: gate_up_proj [E, H, 2*I] (interleaved), down_proj [E, I, H]
    TtRoutedExpert expects HF format per-expert:
        gate_proj: [I, H]  (out_features, in_features)
        up_proj:   [I, H]
        down_proj: [H, I]

    Returns:
        List of dicts, one per expert, with gate_proj, up_proj, down_proj tensors.
    """
    E = config.num_experts
    I = config.intermediate_size
    H = config.hidden_size

    if "gate_up_proj" in state_dict:
        gate_up = state_dict["gate_up_proj"]  # [E, H, 2*I] interleaved
        down = state_dict["down_proj"]  # [E, I, H]
        # Unfuse interleaved: even columns = gate, odd = up
        gate_all = gate_up[..., ::2]  # [E, H, I]
        up_all = gate_up[..., 1::2]  # [E, H, I]
    elif "gate_proj" in state_dict:
        gate_all = state_dict["gate_proj"]  # [E, H, I] or [E, I, H]
        up_all = state_dict["up_proj"]
        down = state_dict["down_proj"]
    else:
        raise ValueError("Expected gate_up_proj or gate_proj in state_dict, got: %s" % list(state_dict.keys()))

    weights_list = []
    for e in range(E):
        # Transpose to HF format (out_features, in_features) for TtRoutedExpert
        weights_list.append(
            {
                "gate_proj": gate_all[e].T.contiguous(),  # [H, I] -> [I, H]
                "up_proj": up_all[e].T.contiguous(),  # [H, I] -> [I, H]
                "down_proj": down[e].T.contiguous(),  # [I, H] -> [H, I]
            }
        )
    return weights_list


def forward_prefill_deepseek(
    hidden_states: ttnn.Tensor,
    topk_expert_indices: ttnn.Tensor,
    topk_expert_weights: ttnn.Tensor,
    config: ThroughputExpertConfig,
    prefill_config: DeepSeekPrefillConfig,
    mesh_device,
    mesh_config=None,
    ccl_manager=None,
) -> ttnn.Tensor:
    """Prefill forward using DeepSeek dispatch/combine.

    Args:
        hidden_states: [seq_per_device, 1, 1, hidden_size] (TP-sharded on last dim)
        topk_expert_indices: [seq_per_device, 1, 1, K] (uint16 from TopKRouter)
        topk_expert_weights: [seq_per_device, 1, 1, K] (bfloat16)
        config: ThroughputExpertConfig
        prefill_config: DeepSeekPrefillConfig with pre-initialized modules
        mesh_device: TTNN mesh device

    Returns:
        [seq_per_device, 1, 1, hidden_size] (TP-sharded)
    """
    pc = prefill_config

    # Step 1: All-gather x across TP axis to get full hidden_dim (skip if already full)
    if mesh_device.shape[1] > 1 and hidden_states.shape[-1] < config.hidden_size:
        x_full = ttnn.all_gather(hidden_states, dim=-1, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear)
    else:
        x_full = hidden_states

    # Ensure ROW_MAJOR for dispatch
    if x_full.layout != ttnn.ROW_MAJOR_LAYOUT:
        x_full = ttnn.to_layout(x_full, ttnn.ROW_MAJOR_LAYOUT)

    # Step 2: Device-side routing setup (offsets + counts on device, no host round-trip)
    if topk_expert_indices.dtype == ttnn.uint16:
        idx_for_routing = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
    else:
        idx_tile = ttnn.to_layout(topk_expert_indices, ttnn.TILE_LAYOUT)
        idx_u16 = ttnn.typecast(idx_tile, ttnn.uint16)
        idx_for_routing = ttnn.to_layout(idx_u16, ttnn.ROW_MAJOR_LAYOUT)

    tt_offsets, tt_counts, _ = pc.routing_setup(
        ttnn_top_k_experts_indices=idx_for_routing,
        num_routed_experts=config.num_experts,
        seq_len_per_chip=pc.seq_len_per_chip,
        num_experts_per_tok=config.num_experts_per_tok,
    )

    # Step 3: Format indices/scores for dispatch (int32, ROW_MAJOR)
    if topk_expert_indices.dtype != ttnn.int32:
        indices_tile = ttnn.to_layout(topk_expert_indices, ttnn.TILE_LAYOUT)
        indices_i32 = ttnn.typecast(indices_tile, ttnn.int32)
        indices_rm = ttnn.to_layout(indices_i32, ttnn.ROW_MAJOR_LAYOUT)
    else:
        indices_rm = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
    scores_rm = ttnn.to_layout(topk_expert_weights, ttnn.ROW_MAJOR_LAYOUT)

    # Save scores for weighted sum (dispatch may consume the input tensors)
    scores_for_reduce = ttnn.clone(scores_rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Step 4: Reshape to 3D for dispatch [dispatch_group_size_per_device, seq, dim]
    x_3d = ttnn.reshape(x_full, (1, pc.seq_len_per_chip, config.hidden_size))
    w_3d = ttnn.reshape(scores_rm, (1, pc.seq_len_per_chip, config.num_experts_per_tok))
    i_3d = ttnn.reshape(indices_rm, (1, pc.seq_len_per_chip, config.num_experts_per_tok))

    # Step 5: Dispatch
    dispatched, metadata = pc.dispatch_module(x_3d, w_3d, i_3d, tt_offsets, pc.tt_dispatch_table)

    # Step 6: Expert compute
    buf = ttnn.squeeze(ttnn.squeeze(dispatched, dim=0), dim=0)
    buf_tiled = ttnn.to_layout(buf, ttnn.TILE_LAYOUT)
    expert_out = pc.routed_expert(buf_tiled, tt_counts)
    expert_out = ttnn.unsqueeze(ttnn.unsqueeze(expert_out, dim=0), dim=0)
    expert_out_rm = ttnn.to_layout(expert_out, ttnn.ROW_MAJOR_LAYOUT)

    # Step 7: Combine
    combined = pc.combine_module(expert_out_rm, metadata, tt_counts)

    # Step 8: Weighted sum + all_reduce
    # GPT-OSS MLP returns full hidden_size (all_reduce, not reduce_scatter).
    combined_tiled = ttnn.to_layout(combined, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w_reduce = ttnn.reshape(scores_for_reduce, (1, 1, pc.seq_len_per_chip, config.num_experts_per_tok))

    # Local weighted sum over topk dim
    w_exp = ttnn.unsqueeze(w_reduce, dim=-1)
    if w_exp.layout != ttnn.TILE_LAYOUT:
        w_exp = ttnn.to_layout(w_exp, ttnn.TILE_LAYOUT)
    weighted = ttnn.mul(combined_tiled, w_exp)
    summed = ttnn.sum(weighted, dim=3)

    # All-reduce across TP axis (full hidden_size on every device)
    if mesh_device.shape[1] > 1:
        output = ttnn.all_reduce(summed, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear)
    else:
        output = summed

    # Match output rank to 4D
    while len(output.shape) > 4:
        output = ttnn.squeeze(output, dim=0)
    while len(output.shape) < 4:
        output = ttnn.unsqueeze(output, dim=0)

    return output
