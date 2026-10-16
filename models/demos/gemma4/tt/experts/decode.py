# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
On-device expert decode forward using sparse_matmul.

Follows gpt-oss experts/decode.py pattern with GeGLU instead of SwiGLU, no bias.
sparse_matmul output is 6D: [batch_dims..., num_experts, seq_tiles, n_dim].
"""

import math

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce

from .weights import ExpertWeights


def _build_sparse_matmul_config(m, n, in0_block_w=1):
    """Build program config for sparse_matmul following gpt-oss pattern."""
    n_tiles = int(math.ceil(n / 32))

    # Find largest divisor of n_tiles fitting in 8×8 grid
    best_cores = 1
    best_cx, best_cy = 1, 1
    for num_cores in range(1, min(65, n_tiles + 1)):
        if n_tiles % num_cores != 0:
            continue
        for cy in range(1, 9):
            if num_cores % cy == 0:
                cx = num_cores // cy
                if cx <= 8 and num_cores > best_cores:
                    best_cores = num_cores
                    best_cx, best_cy = cx, cy
                    break

    per_core_N = n_tiles // best_cores

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(best_cx, best_cy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=per_core_N,
        per_core_M=max(32, m) // 32,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def decode_forward(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config,
    mesh_config=None,
    mesh_device=None,
    ccl_manager=None,
    local_expert_mask=None,
    combine=True,
):
    """On-device expert decode forward using sparse_matmul (the B users sit on dim-2).

    Whole-expert sharding: this chip holds ``weights.num_local_experts`` experts at full width, and
    the replicated dense routing is partitioned to its expert columns (ep axis, then tp axis), so
    every tensor here is E_loc wide and nnz is data-dependent (inferred). ``combine=False``
    returns the local partial sum (the caller completes the cross-lane / cross-axis reduction).

    Args:
        hidden_states: [1, 1, B, hidden_size] on device
        routing_weights: [1, 1, B, num_experts] on device (dense, from router)
    Returns:
        output: [1, 1, B, hidden_size] on device
    """
    from .prefill import apply_geglu_fused, partition_routing_to_local_experts

    batch_size = hidden_states.shape[2]
    num_local = weights.num_local_experts
    intermediate_size = weights.intermediate_size_per_device
    hidden_size = config.hidden_size
    local_routing = partition_routing_to_local_experts(routing_weights, weights, mesh_config)
    # Sparsity [1,1,1,E_loc]: an expert is computed when any user in the batch routes to it; each
    # user's own weight is applied by the routing multiply below.
    if batch_size > 1:
        sparsity_src = ttnn.max(local_routing, dim=2, keepdim=True)
    else:
        sparsity_src = local_routing
    sparsity = ttnn.to_layout(sparsity_src, ttnn.ROW_MAJOR_LAYOUT)
    if sparsity_src is not local_routing:
        sparsity_src.deallocate(True)
    output_tile = ttnn.Tile([32, 32])
    gate_up_config = _build_sparse_matmul_config(batch_size, 2 * intermediate_size)
    down_config = _build_sparse_matmul_config(batch_size, hidden_size)
    # [1,1,B,H] x [1,E_loc,H,2I] -> [1,1,1,E_loc,B_tile,2I]
    up_gate = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_up_proj,
        sparsity=sparsity,
        nnz=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_config,
        dtype=ttnn.bfloat16,
    )
    sm_n = up_gate.shape[-1]
    up_gate = ttnn.reshape(up_gate, (1, num_local, batch_size, sm_n))
    down_input = apply_geglu_fused(up_gate)
    up_gate.deallocate(True)
    down_input = ttnn.reshape(down_input, (1, num_local, batch_size, sm_n // 2))
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        nnz=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=down_config,
        is_input_a_sparse=True,
        dtype=ttnn.bfloat16,
    )
    down_input.deallocate(True)
    sparsity.deallocate(True)
    # down [1, E_loc, B, H] -> weight each expert's rows by its routing score, sum over experts
    routing_3d = ttnn.permute(local_routing, (0, 3, 2, 1))  # [1, E_loc, B, 1]
    next_states = ttnn.mul(down, routing_3d)
    routing_3d.deallocate(True)
    if local_routing is not routing_weights:
        local_routing.deallocate(True)
    next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
    next_states = ttnn.reshape(
        next_states,
        (1, 1, batch_size, hidden_size),
        (1, 1, max(32, batch_size), hidden_size),
    )
    if not combine:
        return next_states
    # Sum the ep rows' partial expert outputs, then the row-parallel down_proj partials over tp
    ep = int(getattr(mesh_config, "ep", 1) or 1) if mesh_config is not None else 1
    if ep > 1:
        next_states = ccl_allreduce(next_states, mesh_config, ccl_manager, axis=mesh_config.ep_axis, group_size=ep)
    if mesh_config is not None and mesh_config.tp > 1:
        next_states = ccl_allreduce(next_states, mesh_config, ccl_manager)

    return next_states
