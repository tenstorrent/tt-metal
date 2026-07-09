# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device expert decode forward using sparse_matmul (seq_len=1).

Mirrors the gemma4 experts decode path with two Qwen changes: SwiGLU (not GeGLU),
and the row-parallel down_proj is combined with the qwen tt_all_reduce, which on the
(1,4) mesh REDUCE-SCATTERS along dim=3 — leaving the output fractured along the hidden
dim, exactly like Qwen36MLP._forward_tp, so the layer's residual add + DistributedNorm
stay aligned. sparse_matmul output is 6D: [batch_dims..., num_experts, seq_tiles, n].
"""

import math

import ttnn
from models.tt_transformers.tt.ccl import tt_all_reduce

from .operations import apply_swiglu
from .weights import ExpertWeights


def _build_sparse_matmul_config(m, n, in0_block_w=1):
    """Program config for sparse_matmul (largest divisor of n_tiles fitting an 8x8 grid)."""
    n_tiles = int(math.ceil(n / 32))

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
    mesh_device=None,
    tt_ccl=None,
    num_devices=1,
    topology=None,
):
    """hidden_states [1,1,S,H] (S=1 decode), routing_weights [1,1,S,E]. Returns [1,1,S,H/tp]."""
    batch_size = hidden_states.shape[2]
    num_experts = config.num_experts
    top_k = config.top_k
    intermediate_size = weights.intermediate_size_per_device

    sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
    output_tile = ttnn.Tile([32, 32])

    gate_up_config = _build_sparse_matmul_config(batch_size, intermediate_size)
    down_config = _build_sparse_matmul_config(batch_size, config.hidden_size)

    gate = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_proj,
        sparsity=sparsity,
        nnz=top_k,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_config,
        dtype=ttnn.bfloat16,
    )
    sm_intermediate = gate.shape[-1]
    gate = ttnn.reshape(gate, (batch_size, num_experts, 1, sm_intermediate))
    gate = ttnn.transpose(gate, 1, 2)
    gate = ttnn.reshape(gate, (batch_size, num_experts, sm_intermediate))

    up = ttnn.sparse_matmul(
        hidden_states,
        weights.up_proj,
        sparsity=sparsity,
        nnz=top_k,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_config,
        dtype=ttnn.bfloat16,
    )
    up = ttnn.reshape(up, (batch_size, num_experts, 1, sm_intermediate))
    up = ttnn.transpose(up, 1, 2)
    up = ttnn.reshape(up, (batch_size, num_experts, sm_intermediate))

    down_input = apply_swiglu(gate, up)

    down_input = ttnn.transpose(down_input, 1, 0)
    down_input = ttnn.reshape(down_input, (1, num_experts, batch_size, sm_intermediate))

    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        nnz=top_k,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=down_config,
        is_input_a_sparse=True,
        dtype=ttnn.bfloat16,
    )

    # down: [1, E, S, H] -> [1, S, E, H]
    next_states = ttnn.permute(down, (0, 2, 1, 3))
    next_states = ttnn.reshape(next_states, (batch_size, num_experts, config.hidden_size))

    # weight each expert's output by its routing score, then sum over experts
    routing_3d = ttnn.reshape(routing_weights, (batch_size, num_experts, 1))
    next_states = ttnn.mul(next_states, routing_3d)
    next_states = ttnn.sum(next_states, dim=1)
    next_states = ttnn.unsqueeze_to_4D(next_states)
    next_states = ttnn.reshape(
        next_states,
        (1, 1, batch_size, config.hidden_size),
        (1, 1, max(32, batch_size), config.hidden_size),
    )

    # Row-parallel down_proj partials -> reduce-scatter (fractured along hidden dim=3),
    # matching Qwen36MLP._forward_tp so residual/DistributedNorm alignment holds.
    if num_devices > 1:
        next_states = tt_all_reduce(
            next_states,
            mesh_device,
            tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return next_states
