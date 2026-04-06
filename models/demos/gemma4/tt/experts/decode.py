# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
On-device expert decode forward using sparse_matmul.

Follows gpt-oss experts/decode.py pattern with GeGLU instead of SwiGLU, no bias.
sparse_matmul output is 6D: [batch_dims..., num_experts, seq_tiles, n_dim].
"""

import math

import ttnn

from .operations import apply_geglu
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
):
    """
    On-device expert forward using sparse_matmul.

    Args:
        hidden_states: [1, 1, seq_len, hidden_size] on device (seq_len=1 for decode)
        routing_weights: [1, 1, seq_len, num_experts] on device (dense, from router)
        weights: ExpertWeights
        config: Gemma4ExpertConfig

    Returns:
        output: [1, 1, seq_len, hidden_size] on device
    """
    batch_size = hidden_states.shape[2]  # seq_len (1 for decode)
    num_experts = config.num_experts
    top_k = config.top_k
    intermediate_size = weights.intermediate_size_per_device

    # Prepare sparsity pattern (must be ROW_MAJOR bfloat16)
    sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
    output_tile = ttnn.Tile([32, 32])

    gate_up_config = _build_sparse_matmul_config(batch_size, intermediate_size)
    down_config = _build_sparse_matmul_config(batch_size, config.hidden_size)

    # Gate projection: [1,1,S,H] × [1,E,H,I] → [1,1,S,E,S_tile,I]
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
    # Reshape 6D → 3D: [batch, num_experts, intermediate]
    gate = ttnn.reshape(gate, (batch_size, num_experts, 1, intermediate_size))
    gate = ttnn.transpose(gate, 1, 2)
    gate = ttnn.reshape(gate, (batch_size, num_experts, intermediate_size))

    # Up projection
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
    up = ttnn.reshape(up, (batch_size, num_experts, 1, intermediate_size))
    up = ttnn.transpose(up, 1, 2)
    up = ttnn.reshape(up, (batch_size, num_experts, intermediate_size))

    # GeGLU activation
    down_input = apply_geglu(gate, up)

    # Prepare for down projection
    down_input = ttnn.transpose(down_input, 1, 0)
    down_input = ttnn.reshape(down_input, (1, num_experts, batch_size, intermediate_size))

    # Down projection: sparse, input is also sparse
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

    # down shape: [1, E, S, H] — permute to [1, S, E, H]
    next_states = ttnn.permute(down, (0, 2, 1, 3))
    # Reshape to [batch, E, H] for weighted sum
    next_states = ttnn.reshape(next_states, (batch_size, num_experts, config.hidden_size))

    # routing_weights: [1, 1, S, E] → reshape to [batch, E, 1] for broadcast mul
    routing_3d = ttnn.reshape(routing_weights, (batch_size, num_experts, 1))
    next_states = ttnn.mul(next_states, routing_3d)

    # Sum across experts dimension
    next_states = ttnn.sum(next_states, dim=1)
    next_states = ttnn.unsqueeze_to_4D(next_states)

    # Reshape to [1, 1, S, H]
    next_states = ttnn.reshape(
        next_states,
        (1, 1, batch_size, config.hidden_size),
        (1, 1, max(32, batch_size), config.hidden_size),
    )

    return next_states
