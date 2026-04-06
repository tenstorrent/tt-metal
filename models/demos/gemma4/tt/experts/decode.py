# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
On-device expert decode forward using sparse_matmul.

Follows gpt-oss experts/decode.py pattern but with GeGLU instead of SwiGLU
and no bias.
"""

import ttnn

from .operations import apply_geglu
from .weights import ExpertWeights


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
        hidden_states: [1, 1, seq_len, hidden_size] on device
        routing_weights: [1, 1, seq_len, num_experts] on device (dense routing from router)
        weights: ExpertWeights with gate/up/down projections
        config: Gemma4ExpertConfig

    Returns:
        output: [1, 1, seq_len, hidden_size] on device
    """
    batch_size = hidden_states.shape[2]
    num_experts = config.num_experts
    top_k = config.top_k
    intermediate_size = weights.intermediate_size_per_device

    # Prepare sparsity pattern for sparse_matmul
    sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
    output_tile = ttnn.Tile([32, 32])

    # Gate projection: sparse matmul across experts
    gate = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_proj,
        sparsity=sparsity,
        nnz=top_k,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        dtype=ttnn.bfloat16,
    )
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

    # Down projection
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        nnz=top_k,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        is_input_a_sparse=True,
        dtype=ttnn.bfloat16,
    )

    # Apply routing weights and reduce
    next_states = ttnn.permute(down, (0, 2, 1, 3))
    next_states = ttnn.reshape(next_states, (batch_size, num_experts, config.hidden_size))

    routing_weights_r = ttnn.permute(routing_weights, (1, 0))
    routing_weights_r = ttnn.reshape(routing_weights_r, (batch_size, num_experts, 1))
    next_states = ttnn.mul(next_states, routing_weights_r)

    # Sum across experts
    next_states = ttnn.sum(next_states, dim=1)
    next_states = ttnn.unsqueeze_to_4D(next_states)

    # Reshape to [1, 1, S, H]
    next_states = ttnn.reshape(
        next_states,
        (1, 1, batch_size, config.hidden_size),
        (1, 1, max(32, batch_size), config.hidden_size),
    )

    return next_states
