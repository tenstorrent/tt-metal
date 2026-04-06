# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
On-device expert prefill forward using sparse_matmul.

For prefill (seq_len > 1), the sequence is reshaped into tile-sized groups:
  [1, 1, seq_len, H] -> [1, seq_len/32, 32, H]

The sparsity pattern is repeated across groups, and nnz is scaled by group_size.
This matches the gpt-oss prefill expert pattern.
"""


import ttnn

from .decode import _build_sparse_matmul_config
from .operations import apply_geglu
from .weights import ExpertWeights

TILE_SIZE = 32


def prefill_forward(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config,
    mesh_config=None,
    mesh_device=None,
    ccl_manager=None,
):
    """
    On-device expert prefill forward using sparse_matmul with tile-grouped sequence.

    Args:
        hidden_states: [1, 1, seq_len, hidden_size] on device (seq_len must be multiple of 32)
        routing_weights: [1, 1, seq_len, num_experts] on device (dense routing from router)
        weights: ExpertWeights
        config: Gemma4ExpertConfig

    Returns:
        output: [1, 1, seq_len, hidden_size] on device
    """
    seq_len = hidden_states.shape[2]
    num_experts = config.num_experts
    top_k = config.top_k
    intermediate_size = weights.intermediate_size_per_device
    hidden_size = config.hidden_size

    assert seq_len % TILE_SIZE == 0, f"Prefill seq_len must be multiple of {TILE_SIZE}, got {seq_len}"
    group_size = seq_len // TILE_SIZE

    # Reshape sequence into tile groups: [1, 1, S, H] -> [1, S/32, 32, H]
    hidden_grouped = ttnn.reshape(hidden_states, (1, group_size, TILE_SIZE, hidden_size))

    # Create sparsity from routing: take the first token's routing as representative
    # (all tokens in a tile group share the same sparsity pattern for sparse_matmul)
    # For simplicity, use the routing from the first token and repeat across groups
    routing_first = ttnn.slice(routing_weights, [0, 0, 0, 0], [1, 1, 1, num_experts])
    sparsity_base = ttnn.to_layout(routing_first, ttnn.ROW_MAJOR_LAYOUT)
    sparsity = ttnn.repeat(sparsity_base, ttnn.Shape((1, 1, group_size, 1)))

    # nnz scales with group_size (each group processes all its tokens independently)
    nnz = num_experts * group_size

    output_tile = ttnn.Tile([32, 32])
    gate_up_config = _build_sparse_matmul_config(TILE_SIZE, intermediate_size)
    down_config = _build_sparse_matmul_config(TILE_SIZE, hidden_size)

    # Gate projection
    gate = ttnn.sparse_matmul(
        hidden_grouped,
        weights.gate_proj,
        sparsity=sparsity,
        nnz=nnz,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_config,
        dtype=ttnn.bfloat16,
    )
    # Reshape: output is [1, group_size, E, 32, I] or similar -> [1, E, S, I]
    gate = ttnn.transpose(gate, 1, 3)
    gate = ttnn.reshape(gate, (1, num_experts, seq_len, intermediate_size))

    # Up projection
    up = ttnn.sparse_matmul(
        hidden_grouped,
        weights.up_proj,
        sparsity=sparsity,
        nnz=nnz,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_config,
        dtype=ttnn.bfloat16,
    )
    up = ttnn.transpose(up, 1, 3)
    up = ttnn.reshape(up, (1, num_experts, seq_len, intermediate_size))

    # GeGLU activation
    down_input = apply_geglu(gate, up)
    down_input = ttnn.reshape(down_input, (1, num_experts, seq_len, intermediate_size))

    # Down projection
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity_base,  # Original unscaled sparsity for down
        nnz=num_experts,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=down_config,
        is_input_a_sparse=True,
        dtype=ttnn.bfloat16,
    )

    # Apply routing weights and reduce
    # down: [1, E, S, H] -> [1, E, S, H]
    next_states = ttnn.reshape(down, (1, num_experts, seq_len, hidden_size))

    # routing_weights: [1, 1, S, E] -> [1, E, S, 1] for broadcast mul
    routing_permuted = ttnn.permute(routing_weights, (0, 3, 2, 1))  # [1, E, S, 1]
    next_states = ttnn.mul(next_states, routing_permuted)

    # Sum across experts dimension -> [1, 1, S, H]
    next_states = ttnn.sum(next_states, dim=1)
    next_states = ttnn.unsqueeze_to_4D(next_states)
    next_states = ttnn.reshape(next_states, (1, 1, seq_len, hidden_size))

    return next_states
