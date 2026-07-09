# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device expert prefill forward using sparse_matmul (seq_len > 1).

Mirrors the gemma4 experts prefill path (all-ones sparsity computes every expert; the
dense routing weights zero out the inactive ones after down_proj) with SwiGLU and the
qwen reduce-scatter. The sequence is chunked to 32 tokens so the sparse_matmul group
dimension (seq_len/32) stays 1 and never overflows the core grid.
"""

import torch

import ttnn
from models.tt_transformers.tt.ccl import tt_all_reduce

from .decode import _build_sparse_matmul_config
from .operations import apply_swiglu
from .weights import ExpertWeights

TILE_SIZE = 32
PREFILL_CHUNK_SIZE = 32


def create_prefill_sparsity(mesh_device, num_experts):
    """All-ones sparsity mask [1,1,1,E] (ROW_MAJOR bf16); routing zeros inactive experts later."""
    is_mesh = hasattr(mesh_device, "shape")
    sparsity = torch.ones(1, 1, 1, num_experts, dtype=torch.bfloat16)
    return ttnn.from_torch(
        sparsity,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )


def _process_prefill_chunk(hidden_states, routing_weights, weights: ExpertWeights, config, prefill_sparsity):
    """One chunk: hidden [1,1,chunk,H], routing [1,1,chunk,E] -> [1,1,chunk,H] (pre-allreduce)."""
    chunk_len = hidden_states.shape[2]
    num_experts = config.num_experts
    hidden_size = config.hidden_size

    group_size = chunk_len // TILE_SIZE
    hidden_grouped = ttnn.reshape(hidden_states, (1, group_size, TILE_SIZE, hidden_size))
    sparsity = ttnn.repeat(prefill_sparsity, (1, 1, group_size, 1))
    nnz = num_experts * group_size

    output_tile = ttnn.Tile([32, 32])
    intermediate_size = weights.intermediate_size_per_device
    gate_up_config = _build_sparse_matmul_config(TILE_SIZE, intermediate_size)
    down_config = _build_sparse_matmul_config(TILE_SIZE, hidden_size)

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
    sm_intermediate = gate.shape[-1]
    gate = ttnn.transpose(gate, 1, 3)
    gate = ttnn.reshape(gate, (1, num_experts, chunk_len, sm_intermediate))

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
    # NB: do NOT deallocate hidden_grouped — it is a reshape *view* of the caller's
    # input x, which Qwen36MoE.forward reuses for the shared expert after the routed
    # experts run. Freeing it here frees x (TT_FATAL: input not allocated).
    up = ttnn.transpose(up, 1, 3)
    up = ttnn.reshape(up, (1, num_experts, chunk_len, sm_intermediate))

    down_input = apply_swiglu(gate, up)
    down_input = ttnn.reshape(down_input, (1, num_experts, chunk_len, sm_intermediate))

    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=prefill_sparsity,
        nnz=num_experts,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=down_config,
        is_input_a_sparse=True,
        dtype=ttnn.bfloat16,
    )
    down_input.deallocate(True)

    next_states = ttnn.reshape(down, (1, num_experts, chunk_len, hidden_size))
    # routing [1,1,S,E] -> [1,E,S,1] broadcast mul to select active experts
    routing_permuted = ttnn.permute(routing_weights, (0, 3, 2, 1))
    next_states = ttnn.mul(next_states, routing_permuted)
    next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
    next_states = ttnn.reshape(next_states, (1, 1, chunk_len, hidden_size))
    return next_states


def prefill_forward(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config,
    prefill_sparsity,
    mesh_device=None,
    tt_ccl=None,
    num_devices=1,
    topology=None,
):
    """hidden_states [1,1,S,H] (S multiple of 32), routing_weights [1,1,S,E]. Returns [1,1,S,H/tp]."""
    seq_len = hidden_states.shape[2]
    assert seq_len % TILE_SIZE == 0, f"Prefill seq_len must be multiple of {TILE_SIZE}, got {seq_len}"

    if seq_len > PREFILL_CHUNK_SIZE:
        hidden_chunks = ttnn.split(hidden_states, PREFILL_CHUNK_SIZE, dim=2)
        routing_chunks = ttnn.split(routing_weights, PREFILL_CHUNK_SIZE, dim=2)
    else:
        hidden_chunks = [hidden_states]
        routing_chunks = [routing_weights]

    result_acc = None
    for h_chunk, r_chunk in zip(hidden_chunks, routing_chunks):
        chunk_result = _process_prefill_chunk(h_chunk, r_chunk, weights, config, prefill_sparsity)
        if result_acc is None:
            result_acc = chunk_result
        else:
            result_concat = ttnn.concat([result_acc, chunk_result], dim=2)
            result_acc.deallocate(True)
            chunk_result.deallocate(True)
            result_acc = result_concat

    # Row-parallel down_proj partials -> reduce-scatter (fractured hidden), matching
    # Qwen36MLP._forward_tp.
    if num_devices > 1:
        result_acc = tt_all_reduce(
            result_acc,
            mesh_device,
            tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return result_acc
