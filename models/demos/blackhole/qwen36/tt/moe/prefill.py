# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device expert prefill forward using sparse_matmul (seq_len > 1).

Mirrors the gpt_oss experts prefill path (all-ones sparsity computes every expert; the
dense routing weights zero out the inactive ones after down_proj) with SwiGLU and the
qwen reduce-scatter. The sequence is processed in chunks of PREFILL_CHUNK_SIZE tokens;
within a chunk gate and up run as ONE fused sparse_matmul over concatenated weights
(N = 2*intermediate), which doubles the N-gridded core count (4 -> 8) vs two separate
matmuls; the fused output stays [up | gate] and is consumed directly by ttnn.swiglu
(up * silu(gate)) — no split slices. group_size = chunk/32
folds into the sparse batch dim so per_core_M stays 1 tile. The down projection's M *is*
the chunk length, so its program config is sized from the real chunk_len (per_core_M =
chunk_len/32); PREFILL_CHUNK_SIZE bounds that so the down grid/L1 never overflows.
"""

import torch

import ttnn
from models.tt_transformers.tt.ccl import tt_all_reduce

from .decode import _build_sparse_matmul_config
from .operations import apply_swiglu
from .weights import ExpertWeights

TILE_SIZE = 32
# Tokens processed per grouped sparse_matmul. Larger = fewer, bigger matmuls (fewer dispatches),
# bounded so the down projection's per_core_M (= chunk/32) fits the core grid / L1. 512 → group_size
# 16, ~16x fewer gate/up/down dispatches than the legacy 32 while staying PCC-clean.
PREFILL_CHUNK_SIZE = 512


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
    """One chunk: hidden [1,1,chunk,H], routing [1,1,chunk,E_local] -> [1,1,chunk,H] (pre-allreduce).

    Expert-parallel: routing_weights is already this device's expert-column slice (E_local =
    num_experts/tp); prefill_sparsity is all-ones over the same E_local experts."""
    chunk_len = hidden_states.shape[2]
    # Per-device expert count (weights + sparsity are already sharded to E_local).
    num_experts = prefill_sparsity.shape[-1]
    hidden_size = config.hidden_size

    group_size = chunk_len // TILE_SIZE
    hidden_grouped = ttnn.reshape(hidden_states, (1, group_size, TILE_SIZE, hidden_size))
    # Per-tile sparsity for gate/up: compute an expert for a 32-token tile only if some token in
    # the tile routes to it (max routing weight over the tile > 0). Real prefill routing is
    # concentrated (~21 of 64 local experts hit per 32-tok tile), so this skips ~2/3 of the
    # all-ones overcompute. nnz varies per tile -> infer (None); a static nnz would deadlock the
    # sparse_matmul mcast receivers (same reason decode uses nnz=None).
    routing_tiled = ttnn.reshape(routing_weights, (1, group_size, TILE_SIZE, num_experts))
    tile_mask = ttnn.max(routing_tiled, dim=2, keepdim=True)  # [1, group, 1, E_local]
    tile_mask = ttnn.to_layout(tile_mask, ttnn.ROW_MAJOR_LAYOUT)
    sparsity = ttnn.reshape(tile_mask, (1, 1, group_size, num_experts))
    nnz = None

    output_tile = ttnn.Tile([32, 32])
    intermediate_size = weights.intermediate_size_per_device
    # up/gate fused into ONE sparse_matmul: N = 2*full_intermediate ([up|gate] concatenated), so
    # the N-gridded core count is 32 (vs 8 for the old intermediate-parallel N=256). M is one
    # 32-row tile per group (group_size folded into the sparse batch dim), per_core_M stays 1.
    # down: M is the full chunk_len, so its per_core_M reflects the real M (= chunk_len/32).
    gate_up_config = _build_sparse_matmul_config(TILE_SIZE, 2 * intermediate_size)
    down_config = _build_sparse_matmul_config(chunk_len, hidden_size)

    up_gate = ttnn.sparse_matmul(
        hidden_grouped,
        weights.gate_up_proj,
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
    up_gate = ttnn.transpose(up_gate, 1, 3)
    up_gate = ttnn.reshape(up_gate, (1, num_experts, chunk_len, 2 * intermediate_size))

    down_input = apply_swiglu(up_gate)
    up_gate.deallocate(True)
    down_input = ttnn.reshape(down_input, (1, num_experts, chunk_len, intermediate_size))

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

    # Expert-parallel: slice the replicated dense routing [1,1,S,E] into this device's
    # contiguous expert columns [1,1,S,E/tp] (matching the dim=1-sharded expert weights),
    # once for the whole sequence before chunking.
    if num_devices > 1:
        routing_weights = ttnn.mesh_partition(routing_weights, dim=3, cluster_axis=1)

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
