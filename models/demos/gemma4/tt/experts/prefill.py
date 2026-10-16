# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
On-device expert prefill forward using sparse_matmul.

Following gpt_oss pattern: sparse_matmul computes ALL experts (all-ones sparsity),
then routing weights are applied after down projection to select active experts.

For prefill (seq_len > 1), the sequence is reshaped into tile-sized groups:
  [1, 1, seq_len, H] -> [1, seq_len/32, 32, H]
"""

import os

import torch

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce

from .decode import _build_sparse_matmul_config
from .operations import apply_geglu
from .weights import ExpertWeights

TILE_SIZE = 32


def create_prefill_sparsity(mesh_device, num_local_experts, mesh_config=None, ep=1):
    """All-ones [1, 1, 1, E_loc] ROW_MAJOR bf16 mask over this chip's whole experts (the routing
    weights zero the inactive ones after the down projection). ``mesh_config``/``ep`` are
    accepted for the older call sites; the expert set is already local under the whole-expert
    sharding, so no per-row mask is needed."""
    is_mesh = hasattr(mesh_device, "shape")
    sparsity = torch.ones(1, 1, 1, num_local_experts, dtype=torch.bfloat16)
    return ttnn.from_torch(
        sparsity,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )


def partition_routing_to_local_experts(routing_weights, weights: ExpertWeights, mesh_config=None):
    """[1, 1, S, E] replicated dense routing -> this chip's [1, 1, S, E_loc] expert columns.

    A 0/1 selection matmul (exact in bf16) rather than a per-device tiled slice, so the local
    width E_loc need not be a tile multiple (it is 4 on the Galaxy).
    """
    if weights.expert_select is None:
        return routing_weights
    return ttnn.matmul(
        routing_weights,
        weights.expert_select,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )


def apply_geglu_fused(up_gate):
    """GeGLU over concatenated [up | gate] on the last dim: up * gelu(gate) (same gelu variant as
    the unfused path, see compute_config). Tiled slices need a tile-aligned half (704 is); the
    unaligned case (small test configs) goes through ROW_MAJOR."""
    b, e, s, n2 = (int(x) for x in up_gate.shape)
    half = n2 // 2
    src = up_gate
    if half % TILE_SIZE:
        src = ttnn.to_layout(up_gate, ttnn.ROW_MAJOR_LAYOUT)
    up = ttnn.slice(src, [0, 0, 0, 0], [b, e, s, half])
    gate = ttnn.slice(src, [0, 0, 0, half], [b, e, s, n2])
    if src is not up_gate:
        src.deallocate(True)
        up = ttnn.to_layout(up, ttnn.TILE_LAYOUT)
        gate = ttnn.to_layout(gate, ttnn.TILE_LAYOUT)
    out = apply_geglu(gate, up)
    gate.deallocate(True)
    up.deallocate(True)
    return out


def _process_prefill_chunk(hidden_states, routing_weights, weights: ExpertWeights, config, prefill_sparsity):
    """One chunk: hidden [1,1,chunk,H], routing [1,1,chunk,E_loc] -> [1,1,chunk,H] (this chip's partial).

    Qwen3.6 MoE pattern: the sequence is grouped into 32-row tiles folded into the sparse batch
    dim; up and gate are ONE sparse_matmul over the fused [up | gate] weight; the per-tile
    sparsity computes an expert for a tile only when some token in it routes there (nnz is
    data-dependent, so it is inferred); every tensor is E_loc experts wide.
    """
    chunk_len = int(hidden_states.shape[2])
    num_local = weights.num_local_experts
    hidden_size = config.hidden_size
    intermediate = weights.intermediate_size_per_device
    group_size = chunk_len // TILE_SIZE
    hidden_grouped = ttnn.reshape(hidden_states, (1, group_size, TILE_SIZE, hidden_size))
    routing_tiled = ttnn.reshape(routing_weights, (1, group_size, TILE_SIZE, num_local))
    tile_mask = ttnn.max(routing_tiled, dim=2, keepdim=True)  # [1, group, 1, E_loc]
    tile_mask = ttnn.to_layout(tile_mask, ttnn.ROW_MAJOR_LAYOUT)
    sparsity = ttnn.reshape(tile_mask, (1, 1, group_size, num_local))
    output_tile = ttnn.Tile([32, 32])
    gate_up_config = _build_sparse_matmul_config(TILE_SIZE, 2 * intermediate)
    down_config = _build_sparse_matmul_config(chunk_len, hidden_size)
    expert_compute_kernel_config = ttnn.init_device_compute_kernel_config(
        hidden_grouped.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    up_gate = ttnn.sparse_matmul(
        hidden_grouped,
        weights.gate_up_proj,
        sparsity=sparsity,
        nnz=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_config,
        compute_kernel_config=expert_compute_kernel_config,
        dtype=ttnn.bfloat16,
    )
    up_gate = ttnn.transpose(up_gate, 1, 3)
    up_gate = ttnn.reshape(up_gate, (1, num_local, chunk_len, 2 * intermediate))
    down_input = apply_geglu_fused(up_gate)
    up_gate.deallocate(True)
    down_input = ttnn.reshape(down_input, (1, num_local, chunk_len, intermediate))
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=prefill_sparsity,
        nnz=num_local,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=down_config,
        is_input_a_sparse=True,
        compute_kernel_config=expert_compute_kernel_config,
        dtype=ttnn.bfloat16,
    )
    down_input.deallocate(True)
    next_states = ttnn.reshape(down, (1, num_local, chunk_len, hidden_size))
    routing_permuted = ttnn.permute(routing_weights, (0, 3, 2, 1))  # [1, E_loc, S, 1]
    next_states = ttnn.mul(next_states, routing_permuted)
    routing_permuted.deallocate(True)
    # Reduce across experts dimension using fast_reduce_nc (gpt_oss pattern)
    next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
    next_states = ttnn.reshape(next_states, (1, 1, chunk_len, hidden_size))
    return next_states


# Maximum tokens per chunk for sparse_matmul prefill.
# sparse_matmul treats the group dimension (seq_len/32) as part of the M dimension,
# which increases num_blocks_y and can overflow the core grid.
# Using chunk_size=32 means group_size=1, so num_blocks_y=1 and the
# blocks come entirely from N-dimension — guaranteed to fit.
# 512 tokens (group_size 16) like the Qwen3.6 MoE: ~16x fewer gate/up/down dispatches than the
# previous 32-token groups while the down projection's per_core_M (= chunk/32) still fits the grid.
PREFILL_CHUNK_SIZE = int(os.environ.get("GEMMA4_MOE_PREFILL_CHUNK", "512"))


def prefill_forward(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config,
    prefill_sparsity,
    mesh_config=None,
    mesh_device=None,
    ccl_manager=None,
    num_local_experts=None,
    combine=True,
):
    """
    On-device expert prefill forward using sparse_matmul.

    Under EP ``prefill_sparsity`` is the device's local-expert mask and ``num_local_experts``
    its expert count; ``combine=False`` returns the local partial sum.

    Following gpt_oss pattern:
    1. Chunk long sequences so group_size fits in the core grid
    2. sparse_matmul gate/up/down with all-ones sparsity (compute ALL experts)
    3. Apply routing weights after down projection to select active experts
    4. Reduce across expert dimension

    Args:
        hidden_states: [1, 1, seq_len, hidden_size] on device (seq_len must be multiple of 32)
        routing_weights: [1, 1, seq_len, num_experts] on device (dense routing from router)
        weights: ExpertWeights
        config: Gemma4ExpertConfig
        prefill_sparsity: [1, 1, 1, num_experts] all-ones sparsity mask (cached, do not deallocate)

    Returns:
        output: [1, 1, seq_len, hidden_size] on device
    """
    seq_len = hidden_states.shape[2]
    hidden_size = config.hidden_size

    assert seq_len % TILE_SIZE == 0, f"Prefill seq_len must be multiple of {TILE_SIZE}, got {seq_len}"

    # This chip's expert columns of the replicated dense routing, once for the whole sequence.
    local_routing = partition_routing_to_local_experts(routing_weights, weights, mesh_config)
    if seq_len > PREFILL_CHUNK_SIZE:
        hidden_chunks = ttnn.split(hidden_states, PREFILL_CHUNK_SIZE, dim=2)
        routing_chunks = ttnn.split(local_routing, PREFILL_CHUNK_SIZE, dim=2)
    else:
        hidden_chunks = [hidden_states]
        routing_chunks = [local_routing]
    chunked = len(hidden_chunks) > 1
    results = []
    for h_chunk, r_chunk in zip(hidden_chunks, routing_chunks):
        results.append(_process_prefill_chunk(h_chunk, r_chunk, weights, config, prefill_sparsity))
        if chunked:  # split() made fresh copies; free them as we go
            h_chunk.deallocate(True)
            r_chunk.deallocate(True)
    if local_routing is not routing_weights:
        local_routing.deallocate(True)
    if len(results) == 1:
        result_acc = results[0]
    else:  # one concat instead of a growing accumulation (quadratic copying)
        result_acc = ttnn.concat(results, dim=2)
        for r in results:
            r.deallocate(True)

    if not combine:
        return result_acc
    # Sum the ep rows' partial expert outputs, then the row-parallel down_proj partials over tp
    ep = int(getattr(mesh_config, "ep", 1) or 1) if mesh_config is not None else 1
    if ep > 1:
        result_acc = ccl_allreduce(result_acc, mesh_config, ccl_manager, axis=mesh_config.ep_axis, group_size=ep)
    if mesh_config is not None and mesh_config.tp > 1:
        result_acc = ccl_allreduce(result_acc, mesh_config, ccl_manager)

    return result_acc
