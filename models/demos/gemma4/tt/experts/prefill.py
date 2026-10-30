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
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.tt.ccl import ccl_allreduce

from .decode import _build_sparse_matmul_config, _decode_sparse_matmul_knobs
from .operations import apply_geglu
from .weights import ExpertWeights

TILE_SIZE = 32


def _prefill_sparse_matmul_knobs(mesh_device):
    """Program-config knobs for the real prefill chunks (chunk_len > 32): on by default on Blackhole
    (``GEMMA4_SPARSE_MM_PREFILL=0`` restores the shipped 8x8 / 1-tile configs), off elsewhere. 4-chip 26B TTFT at
    128 / 2048 / 8192 tokens: 462 / 6263 / 24876 -> 123 / 901 / 3619 ms, logits PCC 1.000000, same first tokens.

    The shipped builder puts the 26B gate_up (N=1408 -> 44 tiles) on 4 cores and the down (N=2816 -> 88 tiles) on
    8, one K tile per multicast block, for every 32-token group x expert pair; with the flag the decode knobs
    (``GEMMA4_SPARSE_MM_GRID`` whole compute grid, ``GEMMA4_SPARSE_MM_IN0_BLOCK_W`` 22) apply to the chunks too:
    gate_up on 44 cores, down on 88. On top of them:

    * ``GEMMA4_SPARSE_MM_PREFILL_IN0_BLOCK_W`` (default 11) K tiles per multicast block;
    * ``GEMMA4_SPARSE_MM_PREFILL_OUT_BLOCK_H`` (default 16) row tiles per M block: a core re-reads its weight
      column once per M block, so a 512-token chunk (16 row tiles) reads every expert's weights 16x at 1 and
      once at 16; the in0 block it multicasts is ``out_block_h * in0_block_w`` bf16 tiles double buffered
      (16 x 11 = 704 KB, Blackhole L1; 16 x 22 overflows it). 2048-token prefill on the 4-chip 26B:
      1 -> 2483 ms, 4 -> 1076, 8 -> 932, 16 x 11 -> 900;
    * ``GEMMA4_SPARSE_MM_PREFILL_OUT_SUBBLOCK_H`` (default 1) output row tiles accumulated per pass (<= out_block_h,
      <= 8 dst tiles). Measured slower than 1 (4 -> 1184 ms, 8 -> 1014 ms at out_block_h 4 / 8);
    * ``GEMMA4_SPARSE_MM_PREFILL_ROUTE_EARLY`` (default 0): apply the routing weights to the GeGLU output
      ([E_loc, chunk, I]) instead of the down-projection output ([E_loc, chunk, H]); exact in real arithmetic
      and 4x less eltwise traffic (-35 ms), but the logits PCC vs the shipped path dropped to 0.63 (first token
      changed) on the 2048-token prompt, so it stays off until that is understood;
    * ``GEMMA4_SPARSE_MM_PREFILL_DENSE_GATE_UP`` (default 1): the gate_up runs the whole chunk (M = chunk rows)
      against every local expert with the all-ones mask, like the down projection, instead of one 32-row group
      per sparse batch entry. The per-group tile mask skips ~13% of the (group, expert) pairs (top-8 of 128 over
      32 tokens) but re-reads each expert's weights once per group (64x per 2048 tokens); the dense form reads
      them once per chunk, needs no group<->expert transpose, and the routing multiply after the down projection
      still zeroes the unselected experts exactly as before.
    """
    if os.environ.get("GEMMA4_SPARSE_MM_PREFILL", "1" if is_blackhole() else "0") != "1":
        return None
    knobs = _decode_sparse_matmul_knobs(mesh_device)
    knobs["in0_block_w"] = max(1, int(os.environ.get("GEMMA4_SPARSE_MM_PREFILL_IN0_BLOCK_W", "11")))
    knobs["out_block_h"] = max(1, int(os.environ.get("GEMMA4_SPARSE_MM_PREFILL_OUT_BLOCK_H", "16")))
    knobs["out_subblock_h"] = max(1, int(os.environ.get("GEMMA4_SPARSE_MM_PREFILL_OUT_SUBBLOCK_H", "1")))
    knobs["route_early"] = os.environ.get("GEMMA4_SPARSE_MM_PREFILL_ROUTE_EARLY", "0") == "1"
    knobs["dense_gate_up"] = os.environ.get("GEMMA4_SPARSE_MM_PREFILL_DENSE_GATE_UP", "1") == "1"
    return knobs


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
    device = hidden_states.device()
    output_tile = ttnn.Tile([32, 32])
    math_fidelity = ttnn.MathFidelity.HiFi4
    # Decode frames (a batched decode step of up to 32 users lands here with chunk_len == 32) take the same
    # GEMMA4_SPARSE_MM_* program-config knobs as the batch-1 path in decode.py; all unset -> the shipped configs.
    # Real prefill chunks take the GEMMA4_SPARSE_MM_PREFILL knobs (see _prefill_sparse_matmul_knobs), default off.
    knobs = _decode_sparse_matmul_knobs(device) if chunk_len <= TILE_SIZE else _prefill_sparse_matmul_knobs(device)
    out_block_h = knobs.get("out_block_h", 1) if knobs is not None else 1
    out_subblock_h = knobs.get("out_subblock_h", 1) if knobs is not None else 1
    dense_gate_up = bool(knobs.get("dense_gate_up", False)) if knobs is not None else False
    route_early = bool(knobs.get("route_early", False)) if knobs is not None else False
    if dense_gate_up:
        # the all-ones [1,1,1,E_loc] mask: the whole chunk against every local expert (down-projection form)
        sparsity = None
    else:
        hidden_grouped = ttnn.reshape(hidden_states, (1, group_size, TILE_SIZE, hidden_size))
        routing_tiled = ttnn.reshape(routing_weights, (1, group_size, TILE_SIZE, num_local))
        tile_mask = ttnn.max(routing_tiled, dim=2, keepdim=True)  # [1, group, 1, E_loc]
        tile_mask = ttnn.to_layout(tile_mask, ttnn.ROW_MAJOR_LAYOUT)
        sparsity = ttnn.reshape(tile_mask, (1, 1, group_size, num_local))
    if knobs is not None and (knobs["max_grid"] is not None or knobs["in0_block_w"] > 1 or knobs["out_subblock_w"] > 1):
        gate_up_config = _build_sparse_matmul_config(
            chunk_len if dense_gate_up else TILE_SIZE,
            2 * intermediate,
            in0_block_w=knobs["in0_block_w"],
            max_grid=knobs["max_grid"],
            k=hidden_size,
            out_subblock_w=knobs["out_subblock_w"],
            out_block_h=out_block_h,
            out_subblock_h=out_subblock_h,
        )
        down_config = _build_sparse_matmul_config(
            chunk_len,
            hidden_size,
            in0_block_w=knobs["in0_block_w"],
            max_grid=knobs["max_grid"],
            k=intermediate,
            out_subblock_w=knobs["out_subblock_w"],
            out_block_h=out_block_h,
            out_subblock_h=out_subblock_h,
        )
    else:
        gate_up_config = _build_sparse_matmul_config(chunk_len if dense_gate_up else TILE_SIZE, 2 * intermediate)
        down_config = _build_sparse_matmul_config(chunk_len, hidden_size)
    if knobs is not None and knobs["fidelity"] is not None:
        math_fidelity = knobs["fidelity"]
    expert_compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    if dense_gate_up:
        # [1,1,chunk,H] x [1,E_loc,H,2I] -> [.., E_loc, chunk, 2I]: experts on the sparse batch dim, no transpose
        up_gate = ttnn.sparse_matmul(
            hidden_states,
            weights.gate_up_proj,
            sparsity=prefill_sparsity,
            nnz=num_local,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_config,
            compute_kernel_config=expert_compute_kernel_config,
            dtype=ttnn.bfloat16,
        )
    else:
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
    routing_permuted = ttnn.permute(routing_weights, (0, 3, 2, 1))  # [1, E_loc, S, 1]
    if route_early:
        # weight each expert's GeGLU rows before the (linear) down projection: same sum, 4x smaller tensor
        weighted = ttnn.mul(down_input, routing_permuted)
        down_input.deallocate(True)
        down_input = weighted
    # Decode frames with GEMMA4_MOE_FRAME_SPARSE_DOWN=1: the down projection skips the experts no user of the
    # frame routes to (the same [1,1,1,E_loc] tile mask the up/gate projection uses, nnz inferred) instead of
    # computing all E_loc experts and zeroing them through the routing weights below. Off by default: real
    # prefill chunks (group_size > 1) keep the dense down projection.
    sparse_down = (
        chunk_len <= TILE_SIZE
        and sparsity is not None
        and os.environ.get("GEMMA4_MOE_FRAME_SPARSE_DOWN", "1" if is_blackhole() else "0") == "1"
    )
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity if sparse_down else prefill_sparsity,
        nnz=None if sparse_down else num_local,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=down_config,
        is_input_a_sparse=True,
        compute_kernel_config=expert_compute_kernel_config,
        dtype=ttnn.bfloat16,
    )
    down_input.deallocate(True)
    next_states = ttnn.reshape(down, (1, num_local, chunk_len, hidden_size))
    if not route_early:
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
