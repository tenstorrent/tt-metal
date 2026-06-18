# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEExperts bulk prefill — TP=4 on QB 4-chip Blackhole.

Follows gpt_oss/tt/experts/prefill.py _process_prefill_chunk pattern, extended
with an outer sequence chunk loop for long-context (S > S_OUTER_CHUNK):

Outer loop (S_OUTER_CHUNK = 32768 tokens per iteration):
  bounds `act [1, 128, S_outer, I]` at 3.87 GB/device — safe with 32 GB/device (QB TP=4).
  Gives 8× fewer outer iterations (8 vs 64 at S=256K) and 8× larger UP sparse_matmul.

Per outer chunk:
  1. Reshape [1, S_outer, H] → [1, G, 32, H]  (G = S_outer // 32)
  2. Up: all-ones sparsity [1,1,G,128] → dense sparse_matmul, nnz=128*G
  3. transpose(1,3) + reshape → [1, 128, S_outer, I];  relu²
  4. Routing weights: [1,1,S_outer,128] → [1,128,S_outer,1] via permute
  5. Down (S_CHUNK=128 inner chunks): is_input_a_sparse=True, sparsity=[1,1,1,128],
     nnz=128 → reshape [1,128,chunk,H] → mul routing → fast_reduce_nc → concat
  6. Outer concat → [1, 1, S, H] partial (all outer chunks, no all_reduce yet)

Single all_reduce at the end (one CCL op regardless of sequence length).
"""

import math

import torch

import ttnn
from ttnn import MeshDevice

from .tp import all_reduce

TILE = 32
N_EXPERTS = 128
HIDDEN_SIZE = 2688

_UP_CORES = (3, 5)
_DOWN_CORES = (7, 6)
_UP_IN0_BLOCK_W = 7
_DOWN_IN0_BLOCK_W = 3

# Outer sequence chunk: max tokens processed through up→act→down in one shot.
# act [1, 128, S_OUTER, I] = 128 * 16384 * 464 * 2 = 1.94 GB/device at S_OUTER=16384.
# QB TP=4 has 32 GB/device but at large ISL (e.g. 262K) the pre-allocated KV cache
# + model weights consume ~26 GB, leaving ~5 GB free with ~407 MB largest contiguous
# block per DRAM bank.  S_OUTER=32768 (3.87 GB) overflows that; S_OUTER=16384 (1.94 GB)
# needs 242 MB/bank which fits.  At S=256K this means 16 outer iterations (vs 8) with
# G=512 rows in the UP sparse_matmul (still 4× better than G=128 at S_OUTER=4096).
# Must be a multiple of TILE=32 (for the UP reshape) and ≥ S_CHUNK=128 (DOWN inner).
S_OUTER_CHUNK = 16384

# Inner DOWN chunk (bounds per_core_M = S_CHUNK//32 = 4).
S_CHUNK = 128

# Sparsity tensors cached per (G, id(mesh)).  All-ones = every expert active.
# ROW_MAJOR layout required by sparse_matmul.
_UP_SPARSITY_CACHE: dict = {}  # (G, id(mesh)) → [1, 1, G, 128] RM all-ones
_DOWN_SPARSITY_CACHE: dict = {}  # id(mesh) → [1, 1, 1, 128] RM all-ones

# Program config cache: same (chunk_S, local_inter) always produces the same config.
_DOWN_CFG_CACHE: dict = {}  # (chunk_S, local_inter) → MatmulMultiCoreReuseMultiCast1DProgramConfig


def _get_up_sparsity(G: int, mesh_device: MeshDevice) -> ttnn.Tensor:
    key = (G, id(mesh_device))
    if key not in _UP_SPARSITY_CACHE:
        ones = torch.ones(1, 1, G, N_EXPERTS, dtype=torch.bfloat16)
        _UP_SPARSITY_CACHE[key] = ttnn.from_torch(
            ones,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return _UP_SPARSITY_CACHE[key]


def _get_down_sparsity(mesh_device: MeshDevice) -> ttnn.Tensor:
    key = id(mesh_device)
    if key not in _DOWN_SPARSITY_CACHE:
        ones = torch.ones(1, 1, 1, N_EXPERTS, dtype=torch.bfloat16)
        _DOWN_SPARSITY_CACHE[key] = ttnn.from_torch(
            ones,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return _DOWN_SPARSITY_CACHE[key]


def _mm_cfg(
    cores_x: int,
    cores_y: int,
    m: int,
    n: int,
    k: int,
    in0_block_w: int,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    Nt = math.ceil(n / 32)
    num_cores = cores_x * cores_y
    per_core_N = math.ceil(Nt / num_cores)
    per_core_M = max(32, m) // 32
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cores_x, cores_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _get_down_cfg(chunk_S: int, local_inter: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    key = (chunk_S, local_inter)
    if key not in _DOWN_CFG_CACHE:
        _DOWN_CFG_CACHE[key] = _mm_cfg(
            *_DOWN_CORES,
            m=chunk_S,
            n=HIDDEN_SIZE,
            k=local_inter,
            in0_block_w=_DOWN_IN0_BLOCK_W,
        )
    return _DOWN_CFG_CACHE[key]


def _moe_experts_bulk_one_outer_chunk(
    mesh_device: MeshDevice,
    h_chunk: ttnn.Tensor,  # [1, S_outer, H] — S_outer divisible by 32; may be a view
    rw_chunk: ttnn.Tensor,  # [1, 1, S_outer, 128] — gate output for this chunk
    up_weights_tt: ttnn.Tensor,
    down_weights_tt: ttnn.Tensor,
) -> ttnn.Tensor:
    """Process one outer sequence chunk.  Returns [1, 1, S_outer, H] partial (before all_reduce)."""
    S_outer = h_chunk.shape[1]
    local_inter = up_weights_tt.shape[-1]
    G = S_outer // TILE
    output_tile = ttnn.Tile([32, 32])

    # UP: [1, S_outer, H] → [1, G, 32, H] (reshape is a view — do not deallocate h_chunk)
    h4d = ttnn.reshape(h_chunk, [1, G, TILE, HIDDEN_SIZE])
    sparsity_up = _get_up_sparsity(G, mesh_device)
    up_cfg = _mm_cfg(*_UP_CORES, m=TILE, n=local_inter, k=HIDDEN_SIZE, in0_block_w=_UP_IN0_BLOCK_W)
    up = ttnn.sparse_matmul(
        h4d,
        up_weights_tt,
        sparsity=sparsity_up,
        nnz=N_EXPERTS * G,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=up_cfg,
        dtype=ttnn.bfloat16,
    )
    # h4d is a reshape view of h_chunk — not deallocated here

    # Reorder → [1, 128, S_outer, I] then relu²
    up = ttnn.transpose(up, 1, 3)
    up = ttnn.reshape(up, [1, N_EXPERTS, S_outer, local_inter])
    act = ttnn.pow(ttnn.relu(up), 2)
    up.deallocate(True)

    # Routing weights: [1, 1, S_outer, 128] → [1, 128, S_outer, 1]
    rw = ttnn.permute(rw_chunk, [0, 3, 2, 1])

    # DOWN inner chunks (S_CHUNK=128 → per_core_M=4)
    if S_outer > S_CHUNK:
        act_chunks = ttnn.split(act, S_CHUNK, dim=2)
        act.deallocate(True)
        rw_chunks = ttnn.split(rw, S_CHUNK, dim=2)
        rw.deallocate(True)
    else:
        act_chunks = [act]
        rw_chunks = [rw]

    sparsity_down = _get_down_sparsity(mesh_device)
    acc = None
    for act_c, rw_c in zip(act_chunks, rw_chunks):
        c_S = act_c.shape[2]
        down = ttnn.sparse_matmul(
            act_c,
            down_weights_tt,
            sparsity=sparsity_down,
            nnz=N_EXPERTS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=_get_down_cfg(c_S, local_inter),
            is_input_a_sparse=True,
            dtype=ttnn.bfloat16,
        )
        act_c.deallocate(True)
        down = ttnn.reshape(down, [1, N_EXPERTS, c_S, HIDDEN_SIZE])
        down = ttnn.mul(down, rw_c, output_tensor=down)
        rw_c.deallocate(True)
        chunk_out = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(down, dims=[1]))
        down.deallocate(True)
        if acc is None:
            acc = chunk_out
        else:
            new_acc = ttnn.concat([acc, chunk_out], dim=2)
            acc.deallocate(True)
            chunk_out.deallocate(True)
            acc = new_acc

    return acc  # [1, 1, S_outer, H] partial (no all_reduce yet)


def moe_experts_prefill_forward(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [1, S, 2688] bf16 on device — caller may pass non-tile-aligned S
    routing_weights: ttnn.Tensor,  # [1, 1, S, 128] bf16 on device — dense routing mask
    up_weights_tt: ttnn.Tensor,  # [1, 128, 2688, 464] bfloat16, col-sharded
    down_weights_tt: ttnn.Tensor,  # [1, 128, 464, 2688] bfloat16, col-sharded
) -> ttnn.Tensor:
    """Bulk prefill MoE routed experts following gpt_oss experts/prefill.py.

    For S ≤ S_OUTER_CHUNK (32768): single outer chunk, no slicing.
    For S > S_OUTER_CHUNK: processes 32K-token outer chunks sequentially, accumulates
    partial [1,1,S,H] before a single all_reduce — one CCL op per E-layer regardless
    of sequence length.

    Returns [1, S, 2688] bfloat16 on device (replicated).
    """
    S = hidden_states.shape[1]

    # Pad to TILE multiple so inner reshape [1,S_outer,H] → [1,G,32,H] is valid.
    # Tokenizer round-trips can produce S that is not divisible by 32 (e.g. 126 for ISL=128).
    S_padded = math.ceil(S / TILE) * TILE
    if S_padded > S:
        pad_len = S_padded - S
        zeros_h = ttnn.from_torch(
            torch.zeros(1, pad_len, HIDDEN_SIZE, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        h_work = ttnn.concat([hidden_states, zeros_h], dim=1)
        zeros_h.deallocate(True)
        zeros_rw = ttnn.from_torch(
            torch.zeros(1, 1, pad_len, N_EXPERTS, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rw_work = ttnn.concat([routing_weights, zeros_rw], dim=2)
        zeros_rw.deallocate(True)
    else:
        # Already aligned — use originals directly (hidden_states may be a view of normed_tt)
        h_work = hidden_states
        rw_work = routing_weights

    if S_padded <= S_OUTER_CHUNK:
        # Single outer chunk — h_work/rw_work are either views (no pad) or new tensors (padded)
        partial = _moe_experts_bulk_one_outer_chunk(mesh_device, h_work, rw_work, up_weights_tt, down_weights_tt)
    else:
        # Multiple outer chunks — ttnn.slice creates independent copies
        acc = None
        for start in range(0, S_padded, S_OUTER_CHUNK):
            end = min(start + S_OUTER_CHUNK, S_padded)
            h_c = ttnn.slice(h_work, [0, start, 0], [1, end, HIDDEN_SIZE])
            rw_c = ttnn.slice(rw_work, [0, 0, start, 0], [1, 1, end, N_EXPERTS])
            chunk_partial = _moe_experts_bulk_one_outer_chunk(mesh_device, h_c, rw_c, up_weights_tt, down_weights_tt)
            h_c.deallocate(True)
            rw_c.deallocate(True)
            if acc is None:
                acc = chunk_partial
            else:
                new_acc = ttnn.concat([acc, chunk_partial], dim=2)
                acc.deallocate(True)
                chunk_partial.deallocate(True)
                acc = new_acc
        partial = acc

    if S_padded > S:
        h_work.deallocate(True)
        rw_work.deallocate(True)

    # Single all_reduce per E-layer: sums partial intermediate-column contributions across TP=4
    partial = all_reduce(partial)  # [1, 1, S_padded, H]

    # Trim padding tokens before reshaping back to [1, S, H]
    if S_padded > S:
        partial = ttnn.slice(partial, [0, 0, 0, 0], [1, 1, S, HIDDEN_SIZE])

    return ttnn.reshape(partial, [1, S, HIDDEN_SIZE])
