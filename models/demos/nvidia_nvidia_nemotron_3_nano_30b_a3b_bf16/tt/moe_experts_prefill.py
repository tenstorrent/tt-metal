# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEExperts bulk prefill — TP=4 on QB 4-chip Blackhole.

Follows gpt_oss/tt/experts/prefill.py _process_prefill_chunk pattern:

1. Reshape [1, S, H] → [1, G, 32, H]  (G = S // 32 tile groups)
2. Up: all-ones sparsity [1,1,G,128] → dense sparse_matmul, nnz=128*G
3. transpose(1,3) + reshape → [1, 128, S, I];  relu²
4. Routing weights: [1,1,S,128] → [1,128,S,1] via permute
5. Down (chunked, S_CHUNK=128): sparse_matmul is_input_a_sparse=True,
   sparsity=[1,1,1,128] all-ones, nnz=128;  reshape → [1,128,S_chunk,H];
   mul routing weights;  fast_reduce_nc(dims=[1]) → [1,1,S_chunk,H];  concat
6. all_reduce across TP devices;  reshape → [1, S, H]
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


def moe_experts_prefill_forward(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [1, S, 2688] bf16 on device — S must be divisible by 32
    routing_weights: ttnn.Tensor,  # [1, 1, S, 128] bf16 on device — dense routing mask
    up_weights_tt: ttnn.Tensor,  # [1, 128, 2688, 464] bfloat16, col-sharded
    down_weights_tt: ttnn.Tensor,  # [1, 128, 464, 2688] bfloat16, col-sharded
) -> ttnn.Tensor:
    """Bulk prefill MoE routed experts following gpt_oss experts/prefill.py.

    Dense UP sparse_matmul (all 128 experts active per tile group), relu²,
    chunked DOWN sparse_matmul, routing-weight scaling, fast_reduce_nc,
    all_reduce for TP.

    Returns [1, S, 2688] bfloat16 on device (replicated).
    """
    S = hidden_states.shape[1]
    local_inter = up_weights_tt.shape[-1]  # 464 at TP=4
    G = S // TILE
    output_tile = ttnn.Tile([32, 32])

    # --- Up projection ---
    # [1, S, H] → [1, G, 32, H]: group tokens into 32-row tile batches
    # reshape returns a VIEW of hidden_states — do not deallocate (caller still needs it)
    h4d = ttnn.reshape(hidden_states, [1, G, TILE, HIDDEN_SIZE])

    # All-ones sparsity: every expert active for every tile group → dense matmul
    sparsity_up = _get_up_sparsity(G, mesh_device)
    up_cfg = _mm_cfg(
        *_UP_CORES,
        m=TILE,
        n=local_inter,
        k=HIDDEN_SIZE,
        in0_block_w=_UP_IN0_BLOCK_W,
    )
    # [1, G, 32, H] × [1, 128, H, I] → output reshaped below
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
    # h4d is a reshape view of hidden_states; caller still needs hidden_states for shared expert

    # Reorder from sparse_matmul layout → [1, 128, S, I]
    # (gpt_oss: transpose(1,3) then reshape — same raw layout)
    up = ttnn.transpose(up, 1, 3)
    up = ttnn.reshape(up, [1, N_EXPERTS, S, local_inter])

    act = ttnn.pow(ttnn.relu(up), 2)  # relu² activation, [1, 128, S, I]
    up.deallocate(True)

    # Routing weights: [1, 1, S, 128] → [1, 128, S, 1] for per-expert broadcast scaling
    rw = ttnn.permute(routing_weights, [0, 3, 2, 1])

    # --- Down projection (chunked over S to bound per_core_M) ---
    # S_CHUNK=128 → per_core_M=4; same chunk size for all ISLs (128, 256, 512, 1024)
    S_CHUNK = 128
    if S > S_CHUNK:
        act_chunks = ttnn.split(act, S_CHUNK, dim=2)
        act.deallocate(True)
        rw_chunks = ttnn.split(rw, S_CHUNK, dim=2)
        rw.deallocate(True)
    else:
        act_chunks = [act]
        rw_chunks = [rw]

    # All-ones down sparsity: all 128 expert input blocks are active
    sparsity_down = _get_down_sparsity(mesh_device)

    acc = None
    for act_chunk, rw_chunk in zip(act_chunks, rw_chunks):
        chunk_S = act_chunk.shape[2]
        down_cfg = _get_down_cfg(chunk_S, local_inter)

        # [1, 128, chunk_S, I] × [1, 128, I, H] — sparse over expert dim
        down = ttnn.sparse_matmul(
            act_chunk,
            down_weights_tt,
            sparsity=sparsity_down,
            nnz=N_EXPERTS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_cfg,
            is_input_a_sparse=True,
            dtype=ttnn.bfloat16,
        )
        act_chunk.deallocate(True)

        # Reshape sparse_matmul output → [1, 128, chunk_S, H]
        down = ttnn.reshape(down, [1, N_EXPERTS, chunk_S, HIDDEN_SIZE])

        # Scale by routing weights and zero out inactive experts
        down = ttnn.mul(down, rw_chunk, output_tensor=down)
        rw_chunk.deallocate(True)

        # Sum over expert dim: [1, 128, chunk_S, H] → [1, 1, chunk_S, H]
        chunk_out = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(down, dims=[1]))
        down.deallocate(True)

        if acc is None:
            acc = chunk_out
        else:
            new_acc = ttnn.concat([acc, chunk_out], dim=2)
            acc.deallocate(True)
            chunk_out.deallocate(True)
            acc = new_acc

    # TP all-reduce: sum partial intermediate-column results across 4 devices
    acc = all_reduce(acc)  # [1, 1, S, H]

    return ttnn.reshape(acc, [1, S, HIDDEN_SIZE])
