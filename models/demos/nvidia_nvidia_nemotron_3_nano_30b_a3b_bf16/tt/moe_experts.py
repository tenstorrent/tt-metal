# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEExperts — TP=4 on QB 4-chip Blackhole.

128 routed NemotronHMLP experts, top-6 per token (top-k=6), intermediate=1856.

Column-parallel TP sharding: up_weight [1,128,2688,1856] is split along dim=3
into [1,128,2688,464] per device; down_weight [1,128,1856,2688] along dim=2 into
[1,128,464,2688] per device.  All 128 experts are present on every device so
sparse_matmul never gets 0 active experts (which caused noc_semaphore_wait hang
with the prior expert-count sharding).  After the expert weighted-sum each
device holds a partial result; all_reduce sums the 4 partial outputs.

Activation: relu2 = relu(x)^2   (NemotronH uses this instead of SwiGLU)
"""

import math

import ttnn
from ttnn import MeshDevice

from .tp import all_reduce

N_EXPERTS = 128
TOP_K = 6
HIDDEN_SIZE = 2688
MOE_INTERMEDIATE = 1856
MOE_INTERMEDIATE_LOCAL = MOE_INTERMEDIATE // 4  # 464 per device at TP=4

# Blackhole BH has an 8×8 usable Tensix grid.  Core-grid constraints for
# MatmulMultiCoreReuseMultiCast1DProgramConfig (decode S=1):
#   The kernel asserts num_cores_with_work == in0_mcast_receiver_num_cores, so
#   ceil(Nt / per_core_N) == num_cores must hold exactly.
#
#   up:   [tokens, 2688] × [2688, 464]  →  Nt=15, Kt=84
#     (3,5)=15 cores → per_core_N=ceil(15/15)=1, ceil(15/1)=15 ✓
#   down: [tokens, 464] × [464, 2688]  →  Nt=84, Kt=15
#     (7,6)=42 cores → per_core_N=ceil(84/42)=2, ceil(84/2)=42 ✓
_UP_CORES = (3, 5)  # 15 cores
_DOWN_CORES = (7, 6)  # 42 cores
_UP_IN0_BLOCK_W = 7  # divides Kt=84
_DOWN_IN0_BLOCK_W = 3  # divides Kt=15


def _mm_config(cores_x, cores_y, m, n, k, in0_block_w, out_subblock_w=1):
    Nt = math.ceil(n / 32)
    num_cores = cores_x * cores_y
    per_core_N = math.ceil(Nt / num_cores)
    per_core_M = max(32, m) // 32
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cores_x, cores_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=1,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def moe_experts_forward(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [tokens, 2688] bf16 on device (replicated)
    routing_weights: ttnn.Tensor,  # [tokens, 128] bf16 on device — replicated sparsity mask
    up_weights_tt: ttnn.Tensor,  # [1, 128, 2688, 464] bfloat16 on device — col-sharded
    down_weights_tt: ttnn.Tensor,  # [1, 128, 464, 2688] bfloat16 on device — col-sharded
) -> ttnn.Tensor:
    """Returns [tokens, 2688] bfloat16 on device (replicated).

    Column-parallel: all 128 experts on every device, each with 1/4 of the
    intermediate columns.  sparse_matmul sparsity selects the 6 active experts
    — all devices always have 6 non-zero entries (no empty-device hang).

      sparse_matmul(h, up_W, sparsity)   → [tokens, 128, 464]  partial intermediate
      relu²                              → same shape
      sparse_matmul(act, down_W, ...)    → [tokens, 128, 2688] partial output
      mul routing + sum over experts     → [tokens, 2688]       partial final
      all_reduce                         → [tokens, 2688]       full output
    """
    tokens = hidden_states.shape[0]
    local_inter = up_weights_tt.shape[-1]  # 464 = MOE_INTERMEDIATE // 4
    output_tile = ttnn.Tile([32, 32])

    # sparsity tensor: sparse_matmul expects ROW_MAJOR [tokens, n_experts]
    sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)

    up_cfg = _mm_config(
        *_UP_CORES, m=tokens, n=local_inter, k=HIDDEN_SIZE, in0_block_w=_UP_IN0_BLOCK_W, out_subblock_w=1
    )
    h4d = ttnn.unsqueeze_to_4D(hidden_states)  # [1, 1, tokens, 2688]

    # Up: [1, 1, tokens, 2688] × [1, 128, 2688, 464] → [1, tokens, 128, 464]
    up = ttnn.sparse_matmul(
        h4d,
        up_weights_tt,
        sparsity=sparsity,
        nnz=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=up_cfg,
        dtype=ttnn.bfloat16,
    )
    up = ttnn.reshape(up, (tokens, N_EXPERTS, local_inter))

    act = ttnn.pow(ttnn.relu(up), 2)

    act = ttnn.transpose(act, 0, 1)
    act = ttnn.reshape(act, (1, N_EXPERTS, tokens, local_inter))

    down_cfg = _mm_config(
        *_DOWN_CORES, m=tokens, n=HIDDEN_SIZE, k=local_inter, in0_block_w=_DOWN_IN0_BLOCK_W, out_subblock_w=1
    )

    # Down: [1, 128, tokens, 464] × [1, 128, 464, 2688] → [1, 128, tokens, 2688]
    down = ttnn.sparse_matmul(
        act,
        down_weights_tt,
        sparsity=sparsity,
        nnz=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=down_cfg,
        is_input_a_sparse=True,
        dtype=ttnn.bfloat16,
    )
    act.deallocate(True)
    sparsity.deallocate(True)

    # down: [1, 128, tokens, 2688] → [tokens, 128, 2688]
    down = ttnn.reshape(down, (N_EXPERTS, tokens, HIDDEN_SIZE))
    down = ttnn.transpose(down, 0, 1)

    # Scale by routing weights and sum over expert dimension → partial [tokens, 2688]
    rw = ttnn.reshape(routing_weights, (tokens, N_EXPERTS, 1))
    down = ttnn.mul(down, rw, output_tensor=down)
    rw.deallocate(True)

    partial = ttnn.sum(down, dim=1)
    down.deallocate(True)

    # Sum partial intermediate-column results across TP devices → full [tokens, 2688]
    return all_reduce(partial)
