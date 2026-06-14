# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEExperts — TP=4 on QB 4-chip Blackhole.

128 routed NemotronHMLP experts, top-6 per token (top-k=6), intermediate=1856.

Device-side implementation using ttnn.sparse_matmul, mirroring the pattern in
models/demos/gpt_oss/tt/experts/decode.py.  All expert weights are pre-stacked
as [1, 128, hidden, intermediate] device tensors (loaded once per layer at
first call, cached in the caller).  The routing-weight sparsity tensor comes
from moe_gate_forward as a dense [tokens, 128] device tensor with zeros for
inactive experts.

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

# Blackhole BH has an 8×8 usable Tensix grid.  Core-grid constraints for
# MatmulMultiCoreReuseMultiCast1DProgramConfig (decode S=1):
#   The kernel asserts num_cores_with_work == in0_mcast_receiver_num_cores, so
#   ceil(Nt / per_core_N) == num_cores must hold exactly.
#
#   up:   [tokens, 2688] × [2688, 1856]  →  Nt=58, Kt=84
#     (5,4)=20 cores → per_core_N=ceil(58/20)=3, ceil(58/3)=20 ✓
#   down: [tokens, 1856] × [1856, 2688]  →  Nt=84, Kt=58
#     (7,6)=42 cores → per_core_N=ceil(84/42)=2, ceil(84/2)=42 ✓
_UP_CORES = (5, 4)  # 20 cores
_DOWN_CORES = (7, 6)  # 42 cores
_UP_IN0_BLOCK_W = 7  # divides Kt=84
_DOWN_IN0_BLOCK_W = 2  # divides Kt=58


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
    routing_weights: ttnn.Tensor,  # [tokens, N/TP] bf16 per device — sharded sparsity mask
    up_weights_tt: ttnn.Tensor,  # [1, N/TP, 2688, 1856] bf16 per device — sharded experts
    down_weights_tt: ttnn.Tensor,  # [1, N/TP, 1856, 2688] bf16 per device — sharded experts
) -> ttnn.Tensor:
    """Returns [tokens, 2688] bfloat16 on device (replicated).

    Expert weights and routing are sharded N/TP per device (N=128, TP=4 → 32 per device).
    Each device computes its local expert outputs, then all_reduce sums partial results:
      sparse_matmul(h, up_W_local, sparsity_local)   → [tokens, N/TP, intermediate]
      relu²                                           → same shape
      sparse_matmul(act, down_W_local, sparsity_local, is_input_a_sparse=True)
                                                      → [tokens, N/TP, hidden]
      mul routing_local + sum local experts           → [tokens, hidden]  (partial)
      all_reduce                                      → [tokens, hidden]  (full)
    """
    tokens = hidden_states.shape[0]
    # Number of experts handled by this device (N_EXPERTS // TP, e.g. 32 for TP=4).
    n_local = routing_weights.shape[-1]
    output_tile = ttnn.Tile([32, 32])

    # sparsity tensor: sparse_matmul expects ROW_MAJOR [tokens, n_local_experts]
    sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)

    up_cfg = _mm_config(
        *_UP_CORES, m=tokens, n=MOE_INTERMEDIATE, k=HIDDEN_SIZE, in0_block_w=_UP_IN0_BLOCK_W, out_subblock_w=1
    )
    h4d = ttnn.unsqueeze_to_4D(hidden_states)  # [1, 1, tokens, 2688]

    # Up: [1, 1, tokens, 2688] × [1, n_local, 2688, 1856] → [1, tokens, n_local, 1856]
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
    up = ttnn.reshape(up, (tokens, n_local, MOE_INTERMEDIATE))

    act = ttnn.pow(ttnn.relu(up), 2)

    act = ttnn.transpose(act, 0, 1)
    act = ttnn.reshape(act, (1, n_local, tokens, MOE_INTERMEDIATE))

    down_cfg = _mm_config(
        *_DOWN_CORES, m=tokens, n=HIDDEN_SIZE, k=MOE_INTERMEDIATE, in0_block_w=_DOWN_IN0_BLOCK_W, out_subblock_w=1
    )

    # Down: [1, n_local, tokens, intermediate] × [1, n_local, intermediate, hidden]
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

    # down: [1, n_local, tokens, 2688] → [tokens, n_local, 2688]
    down = ttnn.reshape(down, (n_local, tokens, HIDDEN_SIZE))
    down = ttnn.transpose(down, 0, 1)

    # Scale by local routing weights and sum over local expert dimension.
    rw = ttnn.reshape(routing_weights, (tokens, n_local, 1))
    down = ttnn.mul(down, rw, output_tensor=down)
    rw.deallocate(True)

    partial = ttnn.sum(down, dim=1)  # [tokens, 2688] partial sum over local experts
    down.deallocate(True)

    # Sum partial results from all TP devices → full expert output.
    return all_reduce(partial)
