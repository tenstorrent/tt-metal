# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device expert decode forward using sparse_matmul (the B decode users sit on dim-2).

Mirrors the gemma4 experts decode path with two Qwen changes: SwiGLU (not GeGLU),
and the row-parallel down_proj is combined with the qwen tt_all_reduce, which on the
(1,4) mesh REDUCE-SCATTERS along dim=3 — leaving the output fractured along the hidden
dim, exactly like Qwen36MLP._forward_tp, so the layer's residual add + DistributedNorm
stay aligned. sparse_matmul output is 6D: [batch_dims..., num_experts, seq_tiles, n].
"""

import math

import ttnn
from models.tt_transformers.tt.ccl import tt_all_reduce

from .operations import apply_swiglu
from .weights import ExpertWeights


def _pick_in0_block_w(k):
    """K tiles per mcast round: half the K extent, i.e. two accumulating blocks.

    The sparse_matmul inner loop runs K/in0_block_w mcast rounds per active expert, and
    in0_block_w=1 makes that sync dominate (measured 60 us per active expert on the gate_up
    shape vs 9 at K/2). Folding all of K into one block instead disables packer_l1_acc
    (which needs num_blocks > 1) and doubles the in1 buffer, which measured slower again
    once per_core_N > 1. Swept on the decode shapes: gate_up K=2048 -> 32, down K=512 -> 8.
    """
    k_tiles = int(math.ceil(k / 32))
    bw = 1
    while bw * 2 <= k_tiles // 2 and k_tiles % (bw * 2) == 0:
        bw *= 2
    return bw


def _build_sparse_matmul_config(m, n, in0_block_w=1, per_core_n=1):
    """Program config for sparse_matmul: n_tiles/per_core_n cores on an 8-wide grid.

    per_core_n also sets out_subblock_w — a 1x1 output subblock leaves the FPU running one
    tile per pass, and widening it to 1x2 (half the cores, two N-tiles each) measured faster
    on both decode shapes at every active-expert count swept. Falls back to one N-tile per
    core when n_tiles does not split that way.
    """
    n_tiles = int(math.ceil(n / 32))
    if per_core_n > 1 and (n_tiles % per_core_n or n_tiles // per_core_n > 64):
        per_core_n = 1
    num_cores = n_tiles // per_core_n
    best_cx = max(d for d in range(1, 9) if num_cores % d == 0)
    best_cy = num_cores // best_cx
    if best_cy > 8:
        per_core_n, num_cores = 1, n_tiles
        best_cx = max(d for d in range(1, 9) if num_cores % d == 0)
        best_cy = num_cores // best_cx

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(best_cx, best_cy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=per_core_n,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=max(32, m) // 32,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def decode_forward(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config,
    mesh_device=None,
    tt_ccl=None,
    num_devices=1,
    topology=None,
    reduce=True,
):
    """hidden_states [1,1,S,H] (S = decode batch), routing_weights [1,1,S,E]. Returns [1,1,S,H/tp]."""
    batch_size = hidden_states.shape[2]
    top_k = config.top_k
    intermediate_size = weights.intermediate_size_per_device
    # Expert-parallel: each device owns num_experts/num_devices experts (weights sharded dim=1).
    num_experts = config.num_experts // num_devices if num_devices > 1 else config.num_experts

    # Slice the replicated routing into this device's expert columns; nnz MUST stay None -- experts split unevenly and a static count deadlocks the mcast receivers.
    if num_devices > 1:
        routing_weights = ttnn.mesh_partition(routing_weights, dim=3, cluster_axis=1)

    # sparse_matmul wants one gate per expert, so collapse the user dim to a union mask; each user's own weight is applied later.
    if batch_size > 1:
        sparsity_src = ttnn.max(routing_weights, dim=2, keepdim=True)  # [1,1,1,E]
        nnz = None
    else:
        sparsity_src = routing_weights
        nnz = None if num_devices > 1 else top_k
    sparsity = ttnn.to_layout(sparsity_src, ttnn.ROW_MAJOR_LAYOUT)
    output_tile = ttnn.Tile([32, 32])

    # up/gate fused into ONE sparse_matmul over concatenated weights (N = 2*full_intermediate),
    # widening the N-gridded core count (8 -> 32) vs the old intermediate-parallel layout; the
    # fused output feeds ttnn.swiglu directly as [up | gate].
    gate_up_config = _build_sparse_matmul_config(
        batch_size, 2 * intermediate_size, _pick_in0_block_w(config.hidden_size), per_core_n=2
    )
    down_config = _build_sparse_matmul_config(
        batch_size, config.hidden_size, _pick_in0_block_w(intermediate_size), per_core_n=2
    )

    up_gate = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_up_proj,
        sparsity=sparsity,
        nnz=nnz,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_config,
        # bfloat8_b output: this expanded result is pure data movement, and the gate/up weights are already bfloat4_b, so bf16 buys no accuracy.
        dtype=ttnn.bfloat8_b,
    )
    # sparse_matmul returns rank 6, EXPERT-major: drop the leading unit dims to [1,E,B,2I] (order-preserving) -- reshaping to user-major would cross users' activations.
    up_gate = ttnn.reshape(up_gate, (1, num_experts, batch_size, up_gate.shape[-1]))
    down_input = apply_swiglu(up_gate)
    up_gate.deallocate(True)
    # swiglu promotes the tile-padded S to logical; retag it back (padded shape unchanged, so
    # this is metadata only) to keep the routing multiply below aligned on the real user count.
    down_input = ttnn.reshape(
        down_input,
        (1, num_experts, batch_size, intermediate_size),
        (1, num_experts, max(32, batch_size), intermediate_size),
    )

    # Apply routing HERE on the intermediate width: down_proj is linear, so it is identical work for a quarter of the elementwise cost.
    routing_per_expert = ttnn.permute(routing_weights, (0, 3, 2, 1))  # [1,1,S,E] -> [1,E,S,1]
    down_input = ttnn.mul(down_input, routing_per_expert)
    routing_per_expert.deallocate(True)

    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        nnz=nnz,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=down_config,
        is_input_a_sparse=True,
        dtype=ttnn.bfloat8_b,  # see gate_up above; this output is summed over experts, then reduce-scattered
    )

    # Routing is already applied (above), so the experts just sum: [1,E,S,H] -> [1,1,S,H].
    next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(down, dims=[1]))
    down.deallocate(True)
    next_states = ttnn.reshape(
        next_states,
        (1, 1, batch_size, config.hidden_size),
        (1, 1, max(32, batch_size), config.hidden_size),
    )

    # Row-parallel down_proj partials -> reduce-scatter (fractured along hidden dim=3),
    # matching Qwen36MLP._forward_tp so residual/DistributedNorm alignment holds.
    if num_devices > 1 and reduce:
        next_states = tt_all_reduce(
            next_states,
            mesh_device,
            tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return next_states
