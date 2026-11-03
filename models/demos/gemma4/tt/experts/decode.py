# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
On-device expert decode forward using sparse_matmul.

Follows gpt-oss experts/decode.py pattern with GeGLU instead of SwiGLU, no bias.
sparse_matmul output is 6D: [batch_dims..., num_experts, seq_tiles, n_dim].
"""

import math
import os
from collections import namedtuple

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.tt.ccl import ccl_allreduce

from .weights import ExpertWeights

# The fused router's routing form (GEMMA4_MOE_FUSED=1, custom_ops/router_topk): ``w`` [1, 1, E_loc, 32] bf16
# ROW_MAJOR (row e = the users' weights of local expert e) and ``sparsity`` [1, 1, 1, E_loc] bf16 ROW_MAJOR.
FusedRouting = namedtuple("FusedRouting", "w sparsity")
# The fused router's form for the decode MoE kernel rows (GEMMA4_MOE_FUSED=1 with GEMMA4_MOE_DECODE_KERNEL=1,
# custom_ops/router_topk.op_packed): ``packed`` [1, 1, 32, 2k] bf16 ROW_MAJOR (per user row: the k sum-normalized,
# per-expert-scaled weights then the k global expert ids as bf16; rows without a user: weight 0, ids 0..k-1).
PackedRouting = namedtuple("PackedRouting", "packed k")


def _decode_sparse_matmul_knobs(mesh_device=None):
    """Program-config knobs for the DECODE sparse_matmuls (Blackhole defaults: the whole compute grid and a 22-tile
    K block; env overrides). On the 4-chip 26B the shipped 8x8 / 1-tile config ran gate_up on 4 cores and down on 8
    with one K tile per mcast block: 32 distinct users 148 -> 75 ms per traced step (6.7 -> 13.4 tok/s/user) with
    these defaults, numerics identical.

    ``GEMMA4_SPARSE_MM_GRID=13x10`` lets the core grid use the whole Blackhole compute grid instead of the 8x8 the
    builder assumes (the 26B gate_up N=1408 -> 44 tiles only fits 4 cores in 8x8 but 11x4 in 13x10; down N=2816 ->
    88 tiles: 8 cores vs 11x8), ``GEMMA4_SPARSE_MM_IN0_BLOCK_W`` the K block (snapped to a divisor of Kt),
    ``GEMMA4_SPARSE_MM_OUT_SUBBLOCK_W`` the output subblock width (snapped to a divisor of per_core_N) and
    ``GEMMA4_SPARSE_MM_FIDELITY=lofi|hifi2|hifi3|hifi4`` the compute fidelity (unset: the op default). ``device``
    caps the grid at the device's compute grid.
    """
    grid = None
    env_grid = os.environ.get("GEMMA4_SPARSE_MM_GRID", "device" if is_blackhole() else "")
    if env_grid:
        if env_grid.lower() in ("device", "full", "max") and mesh_device is not None:
            g = mesh_device.compute_with_storage_grid_size()
            grid = (int(g.x), int(g.y))
        else:
            gx, gy = env_grid.lower().split("x")
            grid = (int(gx), int(gy))
        if mesh_device is not None:
            g = mesh_device.compute_with_storage_grid_size()
            grid = (min(grid[0], int(g.x)), min(grid[1], int(g.y)))
    fidelity = None
    env_fid = os.environ.get("GEMMA4_SPARSE_MM_FIDELITY", "").lower()
    if env_fid:
        fidelity = {
            "lofi": ttnn.MathFidelity.LoFi,
            "hifi2": ttnn.MathFidelity.HiFi2,
            "hifi3": ttnn.MathFidelity.HiFi3,
            "hifi4": ttnn.MathFidelity.HiFi4,
        }[env_fid]
    return {
        "max_grid": grid,
        "in0_block_w": max(1, int(os.environ.get("GEMMA4_SPARSE_MM_IN0_BLOCK_W", "22" if is_blackhole() else "1"))),
        "out_subblock_w": max(1, int(os.environ.get("GEMMA4_SPARSE_MM_OUT_SUBBLOCK_W", "1"))),
        "fidelity": fidelity,
    }


def _largest_divisor_at_most(n, cap):
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def _build_sparse_matmul_config(
    m, n, in0_block_w=1, max_grid=None, k=None, out_subblock_w=1, out_block_h=1, out_subblock_h=1
):
    """Build program config for sparse_matmul following gpt-oss pattern.

    ``max_grid`` (cx, cy) bounds the core rectangle (default the 8x8 the original builder assumed); the largest
    core count dividing the N tiles that fits the rectangle is used, so every core owns ``per_core_N`` output
    tiles and loops over the experts. ``k`` snaps ``in0_block_w`` to a divisor of the K tiles. ``out_block_h``
    (snapped to a divisor of ``per_core_M``) is the M-block height: the kernel re-reads a core's in1 (weight)
    column once per M block, so a prefill chunk of 16 row tiles with the default 1 reads every expert's weights
    16 times; the in0 block it multicasts grows with it (``out_block_h * in0_block_w`` tiles, double buffered).
    ``out_subblock_h`` (snapped to a divisor of ``out_block_h``, ``out_subblock_h * out_subblock_w <= 8`` dst tiles)
    is the number of output row tiles accumulated per pass: with the default 1 x 1 subblock every tile product
    costs two tile loads, an 8 x 1 subblock reuses each in1 tile over eight in0 tiles.
    """
    n_tiles = int(math.ceil(n / 32))
    max_cx, max_cy = max_grid if max_grid is not None else (8, 8)

    # Find largest divisor of n_tiles fitting in the max_cx x max_cy grid
    best_cores = 1
    best_cx, best_cy = 1, 1
    for num_cores in range(1, min(max_cx * max_cy, n_tiles) + 1):
        if n_tiles % num_cores != 0:
            continue
        for cy in range(1, max_cy + 1):
            if num_cores % cy == 0:
                cx = num_cores // cy
                if cx <= max_cx and num_cores > best_cores:
                    best_cores = num_cores
                    best_cx, best_cy = cx, cy
                    break

    per_core_N = n_tiles // best_cores
    if k is not None:
        in0_block_w = _largest_divisor_at_most(int(math.ceil(k / 32)), in0_block_w)
    # dst register file: out_subblock_h (1) * out_subblock_w <= 8; the subblock must divide per_core_N
    out_subblock_w = _largest_divisor_at_most(per_core_N, min(out_subblock_w, 8))
    per_core_M = max(32, m) // 32
    out_block_h = _largest_divisor_at_most(per_core_M, max(1, out_block_h))
    out_subblock_h = _largest_divisor_at_most(out_block_h, max(1, min(out_subblock_h, 8 // out_subblock_w)))

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(best_cx, best_cy),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=out_block_h,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def decode_forward(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config,
    mesh_config=None,
    mesh_device=None,
    ccl_manager=None,
    local_expert_mask=None,
    combine=True,
):
    """On-device expert decode forward using sparse_matmul (the B users sit on dim-2).

    Whole-expert sharding: this chip holds ``weights.num_local_experts`` experts at full width, and
    the replicated dense routing is partitioned to its expert columns (ep axis, then tp axis), so
    every tensor here is E_loc wide and nnz is data-dependent (inferred). ``combine=False``
    returns the local partial sum (the caller completes the cross-lane / cross-axis reduction).

    Args:
        hidden_states: [1, 1, B, hidden_size] on device
        routing_weights: [1, 1, B, num_experts] on device (dense, from router)
    Returns:
        output: [1, 1, B, hidden_size] on device
    """
    from .prefill import apply_geglu_fused, partition_routing_to_local_experts

    batch_size = hidden_states.shape[2]
    num_local = weights.num_local_experts
    intermediate_size = weights.intermediate_size_per_device
    hidden_size = config.hidden_size
    local_routing = partition_routing_to_local_experts(routing_weights, weights, mesh_config)
    # Sparsity [1,1,1,E_loc]: an expert is computed when any user in the batch routes to it; each
    # user's own weight is applied by the routing multiply below.
    if batch_size > 1:
        sparsity_src = ttnn.max(local_routing, dim=2, keepdim=True)
    else:
        sparsity_src = local_routing
    sparsity = ttnn.to_layout(sparsity_src, ttnn.ROW_MAJOR_LAYOUT)
    if sparsity_src is not local_routing:
        sparsity_src.deallocate(True)
    output_tile = ttnn.Tile([32, 32])
    knobs = _decode_sparse_matmul_knobs(mesh_device)
    gate_up_config = _build_sparse_matmul_config(
        batch_size,
        2 * intermediate_size,
        in0_block_w=knobs["in0_block_w"],
        max_grid=knobs["max_grid"],
        k=hidden_size,
        out_subblock_w=knobs["out_subblock_w"],
    )
    down_config = _build_sparse_matmul_config(
        batch_size,
        hidden_size,
        in0_block_w=knobs["in0_block_w"],
        max_grid=knobs["max_grid"],
        k=intermediate_size,
        out_subblock_w=knobs["out_subblock_w"],
    )
    compute_kernel_config = None
    if knobs["fidelity"] is not None:
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            hidden_states.device().arch(),
            math_fidelity=knobs["fidelity"],
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
    # [1,1,B,H] x [1,E_loc,H,2I] -> [1,1,1,E_loc,B_tile,2I]
    up_gate = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_up_proj,
        sparsity=sparsity,
        nnz=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_config,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
    )
    sm_n = up_gate.shape[-1]
    up_gate = ttnn.reshape(up_gate, (1, num_local, batch_size, sm_n))
    down_input = apply_geglu_fused(up_gate)
    up_gate.deallocate(True)
    down_input = ttnn.reshape(down_input, (1, num_local, batch_size, sm_n // 2))
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        nnz=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=down_config,
        is_input_a_sparse=True,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
    )
    down_input.deallocate(True)
    sparsity.deallocate(True)
    # down [1, E_loc, B, H] -> weight each expert's rows by its routing score, sum over experts
    routing_3d = ttnn.permute(local_routing, (0, 3, 2, 1))  # [1, E_loc, B, 1]
    next_states = ttnn.mul(down, routing_3d)
    routing_3d.deallocate(True)
    if local_routing is not routing_weights:
        local_routing.deallocate(True)
    next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
    next_states = ttnn.reshape(
        next_states,
        (1, 1, batch_size, hidden_size),
        (1, 1, max(32, batch_size), hidden_size),
    )
    if not combine:
        return next_states
    # Sum the ep rows' partial expert outputs, then the row-parallel down_proj partials over tp
    ep = int(getattr(mesh_config, "ep", 1) or 1) if mesh_config is not None else 1
    if ep > 1:
        next_states = ccl_allreduce(next_states, mesh_config, ccl_manager, axis=mesh_config.ep_axis, group_size=ep)
    if mesh_config is not None and mesh_config.tp > 1:
        next_states = ccl_allreduce(next_states, mesh_config, ccl_manager)

    return next_states


def decode_forward_fused(
    hidden_states,
    routing: FusedRouting,
    weights: ExpertWeights,
    config,
    mesh_config=None,
    mesh_device=None,
    ccl_manager=None,
    combine=True,
):
    """GEMMA4_MOE_FUSED=1 decode frame (<= 32 users on one row tile, every chip holds whole experts, ep == 1):
    the fused router's ``(w, sparsity)`` drive the gate_up ``sparse_matmul`` (L1 output), ``GegluDecode`` (one
    program for the slices / gelu / mul and the 6-D transpose), the down ``sparse_matmul`` and ``MoeCombine``
    (one program for the routing multiply and the expert sum), then the tp all-reduce. Same sparse_matmul configs
    and fidelity as ``decode_forward``; the sparsity is data-dependent (``nnz`` inferred) as before."""
    from models.demos.gemma4.custom_ops.geglu_decode import GegluDecode
    from models.demos.gemma4.custom_ops.moe_combine import MoeCombine

    batch_size = int(hidden_states.shape[2])
    num_local = weights.num_local_experts
    intermediate_size = weights.intermediate_size_per_device
    hidden_size = config.hidden_size
    output_tile = ttnn.Tile([32, 32])
    knobs = _decode_sparse_matmul_knobs(mesh_device)
    gate_up_config = _build_sparse_matmul_config(
        batch_size,
        2 * intermediate_size,
        in0_block_w=knobs["in0_block_w"],
        max_grid=knobs["max_grid"],
        k=hidden_size,
        out_subblock_w=knobs["out_subblock_w"],
    )
    down_config = _build_sparse_matmul_config(
        batch_size,
        hidden_size,
        in0_block_w=knobs["in0_block_w"],
        max_grid=knobs["max_grid"],
        k=intermediate_size,
        out_subblock_w=knobs["out_subblock_w"],
    )
    compute_kernel_config = None
    if knobs["fidelity"] is not None:
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            hidden_states.device().arch(),
            math_fidelity=knobs["fidelity"],
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
    mem = ttnn.L1_MEMORY_CONFIG if os.environ.get("GEMMA4_MOE_FUSED_L1", "1") == "1" else ttnn.DRAM_MEMORY_CONFIG
    up_gate = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_up_proj,
        sparsity=routing.sparsity,
        nnz=None,
        memory_config=mem,
        output_tile=output_tile,
        program_config=gate_up_config,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
    )
    up_gate = ttnn.reshape(up_gate, (1, num_local, batch_size, 2 * intermediate_size))  # 6-D -> 4-D view
    down_input = GegluDecode.op(up_gate)
    up_gate.deallocate(True)
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=routing.sparsity,
        nnz=None,
        memory_config=mem,
        output_tile=output_tile,
        program_config=down_config,
        is_input_a_sparse=True,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
    )
    down_input.deallocate(True)
    down = ttnn.reshape(down, (1, num_local, batch_size, hidden_size))
    next_states = MoeCombine.op(down, routing.w)
    down.deallocate(True)
    if not combine:
        return next_states
    if mesh_config is not None and mesh_config.tp > 1:
        next_states = ccl_allreduce(next_states, mesh_config, ccl_manager)
    return next_states
