# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma-local TRUE-SPARSE token-gather MoE.

The dense denoise MoE (gemma4 prefill expert path) computes ALL 128 experts for every 32-token
tile-group and then zeros 120/128 via the routing weights — ~137 ms/layer, ~99% of the denoise
step, and the sole wall between the current ~1 t/s and the 30 t/s target. Only the top-8 experts
per token are active, so ~16x of that compute is wasted.

This module replaces it with a GShard-style **capacity dispatch**: gather each active expert's
assigned tokens into a fixed-capacity buffer, run only the active experts' gate/up/down as a
**batched matmul** (one C-token tile per expert), then scatter back weighted by the routing
weights and all-reduce. On the real 26B layer-0 canvas this is **~13x cheaper** (10.5 vs 137 ms)
with PCC 0.9997 vs the dense path (see doc/optimize_perf/bench_ondevice_dispatch.py).

Why this is legal on the (1,4) TP mesh: the canvas input is REPLICATED across TP (experts are
TP-sharded on the intermediate dim, NOT expert-parallel), so gather/scatter over the token dim is
LOCAL per device. Only the down-projection needs the existing TP all-reduce. No cross-device
token dispatch (unlike deepseek's all_to_all, which needs a 16-row ring).

Trace-safety: all ops have fixed shapes and device-resident (UINT32) indices; the fixed step
budget keeps the dispatch program-cache warm. NEVER edits gemma4 — composes over ``moe.router``
and ``moe.experts.weights`` only.
"""

from __future__ import annotations

import os

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.demos.gemma4.tt.experts.operations import apply_geglu

TILE = 32
DEFAULT_CAPACITY = 32


def default_sparse_moe_compute_kernel_config():
    """HiFi2 matches the dense sparse_matmul numerics (PCC 0.9997 vs dense at bf16)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


# Constant/scratch tensors for the dispatch (independent of the routing VALUES) are allocated ONCE
# and reused. ``ttnn.zeros/ones/full`` do host->device writes, which are illegal inside a captured
# trace, so they must live outside it. Cached by (mesh id, S, E, C, top_k). The zero-base buffers
# are only ever consumed (out-of-place scatter never mutates its input), so reuse is correct.
_CONST_CACHE = {}


def _get_dispatch_constants(mesh, S, num_experts, capacity, top_k):
    key = (id(mesh), S, num_experts, capacity, top_k)
    consts = _CONST_CACHE.get(key)
    if consts is None:
        EC = num_experts * capacity
        consts = {
            "ones_sk": ttnn.ones([1, 1, S, top_k], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh),
            "zeros_se": ttnn.zeros([1, 1, S, num_experts], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh),
            "zeros_d": ttnn.zeros([1, 1, S, EC + 1], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh),
            "zeros_c": ttnn.zeros([1, 1, S, EC + 1], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh),
            "dead": ttnn.full([1, 1, S, top_k], float(EC), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh),
        }
        _CONST_CACHE[key] = consts
    return consts


def build_capacity_dispatch(dense_routing, num_experts, capacity, top_k):
    """Build dispatch + combine masks on-device from dense routing (GShard capacity dispatch).

    Args:
        dense_routing: [1, 1, S, E] on device — dense routing weights, exactly ``top_k`` non-zero
            per token (the router output, including per-expert scale).
        num_experts: E
        capacity: C — max tokens per expert (must be a multiple of 32). Tokens beyond C for a hot
            expert are dropped (their contribution is zeroed); pick C so the drop rate is ~0.
        top_k: active experts per token.

    Returns:
        disp: [1, 1, S, E*C] bf16 — one-hot dispatch mask, disp[t, e*C+slot] = 1.
        comb: [1, 1, S, E*C] bf16 — combine mask, comb[t, e*C+slot] = routing_weight(t, e).

    The mapping token t, expert e -> column (e*C + slot) uses ``slot`` = number of earlier tokens
    that also routed to e (an exclusive cumulative count), so each expert's tokens land in a
    contiguous C-wide band. Trace-safe: constant/scratch buffers are preallocated (see
    ``_get_dispatch_constants``); only device compute happens here.
    """
    mesh = dense_routing.device()
    S = dense_routing.shape[2]
    EC = num_experts * capacity
    k = _get_dispatch_constants(mesh, S, num_experts, capacity, top_k)
    ones_sk, zeros_se, zeros_d, zeros_c, dead = (k["ones_sk"], k["zeros_se"], k["zeros_d"], k["zeros_c"], k["dead"])

    # 1. per-token active experts + their routing weights
    vals, idx = ttnn.topk(dense_routing, k=top_k, dim=-1)  # vals [1,1,S,k] bf16, idx uint16
    idx_u = ttnn.typecast(idx, ttnn.uint32)  # gather/scatter index must be UINT32/UINT16
    idx_f = ttnn.typecast(idx, ttnn.float32)

    # 2. active mask [1,1,S,E]
    mask = ttnn.scatter(zeros_se, dim=-1, index=idx_u, src=ones_sk)

    # 3. exclusive slot position within each expert bucket (# earlier tokens on same expert)
    cum = ttnn.cumsum(mask, dim=2, dtype=ttnn.float32)  # inclusive cumsum over tokens
    mask_f = ttnn.typecast(mask, ttnn.float32)
    excl = ttnn.sub(cum, mask_f)  # [1,1,S,E] f32

    # 4. slot per (token, active-expert)
    pos = ttnn.gather(excl, dim=-1, index=idx_u)  # [1,1,S,k] f32

    # 5. dispatch column = e*C + slot; overflow (slot >= C) -> dead column EC (dropped)
    col = ttnn.add(ttnn.mul(idx_f, float(capacity)), pos)  # f32 exact up to 2^24 >> EC
    overflow = ttnn.ge(pos, float(capacity))
    col = ttnn.where(overflow, dead, col)
    valid = ttnn.sub(ones_sk, overflow)  # 1 where kept, 0 where dropped
    vals_valid = ttnn.mul(vals, valid)  # zero the dropped weights
    col_u = ttnn.typecast(col, ttnn.uint32)

    # 6. scatter into the E*C dispatch buffers (+1 dead column, then sliced off)
    disp = ttnn.scatter(zeros_d, dim=-1, index=col_u, src=ones_sk)
    comb = ttnn.scatter(zeros_c, dim=-1, index=col_u, src=vals_valid)
    disp = ttnn.slice(disp, [0, 0, 0, 0], [1, 1, S, EC])
    comb = ttnn.slice(comb, [0, 0, 0, 0], [1, 1, S, EC])

    for t in (vals, idx, idx_u, idx_f, mask, cum, mask_f, excl, pos, col, overflow, valid, vals_valid, col_u):
        t.deallocate(True)
    return disp, comb


# ---------------------------------------------------------------------------------------------
# OPT-004: matmul-geometry tuning of the 5 sparse_matmul calls (opt-in via DG_SPARSE_MOE_TUNED).
#
# The five plain ``ttnn.matmul`` calls in this module (gate/up/down in ``_batched_experts`` plus the
# gather ``disp^T @ hidden`` and combine ``comb @ down`` matmuls) were written for the Lever-A GO/NO-GO
# prototype and pass ONLY ``memory_config`` + ``compute_kernel_config`` — no ``program_config``, so the
# op auto-picks a config. On the real 26B QB2 layer the batched experts read the ~415 MB (bf16) / ~220 MB
# (bfp8) expert bank at only ~46 GB/s effective (~18% of the @256 GB/s roofline). OPT-004 adds explicit
# core-grid + ``in0_block_w`` geometry so the 128 experts pack across the whole compute grid and the
# weight streams in larger K-blocks. Same math (same dtype/fidelity) — PCC must stay at the untuned
# value; this is a pure geometry change. Gated so the current path is bit-identical when the flag is off.
#
# Real per-device shapes on QB2 (mesh (1,4), TP=4): E=128 experts, top_k=8, H=2816 (88 tiles),
# moe_intermediate padded per device I=192 (6 tiles), canvas S=256 (8 tiles), capacity C=32 (1 tile),
# EC=E*C=4096 (128 tiles). BH P150 compute grid = 13x10 = 130 cores/chip.
#
# Key TTNN op-contract facts these builders encode (verified against the matmul device op + factory):
#   * MatmulMultiCoreReuseProgramConfig (batched gate/up/down): the reuse factory FORCES
#     ``per_core_N == Nt`` (the N dim is never split across cores) and distributes the E expert output
#     blocks over the grid via ``split_work_to_cores``. With ``per_core_M == Mt`` there are exactly E=128
#     blocks; ``split_work_to_cores`` then uses ``min(E, grid_cores)`` cores — 128 on BH (1 expert/core,
#     no serialization), 64 on WH (2 experts/core). ``in0_block_w`` (the K-block, must divide Kt) is the
#     only real geometry knob; larger = fewer K passes / bigger DRAM reads, bounded by L1.
#   * MatmulMultiCoreReuseMultiCastProgramConfig (2D, gather/combine): M is parallelized over grid.y,
#     N over grid.x; ``per_core_M = ceil(Mt/gy)``, ``per_core_N = ceil(Nt/gx)`` (ceil is legal — 2D pads).
#   * Subblocks: ``out_subblock_h | per_core_M``, ``out_subblock_w | per_core_N``, product <= 8 (bf16
#     half-dest; the HiFi2/fp32_dest_acc_en=False policy here gives 8). ``Kt % in0_block_w == 0``.
# ---------------------------------------------------------------------------------------------

# in1 (second-operand / weight) block budget in TILES: ``per_core_N * in0_block_w <= this``. Sized so a
# bf16 double-buffered in1 CB stays well under BH's ~1.4 MB usable L1 (176*2 tiles * 2 KB ≈ 720 KB). The
# default bfp8 expert weights leave ~2x more headroom, so this is a conservative floor that is safe for
# either dtype. Drives in0_block_w=22 for gate/up (per_core_N=6) and in0_block_w=2 for down (per_core_N=88).
_IN1_BLOCK_TILE_BUDGET = 176

_TUNED_CONFIG_CACHE = {}


def tuned_configs_enabled():
    """OPT-004 tuned program configs are opt-in; the current (auto-config) path is the default."""
    return os.environ.get("DG_SPARSE_MOE_TUNED", "0") == "1"


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _largest_divisor_leq(n, cap):
    """Largest divisor of ``n`` that is <= ``cap`` (at least 1)."""
    best = 1
    for d in range(1, min(n, cap) + 1):
        if n % d == 0:
            best = d
    return best


def _pick_in0_block_w(k_tiles, per_core_n):
    """Largest divisor of Kt keeping the in1 block within ``_IN1_BLOCK_TILE_BUDGET`` tiles."""
    cap = max(1, _IN1_BLOCK_TILE_BUDGET // max(1, per_core_n))
    return _largest_divisor_leq(k_tiles, cap)


def _pick_out_subblock(per_core_m, per_core_n, max_prod=8):
    """Largest (h, w) with h|per_core_M, w|per_core_N and h*w <= max_prod (dest-register cap)."""
    best_h, best_w, best_prod = 1, 1, 1
    for h in _divisors(per_core_m):
        for w in _divisors(per_core_n):
            prod = h * w
            if prod <= max_prod and prod > best_prod:
                best_h, best_w, best_prod = h, w, prod
    return best_h, best_w


def _device_grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return int(g.x), int(g.y)


def tuned_batched_expert_config(mesh, m_tiles, k_tiles, n_tiles):
    """Program config for a batched ``[1,E,C,·] @ [1,E,·,·]`` expert matmul (gate/up/down).

    per_core_M = Mt (one expert per output block -> E=128 blocks distributed over the grid),
    per_core_N = Nt (forced by the reuse factory), in0_block_w = largest K divisor within L1.
    """
    gx, gy = _device_grid(mesh)
    per_core_M = m_tiles
    per_core_N = n_tiles
    in0_block_w = _pick_in0_block_w(k_tiles, per_core_N)
    sh, sw = _pick_out_subblock(per_core_M, per_core_N)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )


def tuned_2d_matmul_config(mesh, m_tiles, k_tiles, n_tiles):
    """2D systolic program config for the gather / combine matmuls (M over grid.y, N over grid.x)."""
    import math

    gx, gy = _device_grid(mesh)
    per_core_M = math.ceil(m_tiles / gy)
    per_core_N = math.ceil(n_tiles / gx)
    in0_block_w = _pick_in0_block_w(k_tiles, per_core_N)
    sh, sw = _pick_out_subblock(per_core_M, per_core_N)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def build_tuned_configs(mesh, E, C, H, I, S):
    """Build (and cache) the 5 OPT-004 program configs for the real per-device shapes.

    Returns a dict with keys ``gate_up`` / ``down`` (batched) and ``gather`` / ``combine`` (2D).
    Pure host-side construction (no device writes) so it is trace-safe; cached by (mesh, shapes).
    """
    gx, gy = _device_grid(mesh)
    key = (id(mesh), gx, gy, E, C, H, I, S)
    cfgs = _TUNED_CONFIG_CACHE.get(key)
    if cfgs is not None:
        return cfgs
    EC = E * C
    t = TILE
    cfgs = {
        # gate/up: gathered[1,E,C,H] @ w[1,E,H,I]  -> M=C, K=H, N=I
        "gate_up": tuned_batched_expert_config(mesh, C // t, H // t, I // t),
        # down: down_input[1,E,C,I] @ w[1,E,I,H]   -> M=C, K=I, N=H
        "down": tuned_batched_expert_config(mesh, C // t, I // t, H // t),
        # gather: disp^T[1,1,EC,S] @ hidden[1,1,S,H] -> M=EC, K=S, N=H
        "gather": tuned_2d_matmul_config(mesh, EC // t, S // t, H // t),
        # combine: comb[1,1,S,EC] @ down_flat[1,1,EC,H] -> M=S, K=EC, N=H
        "combine": tuned_2d_matmul_config(mesh, S // t, EC // t, H // t),
    }
    _TUNED_CONFIG_CACHE[key] = cfgs
    return cfgs


def _batched_experts(gathered, weights, compute_kernel_config, program_configs=None):
    """Batched gate/up/geglu/down over active experts.

    gathered: [1, E, C, H] — each expert's capacity tokens.
    program_configs: optional dict with ``gate_up`` / ``down`` OPT-004 program configs; None keeps the
        auto-config path (bit-identical to the untuned prototype).
    Returns: [1, E, C, H] partial (TP-sharded down output, pre all-reduce).
    """
    gate_up_pc = program_configs.get("gate_up") if program_configs else None
    down_pc = program_configs.get("down") if program_configs else None
    gate = ttnn.matmul(
        gathered,
        weights.gate_proj,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
        program_config=gate_up_pc,
    )
    up = ttnn.matmul(
        gathered,
        weights.up_proj,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
        program_config=gate_up_pc,
    )
    down_input = apply_geglu(gate, up)
    gate.deallocate(True)
    up.deallocate(True)
    down = ttnn.matmul(
        down_input,
        weights.down_proj,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
        program_config=down_pc,
    )
    down_input.deallocate(True)
    return down


def sparse_experts_forward(
    experts,
    hidden_states,
    dense_routing,
    capacity=DEFAULT_CAPACITY,
    compute_kernel_config=None,
):
    """True-sparse token-gather expert forward — drop-in for ``moe.experts(hidden, routing)``.

    Args:
        experts: gemma4 Gemma4Experts (source of ``.weights``, ``.config``, ``.mesh_config``,
            ``.ccl_manager``). Not mutated.
        hidden_states: [1, 1, S, H] on device (normed expert input, TP-replicated).
        dense_routing: [1, 1, S, E] on device (router output, ``top_k`` non-zero per token).
        capacity: tokens per expert (multiple of 32).

    Returns:
        [1, 1, S, H] on device — expert output, all-reduced across TP.
    """
    assert capacity % TILE == 0, f"capacity must be a multiple of {TILE}, got {capacity}"
    weights = experts.weights
    cfg = experts.config
    mesh_config = experts.mesh_config
    ccl = experts.ccl_manager
    E = cfg.num_experts
    H = cfg.hidden_size
    S = hidden_states.shape[2]
    C = capacity
    EC = E * C
    ckcfg = compute_kernel_config or default_sparse_moe_compute_kernel_config()

    # OPT-004 tuned matmul geometry (opt-in). None -> the auto-config path (bit-identical prototype).
    tuned = (
        build_tuned_configs(hidden_states.device(), E, C, H, weights.intermediate_size_per_device, S)
        if (tuned_configs_enabled())
        else None
    )

    disp, comb = build_capacity_dispatch(dense_routing, E, C, cfg.top_k)

    # gather: dispatched[EC, H] = disp^T @ hidden  ([1,1,EC,S] @ [1,1,S,H])
    disp_t = ttnn.transpose(disp, 2, 3)  # [1,1,EC,S]
    disp.deallocate(True)
    dispatched = ttnn.matmul(
        disp_t,
        hidden_states,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=ckcfg,
        program_config=(tuned["gather"] if tuned else None),
    )
    disp_t.deallocate(True)
    gathered = ttnn.reshape(dispatched, (1, E, C, H))
    dispatched.deallocate(True)

    # experts (batched matmul over active experts only)
    down = _batched_experts(gathered, weights, ckcfg, program_configs=tuned)  # [1, E, C, H] partial
    gathered.deallocate(True)
    down_flat = ttnn.reshape(down, (1, 1, EC, H))
    down.deallocate(True)

    # combine + route-weight: out[S, H] = comb @ down_flat  ([1,1,S,EC] @ [1,1,EC,H])
    out = ttnn.matmul(
        comb,
        down_flat,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=ckcfg,
        program_config=(tuned["combine"] if tuned else None),
    )
    comb.deallocate(True)
    down_flat.deallocate(True)

    # all-reduce across TP after the row-parallel down projection
    if mesh_config is not None and mesh_config.tp > 1:
        out = ccl_allreduce(out, mesh_config, ccl)
    return out
