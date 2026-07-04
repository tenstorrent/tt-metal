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


def _batched_experts(gathered, weights, compute_kernel_config):
    """Batched gate/up/geglu/down over active experts.

    gathered: [1, E, C, H] — each expert's capacity tokens.
    Returns: [1, E, C, H] partial (TP-sharded down output, pre all-reduce).
    """
    gate = ttnn.matmul(
        gathered, weights.gate_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=compute_kernel_config
    )
    up = ttnn.matmul(
        gathered, weights.up_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=compute_kernel_config
    )
    down_input = apply_geglu(gate, up)
    gate.deallocate(True)
    up.deallocate(True)
    down = ttnn.matmul(
        down_input,
        weights.down_proj,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
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

    disp, comb = build_capacity_dispatch(dense_routing, E, C, cfg.top_k)

    # gather: dispatched[EC, H] = disp^T @ hidden  ([1,1,EC,S] @ [1,1,S,H])
    disp_t = ttnn.transpose(disp, 2, 3)  # [1,1,EC,S]
    disp.deallocate(True)
    dispatched = ttnn.matmul(disp_t, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg)
    disp_t.deallocate(True)
    gathered = ttnn.reshape(dispatched, (1, E, C, H))
    dispatched.deallocate(True)

    # experts (batched matmul over active experts only)
    down = _batched_experts(gathered, weights, ckcfg)  # [1, E, C, H] partial
    gathered.deallocate(True)
    down_flat = ttnn.reshape(down, (1, 1, EC, H))
    down.deallocate(True)

    # combine + route-weight: out[S, H] = comb @ down_flat  ([1,1,S,EC] @ [1,1,EC,H])
    out = ttnn.matmul(comb, down_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg)
    comb.deallocate(True)
    down_flat.deallocate(True)

    # all-reduce across TP after the row-parallel down projection
    if mesh_config is not None and mesh_config.tp > 1:
        out = ccl_allreduce(out, mesh_config, ccl)
    return out
