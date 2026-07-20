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

from dataclasses import dataclass
import os

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.experimental.diffusion_gemma.tt.expert_operations import apply_geglu

TILE = 32
DEFAULT_CAPACITY = 32
_EXPERT_FP32_FULL_SYNC_CFG_CACHE = {}


def default_sparse_moe_compute_kernel_config():
    """HiFi2 matches the dense sparse_matmul numerics (PCC 0.9997 vs dense at bf16).

    DG_SPARSE_MOE_HIFI4 (default off) raises the fidelity to HiFi4 to MATCH the gemma4 dense
    reference (models/demos/gemma4/tt/experts/prefill.py uses HiFi4, fp32_dest_acc_en=False):
    the sparse path's HiFi2 vs the dense HiFi4 is a real contributor to the pcc-vs-dense gap, and
    the expert matmuls are weight-bound so the extra math passes are largely hidden behind the DRAM
    read. fp32_dest_acc_en stays False so the tuned out_subblock (product up to 8) remains legal
    (fp32_dest_acc caps the subblock at 4). Flag off -> byte-identical HiFi2 config."""
    fidelity = ttnn.MathFidelity.HiFi4 if os.environ.get("DG_SPARSE_MOE_HIFI4", "0") != "0" else ttnn.MathFidelity.HiFi2
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def expert_compute_kernel_config(tensor, fallback):
    # Blackhole half-DST only exposes four FP32 tiles and corrupts the 8-tile expert
    # subblocks used here. Full-DST synchronization preserves the FP32 accumulation
    # fidelity without changing the BF16 expert outputs. Keep an escape hatch for
    # architecture/performance bisects.
    arch = tensor.device().arch()
    if os.environ.get("DG_SPARSE_EXPERT_FP32_FULL_SYNC", "0") != "1" or arch != ttnn.Arch.BLACKHOLE:
        return fallback
    key = (id(tensor.device()), arch)
    config = _EXPERT_FP32_FULL_SYNC_CFG_CACHE.get(key)
    if config is None:
        config = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            dst_full_sync_en=True,
        )
        _EXPERT_FP32_FULL_SYNC_CFG_CACHE[key] = config
    return config


# DG_MOE_EXPERT_BFP8 (default OFF): cast the (bf16-on-disk) expert gate/up/down weight banks to
# bfloat8_b ONCE (cached) so the weight-bound batched expert matmuls read ~53% of the bytes
# (~415 MB -> ~220 MB), directly attacking the ~68%-of-runtime expert cost. gemma4 is NOT mutated —
# these are cached bfp8 COPIES built from experts.weights and consumed only inside this module. bfp8
# is lower precision than bf16, so this is pcc-risky against the thin 0.98 floor; gated + measured.
# Flag off -> the bf16 weights are used unchanged (byte-identical).
_BFP8_WEIGHTS_CACHE = {}


def expert_weight_bfp8_enabled():
    return os.environ.get("DG_MOE_EXPERT_BFP8", "0") != "0"


class _Bfp8Weights:
    """Minimal shim exposing gate_proj/up_proj/down_proj as bfp8; other attrs proxy to the original."""

    __slots__ = ("gate_proj", "up_proj", "down_proj", "_orig")

    def __init__(self, gate_proj, up_proj, down_proj, orig):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self._orig = orig

    def __getattr__(self, name):
        return getattr(self._orig, name)


def _bfp8_expert_weights(weights):
    """Cached bfp8_b copies of the expert weight banks (built once per weights object)."""
    key = id(weights)
    shim = _BFP8_WEIGHTS_CACHE.get(key)
    if shim is None:
        shim = _Bfp8Weights(
            ttnn.clone(weights.gate_proj, dtype=ttnn.bfloat8_b),
            ttnn.clone(weights.up_proj, dtype=ttnn.bfloat8_b),
            ttnn.clone(weights.down_proj, dtype=ttnn.bfloat8_b),
            weights,
        )
        _BFP8_WEIGHTS_CACHE[key] = shim
    return shim


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


# ---------------------------------------------------------------------------------------------
# MEASUREMENT-ONLY ablation (DG_MOE_DISPATCH_ABLATE, default off): the ~18-op ``build_capacity_dispatch``
# body is a DEPENDENT chain of small ops (topk -> typecast x2 -> scatter -> cumsum -> sub -> gather ->
# mul/add -> ge -> where -> sub -> mul -> typecast -> scatter x2 -> slice x2) that runs once PER LAYER
# PER STEP. In the traced denoise loop, independent ops overlap but a dependent chain cannot — its
# dispatch latency serializes. To measure that serialized cost as a block-latency delta, this flag
# skips the per-call chain entirely and returns a PERSISTENT constant disp/comb built ONCE (on the
# first, eager, pre-capture call), reused for every layer/step. The gather/expert/combine matmuls
# keep identical shapes (values-only difference), so the block-latency delta = the chain's serialized
# replay cost. Output is intentionally WRONG (fixed routing) — this path is for latency evidence only,
# never a committed change. ``sparse_experts_forward`` must NOT deallocate the returned const (guarded).
# ---------------------------------------------------------------------------------------------
_DISPATCH_ABLATE_CACHE = {}


def _dispatch_ablate_enabled():
    return os.environ.get("DG_MOE_DISPATCH_ABLATE", "0") != "0"


def _ablation_const_dispatch(dense_routing, num_experts, capacity, top_k):
    mesh = dense_routing.device()
    S = dense_routing.shape[2]
    key = (id(mesh), S, num_experts, capacity, top_k)
    consts = _DISPATCH_ABLATE_CACHE.get(key)
    if consts is None:
        # First call is the eager warm step (outside any trace) -> build the real disp/comb once and
        # keep them as persistent buffers reused for every subsequent layer/step call.
        consts = _build_capacity_dispatch_impl(dense_routing, num_experts, capacity, top_k)
        _DISPATCH_ABLATE_CACHE[key] = consts
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
    if _dispatch_ablate_enabled():
        return _ablation_const_dispatch(dense_routing, num_experts, capacity, top_k)
    if _dispatch_fused2_enabled():
        return _build_capacity_dispatch_fused2(dense_routing, num_experts, capacity, top_k)
    if _dispatch_fused_enabled():
        return _build_capacity_dispatch_fused(dense_routing, num_experts, capacity, top_k)
    return _build_capacity_dispatch_impl(dense_routing, num_experts, capacity, top_k)


def _dispatch_fused_enabled():
    """DG_MOE_DISPATCH_FUSED (default OFF): shorter, BIT-IDENTICAL dispatch-build chain.

    Step 1 (bench_dispatch_ablation.py) measured the full dependent chain at ~0.9 ms/layer/step
    (12.7% of the @16 block), so shortening it is a real perf lever. This variant removes 4 ops from
    the ~19-op chain WITHOUT changing the kept-column [0:EC] disp/comb values (verify_dispatch_fused.py
    checks bit-exactness incl. capacity overflow):
      * drop ``idx_u`` typecast — topk's uint16 ``idx`` is a legal scatter/gather index directly;
      * drop ``mask_f`` typecast + the [S,E] ``excl`` sub — the exclusive slot is
        ``gather(cum, idx) - 1`` (cum is inclusive; at an active expert cum = excl + 1), one [S,k] sub;
      * drop ``valid`` + ``vals_valid`` — overflow columns are scattered into the dead column EC and
        then sliced off, so pre-zeroing the dropped combine weight is redundant with the slice.
    Default off so the shipped path stays byte-identical until device-validated bit-identical."""
    return os.environ.get("DG_MOE_DISPATCH_FUSED", "0") != "0"


def _build_capacity_dispatch_fused(dense_routing, num_experts, capacity, top_k):
    """Shorter, bit-identical variant of ``_build_capacity_dispatch_impl`` (see ``_dispatch_fused_enabled``)."""
    mesh = dense_routing.device()
    S = dense_routing.shape[2]
    EC = num_experts * capacity
    k = _get_dispatch_constants(mesh, S, num_experts, capacity, top_k)
    ones_sk, zeros_se, zeros_d, zeros_c, dead = (k["ones_sk"], k["zeros_se"], k["zeros_d"], k["zeros_c"], k["dead"])

    # 1. per-token active experts + their routing weights. topk's uint16 idx is a legal
    #    scatter/gather index directly (no uint32 typecast needed).
    vals, idx = ttnn.topk(dense_routing, k=top_k, dim=-1)  # vals [1,1,S,k] bf16, idx uint16
    idx_f = ttnn.typecast(idx, ttnn.float32)

    # 2. active mask [1,1,S,E] + inclusive cumsum over the token dim.
    mask = ttnn.scatter(zeros_se, dim=-1, index=idx, src=ones_sk)
    cum = ttnn.cumsum(mask, dim=2, dtype=ttnn.float32)  # inclusive count of same-expert tokens 0..t

    # 3. exclusive slot per (token, active-expert): at an active expert cum = exclusive + 1, so the
    #    slot is gather(cum, idx) - 1 (one [S,k] sub; no [S,E] mask_f typecast + [S,E] sub).
    pos = ttnn.sub(ttnn.gather(cum, dim=-1, index=idx), 1.0)  # [1,1,S,k] f32

    # 4. dispatch column = e*C + slot; overflow (slot >= C) -> dead column EC (dropped).
    col = ttnn.add(ttnn.mul(idx_f, float(capacity)), pos)  # f32 exact up to 2^24 >> EC
    overflow = ttnn.ge(pos, float(capacity))
    col = ttnn.where(overflow, dead, col)
    col_u = ttnn.typecast(col, ttnn.uint32)

    # 5. scatter into the E*C dispatch buffers (+1 dead column, then sliced off). Overflow rows land
    #    in column EC for BOTH masks and are removed by the slice, so comb scatters vals directly
    #    (no valid/vals_valid pre-zeroing) — the kept columns [0:EC] are identical to the impl path.
    disp = ttnn.scatter(zeros_d, dim=-1, index=col_u, src=ones_sk)
    comb = ttnn.scatter(zeros_c, dim=-1, index=col_u, src=vals)
    disp = ttnn.slice(disp, [0, 0, 0, 0], [1, 1, S, EC])
    comb = ttnn.slice(comb, [0, 0, 0, 0], [1, 1, S, EC])

    for t in (vals, idx, idx_f, mask, cum, pos, col, overflow, col_u):
        t.deallocate(True)
    return disp, comb


# ---------------------------------------------------------------------------------------------
# DG_MOE_DISPATCH_FUSED2 (default OFF): a custom C++ device kernel (``dispatch_build.cpp``, run via
# ``ttnn.generic_op``) that fuses the ENTIRE post-cumsum tail of ``_build_capacity_dispatch_impl``
# into ONE dispatched device op. Step-1 ablation proved skipping the whole build chain drops the @16
# block 3.46->3.02s (12.7%); the removable Python ops were off the critical path, so the *serialized*
# cost lives in the dependent tail: gather(cum,idx) -> sub -> mul/add -> ge -> where -> typecast ->
# scatter(disp) -> scatter(comb) -> slice -> slice. Each ttnn.scatter/gather additionally round-trips
# TILE<->ROW_MAJOR internally, so that tail is many serialized device ops. This variant keeps only
# topk -> scatter(mask) -> cumsum on the critical path and replaces the tail with a single kernel that
# gathers the per-expert count, computes the capacity column, and scatters ones/weights into disp/comb.
# BIT-IDENTICAL to the impl kept columns [0:EC] (verify_dispatch_fused.py). Kernel is JIT-compiled by
# generic_op (no _ttnncpp.so change), so the C++ tree stays byte-identical when the flag is off.
# ---------------------------------------------------------------------------------------------

_FUSED2_KERNEL = "ttnn/cpp/ttnn/operations/data_movement/moe_dispatch_build/device/kernels/dispatch_build.cpp"
_FUSED2_PLAN_CACHE = {}


def _dispatch_fused2_enabled():
    # Default ON: bit-identical to the impl on the kept [0:EC] columns (verify_dispatch_fused.py),
    # denoise-only (S=256/E=128/C=256/top_k=8 — the sole DG shape; prefill uses the ragged path),
    # device-proven +2.3% @48 / +5.2% @12 with matching committed_sha (bench_lever_e2e frozen A/B).
    # Set DG_MOE_DISPATCH_FUSED2=0 to fall back to the pure-ttnn dispatch tail.
    return os.environ.get("DG_MOE_DISPATCH_FUSED2", "1") != "0"


def _round_up(x, m):
    return ((x + m - 1) // m) * m


def _get_fused2_plan(mesh, S, num_experts, capacity, top_k):
    """Static (routing-value-independent) op geometry: per-core token bands + page sizes. Cached."""
    key = (id(mesh), S, num_experts, capacity, top_k)
    plan = _FUSED2_PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    EC = num_experts * capacity
    grid = mesh.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    cores = ttnn.corerange_to_cores(all_cores, row_wise=True)
    num_cores = min(len(cores), S)
    rows_per_core = (S + num_cores - 1) // num_cores
    active = []
    r = 0
    for c in cores:
        start = min(r, S)
        end = min(r + rows_per_core, S)
        if end > start:
            active.append((c, start, end))
        r = end
    active_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for (c, _, _) in active])
    # read sizes = logical row bytes; CB pages rounded up to 32B for L1 validity (no over-read).
    plan = {
        "EC": EC,
        "active": active,
        "core_set": active_core_set,
        "cum_read": num_experts * 4,
        "idx_read": top_k * 4,
        "vals_read": top_k * 2,
        "out_read": EC * 2,
        "cum_cb": _round_up(num_experts * 4, 32),
        "idx_cb": _round_up(top_k * 4, 32),
        "vals_cb": _round_up(top_k * 2, 32),
        "out_cb": _round_up(EC * 2, 32),
    }
    _FUSED2_PLAN_CACHE[key] = plan
    return plan


def _build_fused2_program(plan, cum_rm, idx_rm, vals_rm, disp_rm, comb_rm, num_experts, capacity, top_k):
    cs = plan["core_set"]

    def cb(idx, fmt, page):
        return ttnn.CBDescriptor(
            total_size=page,
            core_ranges=cs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
        )

    cbs = [
        cb(0, ttnn.float32, plan["cum_cb"]),
        cb(1, ttnn.uint32, plan["idx_cb"]),
        cb(2, ttnn.bfloat16, plan["vals_cb"]),
        cb(3, ttnn.bfloat16, plan["out_cb"]),
        cb(4, ttnn.bfloat16, plan["out_cb"]),
    ]

    ct = [top_k, capacity, plan["EC"], plan["cum_read"], plan["idx_read"], plan["vals_read"], plan["out_read"]]
    for tsr in (cum_rm, idx_rm, vals_rm, disp_rm, comb_rm):
        ct.extend(ttnn.TensorAccessorArgs(tsr).get_compile_time_args())

    rt = ttnn.RuntimeArgs()
    addrs = [
        cum_rm.buffer_address(),
        idx_rm.buffer_address(),
        vals_rm.buffer_address(),
        disp_rm.buffer_address(),
        comb_rm.buffer_address(),
    ]
    for core, start, end in plan["active"]:
        rt[core.x][core.y] = addrs + [start, end]

    kernel = ttnn.KernelDescriptor(
        kernel_source=_FUSED2_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cs,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)


def _build_capacity_dispatch_fused2(dense_routing, num_experts, capacity, top_k):
    """Custom-kernel variant of ``_build_capacity_dispatch_impl`` (see ``_dispatch_fused2_enabled``)."""
    mesh = dense_routing.device()
    S = dense_routing.shape[2]
    EC = num_experts * capacity
    plan = _get_fused2_plan(mesh, S, num_experts, capacity, top_k)
    consts = _get_dispatch_constants(mesh, S, num_experts, capacity, top_k)
    ones_sk, zeros_se = consts["ones_sk"], consts["zeros_se"]

    # critical-path prefix kept as-is: topk -> mask scatter -> inclusive cumsum over tokens
    vals, idx = ttnn.topk(dense_routing, k=top_k, dim=-1)  # vals [1,1,S,k] bf16, idx uint16
    mask = ttnn.scatter(zeros_se, dim=-1, index=idx, src=ones_sk)
    cum = ttnn.cumsum(mask, dim=2, dtype=ttnn.float32)  # inclusive count 0..t of same-expert tokens

    idx_u = ttnn.typecast(idx, ttnn.uint32)
    cum_rm = ttnn.to_layout(cum, ttnn.ROW_MAJOR_LAYOUT)
    idx_rm = ttnn.to_layout(idx_u, ttnn.ROW_MAJOR_LAYOUT)
    vals_rm = ttnn.to_layout(vals, ttnn.ROW_MAJOR_LAYOUT)

    disp_rm = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, S, EC]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    comb_rm = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, S, EC]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )

    prog = _build_fused2_program(plan, cum_rm, idx_rm, vals_rm, disp_rm, comb_rm, num_experts, capacity, top_k)
    ttnn.generic_op([cum_rm, idx_rm, vals_rm, disp_rm, comb_rm], prog)

    disp = ttnn.to_layout(disp_rm, ttnn.TILE_LAYOUT)
    comb = ttnn.to_layout(comb_rm, ttnn.TILE_LAYOUT)

    for t in (vals, idx, idx_u, mask, cum, cum_rm, idx_rm, vals_rm, disp_rm, comb_rm):
        t.deallocate(True)
    return disp, comb


def _build_capacity_dispatch_impl(dense_routing, num_experts, capacity, top_k):
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
    """OPT-004 tuned program configs are ON by default (3.47x full-MoE forward vs auto-config, PCC
    0.99967 vs untuned, trace-safe). The auto-config gate/up matmul is ~13x slower (~17 GB/s vs the
    tuned ~235 GB/s ≈ 92% of the weight-traffic roofline), so a run that forgets the flag would
    silently take the slow path. Set ``DG_SPARSE_MOE_TUNED=0`` to force the byte-identical fallback."""
    return os.environ.get("DG_SPARSE_MOE_TUNED", "1") != "0"


# ---------------------------------------------------------------------------------------------
# L1-residency (dg-08 L1 pass): keep the MoE token-gather activation intermediates L1-resident
# across an op boundary instead of round-tripping DRAM. Two self-contained levers:
#   HIGH-1 (gather): the gather matmul writes ``dispatched`` [1,1,EC,H] = 23.1 MB to DRAM, then
#       gate (:296) and up (:303) re-read it (46 MB). Pin it L1 so gate/up read from L1.
#   HIGH-2 (down):   the down matmul writes ``down`` [1,E,C,H] = 23.1 MB to DRAM, then the combine
#       matmul (:385) re-reads ``down_flat`` (23 MB) as its in1. Pin it L1 so combine reads from L1.
# The expert gate/up/down matmuls themselves are WEIGHT-BOUND (~138 MB weight read each, ~92% of the
# 256 GB/s roofline; the ~1.5 MB gate/up outputs are ~1% of their traffic), so MED-5 (L1 gate/up
# outputs) is expected to be ~a no-op and is bundled under mode ``chain`` only for measurement.
# Default OFF -> the path is bit-identical to the DRAM prototype until measured PCC-clean.
# ``both``/``all`` combine the levers. ``out`` (combine output) always stays DRAM: it feeds the
# gemma4 ``ccl_allreduce`` (out-of-gate; MED-7). All matmul math (dtype/fidelity/program_config) is
# unchanged -> a pure placement change; PCC must stay at the DRAM value.
# ---------------------------------------------------------------------------------------------


def moe_l1_mode():
    """DG_MOE_L1 selects which MoE activation intermediates stay L1-resident (opt-in, default off).

    Modes: ``off`` (DRAM, current default) | ``gather`` (HIGH-1) | ``down`` (HIGH-2) |
    ``chain`` (MED-5 gate/up outputs, expected no-op) | ``both`` (gather+down) | ``all``.
    """
    return os.environ.get("DG_MOE_L1", "off").lower()


def _l1_or_dram(use_l1):
    return ttnn.L1_MEMORY_CONFIG if use_l1 else ttnn.DRAM_MEMORY_CONFIG


def fused_gather_enabled():
    """DG_MOE_FUSED_GATHER (fused-MoE increment 3, default OFF): route the expert gate/up matmul
    through a sparse_matmul whose in0 reader GATHERS each expert's token rows directly from
    ``hidden`` (via a per-(expert,slot) gather index built on-device from ``build_capacity_dispatch``
    machinery), deleting the ``disp^T @ hidden`` gather matmul and the [EC,H] materialization.

    NOT yet implemented on the kernel side: the sparse in0 reader only carries the compile gate +
    hook (TTNN_SPARSE_MATMUL_IN0_GATHER -> SPARSE_MATMUL_IN0_GATHER, identity fallback), and the
    ``gather_index`` op input does not exist yet. Enabling this flag therefore raises until the
    kernel gather lands, so no run silently produces wrong output. See
    doc/optimize_perf/fused_moe_kernel.md (increment 3) for the remaining steps and the A/B command.
    """
    return os.environ.get("DG_MOE_FUSED_GATHER", "0") != "0"


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


def _batched_experts(
    gathered,
    weights,
    compute_kernel_config,
    program_configs=None,
    l1_gate_up=False,
    l1_down=False,
):
    """Batched gate/up/geglu/down over active experts.

    gathered: [1, E, C, H] — each expert's capacity tokens.
    program_configs: optional dict with ``gate_up`` / ``down`` OPT-004 program configs; None keeps the
        auto-config path (bit-identical to the untuned prototype).
    l1_gate_up / l1_down: L1-residency levers (see ``moe_l1_mode``). Pure output-placement change.
    Returns: [1, E, C, H] partial (TP-sharded down output, pre all-reduce).
    """
    gate_up_pc = program_configs.get("gate_up") if program_configs else None
    down_pc = program_configs.get("down") if program_configs else None
    gate = ttnn.matmul(
        gathered,
        weights.gate_proj,
        memory_config=_l1_or_dram(l1_gate_up),
        compute_kernel_config=compute_kernel_config,
        program_config=gate_up_pc,
    )
    up = ttnn.matmul(
        gathered,
        weights.up_proj,
        memory_config=_l1_or_dram(l1_gate_up),
        compute_kernel_config=compute_kernel_config,
        program_config=gate_up_pc,
    )
    down_input = apply_geglu(gate, up)
    gate.deallocate(True)
    up.deallocate(True)
    down = ttnn.matmul(
        down_input,
        weights.down_proj,
        memory_config=_l1_or_dram(l1_down),
        compute_kernel_config=compute_kernel_config,
        program_config=down_pc,
    )
    down_input.deallocate(True)
    return down


RAGGED_MAX_M_BLOCKS = 4

# Default token-dim chunk length for long-prompt ragged prefill (see
# ``chunked_ragged_sparse_prefill_forward``). Matches the QB2-validated single-call ceiling.
RAGGED_PREFILL_CHUNK = 4096


@dataclass
class RaggedRouting:
    values: object
    indices: object
    per_expert_scale: object | None


@dataclass
class CompactRaggedMetadata:
    """Fixed-shape device metadata consumed by the traced denoise MoE."""

    slot_token: object
    token_slot: object
    route_weight: object
    sparsity: object
    dense_segment_map: object
    dense_column: object
    max_segments: int
    segment_rows: int


_COMPACT_RAGGED_KERNEL = "models/experimental/diffusion_gemma/tt/kernels/compact_ragged_dispatch.cpp"
_COMPACT_DOWN_TO_DENSE_KERNEL = "models/experimental/diffusion_gemma/tt/kernels/compact_down_to_dense.cpp"
_COMPACT_DENSE_COMBINE_KERNEL = "models/experimental/diffusion_gemma/tt/kernels/compact_dense_combine.cpp"
_COMPACT_RAGGED_PLAN_CACHE = {}
_COMPACT_RAGGED_SCALE_CACHE = {}
_COMPACT_DOWN_TO_DENSE_PLAN_CACHE = {}


def compact_ragged_denoise_enabled():
    """Trace-safe zero-drop ragged denoise MoE selector."""
    return os.environ.get("DG_DENOISE_COMPACT_RAGGED", "0").lower() not in ("0", "false", "no", "off")


def compact_ragged_segment_rows():
    rows = int(os.environ.get("DG_COMPACT_SEGMENT_ROWS", str(TILE)))
    if rows < TILE or rows % TILE != 0:
        raise ValueError(f"DG_COMPACT_SEGMENT_ROWS must be a positive multiple of {TILE}, got {rows}")
    return rows


def compact_ragged_max_segments(sequence_length, num_experts, top_k, segment_rows=TILE):
    """Return a static upper bound for 32-row expert segments."""
    assignments = sequence_length * top_k
    active = min(num_experts, assignments)
    if segment_rows >= sequence_length:
        bound = active
    else:
        bound = active + (assignments - active) // segment_rows
    return _round_up(bound, TILE)


def _compact_scale_row_major(mesh, scale, num_experts):
    key = (id(mesh), id(scale), num_experts)
    cached = _COMPACT_RAGGED_SCALE_CACHE.get(key)
    if cached is not None:
        return cached
    if scale is None:
        cached = ttnn.ones(
            [1, 1, 1, num_experts],
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
        )
    else:
        cached = ttnn.to_layout(scale, ttnn.ROW_MAJOR_LAYOUT)
    _COMPACT_RAGGED_SCALE_CACHE[key] = cached
    return cached


def _compact_ragged_plan(mesh, sequence_length, num_experts, top_k):
    segment_rows = compact_ragged_segment_rows()
    max_segments = compact_ragged_max_segments(sequence_length, num_experts, top_k, segment_rows)
    key = (id(mesh), sequence_length, num_experts, top_k, segment_rows, max_segments)
    cached = _COMPACT_RAGGED_PLAN_CACHE.get(key)
    if cached is not None:
        return cached
    core = ttnn.CoreCoord(0, 0)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    cached = {
        "core": core,
        "core_set": core_set,
        "segment_rows": segment_rows,
        "max_segments": max_segments,
        "max_rows": max_segments * segment_rows,
        "indices_read": top_k * 4,
        "values_read": top_k * 2,
        "scale_read": num_experts * 2,
        "indices_cb_page": _round_up(top_k * 4 + 63, 64),
        "values_cb_page": _round_up(top_k * 2 + 63, 64),
        "scale_cb_page": _round_up(num_experts * 2 + 63, 64),
        "slot_token_write": segment_rows * 4,
        "slot_token_cb_page": _round_up(segment_rows * 4 + 63, 64),
        "token_slot_write": sequence_length * 4,
        "token_slot_cb_page": _round_up(sequence_length * 4 + 63, 64),
        "route_weight_write": sequence_length * 2,
        "route_weight_cb_page": _round_up(sequence_length * 2 + 63, 64),
        "sparsity_write": num_experts * 2,
        "sparsity_cb_page": _round_up(num_experts * 2 + 63, 64),
        "dense_tile_rows": num_experts * (sequence_length // TILE),
        "dense_map_write": num_experts * (sequence_length // TILE) * 4,
        "dense_map_cb_page": _round_up(num_experts * (sequence_length // TILE) * 4 + 63, 64),
    }
    _COMPACT_RAGGED_PLAN_CACHE[key] = cached
    return cached


def _build_compact_ragged_program(
    plan,
    indices_rm,
    values_rm,
    scale_rm,
    slot_token,
    token_slot,
    route_weight,
    sparsity,
    dense_segment_map,
    dense_column,
    sequence_length,
    num_experts,
    top_k,
):
    core_set = plan["core_set"]

    def cb(index, dtype, page_size, total_size=None):
        return ttnn.CBDescriptor(
            total_size=page_size if total_size is None else total_size,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    cbs = [
        cb(0, ttnn.uint32, plan["indices_cb_page"], sequence_length * plan["indices_cb_page"]),
        cb(1, ttnn.bfloat16, plan["values_cb_page"], sequence_length * plan["values_cb_page"]),
        cb(2, ttnn.bfloat16, plan["scale_cb_page"]),
        cb(
            3,
            ttnn.uint32,
            plan["slot_token_cb_page"],
            plan["max_segments"] * plan["slot_token_cb_page"],
        ),
        cb(4, ttnn.uint32, plan["token_slot_cb_page"], top_k * plan["token_slot_cb_page"]),
        cb(5, ttnn.bfloat16, plan["route_weight_cb_page"], top_k * plan["route_weight_cb_page"]),
        cb(6, ttnn.bfloat16, plan["sparsity_cb_page"]),
        cb(7, ttnn.uint32, plan["dense_map_cb_page"]),
        cb(8, ttnn.uint32, plan["token_slot_cb_page"], top_k * plan["token_slot_cb_page"]),
    ]
    compile_args = [
        sequence_length,
        num_experts,
        top_k,
        plan["max_segments"],
        plan["segment_rows"],
        plan["indices_read"],
        plan["values_read"],
        plan["scale_read"],
        plan["slot_token_write"],
        plan["token_slot_write"],
        plan["route_weight_write"],
        plan["sparsity_write"],
        plan["indices_cb_page"],
        plan["values_cb_page"],
        plan["slot_token_cb_page"],
        plan["token_slot_cb_page"],
        plan["route_weight_cb_page"],
        plan["sparsity_cb_page"],
        plan["dense_map_write"],
        plan["dense_map_cb_page"],
    ]
    for tensor in (
        indices_rm,
        values_rm,
        scale_rm,
        slot_token,
        token_slot,
        route_weight,
        sparsity,
        dense_segment_map,
        dense_column,
    ):
        compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())

    runtime_args = ttnn.RuntimeArgs()
    core = plan["core"]
    runtime_args[core.x][core.y] = [
        indices_rm.buffer_address(),
        values_rm.buffer_address(),
        scale_rm.buffer_address(),
        slot_token.buffer_address(),
        token_slot.buffer_address(),
        route_weight.buffer_address(),
        sparsity.buffer_address(),
        dense_segment_map.buffer_address(),
        dense_column.buffer_address(),
    ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=_COMPACT_RAGGED_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=compile_args,
        runtime_args=runtime_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)


def _compact_ragged_metadata(routing, num_experts):
    """Build fixed-shape compact assignment metadata entirely on device."""
    values = routing.values
    indices = routing.indices
    mesh = values.device()
    sequence_length = values.shape[2]
    top_k = values.shape[3]
    plan = _compact_ragged_plan(mesh, sequence_length, num_experts, top_k)

    indices_u32 = ttnn.typecast(indices, ttnn.uint32)
    indices_rm = ttnn.to_layout(indices_u32, ttnn.ROW_MAJOR_LAYOUT)
    values_rm = ttnn.to_layout(values, ttnn.ROW_MAJOR_LAYOUT)
    scale_rm = _compact_scale_row_major(mesh, routing.per_expert_scale, num_experts)
    slot_token = ttnn.allocate_tensor_on_device(
        ttnn.Shape([plan["max_segments"], plan["segment_rows"]]),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    token_slot = ttnn.allocate_tensor_on_device(
        ttnn.Shape([top_k, sequence_length]),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    route_weight = ttnn.allocate_tensor_on_device(
        ttnn.Shape([top_k, sequence_length]),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    sparsity = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, plan["max_segments"], num_experts]),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    dense_segment_map = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, plan["dense_tile_rows"]]),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    dense_column = ttnn.allocate_tensor_on_device(
        ttnn.Shape([top_k, sequence_length]),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    program = _build_compact_ragged_program(
        plan,
        indices_rm,
        values_rm,
        scale_rm,
        slot_token,
        token_slot,
        route_weight,
        sparsity,
        dense_segment_map,
        dense_column,
        sequence_length,
        num_experts,
        top_k,
    )
    ttnn.generic_op(
        [
            indices_rm,
            values_rm,
            scale_rm,
            slot_token,
            token_slot,
            route_weight,
            sparsity,
            dense_segment_map,
            dense_column,
        ],
        program,
    )
    for tensor in (values, indices, indices_u32, indices_rm, values_rm):
        tensor.deallocate(True)
    return CompactRaggedMetadata(
        slot_token=slot_token,
        token_slot=token_slot,
        route_weight=route_weight,
        sparsity=sparsity,
        dense_segment_map=dense_segment_map,
        dense_column=dense_column,
        max_segments=plan["max_segments"],
        segment_rows=plan["segment_rows"],
    )


def _compact_down_to_dense_plan(mesh, num_experts, sequence_length, hidden_size):
    key = (id(mesh), num_experts, sequence_length, hidden_size)
    cached = _COMPACT_DOWN_TO_DENSE_PLAN_CACHE.get(key)
    if cached is not None:
        return cached
    grid = mesh.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    cores = ttnn.corerange_to_cores(all_cores, row_wise=True)
    hidden_tiles = hidden_size // TILE
    total_tiles = num_experts * (sequence_length // TILE) * hidden_tiles
    tiles_per_core = (total_tiles + len(cores) - 1) // len(cores)
    work = []
    start = 0
    for core in cores:
        end = min(total_tiles, start + tiles_per_core)
        if start < end:
            work.append((core, start, end))
        start = end
    cached = {
        "core_set": ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, _, _ in work]),
        "work": work,
        "hidden_tiles": hidden_tiles,
        "tile_bytes": TILE * TILE * 2,
    }
    _COMPACT_DOWN_TO_DENSE_PLAN_CACHE[key] = cached
    return cached


def _build_compact_dense_combine(dense_column, route_weight, num_experts):
    mesh = dense_column.device()
    top_k = dense_column.shape[0]
    sequence_length = dense_column.shape[1]
    output_rm = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, sequence_length, num_experts * sequence_length]),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    grid = mesh.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    cores = ttnn.corerange_to_cores(all_cores, row_wise=True)
    tokens_per_core = (sequence_length + len(cores) - 1) // len(cores)
    work = []
    start = 0
    for core in cores:
        end = min(sequence_length, start + tokens_per_core)
        if start < end:
            work.append((core, start, end))
        start = end
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, _, _ in work])
    column_read = sequence_length * 4
    weight_read = sequence_length * 2
    output_write = num_experts * sequence_length * 2
    column_page = _round_up(column_read + 63, 64)
    weight_page = _round_up(weight_read + 63, 64)
    output_page = _round_up(output_write + 63, 64)

    def cb(index, dtype, page):
        return ttnn.CBDescriptor(
            total_size=page,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    compile_args = [
        sequence_length,
        num_experts,
        top_k,
        column_read,
        weight_read,
        output_write,
        column_page,
        weight_page,
    ]
    for tensor in (dense_column, route_weight, output_rm):
        compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    runtime_args = ttnn.RuntimeArgs()
    for core, start, end in work:
        runtime_args[core.x][core.y] = [
            dense_column.buffer_address(),
            route_weight.buffer_address(),
            output_rm.buffer_address(),
            start,
            end,
        ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=_COMPACT_DENSE_COMBINE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=compile_args,
        runtime_args=runtime_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(
        [dense_column, route_weight, output_rm],
        ttnn.ProgramDescriptor(
            kernels=[kernel],
            semaphores=[],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=top_k * column_page,
                    core_ranges=core_set,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=column_page)
                    ],
                ),
                ttnn.CBDescriptor(
                    total_size=top_k * weight_page,
                    core_ranges=core_set,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=weight_page)
                    ],
                ),
                cb(2, ttnn.bfloat16, output_page),
            ],
        ),
    )
    output = ttnn.to_layout(output_rm, ttnn.TILE_LAYOUT)
    output_rm.deallocate(True)
    return output


def _scatter_compact_down_to_dense(packed_down, dense_segment_map, num_experts, sequence_length, hidden_size):
    mesh = packed_down.device()
    plan = _compact_down_to_dense_plan(mesh, num_experts, sequence_length, hidden_size)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, num_experts * sequence_length, hidden_size]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        mesh,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    map_bytes = num_experts * (sequence_length // TILE) * 4
    map_page = _round_up(map_bytes + 63, 64)
    tile_page = _round_up(plan["tile_bytes"] + 63, 64)

    def cb(index, dtype, page_size):
        return ttnn.CBDescriptor(
            total_size=page_size,
            core_ranges=plan["core_set"],
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    compile_args = [plan["hidden_tiles"], plan["tile_bytes"], map_bytes, map_page]
    for tensor in (dense_segment_map, packed_down, output):
        compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    runtime_args = ttnn.RuntimeArgs()
    for core, start, end in plan["work"]:
        runtime_args[core.x][core.y] = [
            dense_segment_map.buffer_address(),
            packed_down.buffer_address(),
            output.buffer_address(),
            start,
            end,
        ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=_COMPACT_DOWN_TO_DENSE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=plan["core_set"],
        compile_time_args=compile_args,
        runtime_args=runtime_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(
        [dense_segment_map, packed_down, output],
        ttnn.ProgramDescriptor(
            kernels=[kernel],
            semaphores=[],
            cbs=[
                cb(0, ttnn.uint32, map_page),
                cb(1, ttnn.bfloat16, tile_page),
                cb(2, ttnn.bfloat16, tile_page),
            ],
        ),
    )
    return output


_ROUTER_SCALE_HOST_CACHE = {}


def ragged_router_forward(router, hidden_states):
    """Router forward that retains compact top-k metadata instead of scattering dense."""
    normed = router.norm.forward(hidden_states)
    scaled = ttnn.mul(normed, router.scale)
    normed.deallocate(True)
    scaled = ttnn.mul(scaled, router.scalar_root_size)
    expert_scores = ttnn.linear(scaled, router.proj_weight)
    scaled.deallocate(True)
    router_probs = ttnn.softmax(expert_scores, dim=-1)
    expert_scores.deallocate(True)
    top_k_values, top_k_indices = ttnn.topk(router_probs, k=router.top_k, dim=-1)
    router_probs.deallocate(True)
    top_k_sum = ttnn.sum(top_k_values, dim=-1, keepdim=True)
    normalized_values = ttnn.div(top_k_values, top_k_sum)
    top_k_values.deallocate(True)
    top_k_sum.deallocate(True)
    return RaggedRouting(normalized_values, top_k_indices, router.per_expert_scale)


def _compact_primary_program_configs(mesh, m_blocks, intermediate_tiles, hidden_tiles):
    """C=32 geometry that is elementwise exact to the C=256 dense expert path.

    A real-weight sweep found the reduction-compatible K blocks to be 8 for
    gate/up and 2 for down.  The prior OPT-004 gate/up block 22 changes BF16
    accumulation order and is the main compact-path fidelity regression.
    """
    gx, gy = _device_grid(mesh)
    gate_sh, gate_sw = _pick_out_subblock(m_blocks, intermediate_tiles)
    down_sh, down_sw = _pick_out_subblock(m_blocks, hidden_tiles)
    return {
        "gate_up": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=8,
            out_subblock_h=gate_sh,
            out_subblock_w=gate_sw,
            per_core_M=m_blocks,
            per_core_N=intermediate_tiles,
        ),
        "down": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=2,
            out_subblock_h=down_sh,
            out_subblock_w=down_sw,
            per_core_M=m_blocks,
            per_core_N=hidden_tiles,
        ),
    }


def _compact_overflow_program_config(m_blocks, output_width):
    """Sparse overflow geometry with the same K reduction blocks as dense C=256."""
    if output_width == 192:
        grid_x, grid_y, block_w, per_core_n = 6, 1, 8, 1
    elif output_width == 2816:
        grid_x, grid_y, block_w, per_core_n = 11, 4, 2, 2
    else:
        raise ValueError(f"unsupported compact overflow output width: {output_width}")
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=m_blocks,
        out_block_w=per_core_n,
        per_core_M=m_blocks,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def compact_ragged_denoise_forward(experts, hidden_states, routing, compute_kernel_config=None):
    """Static-shape, zero-drop ragged MoE for the traced denoise canvas."""
    weights = experts.weights
    config = experts.config
    mesh_config = experts.mesh_config
    ccl_manager = experts.ccl_manager
    mesh = hidden_states.device()
    sequence_length = hidden_states.shape[2]
    num_experts = config.num_experts
    hidden_size = config.hidden_size
    intermediate = weights.intermediate_size_per_device
    top_k = config.top_k
    if tuple(routing.values.shape) != (1, 1, sequence_length, top_k):
        raise ValueError(
            f"compact denoise routing shape {routing.values.shape} does not match "
            f"(1, 1, {sequence_length}, {top_k})"
        )

    # Dense-compatible combine is the selected compact mode: it preserves the
    # baseline BF16 matmul reduction and therefore the committed trajectory.
    # ``fast_reduce`` remains a diagnostic speed ceiling but is decision-inexact.
    combine_mode = os.environ.get("DG_COMPACT_COMBINE", "dense_compat").lower()
    dense_comb = None
    metadata = _compact_ragged_metadata(routing, num_experts)
    if combine_mode == "dense_compat":
        dense_comb = _build_compact_dense_combine(metadata.dense_column, metadata.route_weight, num_experts)
    max_segments = metadata.max_segments
    segment_rows = metadata.segment_rows
    max_rows = max_segments * segment_rows
    kernel_config = compute_kernel_config or default_sparse_moe_compute_kernel_config()
    hidden_flat = ttnn.reshape(hidden_states, (sequence_length, hidden_size))
    gathered = ttnn.embedding(
        metadata.slot_token,
        hidden_flat,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grouped_input = ttnn.reshape(gathered, (1, max_segments, segment_rows, hidden_size))
    primary_input = ttnn.slice(
        grouped_input,
        [0, 0, 0, 0],
        [1, num_experts, segment_rows, hidden_size],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    primary_configs = (
        _compact_primary_program_configs(
            mesh,
            segment_rows // TILE,
            intermediate // TILE,
            hidden_size // TILE,
        )
        if segment_rows == DEFAULT_CAPACITY
        and os.environ.get("DG_COMPACT_PRIMARY_TUNED", "1").lower() not in ("0", "false", "no", "off")
        else None
    )
    primary_down = _batched_experts(
        primary_input,
        weights,
        kernel_config,
        program_configs=primary_configs,
    )
    primary_flat = ttnn.reshape(primary_down, (num_experts * segment_rows, hidden_size))

    overflow_segments = max_segments - num_experts
    overflow_tensors = []
    if overflow_segments > 0:
        overflow_input = ttnn.slice(
            grouped_input,
            [0, num_experts, 0, 0],
            [1, max_segments, segment_rows, hidden_size],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        overflow_sparsity = ttnn.slice(
            metadata.sparsity,
            [0, 0, num_experts, 0],
            [1, 1, max_segments, num_experts],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate_output = ttnn.empty(
            [1, overflow_segments, segment_rows, intermediate],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        up_output = ttnn.empty(
            [1, overflow_segments, segment_rows, intermediate],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        common = {
            "sparsity": overflow_sparsity,
            "nnz": overflow_segments,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "compute_kernel_config": kernel_config,
            "dtype": ttnn.bfloat16,
        }
        gate = ttnn.sparse_matmul(
            overflow_input,
            weights.gate_proj,
            program_config=_compact_overflow_program_config(segment_rows // TILE, intermediate),
            optional_output_tensor=gate_output,
            **common,
        )
        up = ttnn.sparse_matmul(
            overflow_input,
            weights.up_proj,
            program_config=_compact_overflow_program_config(segment_rows // TILE, intermediate),
            optional_output_tensor=up_output,
            **common,
        )
        overflow_down_input = apply_geglu(gate, up)
        gate.deallocate(True)
        up.deallocate(True)
        down_output = ttnn.empty(
            [1, overflow_segments, segment_rows, hidden_size],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        overflow_down = ttnn.sparse_matmul(
            overflow_down_input,
            weights.down_proj,
            program_config=_compact_overflow_program_config(segment_rows // TILE, hidden_size),
            optional_output_tensor=down_output,
            **common,
        )
        overflow_flat = ttnn.reshape(overflow_down, (overflow_segments * segment_rows, hidden_size))
        packed_down = ttnn.concat([primary_flat, overflow_flat], dim=0)
        overflow_tensors = [overflow_input, overflow_sparsity, overflow_down_input, overflow_flat]
    else:
        packed_down = primary_flat

    combine_tensors = []
    if combine_mode == "dense_compat":
        dense_down = _scatter_compact_down_to_dense(
            packed_down,
            metadata.dense_segment_map,
            num_experts,
            sequence_length,
            hidden_size,
        )
        out = ttnn.matmul(dense_comb, dense_down, compute_kernel_config=kernel_config)
        combine_tensors = [dense_comb, dense_down]
    elif combine_mode == "matmul":
        token_slot = ttnn.permute(metadata.token_slot, (1, 0))
        selected = ttnn.embedding(
            token_slot,
            packed_down,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        selected = ttnn.reshape(selected, (1, sequence_length, top_k, hidden_size))
        route_weight = ttnn.permute(metadata.route_weight, (1, 0))
        route_weight = ttnn.reshape(route_weight, (1, sequence_length, 1, top_k))
        route_weight_tile = ttnn.to_layout(route_weight, ttnn.TILE_LAYOUT)
        out = ttnn.matmul(route_weight_tile, selected, compute_kernel_config=kernel_config)
        out = ttnn.reshape(out, (1, 1, sequence_length, hidden_size))
        combine_tensors = [token_slot, selected, route_weight, route_weight_tile]
    elif combine_mode == "fast_reduce":
        selected = ttnn.embedding(
            metadata.token_slot,
            packed_down,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        selected = ttnn.reshape(selected, (1, top_k, sequence_length, hidden_size))
        route_weight = ttnn.reshape(metadata.route_weight, (1, 1, top_k, sequence_length))
        route_weight = ttnn.permute(route_weight, (0, 2, 3, 1))
        weighted = ttnn.mul(selected, route_weight)
        out = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(weighted, dims=[1]))
        out = ttnn.reshape(out, (1, 1, sequence_length, hidden_size))
        combine_tensors = [selected, route_weight, weighted]
    else:
        raise ValueError(f"unsupported DG_COMPACT_COMBINE mode: {combine_mode}")

    for tensor in (
        metadata.slot_token,
        metadata.token_slot,
        metadata.route_weight,
        metadata.sparsity,
        metadata.dense_segment_map,
        metadata.dense_column,
        gathered,
        primary_input,
        *combine_tensors,
        *overflow_tensors,
    ):
        tensor.deallocate(True)
    if packed_down is not primary_flat:
        packed_down.deallocate(True)
    primary_flat.deallocate(True)
    if mesh_config is not None and mesh_config.tp > 1:
        out = ccl_allreduce(out, mesh_config, ccl_manager)
    return out


try:
    import numba as _numba
except ImportError:  # pragma: no cover - exercised in minimal runtime environments
    _numba = None


if _numba is not None:
    import numpy as np

    @_numba.njit(cache=True)
    def _pack_ragged_assignments(expert_index, num_experts, max_m_blocks):
        sequence_length, top_k = expert_index.shape
        capacity_rows = max_m_blocks * TILE
        max_segments = (sequence_length + capacity_rows - 1) // capacity_rows
        counts = np.zeros(num_experts, np.int32)
        for token in range(sequence_length):
            for k_index in range(top_k):
                counts[expert_index[token, k_index]] += 1

        segment_m_blocks = np.zeros((num_experts, max_segments), np.int32)
        group_counts = np.zeros(max_m_blocks, np.int32)
        for expert in range(num_experts):
            num_segments = (counts[expert] + capacity_rows - 1) // capacity_rows
            for segment in range(num_segments):
                count = min(capacity_rows, counts[expert] - segment * capacity_rows)
                m_blocks = (count + TILE - 1) // TILE
                segment_m_blocks[expert, segment] = m_blocks
                group_counts[m_blocks - 1] += 1

        group_start = np.zeros(max_m_blocks, np.int32)
        total_rows = 0
        for m_blocks in range(1, max_m_blocks + 1):
            group_start[m_blocks - 1] = total_rows
            total_rows += group_counts[m_blocks - 1] * m_blocks * TILE

        group_experts = np.full((max_m_blocks, num_experts * max_segments), -1, np.int32)
        segment_local = np.zeros((num_experts, max_segments), np.int32)
        local_counts = np.zeros(max_m_blocks, np.int32)
        for expert in range(num_experts):
            for segment in range(max_segments):
                m_blocks = segment_m_blocks[expert, segment]
                if m_blocks != 0:
                    local = local_counts[m_blocks - 1]
                    local_counts[m_blocks - 1] += 1
                    segment_local[expert, segment] = local
                    group_experts[m_blocks - 1, local] = expert

        slot_token = np.zeros(total_rows, np.int32)
        slot_valid_bits = np.zeros(total_rows, np.uint16)
        token_slot = np.empty((sequence_length, top_k), np.int32)
        expert_rank = np.zeros(num_experts, np.int32)
        for token in range(sequence_length):
            for k_index in range(top_k):
                expert = expert_index[token, k_index]
                rank = expert_rank[expert]
                expert_rank[expert] += 1
                segment = rank // capacity_rows
                row = rank % capacity_rows
                m_blocks = segment_m_blocks[expert, segment]
                packed_row = group_start[m_blocks - 1] + segment_local[expert, segment] * m_blocks * TILE + row
                slot_token[packed_row] = token
                slot_valid_bits[packed_row] = 0x3F80  # BF16 1.0
                token_slot[token, k_index] = packed_row
        return slot_token, slot_valid_bits, token_slot, group_counts, group_experts, group_start

else:
    _pack_ragged_assignments = None


def _ragged_prefill_program_config(m_blocks, output_width):
    if output_width == 192:
        grid_x, grid_y, block_w, per_core_n = 6, 1, 44, 1
    elif output_width == 2816:
        grid_x, grid_y, block_w, per_core_n = 11, 4, 3, 2
    else:
        raise ValueError(f"unsupported ragged prefill output width: {output_width}")
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=m_blocks,
        out_block_w=per_core_n,
        per_core_M=m_blocks,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _ragged_metadata_host(dense_routing, num_experts, top_k, max_m_blocks=RAGGED_MAX_M_BLOCKS):
    """Pack routed assignments into zero-drop, expert-homogeneous tile groups.

    This CPU metadata builder is intentionally vectorized: the previous
    assignment-by-assignment Python prototype took ~39 ms for 1024 tokens,
    while this implementation takes <1 ms on the same host.
    """
    import torch

    if isinstance(dense_routing, RaggedRouting):
        values = dense_routing.values
        indices = dense_routing.indices
        if values.device().get_num_devices() > 1:
            route_weight = ttnn.to_torch(ttnn.get_device_tensors(values)[0])[0, 0]
            expert_index = ttnn.to_torch(ttnn.get_device_tensors(indices)[0])[0, 0].long()
        else:
            route_weight = ttnn.to_torch(values)[0, 0]
            expert_index = ttnn.to_torch(indices)[0, 0].long()
        values.deallocate(True)
        indices.deallocate(True)
        S = route_weight.shape[0]
        per_token_order = torch.argsort(expert_index, dim=-1)
        expert_index = torch.gather(expert_index, -1, per_token_order)
        route_weight = torch.gather(route_weight, -1, per_token_order)
        if dense_routing.per_expert_scale is not None:
            scale_tensor = dense_routing.per_expert_scale
            scale = _ROUTER_SCALE_HOST_CACHE.get(id(scale_tensor))
            if scale is None:
                if scale_tensor.device().get_num_devices() > 1:
                    scale = ttnn.to_torch(ttnn.get_device_tensors(scale_tensor)[0]).reshape(-1)
                else:
                    scale = ttnn.to_torch(scale_tensor).reshape(-1)
                _ROUTER_SCALE_HOST_CACHE[id(scale_tensor)] = scale
            route_weight = route_weight * scale[expert_index]
    else:
        if dense_routing.device().get_num_devices() > 1:
            routing = ttnn.to_torch(ttnn.get_device_tensors(dense_routing)[0])[0, 0]
        else:
            routing = ttnn.to_torch(dense_routing)[0, 0]
        S = routing.shape[0]
        active_mask = routing != 0
        active_entries = torch.nonzero(active_mask)
        if active_entries.shape[0] == S * top_k:
            # The router contract is exactly top_k nonzero entries per token.
            # nonzero() is row-major, so expert ids are already in the reduction
            # order that matches the dense path (QB2-verified bit-identical: the
            # host-side per-expert scale rounds the same as the on-device bf16
            # multiply, so logits + KV match the shared path exactly).
            expert_index = active_entries[:, -1].reshape(S, top_k)
            route_weight = routing[active_mask].reshape(S, top_k)
        else:
            # Defensive fallback for a future router that can emit exact zero for
            # an active slot or otherwise violates the fixed-top-k contract.
            route_weight, expert_index = torch.topk(routing, top_k, dim=-1)
            per_token_order = torch.argsort(expert_index, dim=-1)
            expert_index = torch.gather(expert_index, -1, per_token_order)
            route_weight = torch.gather(route_weight, -1, per_token_order)

    capacity_rows = max_m_blocks * TILE
    max_segments_per_expert = (S + capacity_rows - 1) // capacity_rows

    if _pack_ragged_assignments is not None:
        (
            slot_token_np,
            slot_valid_bits_np,
            token_slot_np,
            group_counts_np,
            group_experts_np,
            group_start_np,
        ) = _pack_ragged_assignments(
            expert_index.contiguous().numpy(),
            num_experts,
            max_m_blocks,
        )
        groups = []
        for m_blocks in range(1, max_m_blocks + 1):
            group_size = int(group_counts_np[m_blocks - 1])
            if group_size == 0:
                continue
            start = int(group_start_np[m_blocks - 1])
            total_rows = group_size * m_blocks * TILE
            slot_token = torch.from_numpy(slot_token_np[start : start + total_rows].copy())
            slot_valid_bits = torch.from_numpy(slot_valid_bits_np[start : start + total_rows].copy())
            slot_valid = slot_valid_bits.view(torch.bfloat16).reshape(total_rows, 1)
            group_experts = torch.from_numpy(group_experts_np[m_blocks - 1, :group_size].copy()).long()
            sparsity = torch.zeros((1, 1, group_size, num_experts), dtype=torch.bfloat16)
            sparsity[0, 0, torch.arange(group_size), group_experts] = 1
            groups.append((m_blocks, group_size, slot_token, slot_valid, sparsity))
        return (
            groups,
            torch.from_numpy(token_slot_np.copy()),
            route_weight.reshape(S, top_k, 1),
            len(slot_token_np),
        )

    flat_expert = expert_index.reshape(-1)
    flat_token = torch.arange(S).repeat_interleave(top_k)
    flat_k = torch.arange(top_k).repeat(S)
    assignment_order = torch.argsort(flat_expert, stable=True)
    sorted_expert = flat_expert[assignment_order]
    sorted_token = flat_token[assignment_order]
    sorted_k = flat_k[assignment_order]

    expert_counts = torch.bincount(sorted_expert, minlength=num_experts)
    expert_starts = torch.cumsum(expert_counts, 0) - expert_counts
    rank_in_expert = torch.arange(S * top_k) - expert_starts[sorted_expert]
    segment_key = sorted_expert * max_segments_per_expert + rank_in_expert // capacity_rows
    segment_keys, segment_counts = torch.unique_consecutive(segment_key, return_counts=True)
    assignment_segment = torch.repeat_interleave(torch.arange(len(segment_keys)), segment_counts)
    row_in_segment = rank_in_expert % capacity_rows
    segment_m_blocks = (segment_counts + TILE - 1) // TILE

    token_slot = torch.empty((S, top_k), dtype=torch.int32)
    groups = []
    output_offset = 0
    for m_blocks in range(1, max_m_blocks + 1):
        segment_ids = torch.nonzero(segment_m_blocks == m_blocks).flatten()
        if len(segment_ids) == 0:
            continue
        group_size = len(segment_ids)
        rows_per_segment = m_blocks * TILE
        total_rows = group_size * rows_per_segment
        segment_to_group = torch.full((len(segment_keys),), -1, dtype=torch.int64)
        segment_to_group[segment_ids] = torch.arange(group_size)
        assignment_mask = segment_to_group[assignment_segment] >= 0
        packed_row = (
            segment_to_group[assignment_segment[assignment_mask]] * rows_per_segment + row_in_segment[assignment_mask]
        )

        slot_token = torch.zeros(total_rows, dtype=torch.int32)
        slot_valid = torch.zeros((total_rows, 1), dtype=torch.bfloat16)
        slot_token[packed_row] = sorted_token[assignment_mask].to(torch.int32)
        slot_valid[packed_row] = 1
        token_slot[sorted_token[assignment_mask], sorted_k[assignment_mask]] = output_offset + packed_row.to(
            torch.int32
        )

        group_experts = segment_keys[segment_ids] // max_segments_per_expert
        sparsity = torch.zeros((1, 1, group_size, num_experts), dtype=torch.bfloat16)
        sparsity[0, 0, torch.arange(group_size), group_experts] = 1
        groups.append((m_blocks, group_size, slot_token, slot_valid, sparsity))
        output_offset += total_rows

    return groups, token_slot, route_weight.reshape(S, top_k, 1), output_offset


def ragged_sparse_prefill_forward(
    hidden_states,
    routing_weights,
    weights,
    config,
    prefill_sparsity,
    mesh_config=None,
    mesh_device=None,
    ccl_manager=None,
):
    """Zero-drop sparse prefill with compact ragged expert batches.

    QB2-verified bit-identical to the shared 128-expert path (26B-A4B, 30 layers,
    prompts up to 2048 incl. multi-segment packing: logits + KV cache max_abs=0),
    at 26-57x lower prefill latency.
    """
    del prefill_sparsity
    mesh = mesh_device or hidden_states.device()
    S = hidden_states.shape[2]
    E = config.num_experts
    H = config.hidden_size
    I = weights.intermediate_size_per_device
    max_m_blocks = int(os.environ.get("DG_PREFILL_RAGGED_M_BLOCKS", RAGGED_MAX_M_BLOCKS))
    groups, token_slot_host, route_weight_host, packed_rows = _ragged_metadata_host(
        routing_weights, E, config.top_k, max_m_blocks=max_m_blocks
    )
    mapper = ttnn.ReplicateTensorToMesh(mesh) if hasattr(mesh, "shape") else None

    def upload(host_tensor, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            host_tensor,
            dtype=dtype,
            layout=layout,
            device=mesh,
            mesh_mapper=mapper,
        )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    hidden_flat = ttnn.reshape(hidden_states, (S, H))
    down_groups = []
    for m_blocks, group_size, slot_token_host, slot_valid_host, sparsity_host in groups:
        group_rows = group_size * m_blocks * TILE
        slot_token = upload(slot_token_host.reshape(1, group_rows), ttnn.uint32)
        slot_valid = upload(slot_valid_host.reshape(1, group_rows, 1), ttnn.bfloat16)
        sparsity = upload(sparsity_host, ttnn.bfloat16)

        gathered = ttnn.embedding(
            slot_token,
            hidden_flat,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gathered_valid = ttnn.mul(gathered, slot_valid)
        grouped_input = ttnn.reshape(gathered_valid, (1, group_size, m_blocks * TILE, H))
        gate_output = ttnn.empty(
            [1, group_size, m_blocks * TILE, I],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        up_output = ttnn.empty(
            [1, group_size, m_blocks * TILE, I],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        common = {
            "sparsity": sparsity,
            "nnz": group_size,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "compute_kernel_config": compute_kernel_config,
            "dtype": ttnn.bfloat16,
        }
        gate = ttnn.sparse_matmul(
            grouped_input,
            weights.gate_proj,
            program_config=_ragged_prefill_program_config(m_blocks, I),
            optional_output_tensor=gate_output,
            **common,
        )
        up = ttnn.sparse_matmul(
            grouped_input,
            weights.up_proj,
            program_config=_ragged_prefill_program_config(m_blocks, I),
            optional_output_tensor=up_output,
            **common,
        )
        down_input = apply_geglu(gate, up)
        down_output = ttnn.empty(
            [1, group_size, m_blocks * TILE, H],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        down = ttnn.sparse_matmul(
            down_input,
            weights.down_proj,
            program_config=_ragged_prefill_program_config(m_blocks, H),
            optional_output_tensor=down_output,
            **common,
        )
        down_groups.append(ttnn.reshape(down, (group_rows, H)))

        for tensor in (
            slot_token,
            slot_valid,
            sparsity,
            gathered,
            gathered_valid,
            gate,
            up,
            down_input,
        ):
            tensor.deallocate(True)

    packed_down = down_groups[0] if len(down_groups) == 1 else ttnn.concat(down_groups, dim=0)
    if len(down_groups) > 1:
        for tensor in down_groups:
            tensor.deallocate(True)
    assert packed_down.shape[0] == packed_rows

    # Embedding accepts a 2-D index matrix. Store top-k in the leading
    # dimension so fast_reduce_nc can consume it directly without a device
    # permute of the large [S, K, H] selected-expert tensor.
    token_slot = upload(token_slot_host.transpose(0, 1).contiguous(), ttnn.uint32)
    route_weight_transposed = route_weight_host.transpose(0, 1).contiguous()
    route_weight = upload(route_weight_transposed, ttnn.bfloat16)
    selected = ttnn.embedding(
        token_slot,
        packed_down,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    weighted = ttnn.mul(selected, route_weight)
    weighted = ttnn.reshape(weighted, (1, config.top_k, S, H))
    out = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(weighted, dims=[1]))
    out = ttnn.reshape(out, (1, 1, S, H))

    for tensor in (packed_down, token_slot, route_weight, selected, weighted):
        tensor.deallocate(True)
    if mesh_config is not None and mesh_config.tp > 1:
        out = ccl_allreduce(out, mesh_config, ccl_manager)
    return out


def ragged_prefill_chunk_size():
    """Token-dim chunk length for long-prompt ragged prefill (``DG_PREFILL_RAGGED_CHUNK``).

    Defaults to ``RAGGED_PREFILL_CHUNK`` (4096), the largest single-call length the ragged path
    has been exercised at. Must be a positive multiple of the tile height so every chunk (and the
    32-multiple-padded tail) is a legal ragged prefill shape."""
    raw = os.environ.get("DG_PREFILL_RAGGED_CHUNK")
    if raw is None or not raw.strip():
        return RAGGED_PREFILL_CHUNK
    value = int(raw)
    if value <= 0 or value % TILE != 0:
        raise ValueError(f"DG_PREFILL_RAGGED_CHUNK must be a positive multiple of {TILE}, got {value}")
    return value


def chunked_ragged_sparse_prefill_forward(
    hidden_states,
    routing_weights,
    weights,
    config,
    prefill_sparsity,
    mesh_config=None,
    mesh_device=None,
    ccl_manager=None,
):
    """Ragged sparse prefill for prompts longer than one chunk.

    MoE is per-token, so a long prefill is processed in ``ragged_prefill_chunk_size()``-token
    slices along the sequence dim: the full-S ``RaggedRouting`` (computed once by the router hook)
    and ``hidden_states`` are sliced by the same boundaries, each slice runs the QB2-validated
    ``ragged_sparse_prefill_forward`` UNCHANGED (including its per-slice TP all-reduce), and the
    per-chunk ``[1, 1, chunk, H]`` outputs are concatenated on the token dim. This is bit-identical
    to a single full-S ragged call — the router (RMSNorm/softmax/top-k) is strictly per-token so a
    sliced ``RaggedRouting`` equals per-token routing, the ragged FFN is per-token, and TP all-reduce
    is per-element (grouping tokens into chunks cannot change any element). It also keeps every
    intermediate bounded to the single-chunk footprint (the [top_k, S, H] combine reduction and the
    ``top_k*S*H`` index volumes both scale with the chunk length, not the full context), which is
    what lets prefill scale past the ~64K DRAM / ~128K int32-index limits of a single full-S call.

    Drop-in for ``ragged_sparse_prefill_forward`` (identical signature). For ``S <= chunk`` — or a
    non-``RaggedRouting`` argument — it delegates straight through, so the single-chunk path is
    byte-for-byte the validated behavior.
    """
    S = hidden_states.shape[2]
    chunk = ragged_prefill_chunk_size()
    if S <= chunk or not isinstance(routing_weights, RaggedRouting):
        return ragged_sparse_prefill_forward(
            hidden_states,
            routing_weights,
            weights,
            config,
            prefill_sparsity,
            mesh_config=mesh_config,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

    values = routing_weights.values  # [1, 1, S, top_k]
    indices = routing_weights.indices  # [1, 1, S, top_k]
    scale = routing_weights.per_expert_scale  # [1, 1, 1, E] — shared across chunks, NOT sliced
    top_k = values.shape[-1]
    H = hidden_states.shape[3]

    chunk_outputs = []
    for start in range(0, S, chunk):
        end = min(start + chunk, S)  # start is chunk-aligned; S is a 32-multiple upstream
        hidden_chunk = ttnn.slice(hidden_states, [0, 0, start, 0], [1, 1, end, H])
        values_chunk = ttnn.slice(values, [0, 0, start, 0], [1, 1, end, top_k])
        indices_chunk = ttnn.slice(indices, [0, 0, start, 0], [1, 1, end, top_k])
        routing_chunk = RaggedRouting(values_chunk, indices_chunk, scale)
        # ragged_sparse_prefill_forward deallocates values_chunk/indices_chunk (its RaggedRouting
        # input) inside _ragged_metadata_host; it never touches hidden_chunk, so we free that here.
        chunk_outputs.append(
            ragged_sparse_prefill_forward(
                hidden_chunk,
                routing_chunk,
                weights,
                config,
                prefill_sparsity,
                mesh_config=mesh_config,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
            )
        )
        hidden_chunk.deallocate(True)

    values.deallocate(True)
    indices.deallocate(True)
    out = chunk_outputs[0] if len(chunk_outputs) == 1 else ttnn.concat(chunk_outputs, dim=2)
    if len(chunk_outputs) > 1:
        for tensor in chunk_outputs:
            tensor.deallocate(True)
    return out


def sparse_experts_forward(
    experts,
    hidden_states,
    dense_routing,
    capacity=None,
    compute_kernel_config=None,
):
    """True-sparse token-gather expert forward — drop-in for ``moe.experts(hidden, routing)``.

    Args:
        experts: gemma4 Gemma4Experts (source of ``.weights``, ``.config``, ``.mesh_config``,
            ``.ccl_manager``). Not mutated.
        hidden_states: [1, 1, S, H] on device (normed expert input, TP-replicated).
        dense_routing: [1, 1, S, E] on device (router output, ``top_k`` non-zero per token).
        capacity: tokens per expert (multiple of 32); defaults to the canvas
            length so no routed token can be dropped.

    Returns:
        [1, 1, S, H] on device — expert output, all-reduced across TP.
    """
    capacity = hidden_states.shape[2] if capacity is None else capacity
    assert capacity % TILE == 0, f"capacity must be a multiple of {TILE}, got {capacity}"
    if fused_gather_enabled():
        # Increment-3 integration point (see fused_gather_enabled / doc). The in-reader gather is not
        # implemented yet (kernel scaffold only), so fail loudly rather than run the identity-fallback
        # gather and return wrong output. Wiring (once the gather + gather_index op input land):
        # build gather_index[E,C] on-device from the build_capacity_dispatch col/slot machinery
        # (trace-safe, no host round-trip), then replace `disp^T @ hidden` + `_batched_experts` gate/up
        # with sparse_matmul(hidden, w_gate, gather_index=..., sparsity=..., nnz=...).
        raise NotImplementedError(
            "DG_MOE_FUSED_GATHER: in-reader gather kernel not implemented yet (increment-3 scaffold "
            "only). See models/experimental/diffusion_gemma/doc/optimize_perf/fused_moe_kernel.md."
        )
    weights = _bfp8_expert_weights(experts.weights) if expert_weight_bfp8_enabled() else experts.weights
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
        if tuned_configs_enabled() and C == DEFAULT_CAPACITY
        else None
    )

    # L1-residency levers (dg-08 L1 pass; opt-in via DG_MOE_L1, default off -> bit-identical DRAM path)
    mode = moe_l1_mode()
    l1_gather = mode in ("gather", "both", "all")
    l1_down = mode in ("down", "both", "all")
    l1_gate_up = mode in ("chain", "all")

    # DG_MOE_DISPATCH_ABLATE returns a persistent constant disp/comb (measurement-only); it must be
    # reused across layers/steps, so the two deallocates below are guarded off in that mode.
    ablate_dispatch = _dispatch_ablate_enabled()
    disp, comb = build_capacity_dispatch(dense_routing, E, C, cfg.top_k)

    # gather: dispatched[EC, H] = disp^T @ hidden  ([1,1,EC,S] @ [1,1,S,H])
    disp_t = ttnn.transpose(disp, 2, 3)  # [1,1,EC,S]
    if not ablate_dispatch:
        disp.deallocate(True)
    dispatched = ttnn.matmul(
        disp_t,
        hidden_states,
        memory_config=_l1_or_dram(l1_gather),
        compute_kernel_config=ckcfg,
        program_config=(tuned["gather"] if tuned else None),
    )
    disp_t.deallocate(True)
    gathered = ttnn.reshape(dispatched, (1, E, C, H))
    dispatched.deallocate(True)

    # experts (batched matmul over active experts only)
    down = _batched_experts(
        gathered,
        weights,
        expert_compute_kernel_config(gathered, ckcfg),
        program_configs=tuned,
        l1_gate_up=l1_gate_up,
        l1_down=l1_down,
    )  # [1, E, C, H] partial
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
    if not ablate_dispatch:
        comb.deallocate(True)
    down_flat.deallocate(True)

    # all-reduce across TP after the row-parallel down projection
    if mesh_config is not None and mesh_config.tp > 1:
        out = ccl_allreduce(out, mesh_config, ccl)
    return out
