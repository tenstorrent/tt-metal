# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""HOST_ALLOC (P.15, tt/settings.py; owner 2026-10-30, DS41-171): fewer buffer allocations a chunk, the same bytes.

A warm 8x4 chunk made 372 ttnn.allocate_tensor_on_device calls (~84 us of host each, 31 ms a chunk) and as many
frees. This module removes the ones whose buffer can be one load-time buffer:

1. The in-step fabric all_gathers (tt/ccl.py ``all_gather(site=)``): the mHC row exchange (tt/mhc.py fused, 2 a layer),
   the split-norm statistics (tt/norm.py _fork, 1 a layer) and the attention's window kv over TP (tt/attention.py
   _window_packed, 1 a layer) write one persistent output per (site, spec), allocated HERE AT LOAD TIME
   (``reserve_sites``, from the model's shapes) and never freed (ccl.dealloc skips them). Never mid-forward: a buffer
   allocated in a forward and kept shifts later steps' addresses (DS41-146). A call whose output spec has no load-time
   buffer (another chunk length, a box whose shapes this table does not cover) allocates as before (counted,
   ``ccl.SITE_MISSES``). Reuse is safe: each site's result is read and dropped inside its step, each chip runs its
   programs in order, and fabric_all_gather's ready fence keeps a chip from writing a neighbour's output before that
   neighbour started the same call (P.4, the persistent window gathers of P.11 use the same rule).
2. The block MoE (models/demos/moe_block, outside this model's paths: installed on the built block's instances, as
   P.14's prepared_tail):
   - the exchange checks with ``arena_fits`` that its L1 arena would land where its programs baked it, by allocating
     the arena and freeing it, and its call then allocates the same arena again (2 L1 allocations of ~150-180 us a
     layer). ``KeptProbeK3`` keeps the probe as the call's arena (the same address by construction): one allocation.
   - the fused tail (tt/prepared_tail.ShapeKeyedTailA2A, PREPARED_ALL=on) writes a load-time output [S/SP, H/TP]
     instead of allocating one a call (``reserve_tail_out`` from TtMoEBlockLayer's constructor); the model frees the
     MoE output after the layer's mhc_post, so one buffer serves every layer.
3. Fixed-shape intermediates of the attention (ATTN_SP32 on: the chip's own S/(SP TP) query rows), each read and
   freed inside its step: q_a's bfloat8_b input and its projection p (freed by the kv step), _q_stock's scaled q_resid
   and wq_b's output, _out_proj's wo_a input and output (``out`` / ``mm_out``: the op's output_tensor /
   optional_output_tensor), and the window carry from the served cache (the 128 local rows and the owner's 4 gathered
   blocks: ttnn.slice(output_tensor=)). An op writing a given output saves its allocation and the free (8x4,
   tt/perf_host_alloc.py: linear 67 -> 31 us, typecast 52 -> 21, multiply 89 -> 56, slice 171 -> 132).

Every load-time tensor is pinned (ccl.pin: ccl.dealloc returns on its id), and the call sites hand on that object
itself, never the op's new wrapper of its buffer (DS41-149).
"""

from __future__ import annotations

import types

import ttnn

from . import ccl
from .cores import worker_grid
from .settings import settings

_TAIL_OUT: dict = {}  # (id(mesh), chunk) -> (mesh, the fused tail's load-time output)


def on() -> bool:
    return settings.get("HOST_ALLOC") == "on"


def norm_stats_width(mesh, rows: int, cols: int) -> int:
    """ttnn.bringup.rms_norm(stats_out=True)'s output width for [rows, cols] input (32 B): the fork's split plan
    (rms_norm_split_device_operation.cpp split_plan, MODE_STATS_OUT): B <= 2 dividing the column tiles, the fewest
    tile reads on the busiest core, the first on a tie."""
    g = worker_grid(mesh)  # FIX.15: the fork runs on the capped grid (MAX_WORKER_CORES)
    nc = max(1, g.x * g.y)
    ht, wt = rows // 32, cols // 32
    best, bb = None, 1
    for b in range(1, min(2, wt) + 1):
        if wt % b:
            continue
        cost = -(-(ht * b) // nc) * (wt // b)
        if best is None or cost < best:
            best, bb = cost, b
    return 32 * bb


def reserve_sites(model) -> None:
    """Load time (TtDS41Model's constructor): the persistent outputs of the in-step gathers for the model's chunk."""
    if not on() or not ccl.fabric():
        return
    mesh, cfg = model.mesh, model.cfg
    sp, tp = tuple(mesh.shape)
    if tp == 1:
        return
    rows = model.chunk // sp  # the chip's rows of the chunk (S/SP)
    f32, bf16, tile = ttnn.float32, ttnn.bfloat16, ttnn.TILE_LAYOUT
    # mHC row exchange (MHC_FUSED=on, GLUE_FOLD=on): the chips' fp32 partial rows [1, 1, S/SP, 32] gathered on dim 1
    ccl.reserve(mesh, "mhc.row", [1, tp, rows, 32], f32, tile)
    # the column-split norms' partial sums of squares [1, 1, S/SP, 32 B] gathered on dim 3
    w = norm_stats_width(mesh, rows, cfg.dim // tp)
    ccl.reserve(mesh, "norm.stats", [1, 1, rows, w * tp], f32, tile)
    # the attention's window kv rows (ATTN_SP32=on: the chip's own S/(SP TP) rows) gathered over TP to the SP row's W
    ccl.reserve(mesh, "attn.kv_tp", [1, 1, rows, 512], bf16, tile)
    _reserve_attention(model)


def _reserve_attention(model) -> None:
    """The attention's fixed-shape intermediates (ATTN_SP32_PROJ / ATTN_SP32 on: the chip's own S/(SP TP) query rows),
    each read and freed inside its step (tt/attention.py _q_stock / _out_proj, tt/q_a.py): one load-time output a
    site for every layer, written in place (output_tensor / optional_output_tensor). Shapes from a block's weights."""
    from .common import act_dtype, bf16_activations, gemm_dtype

    mesh = model.mesh
    blk = next((b for b in model.blocks.values() if getattr(b, "attention", None) is not None), None)
    if blk is None:
        return
    att, proj = blk.attention, blk.q_a.proj
    if not (att.sp32 and proj.full_rows):
        return
    q = model.tables.geo.q_rows
    gdt, tile = gemm_dtype(), ttnn.TILE_LAYOUT
    # q_a: attn_norm's rows as wq_a | wkv's bfloat8_b input (GEMM_INPUTS), the fused projection's output p (the kv
    # step frees it)
    k, n = int(proj.weight.shape[-2]), int(proj.width)
    ccl.reserve(mesh, "q_a.xq", [1, 1, q, k], gdt, tile)
    ccl.reserve(mesh, "q_a.p", [1, 1, q, n], act_dtype(), tile)
    # _q_stock: q_resid x the softmax scale as wq_b's input, wq_b's bf16 output
    ccl.reserve(mesh, "attn.xq", [1, 1, q, int(att.wq_b.shape[-2])], gdt, tile)
    ccl.reserve(mesh, "attn.qb", [1, 1, q, int(att.wq_b.shape[-1])], ttnn.bfloat16, tile)
    # _out_proj: the grouped heads as wo_a's input, wo_a's output
    ccl.reserve(mesh, "attn.oa", [1, att.g_l, q, att.hpg * att.d], gdt, tile)
    ydt = gdt if (att.glue and bf16_activations()) else ttnn.float32
    ccl.reserve(mesh, "attn.wo_a", [1, att.g_l, q, int(att.wo_a.shape[-1])], ydt, tile)
    # _window_packed's carry from the served cache (CKV_PACK_BEFORE_GATHER, carry_owner): the 128 local rows it packs,
    # and the owner's 4 gathered blocks before their slice_write
    from .attention import WINDOW, WINDOW_BLOCK_BYTES

    rm = ttnn.ROW_MAJOR_LAYOUT
    if model.tables.carry_local() is not None or model.tables.region_index is not None:  # (+ bfp8_tiles, P.18)
        ccl.reserve(mesh, "attn.carry_rows", [1, 1, WINDOW, 512], ttnn.bfloat16, rm)
    if model.tables.carry_local() is not None:
        ccl.reserve(mesh, "attn.carry_mine", [1, 1, WINDOW // 32, WINDOW_BLOCK_BYTES], ttnn.uint8, rm)


def out(mesh, site: str, shape, dtype, layout=ttnn.TILE_LAYOUT):
    """The load-time output of ``site`` for this spec, or None (HOST_ALLOC off, or no such spec: allocate as before)."""
    if not ccl._SITES:
        return None
    return ccl._site_out(mesh, site, [int(d) for d in shape], dtype, layout)


def mm_out(mesh, site: str, x, w, dtype):
    """``out`` for a matmul of x [.., M, K] by w [.., K, N]: the spec [*x.shape[:-1], N]."""
    if not ccl._SITES:
        return None
    s = list(x.shape)
    s[-1] = int(w.shape[-1])
    return ccl._site_out(mesh, site, s, dtype, ttnn.TILE_LAYOUT)


# ---------------------------------------------------------------------------------------------------------------------
# the block MoE
# ---------------------------------------------------------------------------------------------------------------------


def reserve_tail_out(mesh, chunk: int, hidden: int) -> None:
    """Load time (TtMoEBlockLayer): the fused tail's output [1, 1, S/SP, H/TP] bf16 TILE, one for every layer."""
    if not on() or settings.get("PREPARED_ALL") != "on":
        return
    key = (id(mesh), int(chunk))
    hit = _TAIL_OUT.get(key)
    if hit is not None and hit[0] is mesh:
        return
    sp, tp = tuple(mesh.shape)
    t = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, chunk // sp, hidden // tp]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    ccl.pin(mesh, t)
    _TAIL_OUT[key] = (mesh, t)


def tail_out(mesh, chunk: int):
    hit = _TAIL_OUT.get((id(mesh), int(chunk)))
    return hit[1] if hit is not None and hit[0] is mesh else None


def _alloc_arena_kept(self, zero=True):
    """TtK3TokenSplitMoE.alloc_arena (models/demos/deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py) with the arena the
    last ``arena_fits`` probe kept, when there is one; the rest of its body unchanged."""
    from models.demos.deepseek_v3_d_p.tt.moe import tt_k3_token_split_moe as k3

    arena = self.__dict__.pop("_kept_probe", None)
    if arena is None:
        return type(self).alloc_arena(self, zero)
    assert (
        int(arena.buffer_address()) == self.arena_addr
    ), f"the arena landed at {int(arena.buffer_address())}, not {self.arena_addr}: an L1 tensor outlives its op"
    if self.desc is None:
        self.desc, self.l1_args = self.combine.build_l1(self.y, self.out, arena, getattr(self, "_disp", None))
        self._desc_arena = arena
    elif not self.routed_only:
        self.desc = self.combine.rebind(arena)
    if zero:
        k3.prepared_generic_op([arena, self.words], self.zero_desc)
    return arena


def _arena_fits_kept(self):
    """TtK3TokenSplitMoE.arena_fits keeping the probe (it is the arena the call allocates next, at the same address)."""
    old = self.__dict__.pop("_kept_probe", None)
    if old is not None:  # (a probe no call took: give it back first, as the original probe would have)
        ttnn.deallocate(old)
    arena = ttnn.allocate_tensor_on_device(*self._arena_spec)
    ok = int(arena.buffer_address()) == self.arena_addr
    if ok:
        self._kept_probe = arena
    else:
        ttnn.deallocate(arena)
    return ok


def install_moe(blk) -> None:
    """HOST_ALLOC=on: the built MoEBlock's exchange keeps its arena probe (instance methods; the class is unchanged)."""
    if not on():
        return
    k3 = getattr(getattr(blk, "exchange", None), "block", None)
    if k3 is None or not hasattr(k3, "_arena_spec") or not hasattr(k3, "arena_addr"):
        return
    k3.arena_fits = types.MethodType(_arena_fits_kept, k3)
    k3.alloc_arena = types.MethodType(_alloc_arena_kept, k3)


# ---------------------------------------------------------------------------------------------------------------------
# FORK_OUT_BUFFERS (P.16): the bring-up forks' outputs
# ---------------------------------------------------------------------------------------------------------------------
#
# ttnn.bringup.rms_norm / mhc_pre_xing / mhc_post (stats variant) / rotary_embedding_indexed take the caller's output
# buffers (fork options output_tensor, hc_output / y_output, stats_output / stats_scratch). A site whose output is read
# and dropped inside its layer writes ONE load-time buffer for every layer: the next layer's call of that site is
# queued after every reader of this layer's output (each chip runs its programs in order). Outputs kept across a
# layer or a chunk keep allocating (the mHC streams, ffn_hc's hc = the next layer's pending pre, ffn_residual's stats
# row for the next layer, the final norm). The buffers are reserved here AT LOAD TIME (reserve_forks, from the model's
# shapes; never mid-forward, DS41-146) and pinned (ccl.dealloc skips them); a call whose spec has no buffer (another
# box's shapes this table does not cover) allocates as before, counted in FORK_MISSES. The ops return a new Python
# wrapper of the buffer; the call sites hand on the kept object (DS41-149).

_FORKS: dict = {}  # id(mesh) -> (mesh, {(site, shape, dtype): load-time output})
FORK_MISSES: dict = {}  # site -> calls whose output spec had no load-time buffer (allocated as before; diagnostic)

# mhc_post(stats_weight=)'s partials scratch plan (mhc_post_stats_device_operation.cpp mhc_post_stats_plan)
_POST_MAX_COLS, _POST_UNIT_COST, _POST_MAX_PARTIALS = 8, 4, 64


def forks_on() -> bool:
    return settings.get("FORK_OUT_BUFFERS") == "on"


def _reserve_fork(mesh, site: str, shape, dtype) -> None:
    hit = _FORKS.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        hit = _FORKS[id(mesh)] = (mesh, {})
    key = (site, tuple(int(d) for d in shape), dtype)
    if key in hit[1]:
        return
    t = ttnn.allocate_tensor_on_device(ttnn.Shape(list(key[1])), dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
    ccl.pin(mesh, t)
    hit[1][key] = t


def fork_out(mesh, site: str | None, shape, dtype):
    """The load-time output of ``site`` for a [shape] ``dtype`` TILE DRAM output, or None (FORK_OUT_BUFFERS off, a site
    that keeps allocating, or no such spec: allocate as before)."""
    if site is None or not _FORKS:
        return None
    hit = _FORKS.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        return None
    t = hit[1].get((site, tuple(int(d) for d in shape), dtype))
    if t is None:
        FORK_MISSES[site] = FORK_MISSES.get(site, 0) + 1
    return t


def post_scratch_shape(mesh, rows: int, cols: int) -> list[int]:
    """mhc_post(stats_weight=)'s scratch spec [1, 1, R, 32 blocks] for F [1, 1, rows, cols] (the op's plan: the block
    width minimising the busiest core's work, at most 64 partials a row, ties to the wider block)."""
    g = worker_grid(mesh)  # FIX.15: the fork runs on the capped grid (MAX_WORKER_CORES)
    cores = max(1, g.x * g.y)
    col_tiles, row_tiles = cols // 32, rows // 32
    top = min(_POST_MAX_COLS, col_tiles)
    best_cost, best_cb = None, 0
    for cb in range(1, top + 1):
        nb = -(-col_tiles // cb)
        if nb > _POST_MAX_PARTIALS and cb < top:
            continue
        cost = -(-(nb * row_tiles) // cores) * (cb + _POST_UNIT_COST)
        if best_cost is None or cost <= best_cost:
            best_cost, best_cb = cost, cb
    return [1, 1, row_tiles * 32, -(-col_tiles // best_cb) * 32]


def reserve_forks(model) -> None:
    """Load time (TtDS41Model's constructor, after reserve_sites): name every block's in-layer fork sites and reserve one
    output a site from the model's shapes. Off: no site is named, every call allocates (today's path)."""
    if not forks_on():
        return
    from .common import stream_dtype

    mesh, cfg = model.mesh, model.cfg
    sp, tp = tuple(mesh.shape)
    rows = model.chunk // sp  # S/SP: the streams' rows
    hid = cfg.dim // tp  # H/TP: a stream's columns on a chip
    q = model.tables.geo.q_rows  # S/(SP TP): the chip's own rows (ATTN_SP32)
    f32, bf16 = ttnn.float32, ttnn.bfloat16
    for blk in model.blocks.values():
        # mHC (MHC_FUSED=on): attn_hc's hc lives to ffn_collapse, both collapses' y to their norm; ffn_hc's hc is the
        # next layer's pending pre (allocates)
        if getattr(blk, "fused", False):
            blk.attn_hc.out_sites = ("fork.attn_hc.hc", "fork.attn_hc.y")
            blk.ffn_hc.out_sites = (None, "fork.ffn_hc.y")
            _reserve_fork(mesh, "fork.attn_hc.hc", [1, 1, rows, cfg.hc * (cfg.hc + 2)], f32)
            for s in ("fork.attn_hc.y", "fork.ffn_hc.y"):
                _reserve_fork(mesh, s, [1, 1, rows, hid], stream_dtype())
            # the residuals' mhc_post(stats_weight=): attn_residual's row is read by this layer's ffn_hc (ffn's row
            # by the next layer: allocates); the partials scratch is dead when the op ends (one for both)
            blk.attn_res.out_sites = ("fork.post.row", "fork.post.scratch")
            blk.ffn_res.out_sites = (None, "fork.post.scratch")
            _reserve_fork(mesh, "fork.post.row", [1, 1, rows, 32], f32)
            _reserve_fork(mesh, "fork.post.scratch", post_scratch_shape(mesh, rows, hid), f32)
        # the norms: attn_norm (the chip's rows under ATTN_SP32, else the column split), q_norm / kv_norm on the
        # projection's rows, ffn_norm (column split: the stats and the output)
        sp32 = getattr(blk, "sp32", False)
        _name_norm(mesh, blk.attn_norm, "fork.attn_norm", q if sp32 else rows)
        prow = q if blk.q_a.proj.full_rows else rows
        _name_norm(mesh, blk.q_a.norm, "fork.q_norm", prow)
        _name_norm(mesh, blk.kv.norm, "fork.kv_norm", prow)
        _name_norm(mesh, blk.ffn_norm, "fork.ffn_norm", rows)
        # RoPE: kv (the window rows; to_layout / gathered copies are what the cache keeps), q and the output's inverse
        att = getattr(blk, "attention", None)
        if blk.kv.stock:
            blk.kv.rope_site = "fork.rope.kv"
            _reserve_fork(mesh, "fork.rope.kv", [1, 1, prow, blk.kv.k1 - blk.kv.k0], bf16)
        # (ROPE_CHAINS=on, P.22: the q / output chains write sparse_sdpa's q and attn.oa, no rope outputs to keep)
        if att is not None and att.stock and att.sp32 and settings.get("ROPE_CHAINS") != "on":
            att.rope_sites = ("fork.rope.q", "fork.rope.o")
            for s in att.rope_sites:
                _reserve_fork(mesh, s, [1, att.h_l, q, att.d], bf16)


def _name_norm(mesh, norm, site: str, rows: int) -> None:
    """A TtDistributedRmsNorm's site and its load-time outputs (NORM_IMPL=on): [1, 1, rows, width] in its out dtype;
    a column-split norm also its stats [1, 1, rows, 32 B] (fp32)."""
    if norm.impl != "on":
        return
    width = norm.hidden // norm.tp
    norm.site = site
    _reserve_fork(mesh, site, [1, 1, rows, width], norm.out_dtype)
    if norm.split:
        _reserve_fork(mesh, site + ".stats", [1, 1, rows, norm_stats_width(mesh, rows, width)], ttnn.float32)
