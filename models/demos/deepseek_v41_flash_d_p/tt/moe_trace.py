# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MOE_TRACE=moe (P.61, tt/settings.py): the MoE half of every layer replayed from a Metal trace.

Terms
-----
* the MoE half (the segment): the block steps from ffn_norm through ffn_residual (ffn_norm, router, experts,
  shared_expert, moe_add, ffn_residual). On the steady path (OWN_ROWS, FFN_FEED, NORM_IN_MHC, POST_COLLAPSE,
  MOE_BLOCK_SRC=main) that is: the fused head (+ the shared expert), the dispatch, the routed experts + combine, the
  tilize + add, P.56's planner (MOE_REPLICAS=prev) and ffn_residual's mhc_post (the next layer's attn_hc row). None of
  it reads the chunk's start: the same commands every chunk.
* its inputs: attn_residual's new streams (h_mid) and ffn_hc's mixes (the mhc_post hc output), written by the eager
  attn_residual pass straight into load-time buffers here (``input_buffers``: the streams one buffer for every layer,
  the mixes one a layer - they are the layer output's pending pre, read by the next layer), and the head's x, which
  that pass already writes into the block's persistent feed buffer (FFN_FEED). So a replay needs no input copy.
* key: the host facts the captured commands depend on (shapes, the path flags, MOE_REPLICAS prev on / off and its host
  mode, MOE_COUNTS); a call whose key has no trace runs eager; after WARM eager calls of a key the next call is
  captured (programs compiled, the block's lazy buffers and slots made, the planner prepared).
* capture: every DRAM buffer the segment allocates during the capture stays allocated for the trace's life - the
  Python frees of DRAM tensors (tt/ccl.dealloc, ttnn.deallocate) are held, and every top-level op's output is kept -
  so no later allocation can take an address a replay writes (a freed capture buffer would be a hole the eager
  attention could allocate a live tensor in; a replay would then overwrite it). L1 buffers (the block's arena) are
  freed as in the eager call: the arena's address is free at every MoE call (no L1 tensor outlives its op in the
  layer). A graph capture of the same span checks it: a DRAM buffer allocated AND freed inside the capture is an
  error. The trace's own command buffer is allocated by Metal (dynamic mode: top-down in DRAM, no trace region).
* a capture first waits for the chunk's engram uploads (ENGRAM_ASYNC's worker writes on CQ 0; Metal refuses a write
  during a capture); only at the captures.
* host work kept per call (outside the trace): P.56's guard (pre_call: its reset copy of the router bias is an eager
  op, issued only when a layer's plan is not ready - a request's first chunk) and the planner's ordinals
  (post_call_host); the replay itself (execute_trace, non-blocking).

Not traced (eager, as MOE_TRACE=off): the decoder tail's blocks (P.31), DEEP_LAYERS=tail's source layer (its output is
kept across chunks), MOE_REPLICAS runtime / exact or MOE_PREV_HOST=legacy (host-chosen fill per call), any call off
the steady path (no feed, no collapse row).
"""

from __future__ import annotations

import time

from loguru import logger

import ttnn

from . import ccl
from .settings import settings

WARM = 2  # eager calls of a key before its capture
SEG_STEPS = ("ffn_norm", "router", "experts", "shared_expert", "moe_add", "ffn_residual")

_INPUTS: dict = {}  # (id(mesh), shape, dtype) -> (mesh, the streams input buffer, one for every layer)
STATS = {"captures": 0, "replays": 0, "eager": 0, "capture_s": 0.0, "kept_bytes": 0}


def on() -> bool:
    return settings.get("MOE_TRACE") == "moe"


def _dram(t) -> bool:
    try:
        return t.is_allocated() and t.memory_config().buffer_type == ttnn.BufferType.DRAM
    except Exception:  # (a host tensor, a view without a buffer)
        return False


def _alloc(mesh, shape, dtype) -> ttnn.Tensor:
    t = ttnn.allocate_tensor_on_device(
        ttnn.Shape([int(d) for d in shape]), dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    ccl.pin(mesh, t)
    return t


def prepare(model) -> None:
    """Load time (TtDS41Model's constructor, after the forks' sites): a SegmentTrace on every block whose MoE half can
    trace, with its load-time input buffers. Off: nothing (every block runs as before)."""
    if not on():
        return
    from .common import stream_dtype
    from .layout import HC
    from .moe_block_main_adapter import TtMoEBlockMainLayer

    mesh, cfg = model.mesh, model.cfg
    if getattr(model, "tail_of", None) is not None or not (
        model.own and model.fused and model.post_collapse and getattr(model, "norm_in_mhc", False)
    ):
        logger.warning("MOE_TRACE=moe: the model is not on the steady path (OWN_ROWS / POST_COLLAPSE): no trace")
        return
    rows = model.rows
    sdt = stream_dtype()
    key = (id(mesh), (1, 1, rows, HC * cfg.dim), sdt)
    hit = _INPUTS.get(key)
    if hit is None or hit[0] is not mesh:
        hit = _INPUTS[key] = (mesh, _alloc(mesh, key[1], sdt))
    streams = hit[1]
    n = 0
    for i, blk in model.blocks.items():
        if not isinstance(blk.moe_block, TtMoEBlockMainLayer) or i == model.deep_src:
            continue
        if blk.ffn_hc.norm is None or blk.ffn_hc.feed is None:
            continue
        blk.seg = SegmentTrace(blk, streams, _alloc(mesh, (1, 1, rows, HC * (HC + 2)), ttnn.float32))
        n += 1
    logger.info(f"MOE_TRACE=moe: {n} of {len(model.blocks)} layers' MoE half traced (capture after {WARM} eager calls)")


class _Keep:
    """During a capture: the segment's DRAM frees are held and every top-level op's output is kept (the trace's buffers
    stay allocated for its life); L1 frees go through."""

    def __init__(self):
        self.kept: list = []

    def _hold(self, t, *a, **k):
        if _dram(t):
            self.kept.append(t)
            return
        return self._orig_ttnn(t, *a, **k)

    def _hold_ccl(self, t):
        if _dram(t):
            self.kept.append(t)
            return
        return self._orig_ccl(t)

    def _post(self, op, args, kwargs, output):
        self.kept.append(output)

    def __enter__(self):
        self._orig_ccl, self._orig_ttnn = ccl._DEALLOC[0], ttnn.deallocate
        ccl._DEALLOC[0] = self._hold_ccl
        ttnn.deallocate = self._hold
        self._hook = ttnn.register_post_operation_hook(self._post)
        self._hook.__enter__()
        return self

    def __exit__(self, *exc):
        self._hook.__exit__(*exc)
        ccl._DEALLOC[0], ttnn.deallocate = self._orig_ccl, self._orig_ttnn
        return False


def _holes(graph) -> tuple[list, int, int]:
    """From a graph capture of the span: DRAM buffers allocated and freed inside it ([(address, size)]), and the DRAM /
    L1 bytes still allocated by it at its end (the busiest device)."""
    live: dict = {}
    holes = []
    for node in graph:
        kind = node.get("node_type") if isinstance(node, dict) else None
        if kind not in ("buffer_allocate", "buffer_deallocate"):
            continue
        p = node.get("params", {})
        k = (p.get("device_id"), int(p.get("address", 0)), p.get("type"))
        if kind == "buffer_allocate":
            live[k] = int(p.get("size", 0))
        elif k in live:
            size = live.pop(k)
            if k[2] == "DRAM":
                holes.append((k[1], size))
    per = {}
    for (dev, _, typ), size in live.items():
        per.setdefault((dev, typ), 0)
        per[(dev, typ)] += size
    dram = max([v for (d, t), v in per.items() if t == "DRAM"] or [0])
    l1 = max([v for (d, t), v in per.items() if t != "DRAM"] or [0])
    return holes, dram, l1


class _Rec:
    __slots__ = ("tid", "out", "row", "kept", "dram_bytes", "capture_s", "replays", "ids", "marks")


class SegmentTrace:
    """One block's MoE half: eager until its key is warm, then one trace a key, replayed every call."""

    def __init__(self, blk, streams: ttnn.Tensor, mixes: ttnn.Tensor):
        self.blk, self.mesh = blk, blk.mesh
        self.streams, self.mixes = streams, mixes
        self.warm: dict = {}
        self.traces: dict = {}
        self.enabled = True  # (tests: False runs every call eager, the input buffers still written)
        self.steps = [s for s in blk.steps if s.name in SEG_STEPS]
        assert [s.name for s in self.steps] == list(SEG_STEPS), [s.name for s in self.steps]

    def outputs(self) -> dict:
        """attn_residual's output buffers for this layer (TtHcResidual.fused out= / hc_out=)."""
        return dict(out=self.streams, hc_out=self.mixes)

    def _key(self, ctx, h, mix, x):
        e = ctx.extra
        if h is not self.streams or mix is not self.mixes:
            return None
        if not (e.get("ffn_normed") and e.get("ffn_fed")):
            return None
        mk = self.blk.moe_block.trace_key(fed=True)
        if mk is None:
            return None
        nxt = self.blk.model.next_hc(self.blk.layer)
        return (mk, tuple(x.shape), x.dtype, nxt.layer if nxt is not None else None)

    def __call__(self, ctx, h, mix, x, eager):
        """The segment's output (the layer's DevH). ``eager(ctx, h, mix, x)`` runs the steps as MOE_TRACE=off does."""
        key = self._key(ctx, h, mix, x) if self.enabled else None
        if key is None:
            STATS["eager"] += 1
            return eager(ctx, h, mix, x)
        rec = self.traces.get(key)
        if rec is None:
            n = self.warm.get(key, 0)
            if n < WARM:
                self.warm[key] = n + 1
                STATS["eager"] += 1
                return eager(ctx, h, mix, x)
            return self._capture(key, ctx, h, mix, x, eager)
        return self._replay(rec, ctx)

    def _capture(self, key, ctx, h, mix, x, eager):
        from .model import DevH

        m, mesh = self.blk.moe_block, self.mesh
        # ENGRAM_ASYNC=on: the worker uploads the chunk's engram rows on CQ 0 - no write may come during a capture
        self.blk.model.wait_engram()
        m.pre_call()  # (the guard: never inside the capture)
        m._pre_done = True
        keep = _Keep()
        marks = ctx.extra["seg_marks"] = []
        t0 = time.perf_counter()
        ttnn._ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        graph = None
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        ids0 = ttnn._ttnn.get_device_operation_id()  # (the captured programs' runtime ids: [ids0, ids1))
        try:
            with keep:
                out = eager(ctx, h, mix, x)
        finally:
            ids1 = ttnn._ttnn.get_device_operation_id()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            graph = ttnn._ttnn.graph.end_graph_capture()
            m._pre_done = False
        cap_s = time.perf_counter() - t0
        holes, dram, l1 = _holes(graph if isinstance(graph, list) else [])
        if holes:
            ttnn.release_trace(mesh, tid)
            raise RuntimeError(
                f"MOE_TRACE layer {self.blk.layer}: {len(holes)} DRAM buffer(s) allocated and freed inside the capture "
                f"(a later allocation could take them and a replay would overwrite it): {holes[:8]}"
            )
        assert isinstance(out, DevH), type(out)
        row = out.stats[1] if out.stats is not None else ()
        row = row if isinstance(row, tuple) else (row,)
        for t in [out.streams] + [v for v in row if isinstance(v, ttnn.Tensor)]:
            if id(t) not in ccl._KEEP:
                ccl.pin(mesh, t)  # (the next layer frees its input: a trace buffer stays)
        assert out.pre is self.mixes, "the segment's pending pre is ffn_hc's mixes (the layer's load-time buffer)"
        rec = _Rec()
        rec.tid, rec.out, rec.row, rec.kept = tid, out.streams, out.stats, keep.kept
        rec.dram_bytes, rec.capture_s, rec.replays, rec.ids = dram, cap_s, 0, (ids0, ids1)
        rec.marks = [(name, i - ids0) for name, i in marks]
        self.traces[key] = rec
        STATS["captures"] += 1
        STATS["capture_s"] += cap_s
        STATS["kept_bytes"] += dram
        logger.info(
            f"MOE_TRACE layer {self.blk.layer}: captured in {cap_s * 1e3:.1f} ms, {dram / 2**20:.2f} MiB of DRAM a chip "
            f"kept ({l1} B of L1 left allocated), key {key}"
        )
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        return DevH(out.streams, self.mixes, stats=out.stats)

    def _replay(self, rec, ctx):
        from models.demos.common.bringup.testing import profiler

        from .model import DevH

        m = self.blk.moe_block
        e = ctx.extra
        e.pop("ffn_normed", None)  # (the steps' host flags, as the eager steps consume them)
        e.pop("ffn_fed", None)
        self.blk._moe_xq_ready = False
        m.pre_call()
        # (the replay's device-operation ids and its steps' signposts: the profiler charges it here, step by step)
        profiler.trace_replay(*rec.ids, marks=rec.marks)
        ttnn.execute_trace(self.mesh, rec.tid, cq_id=0, blocking=False)
        m.post_call_host()
        rec.replays += 1
        STATS["replays"] += 1
        return DevH(rec.out, self.mixes, stats=rec.row)

    def release(self) -> None:
        for rec in self.traces.values():
            ttnn.release_trace(self.mesh, rec.tid)
        self.traces.clear()
        self.warm.clear()

    def memory(self) -> dict:
        """{key: DRAM bytes a chip the trace keeps} (the command buffer not included)."""
        return {k: r.dram_bytes for k, r in self.traces.items()}
