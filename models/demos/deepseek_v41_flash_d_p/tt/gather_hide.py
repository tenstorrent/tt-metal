# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PACKED_GATHER_HIDE=on (PB.3H, OWNER 2026-11-19 20:55): a packed chunk's per-segment history all-gathers run
CONCURRENT with the attention, on a sub-device of their own.

Off (PB.3A): every segment's history prefetch (tt/history_gather.prefetch: each KV source's served history [0, c hs)
of the segment's slot packed to 4-bit records and all-gathered over the whole mesh into the lane's buffer) runs serially
at the chunk's first engram wait, and each segment's tail (the chunk's own periods, written by the source layer's
compressor) serially at the source layer's table. On one command queue these add up on the device's critical path.

On (this file; deepseek_v3_d_p's sparse_sdpa-beside-push pattern, tt/mla/sparse_kv_push.py + mla._beside_prefetch):
``Split`` is one sub-device manager per mesh: sub-device 0 = the matmul grid (MAX_WORKER_CORES: 11 x 10 from (0, 0) on
the Galaxy, where every matmul and every bring-up fork with ``max_grid`` runs), sub-device 1 = the rest of the grid
(the 12th column: 10 cores, the full-mesh gather's 2 rings x 2 directions x 2 links workers + 2 copy cores). A
``region`` loads it, enqueues gathers on sub-device 1 (high_bw_all_gather's subdevice_id / sub_core_grids, with the
caller-owned semaphores: its allocation-free path for concurrent sub-device dispatch), runs a compute block on
sub-device 0 (every stock non-matmul op's ``sub_cores()`` moved to the matmul grid for the block,
cores.nonmm_on_mm_grid; the block has no fabric op) and clears the manager, which joins both.

Per packed chunk (``Hide``, made by tt/packed_onecall.OneCall when the switch is on and the grid has a free column):
- the history prefetch of every (source, segment) is a pending item instead of an engram-wait job. Each layer's
  attention phase (wq_b, rope chain, sparse_sdpa, rope chain, wo_a, wo_b: ~1.2 ms a layer at 8192 rows) is a region
  that gathers the most urgent source's next items, spread over the layers before that source's first reader
  (source 2's four at layers 0-1, 8's at 2-7, 14's at 8-13). The items' packs (local, sub-device 0's grid) run just
  before the region into record buffers of their own (``_loc``), so no gather reads a buffer a pack writes.
- at a source layer the tails of every segment are gathered beside the indexer's gates and query (matmuls and a rope
  chain), right after the compressor's writes: pack + slice before, the slice_writes after the region; the table
  step then finds them done (history_gather.tail_finish / ``ready``).
- an item not gathered by its source's table (a shorter layer list) is gathered there serially (``flush``).
Same bytes at the same rows as off (data moves; the compute block's ops are row-local on any core count).

Dispatch is in order (one command queue): a program's launch waits for the previous program of ITS sub-device only,
but nothing behind it launches meanwhile. So a region's first gather goes before its compute steps and each further
gather between two steps (its launch waits for the gather before it, while the steps already launched run): a
region of k gathers and n steps puts gather j >= 1 before step min(n - 1, ceil(j n / k)).

Measurement (diagnostic, the walls' profiled run: history_gather.TRACE on): the real-time profiler keeps one open
record a command queue (a program's end is stamped while dispatch waits for workers), so it cannot show two
sub-devices' programs at once. ``Stamps`` instead: a one-core program that writes the chip's wall clock to a slot, on
sub-device 0 before every further gather (its launch waits for the step before: that step's end) and right after it
(launched once the gather before is done: the dispatch stall of the compute = their difference), and at the region's
end on sub-device 0 (the compute's end) then sub-device 1 (the last gather's end, at least the compute's): the
exposed gather time of a region = its stalls + max(0, last gather end - compute end); the rest of the gathers' time
ran beside the compute (``LOG``).

Boxes without a matmul cap (every box but the Galaxy) have no free column: ``Split.of`` is None and the switch does
nothing there (the serial path)."""

from __future__ import annotations

from collections import defaultdict

import ttnn
from models.demos.common.bringup.testing import profiler

from . import history_gather
from .cores import nonmm_on_mm_grid, worker_grid
from .settings import settings


def hide_on() -> bool:
    """PACKED_GATHER_HIDE=on (call-time, bringup/hooks.py batch_call_settings)."""
    return settings.get("PACKED_GATHER_HIDE") == "on"


def _rect(x0: int, y0: int, x1: int, y1: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


STATS = {"regions": 0, "prefetch": 0, "tails": 0, "flushed": 0}  # since load (diagnostic, tests)
LAST_PLAN: dict = {}  # PB.6L (diagnostic): the last packed chunk's piece plan (regions, unplaced, cover)
LOG: list = []  # (diagnostic, with history_gather.TRACE on) per region: (lanes of its gathers, [(kind, slot), ...])

_STAMP_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr auto d_args = TensorAccessorArgs<0>();

void kernel_main() {
    const uint32_t dst = get_arg_val<uint32_t>(0);
    const uint32_t slot = get_arg_val<uint32_t>(1);
    const auto d = TensorAccessor(d_args, dst, 64);
    const uint32_t l1 = get_write_ptr(0);
    volatile tt_reg_ptr uint32_t* clo = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile tt_reg_ptr uint32_t* chi = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    const uint32_t lo = clo[0];
    const uint32_t hi = chi[0];
    volatile tt_l1_ptr uint32_t* b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    b[0] = lo;
    b[1] = hi;
    noc_async_write(l1, d.get_noc_addr(slot), 64);
    noc_async_write_barrier();
}
"""


class Stamps:
    """(diagnostic) The chip wall clock into slot ``i`` of a [1, 1, SLOTS, 16] uint32 DRAM buffer, every chip, from one
    core of sub-device 0 (``compute``) or 1 (``gather``): one prepared program each, the slot a per-run value."""

    SLOTS = 4096

    def __init__(self, mesh, split: "Split"):
        self.mesh, self.n = mesh, 0
        self.buf = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.SLOTS, 16]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        g = mesh.compute_with_storage_grid_size()
        self.cores = {"compute": (0, 0), "gather": (g.x - 1, 0)}
        self._pp = {}

    def _build(self, core):
        sp, tp = tuple(self.mesh.shape)
        crs = _rect(core[0], core[1], core[0], core[1])
        prog = ttnn.ProgramDescriptor()
        prog.cbs = [
            ttnn.CBDescriptor(
                total_size=64, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, 64)]
            )
        ]
        rt = ttnn.RuntimeArgs()
        rt[core[0]][core[1]] = [int(self.buf.buffer_address()), 0]
        prog.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_STAMP_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=list(ttnn.TensorAccessorArgs(self.buf).get_compile_time_args()),
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
        ]
        md = ttnn.MeshProgramDescriptor()
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(sp - 1, tp - 1))] = prog
        return md

    def __call__(self, where: str) -> int:
        """Stamp now (in queue order) from ``where``'s core; returns the slot."""
        pp = self._pp.get(where)
        if pp is None:
            pp = self._pp[where] = ttnn.bringup.prepare_program(self._build(self.cores[where]), [self.buf], [(0, 1)])
        i = self.n % self.SLOTS
        self.n += 1
        ttnn.bringup.run_prepared(pp, [self.buf], [i])
        return i

    def read(self) -> list:
        """[chip] -> uint64 wall clock per slot (host; after a sync)."""
        import torch

        out = []
        for d in ttnn.get_device_tensors(self.buf):
            t = ttnn.to_torch(d).reshape(self.SLOTS, 16).to(torch.int64) & 0xFFFFFFFF
            out.append((t[:, 1] << 32) | t[:, 0])
        return out


_STAMPS: dict = {}


def warm_stamps(mesh) -> None:
    """(diagnostic) Both stamp programs built and run once in a region (their first use compiles: not inside a
    measured run)."""
    st, split = stamps(mesh), Split.of(mesh)
    if st is None:
        return
    mesh.load_sub_device_manager(split.mgr)
    try:
        st("compute")
        st("gather")
    finally:
        mesh.clear_loaded_sub_device_manager()


def stamps(mesh) -> "Stamps | None":
    """(diagnostic) The mesh's Stamps (made on first use) where the grid has a gather sub-device."""
    split = Split.of(mesh)
    if split is None:
        return None
    hit = _STAMPS.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        hit = _STAMPS[id(mesh)] = (mesh, Stamps(mesh, split))
    return hit[1]


class Split:
    """The compute / gather sub-device manager of a mesh (made once, kept for the process)."""

    _BY: dict = {}

    @classmethod
    def of(cls, mesh) -> "Split | None":
        hit = cls._BY.get(id(mesh))
        if hit is None or hit[0] is not mesh:
            g, w = mesh.compute_with_storage_grid_size(), worker_grid(mesh)
            split = cls(mesh, w, g) if (w.y == g.y and w.x < g.x) else None
            hit = cls._BY[id(mesh)] = (mesh, split)
        return hit[1]

    def __init__(self, mesh, w, g):
        self.mesh = mesh
        self.compute = _rect(0, 0, w.x - 1, g.y - 1)
        self.gather = _rect(w.x, 0, g.x - 1, g.y - 1)
        self.mgr = mesh.create_sub_device_manager([ttnn.SubDevice([self.compute]), ttnn.SubDevice([self.gather])], 0)
        self.sub = dict(subdevice_id=ttnn.SubDeviceId(1), sub_core_grids=self.gather)

    def region(self, gathers: list, steps: list, section: str, after_last: bool = False, cover=None, own=None):
        """``gathers`` [(lane, fn(sub))] (each enqueues one gather on sub-device 1) beside the compute ``steps`` (each
        fn(previous result), sub-device 0); returns the last step's result. The first gather before the steps
        (its fence waits for the other chips, the compute needs no one: V3's order), gather j >= 1 before step
        min(n - 1, ceil(j n / k)). ``section``: the profile section the steps are charged to (the gathers: "history_gather").
        PB.6L: ``after_last`` lets a gather go after the last step (n: a 1-step region); ``cover`` (the steps' estimated
        ms, gathers given as (lane, fn, est ms)): gather j >= 1 before the step at which the gathers before it are
        estimated done (the steps taken as equally long), so its launch does not hold the compute launches behind it.
        """
        mesh = self.mesh
        k, n = len(gathers), len(steps)
        at = {}
        if cover is not None:  # PB.6L: by estimated time over the first ``own`` steps (the rest: packs, after them)
            done, m = 0.0, n if own is None else own
            for j, (lane, g, ms) in enumerate(gathers):
                pos = (
                    0 if j == 0 else max(j, min(m, -(-int(round(done * m * 1000)) // max(1, int(round(cover * 1000))))))
                )
                pos = min(pos, n)  # (one gather a step where the steps allow: its launch waits for the one before)
                at.setdefault(pos, []).append((lane, g))
                done += ms
            gathers = [(lane, g) for lane, g, _ in gathers]
        for j, g in enumerate(gathers if cover is None else ()):
            if (
                after_last
            ):  # PB.6L: gather j >= 1 before step ceil(j n / k), or after the last step (n: a 1-step region)
                at.setdefault(0 if j == 0 else max(1, min(n, -(-j * n // k))), []).append(g)
            else:
                at.setdefault(0 if j == 0 or n < 2 else max(1, min(n - 1, -(-j * n // k))), []).append(g)
        st = stamps(mesh) if history_gather.TRACE is not None else None
        log = []
        mesh.load_sub_device_manager(self.mgr)
        try:
            out = None
            if st is not None:
                log.append(("start", None, st("compute")))
            with nonmm_on_mm_grid():
                for i, step in enumerate(steps):
                    for j, (lane, g) in enumerate(at.get(i, ())):
                        stall = st is not None and (i > 0 or j > 0)
                        if stall:
                            log.append(("before", lane, st("compute")))
                        profiler.signpost("history_gather")
                        a = ttnn._ttnn.get_device_operation_id() if st is not None else None
                        g(self.sub)
                        if st is not None:
                            log.append(("ids", lane, (a, ttnn._ttnn.get_device_operation_id())))
                        if stall:
                            log.append(("after", lane, st("compute")))
                    profiler.signpost(section)
                    out = step(out)
                for j, (lane, g) in enumerate(at.get(n, ())):  # (PB.6L: after the last step)
                    if st is not None:
                        log.append(("before", lane, st("compute")))
                    profiler.signpost("history_gather")
                    a = ttnn._ttnn.get_device_operation_id() if st is not None else None
                    g(self.sub)
                    if st is not None:
                        log.append(("ids", lane, (a, ttnn._ttnn.get_device_operation_id())))
                        log.append(("after", lane, st("compute")))
            if st is not None:
                log.append(("end_compute", None, st("compute")))
                log.append(("end_gather", gathers[-1][0], st("gather")))
                LOG.append(log)
            return out
        finally:
            mesh.clear_loaded_sub_device_manager()  # (joins both sub-devices)
            STATS["regions"] += 1


DEC_WEIGHT = 0.5  # the index-key decodes' region against an attention phase (PB.3H walls: ~0.6-1 vs ~1.3-1.5 ms)

# PACKED_LONG_HIDE=on (PB.6L): the planning estimates (ms; they place work, the bytes never depend on them). Measured
# on the 8x4 at 1M (tests/packed/test_long_hide_probe.py: bench and region stamps): a history gather on the gather
# column ~0.15 ms + its bytes at ~64 MB/ms alone, ~2x slower beside a DRAM-bound step (the decodes, the top-k), as
# fast beside the attention phase (sparse_sdpa, matmuls); serially (every core, next to the links) ~2.4x faster.
# A region's cover = the ms of gathering (at the alone rate) it hides: attention phase ~1.7-1.8 ms of compute -> 1.6
# (1M stamps: a 1.5 ms piece beside it, 0.00 ms past on every one of 12; 1.8: 0.03-0.14 past, two pieces 0.4-0.6:
# an overrun on the gather column costs ~2.4x the same rows gathered serially at the table, so the plan under-fills); the tails region at layers 8 / 14 is ~0.27 ms of compute, the
# tail gather fills it -> 0; the index top-k ~3.1e-6 ms a position of the longest lane, half of it (one 9k-row piece
# beside a 3.2 ms top-k: 0.00 past); the decodes (~200 rows/us on 11 x 10 cores, ~17.6 on 4 x 2; the region measured
# 7.8 ms at 1M beside 1.75 ms of pieces, 0 past) at the alone rate.
GATHER_FIXED_MS, GATHER_BYTES_MS = 0.15, 64e6
COVER_ATTN_MS, COVER_TAILS_MS = 1.6, 0.0
DEC_COVER = 1.2  # the decodes' region measured 5.3-8.5 ms at 1M against ~3.9 ms of decodes
# the items' packs (slice + tilize + ckv_bfp4 of a whole history: ~0.4 ms at 1M, DRAM-bound) as compute steps of an
# earlier region: at most this many a region, each adding this much cover
PACKS_A_REGION, PACK_COVER_MS = 2, 0.2
TOPK_MS_A_POSITION, DEC_ROWS_US_WIDE, DEC_ROWS_US_NARROW = 3.1e-6, 200.0, 17.6
DRAM_BOUND_SLOWDOWN = 2.0  # (the top-k's)
MIN_PIECE_ROWS = 512  # a smaller remainder of a region's cover is left unused; of an item, goes with its last piece
# at most a region: a later gather's launch waits for the one before it (and holds the compute launches behind it)
REGION_GATHERS = {"attn": 2, "dec": 4, "topk": 2, "tails": 1}


def long_on() -> bool:
    """PACKED_LONG_HIDE=on (PB.6L; call-time, bringup/hooks.py batch_call_settings)."""
    return settings.get("PACKED_LONG_HIDE") == "on"


def _balance(items: list, slots: list, chunk: int) -> dict:
    """{region key: items} of one source (``slots``: [(key, weight)]): the largest histories first, each to the region
    with the least gathered rows a unit of weight after taking it (a gather's time grows with its rows); within a
    region the largest first (the steps after it hide the rest)."""
    rows = lambda it: (int(it[1].start) // int(chunk)) * (it[2].stripe // it[2].ratio)  # noqa: E731
    load = {k: 0 for k, _ in slots}
    out = {k: [] for k, _ in slots}
    for it in sorted(items, key=rows, reverse=True):
        k = min(slots, key=lambda kw: ((load[kw[0]] + rows(it)) / kw[1], str(kw[0])))[0]
        load[k] += rows(it)
        out[k].append(it)
    return out


_LOC: dict = {}  # (id(mesh), rows, k) -> (mesh, HistoryBuffer): record buffers of the region's k-th pack
_SMALL: dict = {}  # (id(mesh), rows, owners, bfp4, k) -> (mesh, HistoryBuffer): the k-th tail's gather output


def _held(store: dict, key, mesh, make):
    hit = store.get(key)
    if hit is None or hit[0] is not mesh or not hit[1].tensor.is_allocated():
        hit = store[key] = (mesh, make())
    return hit[1]


class Hide:
    """The hidden history gathers of one packed chunk (tt/packed_onecall.OneCall.hide)."""

    def __init__(self, oc, split: Split):
        self.oc, self.split = oc, split
        m = oc.model
        self.mesh, self.chunk = m.mesh, int(m.chunk)
        self.sources = sorted(m.history_sources())
        skipped = m._decoder_set if m.skip_mode == "decode" else frozenset()
        self.order = [i for i in sorted(m.blocks) if i not in skipped]  # (the layers whose attention phase runs)
        self.items = []  # (source, SegRun, compressed cache), source then lane order
        for s in self.sources:
            for r in oc.pk.segs:
                c = r.state.compress.get(s)
                if c is not None and s not in r.state.shared_compress:
                    self.items.append((s, r, c.ckv))
        for r in oc.pk.segs:  # (no history prefetch at the engram wait: the items here; the index keys stay there)
            r.fields["_pf_start"] = None
        self.plan = {}  # source -> {region key: items}
        engram = getattr(m, "engram", None) or {}
        self.dec_layer = min(engram) if engram else None  # (the index keys are decoded at this layer's engram wait)
        self.dec_done = False
        # PB.6L: the pieces' plan over every cover region (``_long_plan``) when an item is longer than one attention
        # phase hides whole (a ~1M lane); shorter histories keep PB.3H's plan (whole items: no piece, pack or region
        # beyond it; 4 x 2k at 139k A/B: the pieces' plan +1.0 ms wall, PB.3H's 0)
        self.long = long_on() and self._long_items()
        if self.long:
            self._long_plan()

    # ------------------------------------------------------------------ buffers
    def _loc(self, ckv, k: int):
        if not history_gather.bfp4_on():
            return None
        R = int(ckv.slab.rows)
        return _held(
            _LOC, (id(self.mesh), R, k), self.mesh, lambda: history_gather.HistoryBuffer(self.mesh, R, None, True)
        )

    def _loc_item(self, ckv, j: int):
        """PB.6L: item j's own record buffer for the chunk (its pieces may run regions after other items' packs)."""
        if not history_gather.bfp4_on():
            return None
        R = int(ckv.slab.rows)
        return _held(
            _LOC,
            (id(self.mesh), R, "item", j),
            self.mesh,
            lambda: history_gather.HistoryBuffer(self.mesh, R, None, True),
        )

    def _small(self, plans: list):
        """The combined tail gather's output: [owners M, width] (M: every plan's rows together)."""
        buf, owners = plans[0][1], plans[0][5]
        rows = owners * sum(p[4] - p[3] for p in plans)
        from .page_copy import dyn

        if dyn("hist"):  # PB.7C: one size for every layout (history_gather.tails_max_rows)
            rows = owners * history_gather.tails_max_rows(plans)
        key = (id(self.mesh), rows, owners, buf.bfp4)
        return _held(_SMALL, key, self.mesh, lambda: history_gather.HistoryBuffer(self.mesh, rows, owners, buf.bfp4))

    # ------------------------------------------------------------------ prefetch items
    def _take(self, i: int) -> list:
        """The items of layer i's region: the most urgent source's, spread evenly over the attention phases left
        before its first reader (the source layer itself)."""
        if not self.items:
            return []
        s = self.items[0][0]
        if s <= i:
            return []  # (its reader runs now or ran: flush)
        return self._slot(("attn", i), i)

    def _slot(self, key, i: int) -> list:
        """The items of region ``key`` (("attn", layer) or ("dec",): the index-key decodes at the first engram wait)
        at layer i: the most urgent source's plan (made at its first region: its items balanced over the regions
        before its reader, the decodes' region weighted DEC_WEIGHT)."""
        if not self.items:
            return []
        s = self.items[0][0]
        if s <= i:
            return []
        if s not in self.plan:
            slots = [(("attn", j), 1.0) for j in self.order if i <= j < s]
            if self.dec_layer is not None and not self.dec_done and i <= self.dec_layer < s:
                slots.append((("dec",), DEC_WEIGHT))
            self.plan[s] = _balance([it for it in self.items if it[0] == s], slots or [(key, 1.0)], self.chunk)
        take = self.plan[s].get(key, [])
        ids = {id(it) for it in take}
        self.items = [it for it in self.items if id(it) not in ids]
        return take

    def decodes(self, steps: list) -> None:
        """The first engram wait's index-key decodes (``steps``: one a segment; local fork ops) as a region beside
        the prefetch gathers planned there."""
        self.dec_done = True
        self.attention(self.dec_layer, steps, key=("dec",))

    def attention(self, i: int, steps: list, key=None):
        """Layer i's attention phase (``steps``: fn(previous result) each) as a region beside its share of the
        pending prefetch gathers; returns the last step's result."""
        if self.long:
            return self._long_region(("attn", i) if key is None else key, steps, "attention")
        items = self._take(i) if key is None else self._slot(key, i)
        plans = []
        for k, (s, r, ckv) in enumerate(items):
            p = history_gather.prefetch_prepare(ckv, r.start, self.chunk, loc=self._loc(ckv, k))
            if p is not None:
                plans.append((r.lane, p))
        if not plans:
            out = None
            for step in steps:
                out = step(out)
            return out
        STATS["prefetch"] += len(plans)
        gathers = [(lane, lambda sub, p=p: history_gather.prefetch_gather(p, sub)) for lane, p in plans]
        return self.split.region(gathers, steps, "attention")

    def flush(self, s: int) -> None:
        """Source s's items not gathered yet (its first reader is here): serially, as off."""
        if self.long:
            self._long_flush(s)
            return
        rest = [it for it in self.items if it[0] == s]
        if not rest:
            return
        self.items = [it for it in self.items if it[0] != s]
        for _, r, ckv in rest:
            history_gather.prefetch(ckv, r.start, self.chunk)
        STATS["flushed"] += len(rest)

    # ------------------------------------------------------------------ tails
    def tails_here(self, block) -> bool:
        """Layer ``block`` is a gathered KV source's own layer (its tails are gathered there)."""
        return block.kv_src == block.layer and block.layer in self.sources

    def tails(self, block, steps: list):
        """At a KV source layer whose history is gathered: every segment's tail (the chunk's own periods, just written
        by the compressor) gathered beside ``steps`` (the indexer's gates and query); returns the last step's result."""

        def serial():
            out = None
            for step in steps:
                out = step(out)
            return out

        s, i = block.kv_src, block.layer
        if s != i or s not in self.sources:
            return serial()
        if self.long:  # PB.6L: the source's items packed (their markers set); their pieces may still run at the top-k
            for j in [j for j, it in enumerate(self.items) if it is not None and it[0] == s]:
                self._pack(j)
        else:
            self.flush(s)
        plans = []
        for k, r in enumerate(self.oc.pk.segs):
            c = r.state.compress.get(s)
            if c is None or s in r.state.shared_compress:
                continue
            ckv = c.ckv
            nb = -(-(r.start + self.chunk) // self.chunk)
            local_rows = nb * ckv.stripe // ckv.ratio  # (tt/sparse_attention._compressed's gather rows)
            p = history_gather.tail_prepare(ckv, r.start, local_rows, loc=self._loc(ckv, k))
            if p is not None:
                plans.append(p)
        pieces = [(lane, g) for lane, g, _ in self._long_gathers(("tails", i))] if self.long else []
        if not plans:
            if pieces:
                return self.split.region(pieces, steps, "indexer", after_last=True)
            return serial()
        STATS["tails"] += len(plans)
        # ONE gather for every segment's tail (each owner's rows of every segment, segment after segment): a tail is
        # latency-bound (~0.45 ms on the gather column for 1.3 MB, PB.3H stamps), four back to back stalled the steps
        small = self._small(plans)
        x = history_gather.tails_input(plans)
        gathers = [(-1, lambda sub: history_gather.tails_gather(plans, x, small, sub))] + pieces
        out = self.split.region(gathers, steps, "indexer", after_last=self.long)
        history_gather.tails_finish(plans, small)
        return out

    def topk(self, i: int, steps: list):
        """PB.6L: layer i's index top-k (local, its own 110 cores) as a region beside the pieces planned there."""
        if not self.long:
            out = None
            for step in steps:
                out = step(out)
            return out
        return self._long_region(("topk", i), steps, "indexer")

    # ------------------------------------------------------------------ PB.6L: pieces over every cover region
    def _long_plan(self) -> None:
        """Every item's history planned as pieces over the cover regions before its source's reader: regions in time
        order (per layer: its tails region at a source layer, its index top-k, its attention phase; the decodes at the
        first engram layer), each given rows worth its estimated cover (``cover``: ms of gathering it hides), the most
        urgent source first, its items largest first, an item cut where a region's cover ends (a piece = an owner's
        row range, ttnn.bringup.high_bw_all_gather gathered_dim_start). What no region before the reader takes is
        gathered serially at the source's table (``_long_flush``: next to the links, ~2.4x faster than on the
        gather column)."""
        m = self.oc.model
        segs = self.oc.pk.segs
        self.packs = {}  # j -> prefetch_prepare plan (None: nothing to gather)
        self.left = {}  # j -> pieces not gathered yet
        rec = self._rec = history_gather.REC if history_gather.bfp4_on() else history_gather.WIDTH * 2
        top = max((int(r.start) for r in segs), default=0)
        cover, order = {}, []
        for i in self.order:
            blk = m.blocks.get(i)
            if blk is not None and i in self.sources and getattr(blk, "kv_src", None) == i:
                order.append(("tails", i))
                cover[("tails", i)] = COVER_TAILS_MS
            if blk is not None and getattr(blk, "indexer", None) is not None and m.cfg.is_index_source(i):
                order.append(("topk", i))
                cover[("topk", i)] = TOPK_MS_A_POSITION * top / DRAM_BOUND_SLOWDOWN
            if self.dec_layer == i:
                order.append(("dec",))
            order.append(("attn", i))
            cover[("attn", i)] = COVER_ATTN_MS
        if self.dec_layer is not None:
            from . import index_keys

            us = 0.0
            for r in segs:
                for i in m.index_sources():
                    c = r.state.compress.get(i)
                    if c is None or not getattr(c.index_k, "k_served", False):
                        continue
                    lo, _ = index_keys._span(c.index_k, r.start)
                    wide = lo >= index_keys.DECODE_WIDE_ROWS
                    us += lo / (DEC_ROWS_US_WIDE if wide else DEC_ROWS_US_NARROW)
            cover[("dec",)] = DEC_COVER * us / 1e3

        def allowed(key, s: int) -> bool:
            if key[0] == "attn":
                return key[1] < s
            if key[0] in ("topk", "tails"):
                return key[1] <= s
            return self.dec_layer < s

        by_src = defaultdict(list)
        for j, (s, r, ckv) in enumerate(self.items):
            key = getattr(ckv, "history_key", None)
            h = (int(r.start) // self.chunk) * (ckv.stripe // ckv.ratio) if isinstance(key, tuple) else 0
            if h > 0:
                by_src[s].append((h, j, int(ckv.owners)))

        def place(cover: dict):
            lplan, unplaced, left = defaultdict(list), [], dict(cover)
            for s in sorted(by_src):
                queue = [[j, 0, h, owners] for h, j, owners in sorted(by_src[s], reverse=True)]
                for key in order:
                    if not queue:
                        break
                    if not allowed(key, s):
                        continue
                    while queue and len(lplan[key]) < REGION_GATHERS[key[0]]:
                        j, a, h, owners = queue[0]
                        room = (left[key] - GATHER_FIXED_MS) * GATHER_BYTES_MS / (owners * rec)
                        if not history_gather.bfp4_on():  # (a bf16 history moves whole: the slab is ND-sharded)
                            room = h - a if room >= (h - a) else 0
                        take = h - a if room >= h - a else int(room) // 32 * 32
                        if h - a - take < MIN_PIECE_ROWS:  # (no tiny last piece: its fixed cost and launch wait)
                            take = h - a if room + MIN_PIECE_ROWS >= h - a else take - MIN_PIECE_ROWS
                        if take < min(MIN_PIECE_ROWS, h - a):
                            break
                        lplan[key].append((j, a, a + take))
                        left[key] -= GATHER_FIXED_MS + owners * take * rec / GATHER_BYTES_MS
                        if a + take >= h:
                            queue.pop(0)
                        else:
                            queue[0][1] = a + take
                for j, a, h, owners in queue:
                    unplaced.append((j, a, h))
            return lplan, unplaced, left

        def firsts(lplan) -> dict:
            """item -> the time index of its first piece's region (its source's attention, the table, if none)"""
            out = {}
            for t, key in enumerate(order):
                for j, _, _ in lplan.get(key, ()):
                    out.setdefault(j, t)
            for s_ in by_src:
                for _, j, _ in by_src[s_]:
                    out.setdefault(j, order.index(("attn", s_)) if ("attn", s_) in order else len(order))
            return out

        # the items' packs as leading compute steps of earlier regions (``pack_at``): each at least one region before
        # its first piece (the region's join orders it before the gather), the earliest needed first; they add cover
        lplan, unplaced, left = place(cover)
        first = firsts(lplan)
        self.pack_at = defaultdict(list)
        if history_gather.bfp4_on():
            todo = sorted(first, key=lambda j: (first[j], j))
            for t, key in enumerate(order):
                if key[0] == "tails":
                    continue
                for j in [j for j in todo if first[j] > t][:PACKS_A_REGION]:
                    self.pack_at[key].append(j)
                    todo.remove(j)
            cover2 = {k: v + PACK_COVER_MS * len(self.pack_at.get(k, ())) for k, v in cover.items()}
            lplan, unplaced, left = place(cover2)
            first = firsts(lplan)
            for t, key in enumerate(order):  # (an item whose first piece moved up to its pack's region: packed at use)
                self.pack_at[key] = [j for j in self.pack_at.get(key, ()) if first[j] > t]
            cover = cover2
        self.lplan, self.unplaced = lplan, unplaced
        for key, units in self.lplan.items():
            for j, a, b in units:
                self.left[j] = self.left.get(j, 0) + 1
        for j, a, b in self.unplaced:
            self.left[j] = self.left.get(j, 0) + 1
        self.cover, self.cover_left = cover, left
        LAST_PLAN.clear()  # (diagnostic: the last chunk's plan)
        LAST_PLAN.update(
            regions={str(k): [(j, a, b) for j, a, b in v] for k, v in self.lplan.items()},
            unplaced=list(self.unplaced),
            cover={str(k): round(v, 3) for k, v in cover.items()},
            packs={str(k): list(v) for k, v in self.pack_at.items() if v},
        )

    def _long_items(self) -> bool:
        """An item whose history (an owner's rows) is more than one attention phase's cover gathers (its gather on the
        gather column would outlast the phase it sits beside)."""
        if not history_gather.bfp4_on():
            return False
        for s, r, ckv in self.items:
            if not isinstance(getattr(ckv, "history_key", None), tuple):
                continue
            h = (int(r.start) // self.chunk) * (ckv.stripe // ckv.ratio)
            if GATHER_FIXED_MS + int(ckv.owners) * h * history_gather.REC / GATHER_BYTES_MS > COVER_ATTN_MS:
                return True
        return False

    def _pack(self, j: int):
        """Item j packed into its own record buffer (once a chunk) and its prefetch marker set; returns its plan."""
        if j in self.packs:
            return self.packs[j]
        s, r, ckv = self.items[j]
        plan = history_gather.prefetch_prepare(ckv, r.start, self.chunk, loc=self._loc_item(ckv, j))
        if plan is not None:
            history_gather.prefetch_mark(plan)
        self.packs[j] = plan
        return plan

    def _long_gathers(self, key) -> list:
        """The planned pieces of region ``key`` as region gathers [(lane, fn(sub))], their items packed now."""
        gathers = []
        for j, a, b in self.lplan.pop(key, []):
            if not self.left.get(j):
                continue
            plan = self._pack(j)
            if plan is None:
                continue

            def g(sub, plan=plan, j=j, a=a, b=b):
                history_gather.prefetch_gather_range(plan, a, b, sub)
                self.left[j] -= 1

            ms = GATHER_FIXED_MS + plan[4] * (b - a) * self._rec / GATHER_BYTES_MS
            gathers.append((self.items[j][1].lane, g, ms))
        STATS["prefetch"] += len(gathers)
        return gathers

    def _long_region(self, key, steps: list, section: str):
        # (the packs after the region's own steps, passing its result through: the gathers are placed by the own
        # steps' time, and the packs' DRAM-bound time covers the last of them)
        packs = [lambda prev, j=j: (self._pack(j), prev)[1] for j in self.pack_at.pop(key, ())]
        n = len(steps)
        steps = list(steps) + packs
        gathers = self._long_gathers(key)
        if not gathers:
            out = None
            for step in steps:
                out = step(out)
            return out
        return self.split.region(
            gathers, steps, section, after_last=True, cover=self.cover.get(key, 0.0) - PACK_COVER_MS * len(packs), own=n
        )

    def _long_flush(self, s: int) -> None:
        """Source s's pieces not gathered yet (a region of the plan did not run): serially, before its reader."""
        rest = 0
        for key in list(self.lplan):
            keep = []
            for j, a, b in self.lplan[key]:
                if self.items[j][0] != s:
                    keep.append((j, a, b))
                    continue
                plan = self._pack(j)
                if plan is not None and self.left.get(j):
                    history_gather.prefetch_gather_range(plan, a, b)
                    self.left[j] -= 1
                    rest += 1
            self.lplan[key] = keep
        keep = []
        for j, a, b in self.unplaced:  # (pieces the plan did not place)
            if self.items[j][0] != s:
                keep.append((j, a, b))
                continue
            plan = self._pack(j)
            if plan is not None and self.left.get(j):
                history_gather.prefetch_gather_range(plan, a, b)
                self.left[j] -= 1
                rest += 1
        self.unplaced = keep
        STATS["flushed"] += rest

    def close(self) -> None:
        self.items = []
        if self.long:
            self.lplan, self.unplaced, self.pack_at = defaultdict(list), [], defaultdict(list)
