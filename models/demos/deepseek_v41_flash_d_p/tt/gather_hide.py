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

    def region(self, gathers: list, steps: list, section: str):
        """``gathers`` [(lane, fn(sub))] (each enqueues one gather on sub-device 1) beside the compute ``steps`` (each
        fn(previous result), sub-device 0); returns the last step's result. The first gather before the steps
        (its fence waits for the other chips, the compute needs no one: V3's order), gather j >= 1 before step
        min(n - 1, ceil(j n / k)). ``section``: the profile section the steps are charged to (the gathers: "history_gather").
        """
        mesh = self.mesh
        k, n = len(gathers), len(steps)
        at = {}
        for j, g in enumerate(gathers):
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
            if st is not None:
                log.append(("end_compute", None, st("compute")))
                log.append(("end_gather", gathers[-1][0], st("gather")))
                LOG.append(log)
            return out
        finally:
            mesh.clear_loaded_sub_device_manager()  # (joins both sub-devices)
            STATS["regions"] += 1


DEC_WEIGHT = 0.5  # the index-key decodes' region against an attention phase (PB.3H walls: ~0.6-1 vs ~1.3-1.5 ms)


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

    # ------------------------------------------------------------------ buffers
    def _loc(self, ckv, k: int):
        if not history_gather.bfp4_on():
            return None
        R = int(ckv.slab.rows)
        return _held(
            _LOC, (id(self.mesh), R, k), self.mesh, lambda: history_gather.HistoryBuffer(self.mesh, R, None, True)
        )

    def _small(self, plans: list):
        """The combined tail gather's output: [owners M, width] (M: every plan's rows together)."""
        buf, owners = plans[0][1], plans[0][5]
        rows = owners * sum(p[4] - p[3] for p in plans)
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
        if not plans:
            return serial()
        STATS["tails"] += len(plans)
        # ONE gather for every segment's tail (each owner's rows of every segment, segment after segment): a tail is
        # latency-bound (~0.45 ms on the gather column for 1.3 MB, PB.3H stamps), four back to back stalled the steps
        small = self._small(plans)
        x = history_gather.tails_input(plans)
        out = self.split.region([(-1, lambda sub: history_gather.tails_gather(plans, x, small, sub))], steps, "indexer")
        history_gather.tails_finish(plans, small)
        return out

    def close(self) -> None:
        self.items = []
