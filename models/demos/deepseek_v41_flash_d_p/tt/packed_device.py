# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PACKED=on (PB.3, owner 2026-11-19; bringup/PACKED_DESIGN.md): the packed chunk on the device.

A packed chunk holds up to GLUED_SEGS segments, each continuing its own slot. The token-wise steps of a layer (engram,
mHC, norms, the MoE, the residuals) run once over all rows, exactly as today. PB.9G (OWNER 2026-11-21, PACKED_CHIPS=on):
every packed chunk runs on CHIP-ALIGNED LANES (tt/packed.GluedChunk(align = rows a chip): each segment owns whole
chips) and its attention steps (q_a, kv, compressor, indexer, attention) run ONCE a layer over every segment
(tt/packed_glued.GluedCall: each chip at its own segment's positions, slot caches and history; per-chunk tables
derived on the device from the chunk header, tt/glued_derive.py). A lone segment at row 0 (``PackedChunk.trivial``) is
today's chunk, bit for bit. Per segment the model's per-chunk fields (``FIELDS``: write_start, skip / snapshot flags,
the prefetch markers) and its active slot state are switched in (``enter`` / ``leave``), and each segment >= 1 reads
its own history-gather and index-key buffers (``bind``: all segments' buffers resident at once, owner 2026-11-19),
bound to the segment's slot caches for this chunk only. PACKED_CHIPS=off: PB.9X's back-to-back glued layout through
the same one call. The fixed 2048-row lanes (PB.3's lane moves) and path (b) (PB.7's per-segment row moves) are
removed: a chunk the one call cannot run raises.
"""

from __future__ import annotations

import importlib as _importlib
from dataclasses import dataclass, field

import ttnn

# DS41-311 (PB.HG): the package attribute fresh after a runner reload
_importlib.import_module(__package__ + ".history_gather")
from . import ccl, history_gather, index_keys
from .common import DRAM, SP_AXIS, TP_AXIS
from .settings import settings

# the model's per-chunk fields begin_chunk sets, switched per segment
FIELDS = ("_write_from", "_skip", "_snap", "_last", "_req", "_tail_mode", "_tail_at", "_pf_start", "_kpf_start")
ATTN_STEPS = ("q_a", "kv", "compressor", "indexer", "attention")
PAD = 0xFFFFFFFF  # the engine's pad id (K-72)


def packed_on() -> bool:
    """PACKED=on (call-time, bringup/hooks.py batch_call_settings)."""
    return settings.get("PACKED") == "on"


def glued_on() -> bool:
    """PB.7 PACKED_GLUED=on (call-time, bringup/hooks.py batch_call_settings): chunks are tt/packed.GluedChunk."""
    return settings.get("PACKED_GLUED") == "on"


def chips_on() -> bool:
    """PB.9G PACKED_CHIPS=on (call-time, bringup/hooks.py batch_call_settings): every packed chunk on chip-aligned
    lanes (each segment owns whole chips; tt/packed.GluedChunk(align = rows a chip)). off: PB.9X's back-to-back glued
    layout (the A/B's previous side)."""
    return settings.get("PACKED_CHIPS") == "on"


STATS = {"onecall": 0, "trivial": 0, "fallback": 0}  # PB.9G: packed chunks by path since load (fallback: none exists;
# a chunk the one call cannot run raises, so it stays 0: the walls / no-rebuild record it as packed_fallback_chunks*)
CHIP_ROWS = [0]  # rows a chip of the loaded model's chunk (TtDS41Model: chunk / (SP TP)); as_run_desc's default


def chip_rows(desc, mesh=None) -> int:
    """Rows a chip of ``desc``'s chunk on ``mesh`` (else the loaded model's)."""
    if mesh is not None:
        return int(desc.chunk) // (int(mesh.shape[SP_AXIS]) * int(mesh.shape[TP_AXIS]))
    assert CHIP_ROWS[0] > 0, "as_run_desc: no mesh and no loaded model (CHIP_ROWS)"
    return CHIP_ROWS[0]


def as_run_desc(desc, mesh=None):
    """The descriptor a chunk runs as. PB.9G (OWNER 2026-11-21): every packed chunk is a glued chunk; with
    PACKED_CHIPS=on its segments sit on chip-aligned lanes (each segment its own whole chips, starts and lengths
    multiples of the rows a chip, its last chip padded; GluedChunk.align), off: back to back (PB.9X). A fixed-lane header
    (tt/packed.ChunkDescriptor) with more than one lane is the glued chunk of its live segments in lane order (the
    fixed 2048-row lane path is removed); a one-lane header is today's single-request chunk."""
    if getattr(desc, "glued", False):
        assert glued_on(), "a glued chunk needs PACKED_GLUED=on"
        return desc.aligned(chip_rows(desc, mesh) if chips_on() else 0)
    if desc.lanes > 1:
        if not glued_on():
            raise ValueError(
                "PB.9G: the fixed 2048-row lane path is removed; a packed header runs as a glued chunk (PACKED_GLUED=on)"
            )
        g = desc.as_glued(settings.get("PREFILL_END"), align=chip_rows(desc, mesh) if chips_on() else 0)
        return desc if g is None else g
    return desc


def routed(desc, run_desc) -> bool:
    """A descriptor that runs as another layout (its tokens / states / input re-laid: run_chunk, make_input)."""
    return run_desc is not desc and getattr(run_desc, "glued", False)


def seg_index(desc, lane: int) -> int:
    """The run descriptor's segment index of ``desc``'s lane (a fixed-lane header: its live lanes in order)."""
    if getattr(desc, "glued", False):
        return int(lane)
    return [ln for ln, _ in desc.live()].index(lane)


def relay_ids(desc, run_desc, ids):
    """Model-row ids [chunk] of ``desc`` -> those of ``run_desc`` (each segment's computed rows moved; PAD elsewhere)."""
    import torch

    if run_desc is desc:
        return ids
    if not getattr(desc, "glued", False):
        return run_desc.pack_ids(desc.lane_seg_ids(ids), pad=PAD)
    out = torch.full((desc.chunk,), PAD, dtype=torch.int64)
    for k in range(desc.lanes):
        n = desc.rows(k)
        out[run_desc.lane_off(k) : run_desc.lane_off(k) + n] = ids[desc.lane_off(k) : desc.lane_off(k) + n]
    return out


@dataclass
class SegRun:
    """One segment of a packed chunk on the device: its lane, the descriptor's Segment, its slot state, its run end
    (PREFILL_END) and the model's per-chunk fields for it (None: a segment that computes nothing, cap <= start)."""

    lane: int
    seg: object
    state: object
    run_end: int
    fields: dict | None = None
    restore: list = field(default_factory=list)  # (object, attribute, value) put back at the chunk's end

    @property
    def start(self) -> int:
        return int(self.seg.start)


_LANE_KEYS: set = set()  # (mesh id, kind, key) of the per-lane buffers acquired (held for the process: v1)


class MoEPadSpan:
    """PB.4 (PACKED_MOE=on): a packed chunk's pad rows out of the MoE dispatch. The block's moe_head (fork
    ttnn.bringup.moe_block, padded_rows.hpp) takes a padded chunk as two one-word uint32 DRAM tensors (actual_start,
    actual_end) and each of its kernels reads them for its OWN SP row only (padded_chunk_rows(start, end, sp_row, ..):
    at start 0 an SP row r holds clamp(end - r W, 0, W) real rows, a prefix). So per device (0, r W + n_r) gives SP row
    r exactly its n_r real rows (ChunkDescriptor.moe_real_prefix): pad rows get no route (not dispatched, not computed
    by the routed experts, not combined; the planner counts only real rows). The words are two persistent tensors of
    the mesh (sharded per device), rewritten only when the counts change (a host write at begin_packed, before the
    chunk's first layer: no transfer inside a layer). A chunk whose rows are all real gets no span (the unpadded
    program, as PACKED_MOE=off)."""

    _BY: dict = {}  # id(mesh) -> (mesh, MoEPadSpan)

    @classmethod
    def of(cls, mesh, chunk: int) -> "MoEPadSpan":
        hit = cls._BY.get((id(mesh), chunk))
        if hit is None or hit[0] is not mesh:
            hit = cls._BY[(id(mesh), chunk)] = (mesh, cls(mesh, chunk))
        return hit[1]

    def __init__(self, mesh, chunk: int):
        assert SP_AXIS == 0, "MoEPadSpan: SP on mesh rows (device order = chip-linear r tp + c)"
        self.mesh, self.chunk = mesh, int(chunk)
        self.sp, self.tp = int(mesh.shape[SP_AXIS]), int(mesh.shape[TP_AXIS])
        self.words = None  # (start, end) device tensors
        self.last = None  # the end words they hold

    def _words(self, vals, device: bool):
        """[len(vals), 1, 1, 1] uint32 row-major sharded per device (device i: vals[i]): on the mesh in DRAM, or
        (device False) the mesh host tensor copy_host_to_device_tensor writes into it."""
        import torch

        t = torch.tensor(vals, dtype=torch.int64).reshape(len(vals), 1, 1, 1)
        kw = dict(device=self.mesh, memory_config=DRAM) if device else {}
        return ttnn.from_torch(
            t,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
            **kw,
        )

    def span(self, desc, valid) -> dict | None:
        """{actual_start, actual_end} for moe_head (None: every row real). ``valid``: [chunk] bool, the rows the
        chunk computes (each live segment's [start, run_end) in its lane)."""
        n, _exact = desc.moe_real_prefix(self.sp, valid)
        w = self.chunk // self.sp
        if all(v == w for v in n):
            return None
        ends = [r * w + n[r] for r in range(self.sp) for _ in range(self.tp)]
        if self.words is None:  # (first padded chunk of the process: two persistent words, pinned)
            self.words = (self._words([0] * len(ends), True), self._words(ends, True))
            for t in self.words:
                ccl.pin(self.mesh, t)
            self.last = ends
        elif ends != self.last:
            ttnn.copy_host_to_device_tensor(self._words(ends, False), self.words[1])
            self.last = ends
        return dict(actual_start=self.words[0], actual_end=self.words[1])


def moe_pad_on() -> bool:
    """PACKED_MOE=on (call-time, bringup/hooks.py batch_call_settings), on the MoE block that takes a padded chunk
    (MOE_IMPL=block, MOE_BLOCK_SRC=block: tt/moe_block_v2_adapter; the other MoE paths run every row as off)."""
    return (
        settings.get("PACKED_MOE") == "on"
        and settings.get("MOE_IMPL") == "block"
        and settings.get("MOE_BLOCK_SRC") == "block"
    )


class PackedChunk:
    """The device side of one packed chunk (TtDS41Model.begin_packed .. end_packed)."""

    def __init__(self, model, desc, segs: list, idle: list):
        self.model, self.desc = model, desc
        self.segs = segs  # SegRun of the segments that compute (lane order)
        self.idle = idle  # SegRun of the segments with cap <= start (snapshot at cap and acks only)
        self.glued = getattr(desc, "glued", False)  # PB.7: a glued chunk that is not a lane chunk (moves by rows)
        self.header = getattr(model, "_packed_inp", None)  # PB.7C: the engine input carrying the chunk header
        self.trivial = len(segs) == 1 and segs[0].lane == 0 and desc.lane_off(0) == 0  # (today's chunk: nothing moves)
        self.prefetched = False  # (TtDS41Model.take_engram_rows: once a chunk)
        self.mover = None  # (PB.9G: no row moves; path (b) and the fixed lanes are removed)
        # PB.9G: every other chunk's attention once a layer over every segment (tt/packed_glued.GluedCall: the
        # one-call path, per-chunk tables derived on the device from the chunk header); a chunk it cannot run raises
        self.oc = None
        if not self.trivial and segs:
            if not self.glued:
                raise ValueError(
                    "PB.9G: a fixed-lane packed chunk (the 2048-row lane path is removed): run as_run_desc"
                )
            from .packed_glued import GluedCall, glued_call_ok

            why = glued_call_ok(model, desc, segs)
            if why is not None:
                raise ValueError(f"PB.9G: this packed chunk is not supported ({why}); no fallback path")
            self.oc = GluedCall(self)
        if segs:
            STATS["trivial" if self.trivial else "onecall" if self.oc is not None else "fallback"] += 1
        # PB.7C: the dynamic parts' chunk kind (tt/page_copy.KIND; a fixed-lane chunk keeps PB.7B's programs)
        from .page_copy import KIND

        KIND[0] = "lanes" if not (self.trivial or self.glued) else "glued" if self.glued else "trivial"
        # PB.4 (PACKED_MOE=on, read per chunk): the pad rows out of the MoE dispatch (MoEPadSpan; None: every row real
        # or the switch off)
        self.moe_span = None
        dv = getattr(self.oc, "dv", None)
        if segs and moe_pad_on() and dv is not None:  # PB.7C attempt 3: the end words derived from the header
            self.moe_span = dv.moe_span(sum(max(0, r.run_end - r.start) for r in segs))
        elif segs and moe_pad_on():
            import torch

            valid = torch.zeros(desc.chunk, dtype=torch.bool)
            for r in segs:
                o = desc.lane_off(r.lane)
                valid[o : o + max(0, r.run_end - r.start)] = True
            self.moe_span = MoEPadSpan.of(model.mesh, desc.chunk).span(desc, valid)

    # ------------------------------------------------------------------ per segment model fields
    def enter(self, r: SegRun) -> None:
        m = self.model
        m.state = r.state
        for f, v in r.fields.items():
            setattr(m, f, v)

    def leave(self, r: SegRun) -> None:
        m = self.model
        for f in FIELDS:
            r.fields[f] = getattr(m, f)

    # ------------------------------------------------------------------ per-lane buffers
    def bind(self) -> None:
        """Every lane >= 1 reads its own history-gather and index-key buffers (lane 0: today's), bound to the slot's
        caches for this chunk (``unbind`` puts the keys back). Acquired once a process (v1: held until exit)."""
        mesh = self.model.mesh
        owned = getattr(self.oc, "owned", False)  # PB.HG: one history buffer a source, no lane buffers
        for r in self.segs:
            if r.lane == 0:
                continue
            st = r.state
            for i, c in st.compress.items():
                if i in st.shared_compress:
                    continue
                ckv = c.ckv
                if getattr(ckv, "history_rows", None) is not None and not owned:
                    orig = ckv.history_key
                    key = (orig if orig is not None else ckv.history_rows, "lane", r.lane)
                    tag = (mesh.id(), "h", key)
                    if tag not in _LANE_KEYS:
                        history_gather.acquire(
                            mesh,
                            ckv.history_rows,
                            key,
                            owners=ckv.owners,
                            tails=ckv._history_tails,
                            bfp4=ckv._history_bfp4,
                        )
                        _LANE_KEYS.add(tag)
                    r.restore.append((ckv, "history_key", orig))
                    ckv.history_key = key
                ik = c.index_k
                if getattr(ik, "k_served", False):
                    orig = ik.key_layer
                    ik.key_layer = (orig, "lane", r.lane)
                    tag = (mesh.id(), "k", index_keys._key(ik))
                    if tag not in _LANE_KEYS:
                        index_keys.acquire(ik)
                        _LANE_KEYS.add(tag)
                    r.restore.append((ik, "key_layer", orig))

    def unbind(self) -> None:
        from .page_copy import KIND

        KIND[0] = "any"
        if self.oc is not None:
            self.oc.close()
        for r in self.segs:
            for obj, name, v in reversed(r.restore):
                setattr(obj, name, v)
            r.restore.clear()

    # ------------------------------------------------------------------ prefetch (P.75 / P.76, every segment)
    def prefetch(self) -> None:
        """At the chunk's first engram wait (as today): every segment's history and index-key prefetch, in lane order,
        each into its own lane's buffers."""
        m = self.model
        hide = self.oc.hide if self.oc is not None else None
        if hide is not None and hide.dec_layer is not None:  # PB.3H: the decodes beside prefetch gathers

            def step(_, r):
                self.enter(r)
                if m._kpf_start is not None:
                    m.prefetch_index_keys()
                self.leave(r)

            hide.decodes([lambda prev, r=r: step(prev, r) for r in self.segs])
            return
        for r in self.segs:
            self.enter(r)
            if m._pf_start is not None:
                m.prefetch_history()
            if m._kpf_start is not None:
                m.prefetch_index_keys()
            self.leave(r)

    # ------------------------------------------------------------------ attention per segment
    def _seg_steps(self, block, r: SegRun, x: ttnn.Tensor, names=ATTN_STEPS):
        """Today's attention steps of ``block`` for segment ``r`` on ``x`` (its rows on lane 0's chips); returns the
        attention output (or None when ``names`` has no attention)."""
        from models.demos.common.bringup.reference.interface import Ctx

        from .model import _free, _tensors

        m = self.model
        self.enter(r)
        rows = m.chunk
        ctx = Ctx(block.layer, r.start, rows, r.state, extra={"end": r.run_end})
        att = getattr(block, "attention", None)
        region = getattr(att, "region", None)
        j = self.segs.index(r)
        if region is not None and j:  # (its own window region: the neighbour sends of back-to-back segments, DS41-324)
            att.region = self._region(att, j)
        env = {"attn_norm": x}
        for s in block.steps:
            if s.name in names:
                env[s.output] = block.fns[s.name](ctx, *[env[n] for n in s.inputs])
        if region is not None:
            att.region = region
        y = env.get("attn_out")
        keep = {id(x)} | {id(t) for t in _tensors(y)}
        for n, v in env.items():
            if n not in ("attn_norm", "attn_out"):
                _free(v, keep)
        ctx.extra.clear()
        st = r.state
        if st.memo is not None and "attention" in names:  # (INDEXER_GLUE=on: as TtDS41Block.__call__'s end)
            for slot in m.memo_last.get(block.layer, ()):
                st.memo.drop(slot)
        self.leave(r)
        return y

    @staticmethod
    def _region(att, j: int) -> ttnn.Tensor:
        """The window region of the layer's j-th segment (j >= 1; 0 is the layer's own): made once (a load-time-like
        buffer, zeros, the region's shape), kept with the attention module."""
        regs = att.__dict__.setdefault("packed_regions", {})
        t = regs.get(j)
        if t is None or not t.is_allocated():
            r = att.region
            t = ttnn.zeros(list(r.shape), dtype=r.dtype, layout=r.layout, device=r.device(), memory_config=DRAM)
            ccl.pin(r.device(), t)
            regs[j] = t
        return t

    def attention(self, block, x: ttnn.Tensor) -> ttnn.Tensor:
        """The layer's attention output for every row of the packed chunk (each segment's rows its own; pad rows
        computed, never read). ``x``: the attention input (attn_norm), the chips' own rows; not consumed."""
        if self.trivial:
            return self._seg_steps(block, self.segs[0], x)
        return self.oc.attention(block, x)  # PB.9G: once over every segment

    def compress_only(self, block, x: ttnn.Tensor) -> None:
        """DECODER_SKIP (kv_only, layer 20): the compressor of every segment (the KV head's served writes)."""
        if self.trivial:
            self._seg_steps(block, self.segs[0], x, names=("compressor",))
            return
        self.oc.compress_only(block, x)


def warm_glued_chunks(chunk: int, slots: list, max_seq: int, mesh=None, replay: bool = True) -> list:
    """PB.9X / PB.9G (rule 20 / OV-68: warm-up is a finite, enumerable set): the glued chunks whose programs cover every
    glued layout. What a layout's programs still depend on is bounded: the number of segments that write compressed
    rows (TtServedRows.write_lanes: one row gather of n W rows, per-segment slices j < n; n <= GLUED_MAX), a fresh
    request's window (cap < 128: the snapshot send's geometry), a continued one's history (prefetch + fixed-size tails;
    PB.6L's range-gather pieces of the longest histories), an encoder replay (PB.9G: in the one call). So: for every
    segment count n, a chunk of fresh requests and one of continued requests (histories of 2 (k + 1) chunks), the last
    segment 17 rows short; continued chunks whose segments are all replays (write_start = start + 128); exact ones
    near the slot's end; a chunk with a 1-token and an idle segment. PACKED_CHIPS=on: every segment on whole chips
    (lengths up to its share of the chips). ``slots``: distinct serving slots (as many as GLUED_MAX, else fewer counts).
    Each chunk's segments fit ``max_seq`` (else the chunk is skipped)."""
    from .packed import GLUED_MAX, GluedChunk, Segment

    out = []
    align = chip_rows(GluedChunk(chunk, (Segment(0, 0, 1),)), mesh) if chips_on() else 0
    n_max = min(GLUED_MAX, len(slots))
    if align:
        n_max = min(n_max, chunk // align)
    far = (max_seq - 2 * chunk) // 64 * 64  # (the longest histories: PB.6L's range-gather pieces start there)
    hs = ("fresh", "cont", "replay") if replay else ("fresh", "cont")  # (replay: DECODER_SKIP=decode only)
    kinds = [(n, h, odd) for n in range(1, n_max + 1) for h in hs for odd in (True,)]
    kinds += [(n, h, False) for n in (2, 4) if n <= n_max for h in ("fresh", "cont", "far")]  # (exact shares)
    kinds += [(n, "far", True) for n in (1, 3, n_max)]
    for n, h, odd in kinds:
        if align:
            L = (chunk // align // n) * align
        else:
            L = (chunk // n) // 64 * 64
        segs = []
        for k in range(n):
            a = {"fresh": 64 * k, "cont": 2 * chunk * (k + 1) + 64 * k, "replay": 2 * chunk * (k + 1) + 64 * k}
            a = a.get(h, far - 64 * 300 * k)
            if a < 0 or a + chunk > max_seq:  # DS41-372: the fits-check before the Segment (it refuses a start < 0)
                break
            ws = a + 128 if h == "replay" else None
            segs.append(Segment(int(slots[k]), a, a + L - (17 if odd and k == n - 1 else 0), ws))
        if len(segs) == n:
            out.append(GluedChunk(chunk, tuple(segs), align=align))
    if n_max >= 3:
        out.append(
            GluedChunk(
                chunk,
                (Segment(int(slots[0]), 128, 129), Segment(int(slots[1]), 64, 764), Segment(int(slots[2]), 0, 1)),
                align=align,
            )
        )
    h = (chunk // 2 // align * align) if align else (chunk // 2 // 64 * 64)
    if settings.get("PACKED_WARM_CONT1") == "on" and slots and 0 < h and h + chunk <= max_seq:
        # PB.FX (DS41-373): a lone segment runs today's single path (PackedChunk.trivial); at a continuing start after
        # a partial (non-whole) chunk of its slot its window carry is read from the slot's ring
        # (attention._ring_carry_pages: to_layout, the owner gather, page copies, untilize, slice_write into the kept
        # tail), a kind no chunk above reaches (their lone segments start fresh or past an unwritten history). Its
        # programs depend on no position (fixed 128-row carry, page-copy offsets / counts as values): one pair, the
        # second chunk at a non-chunk-aligned start, warms it for every start, owner chip and previous layout.
        s = int(slots[0])
        out.append(GluedChunk(chunk, (Segment(s, 0, h),), align=align))
        out.append(GluedChunk(chunk, (Segment(s, h, chunk),), align=align))
    return out


_GLUED_IDS: dict = {}  # id(mesh) -> (mesh, GluedIdsReorder)


def glued_dev_ids(model, desc, inp):
    """PB.9X (OV-69, POS_DYNAMIC=on): a glued chunk's model-row embedding ids made ON THE DEVICE from the engine input
    that carries its header (tt/page_copy.GluedIdsReorder: one SP all_gather + a one-core kernel deriving every
    segment's engine / model offsets from the header); None when the input has no header (tests driving run_chunk with
    host ids) or POS_DYNAMIC=off: the host ids are uploaded then. A new tensor (``model.embed(dev_ids=)`` consumes it).
    """
    from .glued_derive import HDR_WORDS
    from .packed import KV_BLOCK
    from .page_copy import GluedIdsReorder
    from .page_copy import on as pos_dynamic

    if inp is None or not getattr(desc, "glued", False) or not pos_dynamic():
        return None
    sp = int(model.mesh.shape[SP_AXIS])
    if int(inp.shape[-1]) != int(model.chunk) // sp + HDR_WORDS:
        return None
    key = (id(model.mesh), int(getattr(desc, "align", 0)))
    hit = _GLUED_IDS.get(key)
    if hit is None or hit[0] is not model.mesh:
        hit = _GLUED_IDS[key] = (
            model.mesh,
            GluedIdsReorder(model.mesh, model.chunk, model.vocab, HDR_WORDS, KV_BLOCK, int(getattr(desc, "align", 0))),
        )
    return hit[1](inp, desc.prefill_end == "cap")


def run_chunk(model, desc, tokens, states: list, layers: list, last=None, on_layer=None, inp=None):
    """Harness (tests): one chunk of ``desc`` through ``layers`` as PACKED says (read per chunk). on: begin_packed and
    the packed layers; off: today's single-request chunk, which needs exactly one live segment, in lane 0 (the rows of
    a packed chunk's lane 0 are today's chunk rows). ``tokens``: natural-order ids [chunk], PAD_ID past each segment's
    end and in empty lanes. ``on_layer(i, h)`` after each layer (h is freed after it returns). ``inp``: the engine input
    (make_input; PB.7C: a glued chunk's header rides in it). Returns nothing."""
    from .attention import prefill_end
    from .model import SKIPPED

    ids = tokens.reshape(-1).long().clone()
    rd = as_run_desc(desc, model.mesh)  # PB.9G: a glued chunk (chip-aligned lanes or back to back)
    if routed(desc, rd):  # a fixed-lane header / another glued layout re-laid: ids -> the run layout's model rows,
        # states in segment order (the glued chunk's "lane" k = its k-th segment)
        ids = relay_ids(desc, rd, ids)
        if not getattr(desc, "glued", False):
            states = [states[lane] for lane, _ in desc.live()]
    desc = rd
    if getattr(desc, "glued", False):
        ids[desc.used_rows() :] = PAD
    else:
        for lane, s in desc.live():  # (PAD_ID past each segment's run end, as the adapter's read of the engine input)
            n = max(0, prefill_end(s.end) - s.start)
            ids[lane * desc.lane_rows + n : (lane + 1) * desc.lane_rows] = PAD
    if packed_on():
        model.begin_packed(desc, ids, states, last=last, inp=inp)
        try:
            dev_ids = glued_dev_ids(model, desc, inp) if model._packed.segs and layers and layers[0] == 0 else None
            h = model.embed(ids, dev_ids=dev_ids) if model._packed.segs else SKIPPED  # (PB.9X: device ids, OV-69)
            for i in layers:
                h2 = model.layer(i, h, 0, None)
                if h2 is not h:
                    model.free(h)
                h = h2
                if on_layer is not None:
                    on_layer(i, h)
            model.free(h)
        finally:
            model.end_packed()
        return
    live = desc.live()
    assert len(live) == 1 and live[0][0] == 0, "PACKED=off: one segment, in lane 0 (today's chunk)"
    _, s = live[0]
    st = states[0]
    run_end = prefill_end(s.end)
    assert run_end > s.start, "PACKED=off harness: a computed chunk (cap > start)"
    model.state = st
    st.note_tokens(ids, s.start, run_end)
    model.begin_chunk(ids, s.start, last=last, write_start=s.gws)
    h = model.embed(ids)
    for i in layers:
        h2 = model.layer(i, h, s.start, st, run_end)
        model.free(h)
        h = h2
        if on_layer is not None:
            on_layer(i, h)
    model.free(h)


def ids_from_input(mesh, inp, desc):
    """The engine's packed input (uint32 [SP, 1, W] in lane-natural order, sharded over SP, replicated over TP) -> the
    chunk's natural-order ids [chunk] int64, PAD past each segment's run end (PREFILL_END) and in empty lanes: one read
    of mesh column 0's rows (the serving path's one readback a chunk, as the adapter's _host_ids)."""
    import torch

    from .attention import prefill_end

    sp, tp = int(mesh.shape[SP_AXIS]), int(mesh.shape[TP_AXIS])
    dev = ttnn.get_device_tensors(inp)
    w = int(desc.chunk) // sp  # (PB.7C: a glued chunk's input carries its header after each row's ids)
    if settings.get("PACKED_HOST_FAST") == "on":  # PB.3K: column 0's reads enqueued back to back, ONE wait (the last
        # read blocks; the queue completes them in order, so every earlier one has landed when it returns)
        hs = [ttnn.from_device(dev[r * tp], blocking=r == sp - 1) for r in range(sp)]
        got = torch.cat([ttnn.to_torch(h).reshape(-1)[:w].to(torch.int64) & 0xFFFFFFFF for h in hs])
    else:
        got = torch.cat([ttnn.to_torch(dev[r * tp]).reshape(-1)[:w].to(torch.int64) & 0xFFFFFFFF for r in range(sp)])
    rd = as_run_desc(desc, mesh)
    if routed(desc, rd):  # the input is the run layout's: its model rows -> ``desc``'s rows (lane-natural / glued)
        nat = torch.full((desc.chunk,), PAD, dtype=torch.int64)
        mid = rd.model_ids(got, pad=PAD)
        for k, (lane, _) in enumerate(desc.live()):
            n = rd.rows(k)
            nat[desc.lane_off(lane) : desc.lane_off(lane) + n] = mid[rd.lane_off(k) : rd.lane_off(k) + n]
        return nat
    if getattr(desc, "glued", False):  # PB.7: engine rows back to back -> model rows (each segment's computed rows)
        return desc.model_ids(got, pad=PAD)
    nat = torch.full((desc.chunk,), PAD, dtype=torch.int64)
    lr = desc.lane_rows
    for lane, s in desc.live():
        n = max(0, prefill_end(s.end) - s.start)
        nat[lane * lr : lane * lr + n] = got[lane * lr : lane * lr + n]
    return nat


def make_input(mesh, desc, ids_per_lane):
    """The engine's H2D payload of a packed chunk (host side: tests, compile): uint32 [SP, 1, W], lane-natural order
    (DECIDED PB.3), PAD in pad rows and empty lanes."""
    import torch

    sp = int(mesh.shape[SP_AXIS])
    rd = as_run_desc(desc, mesh)
    if routed(desc, rd):  # the run layout is what tt-d-gen sends (PB.9G: chip-aligned lanes)
        if not getattr(desc, "glued", False):
            ids_per_lane = [ids_per_lane[lane] for lane, _ in desc.live()]
        desc = rd
    glued = getattr(desc, "glued", False)
    if glued:  # PB.7: engine order, segments back to back (drafts/tt_d_gen_packed_ask.md)
        nat = desc.engine_ids(ids_per_lane, pad=PAD)
    else:
        nat = desc.pack_ids(ids_per_lane, pad=PAD)
    rows = nat.reshape(sp, 1, desc.chunk // sp)
    if glued and settings.get("PACKED_GLUED_DYNAMIC") == "on":  # PB.7C attempt 3: the chunk header (tt-d-gen's G1
        # fields) after every SP row's ids, in the same upload (tt/glued_derive derives every per-chunk table from it)
        from .glued_derive import HDR_WORDS, header_words

        hdr = torch.tensor(header_words(desc), dtype=torch.int64).view(1, 1, HDR_WORDS).expand(sp, 1, HDR_WORDS)
        rows = torch.cat([rows, hdr], dim=-1)
    return ttnn.from_torch(
        rows.to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])
        ),
    )
