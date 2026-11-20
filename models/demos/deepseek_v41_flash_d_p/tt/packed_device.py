# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PACKED=on (PB.3, owner 2026-11-19; bringup/PACKED_DESIGN.md): the packed chunk on the device, v1.

A packed chunk of ``chunk`` rows (8192) holds up to 4 segments (tt/packed.ChunkDescriptor: lane i = natural rows
[i L, (i + 1) L), L = 2048), each continuing its own slot. The token-wise steps of a layer (engram, mHC, norms, the MoE,
the residuals) run once over all rows, exactly as today. The attention steps (q_a, kv, compressor, indexer, attention:
everything that reads a position, a slot's caches or another row) run once per segment, in a loop.

v1 (this file): today's attention steps are reused UNCHANGED for every segment. They compute chip q's rows at
positions start + q R + d (R = rows a chip), so they are right for a segment whose rows sit on chips [0, lane chips):
lane 0. A segment in lane k > 0 is moved there first: its attention input (the chips' own rows, OWN_ROWS) is gathered
over the mesh axis that holds the lanes, rolled by k lanes and partitioned back (``LaneMover.to_lane0``; stock data
moves, exact), and after its attention the output's lane-0 rows go back to lane k (``LaneMover.from_lane0``). A lone
segment in lane 0 (``PackedChunk.trivial``) moves nothing: today's chunk, bit for bit. Per segment the model's
per-chunk fields (``FIELDS``: write_start, skip / snapshot flags, the prefetch markers) and its active slot state are
switched in (``enter`` / ``leave``), and each lane >= 1 reads its own history-gather and index-key buffers (``bind``:
all segments' buffers resident at once, owner 2026-11-19), bound to the segment's slot caches for this chunk only.

v2 (PB.3H / PB.4, not here): per-chip lane arguments in the attention (one call a layer, no moves, no 4x compute).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import ttnn

from . import ccl, history_gather, index_keys
from .ccl import dealloc
from .common import DRAM, SP_AXIS, TP_AXIS
from .cores import sub_cores
from .settings import settings

# the model's per-chunk fields begin_chunk sets, switched per segment
FIELDS = ("_write_from", "_skip", "_snap", "_last", "_req", "_tail_mode", "_tail_at", "_pf_start", "_kpf_start")
ATTN_STEPS = ("q_a", "kv", "compressor", "indexer", "attention")
PAD = 0xFFFFFFFF  # the engine's pad id (K-72)


def packed_on() -> bool:
    """PACKED=on (call-time, bringup/hooks.py batch_call_settings)."""
    return settings.get("PACKED") == "on"


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


class LaneMover:
    """Moves a lane's rows (each chip's own rows, [1, 1, q, W], chip-linear q = r TP + c) to lane 0's chips and back.
    "sp": a lane is whole SP rows (8x4: 2 SP rows; 4x4: 1): one gather over SP on dim 1. "mesh": a lane is part of an
    SP row (2x4, 2x2): gathers over TP then SP, chip-linear blocks. Data moves only (exact)."""

    def __init__(self, mesh, chunk: int, lanes: int):
        self.mesh = mesh
        self.sp, self.tp = int(mesh.shape[SP_AXIS]), int(mesh.shape[TP_AXIS])
        n = self.sp * self.tp
        lane_rows = chunk // lanes
        per = chunk // n
        assert lane_rows % per == 0, (chunk, lanes, n)
        self.lane_chips = lane_rows // per
        self.mode = "sp" if self.lane_chips % self.tp == 0 else "mesh"
        self.unit = self.lane_chips // self.tp if self.mode == "sp" else self.lane_chips  # blocks a lane
        self.blocks = self.sp if self.mode == "sp" else n
        self._zeros = {}

    def gather(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, q, W] each chip -> [1, blocks, q, W] on every chip of the gathered group (block b = SP row b, or
        chip-linear b)."""
        if self.mode == "sp":
            return ccl.all_gather(t, 1, SP_AXIS)
        a = ccl.all_gather(t, 1, TP_AXIS) if self.tp > 1 else t
        b = ccl.all_gather(a, 0, SP_AXIS) if self.sp > 1 else a
        if a is not t and b is not a:
            dealloc(a)
        shape = list(b.shape)
        return ttnn.reshape(b, [1, shape[0] * shape[1], shape[2], shape[3]])

    def partition(self, g: ttnn.Tensor) -> ttnn.Tensor:
        """[1, blocks, q, W] -> each chip its own block [1, 1, q, W]."""
        if self.mode == "sp":
            return ttnn.mesh_partition(g, dim=1, cluster_axis=SP_AXIS, memory_config=DRAM)
        a = ttnn.mesh_partition(g, dim=1, cluster_axis=SP_AXIS, memory_config=DRAM) if self.sp > 1 else g
        b = ttnn.mesh_partition(a, dim=1, cluster_axis=TP_AXIS, memory_config=DRAM) if self.tp > 1 else a
        if a is not g and b is not a:
            dealloc(a)
        return b

    def _slice(self, g: ttnn.Tensor, b0: int, b1: int) -> ttnn.Tensor:
        s = list(g.shape)
        return ttnn.slice(g, [0, b0, 0, 0], [1, b1, s[2], s[3]], memory_config=DRAM, sub_core_grids=sub_cores(g))

    def to_lane0(self, g: ttnn.Tensor, lane: int) -> ttnn.Tensor:
        """The gathered rows ``g`` -> lane ``lane``'s rows on lane 0's chips (every other chip: some lane's rows)."""
        k = lane * self.unit
        assert 0 < k < self.blocks, (lane, self.unit, self.blocks)
        a, b = self._slice(g, k, self.blocks), self._slice(g, 0, k)
        rolled = ttnn.concat([a, b], dim=1, memory_config=DRAM, sub_core_grids=sub_cores())
        dealloc(a)
        dealloc(b)
        out = self.partition(rolled)
        dealloc(rolled)
        return out

    def lane0_part(self, y: ttnn.Tensor) -> ttnn.Tensor:
        """A segment's output on lane 0's chips -> its lane-0 blocks [1, unit, q, W] on every chip."""
        g = self.gather(y)
        part = self._slice(g, 0, self.unit)
        dealloc(g)
        return part

    def zeros(self, like: ttnn.Tensor) -> ttnn.Tensor:
        """[1, unit, q, W] zeros (an empty lane's rows), made once a shape (a constant, never rebuilt a call)."""
        s = list(like.shape)
        key = (self.unit, s[2], s[3], like.dtype)
        z = self._zeros.get(key)
        if z is None or not z.is_allocated():
            z = ttnn.zeros(
                [1, self.unit, s[2], s[3]],
                dtype=like.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=DRAM,
            )
            ccl.pin(self.mesh, z)
            self._zeros[key] = z
        return z

    def assemble(self, parts: list) -> ttnn.Tensor:
        """Lane-ordered [1, unit, q, W] parts (lane L's rows) -> each chip its own rows [1, 1, q, W]."""
        f = ttnn.concat(parts, dim=1, memory_config=DRAM, sub_core_grids=sub_cores())
        out = self.partition(f)
        dealloc(f)
        return out


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
        self.trivial = len(segs) == 1 and segs[0].lane == 0  # (today's chunk: nothing moves)
        self.prefetched = False  # (TtDS41Model.take_engram_rows: once a chunk)
        self.mover = None if self.trivial else LaneMover(model.mesh, model.chunk, desc.lanes)
        # PB.3A (PACKED_ATTN_ONECALL=on, read per chunk): the attention once a layer over every lane (no moves)
        self.oc = None
        if not self.trivial and segs:
            from .packed_onecall import OneCall, onecall_on

            if onecall_on():
                self.oc = OneCall(self)
        # PB.4 (PACKED_MOE=on, read per chunk): the pad rows out of the MoE dispatch (MoEPadSpan; None: every row real
        # or the switch off)
        self.moe_span = None
        if segs and moe_pad_on():
            import torch

            valid = torch.zeros(desc.chunk, dtype=torch.bool)
            for r in segs:
                o = r.lane * desc.lane_rows
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
        for r in self.segs:
            if r.lane == 0:
                continue
            st = r.state
            for i, c in st.compress.items():
                if i in st.shared_compress:
                    continue
                ckv = c.ckv
                if getattr(ckv, "history_rows", None) is not None:
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
        """The layer's attention output for every row of the packed chunk (each lane its segment's; an empty lane's
        rows zero). ``x``: the attention input (attn_norm), the chips' own rows; not consumed."""
        if self.trivial:
            return self._seg_steps(block, self.segs[0], x)
        if self.oc is not None:  # PB.3A (PACKED_ATTN_ONECALL=on): once over every lane
            return self.oc.attention(block, x)
        mv = self.mover
        g = mv.gather(x) if any(r.lane for r in self.segs) else None
        xs = {r.lane: (x if r.lane == 0 else mv.to_lane0(g, r.lane)) for r in self.segs}
        if g is not None:
            dealloc(g)
        parts = {}
        for r in self.segs:
            xr = xs[r.lane]
            y = self._seg_steps(block, r, xr)
            parts[r.lane] = mv.lane0_part(y)  # (before the next segment: y may be a load-time site, HOST_ALLOC)
            dealloc(y)
            if xr is not x:
                dealloc(xr)
        like = next(iter(parts.values()))
        out = mv.assemble([parts[lane] if lane in parts else mv.zeros(like) for lane in range(self.desc.lanes)])
        for p in parts.values():
            dealloc(p)
        return out

    def compress_only(self, block, x: ttnn.Tensor) -> None:
        """DECODER_SKIP (kv_only, layer 20): the compressor of every segment (the KV head's served writes)."""
        if self.trivial:
            self._seg_steps(block, self.segs[0], x, names=("compressor",))
            return
        if self.oc is not None:  # PB.3A (PACKED_ATTN_ONECALL=on)
            self.oc.compress_only(block, x)
            return
        mv = self.mover
        g = mv.gather(x) if any(r.lane for r in self.segs) else None
        for r in self.segs:
            xr = x if r.lane == 0 else mv.to_lane0(g, r.lane)
            self._seg_steps(block, r, xr, names=("compressor",))
            if xr is not x:
                dealloc(xr)
        if g is not None:
            dealloc(g)


def run_chunk(model, desc, tokens, states: list, layers: list, last=None, on_layer=None):
    """Harness (tests): one chunk of ``desc`` through ``layers`` as PACKED says (read per chunk). on: begin_packed and
    the packed layers; off: today's single-request chunk, which needs exactly one live segment, in lane 0 (the rows of
    a packed chunk's lane 0 are today's chunk rows). ``tokens``: natural-order ids [chunk], PAD_ID past each segment's
    end and in empty lanes. ``on_layer(i, h)`` after each layer (h is freed after it returns). Returns nothing."""
    from .attention import prefill_end
    from .model import SKIPPED

    ids = tokens.reshape(-1).long().clone()
    for lane, s in desc.live():  # (PAD_ID past each segment's run end, as the adapter's read of the engine input)
        n = max(0, prefill_end(s.end) - s.start)
        ids[lane * desc.lane_rows + n : (lane + 1) * desc.lane_rows] = PAD
    if packed_on():
        model.begin_packed(desc, ids, states, last=last)
        try:
            h = model.embed(ids) if model._packed.segs else SKIPPED
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
    got = torch.cat([ttnn.to_torch(dev[r * tp]).reshape(-1).to(torch.int64) & 0xFFFFFFFF for r in range(sp)])
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
    nat = desc.pack_ids(ids_per_lane, pad=PAD)
    return ttnn.from_torch(
        nat.reshape(sp, 1, desc.chunk // sp).to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])
        ),
    )
