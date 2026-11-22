# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.T (OWNER 2026-11-21): SLOT ISOLATION AT CHIP BOUNDARIES, the harness of test_packed_fast_tier and
test_packed_deep_sweep.

Contract (bit-exact, no tolerance): on chip-aligned lanes (PB.9G: rows of two requests never share a chip) each
segment's
    outputs        every computed layer's output rows of the segment (streams and pre, its computed rows),
    served caches  every served record of its slot over [0, cap(end)) after the chunk (window rows / the served
                   128-row snapshot, compressed KV, index keys; every layer) and its host state (engram tails),
    acks           the layer-completion acks the serving entry gives for it (layer range, start, end, in order)
equal those of THE SAME REQUEST ALONE (a one-segment chunk through the same serving entry, in a slot nothing else
ever used, at the same chunk boundaries), whatever its neighbours (steady / replay / split / 1-chip / 1-token /
acks-only / empty chips), with slots reused across chunks (a request in a slot an earlier request left: its records
over [0, end) must not see the leftovers) and segments in any order. Bystanders: a slot not in a chunk is
byte-identical (its WHOLE slabs, every layer and config, every chip) before and after the chunks it sits out.

Both worlds go through ``runtime.prefill_packed`` (tt/runners/adapter.py: the serving entry tt-d-gen drives; the
engine's KV cache, a slot's state bound to its slabs, the host ack sink) under DECODER_SKIP=decode (the served mode:
layers 0-19 compute, layer 20 its KV head, every served layer acked). Every request is real text (the golden prompt at
its own offset). Build counting: program-cache entries + prepare_program calls inside the serving calls after the
engine's own warm-up (``runtime.compile``), every serving call of the test (alone and packed).
Path checks (a test that silently took another path is no evidence): every packed chunk ran the one call
(packed_device.STATS fallback 0); the indexer layers took the ONE ring call (GluedCall._index_dynamic) and, at
histories >= 8192, the long candidate pool (block_max_out, nbv >= ctopk); the history gathers hid (gather_hide
regions > 0).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import torch

import ttnn

DIG_ROWS = 8  # rows of a (segment, layer) output kept as bf16 for the max abs diff where two digests differ


_ORDER_OK = {}  # id(mesh) -> True once the composer's chip order was checked against get_device_tensors


def chips(t) -> list:
    """Every chip's shard of device tensor ``t`` as host tensors, chip-linear (r TP + c, get_device_tensors' order), in
    ONE read (ttnn.to_torch with ConcatMeshToTensor: per-chip reads cost ~10 ms each, thousands a chunk). The order is
    checked against get_device_tensors once a mesh."""
    mesh = t.device()
    n, b = int(mesh.get_num_devices()), int(t.shape[0])
    allc = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    assert allc.shape[0] == n * b, (tuple(allc.shape), tuple(t.shape), n)
    parts = list(allc.reshape(n, b, *allc.shape[1:]).unbind(0))
    if not _ORDER_OK.get(id(mesh)):
        dev = ttnn.get_device_tensors(t)
        for q in (0, 1, n // 2 + 1, n - 1):
            assert torch.equal(
                ttnn.to_torch(dev[q]).reshape(parts[q].shape), parts[q]
            ), f"composer chip order: chip {q}"
        _ORDER_OK[id(mesh)] = True
    return parts


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _sample_idx(n: int) -> torch.Tensor:
    return torch.unique(torch.linspace(0, max(0, n - 1), min(DIG_ROWS, n)).long())


@dataclass
class Req:
    """A request: its text (the golden prompt from ``off``) and its packed-world slot (one slot for its life)."""

    name: str
    off: int
    slot: int
    ref: str = ""  # the request whose alone run is this one's reference (same text, same boundaries); "" = itself
    alone_slot: int = -1

    @property
    def key(self) -> str:
        return self.ref or self.name


@dataclass
class SegRec:
    """What one segment of one chunk produced: per layer output digest (+ a bf16 sample), served digests, acks."""

    out: dict = field(default_factory=dict)  # layer -> sha256
    smp: dict = field(default_factory=dict)  # layer -> bf16 rows (alone world only)
    served: dict = field(default_factory=dict)  # (layer, config) -> sha256
    acks: list = field(default_factory=list)  # (g0, g1, start, end) in order


class AckSink:
    def __init__(self):
        self.acks = []

    def layers_completed(self, g0, g1, slot, start, end):
        self.acks.append((int(g0), int(g1), int(slot), int(start), int(end)))


class PathCount:
    """While installed: the layers whose indexer took GluedCall's ONE ring call (_index_dynamic), those of them whose
    candidate pool was the long one (nbv >= ctopk: block_max_out), and the per-group ring path (_index)."""

    def __init__(self):
        self.one_ring, self.long_pool, self.grouped = set(), set(), set()
        self.undo = []

    def __enter__(self):
        from models.demos.deepseek_v41_flash_d_p.tt import packed_glued as PG

        me = self
        dyn, grp = PG.GluedCall._index_dynamic, PG.GluedCall._index

        def index_dynamic(self, block, *a, **k):
            out = dyn(self, block, *a, **k)
            me.one_ring.add(int(block.layer))
            try:
                from models.demos.deepseek_v41_flash_d_p.tt.indexer import TtCandidateIndexer

                ix = block.indexer
                if isinstance(ix, TtCandidateIndexer):
                    lens = [
                        min(
                            ix._capacity(r.state.compress[block.kv_src].index_k),
                            -(-((r.start + ix.chunk) // ix.ratio) // 32) * 32,
                        )
                        for r in self.geo.segs
                    ]
                    if max(lens) // ix.block >= ix.ctopk:
                        me.long_pool.add(int(block.layer))
            except Exception as e:  # noqa: BLE001 (diagnostic only)
                print(f"PathCount: long-pool check failed: {e}")
            return out

        def index(self, block, *a, **k):
            me.grouped.add(int(block.layer))
            return grp(self, block, *a, **k)

        PG.GluedCall._index_dynamic, PG.GluedCall._index = index_dynamic, index
        self.undo = [(PG.GluedCall, "_index_dynamic", dyn), (PG.GluedCall, "_index", grp)]
        return self

    def __exit__(self, *exc):
        for cls, name, f in reversed(self.undo):
            setattr(cls, name, f)
        self.undo.clear()
        return False


class Bench:
    """The serving runtime over the held model (adapter.build_runtime, the gate's load), the engine's KV cache of
    ``slots`` slots at ``max_seq``, the host ack sink, ``runtime.compile`` (the engine's warm-up), DECODER_SKIP=decode
    for the bench's life (``close`` restores)."""

    def __init__(self, mesh, spec, max_seq: int, slots: int):
        from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
        from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C

        self.mesh, self.spec = mesh, spec
        sp, tp = tuple(mesh.shape)
        chunk = int(spec.data["target"]["chunk"])
        layers = list(spec.layers())
        assert layers == list(range(len(layers))), f"the served layers must be 0..n-1: {layers}"
        self.adapter = get_adapter(C.adapter_name())
        hf = self.adapter.load_hf_config()
        hf.max_seq_len = max_seq
        self.params = PrefillRunParams(
            mesh_shape=(sp, tp),
            num_layers=len(layers),
            first_layer_idx=0,
            is_first_rank=True,
            is_last_rank=True,
            max_seq_len=max_seq,
            chunk_size=chunk,
            num_users=slots,
            capacity_factor=8,
            num_links=2,
            gate_mode_name=self.adapter.default_gate_mode,
            kv_only_last_layer=True,
            weight_cache_path=self.adapter.weight_cache_path((sp, tp)),
        )
        t0 = time.time()
        self.runtime = self.adapter.build_runtime(mesh_device=mesh, hf_config=hf, params=self.params)
        self.model = m = self.runtime.model
        self.chunk = chunk
        self.loaded_skip = m.skip_mode
        if m.skip_mode != "decode":
            m.use_decoder_skip("decode")  # (the served mode; host lists only, FIX.28)
        self.kv = self.adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf, params=self.params)
        self.sink = AckSink()
        self.runtime.set_layer_completion_sink(self.sink)
        print(f"ISOLATION bench: runtime + KV ({slots} slots, max_seq {max_seq}) in {time.time() - t0:.1f} s")
        t0 = time.time()
        self.runtime.compile(self.kv)  # (the engine's warm-up at startup: slots 0..5)
        m.sync()
        self.compile_s = time.time() - t0
        print(f"ISOLATION bench: runtime.compile in {self.compile_s:.1f} s")
        self.builds = 0
        self.serving_calls = 0
        from models.demos.deepseek_v41_flash_d_p.tests import fresh_common as F

        self.pc = F.PrepCount()
        self.build_notes = []
        self.decoder = frozenset(m.decoder)
        # the layers whose outputs are captured: the characteristic ones a decode chunk computes and its last computed
        # layer (a bit-exact difference in any layer before it reaches it); served records are compared at every layer
        computed = [i for i in m.layers if i not in self.decoder]
        self.capture_layers = sorted(({0, 1, 2, 3, 14} & set(computed)) | {computed[-1]})

    def hold(self, slots) -> None:
        """Every slot's state made before any count (a serving process holds them after their first request)."""
        for s in slots:
            self.runtime.state(self.kv, int(s))

    # -------------------------------------------------------------- one serving call
    def run(self, segs: list, texts: list, capture=True) -> tuple:
        """One packed chunk of ``segs`` (tt/packed.Segment, engine order) through runtime.prefill_packed, the ids
        ``texts`` (each segment's [start, end) ids) -> ([SegRec per segment], seconds). The outputs of the computed
        layers in ``capture`` (True: every one) are captured inside the call (each segment's rows of the layer's
        output: streams and pre; host reads only). The serving call runs under a BuildProbe: a build is named."""
        from models.demos.common.bringup.testing.profile import cache_entries
        from models.demos.deepseek_v41_flash_d_p.tt.packed import GluedChunk
        from models.demos.deepseek_v41_flash_d_p.tt.packed_device import as_run_desc

        rt, model, mesh = self.runtime, self.model, self.mesh
        desc = GluedChunk(self.chunk, tuple(segs))
        rd = as_run_desc(desc, mesh)
        assert getattr(rd, "align", 0), "PACKED_CHIPS=on: chip-aligned lanes"
        recs = [SegRec() for _ in segs]
        rows = [rd.rows(k) for k in range(len(segs))]
        offs = [rd.lane_off(k) for k in range(len(segs))]
        layer = model.layer

        from models.demos.deepseek_v41_flash_d_p.tt.layout import HC

        want = None if capture is True else set(capture or ())

        def wrapped(i, h, *a, **k):
            out = layer(i, h, *a, **k)
            if (want is None or int(i) in want) and out is not h and getattr(out, "streams", None) is not None:
                if any(n > 0 for n in rows):
                    st = torch.cat([p.reshape(-1, out.streams.shape[-1]) for p in chips(out.streams)], dim=0)
                    pr = torch.cat([p.reshape(-1, out.pre.shape[-1]) for p in chips(out.pre)], dim=0)[:, :HC]
                for j, rec in enumerate(recs):
                    if rows[j] > 0:
                        o, n = offs[j], rows[j]
                        r = torch.cat([st[o : o + n], pr[o : o + n]], dim=-1).float()
                        rec.out[int(i)] = digest(r)
                        rec.smp[int(i)] = r[_sample_idx(r.shape[0])].to(torch.bfloat16)
            return out

        inp = rt.make_packed_input(desc, [t.long() for t in texts])  # (the engine's H2D: not counted)
        n0 = len(self.sink.acks)
        e0, p0 = cache_entries(mesh), self.pc.n
        t0 = time.time()
        from models.demos.deepseek_v41_flash_d_p.tests import fresh_common as F

        model.layer = wrapped
        try:
            with self.pc, F.BuildProbe(mesh) as pb:
                rt.prefill_packed(inp, self.kv, desc)
                model.sync()
        finally:
            del model.layer
        dt = time.time() - t0
        built = (cache_entries(mesh) - e0) + (self.pc.n - p0)
        if built:
            self.build_notes.append(
                f"{[(s.slot, s.start, s.end, s.write_start) for s in segs]}: {built} {pb.summary(6)}"
            )
            print(f"ISOLATION BUILD {self.build_notes[-1]}", flush=True)
        self.builds += built
        self.serving_calls += 1
        ttnn.deallocate(inp)
        acks = self.sink.acks[n0:]
        slots = [int(s.slot) for s in segs]
        stray = [a for a in acks if a[2] not in slots]
        assert not stray, f"acks for slots not in the chunk: {stray[:4]}"
        for s, rec in zip(segs, recs):
            rec.acks = [(g0, g1, st, en) for g0, g1, sl, st, en in acks if sl == int(s.slot)]
        return recs, dt

    def served(self, seg) -> dict:
        """The segment's slot: every served record of positions [0, cap(end)), every layer (the window's rows and
        the served 128-row snapshot of every chip, compressed KV, index keys: the served bytes), and the host engram
        tails."""
        from models.demos.deepseek_v41_flash_d_p.tt.attention import prefill_end

        st = self.runtime.state(self.kv, int(seg.slot))
        # what the engine serves: [0, cap) (PREFILL_END=cap, P.49: decode receives [0, cap) and replays the rest);
        # rows in [cap, ceil32(end)) are never written (the slot's own leftovers, legitimately unlike a fresh slot's)
        n = int(prefill_end(int(seg.end)))
        out = {}
        for i in self.model.layers:
            c = st.window[i]
            out[(int(i), "0_window_kv")] = digest(window_rows(c, n))
            if getattr(c, "snap", None) is not None:  # (WINDOW_KV_KEEP=snapshot128: the served snapshot, every chip)
                out[(int(i), "snapshot")] = digest(slab_rows(c.snap))
            if i in st.compress:
                for name, cc in st.compress[i].by_config().items():
                    out[(int(i), name)] = digest(compressed_rows(cc, n))
        for i, t in (st.tail or {}).items():
            if isinstance(t, torch.Tensor):
                out[(int(i), "engram_tail")] = digest(t)
        return out

    def slab_digest(self, slot: int) -> dict:
        """A slot's WHOLE slabs (every served layer and config, every chip, all rows): {(layer, config): sha256}."""
        out = {}
        for layer, cfgs in self.kv.contract.slabs(int(slot)).items():
            for name, slab in cfgs.items():
                out[(int(layer), name)] = digest(slab_rows(slab))
        return out

    def close(self) -> None:
        rt = self.runtime
        rt.set_layer_completion_sink(None)
        for key in [k for k in rt._states if k[0] == id(self.kv)]:
            try:
                rt._states.pop(key).free()
            except Exception as e:  # noqa: BLE001
                print(f"state free: {e}")
        for t in self.kv.contract.by_config.values():
            if t.is_allocated():
                ttnn.deallocate(t)
        if self.model.skip_mode != self.loaded_skip:
            self.model.use_decoder_skip(self.loaded_skip)


def slab_rows(slab) -> torch.Tensor:
    """A CacheSlab's whole (slot, layer) rows of every chip [chips, rows, width] (one read)."""
    sl = slab.slice_rows(0, slab.rows)
    out = torch.stack([p.reshape(slab.rows, slab.width) for p in chips(sl)])
    ttnn.deallocate(sl)
    return out


def window_rows(c, length: int) -> torch.Tensor:
    """TtWindowCache.to_host (positions [0, ceil32(length)) in natural order) with the ring read in one call: under
    WINDOW_KV_KEEP=snapshot128 the slot's carry ring (tt/attention._ring_natural: each block from its owner's copy,
    zeros where it holds none); else the model's own read."""
    from models.demos.deepseek_v41_flash_d_p.tt.attention import RING

    if c.ring is None:
        return c.to_host(length)
    p0, p1 = 0, -(-int(length) // 32) * 32
    out = torch.zeros(p1, c.WIDTH, dtype=torch.bfloat16)
    ring = c.ring
    if p1 <= p0 or ring.writer is not c and c.keep == "none" or not c.history:
        return out
    shards = [p.reshape(RING, ring.width) for p in chips(ring.t)]
    for j in range(ring.BLOCKS):
        q = ring.pos[j]
        if q is None:
            continue
        lo, hi = max(q, p0), min(ring.vend[j], p1)
        if lo < hi:
            src = shards[ring.own[j] or 0]
            out[lo - p0 : hi - p0] = src[q % RING + lo - q : q % RING + hi - q]
    return out


def compressed_rows(c, length: int) -> torch.Tensor:
    """A compressed cache's served rows of token positions [0, ceil32(length)) in natural order (tt/compressor
    _owner_shards / _natural: what the engine serves; the 192-byte rows as bytes under fp8), reading only each chip's
    local rows of the chunk periods [0, ceil(length / chunk)) (a whole-slab read is ~0.7 GB a slot at 114688)."""
    slab = c.served if c.fp8 else (c.slab if c.slab is not None else c.served)
    hs = c.stripe // c.ratio  # compressed rows per owner stripe (a chunk period)
    per = c.owners * c.stripe  # token positions a period
    keep = -(-int(length) // 32) * 32
    nb = max(1, -(-keep // per))
    rows = min(nb * hs, c.crows)
    sl = slab.slice_rows(0, rows)
    parts = chips(sl)
    ttnn.deallocate(sl)
    if c.tp_axis is None:  # replicated over TP: mesh column 0's copy of every SP row
        parts = parts[:: c.geo.tp]
    w = slab.width
    parts = [p.reshape(rows, w) for p in parts]
    nat = torch.stack([p.reshape(nb, hs, w) for p in parts], dim=1).reshape(nb * c.owners * hs, w)
    return nat[: keep // c.ratio]


class Verdict:
    """The comparisons of one test (prefix ``packed_fast`` / ``packed_deep``): counts and the max abs diff."""

    def __init__(self):
        self.out_mm = self.served_mm = self.ack_mm = self.bystander_mm = 0
        self.out_diff = 0.0
        self.segments = 0
        self.notes = []

    def compare(self, tag: str, got: SegRec, ref: SegRec, ref_slot: int, slot: int) -> None:
        """``got`` (packed world, slot ``slot``) vs ``ref`` (alone, slot ``ref_slot``)."""
        self.segments += 1
        layers = sorted(set(got.out) | set(ref.out))
        bad = [i for i in layers if got.out.get(i) != ref.out.get(i)]
        if bad:
            self.out_mm += len(bad)
            self.notes.append(f"{tag}: outputs differ at layers {bad} (first {bad[0]})")
        keys = set(got.served) | set(ref.served)
        sbad = sorted(k for k in keys if got.served.get(k) != ref.served.get(k))
        if sbad:
            self.served_mm += len(sbad)
            self.notes.append(f"{tag}: served records differ {sbad[:6]} ({len(sbad)})")
        if got.acks != ref.acks:
            self.ack_mm += 1
            self.notes.append(
                f"{tag}: acks differ: {got.acks[:3]}.. ({len(got.acks)}) vs {ref.acks[:3]}.. ({len(ref.acks)})"
            )

    def out_diff_of(self, got_rows: dict, ref: SegRec) -> None:
        """The max abs diff over the sampled rows of the layers whose digests differ (>= 1e-30)."""
        for i, r in got_rows.items():
            s = ref.smp.get(i)
            if s is None or r.shape != s.shape:
                self.out_diff = float("inf")
                continue
            d = (r.float() - s.float()).abs()
            d = torch.nan_to_num(d, nan=float("inf"))
            self.out_diff = max(self.out_diff, float(d.max()) if d.numel() else 0.0, 1e-30)

    def bystander(self, tag: str, before: dict, after: dict) -> None:
        bad = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        if bad:
            self.bystander_mm += len(bad)
            self.notes.append(f"{tag}: bystander slabs changed {bad[:6]} ({len(bad)})")


class Worlds:
    """The packed world (``chunks``: lists of (request, start, end, write_start), each request in its own packed
    slot; slots reused) against the alone world (each request alone, one segment a chunk, at the same boundaries, in
    a slot of its own nothing else uses). ``bystanders``: (slot, after chunk a, before chunk b): the slot's whole
    slabs equal after chunk a (a = -1: before the first chunk) and before chunk b (b = len(chunks): after the last)."""

    def __init__(self, bench: Bench, text: torch.Tensor, reqs: dict, chunks: list, bystanders=(), label: str = ""):
        self.b, self.text, self.reqs, self.chunks = bench, text, reqs, chunks
        self.bystanders = list(bystanders)
        self.label = label
        for c in chunks:
            names = [x[0] for x in c]
            assert len(names) == len(set(names)), names
            slots = [reqs[n].slot for n in names]
            assert len(slots) == len(set(slots)), f"two segments of one slot in a chunk: {c}"

    def _seg(self, r: Req, start, end, ws, alone: bool):
        from models.demos.deepseek_v41_flash_d_p.tt.packed import Segment

        slot = self.reqs[r.key].alone_slot if alone else r.slot
        assert slot >= 0, (r, alone)
        return Segment(slot, int(start), int(end), None if ws is None else int(ws))

    def alone_names(self) -> list:
        """The requests whose alone runs are references (first appearance order)."""
        out = []
        for c in self.chunks:
            for x in c:
                k = self.reqs[x[0]].key
                if k not in out:
                    out.append(k)
        return out

    def _ids(self, r: Req, start, end) -> torch.Tensor:
        return self.text[r.off + int(start) : r.off + int(end)]

    def alone(self) -> dict:
        """Every request's chunks alone, in its request order -> {(request, start): SegRec}."""
        ref = {}
        by_req = {}
        for c in self.chunks:
            for n, s, e, ws in c:
                k = self.reqs[n].key
                xs = by_req.setdefault(k, [])
                if all(int(x[1]) != int(s) for x in xs):
                    xs.append((k, s, e, ws))
                else:
                    assert (k, s, e, ws) in xs, f"{n}: another chunk at {s} than its reference {k}'s: {xs}"
        t0 = time.time()
        order = [x for xs in by_req.values() for x in xs]
        for name, xs in by_req.items():
            r = self.reqs[name]
            for _, s, e, ws in xs:
                seg = self._seg(r, s, e, ws, alone=True)
                recs, _ = self.b.run([seg], [self._ids(r, s, e)], capture=self.b.capture_layers)
                rec = recs[0]
                rec.served = self.b.served(seg)
                ref[(name, int(s))] = rec
        print(f"ISOLATION {self.label} alone world: {len(order)} chunks in {time.time() - t0:.1f} s", flush=True)
        return ref

    def packed(self, ref: dict, v: Verdict) -> None:
        """The packed world, each chunk compared segment by segment against ``ref`` (the alone world)."""
        b = self.b
        before = {}
        for slot, a, _ in self.bystanders:
            if a == -1:
                before[(slot, a)] = b.slab_digest(slot)
        t0 = time.time()
        for ci, c in enumerate(self.chunks):
            for slot, a, e in self.bystanders:
                if e == ci:
                    v.bystander(
                        f"{self.label} slot {slot} chunks {a + 1}..{e - 1}", before.pop((slot, a)), b.slab_digest(slot)
                    )
            segs = [self._seg(self.reqs[n], s, e, ws, alone=False) for n, s, e, ws in c]
            recs, dt = b.run(segs, [self._ids(self.reqs[n], s, e) for n, s, e, _ in c], capture=b.capture_layers)
            for (n, s, e, ws), seg, rec in zip(c, segs, recs):
                rec.served = b.served(seg)
                tag = f"{self.label} chunk {ci} {n}[{s},{e}{'' if ws is None else f' ws {ws}'}) slot {seg.slot}"
                want = ref[(self.reqs[n].key, int(s))]
                v.compare(tag, rec, want, self.reqs[self.reqs[n].key].alone_slot, seg.slot)
                if any(rec.out.get(i) != want.out.get(i) for i in set(rec.out) | set(want.out)):
                    v.out_diff_of({i: rec.smp[i] for i in rec.out if rec.out.get(i) != want.out.get(i)}, want)
            kinds = ", ".join(f"{n}[{s},{e}{'' if ws is None else '*'})" for n, s, e, ws in c)
            chips = sum(-(-max(1, e - s) // (b.chunk // 32)) for _, s, e, _ in c)
            print(f"ISOLATION {self.label} chunk {ci}: {kinds} ({chips} chips) {dt:.2f} s", flush=True)
            for slot, a, _ in self.bystanders:
                if a == ci:
                    before[(slot, a)] = b.slab_digest(slot)
        for slot, a, e in self.bystanders:
            if e == len(self.chunks) and (slot, a) in before:
                v.bystander(f"{self.label} slot {slot} chunks {a + 1}..end", before.pop((slot, a)), b.slab_digest(slot))
        print(f"ISOLATION {self.label} packed world: {len(self.chunks)} chunks in {time.time() - t0:.1f} s", flush=True)


def live_check(ref: dict, a: tuple, b: tuple) -> None:
    """The comparison is live: two different requests' outputs at one layer differ (a digest of nothing would pass)."""
    ra, rb = ref[a], ref[b]
    common = sorted(set(ra.out) & set(rb.out))
    assert common, f"no common computed layer of {a} and {b}"
    assert ra.out[common[0]] != rb.out[common[0]], f"{a} and {b} have identical outputs at layer {common[0]}"
    assert ra.served and ra.acks, f"{a}: no served records / acks captured"
