# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.NC (OWNER 2026-11-21): an ASSESSMENT, tests only, never a serving switch. How much faster is a chunk at a 1M history
without the per-chunk bf16 -> 4-bit history conversion (tt/history_gather._pack: ttnn.bringup.ckv_bfp4_rows, the
served bf16 rows -> 320 B bfloat4_b records before every gather / owned send)?

Two paths at history NOCONV_POS (default 1024000), steady (PB.9C's ladder method: the real previous chunk untimed
first, testing/profile.warm_up, NOCONV_RUNS (10) timed runs with program-cache misses refused, the median, one
real-time profiled run):
    single8192   today's single-request chunk (PACKED=off, slot 0 of the serving KV cache), one 8192-token segment
    4x2k         chip-aligned 4 x 2048 (PACKED=on, PACKED_GLUED=on, PACKED_CHIPS as shipped), every segment at the
                 history in its own serving slot, segment order rotated every run (as ladder_common.packed_ladder)
Two sides a path:
    A   today's path
    B   every history already held as 4-bit records: after one untimed collect run (prev + timed chunk: the slabs then
        hold every row any chunk of the probe reads), each slot's whole slab [0, R) is packed ONCE (the original
        _pack_rows) into a record buffer of its own, what a 4-bit served cache would hold; then history_gather._pack
        returns that buffer without launching anything (the per-chunk pack skipped: prefetch AND tail packs), and
        _owned_send (PB.HG) reads it in place of the (unwritten) local record buffer. Gathers, owned sends, attention and
        everything else unchanged. A 4-bit served cache would pay the tail's conversion at its compressor write
        instead (2 periods a chunk); B leaves that out too (the upper bound of the saving).
Per side: harness wall, device ms (slowest chip), exposed host (wall - device), the pack's device ms (its programs)
and host ms (the wrapped _pack calls of the timed chunk, median), the traced history ops' device ms by kind
(history_gather.TRACE: pack / gather / tail / write), the profile's history_gather section (the gathers left exposed
outside the hide regions), and the last computing layer's output A vs B (max |diff|: must be 0, the records are a
function of each row).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "<the gate's env>
        scripts/run_safe_pytest.sh --run-all --no-precompile
        models/demos/deepseek_v41_flash_d_p/tests/perf/test_no_conversion_probe.py"
"""

from __future__ import annotations

import importlib
import json
import os
import time
from collections import defaultdict

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden, misses_allowed, warm_up
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

_pc = importlib.reload(_pc)  # (the resident runner keeps helper modules of an earlier request)

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = ("NOCONV_POS", "NOCONV_RUNS", "NOCONV_PATHS", "NOCONV_OUT")
BATCH_POS_VARS = ("NOCONV_POS",)
BATCH_LAYERS = "executed"  # (the ladder's load: layers 0-20 under decode)
CHUNK = _pc.CHUNK
SHIFT = 4096  # (test_packed_walls: segment k reads the text from k x SHIFT)
LENS = (2048, 2048, 2048, 2048)
POS = 1024000


def _knob(name: str, default):
    v = os.environ.get(name, "").strip()
    return type(default)(v) if v else default


def _hg():
    return importlib.import_module("models.demos.deepseek_v41_flash_d_p.tt.history_gather")  # (DS41-311)


class NoConv:
    """history_gather._pack / _owned_send wrapped. mode "a": today's (the pack timed, its op ids kept when ``trace``);
    "collect": today's, every cache seen kept; "b": the cache's pre-packed records returned, nothing launched."""

    def __init__(self, monkeypatch, mesh):
        import ttnn

        self.ttnn, self.mesh, self.hg = ttnn, mesh, _hg()
        self.mode, self.count, self.trace = "a", False, False
        self.seen, self.pre, self.locmap = {}, {}, {}
        self.host_ns, self.calls, self.ids = 0, 0, []
        self.subs, self.unmapped = 0, 0  # side B: owned-send entries read from the pre-packed records / from a loc
        self.orig_pack, self.orig_send, self.orig_rows = self.hg._pack, self.hg._owned_send, self.hg._pack_rows
        monkeypatch.setattr(self.hg, "_pack", self._pack)
        monkeypatch.setattr(self.hg, "_owned_send", self._send)

    def _pack(self, ckv, lo, hi, loc=None):
        if self.mode == "b":
            hit = self.pre.get(id(ckv))
            assert hit is not None, ("PB.NC side B: a cache the collect run never packed", id(ckv), lo, hi)
            if loc is not None:
                self.locmap[id(loc)] = hit.tensor
            if self.count:
                self.calls += 1
            return hit.tensor
        if self.mode == "collect":
            self.seen[id(ckv)] = ckv
        i0 = self.ttnn._ttnn.get_device_operation_id()
        t0 = time.perf_counter_ns()
        try:
            return self.orig_pack(ckv, lo, hi, loc)
        finally:
            if self.count:
                self.host_ns += time.perf_counter_ns() - t0
                self.calls += 1
            if self.trace:
                self.ids.append((i0, self.ttnn._ttnn.get_device_operation_id()))

    def _send(self, buf, R, ckv0, entries, dest, sub=None):
        if self.mode == "b":
            HB = self.hg.HistoryBuffer
            fixed = []
            for t, lo, hi in entries:
                if isinstance(t, HB) and hi > lo:
                    if id(t) in self.locmap:
                        self.subs += 1
                        t = self.locmap[id(t)]
                    else:
                        self.unmapped += 1  # (a local record buffer side B never filled: would read stale bytes)
                fixed.append((t, lo, hi))
            entries = fixed
        return self.orig_send(buf, R, ckv0, entries, dest, sub)

    def prepack(self) -> int:
        """Every collected cache's whole slab -> 320 B records in a buffer of its own (once, untimed)."""
        n = 0
        for k, ckv in self.seen.items():
            if k in self.pre:
                continue
            R = int(ckv.slab.rows)
            buf = self.hg.HistoryBuffer(self.mesh, R, None, True)
            self.orig_rows(ckv, buf, 0, R)  # (the records of every row: what a 4-bit served cache would hold)
            self.pre[k] = buf
            n += 1
        self.ttnn.synchronize_device(self.mesh)
        return n

    def free(self) -> None:
        for b in self.pre.values():
            b.free()
        self.pre.clear()
        self.seen.clear()
        self.locmap.clear()
        self.subs, self.unmapped = 0, 0
        self.mode = "a"


def _dev_ms(recs: dict, d2c: dict, ranges: list) -> float:
    """ms: the slowest chip's summed program time over the device-operation id ranges."""
    ids = set()
    for a, b in ranges:
        ids.update(range(a, b))
    per = defaultdict(float)
    for i in ids:
        for d, e in (recs.get(i) or {}).items():
            per[d2c.get(int(d), int(d))] += e[0]
    return max(per.values()) / 1e6 if per else 0.0


def _measure(mesh, W: NoConv, run, label: str) -> dict:
    """``run(prev, rot, prof=False, keep=False)`` -> (wall ms, profile or None, output or None). PB.9C's method."""
    from models.demos.deepseek_v41_flash_d_p.tests.packed import ladder_common as L

    rot = [0]
    hosts, pcalls = [], []

    def one_run():
        run(True, rot[0])
        W.count, W.host_ns, W.calls = True, 0, 0
        try:
            w = run(False, rot[0])[0]
        finally:
            W.count = False
        hosts.append(W.host_ns / 1e6)
        pcalls.append(W.calls)
        rot[0] += 1
        return round(w, 1)

    w = warm_up(mesh, one_run)
    assert w["warm"], (label, w["entries"][-5:] if "entries" in w else None)
    hosts.clear()
    pcalls.clear()
    walls = []
    misses_allowed(mesh, False)
    try:
        for _ in range(_knob("NOCONV_RUNS", 10)):
            walls.append(one_run())
    finally:
        misses_allowed(mesh, True)
    kept = sorted(walls)
    wall = kept[len(kept) // 2]
    host_pack = sorted(hosts)[len(hosts) // 2]
    # one real-time profiled run: device, sections, the pack's programs, the traced history ops by kind
    hg = _hg()
    run(True, rot[0])
    W.ids = []
    W.trace = True
    hg.TRACE = []
    try:
        _, prof, _ = run(False, rot[0], prof=True)
        trace = list(hg.TRACE)
    finally:
        W.trace = False
        hg.TRACE = None
    rot[0] += 1
    recs, d2c = prof.get("rt_records") or {}, prof.get("dev_to_chip") or {}
    kinds = defaultdict(list)
    for k, _src, _lane, a, b in trace:
        kinds[k].append((a, b))
    by_kind = {k: round(_dev_ms(recs, d2c, v), 3) for k, v in sorted(kinds.items())}
    hist = round(_dev_ms(recs, d2c, [r for v in kinds.values() for r in v]), 3)
    moves = round(_dev_ms(recs, d2c, [r for k, v in kinds.items() if k != "pack" for r in v]), 3)
    dev = L._device(prof)
    phases = {k: round(v, 3) for k, v in L._phases(prof).items()}
    # the output at a fixed rotation (the same rows on both sides)
    run(True, 0)
    _, _, out = run(False, 0, keep=True)
    r = {
        "wall": wall,
        "runs": walls,
        "spread": round(kept[-1] - kept[0], 1),
        "warm_runs": len(w["walls"]),
        "device": round(dev, 2),
        "host_exposed": round(wall - dev, 2),
        "pack_device": round(_dev_ms(recs, d2c, W.ids), 3),
        "pack_host": round(host_pack, 3),
        "pack_calls": pcalls[len(pcalls) // 2] if pcalls else 0,
        "hist_ops_device": hist,
        "hist_moves_device": moves,
        "hist_by_kind": by_kind,
        "hist_section": phases.get("history_gather", 0.0),
        "phases": phases,
    }
    print(
        f"NOCONV {label}: wall {wall} ms (runs {walls}, spread {r['spread']}), device {r['device']} ms, exposed host "
        f"{r['host_exposed']} ms; pack device {r['pack_device']} ms / host {r['pack_host']} ms ({r['pack_calls']} "
        f"calls); history ops device {hist} ms (moves {moves}; {json.dumps(by_kind)}), history_gather section "
        f"{r['hist_section']} ms; phases {json.dumps(phases)}",
        flush=True,
    )
    return r, out


def _last_layer(model, layers) -> int:
    return max(j for j in layers if j not in getattr(model, "kv_only", ()))


def _single(runtime, kv, mesh, layers, text, p: int):
    """single8192 at p on slot 0: run(prev, rot, prof, keep)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import packed_device as PD
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor

    model = runtime.model
    st = runtime.state(kv, 0)
    last = _last_layer(model, layers)

    def run(prev: bool, rot: int, prof: bool = False, keep: bool = False):
        s = p - (CHUNK if prev else 0)
        desc = ChunkDescriptor.single(CHUNK, 0, s, s + CHUNK)
        inp = PD.make_input(mesh, desc, [text[s : s + CHUNK].long()])  # (the engine's H2D: before the clock)
        model.sync()
        kept = []
        on_layer = (lambda i, h: kept.append(model.to_host(h)) if i == last else None) if keep else None
        if prof:
            profiler.enable(mesh, rt=True)
            profiler.set_layer(None)
        res = None
        t0 = time.time()
        try:
            nat = PD.ids_from_input(mesh, inp, desc)
            PD.run_chunk(model, desc, nat, [st], layers, last=False, on_layer=on_layer, inp=inp)
            model.sync()
            wall = (time.time() - t0) * 1e3
            if prof:
                profiler.signpost("end")
                profiler.finish_rt()
                res = profiler.result()
        finally:
            if prof:
                profiler.disable()
            ttnn.deallocate(inp)
        return wall, res, (kept[0] if kept else None)

    return run


def _packed(runtime, kv, mesh, layers, text, p: int):
    """chip-aligned 4 x 2k at p, segment k in slot k, order rotated by ``rot``: run(prev, rot, prof, keep)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import packed_device as PD
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment

    model = runtime.model
    n = len(LENS)
    states = [runtime.state(kv, k) for k in range(n)]
    last = _last_layer(model, layers)

    def run(prev: bool, rot: int, prof: bool = False, keep: bool = False):
        order = [(k + rot) % n for k in range(n)]
        segs, ids, sts = [], [], []
        for k in order:
            s = p - (LENS[k] if prev else 0)
            segs.append(Segment(k, s, s + LENS[k], None))
            ids.append(text[k * SHIFT + s : k * SHIFT + s + LENS[k]].long())
            sts.append(states[k])
        desc = ChunkDescriptor.packed(CHUNK, segs)
        inp = PD.make_input(mesh, desc, ids)
        model.sync()
        kept = []
        on_layer = (lambda i, h: kept.append(model.to_host(h)) if i == last else None) if keep else None
        if prof:
            profiler.enable(mesh, rt=True)
            profiler.set_layer(None)
        res = None
        t0 = time.time()
        try:
            nat = PD.ids_from_input(mesh, inp, desc)
            PD.run_chunk(model, desc, nat, sts, layers, last=False, on_layer=on_layer, inp=inp)
            model.sync()
            wall = (time.time() - t0) * 1e3
            if prof:
                profiler.signpost("end")
                profiler.finish_rt()
                res = profiler.result()
        finally:
            if prof:
                profiler.disable()
            ttnn.deallocate(inp)
        return wall, res, (kept[0] if kept else None)

    return run


def _rec(name: str, value) -> None:
    from models.demos.deepseek_v41_flash_d_p.tests.packed import ladder_common as L

    L.record(name, value)  # (also without the text tag: the gate's threshold names carry none)


@mesh_parametrize(timeout_s=4 * 3600)
def test_no_conversion_probe(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192"
    assert settings.get("DECODER_SKIP") == "decode", "run with DS41_DECODER_SKIP=decode"
    assert _hg().bfp4_on(), "PB.NC measures the 4-bit history path (HISTORY_FORMAT=bfp4)"
    from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as SP

    p = _knob("NOCONV_POS", POS)
    paths = [x for x in os.environ.get("NOCONV_PATHS", "single8192,4x2k").split(",") if x.strip()]
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cap = -(-deploy // CHUNK) * CHUNK  # (the ladder's load: every slot at the deployment context in whole chunks)
    want = max(cap, int(S.data["target"]["seq"]))
    monkeypatch.setitem(S.data["target"], "seq", want)
    monkeypatch.setenv("WALLS_FRESH_SLOTS", str(len(LENS)))  # (one serving slot a segment)
    assert p % 32 == 0 and p >= CHUNK and p + CHUNK <= want, (p, want)
    runtime, adapter, hf, params = SP._runtime(mesh_device, layers, CHUNK, want)
    model = runtime.model
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    text = PositionGolden(S, (len(LENS) - 1) * SHIFT + p + CHUNK, CHUNK).tokens()
    kv = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf, params=params)
    W = NoConv(monkeypatch, mesh_device)
    out = {"position": p, "paths": {}}
    try:
        for path in paths:
            if path == "single8192":
                ctx, run = _pc.packed("off"), _single(runtime, kv, mesh_device, layers, text, p)
            elif path == "4x2k":
                from contextlib import ExitStack

                ctx = ExitStack()
                ctx.enter_context(_pc.packed("on"))
                ctx.enter_context(_pc.switch("PACKED_GLUED", "on"))
                run = _packed(runtime, kv, mesh_device, layers, text, p)
            else:
                raise AssertionError(f"NOCONV_PATHS: unknown path {path}")
            res = {}
            with ctx:
                W.mode = "a"
                res["a"], oa = _measure(mesh_device, W, run, f"{path} p{p} A (today: per-chunk pack)")
                # side B: collect every cache the chunks pack, pre-pack each slab once, then skip the pack
                W.mode = "collect"
                run(True, 0)
                run(False, 0)
                model.sync()
                n = W.prepack()
                W.mode = "b"
                print(f"NOCONV {path}: {n} caches pre-packed as 4-bit records (once, untimed)", flush=True)
                try:
                    res["b"], ob = _measure(mesh_device, W, run, f"{path} p{p} B (history held as 4-bit records)")
                    res["b_owned_entries_substituted"], res["b_owned_entries_unmapped"] = W.subs, W.unmapped
                finally:
                    W.free()
            diff = float((oa.float() - ob.float()).abs().max())
            res["max_abs_diff"] = diff
            res["prepacked_caches"] = n
            a, b = res["a"], res["b"]
            new = CHUNK if path == "single8192" else sum(LENS)
            res["saved_ms"] = round(a["wall"] - b["wall"], 2)
            res["saved_device_ms"] = round(a["device"] - b["device"], 2)
            res["saved_host_exposed_ms"] = round(a["host_exposed"] - b["host_exposed"], 2)
            for side in ("a", "b"):
                r = res[side]
                r["tok_s"] = round(new / (r["wall"] / 1e3), 1)
                pre = f"noconv_{path}_{side}"
                _rec(f"{pre}_wall_ms_p{p}", r["wall"])
                _rec(f"{pre}_runs_p{p}", r["runs"])
                _rec(f"{pre}_device_ms_p{p}", r["device"])
                _rec(f"{pre}_host_exposed_ms_p{p}", r["host_exposed"])
                _rec(f"{pre}_pack_device_ms_p{p}", r["pack_device"])
                _rec(f"{pre}_pack_host_ms_p{p}", r["pack_host"])
                _rec(f"{pre}_hist_ops_device_ms_p{p}", r["hist_ops_device"])
                _rec(f"{pre}_hist_section_ms_p{p}", r["hist_section"])
                _rec(f"{pre}_tok_s_p{p}", r["tok_s"])
            _rec(f"noconv_{path}_max_abs_diff_p{p}", diff)
            _rec(f"noconv_{path}_saved_ms_p{p}", res["saved_ms"])
            _rec(f"noconv_{path}_b_owned_unmapped_p{p}", res["b_owned_entries_unmapped"])
            _rec(f"noconv_{path}_b_owned_substituted_p{p}", res["b_owned_entries_substituted"])
            print(
                f"NOCONV {path} p{p}: A {a['wall']} ms ({a['tok_s']} tok/s) vs B {b['wall']} ms ({b['tok_s']} tok/s): "
                f"saved {res['saved_ms']} ms a chunk (device {res['saved_device_ms']}, exposed host "
                f"{res['saved_host_exposed_ms']}); A vs B output max |diff| {diff}; side B owned-send entries from the "
                f"pre-packed records {res['b_owned_entries_substituted']}, from an unfilled local buffer "
                f"{res['b_owned_entries_unmapped']}",
                flush=True,
            )
            out["paths"][path] = res
            metrics.record(f"noconv_{path}_detail", json.dumps(res)[:200000])
    finally:
        W.free()
        for key in [k for k in runtime._states if k[0] == id(kv)]:
            try:
                runtime._states.pop(key).free()
            except Exception as e:  # noqa: BLE001
                print(f"state free: {e}")
        import ttnn

        for t in kv.contract.by_config.values():
            if t.is_allocated():
                ttnn.deallocate(t)
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    dst = os.environ.get("NOCONV_OUT", "/tmp/pb_nc_noconv.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"NOCONV probe written: {dst}")
    for path, res in out["paths"].items():
        assert res["max_abs_diff"] == 0.0, (path, res["max_abs_diff"])
        assert res["b_owned_entries_unmapped"] == 0, ("side B read a local record buffer it never filled", path)
