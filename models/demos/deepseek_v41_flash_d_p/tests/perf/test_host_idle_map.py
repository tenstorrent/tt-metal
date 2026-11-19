# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.78 (OWNER 2026-11-18): where the device idles in a steady chunk (wall - device ~24 ms at 142k).

Under DECODER_SKIP=decode (as test_skip_positions), at IDLE_POS (default 145408): the real previous chunk, then the
timed chunk, last=False, random ids (testing/profile.PositionGolden). Two kinds of run, after warm-up:

  clean   (no profiler) host timestamps from the chunk's first host call: begin_chunk returned, embed returned, each
          layer returned, each engram wait (take_engram_rows) start / end, the final sync; the worker's per-phase ms
          (model.engram_times). Median of IDLE_RUNS (default 7).
  rt      one run under the program real-time profiler with NO op patching (no per-op host cost): the same host
          markers also record the device-operation id they were at, so every program on every chip is placed in the
          host segment that launched it. Per chip: the device span (first program start to last end), busy time, and
          every idle gap (a program starting after every earlier program ended) charged to the segment of the program
          that ended it. The segment after an engram wait is where the device idled for the engram rows.

Prints both; records nothing. HOST_HIDE is read as the model loaded it (tt/settings.py); run both values to compare.

    DS41_DECODER_SKIP=decode scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/perf/test_host_idle_map.py -s
"""

from __future__ import annotations

import os
import statistics
import time

import pytest

from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.positions import _free
from models.demos.common.bringup.testing.profile import PositionGolden, warm_up
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = ("IDLE_POS", "IDLE_RUNS", "IDLE_SWEEP")
BATCH_POS_VARS = ("IDLE_POS",)  # F288: the plan's model is loaded for the furthest point (core/batch.hints)
BATCH_LAYERS = "executed"  # F285: a performance measurement: the layers it executes, in the resident runner  # F279: per-run, not the model's configuration


class Marks:
    """Host timestamps (ms from t0) and device-operation ids at named points of one chunk."""

    def __init__(self):
        import ttnn

        self._id = ttnn._ttnn.get_device_operation_id
        self.t0 = time.perf_counter()
        self.rows = []

    def __call__(self, name: str):
        self.rows.append((name, (time.perf_counter() - self.t0) * 1e3, int(self._id())))


def _instrument(model, marks_ref):
    """Wrap take_engram_rows (wait start / end) on the instance; host only."""
    orig = model.take_engram_rows

    def take(layer):
        m = marks_ref[0]
        if m is not None:
            m(f"wait{layer}.start")
        out = orig(layer)
        if m is not None:
            m(f"wait{layer}.end")
        return out

    model.take_engram_rows = take
    return orig


def _gaps(recs: dict, marks: list, dev_to_chip: dict) -> dict:
    """Per chip: span, busy, and idle gaps charged to the host segment whose ids launched the program after the gap."""
    import bisect

    ids_at = [m[2] for m in marks]
    names = [m[0] for m in marks]
    per_chip = {}
    for rid, chips in recs.items():
        for d, e in chips.items():
            per_chip.setdefault(dev_to_chip.get(int(d), int(d)), []).append((e[1], e[2], rid))
    out = {}
    for c, progs in per_chip.items():
        progs.sort()
        first, end, busy = progs[0][0], progs[0][0], 0.0
        gaps = {}
        biggest = []
        seg_busy, seg_last = {}, {}  # X.10: device busy of the programs each host segment launched, its last program
        for st, en, rid in progs:
            k2 = bisect.bisect_right(ids_at, rid) - 1
            s2 = names[k2] if k2 >= 0 else "before"
            seg_busy[s2] = seg_busy.get(s2, 0.0) + max(0.0, en - max(st, end))
            if rid >= seg_last.get(s2, (-1, 0))[0]:
                seg_last[s2] = (rid, en - st)
            if st > end:
                k = bisect.bisect_right(ids_at, rid) - 1
                seg = names[k] if k >= 0 else "before"
                gaps[seg] = gaps.get(seg, 0.0) + (st - end)
                biggest.append((st - end, seg, rid))
                busy += en - st
            else:
                busy += max(0.0, en - end)
            end = max(end, en)
        biggest.sort(reverse=True)
        out[c] = {
            "span_ms": (end - first) / 1e6,
            "busy_ms": busy / 1e6,
            "gaps_ms": {k: v / 1e6 for k, v in gaps.items()},
            "top": [(round(g / 1e6, 3), s, r) for g, s, r in biggest[:8]],
            "all": [(g / 1e6, s, r) for g, s, r in biggest],  # X.10: every gap (ms, segment, ending program id)
            "seg_busy": {k: v / 1e6 for k, v in seg_busy.items()},
            "seg_last": {k: v[1] / 1e6 for k, v in seg_last.items()},
        }
    return out


def _instrument_jobs(model, marks_ref, jobs: list):
    """X.10: every engram worker job (fill / fill_tp on the instance) timed on the host clock of the current marks, with
    the device-operation ids the main thread had launched at its start and end, and the job's own phase ms."""
    import ttnn

    gid = ttnn._ttnn.get_device_operation_id
    saved = []
    for layer, rows in model.engram.items():
        for name in ("fill", "fill_tp"):
            orig = getattr(rows, name)

            def wrapped(*a, _orig=orig, _layer=layer, **k):
                m = marks_ref[0]
                t0, i0 = time.perf_counter(), int(gid())
                times = _orig(*a, **k)
                if m is not None:
                    jobs.append(
                        {
                            "layer": _layer,
                            "t0": (t0 - m.t0) * 1e3,
                            "t1": (time.perf_counter() - m.t0) * 1e3,
                            "id0": i0,
                            "id1": int(gid()),
                            "times": dict(times),
                        }
                    )
                return times

            setattr(rows, name, wrapped)
            saved.append((rows, name))
    return saved


def _seg_label(seg: str) -> str:
    """A mark name -> the census label of the host segment (a wait belongs to its layer)."""
    return "L" + seg[4:].split(".")[0] if seg.startswith("wait") else seg


def _split(g: dict, marks: list, jobs: list, census: dict, clean_rows: list, wall: float) -> dict:
    """X.10: one chip's exposed host time (wall - busy) split: start / end latency; per engram layer the gap right after
    its wait (the device idled for the rows) and the gaps of programs launched while a worker job ran (contention);
    the other gaps launch-bound, split by the segment's host share in ttnn op calls (census) vs the rest (Python)."""
    ids = {n: i for n, _, i in marks}
    t_clean = [(n, t) for n, t, _ in clean_rows]
    host_seg = {}
    for (a, ta), (_, tb) in zip(t_clean, t_clean[1:]):
        lab = _seg_label(a)
        host_seg[lab] = host_seg.get(lab, 0.0) + (tb - ta)
    ops_seg = {lab: sum(ms for _, ms in rows) for lab, rows in census.items()}
    out = {"wall": wall, "busy": g["busy_ms"], "span": g["span_ms"], "start_end": wall - g["span_ms"]}
    eng_wait, eng_cont, launch_ops, launch_py, launch_seg = {}, {}, 0.0, 0.0, {}
    for gm, seg, rid in g["all"]:
        # the gap ended by the first programs launched after a wait: the device idled for that layer's rows
        if seg.startswith("wait") and seg.endswith(".end") and rid <= ids[seg] + 3:
            lay = int(seg[4:].split(".")[0])
            eng_wait[lay] = eng_wait.get(lay, 0.0) + gm
            continue
        jl = [j["layer"] for j in jobs if j["id0"] < rid <= j["id1"] + 1]
        if jl:
            eng_cont[jl[0]] = eng_cont.get(jl[0], 0.0) + gm
            continue
        lab = _seg_label(seg)
        f = min(1.0, ops_seg.get(lab, 0.0) / host_seg[lab]) if host_seg.get(lab) else 0.0
        launch_ops += gm * f
        launch_py += gm * (1 - f)
        launch_seg[lab] = launch_seg.get(lab, 0.0) + gm
    out.update(
        engram_wait=eng_wait,
        engram_contention=eng_cont,
        launch_ops=launch_ops,
        launch_python=launch_py,
        launch_by_segment=dict(sorted(launch_seg.items(), key=lambda x: -x[1])[:10]),
    )
    return out


def _clean_replay(g: dict, clean_rows: list, jobs: list, census: dict) -> dict:
    """X.10: the rt run's host is slowed by the profiler (its gaps overstate the launch cost), so replay its device
    work against a clean run's host marks: per host segment (mark to mark, in order) the device starts the segment's
    programs once the host reached the segment and the previous segment's work ended, and ends no earlier than the
    segment's last launch plus that program's time. The idle a segment adds is charged: the segment after an engram
    wait -> the wait; a segment whose host interval overlaps a worker job -> launch-bound while the job ran
    (contention, an upper bound); the rest launch-bound, split by its host share in ttnn op calls (census)."""
    t = [(n, ms) for n, ms, _ in clean_rows]
    ops_seg = {lab: sum(ms for _, ms in rows) for lab, rows in census.items()}
    dev, rows = 0.0, []
    cat = {"engram_wait": 0.0, "at_wait_launch": 0.0, "while_job": 0.0, "launch_ops": 0.0, "launch_python": 0.0}
    waited = {}
    for j in jobs:
        waited[j["layer"]] = waited.get(j["layer"], 0.0)
    for n, ms in t:  # (the main thread's wait for a job: from its job's end, if the wait started before it)
        if n.startswith("wait") and n.endswith(".start"):
            lay = int(n[4:].split(".")[0])
            ends = [j["t1"] for j in jobs if j["layer"] == lay]
            if ends:
                waited[lay] = max(0.0, ends[0] - ms)
    first = None
    for (a, ta), (_, tb) in zip(t, t[1:]):
        busy = g["seg_busy"].get(a, 0.0)
        if busy <= 0.0:
            continue
        if first is None:
            first, dev = ta, ta
        start = max(dev, ta)
        end = max(start + busy, tb + g["seg_last"].get(a, 0.0))
        idle = (start - dev) + (end - start - busy)
        dev = end
        lab = _seg_label(a)
        if a.startswith("wait") and a.endswith(".start"):
            kind = "at_wait"  # the prefetch launches + the main thread's wait for the rows (split below)
        elif any(j["t0"] < tb and ta < j["t1"] for j in jobs):
            kind = "while_job"
        else:
            kind = "launch"
        if kind == "launch":
            f = min(1.0, ops_seg.get(lab, 0.0) / (tb - ta)) if tb > ta else 0.0
            cat["launch_ops"] += idle * f
            cat["launch_python"] += idle * (1 - f)
        elif kind == "at_wait":
            w = min(idle, waited.get(int(a[4:].split(".")[0]), 0.0))
            cat["engram_wait"] += w
            cat["at_wait_launch"] += idle - w
        else:
            cat[kind] += idle
        rows.append((a, round(ta, 2), round(tb, 2), round(busy, 2), round(idle, 2), kind))
    done = t[-1][1]
    cat.update(start=first or 0.0, end=max(0.0, done - dev), device_end=dev, wall=done, busy=g["busy_ms"], rows=rows)
    return cat


def _check_tp_rows(model, mesh, tokens):
    """Every engram layer: fill_tp + the TP gather into a fresh buffer vs fill (the replicated rows), every chip."""
    import torch

    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import ccl
    from models.demos.deepseek_v41_flash_d_p.tt.common import TP_AXIS

    model.sync()
    for i, rows in model.engram.items():
        tail = rows.new_tail()
        ids = tokens.reshape(-1).clone()
        a = rows.allocate(ids.numel())
        rows.fill(a, ids, tail)
        b = rows.allocate_tp(ids.numel())
        rows.fill_tp(b, ids, tail, threads=model._hide_threads)
        full = ttnn.allocate_tensor_on_device(a.shape, a.dtype, a.layout, mesh, ttnn.DRAM_MEMORY_CONFIG)
        ccl.all_gather(b, 2, TP_AXIS, out=full)
        model.sync()
        da, df = ttnn.get_device_tensors(a), ttnn.get_device_tensors(full)
        same = [torch.equal(ttnn.to_torch(x), ttnn.to_torch(y)) for x, y in zip(da, df)]
        print(f"[P.78] layer {i}: TP-sharded rows gathered == replicated rows on {sum(same)} / {len(same)} chips")
        assert all(same), f"layer {i}: chips {[k for k, v in enumerate(same) if not v]} differ"
        for t in (a, b, full):
            ttnn.deallocate(t)


def _census(model, one_run, marks_ref):
    """One run with every outermost FastOperation call timed on the host, keyed by the layer mark it ran under."""
    import ttnn.decorators as D

    out, depth = {}, [0]
    orig = D.FastOperation.__call__

    def call(self, *args, **kwargs):
        if depth[0]:
            return orig(self, *args, **kwargs)
        depth[0] = 1
        t0 = time.perf_counter()
        try:
            return orig(self, *args, **kwargs)
        finally:
            depth[0] = 0
            m = marks_ref[0]
            if m is not None and m.rows:
                lab = m.rows[-1][0]
                lab = lab if not lab.startswith("wait") else "L" + lab[4:].split(".")[0]
                out.setdefault(lab, []).append((self.python_fully_qualified_name, (time.perf_counter() - t0) * 1e3))

    D.FastOperation.__call__ = call
    try:
        one_run()
    finally:
        D.FastOperation.__call__ = orig
    return out


@mesh_parametrize(timeout_s=0)
def test_host_idle_map(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    assert settings.get("DECODER_SKIP") == "decode", "run with DS41_DECODER_SKIP=decode"
    if os.environ.get(settings.env_name("ENGRAM_TABLE")) is None:  # (as test_skip_positions)
        monkeypatch.setenv(settings.env_name("ENGRAM_TABLE"), str(settings.table["ENGRAM_TABLE"].default))
    chunk = int(S.get("target.chunk"))
    p = int(os.environ.get("IDLE_POS", "145408"))
    n_runs = int(os.environ.get("IDLE_RUNS", "7"))
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)  # F282: every chunk here is last=False
    want = max(p + chunk, int(S.data["target"]["seq"]))
    monkeypatch.setitem(S.data["target"], "seq", want)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    print(f"HOST_HIDE={settings.get('HOST_HIDE')} ENGRAM_TABLE={settings.get('ENGRAM_TABLE')}", flush=True)
    ids = PositionGolden(S, p, chunk).tokens()
    prev_t, cur_t = ids[p - chunk : p], ids[p : p + chunk]
    state = model.new_state(p + chunk)
    marks_ref = [None]
    orig = _instrument(model, marks_ref)
    jobs = []
    saved_jobs = _instrument_jobs(model, marks_ref, jobs)
    try:

        def run(tokens, start, m=None, pre=False):
            marks_ref[0] = m
            if m is not None:
                m("begin_chunk")
            model.begin_chunk(tokens, start, last=False)
            if pre:  # X.10: every engram job done (rows on the device) before the clock starts: no engram host work
                model.wait_engram()
                if m is not None:
                    m.t0, m.rows = time.perf_counter(), []
                    m("begin_chunk")
            if m is not None:
                m("embed")
            h = model.embed(tokens)
            for i in layers:
                if m is not None:
                    m(f"L{i}")
                h2 = model.layer(i, h, start, state)
                model.free(h)
                h = h2
            model.free(h)
            if m is not None:
                m("sync")
            marks_ref[0] = None

        def one_run(m=None, pre=False):
            run(prev_t, p - chunk)
            model.sync()
            m = m or Marks()
            m.t0 = time.perf_counter()
            run(cur_t, p, m, pre)
            model.sync()
            m("done")
            return m

        warm_up(mesh_device, lambda: one_run().rows[-1][1])
        if model._engram_full:  # HOST_HIDE=on: the TP-sharded upload + device gather == the replicated upload
            _check_tp_rows(model, mesh_device, cur_t)
        clean, clean_jobs = [], []
        for _ in range(n_runs):
            jobs.clear()
            clean.append(one_run())
            clean_jobs.append(list(jobs))
        walls = [m.rows[-1][1] for m in clean]
        med = sorted(range(n_runs), key=lambda k: walls[k])[n_runs // 2]
        m = clean[med]
        print(f"[P.78] p{p} clean walls {[round(w, 1) for w in walls]} median {walls[med]:.1f} ms", flush=True)
        # X.10: the same chunk with its engram jobs finished before the clock starts (an on-device table's host side,
        # less its id hash / upload): alternating with the shipped runs so drift shows in both
        pre_runs, ship_runs = [], []
        for _ in range(n_runs):
            pre_runs.append(one_run(pre=True))
            ship_runs.append(one_run())
        pw = [r.rows[-1][1] for r in pre_runs]
        sw = [r.rows[-1][1] for r in ship_runs]
        pm = sorted(range(n_runs), key=lambda k: pw[k])[n_runs // 2]
        print(
            f"[X.10] engram jobs done before the clock: walls {[round(w, 1) for w in pw]} median "
            f"{statistics.median(pw):.1f} ms vs shipped (alternating) {[round(w, 1) for w in sw]} median "
            f"{statistics.median(sw):.1f} ms: engram's exposed host {statistics.median(sw) - statistics.median(pw):.2f} ms",
            flush=True,
        )
        lay_p = [(n, t) for n, t, _ in pre_runs[pm].rows if n.startswith("L") or n in ("sync", "embed")]
        print(
            "[X.10] no-engram host per layer (ms): "
            + ", ".join(f"{a}->{b} {tb - ta:.2f}" for (a, ta), (b, tb) in zip(lay_p, lay_p[1:])),
            flush=True,
        )
        print("[P.78] clean host marks (ms from t0): " + ", ".join(f"{n} {t:.2f}" for n, t, _ in m.rows), flush=True)
        for k in ("wait1.start", "wait1.end", "wait14.start", "wait14.end", "sync"):
            vals = [dict((n, t) for n, t, _ in mm.rows).get(k) for mm in clean]
            if all(v is not None for v in vals):
                print(f"[P.78] {k}: median {statistics.median(vals):.2f} ms ({[round(v, 1) for v in vals]})")
        print(f"[P.78] engram worker ms (last run): {model.engram_times}", flush=True)
        lay = [(n, t) for n, t, _ in m.rows if n.startswith("L") or n in ("sync", "embed")]
        steps = [f"{a}->{b} {tb - ta:.2f}" for (a, ta), (b, tb) in zip(lay, lay[1:])]
        print("[P.78] clean host per layer (ms): " + ", ".join(steps), flush=True)
        for j in clean_jobs[med]:
            print(
                f"[X.10] clean engram job layer {j['layer']}: host {j['t0']:.2f} -> {j['t1']:.2f} ms "
                f"({j['t1'] - j['t0']:.2f}), ops launched meanwhile {j['id1'] - j['id0']}, phases "
                + ", ".join(f"{k} {v:.2f}" for k, v in j["times"].items()),
                flush=True,
            )

        # IDLE_SWEEP="defer:threads,...": the clean wall of each (HOST_HIDE_DEFER, HOST_HIDE_THREADS) on this load
        sweep = [x.split(":") for x in os.environ.get("IDLE_SWEEP", "").split(",") if x.strip()]
        if sweep and model._host_hide:
            keep = (model._hide_defer, model._hide_threads)
            res = {}
            for _ in range(2):  # (two rounds, alternating: drift shows as a round difference)
                for d, th in sweep:
                    model._hide_defer, model._hide_threads = int(d), int(th)
                    ws = [one_run().rows[-1][1] for _ in range(n_runs)]
                    res.setdefault((d, th), []).extend(ws)
            for k, ws in res.items():
                print(
                    f"[P.78] sweep defer {k[0]} threads {k[1]}: median {statistics.median(ws):.1f} ms "
                    f"{sorted(round(w, 1) for w in ws)}"
                )
            model._hide_defer, model._hide_threads = keep

        # one clean run with the host time of every outermost ttnn op call (FastOperation), by layer
        census = _census(model, lambda: one_run(), marks_ref)
        tot_ops = sum(ms for rows in census.values() for _, ms in rows)
        n_ops = sum(len(rows) for rows in census.values())
        print(f"[X.10] census: {n_ops} op calls a chunk, {tot_ops:.2f} ms host in op calls", flush=True)
        by_op = {}
        for rows_ in census.values():
            for n_, ms_ in rows_:
                a_ = by_op.setdefault(n_, [0, 0.0])
                a_[0] += 1
                a_[1] += ms_
        top_ops = sorted(by_op.items(), key=lambda x: -x[1][1])[:15]
        print("[X.10] census by op (calls, host ms): " + ", ".join(f"{n_} {c_} {ms_:.2f}" for n_, (c_, ms_) in top_ops))
        for lab in ("L1", "L2", "L3", "L8", "L14"):
            rows = census.get(lab, [])
            tot = sum(ms for _, ms in rows)
            top = sorted(rows, key=lambda x: -x[1])[:8]
            print(
                f"[P.78] census {lab}: {len(rows)} op calls, {tot:.2f} ms in ops; slowest "
                + ", ".join(f"{n} {ms:.2f}" for n, ms in top),
                flush=True,
            )

        # one rt run, no op patching
        run(prev_t, p - chunk)
        model.sync()
        jobs.clear()
        profiler.enable(mesh_device, rt=True)
        try:
            mr = Marks()
            mr.t0 = time.perf_counter()
            run(cur_t, p, mr)
            model.sync()
            mr("done")
            profiler.finish_rt()
            pr = profiler._P
            recs, d2c = pr["rt_records"], pr["dev_to_chip"]
        finally:
            profiler.disable()
        g = _gaps(recs, mr.rows, d2c)
        wall_rt = mr.rows[-1][1]
        print(f"[P.78] rt run wall {wall_rt:.1f} ms; host marks: " + ", ".join(f"{n} {t:.2f}" for n, t, _ in mr.rows))
        crit = max(g, key=lambda c: g[c]["span_ms"])
        for c in sorted(g):
            r = g[c]
            if c == crit or c in (0, 31):
                gs = sorted(r["gaps_ms"].items(), key=lambda x: -x[1])
                print(
                    f"[P.78] chip {c}{' (longest span)' if c == crit else ''}: span {r['span_ms']:.2f} busy "
                    f"{r['busy_ms']:.2f} idle {r['span_ms'] - r['busy_ms']:.2f} ms; idle by segment "
                    + ", ".join(f"{k} {v:.2f}" for k, v in gs[:12])
                    + f"; biggest {r['top']}",
                    flush=True,
                )
        spans = [g[c]["span_ms"] for c in g]
        busys = [g[c]["busy_ms"] for c in g]
        print(
            f"[P.78] chips: span {min(spans):.2f}..{max(spans):.2f}, busy {min(busys):.2f}..{max(busys):.2f} ms; "
            f"wall - span (start + end latency) {wall_rt - max(spans):.2f} ms"
        )
        # X.10: the exposed host time (wall - busy) split engram / launch / other, the longest-span chip and the mean
        for j in jobs:
            print(
                f"[X.10] rt engram job layer {j['layer']}: host {j['t0']:.2f} -> {j['t1']:.2f} ms, op ids "
                f"{j['id0']}..{j['id1']}, phases " + ", ".join(f"{k} {v:.2f}" for k, v in j["times"].items()),
                flush=True,
            )
        sp = {c: _split(g[c], mr.rows, jobs, census, m.rows, wall_rt) for c in g}
        keys = ("start_end", "launch_ops", "launch_python")
        mean = {k: statistics.mean(sp[c][k] for c in sp) for k in keys}
        for kind in ("engram_wait", "engram_contention"):
            lays = sorted({lay for c in sp for lay in sp[c][kind]})
            mean[kind] = {lay: statistics.mean(sp[c][kind].get(lay, 0.0) for c in sp) for lay in lays}
        mean["busy"] = statistics.mean(sp[c]["busy"] for c in sp)
        for lab, r in (("longest-span chip %d" % crit, sp[crit]), ("mean over chips", mean)):
            exposed = wall_rt - r["busy"]
            print(
                f"[X.10] exposed host split ({lab}): wall {wall_rt:.2f} - busy {r['busy']:.2f} = {exposed:.2f} ms: "
                f"start + end latency {r['start_end']:.2f}; engram wait {({k: round(v, 2) for k, v in r['engram_wait'].items()})}; "
                f"engram job contention {({k: round(v, 2) for k, v in r['engram_contention'].items()})}; "
                f"launch-bound: in op calls {r['launch_ops']:.2f}, Python / other {r['launch_python']:.2f}",
                flush=True,
            )
        lb = {k: round(v, 2) for k, v in sp[crit]["launch_by_segment"].items()}
        print(f"[X.10] launch-bound idle by segment (chip {crit}, rt run): {lb}")
        rp = _clean_replay(g[crit], m.rows, clean_jobs[med], census)
        exposed = rp["wall"] - rp["busy"]
        print(
            f"[X.10] clean replay (chip {crit}, the median clean run's host marks): wall {rp['wall']:.2f}, modelled "
            f"device end {rp['device_end']:.2f}, busy {rp['busy']:.2f}, exposed {exposed:.2f} ms = start "
            f"{rp['start']:.2f} + end {rp['end']:.2f} + engram wait {rp['engram_wait']:.2f} + launches at the first "
            f"engram wait (prefetches) {rp['at_wait_launch']:.2f} + launch-bound while an "
            f"engram job ran {rp['while_job']:.2f} + launch-bound in op calls {rp['launch_ops']:.2f} + Python / other "
            f"{rp['launch_python']:.2f} (+ model residual {exposed - rp['start'] - rp['end'] - rp['engram_wait'] - rp['at_wait_launch'] - rp['while_job'] - rp['launch_ops'] - rp['launch_python']:.2f})",
            flush=True,
        )
        print(
            "[X.10] clean replay segments (mark, host from, to, device busy, idle added, kind): "
            + "; ".join(str(r) for r in rp["rows"] if r[4] >= 0.05),
            flush=True,
        )
        try:
            import json
            from pathlib import Path

            d = Path(S.art) / "idle_map"
            d.mkdir(parents=True, exist_ok=True)
            tag = os.environ.get("BRINGUP_POS_TOKENS", "prose")
            rec = {
                "pos": p,
                "tokens": tag,
                "host_hide": settings.get("HOST_HIDE"),
                "clean_walls": walls,
                "clean_marks": [(n, t) for n, t, _ in m.rows],
                "clean_jobs": clean_jobs[med],
                "census_ops_ms": {k: sum(ms for _, ms in v) for k, v in census.items()},
                "census_ops_n": {k: len(v) for k, v in census.items()},
                "rt_wall": wall_rt,
                "rt_marks": mr.rows,
                "rt_jobs": jobs,
                "split": {str(c): {k: v for k, v in sp[c].items()} for c in sp},
                "clean_replay": rp,
                "pre_walls": pw,
                "ship_walls_alternating": sw,
                "pre_marks": [(n, t) for n, t, _ in pre_runs[pm].rows],
                "seg_busy": g[crit]["seg_busy"],
                "crit": crit,
            }
            (d / f"idle_p{p}_t{tag}.json").write_text(json.dumps(rec, indent=1, default=str))
            print(f"[X.10] side record {d / f'idle_p{p}_t{tag}.json'}")
        except OSError as e:
            print(f"[X.10] side record not written: {e}")
    finally:
        model.take_engram_rows = orig
        for rows, name in saved_jobs:
            delattr(rows, name)
        _free(state)
        if loaded != "decode":
            model.use_decoder_skip(loaded)
