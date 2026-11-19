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
        for st, en, rid in progs:
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
        }
    return out


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
    try:

        def run(tokens, start, m=None):
            marks_ref[0] = m
            if m is not None:
                m("begin_chunk")
            model.begin_chunk(tokens, start, last=False)
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

        def one_run(m=None):
            run(prev_t, p - chunk)
            model.sync()
            m = m or Marks()
            m.t0 = time.perf_counter()
            run(cur_t, p, m)
            model.sync()
            m("done")
            return m

        warm_up(mesh_device, lambda: one_run().rows[-1][1])
        if model._engram_full:  # HOST_HIDE=on: the TP-sharded upload + device gather == the replicated upload
            _check_tp_rows(model, mesh_device, cur_t)
        clean = [one_run() for _ in range(n_runs)]
        walls = [m.rows[-1][1] for m in clean]
        med = sorted(range(n_runs), key=lambda k: walls[k])[n_runs // 2]
        m = clean[med]
        print(f"[P.78] p{p} clean walls {[round(w, 1) for w in walls]} median {walls[med]:.1f} ms", flush=True)
        print("[P.78] clean host marks (ms from t0): " + ", ".join(f"{n} {t:.2f}" for n, t, _ in m.rows), flush=True)
        for k in ("wait1.start", "wait1.end", "wait14.start", "wait14.end", "sync"):
            vals = [dict((n, t) for n, t, _ in mm.rows).get(k) for mm in clean]
            if all(v is not None for v in vals):
                print(f"[P.78] {k}: median {statistics.median(vals):.2f} ms ({[round(v, 1) for v in vals]})")
        print(f"[P.78] engram worker ms (last run): {model.engram_times}", flush=True)
        lay = [(n, t) for n, t, _ in m.rows if n.startswith("L") or n in ("sync", "embed")]
        steps = [f"{a}->{b} {tb - ta:.2f}" for (a, ta), (b, tb) in zip(lay, lay[1:])]
        print("[P.78] clean host per layer (ms): " + ", ".join(steps), flush=True)

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
    finally:
        model.take_engram_rows = orig
        _free(state)
        if loaded != "decode":
            model.use_decoder_skip(loaded)
