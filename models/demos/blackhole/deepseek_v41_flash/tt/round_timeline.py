# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Speculative-decode ROUND timeline (instrumentation only; OFF unless DSV41_ROUND_TL=1, then installed by a 3-line hook at the end of tt/spec_model.py).

Nothing in the model code is edited: ``install()`` wraps methods of SpecRunner / AdaptiveSpec / SpecDecoder / the Engram host classes / a few ttnn calls with timers. With the variable unset
nothing is imported and nothing is wrapped, so the hot path is byte-identical to the shipped code.

Host side (per round, ms, perf_counter): ensure, use, force, feed (= hash + rows + pack_prep + upload + other), enqueue (execute_trace, non-blocking), sync (first synchronize_device =
device wait), rb_issue / rb_sync / rb_parse (read-back), round (whole SpecRunner._round), gap_before (post-processing of the previous round: python loop over users + scheduler,
i.e. the time the DEVICE sits idle between two rounds is gap_before + the host phases before enqueue + the read-back), period (start-to-start).
Device side:
  * ``dev_only_ms``: after each run pass the traced round of every used runner is replayed back-to-back N times without any host work (state is destroyed: end of the pass only).
  * op-id phase marks: while a round trace is captured, the device-operation-id range of every phase (embedding, per-layer Engram / layer, head, sampler, accept, commit, drafter) is recorded
    (marks_trace<id>.json); with DSV41_ROUND_TL_PROF=1 (+ TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 and a TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT large enough) the
    ``DSV41_ROUND_TL_PROF_AT``-th round of every runner is bracketed by ReadDeviceProfiler so that tools/round_tl_devprof.py can attribute the traced kernel time to the phases.
Env: DSV41_ROUND_TL=1, DSV41_ROUND_TL_OUT=<dir> (default /tmp/round_tl), DSV41_ROUND_TL_NREP (back-to-back replays, default 40).
"""

import json
import os
import time

import torch

import ttnn

pc = time.perf_counter
OUT = os.environ.get("DSV41_ROUND_TL_OUT", "/tmp/round_tl")
_CUR = [None]  # the record of the round in flight
_REC = []  # all records of the current pass
_STATE = {"pass": 0, "label": "", "last_end": None, "calib": False, "cap": False, "marks": None, "tid_marks": {}, "nround": {}, "prof_rounds": []}


def _ms(t0):
    return (pc() - t0) * 1e3


def _acc(name):
    c = _CUR[0]
    return c, name


def _wrap(fn, name, count=False):
    def w(*a, **k):
        c = _CUR[0]
        if c is None:
            return fn(*a, **k)
        t = pc()
        try:
            return fn(*a, **k)
        finally:
            nm = name
            if count:  # first sync of the round = device wait, later ones = read-back sync
                c["_ns"] = c.get("_ns", 0) + 1
                nm = name if c["_ns"] == 1 else "rb_sync"
            c[nm] = c.get(nm, 0.0) + _ms(t)

    w.__wrapped__ = fn
    return w


def _opid():
    from ttnn import _ttnn

    return int(_ttnn.get_device_operation_id())


def _mk(name, fn, marks):
    def w(*a, **k):
        i0 = _opid()
        try:
            return fn(*a, **k)
        finally:
            marks.append((name, i0, _opid()))

    return w


def _forward_with_marks(orig):
    """SpecDecoder.forward during a trace capture: record the op-id range of every phase (instance-level temporary wrappers)."""

    def fwd(self):
        if not _STATE["cap"]:
            return orig(self)
        marks = []
        patched = []

        def p(obj, attr, name):
            f = getattr(obj, attr)
            setattr(obj, attr, _mk(name, f, marks))
            patched.append((obj, attr))

        p(self.embedding, "forward", "embedding")
        p(self.head, "forward", "head")
        p(self.head, "sample_global", "sampler")
        p(self, "commit", "commit")
        p(self, "_engram_fwd", "engram")
        p(self.drafter, "write_main_full", "drafter_write_main")
        p(self.drafter, "draft_full", "drafter_draft")
        for lid, layer, _ in self.layers:
            p(layer, "forward", f"layer{lid}")
        i0 = _opid()
        try:
            return orig(self)
        finally:
            i1 = _opid()
            for obj, attr in patched:
                try:
                    delattr(obj, attr)
                except AttributeError:
                    pass
            _STATE["marks"] = {"begin": i0, "end": i1, "k": self.n - 1, "U": self.U, "marks": marks}

    return fwd


def _stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return {"mean": sum(xs) / len(xs), "p50": xs[len(xs) // 2], "p95": xs[min(len(xs) - 1, int(0.95 * len(xs)))], "n": len(xs)}


def _dump(tag, extra=None):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"rounds_pass{_STATE['pass']}_{tag}.json")
    with open(path, "w") as f:
        json.dump({"pass": _STATE["pass"], "label": _STATE["label"], "records": _REC, "extra": extra or {}}, f)
    print(f"ROUND_TL wrote {len(_REC)} rounds -> {path}", flush=True)


def _rep(md, fn, nrep):
    fn()
    fn()
    ttnn.synchronize_device(md)
    t = pc()
    for _ in range(nrep):
        fn()
    ttnn.synchronize_device(md)
    return _ms(t) / nrep


def _device_only(runners):
    """Back-to-back replays of the traced round of every used runner, no host work in between (end of a pass; destroys the state). The monolithic (greedy) trace, and when the runner has
    the three traces of the sampled round: A (verify: embedding, 40 layers, head, hidden taps), S (sampler, with the params of the last round), B (accept, commit, drafter), alone and as A+S+B."""
    res = {}
    nrep = int(os.environ.get("DSV41_ROUND_TL_NREP", "40"))
    for r in runners:
        P = getattr(r, "_tl_last_bucket", "none")
        if P == "none" or r.tr.get(P) is None or r.tr[P].get("tid") is None:
            continue
        t = r.tr[P]
        md = r.md
        ex = lambda tid: ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        d = {"bucket": P, "nrep": nrep, "mono_ms": _rep(md, lambda: ex(t["tid"]), nrep)}
        if r.sampled and t.get("tid_a") is not None:
            d["A_verify_ms"] = _rep(md, lambda: ex(t["tid_a"]), nrep)
            from models.demos.blackhole.deepseek_v41_flash.tt.device_sampler import sampling_rows

            Bn = r.B * r.n
            for tag, (tt_, kk_, pp_) in {"greedy": (0.0, 0, 1.0), "sampled": (1.0, 0, 0.95)}.items():
                r.samp.set_params(*sampling_rows([tt_] * Bn, [kk_] * Bn, [pp_] * Bn, torch.Generator().manual_seed(1), Bn))
                d[f"S_{tag}_ms"] = _rep(md, lambda: t["tid_s"].replay(), nrep)
            d["B_tail_ms"] = _rep(md, lambda: ex(t["tid_b"]), nrep)

            def asb():
                ex(t["tid_a"])
                t["tid_s"].replay()
                ex(t["tid_b"])

            d["ASB_ms"] = _rep(md, asb, nrep)
            d["sampler_fast"] = bool(getattr(r.samp, "fast", False))
        res[f"k{r.k}" + ("" if r.draft else "_nodraft")] = d
    return res


def install():
    from models.demos.blackhole.deepseek_v41_flash.tt import spec_model as sm
    from models.demos.blackhole.deepseek_v41_flash.tt.decoder import DSV41Decoder
    from models.demos.blackhole.deepseek_v41_flash.tt.engram import HostEngramRows
    from models.demos.blackhole.deepseek_v41_flash.tt.engram_ragged import RaggedNgramHash
    from models.demos.blackhole.deepseek_v41_flash.tt.spec_decoder import SpecDecoder

    if getattr(sm, "_RTL_INSTALLED", False):
        return
    sm._RTL_INSTALLED = True
    SR, AD = sm.SpecRunner, sm.AdaptiveSpec
    prof_on = os.environ.get("DSV41_ROUND_TL_PROF") == "1"
    prof_at = int(os.environ.get("DSV41_ROUND_TL_PROF_AT", "12"))
    sample_win = int(os.environ.get("DSV41_ROUND_TL_SAMPLE_WIN", "0"))
    _STATE["gen"] = torch.Generator().manual_seed(1234)

    # ---- named phase timers (accumulate into the round in flight) ----
    for cls, attr, name in (
        (SR, "_ensure", "ensure"),
        (SR, "_use", "use"),
        (SR, "_feed", "feed"),
        (SR, "_readback", "readback"),
        (SR, "_set_sample_params", "samp_params"),
        (SpecDecoder, "set_force", "force"),
        (DSV41Decoder, "prepare_packed_inputs", "pack_prep"),
        (DSV41Decoder, "upload_packed_inputs", "upload"),
        (RaggedNgramHash, "__call__", "hash"),
        (HostEngramRows, "rows_all", "rows"),
    ):
        setattr(cls, attr, _wrap(getattr(cls, attr), name))
    ttnn.execute_trace = _wrap(ttnn.execute_trace, "enqueue")
    ttnn.synchronize_device = _wrap(ttnn.synchronize_device, "sync", count=True)
    ttnn.from_device = _wrap(ttnn.from_device, "rb_issue")

    # ---- trace capture flag + op-id marks of the captured round ----
    ob, oe = ttnn.begin_trace_capture, ttnn.end_trace_capture

    def begin(*a, **k):
        _STATE["cap"], _STATE["marks"] = True, None
        return ob(*a, **k)

    def end(md, tid, *a, **k):
        r = oe(md, tid, *a, **k)
        _STATE["cap"] = False
        if _STATE["marks"] is not None:
            os.makedirs(OUT, exist_ok=True)
            with open(os.path.join(OUT, f"marks_trace{int(tid)}.json"), "w") as f:
                json.dump(_STATE["marks"], f)
        return r

    ttnn.begin_trace_capture, ttnn.end_trace_capture = begin, end
    SpecDecoder.forward = _forward_with_marks(SpecDecoder.forward)

    # ---- the round ----
    orig_round, orig_cal = SR._round, SR.calibrate

    def calibrate(self, *a, **k):
        _STATE["calib"] = True
        try:
            return orig_cal(self, *a, **k)
        finally:
            _STATE["calib"] = False
            _STATE["last_end"] = None

    def _round(self, X, base, force=None):
        t0 = pc()
        gap = (t0 - _STATE["last_end"]) * 1e3 if _STATE["last_end"] is not None else None
        key = (id(self), self.k, self.draft)
        nr = _STATE["nround"][key] = _STATE["nround"].get(key, 0) + 1
        do_prof = prof_on and not _STATE["calib"] and nr in (prof_at, prof_at + 1)
        if do_prof:
            ttnn.synchronize_device(self.md)
            ttnn.ReadDeviceProfiler(self.md)
        smode = "g"
        if sample_win and not _STATE["calib"] and self.samp is not None:
            # greedy rows for sample_win rounds, then sampled rows (DSV41_ROUND_TL_SAMPLE=T:top_k:top_p) for sample_win rounds, alternating (same stream, same process)
            smode = "s" if (_STATE["nr_pass"] // sample_win) % 2 else "g"
            if smode == "s":
                t_, k_, p_ = os.environ.get("DSV41_ROUND_TL_SAMPLE", "1.0:0:0.95").split(":")
                self.sample_cfg = dict(temperature=float(t_), top_k=int(k_), top_p=float(p_), gens=_STATE["gen"])
            else:
                self.sample_cfg = None
            _STATE["nr_pass"] += 1
        cur = {"k": self.k, "n": self.n, "T": self.T, "B": self.B, "draft": self.draft, "calib": _STATE["calib"], "t0": t0, "rows": smode}
        _CUR[0] = cur
        try:
            out = orig_round(self, X, base, force)
        finally:
            _CUR[0] = None
        t1 = pc()
        if do_prof:
            ttnn.ReadDeviceProfiler(self.md)
            tid = self.tr[self.cur_bucket]["tid"] if self.cur_bucket in self.tr else None
            _STATE["prof_rounds"].append({"k": self.k, "nround": nr, "trace_id": None if tid is None else int(tid), "bucket": self.cur_bucket})
        cur["round"] = (t1 - t0) * 1e3
        cur["gap_before"] = gap
        cur["bucket"] = self.cur_bucket
        cur["amax_base"] = int(base.max())
        cur["rb_parse"] = cur.get("readback", 0.0) - cur.get("rb_issue", 0.0) - cur.get("rb_sync", 0.0)
        cur["feed_other"] = cur.get("feed", 0.0) - cur.get("hash", 0.0) - cur.get("rows", 0.0) - cur.get("pack_prep", 0.0) - cur.get("upload", 0.0)
        self._tl_last_bucket = self.cur_bucket
        _STATE["last_end"] = t1
        _REC.append(cur)
        return out

    SR._round, SR.calibrate = _round, calibrate

    # ---- passes ----
    def wrap_run(cls, runners_of):
        orig = cls.run

        def run(self, X, base, max_new, *a, **k):
            _STATE["pass"] += 1
            _STATE["last_end"], _STATE["nround"] = None, {}
            _REC.clear()
            _STATE["mode"] = "gs" if os.environ.get("DSV41_ROUND_TL_SAMPLE_WIN") else "g"
            _STATE["label"] = f"{cls.__name__} B={getattr(self, 'B', '?')} policy={k.get('policy') or getattr(self, 'policy', '-')} sample_win={os.environ.get('DSV41_ROUND_TL_SAMPLE_WIN', '-')}"
            _STATE["nr_pass"] = 0
            try:
                return orig(self, X, base, max_new, *a, **k)
            finally:
                extra = {"label": _STATE["label"], "prof_rounds": list(_STATE["prof_rounds"])}
                try:
                    extra["dev_only"] = _device_only(runners_of(self))
                except Exception as e:  # noqa: BLE001 (instrumentation must never break the run)
                    extra["dev_only_error"] = repr(e)
                _dump(f"{getattr(self, 'policy', 'x')}_{_STATE['mode']}", extra)

        cls.run = run

    wrap_run(SR, lambda s: [s])
    wrap_run(AD, lambda s: list(s.runners.values()) + [r for r in (s.runner0, s.probe0) if r is not None])
    print(f"ROUND_TL installed (out {OUT}, prof {prof_on}@{prof_at})", flush=True)
