# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.66 (owner 2026-11-14): the per-layer Python glue (LEAN_PY legacy | on, tt/lean.py), measured on real chunks.

Terms:
  host_ms   = first host call of a chunk to the moment the last layer's ops are enqueued (before the closing sync).
  busy_ms   = host_ms minus the time the main thread waited for the engram worker (ENGRAM_ASYNC) inside the chunk.
  wall_ms   = to the end of the sync after the last layer (the clean wall: no profiler, no instrumentation).
  side      = LEAN_PY legacy | on (tt/lean.set_mode, switched in the loaded model).
  census    = one pass (chunks 1..10, the settings' LEAN_PY): every ccl.dealloc call by its caller (the step
              for the block's frees) with the freed tensor's shape / dtype / buffer type and the binding's host us,
              every Python-side ttnn.allocate_tensor_on_device call, the same way.
  cProfile  = tottime per function, one pass a side (inflated by the profiler; differences between sides only).
Per workload (novel, code): chunks 0..10 of fresh requests back to back (chunk 0 is not counted). One warm pass a side,
then P66_CLEAN rounds with the sides alternating in each round. Means over chunks 1..10, and per side the mean of
the per-round means with the paired difference to the first side and its standard error (DS41-264: one pass a side
cannot split a few ms). Outputs: every side's chunk 1 and chunk 10 output streams must equal the first side's bit
for bit (max |diff| 0).

Output: results/<task>_lean_py.json (record_json) and the log.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_lean_py.py
P66_QUICK=1: 4 layers, chunks 0..3, novel (iterating only). P66_CENSUS=0 skips the census, P66_CPROFILE=0 the cProfile,
P66_OPS=0 the op pass (one pass a side: every ttnn op launch by its calling line, median host us, launches a chunk).
"""

from __future__ import annotations

import cProfile
import os
import pstats
import statistics
import sys
import time
from collections import defaultdict

from loguru import logger

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import begin_chunk, mesh_parametrize, spec
from models.demos.common.bringup.testing.positions import _free

S = spec()
QUICK = os.environ.get("P66_QUICK") == "1"
SEQ = 56320
CHUNKS = 4 if QUICK else 11
WORKLOADS = ("novel",) if QUICK else tuple(os.environ.get("P66_WL", "novel,code").split(","))
SIDES = tuple(os.environ.get("P66_SIDES", "legacy,on").split(","))
CLEAN = int(os.environ.get("P66_CLEAN", "4"))
CENSUS = os.environ.get("P66_CENSUS", "1") == "1"
PROFILE = os.environ.get("P66_CPROFILE", "1") == "1"
OPS = os.environ.get("P66_OPS", "1") == "1"
OPS_WATCH = ("attention.py", "moe_block_main_adapter.py", "residual.py")  # (the files whose op calls P.66 changed)
KEEP_CHUNKS = (1, CHUNKS - 1)
_WAIT = [0.0]


def _set_side(mesh, side: str) -> None:
    from models.demos.deepseek_v41_flash_d_p.tt import lean

    lean.set_mode(side)


def _pass(model, mesh, layers, tokens, prof=None, keep=None, ref=None) -> list:
    """Chunks 0..CHUNKS-1 of a fresh request. -> per chunk {host_ms, wall_ms, wait_ms, busy_ms[, diff]}."""
    from models.demos.deepseek_v41_flash_d_p.tests.workloads import maxdiff as _maxdiff

    chunk = int(S.get("target.chunk"))
    state = model.new_state(SEQ)
    rows = []
    try:
        for c in range(CHUNKS):
            st = c * chunk
            t = tokens[st : st + chunk]
            ttnn.synchronize_device(mesh)
            on = c >= 1
            if prof is not None and on:
                prof.enable()
            _WAIT[0] = 0.0
            t0 = time.perf_counter()
            begin_chunk(model, t, st)
            h = model.embed(t)
            for i in layers:
                h2 = model.layer(i, h, st, state)
                model.free(h)
                h = h2
            t1 = time.perf_counter()
            if prof is not None and on:
                prof.disable()
            ttnn.synchronize_device(mesh)
            t2 = time.perf_counter()
            row = {"host_ms": (t1 - t0) * 1e3, "wall_ms": (t2 - t0) * 1e3, "wait_ms": _WAIT[0] * 1e3}
            row["busy_ms"] = row["host_ms"] - row["wait_ms"]
            if c in KEEP_CHUNKS and hasattr(h, "streams"):
                if keep is not None:
                    keep[c] = ttnn.clone(h.streams)
                if ref is not None and c in ref:
                    row["diff"] = _maxdiff(h.streams, ref[c])
            model.free(h)
            rows.append(row)
    finally:
        _free(state)
    return rows


def _mean(rows, k):
    return statistics.mean(r[k] for r in rows[1:])


# ============================================================================ census
class Census:
    """ccl.dealloc's binding and ttnn.allocate_tensor_on_device, by caller and tensor spec."""

    def __init__(self):
        self.on = False
        self.free = defaultdict(lambda: [0, 0.0])
        self.alloc = defaultdict(lambda: [0, 0.0])

    @staticmethod
    def _caller(depth: int) -> str:
        f = sys._getframe(depth)
        skip = ("ccl.py", "census")
        while f is not None:
            fn = os.path.basename(f.f_code.co_filename)
            name = f.f_code.co_name
            if fn == "model.py" and name in ("_free", "_tensors"):
                f = f.f_back
                continue
            if fn == "model.py" and name == "run" and "step" in f.f_locals:
                return f"step:{f.f_locals['step'].name}"
            if fn == "model.py" and name in ("__call__", "kv_only") and "s" in f.f_locals and "env" in f.f_locals:
                return "block_end"
            if not any(k in fn for k in skip):
                return f"{fn}:{f.f_lineno}:{name}"
            f = f.f_back
        return "?"

    @staticmethod
    def _spec(t) -> str:
        try:
            mc = t.memory_config()
            bt = "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"
            return f"{tuple(t.shape)}|{t.dtype}|{t.layout}|{bt}"
        except Exception as e:  # (a freed tensor: no spec)
            return f"?{type(e).__name__}"

    def wrap_free(self, fn):
        census = self

        def run(t):
            if not census.on:
                return fn(t)
            key = (census._caller(2), census._spec(t))
            t0 = time.perf_counter()
            try:
                return fn(t)
            finally:
                r = census.free[key]
                r[0] += 1
                r[1] += time.perf_counter() - t0

        return run

    def wrap_alloc(self, fn):
        census = self

        def run(*a, **k):
            if not census.on:
                return fn(*a, **k)
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                r = census.alloc[(census._caller(2), str(a[0]) if a else "")]
                r[0] += 1
                r[1] += time.perf_counter() - t0

        return run


def _agg_prof(prof: cProfile.Profile, nchunks: int) -> dict:
    st = pstats.Stats(prof)
    out = {}
    for (fn, line, name), (cc, nc, tt, ct, _callers) in st.stats.items():
        key = f"{os.path.basename(fn)}:{line}:{name}"
        out[key] = (nc / nchunks, tt * 1e6 / nchunks, ct * 1e6 / nchunks)
    return out


@mesh_parametrize(timeout_s=3 * 3600)
def test_lean_py(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tests.workloads import prompt_ids as _ids
    from models.demos.deepseek_v41_flash_d_p.tt import ccl, lean
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    monkeypatch.setenv("DS41_ENGRAM_TABLE", "resident")
    settings.thaw()
    layers = S.layers()[:4] if QUICK else S.layers()
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    setting_mode = "on" if lean.ON[0] else "legacy"
    inner = model._wait_job

    def wait_job(i, _f=inner):
        t = time.perf_counter()
        try:
            return _f(i)
        finally:
            _WAIT[0] += time.perf_counter() - t

    monkeypatch.setattr(model, "_wait_job", wait_job)
    census = Census()
    out: dict = {"sides": list(SIDES), "clean_rounds": CLEAN, "setting": setting_mode}
    for wl in WORKLOADS:
        tokens = _ids(wl)
        res = {s: [] for s in SIDES}
        diffs = defaultdict(list)
        ref: dict = {}
        t0 = time.time()
        for k, s in enumerate(SIDES):  # (warm: every chunk position on every side; the first side keeps the reference)
            _set_side(mesh_device, s)
            rows = _pass(model, mesh_device, layers, tokens, keep=ref if k == 0 else None, ref=ref if k else None)
            for c, r in enumerate(rows):
                if "diff" in r:
                    diffs[s].append(r["diff"])
        if CENSUS and wl == WORKLOADS[0]:
            _set_side(mesh_device, setting_mode)
            free0, alloc0 = ccl._DEALLOC, ttnn.allocate_tensor_on_device
            ccl._DEALLOC = [census.wrap_free(free0[0])]
            ttnn.allocate_tensor_on_device = census.wrap_alloc(alloc0)
            census.on = True
            try:
                _pass(model, mesh_device, layers, tokens)
            finally:
                census.on = False
                ccl._DEALLOC, ttnn.allocate_tensor_on_device = free0, alloc0
            n = CHUNKS  # (every chunk of the pass, chunk 0 included)
            frees = sorted(census.free.items(), key=lambda kv: -kv[1][1])
            allocs = sorted(census.alloc.items(), key=lambda kv: -kv[1][1])
            tot_f = sum(v[0] for v in census.free.values()) / n
            tot_fus = sum(v[1] for v in census.free.values()) * 1e6 / n
            tot_a = sum(v[0] for v in census.alloc.values()) / n
            tot_aus = sum(v[1] for v in census.alloc.values()) * 1e6 / n
            logger.info(
                f"P.66 census ({setting_mode}, a chunk): {tot_f:.0f} frees {tot_fus / 1e3:.2f} ms, "
                f"{tot_a:.0f} Python allocations {tot_aus / 1e3:.2f} ms"
            )
            for (who, spec_), (c, s_) in frees[:60]:
                logger.info(
                    f"P.66 census free {who} {spec_}: {c / n:.1f} a chunk, {s_ * 1e6 / max(c, 1):.1f} us a call"
                )
            for (who, spec_), (c, s_) in allocs[:30]:
                logger.info(
                    f"P.66 census alloc {who} {spec_}: {c / n:.1f} a chunk, {s_ * 1e6 / max(c, 1):.1f} us a call"
                )
            out["census"] = {
                "frees_chunk": tot_f,
                "free_us_chunk": tot_fus,
                "allocs_chunk": tot_a,
                "alloc_us_chunk": tot_aus,
                "free": [
                    {"who": w, "spec": sp, "calls": round(c / n, 2), "us_call": round(s_ * 1e6 / max(c, 1), 2)}
                    for (w, sp), (c, s_) in frees
                ],
                "alloc": [
                    {"who": w, "spec": sp, "calls": round(c / n, 2), "us_call": round(s_ * 1e6 / max(c, 1), 2)}
                    for (w, sp), (c, s_) in allocs
                ],
            }
        for _ in range(CLEAN):
            for s in SIDES:
                _set_side(mesh_device, s)
                res[s].append(_pass(model, mesh_device, layers, tokens, ref=ref))
                diffs[s].extend(r["diff"] for r in res[s][-1] if "diff" in r)
        ops_out = {}
        if OPS:  # (one pass a side: each ttnn op launch by its calling line, inclusive host us; launches a chunk)
            import ttnn.decorators as D

            orig = D.FastOperation.__call__
            for s in SIDES:
                rec = defaultdict(list)

                def call(self, *a, _rec=rec, **k):
                    f = sys._getframe(1)
                    key = f"{self.python_fully_qualified_name}@{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}"
                    t = time.perf_counter()
                    try:
                        return orig(self, *a, **k)
                    finally:
                        _rec[key].append(time.perf_counter() - t)

                _set_side(mesh_device, s)
                D.FastOperation.__call__ = call
                try:
                    _pass(model, mesh_device, layers, tokens)
                finally:
                    D.FastOperation.__call__ = orig
                n = CHUNKS
                ops_out[s] = {
                    "launches_chunk": sum(len(v) for v in rec.values()) / n,
                    "sites": {
                        k: {"calls": round(len(v) / n, 2), "us_median": round(statistics.median(v) * 1e6, 1)}
                        for k, v in rec.items()
                    },
                }
                logger.info(f"P.66 {wl} {s}: {ops_out[s]['launches_chunk']:.1f} ttnn op launches a chunk")
            if "legacy" in ops_out and "on" in ops_out:
                a, b = ops_out["legacy"]["sites"], ops_out["on"]["sites"]
                for k in sorted(set(a) | set(b)):
                    if any(w in k for w in OPS_WATCH):
                        x, y = a.get(k, {}), b.get(k, {})
                        logger.info(
                            f"P.66 {wl} op {k}: legacy {x.get('calls')} calls {x.get('us_median')} us, on {y.get('calls')} "
                            f"calls {y.get('us_median')} us"
                        )
        profs = {}
        if PROFILE:
            for s in SIDES:
                _set_side(mesh_device, s)
                p = cProfile.Profile()
                rows = _pass(model, mesh_device, layers, tokens, prof=p)
                profs[s] = {"host_ms": _mean(rows, "host_ms"), "funcs": _agg_prof(p, CHUNKS - 1)}
        logger.info(f"P.66 {wl}: measured in {time.time() - t0:.0f} s")

        summ: dict = {}
        for s in SIDES:
            cl = res[s]
            summ[s] = {
                k + "_ms": statistics.mean(_mean(r, k + "_ms") for r in cl) if cl else float("nan")
                for k in ("host", "busy", "wait", "wall")
            }
            summ[s]["busy_ms_rounds"] = [round(_mean(r, "busy_ms"), 2) for r in cl]
            summ[s]["wall_ms_rounds"] = [round(_mean(r, "wall_ms"), 2) for r in cl]
            summ[s]["max_diff"] = max(diffs[s]) if diffs[s] else 0.0
            if s in profs:
                summ[s]["cprofile_host_ms"] = profs[s]["host_ms"]
            tag = s
            for k in ("host", "busy", "wall"):
                metrics.record(f"p66_{k}_ms_{tag}_{wl}", round(summ[s][k + "_ms"], 2))
            metrics.record(f"p66_maxdiff_{tag}_{wl}", summ[s]["max_diff"])
            logger.info(
                f"P.66 {wl} {s}: host {summ[s]['host_ms']:.2f} ms, busy {summ[s]['busy_ms']:.2f} (rounds "
                f"{summ[s]['busy_ms_rounds']}), engram wait {summ[s]['wait_ms']:.2f}, wall {summ[s]['wall_ms']:.2f} "
                f"(rounds {summ[s]['wall_ms_rounds']}), max |diff| vs {SIDES[0]} {summ[s]['max_diff']}"
            )
        pairs_out = {}
        a, b = "legacy", "on"
        if a in summ and b in summ and res[a]:
            rec = {}
            for k in ("busy", "wall", "host"):
                pa = [_mean(r, k + "_ms") for r in res[a]]
                pb = [_mean(r, k + "_ms") for r in res[b]]
                d = [y - x for x, y in zip(pa, pb)]
                se = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
                rec[k] = {"d_ms": statistics.mean(d), "se_ms": se}
                metrics.record(f"p66_d_{k}_ms_{wl}", round(statistics.mean(d), 2))
            pairs_out["on-legacy"] = rec
            logger.info(
                f"P.66 {wl} on - legacy: busy {rec['busy']['d_ms']:+.2f} +- {rec['busy']['se_ms']:.2f} ms, "
                f"host {rec['host']['d_ms']:+.2f} +- {rec['host']['se_ms']:.2f}, wall {rec['wall']['d_ms']:+.2f} +- "
                f"{rec['wall']['se_ms']:.2f}"
            )
        prof_out = {}
        if "legacy" in profs and "on" in profs:
            fa, fb = profs["legacy"]["funcs"], profs["on"]["funcs"]
            keys = set(fa) | set(fb)
            pr = sorted(
                (
                    (k, fb.get(k, (0, 0, 0))[1] - fa.get(k, (0, 0, 0))[1], fa.get(k, (0,))[0], fb.get(k, (0,))[0])
                    for k in keys
                ),
                key=lambda x: -abs(x[1]),
            )
            prof_out["diff"] = [
                {"func": k, "d_tottime_us": round(d, 1), "calls_legacy": round(ca, 2), "calls_on": round(cb, 2)}
                for k, d, ca, cb in pr[:80]
            ]
            for r in prof_out["diff"][:40]:
                logger.info(
                    f"P.66 {wl} cprofile {r['func']}: {r['d_tottime_us']:+.1f} us a chunk (calls {r['calls_legacy']} -> "
                    f"{r['calls_on']})"
                )
        for s, p in profs.items():
            top = sorted(p["funcs"].items(), key=lambda kv: -kv[1][1])[:120]
            prof_out[s] = [{"func": k, "calls": round(v[0], 2), "tottime_us": round(v[1], 1)} for k, v in top]
            for k, v in top[:50]:
                logger.info(f"P.66 {wl} {s} cprofile top {k}: {v[1]:.1f} us a chunk, {v[0]:.1f} calls")
        out[wl] = {"sides": summ, "pairs": pairs_out, "cprofile": prof_out, "ops": ops_out}
        for t in ref.values():
            ttnn.deallocate(t)
    metrics.record_json("lean_py", out)
    for wl in WORKLOADS:
        for s in SIDES:
            assert out[wl]["sides"][s]["max_diff"] == 0, f"{wl} {s}: outputs differ from {SIDES[0]}"
