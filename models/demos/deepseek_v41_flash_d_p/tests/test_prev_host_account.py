# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.62 (owner 2026-11-14, diagnostic: no shipped behaviour): why MOE_REPLICAS=prev costs host time over off, call by
call. Accounts the host time of a chunk on both sides until the difference is explained.

Terms:
  host_ms   = first host call of a chunk to the moment the last layer's ops are enqueued (before the closing sync), as
              tests/test_moe_prev_replicas.py (P.60).
  wall_ms   = to the end of the sync after the last layer.
  side      = off (prev switched off in the loaded model: no planner, the stock combine fill, the empty map; P.56's
              `_side`) or prev:<host2>[:<planner>] (MOE_REPLICAS=prev with MOE_PREV_HOST=device, MOE_PREV_HOST2=<host2>
              and, P.62b, MOE_PLANNER=<planner> (default: the setting's), switched in the loaded model:
              expert_prev.set_host2 / set_planner).
  excl      = a wrapped call's exclusive host time (nested wrapped calls charged to the inner one): every ttnn op
              (FastOperation.__call__, C++ and Python forks apart), ttnn.bringup.run_prepared, PrevReplicas.before /
              after (their own Python), transfers and the engram worker's wait (P.57's HostClock).
  python    = a run_block section's host time minus the wrapped calls in it (the model's own Python).
  cProfile  = tottime per function (inflated by the profiler; differences between sides only).
Per workload (novel, code) and side: chunks 0..10 of fresh requests back to back (chunk 0 runs off on every side).
Passes, sides alternating in each round: warm, P62_CLEAN clean rounds (host_ms / wall_ms per chunk), P62_INSTR
instrumented rounds (excl per name and python per section, per chunk), one cProfile round. Means over chunks 1..10.

Output: results/<task>_prev_host_account.json (record_json) and the log: per side the host means, per name the
difference prev - off in us a chunk with calls a chunk, the python sections' differences, the cProfile top differences,
and the residual = clean host difference - the sum of the instrumented differences.

    P62_SIDES=off,prev:legacy,prev:lean PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_prev_host_account.py
P62_QUICK=1: 4 layers, chunks 0..3, novel (iterating only).
"""

from __future__ import annotations

import cProfile
import os
import pstats
import statistics
import time
from collections import defaultdict

from loguru import logger

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import begin_chunk, mesh_parametrize, spec
from models.demos.common.bringup.testing.positions import _free

S = spec()
QUICK = os.environ.get("P62_QUICK") == "1"
SEQ = 56320
CHUNKS = 4 if QUICK else 11
WORKLOADS = ("novel",) if QUICK else tuple(os.environ.get("P62_WL", "novel,code").split(","))
SIDES = tuple(os.environ.get("P62_SIDES", "off,prev:legacy").split(","))
CLEAN = int(os.environ.get("P62_CLEAN", "10"))
INSTR = int(os.environ.get("P62_INSTR", "2"))
PROFILE = os.environ.get("P62_CPROFILE", "1") == "1"
_WAIT = [0.0]  # (the engram worker's waits inside the current chunk, s: every pass, two perf_counter calls a wait)


def _set_side(mesh, side: str) -> None:
    from models.demos.deepseek_v41_flash_d_p.tests.test_moe_prev_replicas import _side
    from models.demos.deepseek_v41_flash_d_p.tt import expert_prev as XP

    if side == "off":
        _side(mesh, "off")
        return
    # (prev:<host2>[:<planner>]; P.62b: _side's mode sets MOE_PREV_HOST2 itself - "device" meant host2 legacy, so
    # P.62's prev:lean side ran P.60's device path; the mode is now lean for host2 lean)
    parts = side.split(":")
    host2 = parts[1]
    XP.set_planner(parts[2] if len(parts) > 2 else None)
    _side(mesh, "prev", "lean" if host2 == "lean" else "device")
    assert XP.host2() == host2, (XP.host2(), host2)


def _chunk(model, layers, tokens, st, state, clock=None):
    if clock is not None:
        clock.set_section("begin_chunk")
        clock.in_bc += 1
        clock.enter("begin_chunk", "begin_chunk")
    try:
        begin_chunk(model, tokens, st)
    finally:
        if clock is not None:
            clock.leave()
            clock.in_bc -= 1
    if clock is not None:
        clock.set_section("embed")
    h = model.embed(tokens)
    for i in layers:
        if clock is not None:
            clock.set_section("layer_glue")
        h2 = model.layer(i, h, st, state)
        model.free(h)
        h = h2
    if clock is not None:
        clock.set_section("layer_glue")
    return h


def _pass(model, mesh, layers, tokens, clock=None, prof=None) -> list:
    """Chunks 0..CHUNKS-1 of a fresh request. -> per chunk {host_ms, wall_ms[, clock snapshot]}."""
    chunk = int(S.get("target.chunk"))
    state = model.new_state(SEQ)
    rows = []
    try:
        for c in range(CHUNKS):
            st = c * chunk
            t = tokens[st : st + chunk]
            ttnn.synchronize_device(mesh)
            on = c >= 1
            if clock is not None and on:
                clock.reset()
                clock.on = True
            if prof is not None and on:
                prof.enable()
            _WAIT[0] = 0.0
            t0 = time.perf_counter()
            h = _chunk(model, layers, t, st, state, clock if (clock is not None and on) else None)
            t1 = time.perf_counter()
            if prof is not None and on:
                prof.disable()
            if clock is not None and on:
                clock.set_section(None)
                clock.on = False
            ttnn.synchronize_device(mesh)
            t2 = time.perf_counter()
            model.free(h)
            row = {"host_ms": (t1 - t0) * 1e3, "wall_ms": (t2 - t0) * 1e3, "wait_ms": _WAIT[0] * 1e3}
            row["busy_ms"] = row["host_ms"] - row["wait_ms"]
            if clock is not None and on:
                row["by_name"] = {f"{cat}:{n}": (r[0], r[1]) for (cat, n), r in clock.by_name.items()}
                row["py"] = {k: v - clock.sec_wrapped.get(k, 0.0) for k, v in clock.sec_time.items()}
            rows.append(row)
    finally:
        _free(state)
    return rows


def _mean(rows, k):
    return statistics.mean(r[k] for r in rows[1:])


def _agg_instr(passes: list) -> dict:
    """Instrumented passes -> {name: (calls a chunk, us a chunk)} and {section: python us a chunk}."""
    names = defaultdict(lambda: [0.0, 0.0])
    py = defaultdict(float)
    host = []
    n = 0
    for rows in passes:
        for r in rows[1:]:
            n += 1
            host.append(r["host_ms"])
            for k, (calls, s) in r["by_name"].items():
                names[k][0] += calls
                names[k][1] += s
            for k, s in r["py"].items():
                py[k] += s
    return {
        "host_ms": statistics.mean(host),
        "names": {k: (v[0] / n, v[1] * 1e6 / n) for k, v in names.items()},
        "py": {k: v * 1e6 / n for k, v in py.items()},
    }


def _agg_prof(prof: cProfile.Profile, nchunks: int) -> dict:
    st = pstats.Stats(prof)
    out = {}
    for (fn, line, name), (cc, nc, tt, ct, _callers) in st.stats.items():
        key = f"{os.path.basename(fn)}:{line}:{name}"
        out[key] = (nc / nchunks, tt * 1e6 / nchunks, ct * 1e6 / nchunks)
    return out


@mesh_parametrize(timeout_s=3 * 3600)
def test_prev_host_account(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tests.test_host_bound import HostClock, _install
    from models.demos.deepseek_v41_flash_d_p.tests.test_moe_prev_replicas import _ids
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    monkeypatch.setenv("DS41_MOE_REPLICAS", "prev")
    monkeypatch.setenv("DS41_MOE_FILL", "piggyback")
    monkeypatch.setenv("DS41_MOE_PREV_HOST", "device")
    monkeypatch.setenv("DS41_ENGRAM_TABLE", "resident")
    monkeypatch.setenv("DS41_MOE_COUNTS", "on")
    monkeypatch.delenv("DS41_MOE_COUNTS_DIR", raising=False)
    settings.thaw()
    layers = S.layers()[:4] if QUICK else S.layers()
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    clock = HostClock()
    _install(monkeypatch, clock, model)
    inner = model._wait_job

    def wait_job(i, _f=inner):
        t = time.perf_counter()
        try:
            return _f(i)
        finally:
            _WAIT[0] += time.perf_counter() - t

    monkeypatch.setattr(model, "_wait_job", wait_job)
    out: dict = {"sides": list(SIDES), "clean_rounds": CLEAN, "instr_rounds": INSTR}
    for wl in WORKLOADS:
        tokens = _ids(wl)
        res = {s: {"clean": [], "instr": []} for s in SIDES}
        t0 = time.time()
        for s in SIDES:  # (warm: every chunk position, both combine modes, the planner)
            _set_side(mesh_device, s)
            _pass(model, mesh_device, layers, tokens)
        for _ in range(CLEAN):
            for s in SIDES:
                _set_side(mesh_device, s)
                res[s]["clean"].append(_pass(model, mesh_device, layers, tokens))
        for _ in range(INSTR):
            for s in SIDES:
                _set_side(mesh_device, s)
                res[s]["instr"].append(_pass(model, mesh_device, layers, tokens, clock=clock))
        profs = {}
        if PROFILE:
            for s in SIDES:
                _set_side(mesh_device, s)
                p = cProfile.Profile()
                rows = _pass(model, mesh_device, layers, tokens, prof=p)
                profs[s] = {"host_ms": _mean(rows, "host_ms"), "funcs": _agg_prof(p, CHUNKS - 1)}
        logger.info(f"P.62 {wl}: measured in {time.time() - t0:.0f} s")

        summ: dict = {}
        for s in SIDES:
            cl = res[s]["clean"]
            per_round = [_mean(r, "host_ms") for r in cl]
            busy = [_mean(r, "busy_ms") for r in cl]
            wait = [_mean(r, "wait_ms") for r in cl]
            summ[s] = {
                "busy_ms": statistics.mean(busy),
                "wait_ms": statistics.mean(wait),
                "busy_ms_rounds": [round(x, 2) for x in busy],
                "host_ms": statistics.mean(per_round),
                "host_ms_rounds": [round(x, 2) for x in per_round],
                "wall_ms": statistics.mean(_mean(r, "wall_ms") for r in cl),
                "host_ms_chunks": [round(statistics.mean(r[c]["host_ms"] for r in cl), 2) for c in range(CHUNKS)],
                "wall_ms_chunks": [round(statistics.mean(r[c]["wall_ms"] for r in cl), 2) for c in range(CHUNKS)],
                "instr": _agg_instr(res[s]["instr"]),
            }
            if s in profs:
                summ[s]["cprofile_host_ms"] = profs[s]["host_ms"]
            logger.info(
                f"P.62 {wl} {s}: host {summ[s]['host_ms']:.2f} ms (rounds {summ[s]['host_ms_rounds']}), busy "
                f"{summ[s]['busy_ms']:.2f} (rounds {summ[s]['busy_ms_rounds']}), engram wait {summ[s]['wait_ms']:.2f}, "
                f"wall {summ[s]['wall_ms']:.2f}, instrumented host {summ[s]['instr']['host_ms']:.2f}"
            )
            metrics.record(f"p62_busy_ms_{s.replace(':', '_')}_{wl}", round(summ[s]["busy_ms"], 2))
            metrics.record(f"p62_wait_ms_{s.replace(':', '_')}_{wl}", round(summ[s]["wait_ms"], 2))
            metrics.record(f"p62_host_ms_{s.replace(':', '_')}_{wl}", round(summ[s]["host_ms"], 2))
            metrics.record(f"p62_wall_ms_{s.replace(':', '_')}_{wl}", round(summ[s]["wall_ms"], 2))
        base = SIDES[0]
        diffs: dict = {}
        for s in SIDES[1:]:
            a, b = summ[base]["instr"], summ[s]["instr"]
            names = set(a["names"]) | set(b["names"])
            rows = []
            for k in names:
                ca, ua = a["names"].get(k, (0.0, 0.0))
                cb, ub = b["names"].get(k, (0.0, 0.0))
                rows.append((k, ub - ua, ca, cb, ua, ub))
            rows.sort(key=lambda x: -abs(x[1]))
            secs = set(a["py"]) | set(b["py"])
            pyd = sorted(((k, b["py"].get(k, 0.0) - a["py"].get(k, 0.0)) for k in secs), key=lambda x: -abs(x[1]))
            d_clean = (summ[s]["host_ms"] - summ[base]["host_ms"]) * 1e3
            pairs = [x - y for x, y in zip(summ[s]["busy_ms_rounds"], summ[base]["busy_ms_rounds"])]
            d_busy = statistics.mean(pairs) * 1e3
            se_busy = (statistics.stdev(pairs) / len(pairs) ** 0.5 * 1e3) if len(pairs) > 1 else float("nan")
            d_wait = (summ[s]["wait_ms"] - summ[base]["wait_ms"]) * 1e3
            d_instr = (b["host_ms"] - a["host_ms"]) * 1e3
            d_names = sum(r[1] for r in rows)
            d_py = sum(v for _, v in pyd)
            rec = {
                "d_host_clean_us": d_clean,
                "d_busy_clean_us": d_busy,
                "se_busy_clean_us": se_busy,
                "d_wait_clean_us": d_wait,
                "d_host_instr_us": d_instr,
                "d_wrapped_us": d_names,
                "d_python_us": d_py,
                "residual_instr_us": d_instr - d_names - d_py,
                "names": [
                    {
                        "name": k,
                        "d_us": round(d, 1),
                        "calls_base": round(ca, 2),
                        "calls": round(cb, 2),
                        "us_base": round(ua, 1),
                        "us": round(ub, 1),
                    }
                    for k, d, ca, cb, ua, ub in rows[:80]
                ],
                "python": [{"section": k, "d_us": round(v, 1)} for k, v in pyd[:40]],
            }
            if base in profs and s in profs:
                fa, fb = profs[base]["funcs"], profs[s]["funcs"]
                keys = set(fa) | set(fb)
                pr = sorted(
                    (
                        (
                            k,
                            fb.get(k, (0, 0, 0))[1] - fa.get(k, (0, 0, 0))[1],
                            fa.get(k, (0, 0, 0))[0],
                            fb.get(k, (0, 0, 0))[0],
                        )
                        for k in keys
                    ),
                    key=lambda x: -abs(x[1]),
                )
                rec["cprofile"] = [
                    {"func": k, "d_tottime_us": round(d, 1), "calls_base": round(ca, 2), "calls": round(cb, 2)}
                    for k, d, ca, cb in pr[:60]
                ]
                rec["d_host_cprofile_us"] = (profs[s]["host_ms"] - profs[base]["host_ms"]) * 1e3
            diffs[s] = rec
            logger.info(
                f"P.62 {wl} {s} - {base}: clean host {d_clean:+.0f} us a chunk = busy {d_busy:+.0f} (+-{se_busy:.0f} "
                f"se) + engram wait {d_wait:+.0f}; instrumented {d_instr:+.0f} = wrapped "
                f"{d_names:+.0f} + python {d_py:+.0f} + residual {rec['residual_instr_us']:+.0f}"
            )
            for r in rec["names"][:30]:
                logger.info(
                    f"P.62 {wl} {s}   {r['name']}: {r['d_us']:+.1f} us a chunk (calls {r['calls_base']} -> "
                    f"{r['calls']}, {r['us_base']:.1f} -> {r['us']:.1f})"
                )
            for r in rec["python"][:15]:
                logger.info(f"P.62 {wl} {s}   python {r['section']}: {r['d_us']:+.1f} us a chunk")
            for r in rec.get("cprofile", [])[:30]:
                logger.info(
                    f"P.62 {wl} {s}   cprofile {r['func']}: {r['d_tottime_us']:+.1f} us a chunk (calls "
                    f"{r['calls_base']} -> {r['calls']})"
                )
        for s in SIDES:  # (the full per-name tables are big: keep the top of each side)
            ins = summ[s]["instr"]
            ins["names"] = dict(sorted(ins["names"].items(), key=lambda x: -x[1][1])[:120])
        out[wl] = {"sides": summ, "diffs": diffs}
    metrics.record_json("prev_host_account", out)
