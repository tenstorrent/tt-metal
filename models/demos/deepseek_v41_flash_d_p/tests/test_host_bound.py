# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.57 (owner 2026-11-14, EXPERIMENTAL, analysis only: no shipped change): where the host time of a chunk goes with
P.56 on (MOE_REPLICAS=prev, MOE_FILL=piggyback) and off, and a one-layer Metal trace prototype.

Terms:
  host_ms   = the time the Python thread needs to issue a chunk: first host call to the moment the last op is enqueued
              (before the closing sync). It includes every wait inside the chunk.
  wall_ms   = first host call to the end of the closing sync (the chunk is done on every chip).
  drain_ms  = wall - host: what the device still runs after the host has issued the last op.
  device_ms = the program real-time profiler (no device profiler, no op mode): each program's start-to-end on its
              slowest chip, summed over the chunk's programs (as P.56). device_span_ms / device_idle_ms: per chip, first
              program start to last program end, and that span minus the chip's summed program time (the device waiting
              for the host), the mean over chips.
  parts of host_ms (an instrumented pass; every wrapped call charged its exclusive time, nested calls to the inner):
    python      (a) Python in the model code: a section's host time minus the wrapped calls in it; sections are
                    run_block's steps (signposts), aggregated over layers (host_ms_py_<step> in the side record);
    op_dispatch (b) outermost ttnn op calls (ttnn.decorators.FastOperation: C++ enqueue, program cache lookup,
                    runtime-arg updates; also the Python of ttnn.bringup Python-registered forks, listed apart);
    prepared    (c) ttnn.bringup.run_prepared (prepared programs, address patching) outside begin_chunk;
    transfer    (d) host <-> device transfers and waits inside the layers (from_torch / to_torch / sync / events, the
                    engram worker join);
    per_chunk   (e) begin_chunk (engram hash + lookup + upload, request bookkeeping), the embedding's id upload, P.56's
                    host guard and planner dispatch wrapper (PrevReplicas.before / after, their run_prepared in (c));
    other       (f) host_ms minus (a..e): the test's loop and anything not wrapped (instrumentation overhead is reported
                    apart: host_ms instrumented - host_ms clean).
  op_calls_chunk = outermost ttnn op calls + run_prepared calls a chunk; programs_chunk = device programs (rt records).

Where (one process, one model load, the shipped settings with MOE_REPLICAS=prev MOE_FILL=piggyback
ENGRAM_TABLE=resident; side off = prev switched off in the loaded model, as tests/test_moe_prev_replicas.py):
  novel10 = chunk 10 (start 51,200) of the ladder's novel prompt after the device's own prefill of chunks 0..9
            (fresh request, back to back; one warm pass, one profiled pass, one clean pass, one instrumented pass);
  p51200  = one warm chunk at position 51,200 (random ids, fresh state, steady_state as the profile; warm until no new
            program, then the clean runs, one profiled run, one instrumented run, and the framework's host profile
            testing/profile.py:host_profile under cProfile, BRINGUP_PROFILE_HOST's).
Trace prototype (side off): layer TRACE_LAYER at the novel chunk 10 (its real state), input cloned from the layer's
real input; untraced runs (each: clone of the input + the layer) until the program cache stops growing, then one
capture of the same and replays. trace_host_ms_layer = execute_trace's host call; trace_device_ms_layer = REPLAYS
replays back to back / REPLAYS (host negligible); untraced_host_ms_layer = issue time of clone + layer,
untraced_device_ms_layer = the real-time profiler's program sum of one untraced run; trace_bit_identical = every
output tensor of the replays equal byte for byte to the untraced output (which must itself repeat bit for bit).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_host_bound.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from collections import defaultdict

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import begin_chunk, spec
from models.demos.common.bringup.testing.positions import _free

S = spec()
SEQ = 56320
CHUNKS = 11  # (novel chunks 0..10; chunk 10 starts at 51,200)
POS = 51200
CLEAN_RUNS = 5
MAX_WARM = 30
TRACE_LAYER = 21  # (dec_reuse, MoE, no engram: the most common layer type)
REPLAYS = 10
TRACE_REGION = 256 * 1024 * 1024
TRANSFER = {
    "from_torch",
    "to_torch",
    "to_device",
    "from_device",
    "synchronize_device",
    "copy_host_to_device_tensor",
    "copy_device_to_host_tensor",
    "record_event",
    "wait_for_event",
    "event_synchronize",
}


# ============================================================================ host clock
class HostClock:
    """Exclusive host time of wrapped calls by category and name, and Python time by section (signposts)."""

    def __init__(self):
        self.on = False
        self.stack: list = []
        self.reset()

    def reset(self):
        self.cat = defaultdict(float)  # (section-group, category) -> s
        self.by_name = defaultdict(lambda: [0, 0.0, 0.0])  # (category, name) -> calls, exclusive s, inclusive s
        self.sec_time = defaultdict(float)  # section -> elapsed s
        self.sec_wrapped = defaultdict(float)  # section -> wrapped exclusive s
        self.section, self.sec_t0 = None, None
        self.in_bc = 0

    def group(self):
        return "per_chunk" if self.in_bc or self.section in ("begin_chunk", "embed") else "layers"

    def set_section(self, name):
        if not self.on:
            return
        t = time.perf_counter()
        if self.section is not None:
            self.sec_time[self.section] += t - self.sec_t0
        self.section, self.sec_t0 = name, t

    def enter(self, cat, name):
        self.stack.append([cat, name, time.perf_counter(), 0.0])

    def leave(self):
        cat, name, t0, child = self.stack.pop()
        dt = time.perf_counter() - t0
        ex = dt - child
        if self.stack:
            self.stack[-1][3] += dt
        r = self.by_name[(cat, name)]
        r[0] += 1
        r[1] += ex
        r[2] += dt
        g = self.group()
        self.cat[(g, cat)] += ex
        if self.section is not None:
            self.sec_wrapped[self.section] += ex

    def wrap(self, fn, cat, name=None):
        clock = self

        def run(*a, **k):
            if not clock.on:
                return fn(*a, **k)
            nm = name
            if nm is None:  # (run_prepared: the calling module and function)
                f = sys._getframe(1)
                nm = f"{os.path.basename(f.f_code.co_filename)}:{f.f_code.co_name}"
            clock.enter(cat, nm)
            try:
                return fn(*a, **k)
            finally:
                clock.leave()

        return run


def _install(monkeypatch, clock: HostClock, model):
    import ttnn.decorators as D

    from models.demos.deepseek_v41_flash_d_p.tt import expert_prev as XP

    orig = D.FastOperation.__call__

    def call(self, *args, **kwargs):
        if not clock.on:
            return orig(self, *args, **kwargs)
        name = self.python_fully_qualified_name
        name = name[5:] if name.startswith("ttnn.") else name
        if name.split(".")[-1] in TRANSFER:
            cat = "transfer"
        elif self.is_cpp_operation:
            cat = "op"
        else:
            cat = "op_py"  # (a Python-registered op: its own Python; ops it calls are charged to them)
        clock.enter(cat, name)
        try:
            return orig(self, *args, **kwargs)
        finally:
            clock.leave()

    monkeypatch.setattr(D.FastOperation, "__call__", call)
    monkeypatch.setattr(ttnn.bringup, "run_prepared", clock.wrap(ttnn.bringup.run_prepared, "prepared"))
    for n in TRANSFER:
        obj = getattr(ttnn, n, None)
        if obj is not None and not isinstance(obj, D.FastOperation):
            monkeypatch.setattr(ttnn, n, clock.wrap(obj, "transfer", n))
    monkeypatch.setattr(XP.PrevReplicas, "before", clock.wrap(XP.PrevReplicas.before, "guard", "prev.before"))
    monkeypatch.setattr(XP.PrevReplicas, "after", clock.wrap(XP.PrevReplicas.after, "guard", "prev.after"))
    monkeypatch.setattr(model, "_wait_job", clock.wrap(model._wait_job, "transfer", "engram_wait"))
    orig_sp = profiler.signpost

    def signpost(name):
        clock.set_section(name)
        orig_sp(name)

    monkeypatch.setattr(profiler, "signpost", signpost)


# ============================================================================ helpers
def _main():
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as A

    return {m.layer: m for m in A._MAIN_LAYERS}


def _side(mesh, side: str) -> None:
    from models.demos.deepseek_v41_flash_d_p.tests.test_moe_prev_replicas import _side as p56_side

    p56_side(mesh, side)


def _ids_novel() -> torch.Tensor:
    from models.demos.deepseek_v41_flash_d_p.tests.test_moe_counts import _ids

    return torch.tensor(_ids("novel"))


def _rt_device(mesh) -> dict:
    """After profiler.finish_rt(): device ms (programs' slowest chip, summed), per-chip span / busy / idle, programs."""
    recs, d2c = profiler._P["rt_records"], profiler._P["dev_to_chip"]
    total = sum(max(e[0] for e in chips.values()) / 1e6 for chips in recs.values() if chips)
    per = defaultdict(lambda: [0.0, float("inf"), 0.0])
    for chips in recs.values():
        for d, e in chips.items():
            c = per[d2c.get(int(d), int(d))]
            c[0] += e[0]
            c[1] = min(c[1], e[1])
            c[2] = max(c[2], e[2])
    spans = [(c[2] - c[1]) / 1e6 for c in per.values()]
    busy = [c[0] / 1e6 for c in per.values()]
    return {
        "device_ms": total,
        "span_ms": statistics.mean(spans) if spans else 0.0,
        "busy_ms": statistics.mean(busy) if busy else 0.0,
        "idle_ms": statistics.mean(s - b for s, b in zip(spans, busy)) if spans else 0.0,
        "programs": len(recs),
    }


def _chunk(model, layers, tokens, st, state, clock=None):
    """One chunk, no sync. Returns the last output (freed by the caller)."""
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
    if layers[-1] == S.num_layers - 1:
        h2 = model.final_norm(h)
        model.free(h)
        h = h2
    return h


def _timed(model, mesh, layers, tokens, st, state, clock=None) -> dict:
    ttnn.synchronize_device(mesh)
    if clock is not None:
        clock.reset()
        clock.on = True
    t0 = time.perf_counter()
    h = _chunk(model, layers, tokens, st, state, clock)
    model.free(h)
    t1 = time.perf_counter()
    if clock is not None:
        clock.set_section(None)
        clock.on = False
    ttnn.synchronize_device(mesh)
    t2 = time.perf_counter()
    return {"host_ms": (t1 - t0) * 1e3, "wall_ms": (t2 - t0) * 1e3}


def _profiled(model, mesh, layers, tokens, st, state) -> dict:
    profiler.enable(mesh, rt=True)
    try:
        t0 = time.perf_counter()
        h = _chunk(model, layers, tokens, st, state)
        model.free(h)
        host = (time.perf_counter() - t0) * 1e3
        profiler.signpost("end")
        profiler.finish_rt()
        out = _rt_device(mesh)
    finally:
        profiler.disable()
    out["host_ms_profiled"] = host
    return out


def _parts(clock: HostClock, host_ms: float) -> dict:
    """The breakdown of one instrumented chunk (ms)."""
    c = {k: v * 1e3 for k, v in clock.cat.items()}
    L = lambda cat: c.get(("layers", cat), 0.0)  # noqa: E731
    layer_secs = {k: v for k, v in clock.sec_time.items() if k not in ("begin_chunk", "embed")}
    python = sum((v - clock.sec_wrapped.get(k, 0.0)) * 1e3 for k, v in layer_secs.items())
    bc_ms = sum(r[2] for (cat, n), r in clock.by_name.items() if cat == "begin_chunk") * 1e3
    embed_ms = clock.sec_time.get("embed", 0.0) * 1e3
    parts = {
        "python": python,
        "op_dispatch": L("op") + L("op_py"),
        "op_dispatch_cpp": L("op"),
        "op_dispatch_pyfork": L("op_py"),
        "prepared": L("prepared"),
        "transfer": L("transfer"),
        "per_chunk": bc_ms + embed_ms + L("guard"),
        "begin_chunk": bc_ms,
        "embed": embed_ms,
        "guard": L("guard"),
    }
    parts["other"] = host_ms - sum(parts[k] for k in ("python", "op_dispatch", "prepared", "transfer", "per_chunk"))
    py_by_sec = {
        k: round((v - clock.sec_wrapped.get(k, 0.0)) * 1e3, 3)
        for k, v in sorted(layer_secs.items(), key=lambda x: -x[1])
    }
    ops = {}
    for (cat, n), (k, ex, inc) in sorted(clock.by_name.items(), key=lambda x: -x[1][1]):
        ops[f"{cat}:{n}"] = {"calls": k, "excl_ms": round(ex * 1e3, 3), "us_per_call": round(ex * 1e6 / max(k, 1), 1)}
    calls = sum(r[0] for (cat, n), r in clock.by_name.items() if cat in ("op", "op_py", "prepared", "transfer"))
    top_level = sum(r[0] for (cat, n), r in clock.by_name.items() if cat in ("op", "op_py", "prepared"))
    secs = {  # (host ms a section, all of it: Python + its ops; sections are run_block's steps over all layers)
        k: round(v * 1e3, 3) for k, v in sorted(clock.sec_time.items(), key=lambda x: -x[1])
    }
    return {
        "parts": parts,
        "python_by_section": py_by_sec,
        "host_by_section": secs,
        "ops": ops,
        "calls_all": calls,
        "op_calls": top_level,
    }


# ============================================================================ the measurements
def _novel(model, mesh, layers, tokens, clock, keep_input=None) -> dict:
    """Chunks 0..10 of a fresh request, four passes: warm, profiled (chunk 10), clean (every chunk), instrumented.
    keep_input: {layer: None} -> the layer's input of chunk 10 cloned into it (trace prototype), state kept."""
    chunk = int(S.get("target.chunk"))
    out = {}
    for kind in ("warm", "profiled", "clean", "instrumented"):
        state = model.new_state(SEQ)
        try:
            rows = []
            for c in range(CHUNKS):
                st = c * chunk
                t = tokens[st : st + chunk]
                last = c == CHUNKS - 1
                if kind == "profiled" and last:
                    rows.append(_profiled(model, mesh, layers, t, st, state))
                elif kind in ("clean", "instrumented"):
                    rows.append(_timed(model, mesh, layers, t, st, state, clock if kind == "instrumented" else None))
                    if kind == "instrumented":
                        rows[-1].update(_parts(clock, rows[-1]["host_ms"]))
                else:
                    h = _chunk(model, layers, t, st, state)
                    model.free(h)
                    ttnn.synchronize_device(mesh)
            out[kind] = rows
        finally:
            _free(state)
    return out


def _position(model, mesh, layers, clock, side) -> dict:
    """One warm chunk at POS (random ids, fresh state), as testing/profile.py at BRINGUP_POS."""
    from models.demos.common.bringup.testing import profile as P

    chunk = int(S.get("target.chunk"))
    vocab = int(S.get("checkpoint.config.vocab_size", 32000) or 32000)
    tokens = torch.randint(0, vocab, (POS + chunk,), generator=torch.Generator().manual_seed(0))[POS:]
    state = model.new_state(POS + chunk)
    out: dict = {}
    try:
        h = _chunk(model, layers, tokens, POS, state)  # (compile)
        model.free(h)
        ttnn.synchronize_device(mesh)
        walls, n, quiet = [], P.cache_entries(mesh), 0
        for _ in range(MAX_WARM):  # (F196 / F200: no new program 2 runs in a row, then the walls settled)
            model.steady_state(POS)
            walls.append(_timed(model, mesh, layers, tokens, POS, state)["wall_ms"])
            m = P.cache_entries(mesh)
            quiet = quiet + 1 if m == n else 0
            n = m
            if quiet >= 2 and (len(walls) >= MAX_WARM // 2 or P.settled(walls)):
                break
        out["warm_walls"] = [round(w, 1) for w in walls]
        runs = []
        for _ in range(CLEAN_RUNS):
            model.steady_state(POS)
            runs.append(_timed(model, mesh, layers, tokens, POS, state))
        out["new_programs_clean"] = (P.cache_entries(mesh) or 0) - (n or 0)
        model.steady_state(POS)
        out["profiled"] = _profiled(model, mesh, layers, tokens, POS, state)
        model.steady_state(POS)
        ins = _timed(model, mesh, layers, tokens, POS, state, clock)
        ins.update(_parts(clock, ins["host_ms"]))
        out["instrumented"] = ins

        def one_run():
            model.steady_state(POS)
            return round(_timed(model, mesh, layers, tokens, POS, state)["wall_ms"], 1)

        os.environ[metrics.SUFFIX_ENV] = f"_{side}_p{POS}"
        try:
            P.host_profile(one_run)  # (F230: the framework's host profile, cProfile, HOST_RUNS runs)
        except Exception as e:  # (a finding, not a failed measurement)
            out["host_profile_error"] = f"{type(e).__name__}: {e}"
            logger.exception("P.57 host profile failed")
        finally:
            os.environ.pop(metrics.SUFFIX_ENV, None)
        out["clean"] = runs
    finally:
        _free(state)
    return out


def _clone_h(h):
    from models.demos.deepseek_v41_flash_d_p.tt.model import DevH

    c = lambda t: ttnn.clone(t, memory_config=t.memory_config())  # noqa: E731
    stats = None
    if h.stats is not None:
        row = h.stats[1]
        stats = (h.stats[0], tuple(c(t) for t in row) if isinstance(row, tuple) else c(row))
    return DevH(c(h.streams), c(h.pre), stats)


def _host_bytes(h) -> list:
    return [
        ttnn.to_torch(d).contiguous().view(torch.uint8).clone() for t in h.tensors() for d in ttnn.get_device_tensors(t)
    ]


def _trace(model, mesh, layers, tokens) -> dict:
    """Side off. The layer's real input of novel chunk 10, then one-layer untraced runs, a capture and replays."""
    from models.demos.common.bringup.testing import profile as P

    chunk = int(S.get("target.chunk"))
    L = TRACE_LAYER
    st = (CHUNKS - 1) * chunk
    state = model.new_state(SEQ)
    res: dict = {"layer": L, "start": st}
    master = None
    try:
        for c in range(CHUNKS):
            s0 = c * chunk
            t = tokens[s0 : s0 + chunk]
            begin_chunk(model, t, s0)
            h = model.embed(t)
            for i in layers:
                if c == CHUNKS - 1 and i == L:
                    master = _clone_h(h)
                h2 = model.layer(i, h, s0, state)
                model.free(h)
                h = h2
            if layers[-1] == S.num_layers - 1:
                h2 = model.final_norm(h)
                model.free(h)
                h = h2
            model.free(h)
            ttnn.synchronize_device(mesh)

        def one():
            if state.memo is not None:
                state.memo.clear()
            x = _clone_h(master)
            out = model.layer(L, x, st, state)
            model.free(x)
            return out

        # untraced: warm until the program cache stops growing (2 runs in a row), then timed
        n, quiet, k = P.cache_entries(mesh), 0, 0
        while quiet < 2 and k < 40:
            o = one()
            ttnn.synchronize_device(mesh)
            k += 1
            m = P.cache_entries(mesh)
            quiet = quiet + 1 if m == n else 0
            n = m
            if quiet < 2 and k < 40:
                model.free(o)
        ref = _host_bytes(o)  # (the last warm run's output)
        model.free(o)
        res["untraced_warm_runs"] = k
        hosts, walls = [], []
        same = True
        for _ in range(REPLAYS):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            o = one()
            t1 = time.perf_counter()
            ttnn.synchronize_device(mesh)
            t2 = time.perf_counter()
            hosts.append((t1 - t0) * 1e3)
            walls.append((t2 - t0) * 1e3)
            got = _host_bytes(o)
            same = same and len(got) == len(ref) and all(torch.equal(a, b) for a, b in zip(got, ref))
            model.free(o)
        res["untraced_host_ms_layer"] = statistics.median(hosts)
        res["untraced_wall_ms_layer"] = statistics.median(walls)
        res["untraced_repeat_identical"] = int(same)
        profiler.enable(mesh, rt=True)
        try:
            o = one()
            profiler.signpost("end")
            profiler.finish_rt()
            dev = _rt_device(mesh)
        finally:
            profiler.disable()
        model.free(o)
        res["untraced_device_ms_layer"] = dev["device_ms"]
        res["untraced_device_idle_ms_layer"] = dev["idle_ms"]
        res["programs_layer"] = dev["programs"]

        # capture: the same clone + layer, then replays
        if state.memo is not None:
            state.memo.clear()
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            x = _clone_h(master)
            traced_out = model.layer(L, x, st, state)
            model.free(x)
        except Exception as e:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            res["trace_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:400]}"
            logger.error(f"P.57 trace capture failed: {res['trace_error']}")
            return res
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        try:
            th = []
            for _ in range(REPLAYS):
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                th.append((time.perf_counter() - t0) * 1e3)
                ttnn.synchronize_device(mesh)
            got = _host_bytes(traced_out)
            ident = len(got) == len(ref) and all(torch.equal(a, b) for a, b in zip(got, ref))
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(REPLAYS):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            back = (time.perf_counter() - t0) * 1e3 / REPLAYS
            got2 = _host_bytes(traced_out)
            ident = ident and all(torch.equal(a, b) for a, b in zip(got2, ref))
        finally:
            ttnn.release_trace(mesh, tid)
        model.free(traced_out)
        res["trace_host_ms_layer"] = statistics.median(th)
        res["trace_device_ms_layer"] = back
        res["trace_bit_identical"] = int(ident)
    finally:
        if master is not None:
            model.free(master)
        _free(state)
    return res


# ============================================================================ the test
@pytest.mark.timeout(3 * 3600)
@pytest.mark.parametrize(
    "device_params",
    [dict(S.hooks().device_params(S), trace_region_size=TRACE_REGION)],
    indirect=True,
    ids=["trace"],
)
@pytest.mark.parametrize("mesh_device", [tuple(S.mesh)], indirect=True, ids=["x".join(map(str, S.mesh))])
def test_host_bound(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    monkeypatch.setenv("DS41_MOE_REPLICAS", "prev")
    monkeypatch.setenv("DS41_MOE_FILL", "piggyback")
    monkeypatch.setenv("DS41_ENGRAM_TABLE", "resident")
    monkeypatch.delenv("DS41_MOE_COUNTS", raising=False)
    settings.thaw()
    assert settings.get("MOE_REPLICAS") == "prev" and settings.get("MOE_FILL") == "piggyback"
    layers = S.layers()
    t0 = time.time()
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    logger.info(f"P.57 model loaded in {time.time() - t0:.0f} s")
    clock = HostClock()
    _install(monkeypatch, clock, model)
    tokens = _ids_novel()
    data: dict = {"settings": getattr(model, "perf_settings", lambda: {})()}
    for side in ("off", "prev"):
        _side(mesh_device, side)
        t1 = time.time()
        nv = _novel(model, mesh_device, layers, tokens, clock)
        ps = _position(model, mesh_device, layers, clock, side)
        logger.info(f"P.57 side {side} measured in {time.time() - t1:.0f} s")
        data[side] = {"novel": nv, "p51200": ps}

    _side(mesh_device, "off")
    try:
        data["trace"] = _trace(model, mesh_device, layers, tokens)
    except Exception as e:  # (a failure of the prototype is a finding, not a failed measurement)
        data["trace"] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:400]}"}
        logger.exception("P.57 trace prototype failed")

    # ------------------------------------------------------------------ metrics
    def rec(name, v):
        metrics.record(name, round(float(v), 3) if isinstance(v, float) else v)

    summary = {}
    for side in ("off", "prev"):
        for where in ("novel10", "p51200"):
            sfx = f"{side}" if where == "novel10" else f"{side}_p51200"
            d = data[side]["novel" if where == "novel10" else "p51200"]
            if where == "novel10":
                clean = d["clean"][-1]
                prof = d["profiled"][-1]
                ins = d["instrumented"][-1]
                steady = d["clean"][1:]
                extra = {
                    "host_ms_mean_c1_10": statistics.mean(r["host_ms"] for r in steady),
                    "wall_ms_mean_c1_10": statistics.mean(r["wall_ms"] for r in steady),
                    "host_ms_c": [round(r["host_ms"], 1) for r in d["clean"]],
                    "wall_ms_c": [round(r["wall_ms"], 1) for r in d["clean"]],
                }
                host, wall = clean["host_ms"], clean["wall_ms"]
            else:
                clean = d["clean"]
                prof = d["profiled"]
                ins = d["instrumented"]
                host = statistics.median(r["host_ms"] for r in clean)
                wall = statistics.median(r["wall_ms"] for r in clean)
                extra = {
                    "host_ms_runs": [round(r["host_ms"], 1) for r in clean],
                    "wall_ms_runs": [round(r["wall_ms"], 1) for r in clean],
                }
            p = ins["parts"]
            rec(f"host_ms_chunk_{sfx}", host)
            rec(f"wall_ms_chunk_{sfx}", wall)
            rec(f"device_ms_chunk_{sfx}", prof["device_ms"])
            rec(f"drain_ms_{sfx}", wall - host)
            rec(f"device_span_ms_{sfx}", prof["span_ms"])
            rec(f"device_idle_ms_{sfx}", prof["idle_ms"])
            rec(f"programs_chunk_{sfx}", prof["programs"])
            rec(f"op_calls_chunk_{sfx}", ins["op_calls"])
            rec(f"host_ms_instrumented_{sfx}", ins["host_ms"])
            rec(f"instrument_overhead_ms_{sfx}", ins["host_ms"] - host)
            for k, v in p.items():
                rec(f"host_ms_{k}_{sfx}", v)
            summary[sfx] = {
                "host_ms": host,
                "wall_ms": wall,
                "device": prof,
                "instrumented": ins,
                **extra,
            }
            print(
                f"P.57 {sfx}: host {host:.1f} ms, wall {wall:.1f}, drain {wall - host:.1f}, device {prof['device_ms']:.1f}"
                f" (span {prof['span_ms']:.1f}, idle {prof['idle_ms']:.1f}, {prof['programs']} programs), op calls "
                f"{ins['op_calls']}; instrumented host {ins['host_ms']:.1f}: "
                + ", ".join(f"{k} {v:.1f}" for k, v in p.items())
            )
            top = list(ins["ops"].items())[:25]
            print(f"P.57 {sfx} top host calls (exclusive): " + "; ".join(f"{k} {v}" for k, v in top))
            print(f"P.57 {sfx} Python by section (ms): {dict(list(ins['python_by_section'].items())[:20])}")
            print(f"P.57 {sfx} host by section (ms, Python + ops): {ins['host_by_section']}")
    # (the unsuffixed headline: novel chunk 10, prev)
    rec("host_ms_chunk", summary["prev"]["host_ms"])
    rec("wall_ms_chunk", summary["prev"]["wall_ms"])
    rec("device_ms_chunk", summary["prev"]["device"]["device_ms"])
    rec("op_calls_chunk", summary["prev"]["instrumented"]["op_calls"])
    tr = data["trace"]
    for k in (
        "trace_host_ms_layer",
        "untraced_host_ms_layer",
        "trace_device_ms_layer",
        "untraced_device_ms_layer",
        "untraced_wall_ms_layer",
        "trace_bit_identical",
        "untraced_repeat_identical",
        "programs_layer",
    ):
        if k in tr:
            rec(k, tr[k])
    rec("trace_captured", int("trace_host_ms_layer" in tr))
    print(f"P.57 trace prototype (layer {TRACE_LAYER}, side off): {tr}")
    metrics.record_json("host_bound", {"mesh": list(mesh_device.shape), "summary": summary, "trace": tr})
