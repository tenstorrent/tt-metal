# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Performance review, measurement half: warm per-section, per-chip device time of one long chunk.

Runs the last chunk of the profile rung (spec ``perf.rung``, default the last rung with ``prefix_from_golden``) after
loading the golden state prefix: once to compile, once unsynced for the real wall time, once with section profiling.
Writes ``<results>/<task>_profile.json`` (read by the dashboard and by the opportunity list) and records
chunk_wall_ms, host_transfers_per_layer, pcc_chunk_out (last layer's output vs the golden), prefill_ms_full (opt-in), prefill_tok_s,
prefill_chunk_ms_c<nn>, device_ms_total, device_ms_<phase>, device_ms_chip<c>, host_overhead_ms, profiled_programs,
and deferred_cpu_steps / deferred_cpu_ms: the steps deferred to op-gen that ran on the CPU bridge in the timed run and
their host time (F46; the profile lists them as ``bridged_steps``, and their transfers are not host_transfers_per_layer).
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import accuracy_guard, cpu_bridge, profiler
from models.demos.common.bringup.testing.host_transfers import HostTransfers


def profile_rung(s) -> str:
    name = s.get("perf.rung")
    if name:
        return name
    cands = [r["name"] for r in s.data["ladder"] if r.get("prefix_from_golden")]
    return cands[-1] if cands else s.data["ladder"][-1]["name"]


def full_prefill(s, model, state, layers, rung, tokens) -> dict | None:
    """The whole target prefill, warm, with nothing read back: embed -> every layer -> final norm for every chunk, one
    device sync at the end. A first pass compiles every position-dependent program; the second is timed
    (prefill_ms_full, prefill_tok_s); a third syncs after each chunk for the per-chunk curve (prefill_chunk_ms_c<nn>).
    Excludes the LM head and sampling (a host LM head would not be device time). Needs a stack that starts at layer 0
    with no gaps: the full model, or a prefix subset such as MiMo's 0-5 of 48 (F42; the final norm only when the stack
    ends at the last layer, and prefill_layers records how many layers were timed)."""
    if not layers or layers != list(range(len(layers))):
        return None
    full_stack = len(layers) == s.num_layers
    seq, chunk = rung["seq"], rung["chunk"]
    n = seq // chunk

    def once(sync_each=False) -> list[float]:
        marks, t0 = [], time.time()
        for c in range(n):
            s0 = c * chunk
            h = model.embed(tokens[s0 : s0 + chunk])
            for i in layers:
                h2 = model.layer(i, h, s0, state)
                model.free(h)
                h = h2
            if c == n - 1 and full_stack:
                h2 = model.final_norm(h)
                model.free(h)
                h = h2
            model.free(h)
            if sync_each:
                model.sync()
                marks.append(time.time())
        model.sync()
        return [t0] + marks + [time.time()]

    once()  # compile
    t = once()
    total = t[-1] - t[0]
    metrics.record("prefill_ms_full", round(total * 1e3, 1))
    metrics.record("prefill_seq", seq)
    metrics.record("prefill_chunk", chunk)
    metrics.record("prefill_tok_s", round(seq / total, 1))
    metrics.record("prefill_layers", len(layers))
    t = once(sync_each=True)
    per = [round((b - a) * 1e3, 1) for a, b in zip(t[:-2], t[1:-1])]
    for c, ms in enumerate(per):
        metrics.record(f"prefill_chunk_ms_c{c:02d}", ms)
    print(
        f"full prefill {seq} tokens in {n} chunks of {chunk}, {len(layers)} layers: {total:.2f}s warm ({seq / total:.0f} tok/s); per chunk {per}"
    )
    return {"tokens": seq, "chunk": chunk, "ms": total * 1e3, "chunk_ms": per}


def want_ops(s) -> bool:
    """The per-op breakdown (spec perf.op_profile or BRINGUP_PROFILE_OPS=1; the final X.3 sets it)."""
    return bool(s.get("perf.op_profile", False)) or os.environ.get("BRINGUP_PROFILE_OPS") == "1"


def op_layers(s, layers: list[int]) -> list[int]:
    """The layers the per-op breakdown runs on: one representative layer per block type by default, or the list in
    BRINGUP_PROFILE_LAYERS ("2" or "0,2"). Op mode syncs and reads the profiler after every ttnn call (about a minute
    per layer), so it never runs over the whole model: the section times of the plain profile already cover every
    layer. A perf change is judged by the e2e chunk time and its own section's time, before and after."""
    env = os.environ.get("BRINGUP_PROFILE_LAYERS", "").strip()
    if env:
        want = sorted({int(x) for x in env.split(",") if x.strip()})
        bad = [i for i in want if i not in layers]
        assert not bad, f"BRINGUP_PROFILE_LAYERS={env}: layers {bad} are not in the profiled rung's layers"
    else:
        want = sorted({s.representative_layer(bt) for bt in s.data.get("block_types", {})} & set(layers))
    assert (
        0 < len(want) <= max(2, len(s.data.get("block_types", {})))
    ), f"op mode on {len(want)} layers: it takes about a minute per layer; pick one layer per block type"
    return want


def op_profile(mesh, run) -> tuple[dict, dict | None]:
    """Two more warm runs. Op mode (F43): per layer and section ("L3.attention.qkv"), the ttnn ops in execution order
    with calls, device programs, device ms (slowest chip per call, summed) and ms per chip. Timeline (F44): the same
    chunk with no syncs; each op row also gets gap_ms (device idle before its programs on the critical chip: dispatch
    and host cost the pipeline did not hide), slot_ms (its end minus the previous op's end) and host_ms (host
    dispatch time of the call)."""
    profiler.enable(mesh, ops=True)
    try:
        run()
        profiler.set_layer(None)
        profiler.signpost("end")
        res = profiler.result()
    finally:
        profiler.disable()
    op_seq, dev_to_chip = res["seq"], res["dev_to_chip"]
    profiler.enable(mesh, timeline=True)
    try:
        t0 = time.perf_counter()
        run()
        host_wall = time.perf_counter() - t0
        progs = profiler.collect_timeline()
        tl_seq = profiler.result()["seq"]
    finally:
        profiler.set_layer(None)
        profiler.disable()
    tl = profiler.align_timeline(op_seq, tl_seq, progs, dev_to_chip)
    if "error" in tl:
        print(f"timeline: {tl['error']}")
    out = {}
    for sec, rows in res["op_ns"].items():
        if sec == "end":
            continue
        out[sec] = [
            {
                "op": r["op"],
                "shape": r["shape"],
                "mem": r.get("mem", ""),
                "calls": r["calls"],
                "programs": r["programs"],
                "ms": round(r["ns"] / 1e6, 4),
                "ms_per_chip": {str(c): round(v / 1e6, 4) for c, v in sorted(r["ns_dev"].items())},
            }
            for r in rows
        ]
    total = sum(r["ms"] for rows in out.values() for r in rows)
    print(f"op profile: {sum(len(v) for v in out.values())} op rows over {len(out)} sections, {total:.1f} ms device")
    summary = None
    if "calls" in tl:
        # the same merge as the op rows (per section, back-to-back repeats of op + shape), then zip onto them
        agg = {}
        for c in tl["calls"]:
            if c["key"] is None or not c.get("programs"):  # op rows only hold calls that launched programs
                continue
            rows = agg.setdefault(c["key"], [])
            if (
                rows
                and rows[-1]["op"] == c["op"]
                and rows[-1]["shape"] == c["shape"]
                and rows[-1]["mem"] == c.get("mem", "")
            ):
                r = rows[-1]
            else:
                r = {
                    "op": c["op"],
                    "shape": c["shape"],
                    "mem": c.get("mem", ""),
                    "gap": 0.0,
                    "slot": 0.0,
                    "host": 0.0,
                    "kernel": 0.0,
                    "n": 0,
                }
                rows.append(r)
            r["gap"] += c["gap_ns"]
            r["slot"] += c["slot_ns"]
            r["host"] += c["host_ns"]
            r["kernel"] += c["kernel_ns"]
            r["n"] += 1
        for sec, rows in out.items():
            trow = agg.get(sec, [])
            if [(r["op"], r["shape"], r["mem"]) for r in trow] != [(o["op"], o["shape"], o["mem"]) for o in rows]:
                continue
            for o, t in zip(rows, trow):
                o.update(
                    gap_ms=round(t["gap"] / 1e6, 4),
                    slot_ms=round(t["slot"] / 1e6, 4),
                    host_ms=round(t["host"] / 1e6, 4),
                    timeline_kernel_ms=round(t["kernel"] / 1e6, 4),
                )
        summary = dict(tl["summary"], host_wall_ms=round(host_wall * 1e3, 3))
        print(
            f"timeline: device {summary['device_timeline_ms']:.1f} ms = kernels {summary['kernel_ms']:.1f} + gaps "
            f"{summary['gap_ms']:.1f} (chip {summary['critical_chip']}); host dispatch {summary['host_dispatch_ms']:.1f} ms; "
            f"host wall {summary['host_wall_ms']:.1f} ms"
        )
    else:
        summary = {"error": tl["error"]}
    return out, summary


def run_profile(s, mesh, rung_name: str | None = None) -> dict:
    accuracy_guard.check(s)  # accuracy first: a perf pick whose frozen test failed in this attempt is not profiled
    rung_name = rung_name or profile_rung(s)
    rung = s.rung(rung_name)
    g = Golden.for_rung(s, rung_name)
    layers = [i for i in s.layers() if i in g.layers]
    chunk = rung["chunk"]
    start = rung["seq"] - chunk
    model = s.hooks().device_model(mesh, s, layers, lm_head=False)
    state = model.new_state(rung["seq"])
    prefix = {i: g.state(i, at=start) for i in layers}

    def load_prefix():
        for i in layers:
            state.load_prefix(i, prefix[i], start)

    load_prefix()
    # Fixed-size state (spec state.fixed) advances on every run of the chunk: reload the prefix before each run.
    reload = load_prefix if s.state_fixed else (lambda: None)
    tokens = g.tokens()[start:]
    starts = {i for k, i in enumerate(layers) if k == 0 or layers[k - 1] != i - 1}

    host = []

    def run(count=False, only=None):
        run_layers = layers if only is None else only
        run_starts = starts if only is None else {k for n, k in enumerate(only) if n == 0 or only[n - 1] != k - 1}
        h = None
        for i in run_layers:
            if i in run_starts:
                if h is not None:
                    model.free(h)
                h = model.embed(tokens) if i == 0 else model.from_host(g.layer(g.n_chunks - 1, i)["in"].float())
            profiler.set_layer(i)
            if count:
                with HostTransfers() as ht:
                    h2 = model.layer(i, h, start, state)
                host.append(ht.total)
            else:
                h2 = model.layer(i, h, start, state)
            model.free(h)
            h = h2
        model.sync()
        if (
            count and only is None
        ):  # the counting run is untimed: also check this chunk is still right (cheap accuracy for perf work)
            got = model.to_host(h).float()
            metrics.record("pcc_chunk_out", metrics.pcc(got, g.layer(g.n_chunks - 1, layers[-1])["out"].float()))
        model.free(h)

    run()  # compile and fill the program cache: performance is measured warm only
    reload()
    cpu_bridge.STATS.reset()
    t0 = time.time()
    run()
    wall = time.time() - t0
    cpu_bridge.record(metrics)
    bridged = {"steps": cpu_bridge.STATS.step_names, "ms": round(cpu_bridge.STATS.ms, 1)}
    reload()
    run(count=True)  # warm, apart from the timed run: host round-trips inside the forward pass (agent rule 5)
    metrics.record("host_transfers_per_layer", max(host))
    # The full-target prefill takes minutes; it is for a final number, not for every perf iteration (spec perf.full_prefill
    # or BRINGUP_FULL_PREFILL=1).
    want_full = s.get("perf.full_prefill", False) or os.environ.get("BRINGUP_FULL_PREFILL") == "1"
    full = full_prefill(s, model, state, layers, rung, g.tokens()) if want_full else None

    reload()
    profiler.enable(mesh)
    run()
    profiler.signpost("end")
    prof = profiler.result()
    profiler.disable()
    if want_ops(s):
        sel = op_layers(s, layers)
        print(f"op profile on layers {sel} (of {len(layers)})")
        metrics.record("op_layers", len(sel))
        reload()
        ops, timeline = op_profile(mesh, lambda: run(only=sel))
    else:
        ops, timeline = None, None
    if ops is not None:
        metrics.record("op_rows", sum(len(v) for v in ops.values()))
        attached = all("gap_ms" in r for rows in ops.values() for r in rows)  # every op row got its timeline columns
        metrics.record("timeline_ok", int(bool(timeline) and "error" not in timeline and attached))
        if timeline and "error" not in timeline:
            for k in ("device_timeline_ms", "kernel_ms", "gap_ms", "host_dispatch_ms", "host_wall_ms"):
                metrics.record(f"timeline_{k}", timeline[k])

    total = sum(prof["kernel_ns"].values())
    assert total > 0, f"device profiler returned no durations; set {profiler.PROFILER_ENV}"
    phases = defaultdict(float)
    for sec, ns in prof["kernel_ns"].items():
        phases[profiler.phase_of(sec)] += ns
    for ph, ns in phases.items():
        metrics.record(f"device_ms_{ph}", round(ns / 1e6, 2))
    for sec, ns in prof["kernel_ns"].items():  # sub-sections a module marks itself ("attention.sdpa")
        if "." in sec:
            metrics.record(f"device_ms_{re.sub(r'[^A-Za-z0-9]+', '_', sec)}", round(ns / 1e6, 2))
    per_chip = defaultdict(float)
    for d in prof["kernel_ns_dev"].values():
        for c, ns in d.items():
            per_chip[c] += ns
    for c, ns in sorted(per_chip.items()):
        metrics.record(f"device_ms_chip{c}", round(ns / 1e6, 2))
    metrics.record("chunk_wall_ms", round(wall * 1e3, 1))
    metrics.record("chunk_start", start)
    metrics.record("chunk_len", chunk)
    metrics.record("device_model_hybrid", int("Hybrid" in type(model).__name__))
    metrics.record("device_ms_total", round(total / 1e6, 2))
    metrics.record("host_overhead_ms", round(wall * 1e3 - total / 1e6, 1))
    metrics.record("profiled_programs", sum(prof["programs"].values()))

    out = {
        "rung": rung_name,
        "chunk": [start, rung["seq"]],
        "layers": layers,
        "wall_ms": wall * 1e3,
        "device_ms": {ph: ns / 1e6 for ph, ns in sorted(phases.items(), key=lambda x: -x[1])},
        "sections_ms": {k: v / 1e6 for k, v in sorted(prof["kernel_ns"].items(), key=lambda x: -x[1])},
        "sections_ms_per_chip": {k: {str(c): v / 1e6 for c, v in d.items()} for k, d in prof["kernel_ns_dev"].items()},
        "programs": prof["programs"],
        "settings": getattr(model, "perf_settings", lambda: {})(),
        "full_prefill": full,
        "bridged_steps": bridged["steps"],  # deferred to op-gen: host time in the wall, not device work
        "deferred_cpu_ms": bridged["ms"],
        **({"ops": ops} if ops else {}),
        **({"timeline": timeline} if timeline else {}),
    }
    path = metrics.results_dir() / f"{metrics.task_id()}_profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    s.profiles_dir.mkdir(parents=True, exist_ok=True)
    (s.profiles_dir / f"{metrics.task_id()}_{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(
        json.dumps(out, indent=1)
    )
    print(f"chunk [{start},{rung['seq']}): wall {wall * 1e3:.0f} ms, device {total / 1e6:.0f} ms")
    for sec, ms in list(out["sections_ms"].items())[:15]:
        print(f"  {sec:40s} {ms:9.1f} ms {100 * ms * 1e6 / total:5.1f}%")
    return out
