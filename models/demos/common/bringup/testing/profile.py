# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Performance review, measurement half: warm per-section, per-chip device time of one long chunk.

Runs the last chunk of the profile rung (spec ``perf.rung``, default the last rung with ``prefix_from_golden``) after
loading the golden state prefix: once to compile, once unsynced for the real wall time, once with section profiling.
F114: the gate first runs it with the device profiler off and BRINGUP_PROFILE_WALL=1 (compile, then WALL_RUNS timed
runs: chunk_wall_ms = the median after the warm-up (F196: no new program; F200: the walls settled), chunk_wall_clean = 1; F119: the harness's
golden-input uploads at layer gaps are timed apart, chunk_wall_gap_ms, and not counted); the profiled run then keeps that chunk_wall_ms and records
its own as chunk_wall_ms_profiled (instrumented kernels and buffer dumps make it slower and noisier).
F155 (owner 2026-10-22): device time comes from the program real-time profiler (testing/profiler.py ``backend()``,
default "rt"): one profiled run, no syncs, gives the sections, the per-op rows of every layer (op mode) and the
pipelined timeline; its wall is recorded as chunk_wall_ms_rt (without op mode, whose per-call bookkeeping is host
work). The device profiler (perf.profiler: device) keeps the old runs (a synced section run, op mode on the
representative layers, a timeline run) for cross-checks.
Writes ``<results>/<task>_profile.json`` (read by the dashboard and by the opportunity list) and records
chunk_wall_ms, host_transfers_per_layer, pcc_chunk_out (last layer's output vs the golden), prefill_ms_full (opt-in), prefill_tok_s,
prefill_chunk_ms_c<nn>, device_ms_total, device_ms_<phase>, device_ms_chip<c>, host_overhead_ms, profiled_programs,
and deferred_cpu_steps / deferred_cpu_ms: the steps deferred to op-gen that ran on the CPU bridge in the timed run and
their host time (F46; the profile lists them as ``bridged_steps``, and their transfers are not host_transfers_per_layer).
"""

from __future__ import annotations

import gzip
import json
import os
import re
import time
from collections import defaultdict

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import accuracy_guard, cpu_bridge
from models.demos.common.bringup.testing import e2e_limits as E
from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.harness import begin_chunk
from models.demos.common.bringup.testing.host_transfers import HostTransfers
from models.demos.common.bringup.testing.ladder import check_states, rel

WALL_RUNS = 10  # F114: timed runs of the clean wall measurement (F196: every one with program-cache misses refused)
WALL_WARM = 10  # F114b: warm-up runs when the mesh cannot count its program cache (selftests); else F196 below
WALL_MAX_WARM = 100  # F196: warm-up runs at most; warm = a run that adds no program-cache entry
WALL_SETTLE_WINDOW = 5  # F200: the host is warm once the last 5 warm-up walls agree within WALL_SETTLE_PCT
WALL_SETTLE_PCT = 5.0
WALL_MAX_SETTLE = 60  # F200: settle runs at most after the cache is warm (a Galaxy chunk takes ~15 s a run)
WALL_TREND_PCT = 2.0  # F205: and the last window's median at most this far below the window before it (no drift)


def cache_entries(mesh) -> int | None:
    """F196: the mesh's program-cache size (None when the mesh cannot tell: a selftest's)."""
    f = getattr(mesh, "num_program_cache_entries", None)
    return int(f()) if f else None


def misses_allowed(mesh, allowed: bool) -> None:
    """F196: refuse program-cache misses (a miss then raises), so a timed run is proven fully warm."""
    f = getattr(mesh, "set_program_cache_misses_allowed", None)
    if f:
        f(allowed)


def record_warm(mesh, before: int | None, what: str) -> None:
    """F197 (owner 2026-10-27): no run is called warm unless it built no program. Records <what>_new_programs (the
    program-cache entries the run added; -1 when the mesh cannot tell) and prints NOT WARM when it built any."""
    after = cache_entries(mesh)
    new = -1 if before is None or after is None else after - before
    metrics.record(f"{what}_new_programs", new)
    if new > 0:
        print(f"NOT WARM: the {what} run built {new} programs (program-cache misses); its host time is not a warm time")


def settled(walls) -> bool:
    """F200: the last WALL_SETTLE_WINDOW walls within WALL_SETTLE_PCT of their median. F205: and no downward drift:
    their median at most WALL_TREND_PCT below the median of the window before (a ~1 %-a-run drift passed the band
    alone: DeepSeek-V4.1 8x4, 200k called settled at 1042 ms after 11 runs while 0k fell to 640 ms over 28)."""
    n = WALL_SETTLE_WINDOW
    w = sorted(walls[-n:])
    if len(w) < n or w[-1] - w[0] > WALL_SETTLE_PCT / 100 * w[n // 2]:
        return False
    prev = sorted(walls[-2 * n : -n])
    return len(prev) == n and w[n // 2] >= (1 - WALL_TREND_PCT / 100) * prev[n // 2]


def warm_up(mesh, one_run) -> dict:
    """F196 (owner 2026-10-27): run until a run adds no program-cache entry (new buffer-address combinations miss the
    cache for a while; the fixed 10 of F114b was a guess), at most WALL_MAX_WARM. F200 (owner 2026-10-27): then keep
    running until the walls settle (host time kept falling 10-25% over 5-6 cache-warm runs), at most WALL_MAX_SETTLE.
    Returns the runs' walls and entries, cache_warm_runs, and settled.
    """
    entries, walls = [cache_entries(mesh)], []
    limit = WALL_MAX_WARM if entries[0] is not None else WALL_WARM
    warm = entries[0] is None
    for _ in range(limit):
        walls.append(one_run())
        entries.append(cache_entries(mesh))
        new = None if entries[-1] is None else entries[-1] - entries[-2]
        print(
            f"F196 warm-up run {len(walls)}: {walls[-1]} ms, program cache {entries[-1]} entries (+{new})", flush=True
        )
        if entries[-1] is not None and entries[-1] == entries[-2]:
            warm = True
            break
    out = {"walls": walls, "entries": entries, "warm": warm, "cache_warm_runs": len(walls), "settled": False}
    if not warm:
        return out
    for _ in range(WALL_MAX_SETTLE):
        if settled(walls):
            break
        walls.append(one_run())
        print(f"F200 settle run {len(walls)}: {walls[-1]} ms", flush=True)
    out["settled"] = settled(walls)
    return out


def profile_rung(s) -> str:
    name = s.get("perf.rung")
    if name:
        return name
    cands = [r["name"] for r in s.data["ladder"] if r.get("prefix_from_golden")]
    return cands[-1] if cands else s.data["ladder"][-1]["name"]


POS_ENV = "BRINGUP_POS"  # F230: profile one chunk starting at this token position (timing only)
HOST_ENV = "BRINGUP_PROFILE_HOST"  # F230: the wall run adds HOST_RUNS chunk runs under cProfile
HOST_RUNS = 5


def position() -> int | None:
    v = os.environ.get(POS_ENV, "").strip()
    return int(v) if v else None


class PositionGolden:
    """F230 (owner 2026-10-27 / 10-30: device and host time of one chunk at 0, 50k, 500k): one chunk at start ``pos``,
    every layer, timing only, as testing/positions.py: the KV prefix is the fresh state's zeros and the tokens random
    ids (the work does not depend on the values). Nothing is compared: no golden exists at such a position."""

    def __init__(self, s, pos: int, chunk: int):
        self.layers = list(s.layers())
        self.n_chunks = (pos + chunk) // chunk
        self.dir = None
        vocab = int(s.get("checkpoint.config.vocab_size", 32000) or 32000)
        self._tokens = torch.randint(0, vocab, (pos + chunk,), generator=torch.Generator().manual_seed(0))

    def tokens(self):
        return self._tokens

    def state(self, i, at):
        return None


def host_profile(one_run) -> None:
    """F230: HOST_RUNS warm chunk runs under cProfile: the per-chunk host time by function, recorded as metrics
    (host_cprofile_ms, host_launch_ms / host_launch_calls: the ttnn op launches) and a side record with the top
    functions; the raw stats beside the results (<task>_host<suffix>.pstats)."""
    import cProfile
    import pstats

    pr, walls = cProfile.Profile(), []
    for _ in range(HOST_RUNS):
        pr.enable()
        walls.append(one_run())
        pr.disable()
    out = metrics.results_dir() / f"{metrics.record_task()}_host{os.environ.get(metrics.SUFFIX_ENV, '')}.pstats"
    out.parent.mkdir(parents=True, exist_ok=True)
    pr.dump_stats(str(out))
    st = pstats.Stats(str(out)).stats
    n = HOST_RUNS
    launch = [v for (f, _, name), v in st.items() if name == "__call__" and f.endswith("decorators.py")]
    metrics.record("host_cprofile_ms", round(sum(walls) / n, 1))
    metrics.record("host_launch_ms", round(sum(v[3] for v in launch) * 1e3 / n, 1))
    metrics.record("host_launch_calls", round(sum(v[1] for v in launch) / n))
    top = sorted(st.items(), key=lambda kv: -kv[1][2])[:25]
    rows = [
        {
            "fn": f"{f.split('/')[-1]}:{ln} {name}",
            "self_ms": round(v[2] * 1e3 / n, 2),
            "cum_ms": round(v[3] * 1e3 / n, 2),
            "calls": round(v[1] / n),
        }
        for (f, ln, name), v in top
    ]
    metrics.record_json(f"host{os.environ.get(metrics.SUFFIX_ENV, '')}", {"runs_ms": walls, "top": rows})
    print(f"PROFILE host: {n} runs under cProfile {walls} ms -> {out}")


def full_prefill(s, model, state, layers, rung, tokens) -> dict | None:
    """The whole target prefill, warm, with nothing read back: embed -> every layer -> final norm for every chunk, one
    device sync at the end. A first pass compiles every position-dependent program; the second is timed
    (prefill_ms_full, prefill_tok_s); a third syncs after each chunk for the per-chunk curve (prefill_chunk_ms_c<nn>).
    Excludes the LM head and sampling (a host LM head would not be device time). Needs a stack that starts at layer 0
    with no gaps: the full model, or a prefix subset such as MiMo's 0-5 of 48 (F42; the final norm only when the stack
    ends at the last layer, and prefill_layers records how many layers were timed)."""
    # F178: a subset with gaps (a partial bring-up's representative layers) is timed back to back, as the device runs
    # them, when every layer's state sources (a KV / index source) are in it; anything else fails loudly (DeepSeek-V4.1
    # X.3: layers 0-3, 14, 20, 21, 24 returned None and the "warm full prefill" was silently skipped)
    if not layers:
        raise ValueError("full prefill: no layers")
    missing = sorted({src for i in layers for src in s.state_sources(i) if src not in layers})
    if missing:
        raise ValueError(f"full prefill: layers {layers} read state from layers {missing} outside the timed stack")
    gapped = layers != list(range(len(layers)))
    full_stack = len(layers) == s.num_layers
    seq, chunk = rung["seq"], rung["chunk"]
    n = seq // chunk

    def once(sync_each=False) -> list[float]:
        marks, t0 = [], time.time()
        for c in range(n):
            s0 = c * chunk
            begin_chunk(model, tokens[s0 : s0 + chunk], s0)  # F65
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
    metrics.record("prefill_gapped", int(gapped))  # F178: the subset back to back (no golden feed between the layers)
    t = once(sync_each=True)
    per = [round((b - a) * 1e3, 1) for a, b in zip(t[:-2], t[1:-1])]
    for c, ms in enumerate(per):
        metrics.record(f"prefill_chunk_ms_c{c:02d}", ms)
    print(
        f"full prefill {seq} tokens in {n} chunks of {chunk}, {len(layers)} layers: {total:.2f}s warm ({seq / total:.0f} tok/s); per chunk {per}"
    )
    return {"tokens": seq, "chunk": chunk, "ms": total * 1e3, "chunk_ms": per}


def want_ops(s, rt: bool = False) -> bool:
    """The per-op breakdown. F187 (owner): every profile on the real-time profiler records it (perf.op_profile, default
    true: the op rows come from the same unsynced run, only host bookkeeping is added), so the dashboard's per-section
    op list is always the profiled code's. On the device profiler (cross-checks) its op mode syncs after every call (a
    minute per layer): only when the spec sets perf.op_profile itself or BRINGUP_PROFILE_OPS=1."""
    if os.environ.get("BRINGUP_PROFILE_OPS") == "1":
        return True
    if rt:
        return bool(s.get("perf.op_profile"))
    return s.has("perf.op_profile") and bool(s.get("perf.op_profile"))


def op_layers(s, layers: list[int], rt: bool = False) -> list[int]:
    """The layers the per-op breakdown covers: BRINGUP_PROFILE_LAYERS ("2" or "0,2") if set; else every layer on the
    real-time profiler (F155: no syncs, the op rows come from the same run as the sections), and one representative
    layer per block type on the device profiler, whose op mode syncs and reads the profiler after every ttnn call
    (about a minute per layer). A perf change is judged by the e2e chunk time and its own section's time."""
    env = os.environ.get("BRINGUP_PROFILE_LAYERS", "").strip()
    if env:
        want = sorted({int(x) for x in env.split(",") if x.strip()})
        bad = [i for i in want if i not in layers]
        assert not bad, f"BRINGUP_PROFILE_LAYERS={env}: layers {bad} are not in the profiled rung's layers"
        if rt:
            return want
    elif rt:
        return list(layers)
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


def rt_ops(prof: dict, sel: list[int] | None) -> dict:
    """F155: the real-time run's op rows in the op-mode schema (profile.json "ops"), timeline columns included; only
    the layers in ``sel``."""
    keep = None if sel is None else {f"L{i}." for i in sel}
    out = {}
    for sec, rows in prof["op_ns"].items():
        if keep is not None and not any(sec.startswith(k) for k in keep):
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
                "gap_ms": round(r["gap_ns"] / 1e6, 4),
                "slot_ms": round(r["slot_ns"] / 1e6, 4),
                "host_ms": round(r["host_ns"] / 1e6, 4),
                "timeline_kernel_ms": round(r["kernel_ns"] / 1e6, 4),
            }
            for r in rows
        ]
    return out


def run_profile(s, mesh, rung_name: str | None = None) -> dict:
    accuracy_guard.check(s)  # accuracy first: a perf pick whose frozen test failed in this attempt is not profiled
    backend = profiler.backend(s)  # F155: "rt" unless the spec / BRINGUP_PROFILER asks for the device profiler
    pos = position()
    if pos is not None:  # F230: one chunk at a position, timing only; every metric carries _p<pos>
        os.environ[metrics.SUFFIX_ENV] = f"_p{pos}"
        chunk = int(s.get("target.chunk"))
        rung_name, rung = f"pos{pos}", {"name": f"pos{pos}", "seq": pos + chunk, "chunk": chunk}
        g = PositionGolden(s, pos, chunk)
        s.data["target"]["seq"] = max(int(s.data["target"]["seq"]), pos + chunk)  # position tables, in memory
    else:
        rung_name = rung_name or profile_rung(s)
        rung = s.rung(rung_name)
        g = Golden.for_rung(s, rung_name)
    metrics.record("scaled_run", int(getattr(s, "scaled", False)))  # F64: the reduced model of a scaled run
    layers = [i for i in s.layers() if i in g.layers]
    chunk = rung["chunk"]
    start = rung["seq"] - chunk
    model = s.hooks().device_model(mesh, s, layers, lm_head=False)
    state = model.new_state(rung["seq"])
    prefix = {i: g.state(i, at=start) for i in layers}

    def load_prefix():
        if pos is not None:  # F230: the fresh state's zeros
            return
        for i in layers:
            state.load_prefix(i, prefix[i], start)

    load_prefix()
    # Fixed-size state (spec state.fixed) advances on every run of the chunk: reload the prefix before each run.
    reload = load_prefix if s.state_fixed else (lambda: None)
    tokens = g.tokens()[start:]
    starts = {i for k, i in enumerate(layers) if k == 0 or layers[k - 1] != i - 1}

    host = []
    gap = [0.0]  # F119: seconds spent uploading golden inputs at layer gaps in the current run

    def run(count=False, only=None, per_layer=False):
        run_layers = layers if only is None else only
        pccs = {}
        run_starts = starts if only is None else {k for n, k in enumerate(only) if n == 0 or only[n - 1] != k - 1}
        h = None
        begin_chunk(model, tokens, start)  # F65
        for i in run_layers:
            if i in run_starts:
                if h is not None:
                    model.free(h)
                if i == 0:
                    h = model.embed(tokens)
                else:  # F119: the golden input at a layer gap is the harness's upload, not the model's time
                    tg = time.time()
                    h = model.from_host(g.layer(g.n_chunks - 1, i)["in"].float())
                    gap[0] += time.time() - tg
            profiler.set_layer(i)
            if count:
                with HostTransfers() as ht:
                    h2 = model.layer(i, h, start, state)
                host.append(ht.total)
            else:
                h2 = model.layer(i, h, start, state)
            if per_layer:  # F122: this layer's output vs the golden's, outside the host-transfer count
                o, w = model.to_host(h2).float(), g.layer(g.n_chunks - 1, i)["out"].float()
                pccs[i] = (metrics.pcc(o, w), rel(o, w))
            model.free(h)
            h = h2
        model.sync()
        if (
            count and only is None and pos is None
        ):  # the counting run is untimed: also check this chunk is still right (cheap accuracy for perf work)
            got = model.to_host(h).float()
            metrics.record("pcc_chunk_out", metrics.pcc(got, g.layer(g.n_chunks - 1, layers[-1])["out"].float()))
        model.free(h)
        return pccs

    run()  # compile and fill the program cache: performance is measured warm only
    if os.environ.get("BRINGUP_PROFILE_WALL") == "1":  # F114: the clean wall run (device profiler off), nothing else
        assert not os.environ.get(
            "TT_METAL_DEVICE_PROFILER"
        ), "BRINGUP_PROFILE_WALL=1 runs with the device profiler off"
        walls, gaps = [], []

        def one_run():
            reload()
            gap[0] = 0.0
            t0 = time.time()
            run()
            gaps.append(round(gap[0] * 1e3, 1))
            return round((time.time() - t0 - gap[0]) * 1e3, 1)

        w = warm_up(mesh, one_run)
        metrics.record("chunk_wall_warm_runs", len(w["walls"]))
        metrics.record("chunk_wall_warm_ms", w["walls"])
        metrics.record("chunk_wall_cache_warm_runs", w["cache_warm_runs"])
        metrics.record("chunk_wall_settled", int(w["settled"]))  # F200: 0 = the timed runs may still be falling
        metrics.record("chunk_wall_loadavg", round(os.getloadavg()[0], 2))  # F200: other host load skews a wall
        if not w["settled"]:
            print(
                f"F200 NOT SETTLED: the last {WALL_SETTLE_WINDOW} warm-up walls differ by more than {WALL_SETTLE_PCT}%"
            )
        if w["entries"][0] is not None:
            metrics.record("program_cache_entries", w["entries"][-1])
            metrics.record("program_cache_entries_runs", w["entries"])
        assert w["warm"], (
            f"F196: not warm after {len(w['walls'])} runs: every run still adds program-cache entries "
            f"({w['entries'][-5:]}); a program keyed on something that changes every run (an address, a position)"
        )
        misses_allowed(mesh, False)  # F196: a timed run that completes hit the cache for every program
        try:
            for _ in range(WALL_RUNS):
                walls.append(one_run())
        finally:
            misses_allowed(mesh, True)
        metrics.record("chunk_wall_cache_hits_only", int(w["entries"][0] is not None))
        kept = sorted(walls)
        med = kept[len(kept) // 2]
        metrics.record("chunk_wall_ms", med)
        metrics.record("chunk_wall_ms_runs", walls)
        metrics.record("chunk_wall_ms_spread", round(kept[-1] - kept[0], 1))
        metrics.record("chunk_wall_gap_ms", sorted(gaps)[len(gaps) // 2])
        metrics.record("chunk_wall_clean", 1)
        print(
            f"PROFILE wall (device profiler off, gap uploads excluded): warm after {len(w['walls'])} runs "
            f"(program cache {w['entries'][0]} -> {w['entries'][-1]} entries; warm-up walls {w['walls']} ms); "
            f"{len(walls)} timed runs, cache misses refused: {walls} ms, median {med}; gap uploads "
            f"{sorted(gaps)[len(gaps) // 2]} ms"
        )
        if os.environ.get(HOST_ENV) == "1":  # F230: warm, after the timed runs
            misses_allowed(mesh, False)
            try:
                host_profile(one_run)
            finally:
                misses_allowed(mesh, True)
        if (
            os.environ.get("BRINGUP_QUICK") == "1" and pos is None
        ):  # F122: the quick A/B also checks every layer against the golden
            reload()
            pcc = run(count=True, per_layer=True)
            drift, failed = (
                E.load(g.dir, rung_name),
                [],
            )  # F131: the ladder's limits, so a `>= limit` gate can judge the quick run
            for i, (v, e) in pcc.items():
                metrics.record(f"pcc_layer_L{i:02d}", v)
                metrics.record(f"err_layer_L{i:02d}", e)  # F135: the rel error gates, as in the ladder
                E.check(drift, f"layer_L{i:02d}", f"err_layer_L{i:02d}", e, failed)
            check_states(state, g, layers, rung["seq"], drift, failed, s.get("tests.waivers"))
            metrics.record("host_transfers_per_layer", max(host) if host else 0)
            worst = max(pcc, key=lambda i: pcc[i][1])
            print(f"PROFILE quick: worst layer rel {pcc[worst][1]:.5f} (L{worst:02d}, PCC {pcc[worst][0]:.5f})")
            print("".join(f"FAIL {f}\n" for f in failed) or "PROFILE quick: every layer and state within its limit")
        return {"wall_ms": med, "wall_runs_ms": walls}
    reload()
    cpu_bridge.STATS.reset()
    gap[0] = 0.0
    n0 = cache_entries(mesh)
    t0 = time.time()
    run()
    wall = time.time() - t0 - gap[0]  # F119
    record_warm(mesh, n0, "plain")
    cpu_bridge.record(metrics)
    bridged = {"steps": cpu_bridge.STATS.step_names, "ms": round(cpu_bridge.STATS.ms, 1)}
    reload()
    run(count=True)  # warm, apart from the timed run: host round-trips inside the forward pass (agent rule 5)
    metrics.record("host_transfers_per_layer", max(host))
    # The full-target prefill takes minutes; it is for a final number, not for every perf iteration (spec perf.full_prefill
    # or BRINGUP_FULL_PREFILL=1).
    want_full = s.get("perf.full_prefill") or os.environ.get("BRINGUP_FULL_PREFILL") == "1"
    full = full_prefill(s, model, state, layers, rung, g.tokens()) if want_full else None

    ops, timeline = None, None
    reload()
    if backend == "rt":  # F155: one run, no syncs: sections, and with op mode the op rows and the timeline
        ops_on = want_ops(s, rt=True)
        gap[0] = 0.0
        profiler.enable(mesh, rt=True, calls=ops_on)
        try:
            n0 = cache_entries(mesh)
            t0 = time.time()
            run()
            rt_wall = time.time() - t0 - gap[0]
            record_warm(mesh, n0, "profiled")
            profiler.set_layer(None)
            profiler.signpost("end")
            profiler.finish_rt()
            prof = profiler.result()
        finally:
            profiler.disable()
        if not ops_on:  # op mode's per-call bookkeeping is host work: then the wall is not the model's
            metrics.record("chunk_wall_ms_rt", round(rt_wall * 1e3, 1))
        if ops_on:
            sel = op_layers(s, layers, rt=True)
            print(f"op profile (real-time profiler, same run) on layers {sel} (of {len(layers)})")
            metrics.record("op_layers", len(sel))
            ops = rt_ops(prof, sel)
            timeline = dict(prof["timeline_rt"], host_wall_ms=round(rt_wall * 1e3, 3))
    else:
        profiler.require_device_profiler()
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
    if ops is not None:
        metrics.record("op_rows", sum(len(v) for v in ops.values()))
        attached = all("gap_ms" in r for rows in ops.values() for r in rows)  # every op row got its timeline columns
        metrics.record("timeline_ok", int(bool(timeline) and "error" not in timeline and attached))
        if timeline and "error" not in timeline:
            for k in ("device_timeline_ms", "kernel_ms", "gap_ms", "host_dispatch_ms", "host_wall_ms"):
                metrics.record(f"timeline_{k}", timeline[k])

    total = sum(prof["kernel_ns"].values())
    assert total > 0, (
        "the real-time profiler returned no program records in any section"
        if backend == "rt"
        else f"device profiler returned no durations; set {profiler.PROFILER_ENV}"
    )
    metrics.record("profiler_rt", int(backend == "rt"))  # F155: where device_ms_* came from
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
    # F114: with a clean wall run first (gate_command adds it), the profiled run's wall is only for reference
    kept = metrics.load(metrics.record_task())
    clean = os.environ.get("BRINGUP_WALL_CLEAN") == "1" and kept.get(metrics.suffixed("chunk_wall_clean"))
    metrics.record("chunk_wall_ms_profiled", round(wall * 1e3, 1))
    if clean:
        wall = kept[metrics.suffixed("chunk_wall_ms")]["value"] / 1e3
    else:
        metrics.record("chunk_wall_ms", round(wall * 1e3, 1))
        metrics.record("chunk_wall_clean", 0)
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
        "profiler": backend,  # F155: "rt" = dispatch start to end per program; "device" = kernel time
        "settings": getattr(model, "perf_settings", lambda: {})(),
        "full_prefill": full,
        "bridged_steps": bridged["steps"],  # deferred to op-gen: host time in the wall, not device work
        "deferred_cpu_ms": bridged["ms"],
        **({"ops": ops} if ops else {}),
        **({"timeline": timeline} if timeline else {}),
    }
    path = metrics.results_dir() / f"{metrics.record_task()}_profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = dict(out)
    if "ops" in slim:  # F148: the per-op rows go gzipped beside it (the repo's 500 KB hook refused a 613 KB profile)
        ops_name = f"{metrics.record_task()}_ops.json.gz"
        with gzip.open(path.parent / ops_name, "wt") as f:
            json.dump(slim.pop("ops"), f)
        slim["ops_file"] = ops_name
    path.write_text(json.dumps(slim, indent=1) + "\n")
    s.profiles_dir.mkdir(parents=True, exist_ok=True)
    (s.profiles_dir / f"{metrics.record_task()}_{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(
        json.dumps(out, indent=1)
    )
    print(f"chunk [{start},{rung['seq']}): wall {wall * 1e3:.0f} ms, device {total / 1e6:.0f} ms")
    for sec, ms in list(out["sections_ms"].items())[:15]:
        print(f"  {sec:40s} {ms:9.1f} ms {100 * ms * 1e6 / total:5.1f}%")
    return out
