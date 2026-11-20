# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.7 (owner 2026-11-20): the GLUED packed prefill's walls, serving-equivalent, vs the fixed-lane path.

DECODER_SKIP=decode (layers 0-20 run), PACKED=on, agentic text (BRINGUP_POS_TOKENS=agentic_cc). Every segment is a
request at its real history position with its own slot state (one capacity for every slot: the deployment context,
OV-62; the long-range history is the state's zeros, as test_packed_walls). Timed: from the engine's device input
(uint32 [SP, 1, W], segments back to back, already on the device) through the ids readback to the sync after the last
layer; tok/s counts REAL new tokens only (each segment's engine rows).

Cases (metric suffix; glued = PACKED_GLUED=on, lanes = off: the accepted 4-lane path):
    4x2k_p139264    four 2048-token segments at 139264 (glued: the lane chunk, so the lane path; vs PB.6L's 74.6k)
    5x1600_p139264  five 1600-token segments at 139264 (5 x 1.6k over 139k; 8000 real rows)
    n<k>            k = 1..6 segments of 8192 / k rows (64-aligned) at 139264: the per-segment overhead
    6x_p1m          six 1344-token segments at 1,040,384 - k x 8192: six slots' histories at ~1M resident (DRAM)
    agentx_m<k>     a request mix sampled from the AgentX per-turn distribution (new tokens and history: quantiles
                    of the 98,827-request replay, BATCHING_PLAN.md: new p25 640 / p50 1.6k / p75 2.9k / p90 5.9k,
                    history p50 139k / p90 547k; requests.json itself is not on this box; DECIDED, seed k), packed
                    as tt-d-gen would: glued (GluedChunk.fill: back to back, <= 6 a chunk, splits at a kv block)
                    and lanes (<= 2048 a lane, a longer request in 2048 pieces over later chunks, 4 lanes a chunk);
                    tok/s = the mix's new tokens / the sum of its chunks' walls.
                    PB.7C (owner 2026-11-20 16:00): FRESH layouts, as serving sees them: every timed run is a NEW mix
                    (seed 1000 k + run, logged), so no timed chunk repeats a layout; GLUED_WARM_MIXES warm-up mixes
                    (seeds 1000 k + 900 + w) run first. Records glued_program_builds_after_warmup_<case> (program-
                    cache entries the timed runs added: a layout-dependent program shows here) and the host ms a new
                    layout costs (glued_new_layout_ms_<case>: every timed mix's chunk walls fresh minus the same
                    chunks run again right after, averaged over every timed chunk), next to the repeated-layout tok/s
                    (glued_serving_tok_s_repeat_<case>, PB.7B's method)
Gathers: per segment the history gathers of a profiled run (tt/history_gather.TRACE, each at its least chip):
the glued path's moves run each segment's gathers serially inside its attention (exposed whole); the lane path's
are hidden beside the attention (PACKED_GATHER_HIDE) and reported by test_packed_walls' stamps method there.

Metrics: glued_indexer_ring_calls_per_layer_<case> (the most ring_indexer_score_dsa calls any layer of the glued side's
timed chunks made, counted at the op; PB.7C: 1), glued_wall_ms_<case>, glued_serving_tok_s_<case>, glued_device_ms_<case> (profiled), glued_path_<case>
(lanes / glued), lanes_* for the lane side, glued_gather_ms_<case>_seg<k> / glued_gather_exposed_ms_<case>_seg<k>,
glued_dram_gib_6x_p1m / glued_dram_headroom_gb_6x_p1m, the table results/PB.7_glued_walls.md.

    BRINGUP_CHUNK=8192 DS41_PACKED=on DS41_PACKED_GLUED=on BRINGUP_POS_TOKENS=agentic_cc DS41_DECODER_SKIP=decode \\
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_walls.py
"""

from __future__ import annotations

import importlib
import os
import random
import time

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import WALL_RUNS, PositionGolden, misses_allowed, warm_up
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc
from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, GluedChunk, Segment
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

_pc = importlib.reload(_pc)  # (the resident runner keeps helper modules of an earlier request: take this file's)
CHUNK, packed, glued = _pc.CHUNK, _pc.packed, _pc.glued

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = ("GLUED_WALL_CASES", "PACKED_WALL_CAP", "GLUED_WALL_RUNS", "GLUED_WARM_MIXES")
BATCH_LAYERS = "executed"  # F285: a performance measurement: the layers it executes (0-20 under decode)
P139 = 139264
P1M = 1040384
SHIFT = 4096  # segment k reads the text from k x SHIFT
GIB = float(1 << 30)
AGENTX_NEW = [(0.0, 64), (0.25, 640), (0.5, 1600), (0.75, 2900), (0.9, 5900), (1.0, 12000)]
AGENTX_HIST = [(0.0, 4096), (0.25, 61440), (0.5, 139264), (0.75, 307200), (0.9, 546816), (1.0, 1000000)]
AGENTX_MIXES = 3
AGENTX_TOKENS = 20000  # new tokens a mix (~2.5 chunks)
WARM_MIXES = 4  # PB.7C: fresh warm-up mixes before the timed fresh ones (GLUED_WARM_MIXES)


def _fresh_seeds(case: str) -> tuple:
    """PB.7C: (warm-up seeds, timed seeds) of an agentx case: every run a new mix (no layout repeats)."""
    k = int(case[len("agentx_m") :])
    runs = int(os.environ.get("GLUED_WALL_RUNS", "") or min(WALL_RUNS, 5))
    warm = int(os.environ.get("GLUED_WARM_MIXES", "") or WARM_MIXES)
    return [1000 * k + 900 + w for w in range(warm)], [1000 * k + r for r in range(runs)]


def _shift(k: int) -> int:
    """Request k's text offset (k mod 6 x SHIFT: a chunk's segments read different stretches of the session)."""
    return (k % 6) * SHIFT


def _cases() -> list:
    v = os.environ.get("GLUED_WALL_CASES", "").strip()
    if v:
        return [c for c in v.split(",") if c]
    return (
        ["4x2k_p139264", "5x1600_p139264"]
        + [f"n{k}" for k in range(1, 7)]
        + ["6x_p1m"]
        + [f"agentx_m{k}" for k in range(1, AGENTX_MIXES + 1)]
    )


def _walls_mod():
    from models.demos.deepseek_v41_flash_d_p.tests.packed import test_packed_walls as W

    return W


def _inv(q: list, u: float) -> int:
    """Piecewise-linear inverse CDF through the quantile points ``q`` [(p, value)]."""
    for (p0, v0), (p1, v1) in zip(q, q[1:]):
        if u <= p1:
            return int(v0 + (v1 - v0) * (u - p0) / max(p1 - p0, 1e-9))
    return int(q[-1][1])


def agentx_mix(seed: int, tokens: int = AGENTX_TOKENS) -> list:
    """[(history start, new tokens)] sampled with ``seed`` until ``tokens`` new tokens: history rounded down to a kv
    block (starts are block-aligned), new tokens >= 1."""
    rnd = random.Random(seed)
    out, tot = [], 0
    while tot < tokens:
        h = _inv(AGENTX_HIST, rnd.random()) // 64 * 64
        n = max(1, _inv(AGENTX_NEW, rnd.random()))
        out.append((h, n))
        tot += n
    return out


def glued_chunks(reqs: list) -> list:
    """Requests [Segment] -> the glued chunks tt-d-gen would send (GluedChunk.fill in arrival order)."""
    out, rest = [], list(reqs)
    while rest:
        c, rest = GluedChunk.fill(CHUNK, rest)
        out.append(c)
    return out


def lane_chunks(reqs: list, lane: int = 2048, lanes: int = 4) -> list:
    """Requests [Segment] -> fixed-lane chunks: each request in pieces of <= ``lane`` tokens (a piece ends on a kv
    block when more follows), up to ``lanes`` requests a chunk (a request's next piece in a later chunk)."""
    queue = [list(_pieces(s, lane)) for s in reqs]
    out = []
    while queue:
        take = queue[:lanes]
        out.append(ChunkDescriptor.packed(CHUNK, [p.pop(0) for p in take], lanes=lanes))
        queue = [p for p in take if p] + queue[lanes:]
    return out


def _pieces(s: Segment, lane: int):
    a = s.start
    while a < s.end:
        b = min(s.end, a + lane)
        if b < s.end:
            b = b // 64 * 64
        yield Segment(s.slot, a, b, s.write_start)
        a = b


def _case_requests(case: str) -> list:
    """[(history start, new tokens)] of a fixed case."""
    if case == "4x2k_p139264":
        return [(P139, 2048)] * 4
    if case == "5x1600_p139264":
        return [(P139, 1600)] * 5
    if case.startswith("n"):
        k = int(case[1:])
        return [(P139, CHUNK // k // 64 * 64)] * k
    if case == "6x_p1m":
        return [(P1M - 2 * k * SHIFT, 1344) for k in range(6)]  # (text P1M - k SHIFT: six different stretches)
    raise KeyError(case)


@mesh_parametrize(timeout_s=4 * 3600)
def test_glued_walls(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192"
    assert settings.get("DECODER_SKIP") == "decode", "the glued walls measure DECODER_SKIP=decode"
    W = _walls_mod()
    full = list(S.layers())
    layers = batch.run_layers(S, full, last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cap = W._cap(deploy)
    want = max(cap, int(S.data["target"]["seq"]))
    monkeypatch.setitem(S.data["target"], "seq", want)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    cases = _cases()
    plans, fresh = {}, {}
    for c in cases:
        if c.startswith("agentx_m"):
            warm_seeds, timed_seeds = _fresh_seeds(c)
            fresh[c] = {sd: agentx_mix(sd) for sd in warm_seeds + timed_seeds}
            plans[c] = fresh[c][timed_seeds[0]]
        else:
            plans[c] = _case_requests(c)
    every = list(plans.values()) + [m for f in fresh.values() for m in f.values()]
    need = max(_shift(k) + h + n for reqs in every for k, (h, n) in enumerate(reqs)) + CHUNK
    assert max(h + n for reqs in every for h, n in reqs) <= cap, "a request past the slots' capacity"
    text = PositionGolden(S, need, CHUNK).tokens()
    dram = W._dram(mesh_device)
    proj = W._Unloaded(model, full) if dram is not None else None
    rows = []
    try:
        for c in cases:
            sides = ("glued", "lanes") if c.startswith("agentx_m") or c in ("5x1600_p139264",) else ("glued",)
            for side in sides:
                rows.append(
                    _measure(model, mesh_device, layers, c, side, plans[c], text, cap, dram, proj, fresh.get(c))
                )
    finally:
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    table = _table(rows, cap, plans)
    metrics.record("glued_walls_table", table)
    out = metrics.results_dir() / f"{metrics.record_task()}_glued_walls.md"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n")
        print(f"walls table -> {out}")
    except OSError as e:
        print(f"walls table not written ({out}): {e}")
    print("\n" + table)


def _plan_chunks(plan: list, text, side: str):
    """A mix [(history, new tokens)] -> (requests, their token rows, the chunks of ``side``, new tokens)."""
    reqs = [Segment(10 + k, h, h + n) for k, (h, n) in enumerate(plan)]
    toks = {s.slot: text[_shift(s.slot - 10) : _shift(s.slot - 10) + s.end].long() for s in reqs}
    chunks = glued_chunks(reqs) if side == "glued" else lane_chunks(reqs)
    return reqs, toks, chunks, sum(s.length for s in reqs)


_RING = {
    "n": 0,
    "max_calls": 0,
}  # PB.7C: ring_indexer_score_dsa calls since the last layer ended; the most a layer made


def _ring_calls() -> dict:
    return _RING


def _counted(fn):
    """run fn() with every ring_indexer_score_dsa call counted (the calls a layer: between two layer ends)."""
    import ttnn

    # (the model's modules read the op's binding docstring at import to see which fork options are built: import
    # them before the op is wrapped, and give the wrapper the op's docstring)
    from models.demos.deepseek_v41_flash_d_p.tt import packed_glued, packed_onecall  # noqa: F401

    ring0 = ttnn.bringup.ring_indexer_score_dsa

    def ring(*a, **k):
        _RING["n"] += 1
        return ring0(*a, **k)

    ring.__doc__ = ring0.__doc__
    ttnn.bringup.ring_indexer_score_dsa = ring
    try:
        return fn()
    finally:
        ttnn.bringup.ring_indexer_score_dsa = ring0


def _measure(model, mesh, layers, case: str, side: str, plan: list, text, cap: int, dram, proj, fresh=None) -> dict:
    """One case on one side: every chunk of the plan timed (median of the warm runs a chunk), fresh slot states for
    every pass (the requests' histories are the states' zeros; the chunks of one pass continue each other).
    ``fresh`` (PB.7C, agentx cases): {seed: mix}; every timed run is a new mix instead (no layout repeats)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import history_gather
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import ids_from_input, make_input, run_chunk

    W = _walls_mod()
    reqs, toks, chunks, new = _plan_chunks(plan, text, side)
    runs = int(os.environ.get("GLUED_WALL_RUNS", "") or min(WALL_RUNS, 5))

    def one_pass(on_chunk=None, profile_chunk=None, chunks=chunks, toks=toks):
        """Every chunk of the plan once on fresh slot states (made at a request's first chunk, freed after its last)
        -> [wall ms a chunk]."""
        last = {}  # slot -> the last chunk that holds it (its state is freed after it: at most a chunk's slots live)
        for j, d in enumerate(chunks):
            for _, s in d.live():
                last[s.slot] = j
        states = {}
        walls = []
        try:
            for j, desc in enumerate(chunks):
                live = [s for _, s in desc.live()]
                for s in live:
                    if s.slot not in states:
                        states[s.slot] = model.new_state(cap)
                ids = [toks[s.slot][s.start : s.end] for s in live]
                if not getattr(desc, "glued", False):
                    ids = [None if s is None else toks[s.slot][s.start : s.end] for s in desc.segments]
                inp = make_input(mesh, desc, ids)
                sts = [None if s is None else states[s.slot] for s in desc.segments]
                model.sync()
                prof = profile_chunk == j
                if prof:
                    profiler.enable(mesh, rt=True)
                    history_gather.TRACE = []
                    profiler.set_layer(None)
                t0 = time.time()

                def on_layer(i, h):  # (PB.7C: the ring calls this layer made)
                    _RING["max_calls"] = max(_RING["max_calls"], _RING["n"])
                    _RING["n"] = 0
                    if on_chunk is not None:
                        on_chunk(i, h)

                try:
                    nat = ids_from_input(mesh, inp, desc)
                    _RING["n"] = 0
                    _counted(lambda: run_chunk(model, desc, nat, sts, layers, last=False, on_layer=on_layer, inp=inp))
                    model.sync()
                    walls.append((time.time() - t0) * 1e3)
                    if prof:
                        profiler.signpost("end")
                        profiler.finish_rt()
                        walls.append((profiler.result(), list(history_gather.TRACE)))
                finally:
                    if prof:
                        history_gather.TRACE = None
                        profiler.disable()
                    ttnn.deallocate(inp)
                for s in live:
                    if last[s.slot] == j:
                        states.pop(s.slot).free()
        finally:
            for st in states.values():
                st.free()
        return walls

    from models.demos.common.bringup.testing.profile import cache_entries

    extra = {}
    with packed("on"), glued("on" if side == "glued" else "off"):
        if fresh is not None:  # PB.7C: every timed run a new mix; the builds the timed runs add are the metric
            warm_seeds, timed_seeds = _fresh_seeds(case)
            print(f"CASE {case} [{side}] fresh layouts: warm-up seeds {warm_seeds}, timed seeds {timed_seeds}")
            for sd in warm_seeds:
                _, tk, ch, _ = _plan_chunks(fresh[sd], text, side)
                one_pass(chunks=ch, toks=tk)
            _ring_calls()["max_calls"] = 0
            e0 = cache_entries(mesh)
            walls_all, new_all, nch, per_mix, reps = [], 0, 0, [], []
            for sd in timed_seeds:
                _, tk, ch, nw = _plan_chunks(fresh[sd], text, side)
                ws = one_pass(chunks=ch, toks=tk)
                # (attempt 3: every timed mix once more right after its fresh run: the same layouts, so whatever a
                # layout leaves warm on the host is warm; fresh - repeat over every timed chunk = the host ms a new
                # layout costs. The repeats do not touch the fresh walls: the next mix is a new layout again)
                rp = one_pass(chunks=ch, toks=tk)
                walls_all += ws
                reps += rp
                new_all += nw
                nch += len(ch)
                per_mix.append((sd, len(ch), round(sum(ws), 1), nw))
            e1 = cache_entries(mesh)
            builds = -1 if e0 is None or e1 is None else e1 - e0
            diffs = [a - b for a, b in zip(walls_all, reps)]
            extra = dict(
                builds=builds,
                per_mix=per_mix,
                new_layout_ms=sum(diffs) / max(1, len(diffs)),
                new_layout_ms_median=sorted(diffs)[len(diffs) // 2] if diffs else 0.0,
                repeat_tok_s=new_all / (sum(reps) / 1e3),
                ring_calls=_ring_calls().get("max_calls", 0),
            )
            med, wall, new, chunks_n = walls_all, sum(walls_all), new_all, nch
            reqs = [r for sd in timed_seeds for r in _plan_chunks(fresh[sd], text, side)[0]]
            # (the paths / profile below: a new mix's chunks, chunk 0 profiled on a layout not seen before)
            _, toks, chunks, _ = _plan_chunks(agentx_mix(1000 * int(case[len("agentx_m") :]) + 99), text, side)
        else:
            _ring_calls()["max_calls"] = 0
            w = warm_up(mesh, lambda: round(sum(one_pass()), 1))
            assert w["warm"], f"{case} {side}: not warm after {len(w['walls'])} runs"
            per = []
            misses_allowed(mesh, False)
            try:
                for _ in range(runs):
                    per.append(one_pass())
            finally:
                misses_allowed(mesh, True)
            med = [sorted(p[j] for p in per)[len(per) // 2] for j in range(len(chunks))]
            wall = sum(med)
            chunks_n = len(chunks)
            extra = dict(ring_calls=_ring_calls().get("max_calls", 0))
        # one profiled pass: the device time and the gathers of the plan's first chunk
        got = one_pass(profile_chunk=0, chunks=chunks, toks=toks)
        prof, trace = got[1]
        dev = W._device_split(prof)
        ser = W._serial_gathers(prof, trace, set())
        peak = None
        if dram is not None and case == "6x_p1m":
            peak = W._peak_run(dram, lambda on_layer: one_pass(on_chunk=on_layer, chunks=chunks, toks=toks))
    paths = ["lanes" if not getattr(d, "glued", False) or d.as_lanes() is not None else "glued" for d in chunks]
    pre = "glued" if side == "glued" else "lanes"
    sfx = f"_{case}"
    tok_s = new / (wall / 1e3)
    W._record_gate_name(f"{pre}_serving_tok_s" + sfx, round(tok_s, 1))
    metrics.record(f"{pre}_wall_ms" + sfx, round(wall, 1))
    metrics.record(f"{pre}_chunk_walls_ms" + sfx, [round(x, 1) for x in med])
    metrics.record(f"{pre}_chunks" + sfx, chunks_n)
    metrics.record(f"{pre}_device_ms_chunk0" + sfx, round(dev["slowest"], 2))
    metrics.record(f"{pre}_path" + sfx, "/".join(paths))
    metrics.record(f"{pre}_rows_real_frac" + sfx, round(new / (CHUNK * chunks_n), 4))
    if side == "glued":  # (PB.7C: the indexer's ring calls a layer of the glued side's chunks)
        W._record_gate_name("glued_indexer_ring_calls_per_layer" + sfx, int(extra.get("ring_calls", 0)))
    if "builds" in extra:
        if side == "glued":
            W._record_gate_name("glued_program_builds_after_warmup" + sfx, int(extra["builds"]))
        metrics.record(f"{pre}_program_builds_after_warmup" + sfx, int(extra["builds"]))
        if side == "glued":  # (attempt 3: gated, glued_new_layout_ms* <= 1.5)
            W._record_gate_name("glued_new_layout_ms" + sfx, round(extra["new_layout_ms"], 2))
        else:
            metrics.record(f"{pre}_new_layout_ms" + sfx, round(extra["new_layout_ms"], 2))
        metrics.record(f"{pre}_new_layout_ms_median" + sfx, round(extra["new_layout_ms_median"], 2))
        metrics.record(f"{pre}_serving_tok_s_repeat" + sfx, round(extra["repeat_tok_s"], 1))
        metrics.record(f"{pre}_fresh_mixes" + sfx, [list(m) for m in extra["per_mix"]])
    if case.startswith("n"):
        metrics.record(f"glued_wall_ms_{case}", round(wall, 1))
    for lane, ms in ser["g"].items():
        metrics.record(f"{pre}_gather_ms{sfx}_seg{lane}", round(ms, 3))
        metrics.record(f"{pre}_gather_exposed_ms{sfx}_seg{lane}", round(ms, 3) if paths[0] == "glued" else -1)
    dr = None
    if peak is not None:
        xb = 0
        if proj is not None:
            st = model.new_state(cap)
            try:
                xb, _ = proj.bytes(st, len(chunks[0].segments))
            finally:
                st.free()
        meas, dep, capg = peak[0] / GIB, (peak[0] + xb) / GIB, dram.capacity / GIB
        dr = (meas, dep)
        metrics.record("glued_dram_gib" + sfx, round(dep, 3))
        metrics.record("glued_dram_measured_gib" + sfx, round(meas, 3))
        W._record_gate_name("glued_dram_headroom_gb" + sfx, round(32.0 - dep, 3))
        metrics.record("glued_dram_headroom_alloc_gib" + sfx, round(capg - dep, 3))
        print(
            f"CASE {case} DRAM a chip: peak {meas:.3f} GiB measured ({peak[1]}), 40-layer {dep:.3f} GiB, 32 - it = {32 - dep:.3f}"
        )
    print(
        f"CASE {case} [{side}] {len(reqs)} requests, {new} new tokens in {chunks_n} chunks ({'/'.join(paths)}): "
        f"walls {[round(x, 1) for x in med]} = {wall:.1f} ms, {tok_s:.0f} new tok/s; chunk 0 device {dev['slowest']:.1f} ms; "
        f"gathers chunk 0 per segment {ser['g']}; {extra}",
        flush=True,
    )
    return {
        "case": case,
        "side": side,
        "reqs": len(reqs),
        "new": new,
        "chunks": chunks_n,
        "segs": [len([s for s in d.segments if s is not None]) for d in chunks],
        "extra": extra,
        "paths": paths,
        "walls": med,
        "wall": wall,
        "tok_s": tok_s,
        "device0": dev["slowest"],
        "gathers": ser["g"],
        "dram": dr,
        "fill": new / (CHUNK * len(chunks)),
    }


def _table(rows: list, cap: int, plans: dict) -> str:
    head = [
        "case",
        "side",
        "requests",
        "new tokens",
        "chunks (segments)",
        "path",
        "real rows",
        "chunk walls ms",
        "wall ms",
        "new tok/s",
        "chunk 0 device ms",
        "chunk 0 gathers ms a segment (exposed)",
        "DRAM GiB 40-layer",
    ]
    lines = [
        "# Glued packed prefill walls (test_glued_walls.py, PB.7 / PB.7C)",
        "",
        f"8192-row chunks; slots at one capacity {cap}; layers 0-20 run (DECODER_SKIP=decode). Wall = engine device "
        "input -> ids readback -> sync after the last layer, per chunk the median of the warm runs; tok/s = real new "
        "tokens / the sum of the chunk walls. glued = PACKED_GLUED=on (a lane chunk runs the lane path; other glued "
        "chunks the per-segment row moves, PB.7 v1); lanes = the accepted 4-lane path (requests > 2048 in pieces).",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for r in rows:
        g = "; ".join(f"seg{k} {v:.2f}" for k, v in sorted(r["gathers"].items()))
        dr = "-" if r["dram"] is None else f"{r['dram'][1]:.2f} (headroom {32 - r['dram'][1]:.2f} of 32)"
        lines.append(
            f"| {r['case']} | {r['side']} | {r['reqs']} | {r['new']} | {r['chunks']} ({'/'.join(map(str, r['segs']))}) | "
            f"{'/'.join(sorted(set(r['paths'])))} | {r['fill']:.0%} | {', '.join(f'{x:.1f}' for x in r['walls'])} | "
            f"{r['wall']:.1f} | {r['tok_s']:.0f} | {r['device0']:.1f} | {g or '-'} | {dr} |"
        )
    by = {(r["case"], r["side"]): r for r in rows}
    lines += ["", "## Glued vs lanes (same requests)", ""]
    for c in sorted({r["case"] for r in rows}):
        a, b = by.get((c, "glued")), by.get((c, "lanes"))
        if a and b:
            lines.append(
                f"- {c}: glued {a['tok_s']:.0f} vs lanes {b['tok_s']:.0f} new tok/s ({a['tok_s'] / b['tok_s'] - 1:+.1%})"
            )
    ns = [by[(f"n{k}", "glued")] for k in range(1, 7) if (f"n{k}", "glued") in by]
    if ns:
        lines += ["", "## Per-segment overhead (139264, 8192 / k rows a segment)", ""]
        for r in ns:
            lines.append(f"- {r['case']}: {r['wall']:.1f} ms ({'/'.join(r['paths'])})")
    fr = [r for r in rows if r.get("extra", {}).get("builds") is not None]
    if fr:
        lines += [
            "",
            "## Fresh AgentX layouts (PB.7C: every timed run a new mix, no layout repeats)",
            "",
            "| case | side | timed mixes (seed, chunks, wall ms, new tokens) | fresh new tok/s | repeated-layout new tok/s "
            "(the last mix again) | host ms a new layout (per chunk) | program builds after warm-up | indexer ring "
            "calls a layer |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in fr:
            x = r["extra"]
            lines.append(
                f"| {r['case']} | {r['side']} | {x['per_mix']} | {r['tok_s']:.0f} | {x['repeat_tok_s']:.0f} | "
                f"{x['new_layout_ms']:.2f} | {x['builds']} | {x.get('ring_calls', '-')} |"
            )
    lines += ["", "AgentX mixes (history, new tokens; agentx cases: the first timed mix):"]
    for c, p in plans.items():
        if c.startswith("agentx_m"):
            lines.append(f"- {c}: {p}")
    return "\n".join(lines)
