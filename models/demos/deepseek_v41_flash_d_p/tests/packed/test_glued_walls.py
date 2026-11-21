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

PB.9F (OWNER 2026-11-20 19:00, "I absolutely need an honest measurement"): WALLS_FRESH=1 runs the FRESH cases only
(tests/packed/fresh_common.py): agentx_m1..3 on both sides (glued and the fixed lanes, the same mixes) and 5x1600
(five 1600-token requests, each its own AgentX history draw: glued), every timed run a NEW mix / draw (seeds logged) in
NEW serving slots (the contract's caches; warm-up, timed and probe draws: warm-up slots disjoint from the others), each
timed run again right after (the warm-repeated number), WALLS_FRESH_PROF timed runs' repeats profiled chunk by chunk
(device ms), WALLS_FRESH_PROBE more draws under the build probe (the ops that built); program-cache misses allowed and
counted (program_builds_after_warmup_<side>_<case>: the fresh runs' entries).
PB.9C (OWNER 2026-11-21, the performance LADDER): LADDER_POS=<histories> runs the ladder instead of the cases
(tests/packed/ladder_common.py): LADDER_CASES=4x2k,5x1600 (paths glued4x2k: four 2048-token segments, one lane-shaped
glued chunk; glued5x1600: five 1600-token segments), every segment its own serving slot at history h, steady and
resumed, the real previous chunk first, warm_up + WALL_RUNS median, one profiled run (device ms), the serving wall
through runtime.prefill_packed with the D2H ack a layer and segment, and at 139264 the serving entry's host split
(glued4x2k). Metrics as test_packed_walls' ladder (ladder_<path>_*); tables: the metrics ladder_<path>_table.

PB.9X: serving's glued warm-up first (tt/packed_device.warm_glued_chunks, adapter.compile's set); + fresh cases
4x2k_p139264 (four 2048-token requests at 139264 + 64 j, |j| <= 256: one lane-shaped glued chunk; its fresh tok/s is
glued_serving_tok_s_4x2k_p139264) and 6x1344 (six requests at AgentX histories); the lanes side's 4-lane chunks run as
glued chunks under PACKED_ROUTE=glued (shipped), the fixed-lane path under PACKED_ROUTE=lanes.
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
from models.demos.deepseek_v41_flash_d_p.tests.packed import fresh_common as _fc
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc
from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, GluedChunk, Segment
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

_pc = importlib.reload(_pc)  # (the resident runner keeps helper modules of an earlier request: take this file's)
_fc = importlib.reload(_fc)
CHUNK, packed, glued = _pc.CHUNK, _pc.packed, _pc.glued

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = (
    "LADDER_POS",
    "LADDER_CASES",
    "LADDER_SERVING",
    "GLUED_WALL_CASES",
    "PACKED_WALL_CAP",
    "GLUED_WALL_RUNS",
    "GLUED_WARM_MIXES",
    "WALLS_FRESH",
    "WALLS_FRESH_SLOTS",
    "WALLS_FRESH_WARM",
    "WALLS_FRESH_RUNS",
    "WALLS_FRESH_PROBE",
    "WALLS_FRESH_PROF",
    "WALLS_FRESH_CASES",
)
BATCH_POS_VARS = ("LADDER_POS",)  # F288: the ladder's histories (the load's tables reach the furthest)
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
FRESH_FIXED = ["4x2k_p139264", "5x1600_p139264", "6x1344"]  # PB.9F / PB.9X: the fresh fixed-shape cases (glued)


def _fresh_seeds(case: str) -> tuple:
    """PB.7C: (warm-up seeds, timed seeds) of an agentx case: every run a new mix (no layout repeats). PB.9F
    (WALLS_FRESH=1): also 5x1600 (k = 0)."""
    k = _case_k(case)
    runs = int(os.environ.get("GLUED_WALL_RUNS", "") or min(WALL_RUNS, 5))
    warm = int(os.environ.get("GLUED_WARM_MIXES", "") or WARM_MIXES)
    return [1000 * k + 900 + w for w in range(warm)], [1000 * k + r for r in range(runs)]


def _case_k(case: str) -> int:
    """A fresh case's seed block: agentx_m<k> k; 5x1600 0 (PB.9F's seeds); PB.9X's cases 7, 8."""
    if case.startswith("agentx_m"):
        return int(case[len("agentx_m") :])
    return {"4x2k_p139264": 7, "6x1344": 8}.get(case, 0)


def _probe_seeds(case: str) -> list:
    """PB.9F: the build probe's draws (new mixes, never timed)."""
    k = _case_k(case)
    return [1000 * k + 500 + j for j in range(_fc.knob("WALLS_FRESH_PROBE", 3))]


def _fresh_mix(case: str, seed: int) -> list:
    """PB.9F: a fresh draw of ``case``: an AgentX mix, or 5x1600's five requests at AgentX history draws."""
    if case.startswith("agentx_m"):
        return agentx_mix(seed)
    if case == "5x1600_p139264":
        d = _fc.Draw(seed)
        return [(d.history(), 1600) for _ in range(5)]
    if case == "4x2k_p139264":  # PB.9X: four 2048-token requests near 139264 (139264 + 64 j, |j| <= 256: new
        # positions every draw), one lane-shaped glued chunk (segments on chip boundaries)
        rnd = random.Random(seed)
        return [(P139 + 64 * rnd.randrange(-256, 257), 2048) for _ in range(4)]
    if case == "6x1344":  # PB.9X: six 1344-token requests at AgentX history draws (6 segments, misaligned)
        d = _fc.Draw(seed)
        return [(d.history(), 1344) for _ in range(6)]
    raise KeyError(case)


def _shift(k: int) -> int:
    """Request k's text offset (k mod 6 x SHIFT: a chunk's segments read different stretches of the session)."""
    return (k % 6) * SHIFT


def _cases() -> list:
    v = os.environ.get("GLUED_WALL_CASES", "").strip()
    if v:
        return [c for c in v.split(",") if c]
    if _fc.on():  # PB.9F: the fresh cases only (PB.9X: + 4 x 2k near 139264 and 6 segments)
        return _fc.cases(FRESH_FIXED + [f"agentx_m{k}" for k in range(1, AGENTX_MIXES + 1)])
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
    from models.demos.deepseek_v41_flash_d_p.tests.packed import ladder_common

    if ladder_common.on():  # PB.9C: the ladder at real fixed histories (module docstring)
        return _ladder(mesh_device, layers, monkeypatch)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cap = W._cap(deploy)
    want = max(cap, int(S.data["target"]["seq"]))
    monkeypatch.setitem(S.data["target"], "seq", want)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    cases = _cases()
    fresh_on = _fc.on()
    plans, fresh = {}, {}
    for c in cases:
        if c.startswith("agentx_m") or (fresh_on and c in FRESH_FIXED):
            warm_seeds, timed_seeds = _fresh_seeds(c)
            k = _case_k(c)
            more = (_probe_seeds(c) + [1000 * k + 99]) if fresh_on else []
            fresh[c] = {sd: _fresh_mix(c, sd) for sd in warm_seeds + timed_seeds + more}
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
    slots = _fc.Slots(model, mesh_device, cap) if fresh_on else None  # PB.9F: serving's slots
    try:
        if fresh_on:  # PB.9X: serving's warm-up of the glued layouts (adapter.compile's set), before any timed chunk
            _warm_glued(model, mesh_device, layers, slots, cap)
        for c in cases:
            sides = ("glued", "lanes") if c.startswith("agentx_m") or c in ("5x1600_p139264",) else ("glued",)
            if fresh_on and not c.startswith("agentx_m"):
                sides = ("glued",)  # (PB.9F: glued 5 x 1.6k; the fixed lanes' fresh 4 x 2k is test_packed_walls')
            for side in sides:
                if not fresh_on:
                    rows.append(
                        _measure(model, mesh_device, layers, c, side, plans[c], text, cap, dram, proj, fresh.get(c))
                    )
                    continue
                name = f"{side}_{c.replace('_p139264', '')}"
                try:
                    r = _measure(model, mesh_device, layers, c, side, plans[c], text, cap, dram, proj, fresh[c], slots)
                    rows.append(r)
                    _fc.record(name, r["extra"]["fresh_row"])
                except Exception as e:  # (owner: a path that fails on fresh positions is recorded, not fixed)
                    if _fc.is_hang(e):
                        raise
                    import traceback

                    traceback.print_exc()
                    _fc.record(name, _fc.failed(name, side, e))
    finally:
        if slots is not None:
            slots.free()
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


def _ladder(mesh, layers, monkeypatch) -> None:
    """PB.9C: the glued paths' ladders (tests/packed/ladder_common.packed_ladder) through the serving runtime's slots
    (adapter.build_runtime through the model's one load path; its KV cache of 5 slots at the walls' capacity)."""
    from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as SP
    from models.demos.deepseek_v41_flash_d_p.tests.packed import ladder_common as L

    L = importlib.reload(L)
    W = _walls_mod()
    cases = L.cases(["4x2k", "5x1600"])
    assert set(cases) <= set(L.CASES), f"unknown ladder cases {cases} (known: {sorted(L.CASES)})"
    route = "glued"  # (PB.9G: PACKED_ROUTE removed; every packed chunk runs glued)
    assert route == "glued", f"the glued ladder runs PACKED_ROUTE=glued (got {route})"
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cap = W._cap(deploy)
    want = max(cap, int(S.data["target"]["seq"]))
    monkeypatch.setitem(S.data["target"], "seq", want)
    nslots = max(len(L.CASES[c]) for c in cases)
    monkeypatch.setenv("WALLS_FRESH_SLOTS", str(nslots))  # (the runtime's KV cache: one slot a segment)
    runtime, adapter, hf, params = SP._runtime(mesh, layers, CHUNK, want)
    model = runtime.model
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    pos = L.positions()
    text = PositionGolden(S, (nslots - 1) * SHIFT + max(pos) + 2048 + CHUNK, CHUNK).tokens()
    kv = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf, params=params)
    print(f"LADDER glued: cases {cases}, histories {pos}, {nslots} slots at {want}", flush=True)
    done = []
    try:
        with packed("on"), glued("on"):
            for c in cases:
                path = f"glued{c}"
                lens = L.CASES[c]
                rows = L.packed_ladder(
                    runtime,
                    kv,
                    mesh,
                    layers,
                    path,
                    lens,
                    lambda segs: GluedChunk(CHUNK, tuple(segs)),
                    text,
                    SHIFT,
                    cap,
                    split=(c == "4x2k"),
                )
                L.ladder_table(path, f"{len(lens)} x {lens[0]}-token segments back to back (glued), each at h", rows)
                done.append(any(r is not None for _, r, _ in rows))
    finally:
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
    assert done and all(done), "a ladder path ran no history"


def _warm_glued(model, mesh, layers, slots, cap: int) -> None:
    """PB.9X: the finite glued warm-up serving runs at startup (tt/packed_device.warm_glued_chunks, as
    adapter.compile), in warm-pool slots 0.. of ``slots``, through run_chunk with the engine input."""
    import torch

    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import (
        ids_from_input,
        make_input,
        run_chunk,
        warm_glued_chunks,
    )

    t0 = time.time()
    chunks = warm_glued_chunks(CHUNK, list(range(min(6, slots.warm))), cap)
    with packed("on"), glued("on"):
        for desc in chunks:
            ids = [torch.zeros(s.length, dtype=torch.int64) for s in desc.segments]
            inp = make_input(mesh, desc, ids)
            nat = ids_from_input(mesh, inp, desc)
            run_chunk(model, desc, nat, [slots.state(s.slot) for s in desc.segments], layers, last=False, inp=inp)
            ttnn.deallocate(inp)
    model.sync()
    print(f"PB.9X glued warm-up: {len(chunks)} chunks in {time.time() - t0:.1f} s", flush=True)


def _plan_chunks(plan: list, text, side: str, slots: list | None = None):
    """A mix [(history, new tokens)] -> (requests, their token rows, the chunks of ``side``, new tokens). ``slots``
    (PB.9F): request k's slot (else 10 + k)."""
    sl = slots or [10 + k for k in range(len(plan))]
    reqs = [Segment(sl[k], h, h + n) for k, (h, n) in enumerate(plan)]
    toks = {sl[k]: text[_shift(k) : _shift(k) + s.end].long() for k, s in enumerate(reqs)}
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


def _measure(
    model, mesh, layers, case: str, side: str, plan: list, text, cap: int, dram, proj, fresh=None, slots=None
) -> dict:
    """One case on one side: every chunk of the plan timed (median of the warm runs a chunk), fresh slot states for
    every pass (the requests' histories are the states' zeros; the chunks of one pass continue each other).
    ``fresh`` (PB.7C, agentx cases): {seed: mix}; every timed run is a new mix instead (no layout repeats).
    ``slots`` (PB.9F, WALLS_FRESH=1: fresh_common.Slots): every request in a serving slot (warm-up mixes in the warm
    pool, the others in the timed pool), each timed mix's repeat profiled chunk by chunk for the first WALLS_FRESH_PROF
    mixes, the build probe over WALLS_FRESH_PROBE more mixes; the fresh row in the result's "fresh_row"."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import history_gather
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import ids_from_input, make_input, run_chunk

    W = _walls_mod()
    reqs, toks, chunks, new = _plan_chunks(plan, text, side)
    runs = int(os.environ.get("GLUED_WALL_RUNS", "") or min(WALL_RUNS, 5))

    devs = []  # (PB.9F: prof_all's device ms a chunk)

    def one_pass(on_chunk=None, profile_chunk=None, chunks=chunks, toks=toks, prof_all=False):
        """Every chunk of the plan once on fresh slot states (made at a request's first chunk, freed after its last)
        -> [wall ms a chunk]. PB.9F: with ``slots`` the requests' serving slots' states (kept); ``prof_all`` profiles
        every chunk (device ms into ``devs``)."""
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
                        states[s.slot] = model.new_state(cap) if slots is None else slots.state(s.slot)
                ids = [toks[s.slot][s.start : s.end] for s in live]
                if not getattr(desc, "glued", False):
                    ids = [None if s is None else toks[s.slot][s.start : s.end] for s in desc.segments]
                inp = make_input(mesh, desc, ids)
                sts = [None if s is None else states[s.slot] for s in desc.segments]
                model.sync()
                prof = profile_chunk == j or prof_all
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
                        if prof_all:
                            devs.append(W._device_split(profiler.result())["slowest"])
                        else:
                            walls.append((profiler.result(), list(history_gather.TRACE)))
                finally:
                    if prof:
                        history_gather.TRACE = None
                        profiler.disable()
                    ttnn.deallocate(inp)
                for s in live:
                    if last[s.slot] == j:
                        st = states.pop(s.slot)
                        if slots is None:
                            st.free()
        finally:
            if slots is None:
                for st in states.values():
                    st.free()
        return walls

    from models.demos.common.bringup.testing.profile import cache_entries

    extra = {}
    # PB.9X: the lanes side packs lane-wise (<= 2048 a piece, 4 a chunk); under PACKED_ROUTE=glued (the shipped value)
    # those headers run the glued path (PACKED_GLUED=on), under PACKED_ROUTE=lanes the fixed-lane path (PB.7's row)
    route = "glued"  # (PB.9G: PACKED_ROUTE removed; every packed chunk runs glued)
    with packed("on"), glued("on" if side == "glued" or route == "glued" else "off"):
        if fresh is not None:  # PB.7C: every timed run a new mix; the builds the timed runs add are the metric
            warm_seeds, timed_seeds = _fresh_seeds(case)
            print(f"CASE {case} [{side}] fresh layouts: warm-up seeds {warm_seeds}, timed seeds {timed_seeds}")
            import random as _random

            def slot_of(sd: int, phase: str):  # (PB.9F: the mix's requests' serving slots, distinct, seeded)
                if slots is None:
                    return None
                return slots.pick(len(fresh[sd]), phase, _random.Random(sd + 77))

            for sd in warm_seeds:
                _, tk, ch, _ = _plan_chunks(fresh[sd], text, side, slot_of(sd, "warm"))
                one_pass(chunks=ch, toks=tk)
            _ring_calls()["max_calls"] = 0
            e0 = cache_entries(mesh)
            walls_all, new_all, nch, per_mix, reps = [], 0, 0, [], []
            fr_rows, fr_rep, fr_prof, fr_builds, fr_draws = [], [], [], 0, []
            fr_pc, fr_preps = _fc.PrepCount(), [0, 0.0]
            nprof = _fc.knob("WALLS_FRESH_PROF", 2)
            for mi, sd in enumerate(timed_seeds):
                rq, tk, ch, nw = _plan_chunks(fresh[sd], text, side, slot_of(sd, "timed"))
                f0 = cache_entries(mesh)
                with fr_pc:  # (PB.9F: prepared programs built in the fresh run: no cache entry of their own)
                    p0, m0 = fr_pc.n, fr_pc.ms
                    ws = one_pass(chunks=ch, toks=tk)
                    fr_preps[0] += fr_pc.n - p0
                    fr_preps[1] += fr_pc.ms - m0
                f1 = cache_entries(mesh)
                built = -1 if f0 is None or f1 is None else f1 - f0
                fr_builds += max(built, 0)
                # (attempt 3: every timed mix once more right after its fresh run: the same layouts, so whatever a
                # layout leaves warm on the host is warm; fresh - repeat over every timed chunk = the host ms a new
                # layout costs. The repeats do not touch the fresh walls: the next mix is a new layout again)
                rp = one_pass(chunks=ch, toks=tk)
                if slots is not None:  # (PB.9F: the fresh row's chunks; the mix's new tokens on its first chunk)
                    fr_rows += [(w, nw if j == 0 else 0) for j, w in enumerate(ws)]
                    fr_rep += [(w, nw if j == 0 else 0) for j, w in enumerate(rp)]
                    if mi < nprof:
                        devs.clear()
                        one_pass(chunks=ch, toks=tk, prof_all=True)
                        fr_prof += [(a, b, d) for a, b, d in zip(ws, rp, devs)]
                    fr_draws.append(
                        {
                            "seed": sd,
                            "requests": [(s.slot, s.start, s.length) for s in rq],
                            "chunks": len(ch),
                            "fresh_ms": [round(x, 1) for x in ws],
                            "repeat_ms": [round(x, 1) for x in rp],
                            "built": built,
                            "prepared": fr_pc.n - p0,
                        }
                    )
                walls_all += ws
                reps += rp
                new_all += nw
                nch += len(ch)
                per_mix.append((sd, len(ch), round(sum(ws), 1), nw))
            e1 = cache_entries(mesh)
            builds = -1 if e0 is None or e1 is None else e1 - e0
            if slots is not None:  # PB.9X: PB.9F's definition (the fresh timed chunks' entries only: never the repeats
                # or the profiled passes, whose profiler programs build once in a fresh process)
                builds = fr_builds
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
            if slots is not None:  # PB.9F: the ops that built (new draws under the build probe, untimed)
                pseeds = _probe_seeds(case)
                pchunks = 0
                with _fc.BuildProbe(mesh) as pb:
                    for sd in pseeds:
                        _, tk, ch, _ = _plan_chunks(fresh[sd], text, side, slot_of(sd, "timed"))
                        one_pass(chunks=ch, toks=tk)
                        pchunks += len(ch)
                pb.print(f"{side} {case}")
                path = (
                    "glued (PACKED_GLUED=on)"
                    if side == "glued"
                    else "4-lane packing sent as glued chunks (PACKED_ROUTE=glued)"
                    if route == "glued"
                    else "fixed 4 x 2048 lanes (PACKED_ROUTE=lanes, PACKED_GLUED=off)"
                )
                extra["fresh_row"] = _fc.summarize(
                    f"{side}_{case.replace('_p139264', '')}",
                    path,
                    fr_rows,
                    fr_rep,
                    fr_prof,
                    fr_builds,
                    pb,
                    {
                        "seeds": {"warm": warm_seeds, "timed": timed_seeds, "probe": pseeds},
                        "draws": fr_draws,
                        "probe_chunks": pchunks,
                    },
                    tuple(fr_preps),
                )
            # (the paths / profile below: a new mix's chunks, chunk 0 profiled on a layout not seen before)
            k99 = 1000 * _case_k(case) + 99
            mix99 = fresh[k99] if k99 in fresh else agentx_mix(k99)
            _, toks, chunks, _ = _plan_chunks(mix99, text, side, slot_of(k99, "timed") if k99 in fresh else None)
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
        from models.demos.deepseek_v41_flash_d_p.tt.packed_device import as_run_desc

        paths = ["glued" if getattr(as_run_desc(d), "glued", False) else "lanes" for d in chunks]
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
        metrics.record(f"{pre}_mixes_timed" + sfx, [list(m) for m in extra["per_mix"]])  # (PB.9F: no "_fresh")
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
