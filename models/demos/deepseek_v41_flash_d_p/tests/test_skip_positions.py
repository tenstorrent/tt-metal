# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.34 (OWNER 2026-11-15): the realistic prefill perf measurement under DECODER_SKIP=decode (P.51), no model change.

The gate runs it with DS41_DECODER_SKIP=decode (layers 0-19 in full, layer 20 only its KV / index head, layers 21-39
never); every other switch at its shipped default: DS41_ENGRAM_TABLE is set to its shipped default (resident) when the
environment leaves it unset, since pytest otherwise reads F222's test value (mmap, DS41-245), which changes the wall.
A model loaded with another DECODER_SKIP (a bring-up batch's, FIX.29) is run as decode through ``use_decoder_skip``
(FIX.28: host lists only) and put back afterwards.

Points p (each a chunk start, chunk 5120 on galaxy_8x4): SKIP_POS (default 51200, 128000, 256000, 512000, 768000,
1024000). At each point, two chunks, each timed on its own:

    steady    begin_chunk(tokens[p : p + chunk], p, last=False): a chunk that is not the request's last, no replay,
              write_start = p; its windows carried from the chunk before (the real one: see below)
    resumed   begin_chunk(tokens[p : p + chunk], p, last=False, write_start=p + 128): the first chunk of a request that
              resumes from a cached prefix: its first 128 rows are encoder replay (layers 0-19, every window truncated at
              p, no history row below p + 128 written), 4992 rows new

How the chunk is preceded: every timed iteration (warm-up, timed, profiled) first runs the REAL previous chunk,
tokens[p - chunk : p] at p - chunk with last=False (untimed, synced), then the timed chunk. So the timed chunk never
follows itself: its window carry (steady) is the previous chunk's tail, the engram tail is the previous chunk's last
ids, and any state a model learns from the chunk before (the replica branch's MoE replicas) comes from a different
chunk, as in a real prefill. No steady_state() (F236) is called: the carry path is the real one.

Long-range caches at the point: as test_profile at BRINGUP_POS (testing/profile.PositionGolden, F230): a fresh state
sized p + chunk (the history caches are the fresh state's zeros), random token ids (seed 0; the work does not depend on
the values, DS41-228: not a routing-dependent measurement). No 1M-token prefill.

Memory: the model is loaded with its position tables (RoPE, window / sparse tables) for the longest point (target.seq
raised in memory, as test_profile does for BRINGUP_POS; never past the deployment's deploy.seq, 1,044,480 on
galaxy_8x4). Before the points the test checks each point against the loaded max_seq and records the allocator's DRAM per chip after the load and after each point's
state allocation and runs (dram_*_gib_p<p>; tests/bringup/test_dram.Dram). A point that does not fit (past max_seq, or
an allocation failure: "Out of Memory" / "Not enough space to allocate") is recorded fits_p<p> = 0 with its reason and
printed, never skipped silently. Anything else, a device hang (dispatch TIMEOUT) above all, fails the test (FIX.36).

Per point and kind (steady / resumed):
    <kind>_wall_ms_p<p>        warm clean wall: first host call (begin_chunk) to the sync after the last layer, the
                               median of WALL_RUNS (10) timed runs with program-cache misses refused, after
                               testing/profile.warm_up (F196 / F200: until no new program and the walls settled);
                               <kind>_wall_ms_spread_p<p>, <kind>_wall_ms_runs_p<p>, <kind>_wall_settled_p<p>
    <kind>_device_ms_p<p>      one more run under the real-time profiler (F155): the slowest chip's total device time
                               (max over chips of its programs' time); <kind>_device_ms_secmax_p<p> = test_profile's
                               device_ms_total (the slowest chip per section, summed); <kind>_device_ms_<phase>_p<p>
                               and the dotted sub-sections as test_profile's device_ms_<phase>; the ops of begin_chunk
                               and the embedding (before test_profile's first section) are the section chunk_input
    <kind>_tok_s_p<p>          tokens / wall: 5120 steady, 4992 (the new rows) resumed
    <kind>_host_ms_p<p>        wall - device (slowest chip)
and skip_positions_table: the markdown table of every point (also printed).

    PYTHONPATH=$PWD DS41_DECODER_SKIP=decode scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_skip_positions.py
"""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import (
    WALL_RUNS,
    PositionGolden,
    cache_entries,
    misses_allowed,
    settled,
    warm_up,
)
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True  # FIX.29: takes the model of a bring-up batch process (core/batch.shareable)
BATCH_ITEM_VARS = ("SKIP_POS",)
BATCH_POS_VARS = ("SKIP_POS",)  # F288: the plan's model is loaded for the furthest point (core/batch.hints)
BATCH_LAYERS = "executed"  # F285: a performance measurement: the layers it executes, in the resident runner  # F279: per-run (the points measured), not the model's configuration
POINTS = (51200, 128000, 256000, 512000, 768000, 1024000)
REPLAY = 128  # the encoder replay rows of a resumed request (DeepSeek's Encoder SWA Bounded Replay, n_win 128)
GIB = float(1 << 30)


def _points() -> list[int]:
    v = os.environ.get("SKIP_POS", "").strip()
    return [int(x) for x in v.split(",") if x.strip()] if v else list(POINTS)


HANG_MARKS = ("timeout", "potential hang", "timed out")
ALLOC_MARKS = ("out of memory", "not enough space to allocate", "allocation failed", "cannot allocate")


def _alloc_failure(e: BaseException) -> str | None:
    """FIX.36: the reason line when ``e`` is a device allocation failure ('does not fit'), else None (re-raise).

    The whole message is read, never only its first line: a dispatch hang's first line is
    "TT_THROW @ .../system_memory_manager.cpp:806: tt::exception" (the word "memory" is the file name) and its second
    "TIMEOUT: device timeout, potential hang detected"; FIX.34's first-line "memory" / "alloc" test filed two hangs
    (P.72, 2026-11-17) as fits_p<p> = 0 and exited 0. A message with a hang mark is never an allocation failure."""
    text = str(e)
    low = text.lower()
    if any(m in low for m in HANG_MARKS):
        return None
    for line in text.splitlines():
        if any(m in line.lower() for m in ALLOC_MARKS):
            return line.strip()
    return None


def _dram(mesh):
    """tests/bringup/test_dram.Dram (allocator bytes per chip), or None when the mesh has no memory view."""
    try:
        from models.demos.deepseek_v41_flash_d_p.tests.bringup.test_dram import Dram

        return Dram(mesh)
    except Exception as e:  # noqa: BLE001
        print(f"DRAM view unavailable: {e}")
        return None


def _device_split(prof: dict) -> dict:
    """The rt profile of one run -> slowest-chip total, test_profile's device_ms_total, per-phase and sub-section ms."""
    per_chip = defaultdict(float)
    for d in prof["kernel_ns_dev"].values():
        for c, ns in d.items():
            per_chip[c] += ns
    phases = defaultdict(float)
    for sec, ns in prof["kernel_ns"].items():
        phases[profiler.phase_of(sec)] += ns
    subs = {re.sub(r"[^A-Za-z0-9]+", "_", sec): ns for sec, ns in prof["kernel_ns"].items() if "." in sec}
    return {
        "slowest": max(per_chip.values()) / 1e6 if per_chip else 0.0,
        "slowest_chip": max(per_chip, key=per_chip.get) if per_chip else -1,
        "secmax": sum(prof["kernel_ns"].values()) / 1e6,
        "phases": {k: v / 1e6 for k, v in phases.items()},
        "subs": {k: v / 1e6 for k, v in subs.items()},
        "programs": sum(prof["programs"].values()),
    }


@mesh_parametrize(timeout_s=4 * 3600)
def test_skip_positions(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    assert settings.get("DECODER_SKIP") == "decode", (
        "FIX.34 measures DECODER_SKIP=decode: run with DS41_DECODER_SKIP=decode "
        f"(got {settings.get('DECODER_SKIP')})"
    )
    if os.environ.get(settings.env_name("ENGRAM_TABLE")) is None:  # (the shipped default, not F222's test value)
        monkeypatch.setenv(settings.env_name("ENGRAM_TABLE"), str(settings.table["ENGRAM_TABLE"].default))
    print(f"ENGRAM_TABLE={settings.get('ENGRAM_TABLE')}")
    chunk = int(S.get("target.chunk"))
    points = _points()
    full = list(S.layers())
    assert full == list(range(S.num_layers)), full
    # F282: no chunk here is a request's last, so under decode layers 21-39 never run: only 0-20 are loaded and looped
    # (testing/batch.run_layers; the same device work as the full list; BRINGUP_RUN_LAYERS=all loads every layer)
    layers = batch.run_layers(S, full, last_chunk=False)

    # The positions the load's tables cover (RoPE, window / sparse tables: load time, max_seq): the longest point, as
    # test_profile sizes them for BRINGUP_POS (target.seq, in memory), never past the deployment's context deploy.seq
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    want = max([p + chunk for p in points if p + chunk <= deploy] + [int(S.data["target"]["seq"])])
    monkeypatch.setitem(S.data["target"], "seq", want)
    t0 = time.time()
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    print(f"model ready in {time.time() - t0:.1f} s (load {getattr(model, 'load_seconds', 0.0):.1f} s)")
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    assert model.skip_mode == "decode" and model.decoder, (model.skip_mode, model.decoder)
    print(f"DECODER_SKIP=decode: decoder {model.decoder[0]}..{model.decoder[-1]}, kv-only {sorted(model.kv_only)}")

    dram = _dram(mesh_device)
    if dram is not None:
        metrics.record("dram_capacity_gib", round(dram.capacity / GIB, 3))
        metrics.record("dram_load_gib_max", round(dram.max() / GIB, 3))
        print(f"DRAM after load: {dram.max() / GIB:.2f} of {dram.capacity / GIB:.2f} GiB a chip ({dram.source})")

    # The memory check first (owner): a point must lie inside the model's position tables (max_seq, load time)
    fits = {}
    for p in points:
        if p < chunk or p % 32:
            fits[p] = f"start {p}: needs p >= {chunk} (a previous chunk) and 32-aligned"
        elif p + chunk > model.max_seq:
            fits[p] = (
                f"p + chunk = {p + chunk} > the model's max_seq {model.max_seq} (position tables sized at load; the "
                f"deployment's context deploy.seq is {deploy})"
            )
        else:
            fits[p] = None
    print("memory / position check: " + "; ".join(f"p{p}: {'fits' if r is None else r}" for p, r in fits.items()))

    rows = []
    try:
        for p in points:
            if fits[p] is not None:
                metrics.record(f"fits_p{p}", 0, reason=fits[p])
                print(f"POINT p{p} NOT RUN: {fits[p]}")
                rows.append((p, None, None, fits[p]))
                continue
            try:
                res = _point(model, mesh_device, layers, p, chunk, dram)
            except RuntimeError as e:  # (an allocation failure: reported with its reason, the next point still runs)
                why = _alloc_failure(e)
                if why is None:  # FIX.36: a hang (dispatch TIMEOUT) or any other error fails the test
                    raise
                msg = why[:300]
                metrics.record(f"fits_p{p}", 0, reason=msg)
                print(f"POINT p{p} DOES NOT FIT: {msg}")
                rows.append((p, None, None, msg))
                continue
            metrics.record(f"fits_p{p}", 1)
            rows.append((p, res["steady"], res["resumed"], None))
    finally:
        if loaded != "decode":
            model.use_decoder_skip(loaded)

    table = _table(rows, chunk)
    metrics.record("skip_positions_table", table)
    print("\n" + table)
    assert all(r[1] is not None for r in rows if r[0] == points[0]), "the first point did not run"


def _point(model, mesh, layers, p: int, chunk: int, dram) -> dict:
    g = PositionGolden(S, p, chunk)  # (random ids, seed 0, as test_profile's BRINGUP_POS)
    ids = g.tokens()
    prev_t, cur_t = ids[p - chunk : p], ids[p : p + chunk]
    state = model.new_state(p + chunk)
    if dram is not None:
        metrics.record(f"dram_state_gib_p{p}", round(dram.max() / GIB, 3))
    out = {}
    try:

        def run(tokens, start, write_start=None, section=False):
            if section:
                profiler.signpost("chunk_input")
            model.begin_chunk(tokens, start, last=False, write_start=write_start)
            h = model.embed(tokens)
            for i in layers:
                h2 = model.layer(i, h, start, state)
                model.free(h)
                h = h2
            model.free(h)

        for kind, ws, new in (("steady", None, chunk), ("resumed", p + REPLAY, chunk - REPLAY)):

            def one_run():
                run(prev_t, p - chunk)  # the real previous chunk, untimed
                model.sync()
                t0 = time.time()
                run(cur_t, p, ws)
                model.sync()
                return round((time.time() - t0) * 1e3, 1)

            w = warm_up(mesh, one_run)
            assert w["warm"], f"p{p} {kind}: not warm after {len(w['walls'])} runs ({w['entries'][-5:]})"
            walls = []
            misses_allowed(mesh, False)  # F196: every timed run hit the program cache
            try:
                for _ in range(WALL_RUNS):
                    walls.append(one_run())
            finally:
                misses_allowed(mesh, True)
            kept = sorted(walls)
            wall = kept[len(kept) // 2]

            run(prev_t, p - chunk)
            model.sync()
            n0 = cache_entries(mesh)
            profiler.enable(mesh, rt=True)
            try:
                profiler.set_layer(None)
                run(cur_t, p, ws, section=True)
                profiler.signpost("end")
                profiler.finish_rt()
                dev = _device_split(profiler.result())
            finally:
                profiler.disable()
            n1 = cache_entries(mesh)
            new_programs = -1 if n0 is None or n1 is None else n1 - n0

            pre = f"{kind}_"
            metrics.record(f"{pre}wall_ms_p{p}", wall)
            metrics.record(f"{pre}wall_ms_runs_p{p}", walls)
            metrics.record(f"{pre}wall_ms_spread_p{p}", round(kept[-1] - kept[0], 1))
            metrics.record(f"{pre}wall_warm_runs_p{p}", len(w["walls"]))
            metrics.record(f"{pre}wall_settled_p{p}", int(settled(w["walls"])))
            metrics.record(f"{pre}device_ms_p{p}", round(dev["slowest"], 2))
            metrics.record(f"{pre}device_slowest_chip_p{p}", dev["slowest_chip"])
            metrics.record(f"{pre}device_ms_secmax_p{p}", round(dev["secmax"], 2))
            metrics.record(f"{pre}device_programs_p{p}", dev["programs"])
            metrics.record(f"{pre}device_new_programs_p{p}", new_programs)
            for ph, ms in dev["phases"].items():
                metrics.record(f"{pre}device_ms_{ph}_p{p}", round(ms, 2))
            for sec, ms in dev["subs"].items():
                metrics.record(f"{pre}device_ms_{sec}_p{p}", round(ms, 2))
            metrics.record(f"{pre}tok_s_p{p}", round(new / (wall / 1e3), 1))
            metrics.record(f"{pre}host_ms_p{p}", round(wall - dev["slowest"], 1))
            out[kind] = {"wall": wall, "spread": round(kept[-1] - kept[0], 1), "tok_s": new / (wall / 1e3), **dev}
            print(
                f"POINT p{p} {kind}: wall {wall} ms (runs {walls}, spread {kept[-1] - kept[0]:.1f}, warm-up "
                f"{len(w['walls'])} runs, settled {settled(w['walls'])}), device {dev['slowest']:.2f} ms slowest chip "
                f"{dev['slowest_chip']} (sections' max summed {dev['secmax']:.2f}; "
                + ", ".join(f"{k} {v:.2f}" for k, v in sorted(dev["phases"].items(), key=lambda x: -x[1]))
                + f"), {new / (wall / 1e3):.0f} tok/s ({new} tokens), profiled run new programs {new_programs}",
                flush=True,
            )
        if dram is not None:
            metrics.record(f"dram_after_gib_p{p}", round(dram.max() / GIB, 3))
    finally:
        state.free()
    return out


def _table(rows, chunk: int) -> str:
    phases = sorted(
        {ph for _, st, rs, _ in rows for r in (st, rs) if r for ph in r["phases"]},
        key=lambda ph: -max((r["phases"].get(ph, 0.0) for _, st, rs, _ in rows for r in (st, rs) if r), default=0),
    )
    head = ["point", "chunk", "tokens", "wall ms (spread)", "device ms", "host ms", "tok/s"] + [
        f"{ph} ms" for ph in phases
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for p, st, rs, why in rows:
        if why is not None:
            lines.append(f"| {p} | - | - | not run: {why} |" + " |" * (len(head) - 4))
            continue
        for kind, r, n in (("steady", st, chunk), ("resumed", rs, chunk - REPLAY)):
            cells = [
                str(p),
                kind,
                str(n),
                f"{r['wall']:.1f} ({r['spread']:.1f})",
                f"{r['slowest']:.1f}",
                f"{r['wall'] - r['slowest']:.1f}",
                f"{r['tok_s']:.0f}",
            ] + [f"{r['phases'].get(ph, 0.0):.1f}" for ph in phases]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
