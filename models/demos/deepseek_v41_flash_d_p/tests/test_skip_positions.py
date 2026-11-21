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

PB.9C (OWNER 2026-11-21, the ladder's serving column; LADDER_SERVING=0 turns it off): per point also the steady chunk
through THE SERVING ENTRY (the adapter's runtime over the same loaded model, adapter.build_runtime -> load_model; its
KV cache of one slot at the load's max_seq): runtime.prefill_chunk(engine input, kv, slot 0, p, p + chunk) with the
per-layer device-op ack tt-d-gen uses (outbound_socket_service_sync into a metadata-only D2HStreamService drained by a
LayerAckService: tests/packed/ladder_common.D2HAcks), the real previous chunk first, warm_up + 5 timed runs, the median:
serving_wall_ms_p<p>, serving_extra_ms_p<p> (- the harness steady wall). At 139264 (AgentX p50) the serving entry's
host work item by item vs the harness's (serving_host_split_p139264: ids readback, un-shuffle, note_tokens,
begin_chunk, reorder_ids, embed, layers' enqueue, D2H ack ops, device tail; also without any ack).

    PYTHONPATH=$PWD DS41_DECODER_SKIP=decode scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_skip_positions.py

PB.9F / MB.1 (OWNER 2026-11-20; PB.MB: MB.1's adapter path replaced PB.9F's run_chunk path): WALLS_FRESH=1 runs the FRESH walls
instead of the points (tests/fresh_common.py): the single-request chunk exactly as the serving engine drives it (the
adapter's runtime, tt/runners/adapter.DS41PrefillRuntime built as prefill_runner builds it, sharing the loaded model:
``runtime.prefill_chunk(engine input, kv, slot_id, actual_start, actual_end, global_write_start)``: the ids readback,
note_tokens / begin_chunk, every served layer to PREFILL_END's run end) with every timed chunk at a NEW position
(AgentX history draws, 64-aligned, >= a chunk) in a NEW serving slot (the adapter's KV cache of WALLS_FRESH_SLOTS
slots; warm-up and timed slots disjoint), program-cache misses allowed and counted. Kinds (case single<chunk>_<kind>):
steady (a full chunk, write_start = start), resumed (a full chunk, global_write_start = start + 128: the encoder
replay) and partial (a request's last chunk: actual_end = start + the last piece of an AgentX new-token draw, >= 64
rows). Each timed iteration runs the real previous chunk [p - chunk, p) in the same slot untimed first, as the points
do. Per kind: WALLS_FRESH_WARM warm-up draws, WALLS_FRESH_RUNS timed draws (each run again right after: the
warm-repeated number), WALLS_FRESH_PROF of them profiled (device ms), WALLS_FRESH_PROBE more draws under the build
probe (the ops that built). The engine's device input is made before the clock (the engine's H2D: not timed). The
model is loaded with the position tables of the walls' slot capacity (the deployment context in whole chunks).

    PYTHONPATH=$PWD BRINGUP_CHUNK=5120 DS41_ENGRAM_TABLE=resident DS41_DECODER_SKIP=decode WALLS_FRESH=1 \\
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_skip_positions.py
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
BATCH_ITEM_VARS = (
    "SKIP_POS",
    "WALLS_FRESH",
    "WALLS_FRESH_SLOTS",
    "WALLS_FRESH_WARM",
    "WALLS_FRESH_RUNS",
    "WALLS_FRESH_PROBE",
    "WALLS_FRESH_PROF",
    "WALLS_FRESH_CASES",
    "LADDER_SERVING",
)
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
    from models.demos.deepseek_v41_flash_d_p.tests import fresh_common

    fresh = fresh_common.on()
    from models.demos.deepseek_v41_flash_d_p.tests.packed import ladder_common as LC

    serve = LC.serving_on() and not fresh  # PB.9C: the serving entry's wall per point (module docstring)
    if fresh:  # MB.1: the walls' one capacity (the deployment context in whole chunks: one load, one max_seq)
        want = max(-(-deploy // chunk) * chunk, int(S.data["target"]["seq"]))
    elif serve:  # (the serving runtime's slots: the walls' one capacity, the deployment context in whole chunks, so
        # this unit and the packed walls / no-rebuild units share one load whichever runs first: F283, DS41-356)
        want = max(want, -(-deploy // chunk) * chunk)
    monkeypatch.setitem(S.data["target"], "seq", want)
    t0 = time.time()
    srv = None
    if fresh:  # MB.1: the serving runtime loads it, through the model's one load path (the batch's model in a batch)
        rt = _runtime(mesh_device, layers, chunk, want)
        model = rt[0].model
    elif serve:  # PB.9C: the same, one slot (the points' own states stay the harness's)
        monkeypatch.setenv("WALLS_FRESH_SLOTS", str(1 + LC.SPLIT_RUNS))  # (slot 0 + the split's fresh slots)
        rt = _runtime(mesh_device, layers, chunk, want)
        model = rt[0].model
    else:
        model = batch.device_model(S, mesh_device, layers, lm_head=False)
    print(f"model ready in {time.time() - t0:.1f} s (load {getattr(model, 'load_seconds', 0.0):.1f} s)")
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    assert model.skip_mode == "decode" and model.decoder, (model.skip_mode, model.decoder)
    print(f"DECODER_SKIP=decode: decoder {model.decoder[0]}..{model.decoder[-1]}, kv-only {sorted(model.kv_only)}")

    if fresh:
        try:
            _fresh_walls(rt, mesh_device, chunk, want)
        finally:
            if loaded != "decode":
                model.use_decoder_skip(loaded)
        return

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
    if serve:
        runtime, adapter, hf, params = rt
        kv = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf, params=params)
        acks = None
        try:
            acks = LC.D2HAcks(mesh_device)
        except Exception as e:  # noqa: BLE001 (recorded: the serving column then says why it is missing)
            print(f"SERVING: D2H ack service not available: {type(e).__name__}: {e}", flush=True)
            LC.record("serving_error", f"{type(e).__name__}: {e}"[:300])
        srv = (runtime, kv, acks)
    try:
        for p in points:
            if fits[p] is not None:
                metrics.record(f"fits_p{p}", 0, reason=fits[p])
                print(f"POINT p{p} NOT RUN: {fits[p]}")
                rows.append((p, None, None, fits[p]))
                continue
            try:
                res = _point(model, mesh_device, layers, p, chunk, dram, srv)
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
        if srv is not None:
            _free_serving(srv)
        if loaded != "decode":
            model.use_decoder_skip(loaded)

    table = _table(rows, chunk)
    metrics.record("skip_positions_table", table)
    print("\n" + table)
    assert all(r[1] is not None for r in rows if r[0] == points[0]), "the first point did not run"


def _free_serving(srv) -> None:
    import ttnn

    runtime, kv, acks = srv
    if acks is not None:
        err = acks.close()
        if err:
            metrics.record("serving_error", err)
            print(f"SERVING: D2H ack service: {err}", flush=True)
    for key in [k for k in runtime._states if k[0] == id(kv)]:
        try:
            runtime._states.pop(key).free()
        except Exception as e:  # noqa: BLE001
            print(f"state free: {e}")
    for t in kv.contract.by_config.values():
        if t.is_allocated():
            ttnn.deallocate(t)


def _point(model, mesh, layers, p: int, chunk: int, dram, srv=None) -> dict:
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
        if srv is not None and srv[2] is not None:  # PB.9C: the serving entry's wall (module docstring)
            out["serving"] = _serving_point(model, mesh, p, chunk, ids, srv, out, run, (prev_t, cur_t))
            out["steady"]["serving_wall"] = out["serving"]["wall"]
        if dram is not None:
            metrics.record(f"dram_after_gib_p{p}", round(dram.max() / GIB, 3))
    finally:
        state.free()
    return out


def _serving_point(model, mesh, p: int, chunk: int, ids, srv, out: dict, run, toks) -> dict:
    """PB.9C: the steady chunk at ``p`` through runtime.prefill_chunk with the D2H ack a layer (slot 0 of the
    runtime's KV cache; its real previous chunk first), warm_up + SERVING_RUNS timed, the median; at SPLIT_POS the host
    split (serving with ack, serving without any ack, the harness's run of the same chunk)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R
    from models.demos.deepseek_v41_flash_d_p.tests.packed import ladder_common as LC

    runtime, kv, acks = srv
    sp = int(tuple(mesh.shape)[0])
    saved = model.state  # (the adapter's _run makes its slot's state the active one: the harness never sets it)

    def srun(s: int, hs=None, ack: bool = True, slot: int = 0) -> float:
        e = s + chunk
        inp = C.payload_tensor(mesh, R.server_payload(ids[s:e].tolist(), s, e, chunk=chunk, sp=sp))  # (engine H2D)
        model.sync()
        if hs is not None:
            hs.begin()
        t0 = time.time()
        try:
            if ack:
                with acks:
                    runtime.prefill_chunk(
                        inp,
                        kv,
                        slot_id=slot,
                        actual_start=s,
                        actual_end=e,
                        d2h_service=acks.d2h,
                        metadata_msg=acks.record,
                    )
            else:
                runtime.prefill_chunk(inp, kv, slot_id=slot, actual_start=s, actual_end=e)
            t1 = time.time()
            model.sync()
            wall = (time.time() - t0) * 1e3
            if hs is not None:
                hs.add("device tail (sync after the last enqueue)", (time.time() - t1) * 1e3)
                hs.end(wall)
        finally:
            model.state = saved
            ttnn.deallocate(inp)
        missing = acks.drain()
        assert missing == 0, f"{missing} D2H acks not seen after the chunk ({acks.sent} sent)"
        return wall

    def one_run():
        srun(p - chunk)  # the real previous chunk, untimed
        return round(srun(p), 1)

    sv = LC.time_kind(mesh, one_run, runs=LC.SERVING_RUNS)
    steady = out["steady"]["wall"]
    metrics.record(f"serving_wall_ms_p{p}", sv["wall"])
    metrics.record(f"serving_wall_ms_runs_p{p}", sv["runs"])
    metrics.record(f"serving_extra_ms_p{p}", round(sv["wall"] - steady, 1))
    metrics.record("serving_acks_per_chunk", len(runtime.layer_ids))
    print(
        f"POINT p{p} serving entry (prefill_chunk + D2H ack a layer): wall {sv['wall']} ms (runs {sv['runs']}), "
        f"{sv['wall'] - steady:+.1f} ms over the harness steady wall {steady}",
        flush=True,
    )
    res = {**sv}
    if p == LC.SPLIT_POS:
        res["split"] = _single_split(model, runtime, kv, mesh, p, chunk, srun, run, ids, saved)
        metrics.record(f"serving_host_split_p{p}", res["split"])
        LC.print_split(f"single{chunk}", p, res["split"])
    return res


def _single_split(model, runtime, kv, mesh, p: int, chunk: int, srun, run, text, saved) -> dict:
    """PB.9C: the single-request chunk's host work item by item, LC.SPLIT_RUNS chunks a side (each after its real
    previous chunk), every item's median: the harness (begin_chunk on host ids, embed with its ids upload, the layers,
    the device tail) vs the serving entry with the D2H ack and without any (the ids readback, the un-shuffle,
    note_tokens, begin_chunk, reorder_ids, embed, the layers, the ack ops, the device tail)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tests.packed import ladder_common as LC
    from models.demos.deepseek_v41_flash_d_p.tt import cores as CORES
    from models.demos.deepseek_v41_flash_d_p.tt import model as MM

    out = {}
    for side in ("harness", "serving", "serving_noack", "harness_fresh", "serving_fresh"):
        hs = LC.HostSplit()
        walls = []
        try:
            hs.wrap(ttnn, "to_torch", "ids readback (to_torch, mesh column 0's SP rows)")
            hs.wrap(ttnn, "from_device", "ids readback (to_torch, mesh column 0's SP rows)")
            hs.wrap(MM, "from_torch", "ids upload (from_torch)")
            hs.wrap(CORES, "from_torch", "ids upload (from_torch)")
            hs.wrap(model, "begin_chunk", "begin_chunk (fields, engram tails + jobs submitted)", instance=True)
            hs.wrap(model, "embed", "embed (enqueue)", instance=True)
            hs.wrap(model, "layer", "layers (host enqueue, engram waits included)", instance=True)
            if not side.startswith("harness"):
                hs.wrap(runtime, "_host_ids", "ids un-shuffle (server order -> natural, host)", instance=True)
                for k in range(LC.SPLIT_RUNS + 1):
                    hs.wrap(runtime.state(kv, k), "note_tokens", "note_tokens (slot id history + engram tails)", True)
                hs.wrap(model, "reorder_ids", "reorder_ids (natural-order ids on the device: enqueue)", instance=True)
                hs.wrap(ttnn.experimental.deepseek_prefill, "outbound_socket_service_sync", "D2H ack ops (enqueue)")
            for j in range(LC.SPLIT_RUNS):
                # (_fresh: every chunk at a position not run before, p - 64 (j + 1); serving_fresh also in a slot of its
                # own, 1 + j: the fresh-traffic host cost PB.MB's fresh walls carried)
                q = p - 64 * (j + 1) if side.endswith("fresh") else p
                if side.startswith("harness"):
                    model.state = saved
                    run(text[q - chunk : q], q - chunk)  # the real previous chunk, untimed
                    model.sync()
                    hs.begin()
                    t0 = time.time()
                    run(text[q : q + chunk], q)
                    t1 = time.time()
                    model.sync()
                    w = (time.time() - t0) * 1e3
                    walls.append(w)
                    hs.add("device tail (sync after the last enqueue)", (time.time() - t1) * 1e3)
                    hs.end(w)
                else:
                    ack = side != "serving_noack"
                    slot = 1 + j if side.endswith("fresh") else 0
                    srun(q - chunk, ack=ack, slot=slot)
                    walls.append(srun(q, hs, ack=ack, slot=slot))
        finally:
            hs.restore()
        out[side] = {
            "wall": sorted(walls)[len(walls) // 2],
            "walls": [round(w, 2) for w in walls],
            "items": hs.items(),
            "calls": dict(hs.calls),
            "engram": LC._engram_ms(model),
        }
    return out


def _table(rows, chunk: int) -> str:
    phases = sorted(
        {ph for _, st, rs, _ in rows for r in (st, rs) if r for ph in r["phases"]},
        key=lambda ph: -max((r["phases"].get(ph, 0.0) for _, st, rs, _ in rows for r in (st, rs) if r), default=0),
    )
    head = ["point", "chunk", "tokens", "wall ms (spread)", "device ms", "host ms", "tok/s", "serving wall ms"] + [
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
                f"{r['serving_wall']:.1f}" if r.get("serving_wall") is not None else "-",
            ] + [f"{r['phases'].get(ph, 0.0):.1f}" for ph in phases]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------- MB.1 fresh walls (PB.9F)
FRESH_KINDS = ("steady", "resumed", "partial")
FRESH_SEED = {"steady": 1, "resumed": 2, "partial": 3}


def _runtime(mesh, layers, chunk: int, cap: int):
    """The serving runtime as prefill_runner builds it (adapter.build_runtime over ``layers`` at max_seq ``cap``; it
    loads through the model's one load path, adapter.load_model: in a batch the held model, never a second copy).
    Returns (runtime, adapter, hf config, params); the KV cache and compile come later (``_fresh_walls``)."""
    from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
    from models.demos.deepseek_v41_flash_d_p.tests import fresh_common as F
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C

    assert list(layers) == list(range(len(layers))), f"the served layers must be 0..n-1: {layers}"
    sp, tp = tuple(mesh.shape)
    adapter = get_adapter(C.adapter_name())
    hf = adapter.load_hf_config()
    hf.max_seq_len = cap
    params = PrefillRunParams(
        mesh_shape=(sp, tp),
        num_layers=len(layers),
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=cap,
        chunk_size=chunk,
        num_users=F.knob("WALLS_FRESH_SLOTS", 32),
        capacity_factor=8,
        num_links=2,
        gate_mode_name=adapter.default_gate_mode,
        kv_only_last_layer=True,
        weight_cache_path=adapter.weight_cache_path((sp, tp)),
    )
    runtime = adapter.build_runtime(mesh_device=mesh, hf_config=hf, params=params)
    assert runtime.layer_ids == list(layers), (runtime.layer_ids, list(layers))
    return runtime, adapter, hf, params


def _fresh_walls(rt, mesh, chunk: int, cap: int) -> None:
    """MB.1: the single-request chunk's FRESH walls, a case a kind (module docstring). ``rt``: ``_runtime``'s."""
    from models.demos.deepseek_v41_flash_d_p.tests import fresh_common as F

    runtime, adapter, hf, params = rt
    model = runtime.model
    hi = min(cap, model.max_seq) - chunk  # (the last start whose chunk fits the slot and the position tables)
    text = PositionGolden(S, hi, chunk).tokens()  # (hi + chunk tokens of the text: any draw's previous and own chunk)
    t0 = time.time()
    kv = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf, params=params)  # (the engine-owned caches)
    runtime.compile(kv)  # (the engine's cold chunk at startup)
    print(f"FRESH KV cache ({params.num_users} slots at {cap}) + runtime.compile in {time.time() - t0:.1f} s")
    slots = F.Slots(runtime, kv)
    try:
        for kind in F.cases(FRESH_KINDS):
            case = f"single{chunk}_{kind}"
            try:
                row = _fresh_kind(runtime, kv, mesh, chunk, kind, case, hi, text, slots)
            except Exception as e:  # (owner: a path that fails on fresh positions is recorded, not fixed)
                if F.is_hang(e):
                    raise
                import traceback

                traceback.print_exc()
                row = F.failed(case, "single (adapter prefill_chunk)", e)
            F.record(case, row)
    finally:
        slots.free()


def _fresh_kind(runtime, kv, mesh, chunk: int, kind: str, case: str, hi: int, text, slots) -> dict:
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tests import fresh_common as F
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R

    model = runtime.model
    sp = int(tuple(mesh.shape)[0])
    nwarm, nrun = F.knob("WALLS_FRESH_WARM", 4), F.knob("WALLS_FRESH_RUNS", 8)
    nprobe, nprof = F.knob("WALLS_FRESH_PROBE", 3), F.knob("WALLS_FRESH_PROF", 3)
    base = 100000 * FRESH_SEED[kind] + chunk
    seeds = {
        "warm": [base + 900 + w for w in range(nwarm)],
        "timed": [base + r for r in range(nrun)],
        "probe": [base + 500 + j for j in range(nprobe)],
    }
    print(f"FRESH {case}: seeds {seeds}", flush=True)

    def draw(seed: int, phase: str) -> dict:
        d = F.Draw(seed)
        p = d.history(lo=chunk, hi=hi)
        slot = slots.pick(1, phase, d.rnd)[0]
        if kind == "steady":
            end, ws, new = p + chunk, None, chunk
        elif kind == "resumed":
            end, ws, new = p + chunk, p + REPLAY, chunk - REPLAY
        else:  # a request's last chunk: the last piece of its new tokens (>= 64 rows: PREFILL_END=cap computes rows)
            while True:
                m = (d.new_tokens() - 1) % chunk + 1
                if m >= 64:
                    break
            end, ws, new = p + m, None, m
        return {"seed": seed, "slot": slot, "start": p, "end": end, "write_start": ws, "new": new}

    def run(dr: dict, prev: bool, prof: bool = False):
        if prev:  # the real previous chunk of the request (untimed)
            s, e, ws = dr["start"] - chunk, dr["start"], None
        else:
            s, e, ws = dr["start"], dr["end"], dr["write_start"]
        inp = C.payload_tensor(mesh, R.server_payload(text[s:e].tolist(), s, e, chunk=chunk, sp=sp))  # (engine H2D)
        model.sync()
        if prof:
            profiler.enable(mesh, rt=True)
            profiler.set_layer(None)
        t0 = time.time()
        dev = None
        try:
            runtime.prefill_chunk(inp, kv, slot_id=dr["slot"], actual_start=s, actual_end=e, global_write_start=ws)
            model.sync()
            wall = (time.time() - t0) * 1e3
            if prof:
                profiler.signpost("end")
                profiler.finish_rt()
                dev = _device_split(profiler.result())["slowest"]
        finally:
            if prof:
                profiler.disable()
            ttnn.deallocate(inp)
        return wall, dev

    for sd in seeds["warm"]:
        dr = draw(sd, "warm")
        run(dr, True)
        w, _ = run(dr, False)
        print(f"FRESH {case} warm-up {dr}: {w:.1f} ms", flush=True)
    fresh, repeat, prof, builds, draws = [], [], [], 0, []
    pc, preps = F.PrepCount(), [0, 0.0]  # (prepared programs built in the fresh runs: no cache entry of their own)
    for i, sd in enumerate(seeds["timed"]):
        dr = draw(sd, "timed")
        e0 = cache_entries(mesh)
        with pc:
            p0, m0 = pc.n, pc.ms
            run(dr, True)
            wf, _ = run(dr, False)
            preps[0] += pc.n - p0
            preps[1] += pc.ms - m0
        e1 = cache_entries(mesh)
        built = -1 if e0 is None or e1 is None else e1 - e0
        builds += max(built, 0)
        run(dr, True)  # the warm-repeated run: the same chunks again (positions, slot seen)
        wr, _ = run(dr, False)
        fresh.append((wf, dr["new"]))
        repeat.append((wr, dr["new"]))
        dev = None
        if i < nprof:
            run(dr, True)
            _, dev = run(dr, False, prof=True)
            prof.append((wf, wr, dev))
        draws.append(
            {
                **dr,
                "fresh_ms": round(wf, 1),
                "repeat_ms": round(wr, 1),
                "built": built,
                "prepared": pc.n - p0,
                "device_ms": dev,
            }
        )
        print(f"FRESH {case} timed {dr}: fresh {wf:.1f} ms (+{built} programs), repeated {wr:.1f} ms, device {dev}")
    with F.BuildProbe(mesh) as pb:
        for sd in seeds["probe"]:
            dr = draw(sd, "timed")
            run(dr, True)
            run(dr, False)
    pb.print(case)
    extra = {"seeds": seeds, "draws": draws, "probe_chunks": 2 * len(seeds["probe"]), "chunk": chunk}
    return F.summarize(case, "single (adapter prefill_chunk)", fresh, repeat, prof, builds, pb, extra, tuple(preps))
