# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.3 (owner 2026-11-19): the packed prefill's steady walls, serving-equivalent (bringup/PACKED_DESIGN.md s9 item 4).

DECODER_SKIP=decode (layers 0-19, layer 20's KV head, 21-39 never: the layers loaded and looped, BATCH_LAYERS
executed), PACKED=on, agentic text (BRINGUP_POS_TOKENS=agentic_cc; lane k reads the text shifted by k x 4096 tokens, so
the four lanes are four different stretches of the session). Each segment is a request at its real history position
with its own slot state (fresh caches sized to it, as test_skip_positions: the long-range history is the state's
zeros; its window carry and engram tail are the real previous chunk's).

Cases (metric suffix):
    4x2k_p139264    four 2048-token segments, each at 139264
    mixed           2048 tokens each at 61440 / 139264 / 307200 / 546816 (61k / 139k / 308k / 547k)
    1x2k_p139264    one 2048-token segment and three empty lanes (the per-segment overhead)
Every iteration (warm-up, timed, profiled) first runs the REAL previous packed chunk (each segment at p - 2048,
untimed, synced), then the timed chunk; the slots change lanes every chunk (rotated by one lane each iteration: no
lane keeps a slot's device state; every segment's history is gathered from its own slot's caches every chunk).
Timed: from the engine's device input (uint32 [SP, 1, W] lane-natural, already on the device: the engine's H2D) through
the ids readback (tt/packed_device.ids_from_input: the one readback a chunk) to the sync after the last layer. Median
of WALL_RUNS warm runs with program-cache misses refused (testing/profile.warm_up / misses_allowed). Then one run
under the real-time profiler: the slowest chip's device time and its phases.

Slots (OV-62, serving's layout): every lane's slot state has ONE capacity, the deployment context rounded up to whole
chunks (PACKED_WALL_CAP overrides: a smaller one where four of them do not fit next to the model; printed with the
allocator's DRAM), so the one-ring indexer path is the one measured in every case, as tt-d-gen's equal slots take it.

History gathers (PB.3H): in the profiled run every history gather / prep op is traced (tt/history_gather.TRACE: its
device-operation ids, source layer and lane). The real-time profiler keeps one open record a command queue, so it
cannot show programs of two sub-devices at once (tt/gather_hide.py): a gather run beside the attention
(PACKED_GATHER_HIDE=on) is measured by device clock stamps instead (gather_hide.Stamps / LOG): its exposed time = the
compute's dispatch stalls on it + max(0, its end - the compute's end); a gather run serially is exposed whole (its
real-time record). The reference cost of every gather is the same chunk's serial gathers (one more profiled run with
the switch off, same lanes: the profiler's records of serial programs are exact). hidden = reference - exposed.
GATHER_RUNS profiled chunks a side, per segment the median (a host hiccup inside a region skews one run).
A full-mesh gather ends when every chip has reached it: on a chip that reached it early its time includes waiting for
the others (the lanes' skew, which the next collective absorbs in the serial design as well). So every gather (and
every region) is charged on the chip that waited least: its own cost (the reference's per-gather minimum over chips,
a region's exposure minimum over chips); packed_gather_exposed_worst_chip_ms_<case> keeps the worst chip's sum.
Metrics: packed_gather_ms_<case> (the reference),
packed_gather_hidden_ms_<case>, packed_gather_exposed_ms_<case>, packed_gather_exposed_frac_<case> (gated <= 0.25 for
4x2k_p139264 and mixed, recorded also without the text tag: the gate's names), the same per segment (_seg<k>: the case's k-th position; _segtails: the combined tails),
packed_gather_prep_ms_<case> (the packs / slices / slice_writes, local ops), packed_gather_regions_<case>.

Metrics: packed_wall_ms_<case> (gated > 0), packed_serving_tok_s_<case> (new tokens / wall), packed_device_ms_<case>,
packed_host_exposed_ms_<case> (wall - device), packed_device_ms_<phase>_<case>, packed_spread_ms_<case> and packed_runs_ms_<case> (the timed walls),
packed_dram_gib_<case> (the allocator's DRAM a chip with every segment's state and buffers held), packed_walls_table.

    BRINGUP_CHUNK=8192 DS41_PACKED=on BRINGUP_POS_TOKENS=agentic_cc DS41_DECODER_SKIP=decode \\
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/packed/test_packed_walls.py
"""

from __future__ import annotations

import importlib
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
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc
from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

_pc = importlib.reload(_pc)  # (the resident runner keeps helper modules of an earlier request: take this file's)
CHUNK, packed = _pc.CHUNK, _pc.packed

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = ("PACKED_WALL_CASES", "PACKED_WALL_CAP", "PB3H_STAMP_DUMP")
BATCH_LAYERS = "executed"  # F285: a performance measurement: the layers it executes (0-20 under decode)
LANE_ROWS = 2048
SHIFT = 4096  # lane k reads the text shifted by k x SHIFT
CASES = {
    "4x2k_p139264": [139264, 139264, 139264, 139264],
    "mixed": [61440, 139264, 307200, 546816],
    "1x2k_p139264": [139264, None, None, None],
}
GIB = float(1 << 30)
GATHER_RUNS = 3  # PB.3H: profiled chunks a side for the gather metrics (per segment the median)


def _cases() -> list:
    v = os.environ.get("PACKED_WALL_CASES", "").strip()
    return [c for c in v.split(",") if c] if v else list(CASES)


def _device_split(prof: dict) -> dict:
    per_chip = defaultdict(float)
    for d in prof["kernel_ns_dev"].values():
        for c, ns in d.items():
            per_chip[c] += ns
    phases = defaultdict(float)
    for sec, ns in prof["kernel_ns"].items():
        phases[profiler.phase_of(sec)] += ns
    return {
        "slowest": max(per_chip.values()) / 1e6 if per_chip else 0.0,
        "phases": {re.sub(r"[^A-Za-z0-9]+", "_", k): v / 1e6 for k, v in phases.items()},
    }


def _dram(mesh):
    try:
        from models.demos.deepseek_v41_flash_d_p.tests.bringup.test_dram import Dram

        return Dram(mesh)
    except Exception as e:  # noqa: BLE001
        print(f"DRAM view unavailable: {e}")
        return None


@mesh_parametrize(timeout_s=4 * 3600)
def test_packed_walls(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192"
    assert settings.get("DECODER_SKIP") == "decode", "the packed walls measure DECODER_SKIP=decode"
    # (no load-time switch set here, OV-61 / F283: the gate's one performance load serves every packed perf test)
    full = list(S.layers())
    layers = batch.run_layers(S, full, last_chunk=False)
    cases = _cases()
    top = max(p for c in cases for p in CASES[c] if p is not None)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    assert top + CHUNK <= deploy + CHUNK, (top, deploy)
    cap = _cap(deploy)
    assert cap >= top + CHUNK, ("PACKED_WALL_CAP below the cases' positions", cap, top)
    want = max(cap, int(S.data["target"]["seq"]))
    monkeypatch.setitem(S.data["target"], "seq", want)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    assert model.chunk == CHUNK and model.max_seq >= top + CHUNK, (model.chunk, model.max_seq, top)
    g = PositionGolden(S, top + 3 * SHIFT, CHUNK)
    text = g.tokens()
    dram = _dram(mesh_device)
    rows = []
    try:
        for case in cases:
            rows.append((case, _case(model, mesh_device, layers, case, text, dram, cap)))
    finally:
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    head = [
        "case",
        "positions",
        "new tokens",
        "wall ms (spread)",
        "device ms",
        "host exposed ms",
        "new tok/s",
        "history gather ms (hidden)",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for case, r in rows:
        lines.append(
            f"| {case} | {r['pos']} | {r['new']} | {r['wall']} ({r['spread']}) | {r['device']:.1f} | "
            f"{r['wall'] - r['device']:.1f} | {r['tok_s']:.0f} | {r['gather'][0]:.2f} ({r['gather'][1]:.2f}) |"
        )
    table = "\n".join(lines)
    metrics.record("packed_walls_table", table)
    print("\n" + table)


def _cap(deploy: int) -> int:
    """OV-62: every slot's capacity (one for all lanes): PACKED_WALL_CAP, else the deployment context in whole chunks."""
    v = os.environ.get("PACKED_WALL_CAP", "").strip()
    return int(v) if v else -(-deploy // CHUNK) * CHUNK


def _serial_gathers(prof: dict, trace: list, skip: set) -> dict:
    """The traced history gathers (kinds gather / tail) the real-time profiler recorded, but those launched inside a
    gather_hide region (``skip``: their ids): {lane: ms} with each gather at its least chip time ("g"), {lane: ms} on
    the worst chip ("worst": the chip with the largest sum), and the prep ops' ms (per-op least chip time)."""
    recs, chip_of = prof.get("rt_records") or {}, prof.get("dev_to_chip") or {}
    g, prep, per = defaultdict(float), 0.0, defaultdict(lambda: defaultdict(float))
    for k, _src, lane, a, b in trace:
        for i in range(a, b):
            if i in skip or not recs.get(i):
                continue
            t = {chip_of.get(int(d), int(d)): (e[2] - e[1]) / 1e6 for d, e in recs[i].items()}
            if k in ("gather", "tail"):
                g[lane] += min(t.values())
                for c, v in t.items():
                    per[c][lane] += v
            else:
                prep += min(t.values())
    worst = max(per.values(), key=lambda v: sum(v.values())) if per else {}
    return {"g": dict(g), "worst": dict(worst), "prep": prep}


def _region_exposed(log: list, clocks: list, ghz: float) -> tuple:
    """From gather_hide.LOG and the stamps' clocks [chip][slot]: ({lane: exposed ms}: every region on the chip with
    its least exposure, {lane: ms} the worst chip's sum, the ids of the gathers launched in regions). A stall (the
    stamps "before" / "after" a gather j >= 1: the compute's launch waited for the gather before it) is charged to
    that gather's lane; the region's end (last gather end - compute end) to the last gather's lane."""
    least, total, ids = defaultdict(float), defaultdict(lambda: defaultdict(float)), set()
    for reg in log:
        out = defaultdict(lambda: defaultdict(float))
        prev, before, owed, endc = None, None, None, None
        for kind, lane, v in reg:
            if kind == "ids":
                ids.update(range(*v))
            elif kind == "before":
                before, owed = v, prev  # (the launch after it waits for the gather before: ``prev``)
            elif kind == "after":
                for c, t in enumerate(clocks):
                    out[c][owed] += max(0, int(t[v]) - int(t[before])) / ghz / 1e6
            elif kind == "end_compute":
                endc = v
            elif kind == "end_gather":
                for c, t in enumerate(clocks):
                    out[c][lane] += max(0, int(t[v]) - int(t[endc])) / ghz / 1e6
            if kind == "ids":
                prev = lane
        if out:
            c = min(out, key=lambda c: sum(out[c].values()))
            for lane, v in out[c].items():
                least[lane] += v
            for c, lanes in out.items():
                for lane, v in lanes.items():
                    total[c][lane] += v
    worst = max(total.values(), key=lambda v: sum(v.values())) if total else {}
    return dict(least), dict(worst), ids


def _gather_metrics(reg, ser, ref, regions: int, ghz: float) -> dict:
    """Per lane (reference ms, exposed ms), totals, the worst chip's exposed sum, prep ms (``reg``: _region_exposed)."""
    least, worst, _ = reg
    lanes = sorted(set(ref["g"]) | set(ser["g"]) | set(least), key=lambda k: (k is None, k))
    per = {k: (ref["g"].get(k, 0.0), ser["g"].get(k, 0.0) + least.get(k, 0.0)) for k in lanes}
    worst_exp = sum(worst.values()) + sum(ser["worst"].values())
    return {
        "ms": sum(v[0] for v in per.values()),
        "exposed": sum(v[1] for v in per.values()),
        "worst": worst_exp,
        "lanes": per,
        "prep": ser["prep"],
        "regions": regions,
        "ghz": ghz,
    }


def _gather_metrics_runs(runs_on: list, runs_ref: list, ghz: float) -> dict:
    """GATHER_RUNS profiled chunks a side: per segment the median of its runs' exposed and reference ms (a host
    hiccup inside one region of one run skews that run only)."""
    import statistics

    one = [_gather_metrics(reg, ser, ref, n, ghz) for (reg, ser, n), ref in zip(runs_on, runs_ref)]
    lanes = sorted({k for g in one for k in g["lanes"]}, key=lambda k: (k is None, k))
    per = {
        k: (
            statistics.median(g["lanes"].get(k, (0.0, 0.0))[0] for g in one),
            statistics.median(g["lanes"].get(k, (0.0, 0.0))[1] for g in one),
        )
        for k in lanes
    }
    print(
        f"history gathers, each run (reference / exposed ms): {[(round(g['ms'], 3), round(g['exposed'], 3)) for g in one]}"
    )
    return {
        "ms": sum(v[0] for v in per.values()),
        "exposed": sum(v[1] for v in per.values()),
        "worst": statistics.median(g["worst"] for g in one),
        "lanes": per,
        "prep": statistics.median(g["prep"] for g in one),
        "regions": one[0]["regions"],
        "ghz": ghz,
    }


def _ghz(mesh, st) -> float:
    """The stamps' clock rate (ticks a ns): two stamps 0.2 s apart on the host clock (chip 0)."""
    import ttnn

    t0 = time.perf_counter()
    a = st("compute")
    ttnn.synchronize_device(mesh)
    time.sleep(0.2)
    t1 = time.perf_counter()
    b = st("compute")
    ttnn.synchronize_device(mesh)
    c = st.read()[0]
    return float(int(c[b]) - int(c[a])) / ((t1 - t0) * 1e9)


def _record_gate_name(name: str, value) -> None:
    """The gate's threshold names carry no text tag: record the metric under both names."""
    metrics.record(name, value)
    tok = os.environ.pop("BRINGUP_POS_TOKENS", None)
    try:
        metrics.record(name, value)
    finally:
        if tok is not None:
            os.environ["BRINGUP_POS_TOKENS"] = tok


def _case(model, mesh, layers, case: str, text, dram, cap: int) -> dict:
    from models.demos.deepseek_v41_flash_d_p.tt import gather_hide, history_gather
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import ids_from_input, make_input, run_chunk

    pos = CASES[case]
    live = [k for k, p in enumerate(pos) if p is not None]
    states = {k: model.new_state(cap) for k in live}  # (OV-62: one capacity, serving's slots)
    toks = {k: text[k * SHIFT : k * SHIFT + pos[k] + LANE_ROWS].long() for k in live}
    it = [0]

    def chunk(prev: bool):
        """The packed chunk of this iteration: lane (k + it) % 4 holds segment k (slots move every chunk)."""
        lanes = [None] * 4
        for k in live:
            lanes[(k + it[0]) % 4] = k
        segs, ids = [], []
        for k in lanes:
            if k is None:
                segs.append(None)
                ids.append(None)
                continue
            s = pos[k] - (LANE_ROWS if prev else 0)
            segs.append(Segment(10 + k, s, s + LANE_ROWS))
            ids.append(toks[k][s : s + LANE_ROWS])
        desc = ChunkDescriptor.packed(CHUNK, segs)
        return desc, [None if k is None else states[k] for k in lanes], ids

    def run(prev: bool, timed: bool = False):
        desc, sts, ids = chunk(prev)
        inp = make_input(mesh, desc, ids)  # (the engine's H2D: not timed)
        model.sync()
        t0 = time.time()
        nat = ids_from_input(mesh, inp, desc)
        run_chunk(model, desc, nat, sts, layers, last=False)
        model.sync()
        wall = (time.time() - t0) * 1e3
        import ttnn

        ttnn.deallocate(inp)
        return wall

    def one_run():
        run(True)
        w = run(False)
        it[0] += 1
        return round(w, 1)

    try:
        with packed("on"):
            w = warm_up(mesh, one_run)
            assert w["warm"], f"{case}: not warm after {len(w['walls'])} runs"
            walls = []
            misses_allowed(mesh, False)
            try:
                for _ in range(WALL_RUNS):
                    walls.append(one_run())
            finally:
                misses_allowed(mesh, True)
            kept = sorted(walls)
            wall = kept[len(kept) // 2]

            def profiled():
                profiler.enable(mesh, rt=True)
                history_gather.TRACE = []
                gather_hide.LOG.clear()
                try:
                    profiler.set_layer(None)
                    run(False)
                    profiler.signpost("end")
                    profiler.finish_rt()
                    return profiler.result(), list(history_gather.TRACE), list(gather_hide.LOG)
                finally:
                    history_gather.TRACE = None
                    profiler.disable()

            gather_hide.warm_stamps(mesh)  # (the stamp programs' first use compiles: before the measured run)
            st = gather_hide.stamps(mesh)
            ghz = _ghz(mesh, st) if st is not None else 1.0
            runs_on, runs_ref = [], []
            for k in range(GATHER_RUNS):  # (each with the real previous chunk first; the lanes of this iteration)
                run(True)
                c0 = cache_entries(mesh)
                prof, trace, log = profiled()
                if k == 0:
                    dev = _device_split(prof)
                    n0, n1 = c0, cache_entries(mesh)
                clocks = st.read() if (st is not None and log) else []
                reg = _region_exposed(log, clocks, ghz)
                runs_on.append((reg, _serial_gathers(prof, trace, reg[2]), len(log)))
                dump = os.environ.get("PB3H_STAMP_DUMP", "").strip()  # (diagnostic: the raw stamps, offline reading)
                if dump and clocks:
                    import json

                    used = sorted({v for reg_ in log for k_, _, v in reg_ if k_ not in ("ids",)})
                    with open(f"{dump}.{case}.{k}.json", "w") as f:
                        json.dump({"log": log, "ghz": ghz, "clocks": {i: [int(c[i]) for c in clocks] for i in used}}, f)
                if settings.get("PACKED_GATHER_HIDE") == "on":  # (the reference: the same chunk's gathers, serial)
                    with _pc.switch("PACKED_GATHER_HIDE", "off"):
                        run(True)
                        prof_off, trace_off, _ = profiled()
                    runs_ref.append(_serial_gathers(prof_off, trace_off, set()))
                else:
                    runs_ref.append(runs_on[-1][1])
            gs = _gather_metrics_runs(runs_on, runs_ref, ghz)
            seg_of = {(k + it[0]) % 4: k for k in live}  # (the profiled chunk's lanes: segment k in lane (k + it) % 4)
            it[0] += 1
        new = LANE_ROWS * len(live)
        sfx = f"_{case}"
        metrics.record("packed_wall_ms" + sfx, wall)
        metrics.record("packed_runs_ms" + sfx, walls)
        metrics.record("packed_spread_ms" + sfx, round(kept[-1] - kept[0], 1))
        metrics.record("packed_wall_settled" + sfx, int(settled(w["walls"])))
        metrics.record("packed_serving_tok_s" + sfx, round(new / (wall / 1e3), 1))
        metrics.record("packed_device_ms" + sfx, round(dev["slowest"], 2))
        metrics.record("packed_host_exposed_ms" + sfx, round(wall - dev["slowest"], 1))
        metrics.record("packed_profiled_new_programs" + sfx, -1 if n0 is None or n1 is None else n1 - n0)
        for ph, ms in dev["phases"].items():
            metrics.record(f"packed_device_ms_{ph}" + sfx, round(ms, 2))
        if dram is not None:
            metrics.record("packed_dram_gib" + sfx, round(dram.max() / GIB, 3))
        exp = gs["exposed"]
        frac = exp / gs["ms"] if gs["ms"] > 0 else 0.0
        metrics.record("packed_gather_ms" + sfx, round(gs["ms"], 3))
        metrics.record("packed_gather_hidden_ms" + sfx, round(gs["ms"] - exp, 3))
        metrics.record("packed_gather_exposed_ms" + sfx, round(exp, 3))
        _record_gate_name("packed_gather_exposed_frac" + sfx, round(frac, 4))
        metrics.record("packed_gather_prep_ms" + sfx, round(gs["prep"], 3))
        metrics.record("packed_gather_regions" + sfx, gs["regions"])
        metrics.record("packed_gather_exposed_worst_chip_ms" + sfx, round(gs["worst"], 3))
        names = {lane: ("tails" if lane == -1 else seg_of.get(lane, f"lane{lane}")) for lane in gs["lanes"]}
        for lane, (ms, e) in gs["lanes"].items():  # (segment k = the case's k-th position; "tails": every segment's)
            metrics.record(f"packed_gather_ms{sfx}_seg{names[lane]}", round(ms, 3))
            metrics.record(f"packed_gather_hidden_ms{sfx}_seg{names[lane]}", round(ms - e, 3))
            metrics.record(f"packed_gather_exposed_ms{sfx}_seg{names[lane]}", round(e, 3))
        lanes_s = ", ".join(f"seg {names[k]} (lane {k}): {v[0]:.3f} / {v[1]:.3f}" for k, v in gs["lanes"].items())
        print(
            f"CASE {case} history gathers (each at its least chip): reference (serial) {gs['ms']:.3f} ms, exposed {exp:.3f}, "
            f"hidden {gs['ms'] - exp:.3f} (exposed frac {frac:.3f}); per segment (reference / exposed ms) {lanes_s}; "
            f"{gs['regions']} regions; worst chip's exposed sum {gs['worst']:.3f} ms; prep ops {gs['prep']:.3f} ms; clock "
            f"{gs['ghz']:.4f} GHz; slot capacity {cap}",
            flush=True,
        )
        print(
            f"CASE {case} positions {pos}: wall {wall} ms (runs {walls}), device {dev['slowest']:.2f} ms, "
            f"{new / (wall / 1e3):.0f} new tok/s; phases "
            + ", ".join(f"{k} {v:.2f}" for k, v in sorted(dev["phases"].items(), key=lambda x: -x[1])),
            flush=True,
        )
        return {
            "pos": "/".join(str(p) for p in pos if p is not None),
            "new": new,
            "wall": wall,
            "spread": round(kept[-1] - kept[0], 1),
            "device": dev["slowest"],
            "tok_s": new / (wall / 1e3),
            "gather": (gs["ms"], gs["ms"] - gs["exposed"]),
        }
    finally:
        for st in states.values():
            st.free()
