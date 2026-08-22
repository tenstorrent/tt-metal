#!/usr/bin/env python3
"""Selftest for the lane-ET e2e-metric overhaul (owner ratification
2026-08-21): dual-metric measurement (diagnostic zone + drain-inclusive
KERNEL zone from ONE device run), KERNEL-decided verdicts, v2 baseline
wiring, the verdict-metric DELTA report, and the ES-F1 TENSIX-timeout
flush + verify + DEVICE-POISONED marking.

Toolchain-free: Sweep objects are built via object.__new__ with stubbed
device/classify plumbing — silicon(), _perf_value, _row_verdict,
_issue_slot_check, _kernel_cell_gate, the delta section and the ES-F1
helpers are the REAL code.
"""

import os
import pathlib
import stat
import sys
import tempfile
import types

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_2x2 as sweep  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"SELFTEST PASS: {name}")
    else:
        print(f"SELFTEST FAIL: {name} {detail}")
        FAILS.append(name)


def mk_sweep(ev, **a_over):
    sw = object.__new__(sweep.Sweep)
    args = dict(
        force=False,
        dry_run=False,
        knob_silicon_rows=None,
        knob_attribution=False,
        skip_craq_gate=True,
        classify_workers=1,
        priority_ops=None,
        prev_run=None,
        baseline=None,
        kernel_baseline=None,
        allow_win_to_parity=False,
        max_drift_pct=5.0,
        max_abs_drift_pct=10.0,
        red_loss_growth_pct=5.0,
    )
    args.update(a_over)
    sw.a = types.SimpleNamespace(**args)
    sw.ev = pathlib.Path(ev)
    sw.ev.mkdir(parents=True, exist_ok=True)
    sw.reds = []
    sw.notes = []
    sw.reused = []
    return sw


def mk_row(
    op, nodes, kind="full2x2", marker="TILE_LOOP", per_tile=True, lb=None, sem_class=""
):
    row = {
        "op": op,
        "corpus_id": f"cid__{op}",
        "kind": kind,
        "marker": marker,
        "metric": "MATH_ISOLATE",
        "per_tile": per_tile,
        "issue_slot_lb": lb,
        "pin_flags": "",
        "extra_env": {},
        "sel_extra_env": {"sem": {}, "hand": {}},
        "schedule": "nightly",
        "sem_class": sem_class,
        "craq_archs": "bh",
        "nodes": {s: nodes.get(s, "") for s in sweep.SELECTORS},
        "note": "",
    }
    return row


# ---------------- 1. dual-metric CSV parsing ----------------

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    sw = mk_sweep(td / "ev")
    row = mk_row("dualop", {"sem-perf": "perf.py::t"})
    work = sw.ev / "dualop" / "silicon" / "sem-perf" / "r1-on" / "perf_data" / "mod"
    work.mkdir(parents=True)
    (work / "mod.post.csv").write_text(
        "marker,tile_cnt,mean(MATH_ISOLATE)\n"
        "INIT,8,100.0\n"
        "TILE_LOOP,8,1150.0\n"
        "KERNEL,8,147000.0\n"
    )
    diag = sw._perf_value(row, "sem-perf", "r1", "on")
    kern = sw._kernel_value(row, "sem-perf", "r1", "on")
    check(
        "dual parse: diag = TILE_LOOP mean / tile_cnt (per-tile convention)",
        diag == 1150.0 / 8,
        f"got {diag}",
    )
    check(
        "dual parse: KERNEL cell = absolute mean at the KERNEL marker "
        "(no per-tile division)",
        kern == 147000.0,
        f"got {kern}",
    )
    check(
        "kernel_scope is distinct from row_scope (no baseline keyspace collision)",
        sweep.kernel_scope(row) == "KERNEL_MATH_ISOLATE_E2E"
        and sweep.kernel_scope(row) != sweep.row_scope(row),
        f"{sweep.kernel_scope(row)} vs {sweep.row_scope(row)}",
    )

# ---------------- 2. silicon(): kernel cells, ratios, verdict class ----------------


def mk_silicon_sweep(ev, diag_vals, kern_vals):
    """diag_vals/kern_vals: {(sel, leg): value-or-None}."""
    sw = mk_sweep(ev)

    def fake_device_job(
        row, sel, label, leg, flags, tag="silicon", expected_texts=None
    ):
        return 0

    sw._device_job = fake_device_job

    def fake_perf(row, sel, label, leg, tag="silicon", marker=None, per_tile=None):
        vals = kern_vals if marker == sweep.KERNEL_MARKER else diag_vals
        return vals.get((sel, leg))

    sw._perf_value = fake_perf
    sw._classify_texts = lambda row, sel, leg, tag="classify": {"math.elf": "x"}
    sw._macro_lb_gate = lambda row, cls, result: None
    sw._measured_flags_gate = lambda row, cls, result: None
    return sw


CLS_CHANGED = {
    "sem-corr": {"status": "OK", "all": "IDENTICAL"},
    "hand-corr": {"status": "OK", "all": "IDENTICAL"},
    "sem-perf": {"status": "OK", "all": "CHANGED"},
    "hand-perf": {"status": "OK", "all": "IDENTICAL"},
}
NODES = {
    "sem-corr": "t.py::corr[i:1]",
    "sem-perf": "t.py::perf[i:1]",
    "hand-corr": "t.py::corr[i:0]",
    "hand-perf": "t.py::perf[i:0]",
}

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    # diag says WIN (-10%), kernel says PARITY (+0.2%): the class must be
    # PARITY (kernel-decided) and the delta section must list the row.
    diag = {
        ("sem-perf", "off"): 120.0,
        ("sem-perf", "on"): 90.0,
        ("hand-perf", "off"): 100.0,
    }
    kern = {
        ("sem-perf", "off"): 10100.0,
        ("sem-perf", "on"): 10020.0,
        ("hand-perf", "off"): 10000.0,
    }
    sw = mk_silicon_sweep(td / "ev", diag, kern)
    row = mk_row("compress", NODES)
    res = sw.silicon(row, CLS_CHANGED)
    kc = res["kernel_cells"]
    check(
        "silicon(): kernel cells populated for every measured leg "
        "(hand fold fills both hand cells)",
        kc.get("sem_off") == 10100.0
        and kc.get("sem_on") == 10020.0
        and kc.get("hand_off") == 10000.0
        and kc.get("hand_on") == 10000.0,
        f"got {kc}",
    )
    check(
        "silicon(): kernel_causal/kernel_vs_hand computed from kernel cells",
        abs(res["kernel_vs_hand_pct"] - 0.2) < 1e-9
        and abs(res["kernel_causal_pct"] - (100.0 * (10020 - 10100) / 10100)) < 1e-9,
        f"{res.get('kernel_vs_hand_pct')}, {res.get('kernel_causal_pct')}",
    )
    check(
        "verdict class is KERNEL-decided (diag WIN -10% vs kernel PARITY +0.2%)",
        sweep.Sweep._row_class(res) == "PARITY"
        and sweep.Sweep._diag_row_class(res) == "WIN",
        f"class={sweep.Sweep._row_class(res)} diag={sweep.Sweep._diag_row_class(res)}",
    )
    check("no reds from a fully-parsed dual-metric row", not sw.reds, str(sw.reds))
    # delta section lists the row + writes KERNEL-DELTA.md
    lines = sw._kernel_delta_section([res])
    joined = "\n".join(lines)
    check(
        "delta section lists the class-changing row with both ratios",
        "compress" in joined and "1 row(s) change class" in joined,
        joined,
    )
    check(
        "KERNEL-DELTA.md written standalone",
        (sw.ev / "KERNEL-DELTA.md").is_file()
        and "compress" in (sw.ev / "KERNEL-DELTA.md").read_text(),
    )

    # legacy payload (no kernel_cells key) keeps the diagnostic banding
    legacy = {"cells": {"sem_on": 90.0}, "vs_hand_pct": -10.0}
    check(
        "legacy result payloads (no kernel_cells) keep diagnostic banding",
        sweep.Sweep._row_class(legacy) == "WIN",
    )
    # a dual-metric payload with an unparsable kernel ratio is UNMEASURED,
    # never silently diagnostic-banded
    half = {"cells": {"sem_on": 90.0}, "kernel_cells": {}, "vs_hand_pct": -10.0}
    check(
        "dual-metric payload without a kernel ratio is UNMEASURED (no "
        "silent diagnostic fallback)",
        sweep.Sweep._row_class(half) == "UNMEASURED",
    )

# refusal mirroring + kernel-unparsable RED
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    cls_ident = dict(CLS_CHANGED, **{"sem-perf": {"status": "OK", "all": "IDENTICAL"}})
    sw = mk_silicon_sweep(
        td / "ev", {("hand-perf", "off"): 100.0}, {("hand-perf", "off"): 9000.0}
    )
    res = sw.silicon(mk_row("refop", NODES), cls_ident)
    check(
        "sem refusal mirrors into kernel_cells (REFUSAL_BYTE_IDENTICAL)",
        res["kernel_cells"].get("sem_off") == "REFUSAL_BYTE_IDENTICAL"
        and res["kernel_cells"].get("sem_on") == "REFUSAL_BYTE_IDENTICAL",
        str(res["kernel_cells"]),
    )
    check("refusal row class stays REFUSAL", sweep.Sweep._row_class(res) == "REFUSAL")

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    # diag parses, kernel does NOT -> RED (verdict zone missing)
    sw = mk_silicon_sweep(
        td / "ev",
        {
            ("sem-perf", "off"): 120.0,
            ("sem-perf", "on"): 90.0,
            ("hand-perf", "off"): 100.0,
        },
        {},
    )
    res = sw.silicon(mk_row("nokern", NODES), CLS_CHANGED)
    check(
        "kernel-unparsable with measured diag cells is RED (fail closed, "
        "no silent diagnostic fallback)",
        any("KERNEL cell is unparsable" in r for r in sw.reds),
        str(sw.reds),
    )

# ---------------- 3. issue-slot gate is diagnostic-only ----------------

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    sw = mk_sweep(td / "ev")
    row = mk_row("lbop", NODES, lb=50.0)
    res = {
        "cells": {"sem_on": 20.0, "sem_off": 80.0},
        "kernel_cells": {"sem_on": 9000.0, "sem_off": 9100.0},
        "notes": [],
    }
    sw._issue_slot_check(row, res)
    check(
        "issue-slot: diag reading below lb invalidates the DIAG cell only",
        res["cells"]["sem_on"] == "INVALID_MARKER"
        and res["kernel_cells"]["sem_on"] == 9000.0,
        str(res),
    )
    check(
        "issue-slot: with a KERNEL twin the diag invalidation is a note, "
        "not a run-blocking red",
        not sw.reds
        and any("verdict decided by the KERNEL cell" in n for n in res["notes"]),
        f"reds={sw.reds} notes={res['notes']}",
    )
    # without a kernel twin the invalidation still escalates
    sw2 = mk_sweep(td / "ev2")
    res2 = {"cells": {"sem_on": 20.0}, "kernel_cells": {}, "notes": []}
    sw2._issue_slot_check(row, res2)
    check(
        "issue-slot: diag-invalid WITHOUT a KERNEL twin stays RED",
        any("no parsable KERNEL cell" in r for r in sw2.reds),
        str(sw2.reds),
    )

# ---------------- 4. _row_verdict: kernel anchors + handover rule ----------------


def mk_result(op, diag_cells, kernel_cells=None, **extra):
    r = {
        "op": op,
        "corpus_id": f"cid__{op}",
        "kind": "full2x2",
        "marker": "TILE_LOOP",
        "scope": "TILE_LOOP_MATH_ISOLATE_PER_TILE",
        "cells": diag_cells,
        "runs": {},
        "notes": [],
    }
    if kernel_cells is not None:
        r["kernel_cells"] = kernel_cells
        r["kernel_scope"] = "KERNEL_MATH_ISOLATE_E2E"
    r.update(extra)
    return r


with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    sw = mk_sweep(td / "ev")
    cid = "cid__vop"
    dscope = "TILE_LOOP_MATH_ISOLATE_PER_TILE"
    kscope = "KERNEL_MATH_ISOLATE_E2E"
    dbase = {
        (cid, dscope, "vop:sem_off"): [100.0],
        (cid, dscope, "vop:sem_on"): [80.0],
        (cid, dscope, "vop:hand_on"): [85.0],
    }
    kbase = {
        (cid, kscope, "vop:sem_off"): [10000.0],
        (cid, kscope, "vop:sem_on"): [9500.0],
        (cid, kscope, "vop:hand_on"): [9600.0],
    }

    # (a) kernel WIN->LOSS flip = RED at full severity
    res = mk_result(
        "vop",
        {"sem_off": 100.0, "sem_on": 80.0, "hand_on": 85.0},
        {"sem_off": 10000.0, "sem_on": 9700.0, "hand_on": 9500.0},
        causal_pct=-20.0,
        vs_hand_pct=-5.88,
        kernel_causal_pct=-3.0,
        kernel_vs_hand_pct=2.1,
    )
    v = sw._row_verdict(res, dbase, {}, {}, kbase, {})
    check(
        "kernel WIN->LOSS FLIP vs v2 anchors is RED",
        v["rag"] == "RED"
        and any(
            "kernel vs_hand WIN→LOSS FLIP" in x and "RED" in x for x in v["verdicts"]
        ),
        str(v["verdicts"]),
    )

    # (b) diag flip is DEMOTED to YELLOW once kernel anchors exist
    res = mk_result(
        "vop",
        {"sem_off": 100.0, "sem_on": 90.0, "hand_on": 85.0},
        {"sem_off": 10000.0, "sem_on": 9500.0, "hand_on": 9600.0},
        causal_pct=-10.0,
        vs_hand_pct=5.88,  # diag says loss where baseline said win
        kernel_causal_pct=-5.0,
        kernel_vs_hand_pct=-1.04,  # kernel still a win, matches anchors
    )
    v = sw._row_verdict(res, dbase, {}, {}, kbase, {})
    check(
        "diag WIN->LOSS flip is capped YELLOW when the row has kernel anchors",
        v["rag"] == "YELLOW"
        and any(
            "diag vs_hand WIN→LOSS FLIP" in x and "YELLOW" in x for x in v["verdicts"]
        ),
        str(v["verdicts"]),
    )

    # (c) HANDOVER: same diag flip with NO kernel anchors stays RED
    v = sw._row_verdict(res, dbase, {}, {}, {}, {})
    check(
        "handover rule: diag WIN->LOSS flip stays RED while v2 is unseeded",
        v["rag"] == "RED"
        and any("vs_hand WIN→LOSS FLIP" in x and "RED" in x for x in v["verdicts"]),
        str(v["verdicts"]),
    )

    # (d) dead KERNEL cell with v2 history = RED
    res = mk_result(
        "vop",
        {"sem_off": 100.0, "sem_on": 80.0},
        {"sem_off": None, "sem_on": None},
    )
    v = sw._row_verdict(res, {}, {}, {}, kbase, {})
    check(
        "dead KERNEL cell with v2 baseline history is INVALID_METRIC RED",
        v["rag"] == "RED" and any("KERNEL cell(s)" in x for x in v["verdicts"]),
        str(v["verdicts"]),
    )

    # (e) INVALID_MARKER diag cell with a numeric kernel twin = YELLOW;
    #     without = RED (legacy payload)
    res = mk_result(
        "vop",
        {"sem_on": "INVALID_MARKER"},
        {"sem_on": 9500.0},
        kernel_vs_hand_pct=None,
    )
    v = sw._row_verdict(res, {}, {}, {}, {}, {})
    check(
        "INVALID_MARKER diag cell with KERNEL twin is YELLOW",
        v["rag"] == "YELLOW" and any("diag INVALID_MARKER" in x for x in v["verdicts"]),
        str(v["verdicts"]),
    )
    legacy = mk_result("vop", {"sem_on": "INVALID_MARKER"})
    v = sw._row_verdict(legacy, {}, {}, {}, {}, {})
    check(
        "legacy INVALID_MARKER (no kernel cells) keeps the RED",
        v["rag"] == "RED",
        str(v["verdicts"]),
    )

    # (f) kernel abs-drift RED, diag abs-drift demoted when anchored
    res = mk_result(
        "vop",
        {"sem_on": 80.0},
        {"sem_on": 12000.0},  # +26% over the 9500 anchor
    )
    v = sw._row_verdict(res, dbase, {}, {}, kbase, {})
    check(
        "kernel per-cell ABS drift beyond threshold is RED",
        v["rag"] == "RED"
        and any("kernel sem_on ABS CYCLES" in x for x in v["verdicts"]),
        str(v["verdicts"]),
    )
    res = mk_result(
        "vop",
        {"sem_on": 95.0},  # +18.75% over the 80 diag anchor
        {"sem_on": 9500.0},  # matches kernel anchor
    )
    v = sw._row_verdict(res, dbase, {}, {}, kbase, {})
    check(
        "diag per-cell ABS drift is capped YELLOW when kernel anchors exist",
        v["rag"] == "YELLOW"
        and any("diag sem_on ABS CYCLES" in x and "YELLOW" in x for x in v["verdicts"]),
        str(v["verdicts"]),
    )
    v = sw._row_verdict(res, dbase, {}, {}, {}, {})
    check(
        "handover rule: diag ABS drift stays RED while v2 is unseeded",
        v["rag"] == "RED",
        str(v["verdicts"]),
    )

# ---------------- 5. ES-F1: timeout scan, poisoned marking, flush+verify ----------------

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    log = td / "log.txt"
    log.write_text("blah\nTENSIX TIMED OUT waiting for math\n")
    check(
        "_scan_device_timeout: log text detected",
        sweep.Sweep._scan_device_timeout(log, 1),
    )
    log.write_text("all fine\n1 passed\n")
    check(
        "_scan_device_timeout: rc=124 (killed by timeout) detected",
        sweep.Sweep._scan_device_timeout(log, 124),
    )
    check(
        "_scan_device_timeout: clean log + rc 0 is clean",
        not sweep.Sweep._scan_device_timeout(log, 0),
    )

    # poisoned marking surfaces in the batched assembly RED
    sw = mk_sweep(td / "ev")
    work = td / "leg"
    work.mkdir()
    (work / "rc.txt").write_text("1\n")
    (work / "log.txt").write_text("boom\n")
    sweep.Sweep._mark_poisoned_leg(
        work,
        "TENSIX timeout in this batched session — co-scheduled failure is a collateral suspect, not a proven kernel failure",
    )
    row = mk_row("pop", NODES)
    rc = sw._batched_leg_verdict(row, "sem-perf", "r1", "on", work, {"k": 1}, None)
    check(
        "batched assembly surfaces DEVICE-POISONED in the RED",
        rc != 0 and any("DEVICE-POISONED" in r for r in sw.reds),
        str(sw.reds),
    )

    # flush missing -> stays poisoned + RED
    sw = mk_sweep(td / "ev2")
    os.environ["SWEEP_FLUSH_SH"] = str(td / "no-such-flush.sh")
    sw._flush_and_verify("selftest-context")
    check(
        "flush script missing: device stays POISONED + RED",
        sw.device_state == "poisoned"
        and any("flush script is missing" in r for r in sw.reds),
        f"state={sw.device_state} reds={sw.reds}",
    )

    # flush ok + verify passes -> state returns to clean
    sw = mk_sweep(td / "ev3")
    flush = td / "flush.sh"
    flush.write_text("#!/bin/sh\necho flushed\nexit 0\n")
    flush.chmod(flush.stat().st_mode | stat.S_IEXEC)
    fake_py = td / "fakepy"
    fake_py.write_text('#!/bin/sh\necho "1 passed"\nexit 0\n')
    fake_py.chmod(fake_py.stat().st_mode | stat.S_IEXEC)
    sw.python = fake_py
    os.environ["SWEEP_FLUSH_SH"] = str(flush)
    sw._flush_and_verify("selftest-context")
    check(
        "flush + verify PASS returns device_state to clean (evidence dir written)",
        sw.device_state == "clean"
        and (sw.ev / "device-flush" / "flush-01" / "flush.log").is_file()
        and any("verify PASS" in n for n in sw.notes),
        f"state={sw.device_state} notes={sw.notes} reds={sw.reds}",
    )

    # flush ok + verify FAILS -> stays poisoned + RED
    sw = mk_sweep(td / "ev4")
    fake_bad = td / "fakebad"
    fake_bad.write_text('#!/bin/sh\necho "1 failed"\nexit 1\n')
    fake_bad.chmod(fake_bad.stat().st_mode | stat.S_IEXEC)
    sw.python = fake_bad
    sw._flush_and_verify("selftest-context")
    check(
        "flush ok but verify FAIL: device stays POISONED + RED",
        sw.device_state == "poisoned" and any("verify FAILED" in r for r in sw.reds),
        f"state={sw.device_state} reds={sw.reds}",
    )
    del os.environ["SWEEP_FLUSH_SH"]

# ---------------- 6. ops.tsv v4 wiring ----------------

rows = sweep.load_config(HERE / "sweep_2x2_ops.tsv")
by = {r["op"]: r for r in rows}
check(
    "ops v4: topk-perf is a measured full2x2 row (typed-vs-hand A/B), "
    "sem_class=measure-identical forces the one-leg measurement",
    by["topk-perf"]["kind"] == "full2x2"
    and by["topk-perf"]["sem_class"] == "measure-identical"
    and sweep.fresh_body_row(by["topk-perf"])
    and by["topk-perf"]["marker"] == "TOPK_BODY"
    and not by["topk-perf"]["per_tile"],
    str({k: by["topk-perf"][k] for k in ("kind", "marker", "sem_class", "per_tile")}),
)
check(
    "ops v4: topkxl is a measured semantic row at KERNEL scope with the "
    "new device-profile vehicle",
    by["topkxl"]["kind"] == "semantic"
    and by["topkxl"]["marker"] == "KERNEL"
    and "test_topk_xl_device_profile" in by["topkxl"]["nodes"]["sem-perf"]
    and sweep.fresh_body_row(by["topkxl"]),
    str(by["topkxl"]["nodes"]),
)
check(
    "ops v4: topkmetal is an SKIP_ALIAS (no double-booking of the topk kernel)",
    by["topkmetal"]["kind"] == "skip" and "SKIP_ALIAS" in by["topkmetal"]["note"],
    by["topkmetal"]["note"][:120],
)
check(
    "ops v4: gmg rows stay skipped with the metric excuse re-based honestly",
    by["gmgsingleface"]["kind"] == "skip"
    and "LIFTED" in by["gmgsingleface"]["note"]
    and "gmg_sanitize_scratch" in by["gmgsingleface"]["note"],
    by["gmgsingleface"]["note"][:160],
)
# Lane FK (2026-08-22): ema-fresh's SKIP_NOT_FEASIBLE verdict is DISSOLVED
# by the typed cross-lane surface (fresh_cpp/ema.h, sfpi::transp8 + typed
# cross-call state) — the row is now a measured full2x2.  cumsum-fresh
# likewise.  moegatetopk-fresh and sfpureduce-fresh keep their skips (their
# blockers — bitonic SFPSHFT2 networks raced on the SAME vehicle as the
# lifted arms, and the S4 subvec_shflror1 ruling — are owned elsewhere).
check(
    "ops v4: remaining feasibility-class skips are untouched (moegatetopk/sfpureduce)",
    all(
        by[o]["kind"] == "skip" and "SKIP" in by[o]["note"]
        for o in ("moegatetopk-fresh", "sfpureduce-fresh")
    ),
)
check(
    "ops v4: ema-fresh feasibility skip DISSOLVED by the cross-lane surface (lane FK)",
    by["ema-fresh"]["kind"] == "full2x2"
    and "DISSOLVED" in by["ema-fresh"]["note"]
    and by["cumsum-fresh"]["kind"] == "full2x2",
)

# invalid sem_class must refuse at load
with tempfile.TemporaryDirectory() as td:
    bad = pathlib.Path(td) / "ops.tsv"
    hdr = "op\tcorpus_id\tkind\tmarker\tcraq_archs\tsem_corr\tsem_perf\thand_corr\thand_perf\tnote\tmetric\tper_tile\tissue_slot_lb\tpin_flags\textra_env\tsem_extra_env\thand_extra_env\tschedule\tsem_class\n"
    bad.write_text(
        hdr
        + "x\tcid\tsemantic\tKERNEL\tbh\tc.py::t\t\t\t\tn\tMATH_ISOLATE\t0\t\t\t\t\t\tnightly\tbogus\n"
    )
    try:
        sweep.load_config(bad)
        check("ops v4: invalid sem_class refuses at load", False)
    except SystemExit as e:
        check(
            "ops v4: invalid sem_class refuses at load",
            "sem_class" in str(e),
            str(e),
        )

# ---------------- 7. scoreboard + wrapper + lint wiring ----------------

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    sw = mk_sweep(td / "ev")
    sw.info = {
        "cc1plus_sha256": "c" * 64,
        "compiler_sha256": "d" * 64,
        "sim_bh_sha256": "e" * 64,
    }
    res = mk_result(
        "sbop",
        {"sem_on": 80.0, "hand_on": 85.0},
        {"sem_on": 9500.0, "hand_on": 9600.0},
        kernel_vs_hand_pct=-1.04,
        vs_hand_pct=-5.88,
    )
    res["kind"] = "full2x2"
    sw.emit_scoreboard([res], [])
    tsv = (sw.ev / "scoreboard.tsv").read_text()
    check(
        "scoreboard.tsv carries KERNEL-scope rows (v2 seeding source)",
        "KERNEL_MATH_ISOLATE_E2E\tsbop:sem_on\t9500.0" in tsv
        and "TILE_LOOP_MATH_ISOLATE_PER_TILE\tsbop:sem_on\t80.0" in tsv,
        tsv,
    )
    md = (sw.ev / "SCOREBOARD.md").read_text()
    check(
        "SCOREBOARD.md carries the e2e columns",
        "e2e vs hand" in md and "9500.000" in md,
        md,
    )

wiring = []
for w in ("weekly_bh_sweep.sh", "nightly_bh_sweep.sh", "headline_bh_sweep.sh"):
    t = (HERE / w).read_text()
    wiring.append("--kernel-baseline" in t and "_v2.tsv" in t)
check("wrappers pass --kernel-baseline when the v2 baseline exists", all(wiring))
check(
    "conf carries the verdict-metric ratification + ES-F1 flush config",
    "END-TO-END DEVICE KERNEL TIME" in (HERE / "sweep_2x2.conf").read_text()
    and "SWEEP_FLUSH_SH" in (HERE / "sweep_2x2.conf").read_text(),
)
check(
    "conf_lint anchors both baselines (R5b/R6b for the v2 header)",
    "R5b" in (HERE / "conf_lint.sh").read_text(),
)

# ---------------- verdict ----------------

if FAILS:
    print(f"\nSELFTEST: {len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("\nSELFTEST: ALL GREEN")
