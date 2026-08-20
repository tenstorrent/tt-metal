#!/usr/bin/env python3
"""Self-test for the sweep_2x2.py pipeline overhaul (laneDC, owner order).

Drives the REAL sweep code (imported, not re-implemented) with filesystem
fixtures and stubbed workers — no toolchain, no simulator, no device:

  1. PIPELINING (kill the phase barrier): a 3-op run through the real
     _pipeline_run/_admission_waves where the wave-1 classify stub BLOCKS
     until it has seen wave-0 silicon execute — the test can only pass if
     silicon batches genuinely begin while classify continues (a
     phase-barrier implementation deadlocks and fails by timeout).  Wave
     scoping, result order, and gating-thread SystemExit forwarding are
     asserted too.
  2. ROW PRIORITY SCHEDULING: --priority-ops jumps the queue entirely (in
     the order given); rows with prior CHANGED verdicts (or no history)
     order before rows whose prev-run verdict says IDENTICAL or whose
     baseline class is refusal (re-baseline rows last); stable otherwise.
  3. CROSS-PIN CELL REUSE: a synthetic --prev-run root with a
     jobkey+hash-matching cell is ADOPTED (evidence copied, REUSED_FROM.txt
     marker, resume returns green without re-running); hash-MISmatching,
     tampered-jobkey, tampered-hash and failed-rc cells are REFUSED and
     re-run; expected_texts=None and --force never adopt; with two prev
     roots the NEWEST (first) wins; the batched executor's _job_cached
     adopts under the same checks.
  3b. SOURCE-ROOT PROVENANCE GATE (wave-12 ledger 19): a root with a
     recorded FOREIGN pin adopts with the pin recorded (REUSED_FROM.txt +
     reused_cells source_pin) and a loud CROSS-PIN ADOPTION line; a root
     with NO pin record, a QUARANTINED marker or a CONTAMINATION-NOTE.md
     REFUSES; a craq-gate-tainted root refuses against an untainted run
     and adopts against a taint-matched run WITH the taint line propagated
     into this run's MANIFEST; transitive adoption preserves the full
     origin chain oldest-first (the silicon run is always entry 0).
  4. ROW VERDICT STREAMING: _emit_row_verdict writes
     <evidence-root>/<op>/ROW-VERDICT.json whose verdict/details/report_row
     equal the final report()'s row line for the same op, byte-exact, and
     whose rag matches the report's overall verdict on a single-row run.

Run standalone or from the nightly/weekly wrappers; exits nonzero on any
failure so a broken gate can never bless a sweep.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import threading

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sweep_2x2", HERE / "sweep_2x2.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

FAILS = []

# Synthetic toolchain pins for the source-root provenance gate fixtures:
# RUNPIN is THIS run's cc1plus pin (mk_sweep wires it into sw.info the way
# the real preflight does); the others mark foreign/prior runs.
RUNPIN = "c" * 64
FOREIGNPIN = "d" * 64
ORIGINPIN = "e" * 64
HOPPIN = "f" * 64


def check(name, cond, detail=""):
    if cond:
        print(f"SELFTEST PASS: {name}")
    else:
        print(f"SELFTEST FAIL: {name} {detail}")
        FAILS.append(name)


def mk_row(op, kind="semantic", nodes=None, corpus_id=None):
    return {
        "op": op,
        "corpus_id": corpus_id or f"fix__{op}",
        "kind": kind,
        "marker": "TILE_LOOP",
        "metric": "MATH_ISOLATE",
        "per_tile": True,
        "issue_slot_lb": None,
        "pin_flags": "",
        "extra_env": {},
        "sel_extra_env": {"sem": {}, "hand": {}},
        "nodes": {
            sel: (nodes or {}).get(sel, "")
            for sel in ("sem-corr", "sem-perf", "hand-corr", "hand-perf")
        },
    }


def mk_sweep(ev, **args):
    sw = object.__new__(sweep.Sweep)
    base = dict(
        force=False,
        dry_run=True,
        knob_silicon_rows=None,
        knob_attribution=False,
        skip_craq_gate=True,
        classify_workers=2,
        priority_ops=None,
        admit_wave_rows=8,
        prev_run=None,
        baseline=None,
    )
    base.update(args)
    sw.a = argparse.Namespace(**base)
    sw.ev = pathlib.Path(ev)
    sw.ev.mkdir(parents=True, exist_ok=True)
    sw.python = pathlib.Path(sys.executable)
    sw.reds = []
    sw.reused = []
    sw.exec_mode = "batched"
    sw.info = {
        "cc1plus_sha256": RUNPIN,
        "compiler_sha256": "selftest",
        "sim_bh_sha256": "",
    }
    return sw


# ---------------- 1. pipelining: silicon overlaps classify ----------------
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    sw = mk_sweep(td / "ev", priority_ops=["op0"], admit_wave_rows=1)
    rows = [mk_row(f"op{i}", nodes={"sem-corr": f"t.py::t[{i}]"}) for i in range(3)]

    waves = sw._admission_waves(rows)
    check(
        "pipeline: --priority-ops forms wave 0, later rows in "
        "--admit-wave-rows waves",
        [[r["op"] for r in w] for w in waves] == [["op0"], ["op1"], ["op2"]],
        waves,
    )
    sw2 = mk_sweep(td / "ev2")
    check(
        "pipeline: without priority ops the first wave is small "
        f"({sweep.Sweep._FIRST_WAVE_ROWS} rows) so silicon starts early",
        [len(w) for w in sw2._admission_waves([mk_row(f"o{i}") for i in range(12)])]
        == [3, 8, 1],
    )

    events = []
    wave0_silicon_done = threading.Event()

    def fake_prewarm(wrows, phases):
        ops = [r["op"] for r in wrows]
        if "op1" in ops:
            # The proof of pipelining: wave-1 classify refuses to complete
            # until wave-0 SILICON has executed.  A phase-barrier flow
            # (classify everything, then device work) deadlocks here and
            # the timeout fails the test.
            saw = wave0_silicon_done.wait(timeout=60)
            events.append(("classify-w1-saw-wave0-silicon", saw))
        events.append(("classify", tuple(ops)))

    def fake_gate_one(row, phases):
        return (row, {"sem-corr": {"status": "OK", "all": "CHANGED"}}, None)

    def fake_gate_rows(prelim):
        return [("go", r, c, a) for r, c, a in prelim]

    def fake_silicon_phase(slots, wave=None):
        events.append(("silicon", wave, tuple(s[1]["op"] for s in slots)))
        if wave == "w0":
            wave0_silicon_done.set()
        return [{"op": s[1]["op"]} for s in slots]

    sw._classify_prewarm = fake_prewarm
    sw._gate_one_row = fake_gate_one
    sw._gate_rows = fake_gate_rows
    sw._silicon_phase = fake_silicon_phase

    results = sw._pipeline_run(["classify", "silicon", "report"], rows)
    check(
        "pipeline: silicon batches begin BEFORE classify completes "
        "(wave-1 classify observed wave-0 silicon; no deadlock)",
        ("classify-w1-saw-wave0-silicon", True) in events,
        events,
    )
    check(
        "pipeline: every wave executed with its own wave scope",
        [e for e in events if e[0] == "silicon"]
        == [
            ("silicon", "w0", ("op0",)),
            ("silicon", "w1", ("op1",)),
            ("silicon", "w2", ("op2",)),
        ],
        events,
    )
    check(
        "pipeline: results stream in admission (priority) order",
        [r["op"] for r in results] == ["op0", "op1", "op2"],
        results,
    )

    # gating-thread failure forwarding: SystemExit must re-raise in main
    sw3 = mk_sweep(td / "ev3")

    def exploding_prewarm(wrows, phases):
        raise SystemExit("TOOLCHAIN CHANGED MID-RUN (selftest)")

    sw3._classify_prewarm = exploding_prewarm
    try:
        sw3._pipeline_run(["classify", "silicon"], [mk_row("boom")])
        check("pipeline: gating SystemExit re-raises in the main thread", False)
    except SystemExit as e:
        check(
            "pipeline: gating SystemExit re-raises in the main thread",
            "TOOLCHAIN CHANGED" in str(e),
            str(e),
        )

# ---------------- 2. row priority scheduling ----------------
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    prev = td / "prev-run"
    baseline = td / "baseline.tsv"
    baseline.write_text(
        "# schema=2 fixture\n"
        "id\tarch\tmetric\tscope\tselector\tcycles\tstatus\texpected_class\t"
        "compiler_sha\tprovenance\n"
        "fix__opC\tbh\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\t"
        "opC:sem_off\t\trefusal\trefusal\tselftest\tfixture\n"
    )
    for op, verdict in (("opA", "CHANGED"), ("opB", "IDENTICAL")):
        vd = prev / op / "classify" / "sem-perf"
        vd.mkdir(parents=True)
        (vd / "verdict.json").write_text(
            json.dumps({"selector": "sem-perf", "status": "OK", "all": verdict})
        )
    nodes = {"sem-perf": "p.py::t", "sem-corr": "t.py::t"}
    rows = [mk_row(op, nodes=nodes) for op in ("opA", "opB", "opC", "opD")]

    sw = mk_sweep(td / "ev", prev_run=[prev], baseline=baseline)
    sw.rows = [dict(r) for r in rows]
    sw._order_rows()
    check(
        "priority: expected-CHANGED/unknown rows first, expected "
        "byte-identical (prev IDENTICAL verdict / baseline refusal) last",
        [r["op"] for r in sw.rows] == ["opA", "opD", "opB", "opC"],
        [r["op"] for r in sw.rows],
    )

    sw = mk_sweep(
        td / "ev4", prev_run=[prev], baseline=baseline, priority_ops=["opC", "opB"]
    )
    sw.rows = [dict(r) for r in rows]
    sw._order_rows()
    check(
        "priority: --priority-ops jumps the queue entirely, in the order "
        "given (even expected-identical rows)",
        [r["op"] for r in sw.rows] == ["opC", "opB", "opA", "opD"],
        [r["op"] for r in sw.rows],
    )

    # a CHANGED verdict in the CURRENT run root outranks the baseline hint
    sw = mk_sweep(td / "ev5", baseline=baseline)
    vd = sw.ev / "opC" / "classify" / "sem-perf"
    vd.mkdir(parents=True)
    (vd / "verdict.json").write_text(
        json.dumps({"selector": "sem-perf", "status": "OK", "all": "CHANGED"})
    )
    sw.rows = [dict(r) for r in rows]
    sw._order_rows()
    check(
        "priority: a fresh CHANGED verdict outranks the baseline refusal "
        "hint (hints never pin a row to the back)",
        [r["op"] for r in sw.rows].index("opC") < 3,
        [r["op"] for r in sw.rows],
    )

# ---------------- 3. cross-pin cell reuse ----------------
NODE = "p.py::test_x[mathop:Fix-impl:1]"
FLAGS = "-mfix-flags"


def fab_cell(root, jobkey, texts, rc="0", log="1 passed in 1.00s\n"):
    work = root / "fixop" / "silicon" / "sem-perf" / "r1-off"
    work.mkdir(parents=True, exist_ok=True)
    (work / "rc.txt").write_text(f"{rc}\n")
    (work / "log.txt").write_text(log)
    (work / "TEXT_HASHES.txt").write_text(
        "".join(f"k{i}.elf\ttext:{t}\telf:{t}\n" for i, t in enumerate(texts))
    )
    (work / "jobkey.json").write_text(json.dumps(jobkey) + "\n")
    (work / "node.txt").write_text(NODE + "\n")
    return work


def reuse_case(
    name,
    prev_roots,
    want_adopt,
    key_over=None,
    texts=None,
    rc="0",
    expected=("aaa",),
    force=False,
    want_from=None,
):
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        row = mk_row("fixop", nodes={"sem-perf": NODE, "sem-corr": "t.py::t"})
        goodkey = {
            "node": NODE,
            "flags": FLAGS,
            "extra_env": {},
            "tag": "silicon",
            "mode": "serial",
        }
        roots = []
        for i in range(prev_roots):
            root = td / f"prev{i}"
            fab_cell(
                root,
                dict(goodkey, **(key_over or {})),
                list(texts or ["aaa"]),
                rc=rc,
            )
            # same-pin, untainted source root (the cell-level cases here
            # isolate the CELL checks; the source-ROOT provenance gate has
            # its own case block below)
            (root / "PIN_STAMP").write_text(RUNPIN + "\n")
            roots.append(root)
        sw = mk_sweep(td / "ev", prev_run=roots, force=force)
        sw.exec_mode = "serial"
        work = sw.ev / "fixop" / "silicon" / "sem-perf" / "r1-off"
        sw._device_job(
            row, "sem-perf", "r1", "off", FLAGS, expected_texts=list(expected)
        )
        adopted = (work / "REUSED_FROM.txt").is_file() and (work / "rc.txt").is_file()
        ok = adopted == want_adopt
        if ok and want_adopt and want_from is not None:
            ok = (
                f"reused-from:{td / want_from}"
                in (work / "REUSED_FROM.txt").read_text()
            )
        check(name, ok, f"(adopted={adopted}, expected {want_adopt})")
        return sw


reuse_case(
    "reuse: jobkey+hash-matched prev cell is ADOPTED with a REUSED_FROM "
    "marker (resume green, no re-run)",
    prev_roots=1,
    want_adopt=True,
)
reuse_case(
    "reuse: two prev roots — the NEWEST (first listed) wins",
    prev_roots=2,
    want_adopt=True,
    want_from="prev0",
)
reuse_case(
    "reuse: hash-MISmatching prev cell is refused (re-run)",
    prev_roots=1,
    want_adopt=False,
    texts=["bbb"],
)
reuse_case(
    "reuse: tampered jobkey (flags edited) is refused",
    prev_roots=1,
    want_adopt=False,
    key_over={"flags": "-mtampered"},
)
reuse_case(
    "reuse: tampered TEXT_HASHES (edited after the run) is refused",
    prev_roots=1,
    want_adopt=False,
    texts=["aaa", "extra"],
)
reuse_case(
    "reuse: failed prev cell (rc!=0) is refused",
    prev_roots=1,
    want_adopt=False,
    rc="1",
)
reuse_case(
    "reuse: execution-mode mismatch (batched prev cell, serial run) refused",
    prev_roots=1,
    want_adopt=False,
    key_over={"mode": "batched"},
)
reuse_case(
    "reuse: --force never adopts",
    prev_roots=1,
    want_adopt=False,
    force=True,
)

# expected_texts=None never adopts (no classify reference = no trust)
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    goodkey = {
        "node": NODE,
        "flags": FLAGS,
        "extra_env": {},
        "tag": "silicon",
        "mode": "serial",
    }
    fab_cell(td / "prev0", goodkey, ["aaa"])
    (td / "prev0" / "PIN_STAMP").write_text(RUNPIN + "\n")
    sw = mk_sweep(td / "ev", prev_run=[td / "prev0"])
    sw.exec_mode = "serial"
    row = mk_row("fixop", nodes={"sem-perf": NODE, "sem-corr": "t.py::t"})
    sw._device_job(row, "sem-perf", "r1", "off", FLAGS, expected_texts=None)
    work = sw.ev / "fixop" / "silicon" / "sem-perf" / "r1-off"
    check(
        "reuse: expected_texts=None never adopts",
        not (work / "REUSED_FROM.txt").is_file(),
    )

# batched executor path: _job_cached adopts under the same checks
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    row = mk_row("fixop", nodes={"sem-perf": NODE, "sem-corr": "t.py::t"})
    batchedkey = {
        "node": NODE,
        "flags": FLAGS,
        "extra_env": {},
        "tag": "silicon",
        "mode": "batched",
    }
    fab_cell(td / "prev0", batchedkey, ["aaa"])
    (td / "prev0" / "PIN_STAMP").write_text(RUNPIN + "\n")
    sw = mk_sweep(td / "ev", prev_run=[td / "prev0"], dry_run=False)
    hf = sw.ev / "fixop" / "classify" / "sem-perf" / "hashes-off.txt"
    hf.parent.mkdir(parents=True)
    hf.write_text("k0.elf\ttext:aaa\telf:aaa\n")
    job = sw._mk_job(row, "sem-perf", "r1", "off", FLAGS, "perf")
    cached = sw._job_cached(job)
    check(
        "reuse: batched executor _job_cached adopts a matching prev cell "
        "(leg leaves the pending set) and books provenance",
        cached
        and (job["work"] / "REUSED_FROM.txt").is_file()
        and sw.reused
        and sw.reused[0]["reused_from"].endswith("prev0"),
        (cached, sw.reused),
    )
    # scoreboard provenance
    sw.info = {
        "cc1plus_sha256": "cc",
        "compiler_sha256": "dd",
        "sim_bh_sha256": "",
    }
    sw.emit_scoreboard([], [])
    payload = json.loads((sw.ev / "scoreboard.json").read_text())
    check(
        "reuse: scoreboard.json carries reused_cells provenance "
        "(leg + source_pin + source_taint + origin_root)",
        payload.get("reused_cells")
        and payload["reused_cells"][0]["leg"] == "fixop/silicon/sem-perf/r1-off"
        and payload["reused_cells"][0].get("source_pin") == RUNPIN[:12]
        and payload["reused_cells"][0].get("source_taint") is False
        and str(payload["reused_cells"][0].get("origin_root", "")).endswith("prev0"),
        payload.get("reused_cells"),
    )

# ---------------- 3b. source-root provenance gate (wave-12 ledger 19) ------


def prov_case(name, want_adopt, setup, skip_craq_gate=True, extra_checks=None):
    """One constructed source-root negative/positive: fabricate a cell that
    passes every CELL-level check (green, jobkey, .text match), shape the
    ROOT's provenance via setup(root), and assert the adoption verdict —
    the gate under test is the ROOT gate alone."""
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        row = mk_row("fixop", nodes={"sem-perf": NODE, "sem-corr": "t.py::t"})
        goodkey = {
            "node": NODE,
            "flags": FLAGS,
            "extra_env": {},
            "tag": "silicon",
            "mode": "serial",
        }
        root = td / "prev0"
        fab_cell(root, goodkey, ["aaa"])
        setup(root)
        sw = mk_sweep(td / "ev", prev_run=[root], skip_craq_gate=skip_craq_gate)
        sw.exec_mode = "serial"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            sw._device_job(
                row, "sem-perf", "r1", "off", FLAGS, expected_texts=["aaa"]
            )
        work = sw.ev / "fixop" / "silicon" / "sem-perf" / "r1-off"
        adopted = (work / "REUSED_FROM.txt").is_file()
        ok = adopted == want_adopt
        detail = f"(adopted={adopted}, expected {want_adopt})"
        if ok and extra_checks:
            ok, detail = extra_checks(sw, work, out.getvalue())
        check(name, ok, detail)


def _foreign_pin_checks(sw, work, stdout):
    marker = (work / "REUSED_FROM.txt").read_text()
    ok = (
        f"pin:{FOREIGNPIN}" in marker
        and sw.reused
        and sw.reused[0].get("source_pin") == FOREIGNPIN[:12]
        and sw.reused[0].get("source_taint") is False
        and str(sw.reused[0].get("origin_root", "")).endswith("prev0")
        and "CROSS-PIN ADOPTION" in stdout
        and FOREIGNPIN[:12] in stdout
    )
    return ok, (marker, sw.reused, stdout)


prov_case(
    "provenance: FOREIGN-pin root (pin recorded) ADOPTS — source_pin in "
    "REUSED_FROM + reused_cells, loud CROSS-PIN ADOPTION line",
    want_adopt=True,
    setup=lambda root: (root / "PIN_STAMP").write_text(FOREIGNPIN + "\n"),
    extra_checks=_foreign_pin_checks,
)
prov_case(
    "provenance: root with NO pin record (no PIN_STAMP, no preflight.json) "
    "is REFUSED fail-closed",
    want_adopt=False,
    setup=lambda root: None,
)
prov_case(
    "provenance: QUARANTINED-marked root is REFUSED even with a matching "
    "pin record",
    want_adopt=False,
    setup=lambda root: (
        (root / "PIN_STAMP").write_text(RUNPIN + "\n"),
        (root / "QUARANTINED").write_text(""),
    ),
)
prov_case(
    "provenance: CONTAMINATION-NOTE.md root is REFUSED even with a "
    "matching pin record",
    want_adopt=False,
    setup=lambda root: (
        (root / "PIN_STAMP").write_text(RUNPIN + "\n"),
        (root / "CONTAMINATION-NOTE.md").write_text("known-mixed\n"),
    ),
)


def _tainted_preflight(root):
    (root / "preflight.json").write_text(
        json.dumps({"cc1plus_sha256": RUNPIN, "craq_gate_skipped": True}) + "\n"
    )


prov_case(
    "provenance: craq-gate-TAINTED root is REFUSED by an untainted run "
    "(taint mismatch)",
    want_adopt=False,
    setup=_tainted_preflight,
    skip_craq_gate=False,
)


def _taint_match_checks(sw, work, stdout):
    manifest = sw.ev / "MANIFEST.txt"
    ok = (
        sw.reused
        and sw.reused[0].get("source_taint") is True
        and manifest.is_file()
        and "ADOPTED-CELL TAINT" in manifest.read_text()
        and "craq_gate_skipped" in manifest.read_text()
    )
    return ok, (sw.reused, manifest.read_text() if manifest.is_file() else "")


prov_case(
    "provenance: taint-MATCHED run (both --skip-craq-gate) ADOPTS and "
    "propagates the taint line into this run's MANIFEST",
    want_adopt=True,
    setup=_tainted_preflight,
    skip_craq_gate=True,
    extra_checks=_taint_match_checks,
)

# transitive chain: origin (touched silicon) -> runB adopts -> runC adopts
# from runB; the final REUSED_FROM.txt must list the FULL chain oldest-first
# with per-hop pins — never overwrite/launder the origin.
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    row = mk_row("fixop", nodes={"sem-perf": NODE, "sem-corr": "t.py::t"})
    goodkey = {
        "node": NODE,
        "flags": FLAGS,
        "extra_env": {},
        "tag": "silicon",
        "mode": "serial",
    }
    origin = td / "origin-run"
    fab_cell(origin, goodkey, ["aaa"])
    (origin / "PIN_STAMP").write_text(ORIGINPIN + "\n")
    swb = mk_sweep(td / "runB", prev_run=[origin])
    swb.exec_mode = "serial"
    with contextlib.redirect_stdout(io.StringIO()):
        swb._device_job(row, "sem-perf", "r1", "off", FLAGS, expected_texts=["aaa"])
    (td / "runB" / "PIN_STAMP").write_text(HOPPIN + "\n")
    swc = mk_sweep(td / "runC", prev_run=[td / "runB"])
    swc.exec_mode = "serial"
    with contextlib.redirect_stdout(io.StringIO()):
        swc._device_job(row, "sem-perf", "r1", "off", FLAGS, expected_texts=["aaa"])
    marker_f = td / "runC" / "fixop" / "silicon" / "sem-perf" / "r1-off" / (
        "REUSED_FROM.txt"
    )
    marker = marker_f.read_text() if marker_f.is_file() else "(no marker)"
    check(
        "provenance: TRANSITIVE adoption preserves the full chain "
        "oldest-first (silicon origin is entry 0, per-hop pins recorded, "
        "origin_root booked)",
        f"  {origin} pin:{ORIGINPIN}\n  {td / 'runB'} pin:{HOPPIN}\n" in marker
        and swc.reused
        and swc.reused[0].get("origin_root") == str(origin)
        and swc.reused[0].get("source_pin") == HOPPIN[:12],
        (marker, getattr(swc, "reused", None)),
    )

# ---------------- 4. row verdict streaming ----------------
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    baseline = td / "baseline.tsv"
    baseline.write_text(
        "# schema=2 fixture\n"
        "id\tarch\tmetric\tscope\tselector\tcycles\tstatus\texpected_class\t"
        "compiler_sha\tprovenance\n"
        "fix__winop\tbh\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\t"
        "winop:sem_off\t100.0\tmeasured\twin\tselftest\tfixture\n"
        "fix__winop\tbh\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\t"
        "winop:sem_on\t80.0\tmeasured\twin\tselftest\tfixture\n"
        "fix__winop\tbh\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\t"
        "winop:hand_on\t82.0\tmeasured\twin\tselftest\tfixture\n"
    )
    result = {
        "op": "winop",
        "corpus_id": "fix__winop",
        "kind": "semantic",
        "marker": "TILE_LOOP",
        "scope": "TILE_LOOP_MATH_ISOLATE_PER_TILE",
        "classify": {},
        "cells": {"sem_off": 100.0, "sem_on": 79.0, "hand_on": 82.0},
        "runs": {},
        "notes": [],
        "causal_pct": -21.0,
        "vs_hand_pct": -3.66,
    }
    sw = mk_sweep(
        td / "ev",
        baseline=baseline,
        max_drift_pct=5.0,
        max_abs_drift_pct=10.0,
        red_loss_growth_pct=5.0,
        allow_win_to_parity=False,
        skip_craq_gate=False,
    )
    payload = sw._emit_row_verdict(result)
    vf = sw.ev / "winop" / "ROW-VERDICT.json"
    check(
        "streaming: ROW-VERDICT.json written at row completion with cells, "
        "vs_hand %, class band and verdict",
        vf.is_file()
        and payload["cells"]["sem_on"] == 79.0
        and payload["vs_hand_pct"] == -3.66
        and payload["class"] == "WIN"
        and payload["verdict"] == "ok",
        payload,
    )
    rag = sw.report([result], [])
    report = (sw.ev / "REPORT.md").read_text()
    check(
        "streaming: ROW-VERDICT content == the final report row for the "
        "same op (byte-equal table line)",
        payload["report_row"] in report.splitlines(),
        (payload["report_row"], report),
    )
    check(
        "streaming: streamed rag matches the report overall on a " "single-row run",
        rag == payload["rag"] == "GREEN",
        (rag, payload["rag"]),
    )
    on_disk = json.loads(vf.read_text())
    check(
        "streaming: the on-disk JSON is the emitted payload",
        on_disk == payload,
    )
    # a REGRESSING row streams RED at completion time, same as the report
    bad = dict(
        result,
        cells={"sem_off": 200.0, "sem_on": 190.0},
        causal_pct=-5.0,
        vs_hand_pct=131.7,
    )
    sw2 = mk_sweep(
        td / "ev2",
        baseline=baseline,
        max_drift_pct=5.0,
        max_abs_drift_pct=10.0,
        red_loss_growth_pct=5.0,
        allow_win_to_parity=False,
        skip_craq_gate=False,
    )
    payload2 = sw2._emit_row_verdict(bad)
    rag2 = sw2.report([bad], [])
    check(
        "streaming: a regressing row streams RED at completion, matching "
        "the final report verdict",
        payload2["rag"] == "RED" == rag2
        and payload2["report_row"] in (sw2.ev / "REPORT.md").read_text(),
        (payload2["rag"], rag2),
    )

if FAILS:
    print(f"sweep-core-overhaul self-test: FAILED ({len(FAILS)}: {', '.join(FAILS)})")
    sys.exit(1)
print(
    "sweep-core-overhaul self-test: ALL GREEN (pipelining proven by "
    "blocking handshake — silicon executes while classify continues, wave "
    "scoping + admission order + SystemExit forwarding; priority order "
    "measure-first with --priority-ops queue jump and fresh-verdict "
    "override; cross-pin reuse adopts jobkey+hash-matched prev cells with "
    "REUSED_FROM provenance and refuses mismatch/tamper/failed/mode/force/"
    "no-reference; source-root provenance gate refuses quarantined/"
    "contaminated/unpinned/taint-mismatched roots, records foreign pins "
    "loudly, propagates matched taint to MANIFEST and preserves transitive "
    "origin chains; ROW-VERDICT.json streams the byte-equal report row and "
    "rag at completion time)"
)
