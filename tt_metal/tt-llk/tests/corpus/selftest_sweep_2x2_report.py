#!/usr/bin/env python3
"""Self-test for sweep_2x2.py report() class-aware flip detection (defect D4,
PULL_ANALYSIS-20260817 §2c) and the sweep-hardening round-2 fixes
(adversarial review 2026-08-16).

Drives the REAL report() (and _device_job/_bh_craq_gate/load_config) with
synthetic fixtures and asserts:
  1. WIN -> byte-identical REFUSAL  = RED   (the D4 hole: was GREEN);
  2. refusal -> refusal             = GREEN;
  3. refusal -> changed/measured    = flagged notice (YELLOW), never silent;
  4. win -> win (within drift)      = GREEN;
  5. INVALID_MARKER cell            = RED.
Round-2 regression cases (each is a verifier reproduction recipe):
  6. all-None cells on a row with baseline history = INVALID_METRIC RED
     (was: 'no silicon cells this run', GREEN);
  7. uniform slowdown (+50% both legs, ratios preserved) = ABS CYCLES RED
     (was: GREEN — ratio-only acceptance);
  8. hand-leg 2x regression on a refusal-class row = RED
     (was: GREEN — hand cells fed no comparison without a measured sem leg);
  9. win eroding to parity (-5% -> -0.3%) = RED by default,
     YELLOW under --allow-win-to-parity;
 10. loss growing beyond --red-loss-growth-pct (+10% -> +16%) = RED
     (was: YELLOW, exit 0, unalertable);
 11. YELLOW rows show verdict column 'YELLOW', never 'ok';
 12. _device_job cache: expected_texts=None never reuses; a changed node id
     or extra_env (jobkey) never reuses; a keyed hash-matched cache reuses;
 13. the BH CRAQ silicon gate rejects verdicts keyed to another
     cc1plus/simulator/tt-metal head, accepts this run's keys;
 14. ops-load validation: a perf selector without its correctness selector
     fails loudly.

Run standalone or from the nightly/weekly wrappers; exits nonzero on any
failure so a broken gate can never bless a sweep.
"""
import argparse
import hashlib
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sweep_2x2 import Sweep, load_config  # noqa: E402

BASELINE = """\
# schema=2; synthetic self-test baseline; expected_class column drives the class-aware gate
id\tarch\tchip_class\tmetric\tscope\tselector\tcycles\tstatus\texpected_class\tcompiler_sha\tprovenance
fix__winop\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\twinop:sem_off\t100.0\tmeasured\twin\tselftest\tfixture
fix__winop\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\twinop:sem_on\t80.0\tmeasured\twin\tselftest\tfixture
fix__winop\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\twinop:hand_off\t80.0\tmeasured\twin\tselftest\tfixture
fix__winop\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\twinop:hand_on\t80.0\tmeasured\twin\tselftest\tfixture
fix__refop\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\trefop:sem_off\t\trefusal\trefusal\tselftest\tfixture
fix__refop\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\trefop:hand_off\t50.0\tmeasured\trefusal\tselftest\tfixture
fix__refop\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\trefop:hand_on\t50.0\tmeasured\trefusal\tselftest\tfixture
fix__smallwin\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\tsmallwin:sem_off\t100.0\tmeasured\twin\tselftest\tfixture
fix__smallwin\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\tsmallwin:sem_on\t95.0\tmeasured\twin\tselftest\tfixture
fix__lossop\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\tlossop:sem_off\t100.0\tmeasured\tloss\tselftest\tfixture
fix__lossop\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\tlossop:sem_on\t110.0\tmeasured\tloss\tselftest\tfixture
fix__shared\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\ttwinop-fresh:sem_off\t100.0\tmeasured\tloss\tselftest\tGE-F1 fixture: fresh twin, measured
fix__shared\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\ttwinop-fresh:sem_on\t110.0\tmeasured\tloss\tselftest\tGE-F1 fixture: fresh twin, measured
fix__shared\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\ttwinop:sem_off\t\trefusal_byte_identical\trefusal\tselftest\tGE-F1 fixture: parent refusal AFTER the twin (the overwrite direction)
fix__shared\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\ttwinop:sem_on\t\trefusal_byte_identical\trefusal\tselftest\tGE-F1 fixture: parent refusal AFTER the twin (the overwrite direction)
"""


def make_result(op, corpus_id, cells, **extra):
    r = {
        "op": op,
        "corpus_id": corpus_id,
        "kind": "semantic",
        "marker": "TILE_LOOP",
        "scope": "TILE_LOOP_MATH_ISOLATE_PER_TILE",
        "classify": {},
        "cells": cells,
        "runs": {},
        "notes": [],
    }
    r.update(extra)
    return r


def run_report(tmp, name, results, **overrides):
    ev = tmp / name
    ev.mkdir()
    sw = object.__new__(Sweep)
    args = dict(
        baseline=tmp / "baseline.tsv",
        prev_run=None,
        max_drift_pct=5.0,
        max_abs_drift_pct=10.0,
        red_loss_growth_pct=5.0,
        allow_win_to_parity=False,
    )
    args.update(overrides)
    sw.a = argparse.Namespace(**args)
    sw.ev = ev
    sw.reds = []
    rag = sw.report(results, [])
    return rag, (ev / "REPORT.md").read_text()


def main():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="selftest-sweep-report-"))
    (tmp / "baseline.tsv").write_text(BASELINE)
    failures = []

    def check(name, want_rag, got_rag, report, want_text=None):
        ok = got_rag == want_rag and (want_text is None or want_text in report)
        print(
            f"SELFTEST {'PASS' if ok else 'FAIL'}: {name} "
            f"(rag={got_rag}, expected {want_rag})"
        )
        if not ok:
            failures.append(name)
            print(report)

    # 1. The D4 hole: a prior measured WIN row that becomes a byte-identical
    #    refusal must be RED, not "refusal byte-identical: GREEN".
    rag, rep = run_report(
        tmp,
        "case1",
        [
            make_result(
                "winop",
                "fix__winop",
                {
                    "sem_off": "REFUSAL_BYTE_IDENTICAL",
                    "sem_on": "REFUSAL_BYTE_IDENTICAL",
                },
            ),
        ],
    )
    check("win→refusal flip is RED", "RED", rag, rep, "WIN→REFUSAL FLIP")

    # 2. Expected refusal stays a refusal: GREEN.
    rag, rep = run_report(
        tmp,
        "case2",
        [
            make_result(
                "refop",
                "fix__refop",
                {
                    "sem_off": "REFUSAL_BYTE_IDENTICAL",
                    "sem_on": "REFUSAL_BYTE_IDENTICAL",
                },
            ),
        ],
    )
    check(
        "refusal→refusal is GREEN",
        "GREEN",
        rag,
        rep,
        "refusal byte-identical (baseline class refusal): GREEN",
    )

    # 3. Expected refusal now changed and measured: flagged notice (YELLOW).
    rag, rep = run_report(
        tmp,
        "case3",
        [
            make_result(
                "refop",
                "fix__refop",
                {"sem_off": 100.0, "sem_on": 90.0},
                causal_pct=-10.0,
            ),
        ],
    )
    check("refusal→changed is flagged YELLOW", "YELLOW", rag, rep, "REFUSAL→CHANGED")

    # 3b. GE-F1 INJECTION (lane GF): a corpus TU id SHARED between a parent
    #     op and its fresh twin (mulint32 / mulint32-fresh class).  The
    #     parent's refusal_byte_identical baseline rows come AFTER the
    #     twin's measured rows in the TSV — with the (id, scope)-keyed
    #     class map they OVERWROTE the twin's class and every twin re-run
    #     minted a structural REFUSAL→CHANGED YELLOW no anchor refresh
    #     could clear.  With the selector-op-prefix keying: the measured
    #     twin is GREEN (no refusal notice), the parent's refusal stays a
    #     GREEN expected refusal.
    rag, rep = run_report(
        tmp,
        "case3b-twin",
        [
            make_result(
                "twinop-fresh",
                "fix__shared",
                {"sem_off": 100.0, "sem_on": 110.0},
                causal_pct=10.0,
            ),
        ],
    )
    ok = rag == "GREEN" and "REFUSAL→CHANGED" not in rep
    print(
        f"SELFTEST {'PASS' if ok else 'FAIL'}: GE-F1 fresh twin on a shared "
        f"TU id books its OWN measured class — no parent-refusal bleed "
        f"(rag={rag}, expected GREEN without REFUSAL→CHANGED)"
    )
    if not ok:
        failures.append("GE-F1 fresh twin no refusal bleed")
        print(rep)
    rag, rep = run_report(
        tmp,
        "case3b-parent",
        [
            make_result(
                "twinop",
                "fix__shared",
                {
                    "sem_off": "REFUSAL_BYTE_IDENTICAL",
                    "sem_on": "REFUSAL_BYTE_IDENTICAL",
                },
            ),
        ],
    )
    check(
        "GE-F1 parent refusal on the shared TU id keeps its refusal class",
        "GREEN",
        rag,
        rep,
        "baseline class refusal",
    )

    # 4. Win preserved within drift: GREEN.
    rag, rep = run_report(
        tmp,
        "case4",
        [
            make_result(
                "winop",
                "fix__winop",
                {"sem_off": 100.0, "sem_on": 80.5},
                causal_pct=-19.5,
            ),
        ],
    )
    check("win→win is GREEN", "GREEN", rag, rep)

    # 5. INVALID_MARKER cell gates RED.
    rag, rep = run_report(
        tmp,
        "case5",
        [
            make_result(
                "winop", "fix__winop", {"sem_off": 100.0, "sem_on": "INVALID_MARKER"}
            ),
        ],
    )
    check("INVALID_MARKER is RED", "RED", rag, rep, "INVALID_MARKER")

    # 5b. GE-F2 (lane GF): an executed perf leg with EMPTY samples stamps a
    #     'GE-F2 FATAL' assembly note — the row verdict must carry the RED
    #     (acceptance 1e), so streamed ROW-VERDICT.json and the report agree
    #     and an empty-samples row can never read 'ok'.
    rag, rep = run_report(
        tmp,
        "case5b",
        [
            make_result(
                "winop",
                "fix__winop",
                {"sem_off": 100.0, "sem_on": 80.5},
                causal_pct=-19.5,
                notes=[
                    "GE-F2 FATAL: winop/hand-perf leg 'on': EMPTY diag+kernel "
                    "perf samples on an executed device leg"
                ],
            ),
        ],
    )
    check(
        "GE-F2 FATAL empty-samples note gates the row RED",
        "RED",
        rag,
        rep,
        "EMPTY SAMPLES ON EXECUTED PERF LEG",
    )

    # ---- sweep-hardening round 2 regression cases ----

    # 6. Verifier fixture A: baseline row with measured cycles, this run's
    #    cells all None (device 'passed', metric unparsable) — must be RED,
    #    never 'no silicon cells this run' GREEN.
    rag, rep = run_report(
        tmp,
        "case6",
        [make_result("winop", "fix__winop", {"sem_off": None, "sem_on": None})],
    )
    check(
        "all-None cells with baseline history are RED",
        "RED",
        rag,
        rep,
        "INVALID_METRIC",
    )

    # 7. Verifier fixture B: uniform +50% slowdown, ratios preserved
    #    (causal -20% == baseline -20%, vs_hand 0%) — must be RED on the
    #    absolute cycle axis.
    rag, rep = run_report(
        tmp,
        "case7",
        [
            make_result(
                "winop",
                "fix__winop",
                {
                    "sem_off": 150.0,
                    "sem_on": 120.0,
                    "hand_off": 120.0,
                    "hand_on": 120.0,
                },
                causal_pct=-20.0,
                vs_hand_pct=0.0,
            ),
        ],
    )
    check("uniform slowdown (ratios preserved) is RED", "RED", rag, rep, "ABS CYCLES")

    # 8. Verifier fixture E: refusal-class row, sem still refuses, hand leg
    #    regresses 2x (50 -> 100) — must be RED, not blessed via the sem
    #    refusal branch.
    rag, rep = run_report(
        tmp,
        "case8",
        [
            make_result(
                "refop",
                "fix__refop",
                {
                    "sem_off": "REFUSAL_BYTE_IDENTICAL",
                    "sem_on": "REFUSAL_BYTE_IDENTICAL",
                    "hand_off": 100.0,
                    "hand_on": 100.0,
                },
            ),
        ],
    )
    check("hand-leg regression on a refusal row is RED", "RED", rag, rep, "ABS CYCLES")

    # 9. Verifier fixture C: baseline win -5.0% eroding to -0.3% (drift 4.7
    #    < 5, previously GREEN forever) — WIN→PARITY is RED by default...
    rag, rep = run_report(
        tmp,
        "case9",
        [
            make_result(
                "smallwin",
                "fix__smallwin",
                {"sem_off": 100.0, "sem_on": 99.7},
                causal_pct=-0.3,
            ),
        ],
    )
    check("win→parity erosion is RED by default", "RED", rag, rep, "WIN→PARITY")
    #    ...and YELLOW under the documented override.
    rag, rep = run_report(
        tmp,
        "case9b",
        [
            make_result(
                "smallwin",
                "fix__smallwin",
                {"sem_off": 100.0, "sem_on": 99.7},
                causal_pct=-0.3,
            ),
        ],
        allow_win_to_parity=True,
    )
    check(
        "win→parity is YELLOW under --allow-win-to-parity",
        "YELLOW",
        rag,
        rep,
        "WIN→PARITY",
    )

    # 10. Verifier fixture D (bounded to isolate the loss-growth axis from
    #     the abs-cycle axis): baseline loss +10% growing to +16% (growth
    #     6pp > 5pp; sem_on abs drift +5.45% < 10%) — RED, not YELLOW.
    rag, rep = run_report(
        tmp,
        "case10",
        [
            make_result(
                "lossop",
                "fix__lossop",
                {"sem_off": 100.0, "sem_on": 116.0},
                causal_pct=16.0,
            ),
        ],
    )
    check("loss growth beyond threshold is RED", "RED", rag, rep, "LOSS GREW")

    # 11. A YELLOW row must display 'YELLOW' in the verdict column, not 'ok'
    #     (drift-only fixture: causal -20% -> -13%, drift 7 > 5, all cells
    #     within abs threshold).
    rag, rep = run_report(
        tmp,
        "case11",
        [
            make_result(
                "winop",
                "fix__winop",
                {"sem_off": 100.0, "sem_on": 87.0, "hand_off": 80.0, "hand_on": 80.0},
                causal_pct=-13.0,
                vs_hand_pct=8.75,
            ),
        ],
    )
    check(
        "YELLOW row shows YELLOW verdict column",
        "YELLOW",
        rag,
        rep,
        "| winop | YELLOW |",
    )

    # 12. _device_job cache integrity (verifier recipe: fabricated GREEN
    #     cache).  Cached-reuse returns BEFORE the work dir is rebuilt, so
    #     rc.txt surviving == reuse; rc.txt gone == re-measured (dry-run
    #     stops before any device access).
    def fab_cache(sw, row, sel, label_leg, texts, jobkey):
        work = sw.ev / row["op"] / "silicon" / sel / label_leg
        work.mkdir(parents=True, exist_ok=True)
        (work / "rc.txt").write_text("0\n")
        (work / "log.txt").write_text("1 passed in 1.00s\n")
        (work / "TEXT_HASHES.txt").write_text(
            "".join(f"k.elf\ttext:{t}\telf:{t}\n" for t in texts)
        )
        (work / "jobkey.json").write_text(json.dumps(jobkey) + "\n")
        return work

    def cache_sweep(name):
        sw = object.__new__(Sweep)
        sw.a = argparse.Namespace(force=False, dry_run=True)
        sw.ev = tmp / name
        sw.ev.mkdir()
        sw.python = pathlib.Path(sys.executable)
        sw.reds = []
        return sw

    row = {
        "op": "cacheop",
        "nodes": {"sem-perf": "t.py::test_x[a-1]"},
        "extra_env": {"K": "V"},
    }
    flags = "-mflag"
    goodkey = {
        "node": row["nodes"]["sem-perf"],
        "flags": flags,
        "extra_env": {"K": "V"},
        "tag": "silicon",
        # execution-context key (laneBU batched silicon): serial and batched
        # cells never mix inside one row's samples
        "mode": "serial",
    }

    def cache_case(name, texts, jobkey, expected_texts, want_reuse, row_over=None):
        sw = cache_sweep(name)
        r = dict(row, **(row_over or {}))
        work = fab_cache(sw, r, "sem-perf", "r1-off", texts, jobkey)
        sw._device_job(r, "sem-perf", "r1", "off", flags, expected_texts=expected_texts)
        reused = (work / "rc.txt").is_file()
        ok = reused == want_reuse
        print(
            f"SELFTEST {'PASS' if ok else 'FAIL'}: {name} "
            f"(reused={reused}, expected reuse={want_reuse})"
        )
        if not ok:
            failures.append(name)

    cache_case(
        "cache: expected_texts=None never reuses",
        ["aaa"],
        goodkey,
        None,
        want_reuse=False,
    )
    cache_case(
        "cache: keyed hash-matched cache reuses",
        ["aaa"],
        goodkey,
        ["aaa"],
        want_reuse=True,
    )
    cache_case(
        "cache: changed node id never reuses",
        ["aaa"],
        goodkey,
        ["aaa"],
        want_reuse=False,
        row_over={"nodes": {"sem-perf": "t.py::test_x[b-2]"}},
    )
    cache_case(
        "cache: changed extra_env never reuses",
        ["aaa"],
        goodkey,
        ["aaa"],
        want_reuse=False,
        row_over={"extra_env": {"K": "OTHER"}},
    )
    cache_case(
        "cache: execution-mode mismatch never reuses (batched cell, serial run)",
        ["aaa"],
        dict(goodkey, mode="batched"),
        ["aaa"],
        want_reuse=False,
    )
    cache_case(
        "cache: .text mismatch never reuses",
        ["aaa"],
        goodkey,
        ["bbb"],
        want_reuse=False,
    )

    # 13. BH CRAQ silicon gate is KEYED: a legs-all-PASS verdict from another
    #     cc1plus/simulator/tree must not open it (verifier recipe: seeded
    #     stale verdict opened the gate for 14 device jobs).
    def gate_sweep(name):
        sw = object.__new__(Sweep)
        sim_dir = tmp / name / "sim"
        sim_dir.mkdir(parents=True)
        (sim_dir / "libttsim.so").write_bytes(b"fake-simulator")
        (sim_dir / "soc_descriptor.yaml").write_text("fake: yes\n")
        sw.a = argparse.Namespace(sim_bh=sim_dir / "libttsim.so", force=False)
        sw.ev = tmp / name / "ev"
        sw.ev.mkdir(parents=True)
        sw.info = {
            "cc1plus_sha256": "cc1-this-run",
            "tt_metal_head": "head-this-run",
        }
        sw.reds = []
        sim_sha = hashlib.sha256(b"fake-simulator").hexdigest()
        return sw, sim_sha

    def gate_case(name, verdict, want_open):
        sw, sim_sha = gate_sweep(name)
        verdict = dict(verdict, sim_sha_actual=sim_sha)
        vd = sw.ev / "gateop" / "craq" / "sem-corr-bh"
        vd.mkdir(parents=True)
        if verdict.get("sim_sha256") == "USE_ACTUAL":
            verdict["sim_sha256"] = sim_sha
        vd.joinpath("verdict.json").write_text(json.dumps(verdict))
        got = sw._bh_craq_gate({"op": "gateop"})
        ok = got == want_open
        print(
            f"SELFTEST {'PASS' if ok else 'FAIL'}: {name} "
            f"(gate={'open' if got else 'closed'}, expected "
            f"{'open' if want_open else 'closed'})"
        )
        if not ok:
            failures.append(name)

    legs_pass = {"legs": {"off": "PASS", "on": "PASS"}, "status": "OK"}
    gate_case(
        "gate: stale-key verdict never opens the gate",
        dict(
            legs_pass,
            cc1plus_sha256="cc1-OLD",
            sim_sha256="USE_ACTUAL",
            tt_metal_head="head-OLD",
        ),
        want_open=False,
    )
    gate_case(
        "gate: unkeyed legacy verdict never opens the gate",
        dict(legs_pass),
        want_open=False,
    )
    gate_case(
        "gate: this-run keyed verdict opens the gate",
        dict(
            legs_pass,
            cc1plus_sha256="cc1-this-run",
            sim_sha256="USE_ACTUAL",
            tt_metal_head="head-this-run",
        ),
        want_open=True,
    )
    gate_case(
        "gate: keyed verdict with a failing leg never opens the gate",
        dict(
            legs_pass,
            legs={"off": "PASS", "on": "FAIL(rc=1)"},
            cc1plus_sha256="cc1-this-run",
            sim_sha256="USE_ACTUAL",
            tt_metal_head="head-this-run",
        ),
        want_open=False,
    )

    # 14. ops-load validation: a perf selector without its correctness
    #     selector must fail loudly at load time.
    ops_fixture = tmp / "bad_ops.tsv"
    ops_fixture.write_text(
        "op\tcorpus_id\tkind\tmarker\tcraq_archs\tsem_corr\tsem_perf\t"
        "hand_corr\thand_perf\tnote\n"
        "badop\tfix__badop\tfull2x2\tTILE_LOOP\tbh\tt.py::corr\tt.py::perf\t\t"
        "t.py::handperf\thand perf without hand corr\n"
    )
    try:
        load_config(ops_fixture)
        print("SELFTEST FAIL: perf-without-corr row loaded without error")
        failures.append("ops-load validation")
    except SystemExit as e:
        ok = "hand-corr" in str(e) or "hand_corr" in str(e) or "corr" in str(e)
        print(
            f"SELFTEST {'PASS' if ok else 'FAIL'}: ops-load rejects perf "
            f"without corr (SystemExit: {e})"
        )
        if not ok:
            failures.append("ops-load validation")

    if failures:
        print(f"report self-test: FAILED ({', '.join(failures)})")
        return 1
    print(
        "report self-test: ALL GREEN "
        "(win→refusal=RED, refusal→refusal=GREEN, refusal→changed=YELLOW, "
        "win→win=GREEN, INVALID_MARKER=RED, all-None=INVALID_METRIC RED, "
        "uniform-slowdown=RED, refusal-hand-regression=RED, "
        "win→parity=RED(default)/YELLOW(override), loss-growth=RED, "
        "YELLOW column, device-job cache keying, keyed CRAQ gate, "
        "ops-load perf-requires-corr)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
