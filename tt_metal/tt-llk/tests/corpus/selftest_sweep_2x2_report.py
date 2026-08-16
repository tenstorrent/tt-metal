#!/usr/bin/env python3
"""Self-test for sweep_2x2.py report() class-aware flip detection (defect D4,
PULL_ANALYSIS-20260817 §2c).

Drives the REAL report() with synthetic fixtures and asserts:
  1. WIN -> byte-identical REFUSAL  = RED   (the D4 hole: was GREEN);
  2. refusal -> refusal             = GREEN;
  3. refusal -> changed/measured    = flagged notice (YELLOW), never silent;
  4. win -> win (within drift)      = GREEN;
  5. INVALID_MARKER cell            = RED.

Run standalone or from the nightly/weekly wrappers; exits nonzero on any
failure so a broken gate can never bless a sweep.
"""
import argparse
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sweep_2x2 import Sweep  # noqa: E402

BASELINE = """\
# schema=2; synthetic self-test baseline; expected_class column drives the class-aware gate
id\tarch\tchip_class\tmetric\tscope\tselector\tcycles\tstatus\texpected_class\tcompiler_sha\tprovenance
fix__winop\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\twinop:sem_off\t100.0\tmeasured\twin\tselftest\tfixture
fix__winop\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\twinop:sem_on\t80.0\tmeasured\twin\tselftest\tfixture
fix__refop\tbh\tp150\tdevice_cycles\tTILE_LOOP_MATH_ISOLATE_PER_TILE\trefop:sem_off\t\trefusal\trefusal\tselftest\tfixture
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


def run_report(tmp, name, results):
    ev = tmp / name
    ev.mkdir()
    sw = object.__new__(Sweep)
    sw.a = argparse.Namespace(
        baseline=tmp / "baseline.tsv", prev_run=None, max_drift_pct=5.0
    )
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

    if failures:
        print(f"report self-test: FAILED ({', '.join(failures)})")
        return 1
    print(
        "report self-test: ALL GREEN "
        "(win→refusal=RED, refusal→refusal=GREEN, refusal→changed=YELLOW, "
        "win→win=GREEN, INVALID_MARKER=RED)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
