#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# selftest for prove_all.py — guards ROUTING + CENSUS MATH and exercises the
# engines end-to-end on 3 known ops:
#     zeropad-fresh   (formal_equiv, known EQUAL     -> SMT-PROVEN-ALL-INPUTS)
#     smoothstep-fresh(formal_equiv, known DIVERGENT -> DIVERGENCE-CERTIFIED)
#     binary-bcast    (classify,    known SCOPE      -> NOT-EXHAUSTIBLE)
#
# Part A is a fast pure-unit check of the precedence join + census (no sim).
# Part B runs the real driver over the 3 ops and checks the emitted ledger.
#
# Exit 0 = PASS.

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
import prove_all as PA  # noqa: E402


def part_a_unit():
    """Precedence + census must obey the strict order and machine-certified rule."""
    # silicon supersedes a divergent engine verdict
    r = PA.join_op(
        "x",
        "WIN",
        {"engine": "bitexact", "class": "DIVERGENCE-CERTIFIED", "verdict": "DIVERGENT"},
        {
            "x": {
                "silicon_class": "SILICON-EXHAUSTIVE",
                "silicon_verdict": "BIT-EXACT-ALL-INPUTS",
            }
        },
        {},
    )
    assert r["provability_class"] == "SILICON-EXHAUSTIVE", r
    assert r["machine_certified_equal"] == "YES", r

    # domain overlay upgrades a base DIVERGENT to SMT-PROVEN-DOMAIN
    r = PA.join_op(
        "y",
        "WIN",
        {
            "engine": "formal_equiv",
            "class": "DIVERGENCE-CERTIFIED",
            "verdict": "DIVERGENT",
        },
        {},
        {
            "y": {
                "domain_class": "SMT-PROVEN-DOMAIN",
                "jo_verdict": "PROVEN-EQUIV-ON-DOCUMENTED-DOMAIN",
            }
        },
    )
    assert r["provability_class"] == "SMT-PROVEN-DOMAIN", r
    assert r["machine_certified_equal"] == "no", r  # domain != full machine-certified

    # a pure SMT-all-inputs stays certified-equal
    r = PA.join_op(
        "z",
        "WIN",
        {
            "engine": "formal_equiv",
            "class": "SMT-PROVEN-ALL-INPUTS",
            "verdict": "PROVEN-EQUIV-ALL-INPUTS",
        },
        {},
        {},
    )
    assert (
        r["provability_class"] == "SMT-PROVEN-ALL-INPUTS"
        and r["machine_certified_equal"] == "YES"
    ), r

    # classify not-exhaustible is unreached
    r = PA.join_op(
        "w",
        "PARITY",
        {
            "engine": "classify",
            "class": "NOT-EXHAUSTIBLE",
            "verdict": "NOT-EXHAUSTIBLE",
        },
        {},
        {},
    )
    assert (
        r["provability_class"] == "NOT-EXHAUSTIBLE"
        and r["machine_certified_equal"] == "no"
    ), r

    # census counts what the join produced
    rows = [
        {
            "op": "a",
            "provability_class": "SILICON-EXHAUSTIVE",
            "machine_certified_equal": "YES",
        },
        {
            "op": "b",
            "provability_class": "SMT-PROVEN-ALL-INPUTS",
            "machine_certified_equal": "YES",
        },
        {
            "op": "c",
            "provability_class": "DIVERGENCE-CERTIFIED",
            "machine_certified_equal": "no",
        },
    ]
    cen = PA.census(rows)
    assert cen["SILICON-EXHAUSTIVE"] == 1 and cen["SMT-PROVEN-ALL-INPUTS"] == 1
    mc = sum(1 for r in rows if r["machine_certified_equal"] == "YES")
    assert mc == 2
    print("PART A (unit precedence + census): PASS")


def part_a_manifest():
    """Routing sanity: the 3 selftest ops route to the expected engines."""
    board = PA.load_board()
    man = PA.load_manifest(board)
    exp = {
        "zeropad-fresh": "formal_equiv",
        "smoothstep-fresh": "formal_equiv",
        "binary-bcast": "classify",
    }
    for op, eng in exp.items():
        assert man[op]["engine"] == eng, (op, man[op]["engine"], eng)
    print("PART A (manifest routing of 3 selftest ops): PASS")


def part_b_live():
    """Run the driver end-to-end on the 3 ops; check the ledger classes."""
    tmp = Path(tempfile.mkdtemp(prefix="prove_all_selftest_"))
    ops = "zeropad-fresh,smoothstep-fresh,binary-bcast"
    print(f"PART B: running prove_all --only {ops} (out={tmp}) ...")
    r = subprocess.run(
        [
            sys.executable,
            str(HERE / "prove_all.py"),
            "--only",
            ops,
            "--out",
            str(tmp),
            "--timeout",
            "600",
        ],
        capture_output=True,
        text=True,
    )
    sys.stdout.write(r.stdout[-1500:])
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-1500:])
        raise SystemExit("driver exited non-zero")
    ledger = tmp / "MASTER-COVERAGE-LEDGER.tsv"
    got = {}
    for row in PA.read_tsv(ledger):
        got[row["op"]] = row["provability_class"]
    expect = {
        "zeropad-fresh": "SMT-PROVEN-ALL-INPUTS",
        "smoothstep-fresh": "DIVERGENCE-CERTIFIED",
        "binary-bcast": "NOT-EXHAUSTIBLE",
    }
    ok = True
    for op, cls in expect.items():
        actual = got.get(op)
        flag = "OK" if actual == cls else "MISMATCH"
        if actual != cls:
            ok = False
        print(f"  {op:20s} expect={cls:24s} got={actual} [{flag}]")
    # census of the 3-op run must be internally consistent
    rows = PA.read_tsv(ledger)
    assert len(rows) == 3, rows
    if not ok:
        raise SystemExit("PART B: ledger class mismatch")
    print("PART B (live 3-op end-to-end): PASS")


if __name__ == "__main__":
    part_a_unit()
    part_a_manifest()
    part_b_live()
    print("\nSELFTEST: ALL PASS")
