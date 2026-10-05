#!/usr/bin/env python3
"""Focused device-free tests for whole-campaign numeric admission."""

from __future__ import annotations

import tempfile
from pathlib import Path
import subprocess
import sys

import galaxy_numeric_admission as admission


def corr(op: str, leg: str, count: int, out: int, classes: str, *, graded=None) -> str:
    fields = [
        "SFPU_CORRECTNESS",
        f"op={op}",
        f"leg={leg}",
        f"patterns={count}",
        f"n_out_of_tol={out}",
        "max_bf16_ulp=9",
        f"within_contract={'True' if out == 0 else 'False'}",
        f"class_ulp={classes}",
    ]
    if graded is not None:
        fields += [f"n_graded={graded[0]}", f"n_out_graded={graded[1]}"]
    return ",".join(fields) + "\n"


def make_root(tmp: Path, op="op", verdict="DIVERGENT", covered=20) -> Path:
    root = tmp / op
    root.mkdir()
    (root / f"{op}-VERDICT.txt").write_text(
        f"OP={op} VERDICT={verdict} covered={covered} full_space={covered}\n"
    )
    return root


def band(root: Path, chip: int, name: str, sem: str, hand: str) -> None:
    out = root / f"slice-{chip}" / "bands"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}-sem.txt.corr").write_text(sem)
    (out / f"{name}-hand.txt.corr").write_text(hand)


def test_global_max_not_shard_and(tmp: Path) -> None:
    root = make_root(tmp)
    # Slice 0 would fail a local 8 <= 5 comparison.  Globally the hand maximum
    # is 9 from slice 1, so candidate max 8 <= hand max 9 and must pass.
    band(root, 0, "b0", corr("op", "sem", 10, 0, "in_domain_finite_normal:10:8"),
         corr("op", "hand", 10, 0, "in_domain_finite_normal:10:5"))
    band(root, 1, "b1", corr("op", "sem", 10, 0, "in_domain_finite_normal:10:2"),
         corr("op", "hand", 10, 0, "in_domain_finite_normal:10:9"))
    got = admission.aggregate(root, "op")
    assert got["numeric_admission"] == "PASS", got
    assert got["equivalence"]["status"] == "DIVERGENT", got


def test_better_semantic_arm_admitted(tmp: Path) -> None:
    root = make_root(tmp, op="better", covered=10)
    band(root, 0, "b0", corr("better", "sem", 10, 0, "in_domain_finite_normal:10:3"),
         corr("better", "hand", 10, 2, "in_domain_finite_normal:10:9"))
    got = admission.aggregate(root, "better")
    assert got["semantic_absolute"] == "PASS", got
    assert got["hand_absolute"] == "FAIL", got
    assert got["ulp_nonregression"] == "PASS", got
    assert got["numeric_admission"] == "PASS", got


def test_semantic_failure_and_missing_oracle(tmp: Path) -> None:
    root = make_root(tmp, op="semfail", covered=10)
    band(root, 0, "b0", corr("semfail", "sem", 10, 1, "in_domain_finite_normal:10:9"),
         corr("semfail", "hand", 10, 0, "in_domain_finite_normal:10:3"))
    got = admission.aggregate(root, "semfail")
    assert got["numeric_admission"] == "FAIL", got
    assert got["reason"] == "semantic-arm-outside-absolute-contract", got

    missing = make_root(tmp, op="missing", covered=10)
    got = admission.aggregate(missing, "missing")
    assert got["numeric_admission"] == "NO_ORACLE", got
    run = subprocess.run(
        [sys.executable, str(Path(admission.__file__)), str(missing), "missing"],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0, "NO_ORACLE CLI result must fail closed"


def test_graded_contract_and_completeness(tmp: Path) -> None:
    root = make_root(tmp, op="graded", covered=10)
    # Global tolerance includes undefined inputs, but the graded/defined domain
    # is clean.  This is an absolute semantic pass, matching the streamer rule.
    band(root, 0, "b0", corr("graded", "sem", 10, 4, "in_domain_finite_normal:10:3", graded=(6, 0)),
         corr("graded", "hand", 10, 5, "in_domain_finite_normal:10:4", graded=(6, 1)))
    got = admission.aggregate(root, "graded")
    assert got["semantic_absolute"] == "PASS", got
    assert got["hand_absolute"] == "FAIL", got
    assert got["numeric_admission"] == "PASS", got

    # Full campaign verdict says 10 patterns but only 9 have oracle records.
    incomplete = make_root(tmp, op="short", covered=10)
    band(incomplete, 0, "b0", corr("short", "sem", 9, 0, "in_domain_finite_normal:9:2"),
         corr("short", "hand", 9, 0, "in_domain_finite_normal:9:2"))
    got = admission.aggregate(incomplete, "short")
    assert got["numeric_admission"] == "INCOMPLETE", got


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_global_max_not_shard_and(tmp)
        test_better_semantic_arm_admitted(tmp)
        test_semantic_failure_and_missing_oracle(tmp)
        test_graded_contract_and_completeness(tmp)
    print("PASS galaxy numeric admission (global maxima, uplift, refusal, coverage)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
