#!/usr/bin/env python3
"""Device-free focused tests for exhaustive_campaign_plan.py."""

import json
import tempfile
from pathlib import Path

import exhaustive_campaign_plan as P


def record(op, node, kind="full2x2"):
    return {"op": op, "kind": kind, "sem_corr": node, "hand_corr": node + "-hand"}


def search(op):
    return {
        "proposal": {
            "frozen_baseline_flags": "-mbaseline",
            "selection": {"flags": f"-mselected-{op}"},
        }
    }


def main():
    corpus = {
        "signbit": record("signbit", "test_sfpu_unary.py::u[formats:Float32->Float32]"),
        "abs": record("abs", "test_sfpu_unary.py::test_causal_lift_fresh_cpp[Abs]"),
        "binarypow-fresh": record("binarypow-fresh", "test_sfpu_binary.py::b[formats:Float16_b->Float16_b]"),
        "addint": record("addint", "test_sfpu_binary.py::b[formats:Int32->Int32]"),
        "where": record("where", "test_sfpu_ternary.py::t[formats:Float16_b->Float16_b]"),
        "welford": record("welford", "test_sfpu_welford_prefix_snapshot.py::w"),
        "ignored": record("ignored", "test_sfpu_unary.py::u", kind="semantic"),
    }
    chosen = {op: search(op) for op in corpus}
    old = P.UNARY_32
    try:
        P.UNARY_32 = frozenset({"signbit"})
        rows = P.plan(
            corpus,
            chosen,
            {"signbit": {"covered": P.TWO32, "full_space": P.TWO32}},
            "-mbaseline",
        )
    finally:
        P.UNARY_32 = old
    by_op = {row["op"]: row for row in rows}
    assert len(rows) == 6
    assert by_op["signbit"]["state"] == "COMPLETE"
    assert by_op["abs"]["full_space"] == P.TWO16
    assert by_op["binarypow-fresh"]["full_space"] == P.TWO32
    assert by_op["addint"]["category"] == "binary_nonexhaustive"
    assert by_op["where"]["category"] == "ternary_bf16_nonexhaustive"
    assert by_op["welford"]["category"] == "structural_unhooked"

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        P.write_outputs(out, rows)
        assert (out / "roster-unary-bf16.txt").read_text() == "abs\n"
        assert (out / "roster-unary-32.txt").read_text() == ""
        assert (out / "roster-binary-bf16-pair.txt").read_text() == "binarypow-fresh\n"
        assert not (out / "ops-unary-bf16.tsv").exists()
        assert (out / "tri-ops-unary-bf16.tsv").read_text().startswith(
            "op\ta_selected_sem_node\tb_baseline_sem_node\tc_baseline_hand_node\n"
        )
        assert (out / "roster-class-stratified.txt").read_text() == "addint\nwhere\n"
        assert "-mselected-abs" in (out / "selected-flags.tsv").read_text()
        assert "-mbaseline" in (out / "baseline-flags.tsv").read_text()
        tri = (out / "tri-profiles.tsv").read_text()
        assert "\r" not in tri
        assert "a_selected_sem_node" in tri
        assert "test_sfpu_unary.py::test_causal_lift_fresh_cpp[Abs]" in tri
        assert json.loads((out / "plan.json").read_text())["states"] == {
            "COMPLETE": 1, "GAP": 2, "NONEXHAUSTIVE": 3
        }

    bad = {op: search(op) for op in corpus}
    bad["abs"]["proposal"]["frozen_baseline_flags"] = "-mother-baseline"
    try:
        P.plan(corpus, bad, {}, "-mbaseline")
    except ValueError as error:
        assert "abs: proposal.frozen_baseline_flags disagrees" in str(error)
    else:
        raise AssertionError("per-op/global baseline disagreement was accepted")
    print("exhaustive_campaign_plan selftest: PASS")


if __name__ == "__main__":
    main()
