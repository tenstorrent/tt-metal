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
    assert P.UNARY_32.isdisjoint(P.COMPOSITE_UNARY_PAIRS)
    assert P.UNARY_32.isdisjoint(P.EXACT_BF16_UNARY_PAIRS)
    corpus = {
        "signbit": record("signbit", "test_sfpu_unary.py::u[formats:Float32->Float32]"),
        "abs": record("abs", "test_sfpu_unary.py::test_causal_lift_fresh_cpp[Abs]"),
        "expm1cw-fresh": record(
            "expm1cw-fresh",
            P.EXACT_BF16_UNARY_PAIRS["expm1cw-fresh"][0],
        ),
        "i1-fresh": record("i1-fresh", P.EXACT_BF16_UNARY_PAIRS["i1-fresh"][0]),
        "geluappx-fresh": record(
            "geluappx-fresh",
            P.COMPOSITE_UNARY_PAIRS["geluappx-fresh"][0],
        ),
        "hardshrink-fresh": record(
            "hardshrink-fresh",
            P.COMPOSITE_UNARY_PAIRS["hardshrink-fresh"][0],
        ),
        "relu": record("relu", P.COMPOSITE_UNARY_PAIRS["relu"][0]),
        "binarypow-fresh": record(
            "binarypow-fresh", "test_sfpu_binary.py::b[formats:Float16_b->Float16_b]"
        ),
        "addint": record("addint", "test_sfpu_binary.py::b[formats:Int32->Int32]"),
        "where": record(
            "where", "test_sfpu_ternary.py::t[formats:Float16_b->Float16_b]"
        ),
        "welford": record("welford", "test_sfpu_welford_prefix_snapshot.py::w"),
        "ignored": record("ignored", "test_sfpu_unary.py::u", kind="semantic"),
    }
    for op, (_, hand) in {
        **P.EXACT_BF16_UNARY_PAIRS,
        **P.COMPOSITE_UNARY_PAIRS,
    }.items():
        corpus[op]["hand_corr"] = hand
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
        stale_count_rows = P.plan(
            corpus,
            chosen,
            {
                "expm1cw-fresh": {
                    "covered": P.TWO32,
                    "full_space": P.TWO32,
                }
            },
            "-mbaseline",
        )
    finally:
        P.UNARY_32 = old
    by_op = {row["op"]: row for row in rows}
    assert len(rows) == 11
    assert by_op["signbit"]["state"] == "COMPLETE"
    assert by_op["abs"]["full_space"] == P.TWO16
    assert by_op["expm1cw-fresh"]["full_space"] == P.TWO16
    assert by_op["i1-fresh"]["category"] == "unary_bf16_exhaustive"
    assert by_op["geluappx-fresh"]["category"] == "structural_unhooked"
    assert by_op["geluappx-fresh"]["full_space"] is None
    assert by_op["hardshrink-fresh"]["category"] == "structural_unhooked"
    assert by_op["hardshrink-fresh"]["full_space"] is None
    assert by_op["relu"]["category"] == "structural_unhooked"
    assert by_op["binarypow-fresh"]["full_space"] == P.TWO32
    assert by_op["addint"]["category"] == "binary_nonexhaustive"
    assert by_op["where"]["category"] == "ternary_bf16_nonexhaustive"
    assert by_op["welford"]["category"] == "structural_unhooked"
    assert {row["op"]: row for row in stale_count_rows}["expm1cw-fresh"][
        "state"
    ] == "GAP"

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        P.write_outputs(out, rows)
        assert (out / "roster-unary-bf16.txt").read_text() == (
            "abs\nexpm1cw-fresh\ni1-fresh\n"
        )
        assert (out / "roster-unary-32.txt").read_text() == ""
        assert (out / "roster-binary-bf16-pair.txt").read_text() == "binarypow-fresh\n"
        assert not (out / "ops-unary-bf16.tsv").exists()
        assert (
            (out / "tri-ops-unary-bf16.tsv")
            .read_text()
            .startswith(
                "op\ta_selected_sem_node\tb_baseline_sem_node\tc_baseline_hand_node\n"
            )
        )
        assert (out / "roster-class-stratified.txt").read_text() == "addint\nwhere\n"
        assert "-mselected-abs" in (out / "selected-flags.tsv").read_text()
        assert "-mbaseline" in (out / "baseline-flags.tsv").read_text()
        tri = (out / "tri-profiles.tsv").read_text()
        assert "\r" not in tri
        assert "a_selected_sem_node" in tri
        assert "test_sfpu_unary.py::test_causal_lift_fresh_cpp[Abs]" in tri
        assert json.loads((out / "plan.json").read_text())["states"] == {
            "COMPLETE": 1,
            "GAP": 4,
            "NONEXHAUSTIVE": 6,
        }

    stale = {key: value.copy() for key, value in corpus.items()}
    stale["expm1cw-fresh"]["sem_corr"] += "-changed"
    try:
        P.classify("expm1cw-fresh", stale["expm1cw-fresh"])
    except ValueError as error:
        assert "exhaustive format binding is stale" in str(error)
    else:
        raise AssertionError("a rebound exact-format selector was accepted")

    stale_hand = {key: value.copy() for key, value in corpus.items()}
    stale_hand["i1-fresh"]["hand_corr"] += "-changed"
    try:
        P.classify("i1-fresh", stale_hand["i1-fresh"])
    except ValueError as error:
        assert "exhaustive format binding is stale" in str(error)
    else:
        raise AssertionError("a rebound exact-format hand selector was accepted")

    stale_composite = {key: value.copy() for key, value in corpus.items()}
    stale_composite["hardshrink-fresh"][
        "sem_corr"
    ] = "test_sfpu_unary.py::test_causal_lift_fresh_cpp[Hardshrink-fresh_cpp]"
    try:
        P.classify("hardshrink-fresh", stale_composite["hardshrink-fresh"])
    except ValueError as error:
        assert "exhaustive format binding is stale" in str(error)
    else:
        raise AssertionError("a rebound composite selector was accepted")

    stale_gelu = {key: value.copy() for key, value in corpus.items()}
    stale_gelu["geluappx-fresh"]["hand_corr"] += "-changed"
    try:
        P.classify("geluappx-fresh", stale_gelu["geluappx-fresh"])
    except ValueError as error:
        assert "exhaustive format binding is stale" in str(error)
    else:
        raise AssertionError("a rebound GELU composite selector was accepted")

    # Bind the focused checks to the checked-in corpus, not only toy records.
    checked_in = P.read_corpus(
        Path(__file__).resolve().parents[1] / "sweep_2x2_ops.tsv"
    )
    for op in P.EXACT_BF16_UNARY_PAIRS:
        category, fmt, _, full_space, _ = P.classify(op, checked_in[op])
        assert (category, fmt, full_space) == (
            "unary_bf16_exhaustive",
            "Float16_b",
            P.TWO16,
        )
    for op in P.COMPOSITE_UNARY_PAIRS:
        category, _, _, full_space, _ = P.classify(op, checked_in[op])
        assert category == "structural_unhooked"
        assert full_space is None

    # Exercise the real classification sets together, not just individual
    # selectors or the reduced fixture's replacement UNARY_32 set.
    full_plan = P.plan(
        checked_in,
        {op: search(op) for op in checked_in},
        {},
        "-mbaseline",
    )
    assert {row["op"] for row in full_plan} == {
        op for op, row in checked_in.items() if row["kind"] == "full2x2"
    }
    assert not any(row["state"] == "COMPLETE" for row in full_plan)

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
