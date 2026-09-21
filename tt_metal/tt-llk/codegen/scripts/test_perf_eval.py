# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for perf_eval.py — intent-aware perf regression judgement."""

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "perf_eval", Path(__file__).parent / "perf_eval.py"
)
perf_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_eval)


HEADER = (
    "mathop,dest_acc,tile_cnt,marker,"
    "mean(L1_TO_L1),std(L1_TO_L1),mean(MATH_ISOLATE),TEXT_SIZE(L1_TO_L1)"
)


def _csv(path: Path, mathop: str, tile_loop_cycles: float) -> Path:
    # One TILE_LOOP row (the compared marker) + an INIT row that must be ignored.
    # MATH_ISOLATE tracks the headline so the per-thread breakdown is meaningful.
    rows = [
        HEADER,
        f"{mathop},DestAccumulation.No,8,INIT,457.0,0.0,176.0,12484",
        f"{mathop},DestAccumulation.No,8,TILE_LOOP,{tile_loop_cycles},0.0,{tile_loop_cycles},12484",
    ]
    path.write_text("\n".join(rows) + "\n")
    return path


def _eval(
    current_csv, baseline_csv, *, op, goal, primary_metric=perf_eval.PRIMARY_METRIC
):
    cur = perf_eval._read_csv(current_csv)
    base = perf_eval._read_csv(baseline_csv) if baseline_csv else []
    return perf_eval.evaluate(
        cur,
        base,
        op=op,
        goal=goal,
        noise_pct=3.0,
        regress_pct=3.0,
        improve_pct=2.0,
        primary_metric=primary_metric,
    )


def test_regression_under_no_regress_is_miss(tmp_path):
    base = _csv(tmp_path / "b.csv", "MathOperation.Reciprocal", 600.0)
    cur = _csv(tmp_path / "c.csv", "MathOperation.Reciprocal", 660.0)  # +10%
    r = _eval(cur, base, op="Reciprocal", goal="no_regress")
    assert r["verdict"] == "regressed"
    assert r["exit_code"] == 1
    assert r["delta_pct_worst"] > 3.0


def test_regression_includes_thread_breakdown(tmp_path):
    # The worst variant must carry a per-thread breakdown so the worker can
    # localize which Tensix thread the change slowed down.
    base = _csv(tmp_path / "b.csv", "MathOperation.Reciprocal", 600.0)
    cur = _csv(tmp_path / "c.csv", "MathOperation.Reciprocal", 660.0)  # +10%
    r = _eval(cur, base, op="Reciprocal", goal="no_regress")
    wv = r["worst_variant"]
    assert "thread_breakdown" in wv
    assert "mean(MATH_ISOLATE)" in wv["thread_breakdown"]
    assert abs(wv["thread_breakdown"]["mean(MATH_ISOLATE)"]["delta_pct"] - 10.0) < 0.01
    # Internal raw-row refs must not leak into the result.
    assert "_cur" not in wv and "_base" not in wv


def test_within_noise_is_neutral_pass(tmp_path):
    base = _csv(tmp_path / "b.csv", "MathOperation.Reciprocal", 600.0)
    cur = _csv(tmp_path / "c.csv", "MathOperation.Reciprocal", 606.0)  # +1%
    r = _eval(cur, base, op="Reciprocal", goal="no_regress")
    assert r["verdict"] == "neutral"
    assert r["exit_code"] == 0


def test_improvement_under_improve_passes(tmp_path):
    base = _csv(tmp_path / "b.csv", "MathOperation.Reciprocal", 600.0)
    cur = _csv(tmp_path / "c.csv", "MathOperation.Reciprocal", 540.0)  # -10%
    r = _eval(cur, base, op="Reciprocal", goal="improve")
    assert r["verdict"] == "improved"
    assert r["exit_code"] == 0


def test_not_improved_under_improve_is_miss(tmp_path):
    base = _csv(tmp_path / "b.csv", "MathOperation.Reciprocal", 600.0)
    cur = _csv(tmp_path / "c.csv", "MathOperation.Reciprocal", 601.0)  # flat
    r = _eval(cur, base, op="Reciprocal", goal="improve")
    assert r["verdict"] == "not_improved"
    assert r["exit_code"] == 1


def test_regression_under_improve_is_miss(tmp_path):
    base = _csv(tmp_path / "b.csv", "MathOperation.Reciprocal", 600.0)
    cur = _csv(tmp_path / "c.csv", "MathOperation.Reciprocal", 700.0)  # slower
    r = _eval(cur, base, op="Reciprocal", goal="improve")
    assert r["verdict"] == "regressed"
    assert r["exit_code"] == 1


def test_missing_baseline_is_not_comparable(tmp_path):
    cur = _csv(tmp_path / "c.csv", "MathOperation.Reciprocal", 600.0)
    r = _eval(cur, None, op="Reciprocal", goal="no_regress")
    assert r["verdict"] == "no_baseline"
    assert r["reason_code"] == "baseline_rows_missing"
    assert r["exit_code"] == 2


def test_op_filter_excludes_other_ops(tmp_path):
    # Baseline has only Sqrt; current has Reciprocal -> no comparable variants.
    base = _csv(tmp_path / "b.csv", "MathOperation.Sqrt", 600.0)
    cur = _csv(tmp_path / "c.csv", "MathOperation.Reciprocal", 600.0)
    r = _eval(cur, base, op="Reciprocal", goal="no_regress")
    assert r["verdict"] == "no_baseline"
    assert r["reason_code"] == "baseline_rows_missing"
    assert r["exit_code"] == 2


def test_no_current_rows_not_measured(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    r = _eval(empty, None, op=None, goal="no_regress")
    assert r["verdict"] == "not_measured"
    assert r["reason_code"] == "current_rows_missing"
    assert r["exit_code"] == 2


def test_empty_current_selection_is_a_plan_defect_not_missing_measurements(tmp_path):
    cur = _csv(tmp_path / "current.csv", "MathOperation.Sqrt", 600.0)
    base = _csv(tmp_path / "baseline.csv", "MathOperation.Reciprocal", 600.0)
    result = _eval(cur, base, op="Reciprocal", goal="no_regress")
    assert result["verdict"] == "not_measured"
    assert result["reason_code"] == "current_selection_empty"
    assert result["exit_code"] == 2
    assert result["measured"] is False
    assert result["coverage"]["current_rows"] == 0
    assert result["coverage"]["comparison_complete"] is False


# --- 0.5% noise floor (perf team) ------------------------------------------


def _eval_noise_floor(current_csv, baseline_csv, *, op, goal):
    cur = perf_eval._read_csv(current_csv)
    base = perf_eval._read_csv(baseline_csv) if baseline_csv else []
    return perf_eval.evaluate(
        cur, base, op=op, goal=goal, noise_pct=0.5, regress_pct=0.5, improve_pct=0.5
    )


def test_within_noise_floor_is_neutral(tmp_path):
    base = _csv(tmp_path / "b.csv", "MathOperation.Reciprocal", 600.0)
    cur = _csv(tmp_path / "c.csv", "MathOperation.Reciprocal", 602.4)  # +0.4% < 0.5
    r = _eval_noise_floor(cur, base, op="Reciprocal", goal="no_regress")
    assert r["verdict"] == "neutral"
    assert r["exit_code"] == 0


def test_above_noise_floor_is_regression(tmp_path):
    base = _csv(tmp_path / "b.csv", "MathOperation.Reciprocal", 600.0)
    cur = _csv(tmp_path / "c.csv", "MathOperation.Reciprocal", 604.2)  # +0.7% > 0.5
    r = _eval_noise_floor(cur, base, op="Reciprocal", goal="no_regress")
    assert r["verdict"] == "regressed"
    assert r["exit_code"] == 1


def test_shipped_cli_defaults_use_half_pct_noise_floor(tmp_path):
    # Verify the *defaults* (no threshold flags) honor the 0.5% noise floor:
    # +0.4% must pass (exit 0), +0.7% must flag a regression (exit 1).
    script = Path(__file__).parent / "perf_eval.py"
    base = _csv(tmp_path / "b.csv", "MathOperation.Reciprocal", 600.0)

    near = _csv(tmp_path / "near.csv", "MathOperation.Reciprocal", 602.4)  # +0.4%
    over = _csv(tmp_path / "over.csv", "MathOperation.Reciprocal", 604.2)  # +0.7%

    def run(current):
        return subprocess.run(
            [
                sys.executable,
                str(script),
                "--current",
                str(current),
                "--baseline",
                str(base),
                "--op",
                "Reciprocal",
                "--goal",
                "no_regress",
            ],
            capture_output=True,
            text=True,
        ).returncode

    assert run(near) == 0  # within noise -> not flagged
    assert run(over) == 1  # beyond noise -> regression


def test_isolate_only_csv_requires_explicit_selection(tmp_path):
    # Matches the archived feature-support run's schema: there is no L1 metric.
    header = "variant,marker,mean(MATH_ISOLATE),TEXT_SIZE(MATH_ISOLATE)\n"
    base = tmp_path / "baseline.csv"
    cur = tmp_path / "current.csv"
    base.write_text(header + "a,TILE_LOOP,100,2000\nb,TILE_LOOP,200,2000\n")
    cur.write_text(header + "a,TILE_LOOP,100,2100\nb,TILE_LOOP,200,2100\n")

    missing = _eval(cur, base, op=None, goal="no_regress")
    assert missing["verdict"] == "missing_metric"
    assert missing["exit_code"] == 2
    assert "mean(L1_TO_L1)" in missing["reason"]

    result = _eval(
        cur, base, op=None, goal="no_regress", primary_metric="mean(MATH_ISOLATE)"
    )
    assert result["verdict"] == "neutral"
    assert result["variants_compared"] == 2
    assert result["primary_metric"] == "mean(MATH_ISOLATE) @ TILE_LOOP"


def test_selected_metric_is_not_replaced_by_more_favorable_metric(tmp_path):
    base = _csv(tmp_path / "base.csv", "MathOperation.Reciprocal", 600.0)
    cur = _csv(tmp_path / "current.csv", "MathOperation.Reciprocal", 660.0)
    # L1 regresses by 10%, whereas MATH improves by 10%.
    cur.write_text(cur.read_text().replace("660.0,0.0,660.0", "660.0,0.0,540.0"))
    assert _eval(cur, base, op=None, goal="no_regress")["verdict"] == "regressed"
    isolated = _eval(
        cur, base, op=None, goal="improve", primary_metric="mean(MATH_ISOLATE)"
    )
    assert isolated["verdict"] == "improved"
    assert isolated["delta_pct_worst"] == -10.0


@pytest.mark.parametrize("source", ["current", "baseline"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "", "bad", "0", "-1"])
def test_invalid_selected_measurement_cannot_silently_pass(tmp_path, source, value):
    # A valid neutral variant must not hide a second malformed measurement.
    header = "variant,marker,mean(MATH_ISOLATE)\n"
    files = {name: tmp_path / f"{name}.csv" for name in ("current", "baseline")}
    for name, path in files.items():
        second = value if name == source else "100"
        path.write_text(header + f"good,TILE_LOOP,100\nbad,TILE_LOOP,{second}\n")
    result = _eval(
        files["current"],
        files["baseline"],
        op=None,
        goal="no_regress",
        primary_metric="mean(MATH_ISOLATE)",
    )
    assert result["exit_code"] == 2
    assert result["verdict"] == "invalid_measurement"
    assert source in result["reason"]
    json.dumps(result, allow_nan=False)


def test_missing_selected_baseline_metric_is_not_comparable(tmp_path):
    cur = _csv(tmp_path / "current.csv", "MathOperation.Reciprocal", 600.0)
    base = tmp_path / "baseline.csv"
    base.write_text(cur.read_text().replace("mean(MATH_ISOLATE)", "std(MATH_ISOLATE)"))
    result = _eval(
        cur, base, op=None, goal="no_regress", primary_metric="mean(MATH_ISOLATE)"
    )
    assert result["verdict"] == "missing_metric"
    assert "baseline" in result["reason"]


def test_unsupported_metric_rejected_by_api(tmp_path):
    cur = _csv(tmp_path / "current.csv", "MathOperation.Reciprocal", 600.0)
    with pytest.raises(ValueError, match="unsupported primary metric"):
        _eval(cur, cur, op=None, goal="no_regress", primary_metric="std(L1_TO_L1)")


@pytest.mark.parametrize("name", ["noise_pct", "regress_pct", "improve_pct"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_negative_threshold_rejected(name, value):
    thresholds = dict(noise_pct=0.5, regress_pct=0.5, improve_pct=0.5)
    thresholds[name] = value
    with pytest.raises(ValueError, match="finite and non-negative"):
        perf_eval.evaluate([], [], op=None, goal="no_regress", **thresholds)


def test_cli_records_explicit_metric_and_rejects_invalid_configuration(tmp_path):
    cur = tmp_path / "current.csv"
    cur.write_text("variant,marker,mean(MATH_ISOLATE)\na,TILE_LOOP,100\n")
    out = tmp_path / "result.json"
    command = [
        sys.executable,
        str(Path(__file__).parent / "perf_eval.py"),
        "--current",
        str(cur),
        "--baseline",
        str(cur),
        "--json-out",
        str(out),
    ]
    result = subprocess.run(
        command + ["--primary-metric", "mean(MATH_ISOLATE)"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "mean(MATH_ISOLATE) @ TILE_LOOP" in result.stdout
    assert (
        json.loads(out.read_text())["primary_metric"]
        == "mean(MATH_ISOLATE) @ TILE_LOOP"
    )

    for flags in (
        ["--primary-metric", "mean(UNKNOWN)"],
        ["--regress-pct", "nan"],
        ["--improve-pct", "inf"],
    ):
        invalid = subprocess.run(command + flags, capture_output=True, text=True)
        assert invalid.returncode == 2
        assert "error:" in invalid.stderr


def _variant_csv(path, variants, *, include_dest_acc=True):
    columns = ["dest_acc", "tile_cnt", "marker", "mean(L1_TO_L1)"]
    if not include_dest_acc:
        columns.remove("dest_acc")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for acc, cycles in variants:
            row = {
                "dest_acc": acc,
                "tile_cnt": "1",
                "marker": "TILE_LOOP",
                "mean(L1_TO_L1)": cycles,
            }
            writer.writerow({key: row[key] for key in columns})
    return path


def test_dest_acc_mismatch_is_not_a_comparable_variant(tmp_path):
    cur = _variant_csv(tmp_path / "cur.csv", [("Yes", 100)])
    base = _variant_csv(tmp_path / "base.csv", [("No", 100)])
    result = _eval(cur, base, op=None, goal="no_regress")
    assert result["exit_code"] == 2 and result["verdict"] == "no_baseline"
    assert result["reason_code"] == "no_matching_baseline_variants"
    assert result["coverage"]["matched_variants"] == 0
    assert result["coverage"]["current_only_variants"] == 1
    assert result["coverage"]["baseline_only_variants"] == 1


@pytest.mark.parametrize("goal", ["no_regress", "improve"])
def test_new_current_variant_prevents_whole_sweep_success_without_claiming_regression(
    tmp_path, goal
):
    cur = _variant_csv(tmp_path / "cur.csv", [("No", 100), ("Yes", 300)])
    base = _variant_csv(tmp_path / "base.csv", [("No", 100)])
    result = _eval(cur, base, op=None, goal=goal)
    assert result["exit_code"] == 2 and result["verdict"] == "no_baseline"
    assert result["reason_code"] == "incomplete_baseline_coverage"
    assert result["measured"] is True
    assert result["matched_verdict"] in ("neutral", "not_improved")
    assert result["variants_compared"] == 1
    assert result["coverage"] == {
        "current_rows": 2,
        "baseline_rows": 1,
        "current_variants": 2,
        "baseline_variants": 1,
        "matched_variants": 1,
        "current_only_variants": 1,
        "baseline_only_variants": 0,
        "duplicate_current_rows": 0,
        "duplicate_baseline_rows": 0,
        "comparison_complete": False,
    }
    assert "do not certify" in result["reason"]


@pytest.mark.parametrize("missing_in", ["current", "baseline"])
def test_asymmetric_configuration_columns_cannot_certify_same_variant(
    tmp_path, missing_in
):
    # Omitting current dest_acc used to match against baseline No, regardless
    # of the actual current configuration. The converse is equally unprovable.
    cur = _variant_csv(
        tmp_path / "cur.csv", [("Yes", 100)], include_dest_acc=missing_in != "current"
    )
    base = _variant_csv(
        tmp_path / "base.csv", [("No", 100)], include_dest_acc=missing_in != "baseline"
    )
    result = _eval(cur, base, op=None, goal="no_regress")
    assert result["exit_code"] == 2 and result["verdict"] == "no_baseline"
    assert result["reason_code"] == "variant_schema_mismatch"
    assert "configuration columns differ" in result["reason"]
    assert result["coverage"]["matched_variants"] is None
    assert result["coverage"]["comparison_complete"] is False


@pytest.mark.parametrize("source", ["current", "baseline"])
@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_variant_keys_are_rejected_independent_of_csv_order(
    tmp_path, source, reverse
):
    duplicates = [("Yes", 50), ("Yes", 100)]
    if reverse:
        duplicates.reverse()
    cur = _variant_csv(
        tmp_path / "cur.csv", duplicates if source == "current" else [("Yes", 100)]
    )
    base = _variant_csv(
        tmp_path / "base.csv", duplicates if source == "baseline" else [("Yes", 100)]
    )
    result = _eval(cur, base, op=None, goal="no_regress")
    assert result["exit_code"] == 2 and result["verdict"] == "no_baseline"
    assert result["reason_code"] == "duplicate_variant_keys"
    assert "duplicate variant keys" in result["reason"]
    assert result["coverage"][f"duplicate_{source}_rows"] == 1
    assert result["coverage"]["comparison_complete"] is False


def test_broader_baseline_is_allowed_and_its_extra_variants_are_disclosed(tmp_path):
    cur = _variant_csv(tmp_path / "cur.csv", [("Yes", 100)])
    base = _variant_csv(tmp_path / "base.csv", [("No", 200), ("Yes", 100)])
    result = _eval(cur, base, op=None, goal="no_regress")
    assert result["exit_code"] == 0 and result["verdict"] == "neutral"
    assert result["coverage"]["comparison_complete"] is True
    assert result["coverage"]["matched_variants"] == 1
    assert result["coverage"]["current_only_variants"] == 0
    assert result["coverage"]["baseline_only_variants"] == 1


def test_partial_overlap_retains_proven_regression_without_full_coverage_claim(
    tmp_path,
):
    cur = _variant_csv(tmp_path / "cur.csv", [("No", 120), ("Yes", 100)])
    base = _variant_csv(tmp_path / "base.csv", [("No", 100)])
    result = _eval(cur, base, op=None, goal="no_regress")
    assert result["exit_code"] == 1 and result["verdict"] == "regressed"
    assert result["reason_code"] == "incomplete_baseline_coverage"
    assert result["matched_verdict"] == "regressed"
    assert result["coverage"]["comparison_complete"] is False
    assert "no matching baseline" in result["reason"]


def test_cli_incomplete_coverage_is_exit_two_and_reports_counts(tmp_path):
    cur = _variant_csv(tmp_path / "cur.csv", [("No", 100), ("Yes", 300)])
    base = _variant_csv(tmp_path / "base.csv", [("No", 100)])
    out = tmp_path / "result.json"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "perf_eval.py"),
            "--current",
            str(cur),
            "--baseline",
            str(base),
            "--goal",
            "no_regress",
            "--json-out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "matched=1 current-only=1 baseline-only=0 complete=False" in result.stdout
    saved = json.loads(out.read_text())
    assert saved["verdict"] == "no_baseline"
    assert saved["reason_code"] == "incomplete_baseline_coverage"
