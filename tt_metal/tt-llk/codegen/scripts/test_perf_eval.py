# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for perf_eval.py — intent-aware perf regression judgement."""

import importlib.util
import subprocess
import sys
from pathlib import Path

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


def _eval(current_csv, baseline_csv, *, op, goal):
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
    assert r["exit_code"] == 2


def test_op_filter_excludes_other_ops(tmp_path):
    # Baseline has only Sqrt; current has Reciprocal -> no comparable variants.
    base = _csv(tmp_path / "b.csv", "MathOperation.Sqrt", 600.0)
    cur = _csv(tmp_path / "c.csv", "MathOperation.Reciprocal", 600.0)
    r = _eval(cur, base, op="Reciprocal", goal="no_regress")
    assert r["verdict"] == "no_baseline"
    assert r["exit_code"] == 2


def test_no_current_rows_not_measured(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    r = _eval(empty, None, op=None, goal="no_regress")
    assert r["verdict"] == "not_measured"
    assert r["exit_code"] == 2


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
