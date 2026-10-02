# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the canonical perf compare module (tt_metal/tt-llk/perf).

This is the module the PR gate runs, so the two-clause rule it enforces —
more than ``threshold`` slower AND more than ``min_cycles`` slower — is what
most of these tests cover.

Run: pytest test_perf_regression_compare.py
"""

import pathlib
import sys

import pandas as pd
import pytest

# tt-llk holds a hyphen, so `perf` is not an importable package path; add its
# directory the way both real callers reach it — by filesystem location.
_PERF = pathlib.Path(__file__).parents[2] / "perf"
sys.path.insert(0, str(_PERF))
from regression_compare import (  # noqa: E402
    DEFAULT_MIN_CYCLES,
    DEFAULT_THRESHOLD,
    _medians,
    compare_runs,
    render_report,
)

# Magnitudes are realistic cycle counts, not toy numbers: the verdict depends on
# an absolute cycle floor, so a test written at 100 cycles would prove nothing
# about a TILE_LOOP measured in thousands.
_INIT = 900.0
_TILE_LOOP = 2000.0


def _csv(
    tmp_path,
    name,
    init,
    tile_loop,
    run_type="MATH_ISOLATE",
    tile_cnt=4,
    module="perf_x",
):
    # The real layout: <run>/<module>/<module>.csv, one directory per run here.
    path = tmp_path / name.removesuffix(".csv") / f"{module}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "marker": ["INIT", "TILE_LOOP"],
            "tile_cnt": [tile_cnt, tile_cnt],  # a config column -> part of the key
            f"mean({run_type})": [init, tile_loop],
        }
    ).to_csv(path, index=False)
    return str(path)


def _sides(tmp_path, base, cur, **kw):
    baseline = [_csv(tmp_path, f"b{i}.csv", *base, **kw) for i in range(3)]
    current = [_csv(tmp_path, f"c{i}.csv", *cur, **kw) for i in range(3)]
    return current, baseline


def _markers(records):
    return {r["marker"] for r in records}


# --- the two-clause rule ----------------------------------------------------


def test_both_clauses_met_is_a_regression(tmp_path):
    """+10% and +200 cycles on TILE_LOOP. Both clauses hold."""
    current, baseline = _sides(tmp_path, (_INIT, _TILE_LOOP), (_INIT, _TILE_LOOP + 200))

    result = compare_runs(current, baseline)

    assert _markers(result["regressions"]) == {"TILE_LOOP"}
    assert abs(result["regressions"][0]["delta"] - 0.10) < 1e-9


def test_percentage_alone_is_not_a_regression(tmp_path):
    """INIT moves +2.2%, which clears the threshold, but only 20 cycles.

    This is the case the cycle floor exists for: a small marker with jitter.
    """
    current, baseline = _sides(tmp_path, (_INIT, _TILE_LOOP), (_INIT + 20, _TILE_LOOP))

    result = compare_runs(current, baseline)

    assert not result["regressions"]
    assert result["noise_filtered"] == 1


def test_cycles_alone_is_not_a_regression(tmp_path):
    """TILE_LOOP moves +38 cycles, over the floor, but only 1.9% of a big number."""
    current, baseline = _sides(
        tmp_path, (_INIT, _TILE_LOOP), (_INIT, _TILE_LOOP * 1.019)
    )

    assert (_TILE_LOOP * 0.019) > DEFAULT_MIN_CYCLES  # the cycle clause holds
    assert 0.019 < DEFAULT_THRESHOLD  # the percentage clause does not

    result = compare_runs(current, baseline)

    assert not result["regressions"]


def test_improvement_is_symmetric(tmp_path):
    current, baseline = _sides(tmp_path, (_INIT, _TILE_LOOP), (_INIT, _TILE_LOOP - 200))

    result = compare_runs(current, baseline)

    assert _markers(result["improvements"]) == {"TILE_LOOP"}
    assert not result["regressions"]


def test_min_cycles_zero_disables_the_absolute_clause(tmp_path):
    current, baseline = _sides(tmp_path, (_INIT, _TILE_LOOP), (_INIT + 20, _TILE_LOOP))

    result = compare_runs(current, baseline, min_cycles=0)

    assert _markers(result["regressions"]) == {"INIT"}


# --- points and coverage ----------------------------------------------------


def test_unchanged_run_still_compares_every_point(tmp_path):
    current, baseline = _sides(tmp_path, (_INIT, _TILE_LOOP), (_INIT, _TILE_LOOP))

    result = compare_runs(current, baseline)

    assert not result["regressions"]
    assert len(result["records"]) == 2


def test_new_config_is_reported_not_flagged(tmp_path):
    baseline = [_csv(tmp_path, "b.csv", _INIT, _TILE_LOOP, tile_cnt=4)]
    # tile_cnt=8 exists on the current side only, so it has nothing to compare to.
    current = [_csv(tmp_path, "c.csv", _INIT, _TILE_LOOP * 4, tile_cnt=8)]

    result = compare_runs(current, baseline)

    assert len(result["new_points"]) == 2
    assert not result["regressions"]
    assert not result["records"]  # nothing was actually compared


def test_run_type_filter_keeps_only_what_was_asked_for(tmp_path):
    baseline = [_csv(tmp_path, "b.csv", _INIT, _TILE_LOOP, run_type="L1_TO_L1")]
    current = [_csv(tmp_path, "c.csv", _INIT, _TILE_LOOP + 200, run_type="L1_TO_L1")]

    assert compare_runs(current, baseline, run_types="L1_TO_L1")["regressions"]
    # A filter that matches nothing must empty the comparison, not pass it.
    assert not compare_runs(current, baseline, run_types="MATH_ISOLATE")["records"]


# --- the median ------------------------------------------------------------


def test_median_ignores_one_wild_iteration(tmp_path):
    """Three iterations, one of them absurd. The median must not move."""
    baseline = [_csv(tmp_path, f"b{i}.csv", _INIT, _TILE_LOOP) for i in range(3)]
    current = [
        _csv(tmp_path, "c0.csv", _INIT, _TILE_LOOP),
        _csv(tmp_path, "c1.csv", _INIT, _TILE_LOOP),
        _csv(tmp_path, "c2.csv", _INIT, _TILE_LOOP * 10),
    ]

    result = compare_runs(current, baseline)

    assert not result["regressions"]


def test_medians_tolerate_frames_with_different_columns(tmp_path):
    """Iterations need not agree on their sweep columns.

    A column absent from one frame must stay out of that frame's point key,
    rather than merging its rows into another point.
    """
    wide = pd.DataFrame(
        {
            "marker": ["TILE_LOOP"],
            "tile_cnt": [4],
            "math_fidelity": ["HiFi4"],
            "mean(L1_TO_L1)": [_TILE_LOOP],
        }
    )
    narrow = pd.DataFrame(
        {"marker": ["TILE_LOOP"], "tile_cnt": [4], "mean(L1_TO_L1)": [_TILE_LOOP]}
    )

    medians = _medians([wide, narrow])

    assert len(medians) == 2  # two distinct points, not one merged point


def test_medians_of_no_frames_is_empty():
    assert _medians([]) == {}
    assert _medians([pd.DataFrame()]) == {}


def test_points_csv_streams_every_point_with_its_config(tmp_path):
    from regression_compare import _write_points_csv

    records = [
        {
            "marker": "TILE_LOOP",
            "run_type": "L1_TO_L1",
            "current": 2130.0,
            "baseline": 2000.0,
            "delta": 0.065,
            "abs_delta": 130.0,
            "test_module": "perf_matmul",
            "config": (("tile_cnt", 2),),
        },
        {
            "marker": "INIT",
            "run_type": "L1_TO_L1",
            "current": 90.0,
            "baseline": 100.0,
            "delta": -0.10,
            "abs_delta": -10.0,
            "test_module": "perf_pack",
            "config": (("dst_index", 0),),
        },
    ]
    out = tmp_path / "p.csv"
    assert _write_points_csv(records, str(out)) is True
    frame = pd.read_csv(out)

    assert list(frame["marker"]) == ["TILE_LOOP", "INIT"]  # worst delta first
    assert set(frame.columns) >= {"tile_cnt", "dst_index", "test_module"}
    assert frame.loc[0, "delta_pct"] == 6.5


def test_points_csv_writes_nothing_when_there_is_nothing(tmp_path):
    from regression_compare import _write_points_csv

    out = tmp_path / "p.csv"
    assert _write_points_csv([], str(out)) is False
    assert not out.exists()


def test_identical_configs_share_one_pair_object():
    from regression_compare import _point_key

    a = _point_key("INIT", [("tile_cnt", 2), ("dst_index", 0)])
    b = _point_key("TILE_LOOP", [("tile_cnt", 2), ("dst_index", 0)])
    assert a[1] == b[1]
    for x, y in zip(a[1], b[1]):
        assert x is y


def _record(module, marker, **config):
    return {
        "marker": marker,
        "run_type": "L1_TO_L1",
        "current": 110.0,
        "baseline": 100.0,
        "delta": 0.10,
        "abs_delta": 10.0,
        "test_module": module,
        "config": tuple(sorted(config.items())),
    }


def test_report_table_names_the_test_and_keeps_the_whole_config():
    from regression_compare import _delta_table

    wide = {f"param_{i}": f"a_long_enough_value_{i}" for i in range(20)}
    text = "\n".join(
        _delta_table(
            [_record("perf_eltwise_unary_sfpu", "TILE_LOOP", **wide)],
            caption="Top 1 regressions",
        )
    )

    assert "| test |" in text
    assert "perf_eltwise_unary_sfpu" in text
    assert "…" not in text
    for key, value in wide.items():
        assert f"`{key}={value}`" in text


def test_report_table_puts_the_distinguishing_parameter_first():
    from regression_compare import _delta_table

    rows = [
        _record("perf_pack", "INIT", shared="x", tile_cnt=2),
        _record("perf_pack", "INIT", shared="x", tile_cnt=8),
    ]
    body = [l for l in _delta_table(rows, caption="c") if l.startswith("1. ")][0]
    assert body.startswith("1. `tile_cnt=2`")


def test_defaults_are_the_measured_ones():
    # perf_compare_commits.sh and the gate both lean on these; keep them pinned.
    assert DEFAULT_THRESHOLD == 0.02
    assert DEFAULT_MIN_CYCLES == 30.0


def test_report_names_the_verdict_and_the_regression(tmp_path):
    current, baseline = _sides(
        tmp_path, (_INIT, _TILE_LOOP), (_INIT, _TILE_LOOP * 1.10)
    )
    result = compare_runs(current, baseline)
    text = render_report(
        result,
        threshold=DEFAULT_THRESHOLD,
        test="LLK perf",
        baseline_sha="aaa",
        current_sha="bbb",
    )
    assert "REGRESSIONS FOUND" in text
    assert "TILE_LOOP" in text and "+10.0%" in text


def test_clean_report_says_so(tmp_path):
    current, baseline = _sides(tmp_path, (_INIT, _TILE_LOOP), (_INIT, _TILE_LOOP))
    text = render_report(
        compare_runs(current, baseline),
        threshold=DEFAULT_THRESHOLD,
        test="LLK perf",
        baseline_sha="aaa",
        current_sha="aaa",
    )
    assert "no regressions" in text


def test_two_modules_on_one_config_stay_separate_points(tmp_path):
    """perf_a regresses 20%, perf_b does not; the regression is not halved."""
    base = [
        _csv(tmp_path, "b0.csv", _INIT, _TILE_LOOP, module="perf_a"),
        _csv(tmp_path, "b1.csv", _INIT, _TILE_LOOP, module="perf_b"),
    ]
    cur = [
        _csv(tmp_path, "c0.csv", _INIT, _TILE_LOOP * 1.2, module="perf_a"),
        _csv(tmp_path, "c1.csv", _INIT, _TILE_LOOP, module="perf_b"),
    ]

    result = compare_runs(cur, base)

    (reg,) = result["regressions"]
    assert reg["test_module"] == "perf_a"
    assert abs(reg["delta"] - 0.20) < 1e-9


def test_run_type_presets_select_the_right_metrics(tmp_path):
    base = [_csv(tmp_path, "b.csv", _INIT, _TILE_LOOP, run_type="MATH_ISOLATE")]
    cur = [_csv(tmp_path, "c.csv", _INIT, _TILE_LOOP + 200, run_type="MATH_ISOLATE")]

    assert compare_runs(cur, base, run_types="ALL_ISOLATION_MODES")["regressions"]
    assert compare_runs(cur, base, run_types="ALL_MODES")["regressions"]
    assert not compare_runs(cur, base, run_types="L1_TO_L1")["records"]


def test_points_that_differ_only_in_marker_and_run_type_are_one_finding():
    from regression_compare import _delta_table

    rows = [
        dict(_record("perf_eltwise_unary_sfpu", marker, mathop="Square"), run_type=rt)
        for marker in ("TILE_LOOP", "KERNEL")
        for rt in ("L1_TO_L1", "MATH_ISOLATE")
    ]
    text = "\n".join(_delta_table(rows, caption="c"))
    body = [l for l in text.splitlines() if l.startswith("| 1 |")]
    assert len(body) == 1 and "| 4 |" in body[0]
    assert "KERNEL, TILE_LOOP" in body[0] and "L1_TO_L1, MATH_ISOLATE" in body[0]
    assert not any(l.startswith("| 2 |") for l in text.splitlines())


def test_a_module_threshold_overrides_the_default(tmp_path):
    rise = (_INIT, _TILE_LOOP * 1.08)
    noisy = _sides(tmp_path / "a", (_INIT, _TILE_LOOP), rise, module="perf_matmul")
    quiet = _sides(tmp_path / "b", (_INIT, _TILE_LOOP), rise, module="perf_pack")
    result = compare_runs(
        noisy[0] + quiet[0],
        noisy[1] + quiet[1],
        threshold=0.03,
        module_thresholds={"perf_matmul": 0.10},
    )
    assert {r["test_module"] for r in result["regressions"]} == {"perf_pack"}


def test_the_report_names_the_module_thresholds(tmp_path):
    current, baseline = _sides(tmp_path, (_INIT, _TILE_LOOP), (_INIT, _TILE_LOOP + 200))
    text = render_report(
        compare_runs(current, baseline),
        threshold=0.03,
        test="t",
        baseline_sha="a",
        current_sha="b",
        module_thresholds={"perf_matmul": 0.10},
    )
    assert "Per-module thresholds: perf_matmul 10%." in text


def test_a_bad_module_threshold_is_refused():
    from regression_compare import _module_thresholds

    assert _module_thresholds("") == {}
    assert _module_thresholds("perf_matmul=0.25, perf_math_matmul=0.25") == {
        "perf_matmul": 0.25,
        "perf_math_matmul": 0.25,
    }
    with pytest.raises(SystemExit):  # allow-pytest.raises: no expect_error in LLK suite
        _module_thresholds("perf_matmul")


def test_a_run_type_preset_selects_its_run_types(tmp_path):
    rise = (_INIT, _TILE_LOOP + 200)
    iso = _sides(tmp_path / "i", (_INIT, _TILE_LOOP), rise, run_type="PACK_ISOLATE")
    l1 = _sides(tmp_path / "l", (_INIT, _TILE_LOOP), rise, run_type="L1_TO_L1")
    result = compare_runs(
        iso[0] + l1[0], iso[1] + l1[1], run_types="ALL_ISOLATION_MODES"
    )
    assert {r["run_type"] for r in result["regressions"]} == {"PACK_ISOLATE"}
