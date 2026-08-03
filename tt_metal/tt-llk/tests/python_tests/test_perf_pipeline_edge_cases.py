# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Adversarial edge cases — deliberately trying to break the pipeline.

Degenerate inputs, hostile names, malformed metadata, and numeric corner cases
across convert / migrate / dashboard / compare. Each asserts either graceful
handling or a loud, correct failure.
"""

import pandas as pd
import pyarrow.parquet as pq
import pytest
from helpers.perf_compare import compare_to_history
from helpers.perf_dashboard import dashboard_from_parquet
from helpers.perf_migrate import discover_runs, migrate_runs
from helpers.perf_parquet import (
    convert_csvs_to_parquet,
    parquet_to_csvs,
    write_run_batch,
)

PROV = dict(
    commit_sha="c",
    arch="wormhole",
    run_id="r",
    timestamp="t",
    pipeline="PR",
    pr_number=None,
)


def _csv(path, df):
    df.to_csv(path, index=False)
    return str(path)


def _one(mean=1.0):
    return pd.DataFrame(
        {"marker": ["INIT"], "tile_cnt": [4], "mean(MATH_ISOLATE)": [mean]}
    )


# ── degenerate inputs ─────────────────────────────────────────────────────────


def test_empty_csv_header_only(tmp_path):
    p = _csv(tmp_path / "perf_x.csv", pd.DataFrame({"marker": [], "tile_cnt": []}))
    out = tmp_path / "o.parquet"
    convert_csvs_to_parquet([p], out, **PROV)
    assert pq.read_table(out).num_rows == 0


def test_convert_no_csvs_is_empty_batch(tmp_path):
    out = tmp_path / "o.parquet"
    convert_csvs_to_parquet([], out, **PROV)
    assert pq.read_table(out).num_rows == 0


def test_missing_marker_fails_loud(tmp_path):
    p = _csv(tmp_path / "perf_x.csv", pd.DataFrame({"tile_cnt": [4]}))  # no marker
    with pytest.raises(ValueError, match="marker"):
        convert_csvs_to_parquet([p], tmp_path / "o.parquet", **PROV)


# ── hostile names (path safety) ───────────────────────────────────────────────


def test_dashboard_test_name_with_slash(tmp_path):
    b = tmp_path / "b.parquet"
    write_run_batch({"weird/../name": _one()}, b, **PROV)
    out = tmp_path / "out"
    written = dashboard_from_parquet(b, out)
    for path in written.values():
        assert out.resolve() in path.resolve().parents  # never escapes out_dir


def test_parquet_to_csvs_test_name_with_slash(tmp_path):
    b = tmp_path / "b.parquet"
    write_run_batch({"a/b": _one()}, b, **PROV)
    out = tmp_path / "out"
    written = parquet_to_csvs(b, out)
    for path in written.values():
        assert out.resolve() in path.resolve().parents


# ── migrate robustness ────────────────────────────────────────────────────────


def test_discover_runs_malformed_sidecar(tmp_path):
    run_dir = tmp_path / "run1" / "perf_data"
    run_dir.mkdir(parents=True)
    _csv(run_dir / "perf_a.csv", _one())
    (tmp_path / "run1" / "run_meta.json").write_text("{ not valid json")
    runs = discover_runs(tmp_path)  # must not crash
    assert runs[0].commit_sha == "unknown"


def test_migrate_empty_archive(tmp_path):
    (tmp_path / "empty").mkdir()
    report = migrate_runs(discover_runs(tmp_path / "empty"), tmp_path / "out")
    assert report == {}


# ── compare robustness ────────────────────────────────────────────────────────


def test_compare_empty_history(tmp_path):
    cur = tmp_path / "c.parquet"
    write_run_batch({"perf_a": _one()}, cur, **PROV)
    result = compare_to_history(cur, [], threshold=0.05)
    assert result["regressions"] == []
    assert len(result["new_points"]) == 1


def test_compare_zero_baseline_no_crash(tmp_path):
    h = tmp_path / "h.parquet"
    write_run_batch({"perf_a": _one(0.0)}, h, **PROV)
    c = tmp_path / "c.parquet"
    write_run_batch({"perf_a": _one(5.0)}, c, **PROV)
    result = compare_to_history(c, [h], threshold=0.05)  # baseline 0 -> no div-by-zero
    assert isinstance(result["records"], list)
