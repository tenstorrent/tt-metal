# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests for historical CSV -> Parquet migration (Milestone 3)."""

import json
from dataclasses import replace

import pandas as pd
import pyarrow.parquet as pq
from helpers.perf.migrate import (
    MigrationRun,
    discover_runs,
    migrate_runs,
    summarize_coverage,
)
from helpers.perf.wide_schema import DB_SCHEMA


def _csv(dir_, name, df):
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / name
    df.to_csv(path, index=False)
    return path


def _rows_a():
    return pd.DataFrame(
        {
            "marker": ["INIT", "TILE_LOOP"],
            "tile_cnt": [4, 4],
            "mean(MATH_ISOLATE)": [1.0, 2.0],
        }
    )


def _run(src_dir, run_id, arch, tests):
    run_dir = src_dir / run_id
    paths = tuple(_csv(run_dir, f"{name}.csv", df) for name, df in tests.items())
    return MigrationRun(
        run_id=run_id,
        arch=arch,
        commit_sha="c0ffee",
        timestamp="2026-01-01T00:00:00",
        pipeline="nightly",
        csv_paths=paths,
    )


def test_migrate_one_batch_per_run(tmp_path):
    src = tmp_path / "src"
    runs = [
        _run(src, "run_wh", "wormhole", {"perf_a": _rows_a()}),
        _run(src, "run_bh", "blackhole", {"perf_a": _rows_a()}),
    ]
    out = tmp_path / "out"

    migrate_runs(runs, out)

    assert (out / "run_wh.parquet").exists()
    assert (out / "run_bh.parquet").exists()
    table = pq.read_table(out / "run_wh.parquet")
    assert table.schema.names == [c.name for c in DB_SCHEMA]
    df = table.to_pandas()
    assert set(df["arch"]) == {"wormhole"}
    assert set(df["run_id"]) == {"run_wh"}


def test_migrate_deterministic_across_location_and_order(tmp_path):
    # Same run built under two different roots, with CSVs fed in opposite order.
    # The sorted() calls in migrate_runs must make both produce byte-identical files
    # (source-location independence + CSV-ordering independence + codec determinism).
    tests = {"perf_a": _rows_a(), "perf_b": _rows_a()}
    r1 = _run(tmp_path / "src1", "run_x", "wormhole", tests)
    r2 = _run(tmp_path / "src2", "run_x", "wormhole", tests)
    r2 = replace(r2, csv_paths=tuple(reversed(r2.csv_paths)))

    migrate_runs([r1], tmp_path / "out1")
    migrate_runs([r2], tmp_path / "out2")

    b1 = (tmp_path / "out1" / "run_x.parquet").read_bytes()
    b2 = (tmp_path / "out2" / "run_x.parquet").read_bytes()
    assert b1 == b2


def test_migrate_lenient_reports_coverage(tmp_path):
    dirty = pd.DataFrame(
        {"marker": ["INIT"], "made_up_col": [9], "value_bits": ["2.0f"]}
    )
    run = _run(tmp_path / "src", "run_x", "wormhole", {"perf_x": dirty})
    out = tmp_path / "out"

    report = migrate_runs([run], out)

    assert (out / "run_x.parquet").exists()  # written despite dirty data
    assert report["run_x"]["dropped_columns"]["perf_x"] == ["made_up_col"]
    assert "value_bits" in report["run_x"]["coerced_values"]["perf_x"]


def test_migrate_one_dirty_run_does_not_abort_others(tmp_path):
    # A run that raises (marker-less frame -> mandatory 'marker' all-NULL ->
    # validate_batch raises) must be isolated: earlier AND later runs still migrate,
    # the batch is recorded failed, and the whole call never raises.
    src = tmp_path / "src"
    good_before = _run(src, "a_good", "wormhole", {"perf_a": _rows_a()})
    bad = _run(src, "b_bad", "wormhole", {"perf_x": pd.DataFrame({"tile_cnt": [4]})})
    good_after = _run(src, "c_good", "wormhole", {"perf_a": _rows_a()})
    out = tmp_path / "out"

    report = migrate_runs([good_before, bad, good_after], out)  # must not raise

    assert (out / "a_good.parquet").exists()
    assert (out / "c_good.parquet").exists()
    assert not (out / "b_bad.parquet").exists()
    assert not (out / "b_bad.parquet.tmp").exists()  # partial temp cleaned up
    assert report["b_bad"]["failed"] is True
    assert "error" in report["b_bad"]
    assert report["a_good"]["skipped"] is False


def test_migrate_zero_byte_csv_run_fails_gracefully(tmp_path):
    # A run whose only CSV is zero-byte (EmptyDataError) is recorded failed, not raised.
    src = tmp_path / "src"
    run_dir = src / "run_x"
    run_dir.mkdir(parents=True)
    empty = run_dir / "perf_x.csv"
    empty.write_text("")
    run = MigrationRun(
        run_id="run_x",
        arch="wormhole",
        commit_sha="c0ffee",
        timestamp="T",
        pipeline="nightly",
        csv_paths=(empty,),
    )
    out = tmp_path / "out"

    report = migrate_runs([run], out)  # must not raise

    assert report["run_x"]["failed"] is True
    assert not (out / "run_x.parquet").exists()


def test_summarize_counts_coerced_values_not_columns(tmp_path):
    # Two bad values in one column must report "2 coerced value(s)", not "1".
    dirty = pd.DataFrame(
        {"marker": ["INIT", "TILE_LOOP"], "value_bits": ["2.0f", "3.0f"]}
    )
    run = _run(tmp_path / "src", "run_x", "wormhole", {"perf_x": dirty})

    report = migrate_runs([run], tmp_path / "out")
    line = summarize_coverage(report)

    assert "2 coerced value(s)" in line
    assert "1 coerced" not in line


def test_migrate_skips_already_migrated_with_full_report_shape(tmp_path):
    run = _run(tmp_path / "src", "run_x", "wormhole", {"perf_a": _rows_a()})
    out = tmp_path / "out"

    migrate_runs([run], out)
    report = migrate_runs([run], out)  # second pass

    entry = report["run_x"]
    assert entry["skipped"] is True
    # documented shape: every entry carries these 5 keys, even on skip
    assert set(entry) >= {
        "arch",
        "csv_count",
        "dropped_columns",
        "coerced_values",
        "skipped",
    }
    assert entry["csv_count"] == 1


def test_discover_excludes_counters_post_and_empty(tmp_path):
    root = tmp_path / "archive"
    d = root / "nightly_1" / "perf_data"
    _csv(d, "perf_a.csv", _rows_a())
    _csv(d, "perf_a.counters.csv", _rows_a())  # per-worker counter dump -> excluded
    _csv(d, "perf_a.post.csv", _rows_a())  # post-processed twin -> excluded
    (d / "perf_empty.csv").write_text("")  # zero-byte -> excluded

    (run,) = discover_runs(root)

    assert sorted(p.name for p in run.csv_paths) == ["perf_a.csv"]


def test_discover_sidecar_lenient_and_null_fallback(tmp_path):
    root = tmp_path / "archive"
    # broken JSON sidecar -> ignored, folder arch used, discovery not aborted
    d1 = root / "nightly-wormhole-1"
    _csv(d1 / "perf_data", "perf_a.csv", _rows_a())
    (d1 / "run_meta.json").write_text("{not valid json")
    # explicit null arch -> falls back to folder-parsed arch (not None)
    d2 = root / "run-blackhole-2"
    _csv(d2 / "perf_data", "perf_a.csv", _rows_a())
    (d2 / "run_meta.json").write_text(json.dumps({"arch": None, "commit_sha": "abc"}))

    runs = {r.run_id: r for r in discover_runs(root)}

    assert runs["nightly-wormhole-1"].arch == "wormhole"
    assert runs["run-blackhole-2"].arch == "blackhole"
    assert runs["run-blackhole-2"].commit_sha == "abc"


def test_discover_runs_parses_arch_and_sidecar(tmp_path):
    root = tmp_path / "archive"
    _csv(root / "run-blackhole-1" / "perf_data", "perf_a.csv", _rows_a())
    sidecar_dir = root / "nightly_777"
    _csv(sidecar_dir / "perf_data", "perf_a.csv", _rows_a())
    (sidecar_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "arch": "quasar",
                "commit_sha": "deadbeef",
                "timestamp": "T",
                "pipeline": "pr",
            }
        )
    )

    runs = {r.run_id: r for r in discover_runs(root)}

    # arch parsed from the folder name, no sidecar -> commit unknown
    assert runs["run-blackhole-1"].arch == "blackhole"
    assert runs["run-blackhole-1"].commit_sha == "unknown"
    # sidecar overrides provenance
    assert runs["nightly_777"].arch == "quasar"
    assert runs["nightly_777"].commit_sha == "deadbeef"
    assert runs["nightly_777"].pipeline == "pr"
