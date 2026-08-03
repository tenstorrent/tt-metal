# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests for historical CSV -> Parquet migration (Milestone 3)."""

import json

import pandas as pd
import pyarrow.parquet as pq
from helpers.perf_migrate import MigrationRun, discover_runs, migrate_runs
from helpers.perf_wide_schema import DB_SCHEMA


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


def test_migrate_is_deterministic(tmp_path):
    runs = [_run(tmp_path / "src", "run_x", "wormhole", {"perf_a": _rows_a()})]

    migrate_runs(runs, tmp_path / "out1")
    migrate_runs(runs, tmp_path / "out2")

    a = pq.read_table(tmp_path / "out1" / "run_x.parquet")
    b = pq.read_table(tmp_path / "out2" / "run_x.parquet")
    assert a.equals(b)


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


def test_migrate_skips_already_migrated(tmp_path):
    run = _run(tmp_path / "src", "run_x", "wormhole", {"perf_a": _rows_a()})
    out = tmp_path / "out"

    migrate_runs([run], out)
    report = migrate_runs([run], out)  # second pass

    assert report["run_x"]["skipped"] is True


def test_discover_runs_parses_arch_and_sidecar(tmp_path):
    root = tmp_path / "archive"
    _csv(root / "outer_perf-data-blackhole-1" / "perf_data", "perf_a.csv", _rows_a())
    sidecar_dir = root / "nightly_777"
    _csv(sidecar_dir / "perf_data", "perf_a.csv", _rows_a())
    (sidecar_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "arch": "quasar",
                "commit_sha": "deadbeef",
                "timestamp": "T",
                "pipeline": "PR",
            }
        )
    )

    runs = {r.run_id: r for r in discover_runs(root)}

    # arch parsed from the folder name, no sidecar -> commit unknown
    assert runs["outer_perf-data-blackhole-1"].arch == "blackhole"
    assert runs["outer_perf-data-blackhole-1"].commit_sha == "unknown"
    # sidecar overrides provenance
    assert runs["nightly_777"].arch == "quasar"
    assert runs["nightly_777"].commit_sha == "deadbeef"
    assert runs["nightly_777"].pipeline == "PR"
