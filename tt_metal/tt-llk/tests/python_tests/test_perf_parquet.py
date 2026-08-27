# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free Parquet output tests (Milestone 2 / #51249).

Checks that a report DataFrame writes to a typed Parquet file whose schema is the
shared wide schema (DB_SCHEMA), that columns a test did not emit become NULL
(not dropped), and that CSV and Parquet carry the same data. Needs pyarrow, no chip.
"""

import pandas as pd
import pyarrow.parquet as pq
import pytest
from helpers.metrics import export_metrics
from helpers.perf.parquet import (
    align_to_schema,
    arrow_schema,
    build_run_batch,
    convert_csvs_to_parquet,
    parquet_to_csvs,
    stamp_provenance,
    to_table,
    write_parquet,
    write_run_batch,
)
from helpers.perf.schema import MARKER, METRIC_BASES, RUN_TYPE_NAMES
from helpers.perf.test_schemas import PERF_TEST_SCHEMAS
from helpers.perf.wide_schema import DB_SCHEMA, DROPPED_COLUMNS

_RUN_PROV = dict(
    commit_sha="abc123",
    arch="wormhole",
    run_id="42",
    timestamp="2026-01-01T00:00:00",
    pipeline="PR",
    pr_number="7",
)


def _output_row():
    """What one perf test emits: a few OUTPUT columns, the rest absent."""
    # Real perf reports carry three profiler markers per config: INIT, KERNEL,
    # TILE_LOOP (see the marker values in nightly CSVs).
    return pd.DataFrame(
        {
            "marker": ["INIT", "KERNEL", "TILE_LOOP"],
            "mean(MATH_ISOLATE)": [10.5, 15.0, 20.0],
            "std(MATH_ISOLATE)": [1.0, 1.5, 2.0],
            "tile_cnt": [4, 4, 4],
            "loop_factor": [1, 1, 1],
            "approx_mode": ["No", "No", "No"],
        }
    )


def _stamp_provenance(df):
    return stamp_provenance(df, test_name="perf_x", **_RUN_PROV)


def test_arrow_schema_matches_db_schema():
    schema = arrow_schema()
    assert schema.names == [c.name for c in DB_SCHEMA]
    for field, col in zip(schema, DB_SCHEMA):
        assert field.nullable == col.nullable


def test_report_round_trips_through_parquet(tmp_path):
    df = _stamp_provenance(_output_row())
    path = tmp_path / "batch.parquet"
    write_parquet(df, path)

    table = pq.read_table(path)
    assert table.schema.names == [c.name for c in DB_SCHEMA]

    back = table.to_pandas()
    # every source row survives, values intact, provenance stamped on each row.
    assert table.num_rows == len(df)
    assert list(back["marker"]) == ["INIT", "KERNEL", "TILE_LOOP"]
    assert list(back["mean(MATH_ISOLATE)"].dropna()) == [10.5, 15.0, 20.0]
    assert set(back["arch"]) == {"wormhole"}
    assert set(back["commit_sha"]) == {"abc123"}
    assert set(back["test_name"]) == {"perf_x"}
    # tmp_path is torn down by pytest, so the file needs no manual cleanup.


def test_missing_columns_are_null_not_dropped():
    df = _stamp_provenance(_output_row())
    table = to_table(df)
    # a config column this test never emits is present, and entirely NULL
    idx = table.schema.get_field_index("num_faces")
    assert idx != -1
    assert table.column(idx).null_count == len(df)


def test_csv_and_parquet_agree(tmp_path):
    df = _stamp_provenance(_output_row())
    aligned = align_to_schema(df)

    csv_path = tmp_path / "r.csv"
    aligned.to_csv(csv_path, index=False)
    pq_path = tmp_path / "r.parquet"
    write_parquet(df, pq_path)

    from_csv = pd.read_csv(csv_path)
    from_pq = pq.read_table(pq_path).to_pandas()
    assert list(from_csv.columns) == list(from_pq.columns)
    assert list(from_csv["mean(MATH_ISOLATE)"].dropna()) == list(
        from_pq["mean(MATH_ISOLATE)"].dropna()
    )


# ── Run-level publication ─────────────────────────────────────────────────────


def _output_row_b():
    """A second test emitting a different column set (no MATH_ISOLATE)."""
    return pd.DataFrame(
        {
            "marker": ["INIT", "KERNEL", "TILE_LOOP"],
            "mean(PACK_ISOLATE)": [5.0, 5.5, 6.0],
            "num_faces": [4, 4, 4],
            "tile_cnt": [2, 2, 2],
        }
    )


def test_run_batch_compacts_multiple_tests():
    # Two tests with different columns -> one batch, one schema, rows summed.
    table = build_run_batch(
        {"perf_a": _output_row(), "perf_b": _output_row_b()}, **_RUN_PROV
    )
    assert table.schema.names == [c.name for c in DB_SCHEMA]

    df = table.to_pandas()
    assert len(df) == 6  # 3 markers x 2 tests
    assert set(df["test_name"]) == {"perf_a", "perf_b"}
    # A column only one test emits is NULL on the other test's rows.
    assert df[df["test_name"] == "perf_b"]["mean(MATH_ISOLATE)"].isna().all()
    assert df[df["test_name"] == "perf_a"]["num_faces"].isna().all()


def test_run_batch_rejects_missing_mandatory_provenance():
    prov = dict(_RUN_PROV, commit_sha=None)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="commit_sha"
    ):
        build_run_batch({"perf_a": _output_row()}, **prov)


def test_pipeline_column_distinguishes_pr_and_nightly():
    # PR and nightly share the schema; only execution metadata differs.
    prov = dict(_RUN_PROV, pipeline="nightly", pr_number=None)
    table = build_run_batch({"perf_a": _output_row()}, **prov)
    assert set(table.to_pandas()["pipeline"]) == {"nightly"}


def test_write_run_batch_is_one_file_per_run(tmp_path):
    path = tmp_path / "run_42.parquet"
    write_run_batch(
        {"perf_a": _output_row(), "perf_b": _output_row_b()}, path, **_RUN_PROV
    )
    assert path.exists()
    table = pq.read_table(path)
    assert table.schema.names == [c.name for c in DB_SCHEMA]
    assert table.num_rows == 6


# ── CSV -> Parquet conversion ─────────────────────────────────────────────────


def _write_csv(tmp_path, name, df):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_convert_compacts_multiple_csvs(tmp_path):
    a = _write_csv(tmp_path, "perf_a.csv", _output_row())
    b = _write_csv(tmp_path, "perf_b.csv", _output_row_b())
    out = tmp_path / "run.parquet"

    convert_csvs_to_parquet([a, b], out, **_RUN_PROV)

    table = pq.read_table(out)
    assert table.schema.names == [c.name for c in DB_SCHEMA]
    assert set(table.to_pandas()["test_name"]) == {"perf_a", "perf_b"}


def test_convert_lenient_drops_and_reports_unknown_columns(tmp_path):
    df = pd.DataFrame({"marker": ["INIT"], "tile_cnt": [4], "made_up_col": [9]})
    p = _write_csv(tmp_path, "perf_x.csv", df)

    diag = convert_csvs_to_parquet(
        [p], tmp_path / "out.parquet", strict=False, **_RUN_PROV
    )

    assert diag["unknown_columns"]["perf_x"] == ["made_up_col"]
    assert "made_up_col" not in pq.read_table(tmp_path / "out.parquet").schema.names


def test_convert_lenient_coerces_and_reports_bad_values(tmp_path):
    # value_bits is int64 in the schema; "2.0f" can't parse -> NULL, and reported.
    df = pd.DataFrame({"marker": ["INIT"], "value_bits": ["2.0f"], "tile_cnt": [4]})
    p = _write_csv(tmp_path, "perf_x.csv", df)

    diag = convert_csvs_to_parquet(
        [p], tmp_path / "out.parquet", strict=False, **_RUN_PROV
    )

    assert "value_bits" in diag["coerced_values"]["perf_x"]
    table = pq.read_table(tmp_path / "out.parquet")
    idx = table.schema.get_field_index("value_bits")
    assert table.column(idx).null_count == 1


def test_convert_strict_raises_on_unknown_column(tmp_path):
    # Default strict=True: a column not in the schema is data loss -> fail loud.
    df = pd.DataFrame({"marker": ["INIT"], "tile_cnt": [4], "made_up_col": [9]})
    p = _write_csv(tmp_path, "perf_x.csv", df)
    out = tmp_path / "out.parquet"

    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="made_up_col"
    ):
        convert_csvs_to_parquet([p], out, **_RUN_PROV)
    assert not out.exists()  # nothing written on a lossy conversion


def test_convert_strict_raises_on_bad_value(tmp_path):
    # Default strict=True: an unparsable value would become NULL -> fail loud.
    df = pd.DataFrame({"marker": ["INIT"], "value_bits": ["2.0f"], "tile_cnt": [4]})
    p = _write_csv(tmp_path, "perf_x.csv", df)
    out = tmp_path / "out.parquet"

    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="value_bits"
    ):
        convert_csvs_to_parquet([p], out, **_RUN_PROV)
    assert not out.exists()


def test_convert_stringifies_bool_valued_string_column(tmp_path):
    # unpack_to_dest is "string" in the schema but a bool in the CSV -> "True"/"False".
    df = pd.DataFrame(
        {"marker": ["INIT", "TILE_LOOP"], "unpack_to_dest": [True, False]}
    )
    p = _write_csv(tmp_path, "perf_x.csv", df)

    convert_csvs_to_parquet([p], tmp_path / "out.parquet", **_RUN_PROV)

    back = pq.read_table(tmp_path / "out.parquet").to_pandas()
    assert list(back["unpack_to_dest"]) == ["True", "False"]


def test_convert_drops_dropped_columns_strict_safe(tmp_path):
    # TEXT_SIZE(...) columns are in DROPPED_COLUMNS (per-stage ELF code size, not
    # used by the gate): the converter drops them and does NOT trip the strict
    # unknown-column guard.
    df = pd.DataFrame(
        {
            "marker": ["INIT"],
            "num_blocks": [4],
            "TEXT_SIZE(MATH_ISOLATE)": [2048],
            "tile_cnt": [2],
        }
    )
    p = _write_csv(tmp_path, "perf_x.csv", df)

    diag = convert_csvs_to_parquet([p], tmp_path / "out.parquet", **_RUN_PROV)

    assert diag["unknown_columns"] == {}  # not flagged as drift
    names = pq.read_table(tmp_path / "out.parquet").schema.names
    assert "num_blocks" in names
    assert "TEXT_SIZE(MATH_ISOLATE)" not in names


def test_convert_keeps_num_blocks_columns(tmp_path):
    # input_/output_num_blocks are real schema columns now: they must survive the
    # round-trip, not be dropped.
    df = pd.DataFrame(
        {
            "marker": ["INIT"],
            "num_blocks": [4],
            "input_num_blocks": [4],
            "output_num_blocks": [4],
            "tile_cnt": [2],
        }
    )
    p = _write_csv(tmp_path, "perf_x.csv", df)

    diag = convert_csvs_to_parquet([p], tmp_path / "out.parquet", **_RUN_PROV)

    assert diag["unknown_columns"] == {}
    names = pq.read_table(tmp_path / "out.parquet").schema.names
    assert "input_num_blocks" in names
    assert "output_num_blocks" in names


def test_catalog_columns_are_in_db_schema():
    # Every WH/BH per-test CSV column must be in the published WH/BH table (or
    # intentionally dropped). Quasar has its own table — see
    # test_perf_parquet_quasar.py.
    schema_names = {c.name for c in DB_SCHEMA} | DROPPED_COLUMNS
    missing = {}
    for test, entry in PERF_TEST_SCHEMAS.items():
        unknown = sorted(set(entry["columns"]) - schema_names)
        if unknown:
            missing[test] = unknown
    assert not missing, (
        "WH/BH perf-test catalog column(s) are not in "
        "helpers.perf.wide_schema.DB_SCHEMA and would be dropped from Parquet: "
        f"{missing}. Add them as nullable columns (or to DROPPED_COLUMNS if they "
        "must not be published)."
    )


def test_counter_metric_columns_are_accounted_for():
    # Drives the real emitter, not the schema's constants, so a rename fails here.
    schema_names = {c.name for c in DB_SCHEMA} | DROPPED_COLUMNS
    unknown = set()
    for run_type in sorted(RUN_TYPE_NAMES):
        for metric in sorted(METRIC_BASES):
            for computed in (
                [{"zone": "ZONE_1", metric: 1.0}],
                [{"zone": "ZONE_1", metric: 1.0}, {"zone": "ZONE_1", metric: 2.0}],
            ):
                frame = export_metrics(computed, run_type, ["INIT", "TILE_LOOP"])
                unknown |= {
                    c for c in frame.columns if c != MARKER and c not in schema_names
                }
    assert not unknown, (
        "helpers.metrics.export_metrics emits counter metric column(s) that are "
        "neither published nor dropped, so a run with --enable-perf-counters fails "
        f"the strict writer: {sorted(unknown)}"
    )


def test_counter_metrics_stay_out_of_the_published_table():
    # The published table is a data contract (DATA-1652) and counter metrics are
    # not part of it: no pipeline passes --enable-perf-counters, so every one
    # would be NULL in every published row.
    published = {c.name for c in DB_SCHEMA}
    leaked = sorted(
        c for c in published if any(c.endswith("_" + m) for m in METRIC_BASES)
    )
    assert not leaked, (
        "counter metric column(s) reached the published table: "
        f"{leaked[:5]}{'...' if len(leaked) > 5 else ''}"
    )


# ── Parquet -> CSV (reverse conversion) ───────────────────────────────────────


def test_parquet_to_csvs_splits_by_test(tmp_path):
    batch = tmp_path / "batch.parquet"
    write_run_batch(
        {"perf_a": _output_row(), "perf_b": _output_row_b()}, batch, **_RUN_PROV
    )

    written = parquet_to_csvs(batch, tmp_path / "csvs")

    assert set(written) == {"perf_a", "perf_b"}
    a = pd.read_csv(written["perf_a"])
    assert "commit_sha" not in a.columns  # provenance dropped
    assert "mean(MATH_ISOLATE)" in a.columns  # this test's own column kept
    assert "num_faces" not in a.columns  # only perf_b emits it -> NULL -> dropped


def test_round_trip_parquet_csv_parquet(tmp_path):
    # Parquet -> per-test CSVs -> Parquet reproduces the same batch.
    batch1 = tmp_path / "batch1.parquet"
    write_run_batch(
        {"perf_a": _output_row(), "perf_b": _output_row_b()}, batch1, **_RUN_PROV
    )

    written = parquet_to_csvs(batch1, tmp_path / "csvs")
    batch2 = tmp_path / "batch2.parquet"
    convert_csvs_to_parquet(list(written.values()), batch2, **_RUN_PROV)

    t1 = pq.read_table(batch1)
    t2 = pq.read_table(batch2)
    assert t1.schema.names == t2.schema.names

    key = ["test_name", "marker"]
    d1 = t1.to_pandas().sort_values(key).reset_index(drop=True)
    d2 = t2.to_pandas().sort_values(key)[d1.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(d1, d2)
