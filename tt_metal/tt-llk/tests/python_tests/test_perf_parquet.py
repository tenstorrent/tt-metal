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
from helpers.perf_parquet import (
    align_to_schema,
    arrow_schema,
    build_run_batch,
    stamp_provenance,
    to_table,
    write_parquet,
    write_run_batch,
)
from helpers.perf_wide_schema import DB_SCHEMA

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
    return pd.DataFrame(
        {
            "marker": ["INIT", "TILE_LOOP"],
            "mean(MATH_ISOLATE)": [10.5, 20.0],
            "std(MATH_ISOLATE)": [1.0, 2.0],
            "tile_cnt": [4, 4],
            "loop_factor": [1, 1],
            "approx_mode": ["No", "No"],
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
    assert list(back["mean(MATH_ISOLATE)"].dropna()) == [10.5, 20.0]
    assert set(back["arch"]) == {"wormhole"}


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
            "marker": ["INIT", "TILE_LOOP"],
            "mean(PACK_ISOLATE)": [5.0, 6.0],
            "num_faces": [4, 4],
            "tile_cnt": [2, 2],
        }
    )


def test_run_batch_compacts_multiple_tests():
    # Two tests with different columns -> one batch, one schema, rows summed.
    table = build_run_batch(
        {"perf_a": _output_row(), "perf_b": _output_row_b()}, **_RUN_PROV
    )
    assert table.schema.names == [c.name for c in DB_SCHEMA]

    df = table.to_pandas()
    assert len(df) == 4  # 2 markers x 2 tests
    assert set(df["test_name"]) == {"perf_a", "perf_b"}
    # A column only one test emits is NULL on the other test's rows.
    assert df[df["test_name"] == "perf_b"]["mean(MATH_ISOLATE)"].isna().all()
    assert df[df["test_name"] == "perf_a"]["num_faces"].isna().all()


def test_run_batch_rejects_missing_mandatory_provenance():
    prov = dict(_RUN_PROV, commit_sha=None)
    with pytest.raises(ValueError, match="commit_sha"):
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
    assert table.num_rows == 4
