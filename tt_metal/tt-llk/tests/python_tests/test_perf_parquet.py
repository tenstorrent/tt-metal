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
from helpers.perf_parquet import align_to_schema, arrow_schema, to_table, write_parquet
from helpers.perf_wide_schema import DB_SCHEMA


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
    df = df.copy()
    df["test_name"] = "perf_x"
    df["commit_sha"] = "abc123"
    df["arch"] = "wormhole"
    df["run_id"] = "42"
    df["timestamp"] = "2026-01-01T00:00:00"
    df["pipeline"] = "PR"
    df["pr_number"] = "7"
    return df


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
