# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Typed Parquet output for LLK performance reports (Milestone 2).

Publishes a run's per-test reports as one immutable, typed Parquet batch whose
physical schema is the shared wide schema (perf_wide_schema.DB_SCHEMA):

  - stamp_provenance adds the run-context columns CI knows (commit, arch, ...).
  - build_run_batch aligns every per-test frame to the schema (columns a test
    did not emit become NULL), stamps provenance, and compacts them into ONE
    run-level table. One file per run, not per test.
  - align_to_schema / to_table fill missing columns with NULL and cast each
    column to its declared Arrow type (nullable-int safe).

Needs pyarrow, but no device libraries — builds and validates without hardware.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .perf_wide_schema import DB_SCHEMA, MANDATORY

_ARROW_TYPES = {
    "int64": pa.int64(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
    "string": pa.string(),
}


def arrow_schema(columns=DB_SCHEMA) -> pa.Schema:
    """pyarrow schema from the wide-schema Column list (name, type, nullability)."""
    return pa.schema(
        [pa.field(c.name, _ARROW_TYPES[c.dtype], nullable=c.nullable) for c in columns]
    )


def align_to_schema(df, columns=DB_SCHEMA):
    """Reindex df to the schema: add missing columns as NULL, drop extras, order."""
    return df.reindex(columns=[c.name for c in columns])


def to_table(df, columns=DB_SCHEMA) -> pa.Table:
    """Build a typed Arrow table: align, then cast each column to its schema type.

    Building column-by-column with ``from_pandas=True`` maps NaN/None to null and
    coerces to the declared type, which handles nullable ints (a pandas int column
    with a NULL is float64) that a whole-frame conversion would reject.
    """
    schema = arrow_schema(columns)
    aligned = align_to_schema(df, columns)
    arrays = [
        pa.array(aligned[field.name], type=field.type, from_pandas=True)
        for field in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def write_parquet(df, path, columns=DB_SCHEMA, compression="zstd"):
    """Align df to the schema and write one immutable typed Parquet file."""
    pq.write_table(to_table(df, columns), path, compression=compression)


# ── Run-level publication ─────────────────────────────────────────────────────


def stamp_provenance(
    df, *, test_name, commit_sha, arch, run_id, timestamp, pipeline, pr_number=None
):
    """Add the run-context columns (added by CI, not produced by the test).

    ``test_name`` is per test; the rest identify the run. ``pipeline`` ("PR" or
    "nightly") is how PR and nightly rows share one schema but stay distinguishable.
    """
    out = df.copy()
    out["test_name"] = test_name
    out["commit_sha"] = commit_sha
    out["arch"] = arch
    out["run_id"] = run_id
    out["timestamp"] = timestamp
    out["pipeline"] = pipeline
    out["pr_number"] = pr_number
    return out


def validate_batch(table):
    """Every mandatory (non-nullable) column must be fully populated."""
    for name in MANDATORY:
        null_count = table.column(name).null_count
        if null_count:
            raise ValueError(
                f"run batch has {null_count} NULL(s) in mandatory column '{name}'"
            )


def build_run_batch(
    test_frames, *, commit_sha, arch, run_id, timestamp, pipeline, pr_number=None
):
    """Compact a run's per-test frames into one run-level table.

    ``test_frames`` maps test_name -> that test's report DataFrame. Each frame is
    stamped with run provenance and concatenated; ``to_table`` then aligns the
    union to DB_SCHEMA (missing columns -> NULL) and casts types. One row is one
    test configuration in one execution context.
    """
    stamped = [
        stamp_provenance(
            df,
            test_name=name,
            commit_sha=commit_sha,
            arch=arch,
            run_id=run_id,
            timestamp=timestamp,
            pipeline=pipeline,
            pr_number=pr_number,
        )
        for name, df in test_frames.items()
    ]
    combined = (
        pd.concat(stamped, ignore_index=True, sort=False)
        if stamped
        else pd.DataFrame(columns=[c.name for c in DB_SCHEMA])
    )
    table = to_table(combined)
    validate_batch(table)
    return table


def write_run_batch(test_frames, path, *, compression="zstd", **provenance):
    """Write a run's per-test frames as one immutable typed Parquet batch."""
    pq.write_table(
        build_run_batch(test_frames, **provenance), path, compression=compression
    )
