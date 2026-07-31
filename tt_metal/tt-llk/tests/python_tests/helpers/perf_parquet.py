# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parquet output for LLK performance reports

Publishes a run's per-test reports as one immutable, typed Parquet batch whose
physical schema is the shared wide schema (perf_wide_schema.DB_SCHEMA).

Data flow
---------
Two entry points both produce {test_name: DataFrame}, then hand it to one
shared core (build_run_batch):

  A) live run   -->  {test_name: DataFrame}              (already in memory)

  B) CSV files  -->  convert_csvs_to_parquet(paths, out, **prov)
                       for each csv:
                         _test_name_from_csv      name from filename
                         pd.read_csv              read the text table
                         cols not in schema    -> diagnostics: DROPPED
                         _coerce_frame_to_schema
                           text -> int / float / bool
                           won't parse         -> NULL, diagnostics: COERCED
                     -->  {test_name: cleaned DataFrame}

                                   |  (both paths)
                                   v
  build_run_batch(frames, **provenance)                        <-- CORE
      1  stamp_provenance      add test_name + run-context columns
      2  pd.concat             stack every test's rows into one frame
      3  to_table              enforce the schema (align + cast; below)
      4  validate_batch        mandatory columns must not be NULL
                                   |
                                   v
  to_table(df)
      align_to_schema          missing cols -> NULL, drop extras, order
      arrow_schema             column types, via _ARROW_TYPES
      pa.array per column      cast to type; string cols stringify values
                                   |
                                   v
  pq.write_table(..., zstd)     -->     one immutable  run.parquet

Entry points:
  write_run_batch          = build_run_batch + write        (frames in memory)
  convert_csvs_to_parquet  = read + coerce + build + write  (csv files on disk)

Notes:
  * One file per RUN, not per test: build_run_batch compacts every test's rows.
  * A test emits only its own columns; align_to_schema fills the rest with NULL.
  * Provenance (commit/arch/run_id/...) is added by CI here, not by the test.
  * No device libraries (pandas + pyarrow only): builds and tests hardware-free.
"""

import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .perf_wide_schema import DB_SCHEMA, MANDATORY

_BOOL_MAP = {
    True: True,
    False: False,
    "True": True,
    "False": False,
    "true": True,
    "false": False,
    1: True,
    0: False,
}

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
    arrays = []
    for field in schema:
        col = aligned[field.name]
        # A string column may hold non strings (a bool unpack_to_dest, an enum
        # dest_acc); stringify non-null values so Arrow accepts them, keep nulls.
        if field.type == pa.string():
            col = col.map(lambda v: v if pd.isna(v) else str(v))
        arrays.append(pa.array(col, type=field.type, from_pandas=True))
    return pa.Table.from_arrays(arrays, schema=schema)


def write_parquet(df, path, columns=DB_SCHEMA, compression="zstd"):
    """Align df to the schema and write one immutable typed Parquet file."""
    pq.write_table(to_table(df, columns), path, compression=compression)


#  Run-level publication


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


#  CSV -> Parquet conversion


def _test_name_from_csv(path) -> str:
    """perf_fast_untilize[.post|.counters].csv -> perf_fast_untilize."""
    name = Path(path).name
    return re.sub(r"\.(?:post|counters)?\.?csv$", "", name).rstrip(".")


def _coerce_frame_to_schema(df, schema_by_name):
    """Coerce each known column to its declared type. A value that does not fit
    (e.g. value_bits "2.0f" in an int column) becomes NULL and is reported, so a
    dirty CSV converts instead of crashing. Returns (coerced_df, report)."""
    out = df.copy()
    report = {}
    for col in out.columns:
        spec = schema_by_name.get(col)
        if spec is None:
            continue  # unknown column; align_to_schema drops it
        before = out[col]
        if spec.dtype in ("int64", "float64"):
            after = pd.to_numeric(before, errors="coerce")
        elif spec.dtype == "bool":
            after = before.map(_BOOL_MAP)
        else:
            continue  # string: cast happens at Arrow conversion
        bad = before.notna() & after.isna()
        if bad.any():
            report[col] = {
                "type": spec.dtype,
                "bad": int(bad.sum()),
                "example": before[bad].iloc[0],
            }
        out[col] = after
    return out, report


def _lossy_conversion_message(diagnostics) -> str:
    lines = ["CSV -> Parquet conversion is not lossless (strict mode):"]
    for test, cols in sorted(diagnostics["unknown_columns"].items()):
        lines.append(f"  {test}: columns not in schema, would be DROPPED: {cols}")
    for test, report in sorted(diagnostics["coerced_values"].items()):
        for col, info in sorted(report.items()):
            lines.append(
                f"  {test}: {col} ({info['type']}) has {info['bad']} value(s) that "
                f"don't fit (e.g. {info['example']!r}), would become NULL"
            )
    lines.append(
        "Fix the schema (add the column / correct the type), or pass strict=False."
    )
    return "\n".join(lines)


def convert_csvs_to_parquet(
    csv_paths, out_path, *, compression="zstd", strict=True, **provenance
):
    """Convert a run's per-test CSVs into one typed Parquet batch.

    Reads each CSV, coerces its columns to the schema types, and reuses
    build_run_batch to align + stamp provenance + compact.

    With strict=True (default) the conversion must be lossless: if any column
    would be dropped (not in the schema) or any value coerced to NULL, it raises
    BEFORE writing and no file is produced. Pass strict=False for a best-effort
    conversion that writes anyway. Either way it returns the diagnostics: per
    test, the dropped columns and the coerced values.
    """
    schema_by_name = {c.name: c for c in DB_SCHEMA}
    frames = {}
    diagnostics = {"unknown_columns": {}, "coerced_values": {}}
    for path in csv_paths:
        name = _test_name_from_csv(path)
        df = pd.read_csv(path)
        unknown = sorted(set(df.columns) - set(schema_by_name))
        if unknown:
            diagnostics["unknown_columns"][name] = unknown
        coerced, report = _coerce_frame_to_schema(df, schema_by_name)
        if report:
            diagnostics["coerced_values"].setdefault(name, {}).update(report)
        # Same test can appear across arch dirs / shards: accumulate its rows..
        if name in frames:
            frames[name] = pd.concat([frames[name], coerced], ignore_index=True)
        else:
            frames[name] = coerced

    if strict and (diagnostics["unknown_columns"] or diagnostics["coerced_values"]):
        raise ValueError(_lossy_conversion_message(diagnostics))

    table = build_run_batch(frames, **provenance)
    pq.write_table(table, out_path, compression=compression)
    return diagnostics
