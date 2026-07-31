# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Typed Parquet output for LLK performance reports (Milestone 2).

Writes a report DataFrame to an immutable, typed Parquet file whose physical
schema is the shared wide schema (perf_wide_schema.DB_SCHEMA). ``align_to_schema``
fills columns a test did not emit with NULL and orders them, so every published
Parquet file carries one identical schema regardless of which test produced the
rows.

Needs pyarrow, but no device libraries — builds and validates without hardware.
"""

import pyarrow as pa
import pyarrow.parquet as pq

from .perf_wide_schema import DB_SCHEMA

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
