# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Local DuckDB stand-in for ``PerfWarehouse`` — DELETABLE once Snowflake is live.

Nothing in the pipeline imports this directly; ``get_warehouse()`` imports it
lazily only when ``PERF_WAREHOUSE=duckdb``. DuckDB is a good stand-in for Snowflake
here: embedded (no server), reads Parquet natively, and its analytical SQL (window
functions, ``QUALIFY``, CTEs) is close enough that the compare/dashboard queries
transfer with little change.

To retire it: delete this file, drop the ``duckdb`` branch in ``get_warehouse``, and
remove ``duckdb`` from requirements. No other code changes.
"""

import os
from typing import Optional

import pandas as pd

from .warehouse import DEFAULT_TABLE, PerfWarehouse


class DuckDBWarehouse(PerfWarehouse):
    """One embedded DuckDB file (or in-memory), Parquet-native.

    The table is created from the first Parquet's schema (inference) and appended
    to on later loads. In production the table is created from DB_SCHEMA / the
    generated DDL; here inference keeps the stand-in decoupled from schema version.
    """

    def __init__(self, path: Optional[str] = None, table: str = DEFAULT_TABLE):
        import duckdb

        self.table = table
        self.con = duckdb.connect(path or os.environ.get("PERF_DUCKDB", ":memory:"))

    def load(self, parquet_path: str) -> int:
        # `read_parquet` accepts a single path or a glob; both work here.
        self.con.execute(
            f'CREATE TABLE IF NOT EXISTS "{self.table}" AS '
            "SELECT * FROM read_parquet(?) LIMIT 0",
            [parquet_path],
        )
        self.con.execute(
            f'INSERT INTO "{self.table}" SELECT * FROM read_parquet(?)', [parquet_path]
        )
        return int(
            self.con.execute(f'SELECT count(*) FROM "{self.table}"').fetchone()[0]
        )

    def query(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()
