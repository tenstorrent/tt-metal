# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Storage seam for LLK perf data.

Everything downstream (loader, dashboard, compare) talks to a ``PerfWarehouse``
instead of to a specific database. A local DuckDB backend stands in now; a
Snowflake backend drops in later, chosen by the ``PERF_WAREHOUSE`` env var. This
lets the whole pipeline be built and tested before the real table exists.

    wh = get_warehouse()            # duckdb (default) | snowflake
    wh.load("run-wormhole.parquet") # ingest one run's typed Parquet batch
    df = wh.query("SELECT ...")     # analytical read (dashboard / compare use this)

DuckDB is a good stand-in: embedded (no server), reads Parquet natively, and its
analytical SQL is close to Snowflake's, so the compare/dashboard queries transfer
with little change. Real SQL differences (MERGE, some date fns, VARIANT) are kept
inside the two backends' ``load`` methods.
"""

import os
from typing import Optional

import pandas as pd

DEFAULT_TABLE = "llk_perf"


class PerfWarehouse:
    """Interface both backends implement. Downstream code depends only on this."""

    def load(self, parquet_path: str) -> int:  # pragma: no cover - interface
        """Ingest one run's Parquet batch into the table. Returns total row count."""
        raise NotImplementedError

    def query(self, sql: str) -> pd.DataFrame:  # pragma: no cover - interface
        """Run analytical SQL and return the result as a DataFrame."""
        raise NotImplementedError


class DuckDBWarehouse(PerfWarehouse):
    """Local stand-in. One embedded DuckDB file (or in-memory), Parquet-native.

    The table is created from the first Parquet's schema (inference) and appended
    to on later loads. In production the table is created from DB_SCHEMA / the
    generated DDL; here inference keeps the prototype decoupled from schema version.
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


class SnowflakeWarehouse(PerfWarehouse):
    """Real backend — implemented when the data team's table is ready.

    ``load`` will either ``write_pandas`` or PUT-to-stage + ``COPY INTO`` (open
    question Q1); ``query`` runs against the live table. Same two methods, so
    dashboard/compare are unchanged when we switch ``PERF_WAREHOUSE=snowflake``.
    """

    def __init__(self, **creds):
        self.creds = creds or _snowflake_creds_from_env()

    def load(self, parquet_path: str) -> int:  # pragma: no cover - needs Snowflake
        raise NotImplementedError("SnowflakeWarehouse.load: wire when the table exists")

    def query(self, sql: str) -> pd.DataFrame:  # pragma: no cover - needs Snowflake
        raise NotImplementedError(
            "SnowflakeWarehouse.query: wire when the table exists"
        )


def _snowflake_creds_from_env() -> dict:
    keys = ("account", "user", "role", "warehouse", "database", "schema")
    return {k: os.environ.get(f"SNOWFLAKE_{k.upper()}") for k in keys}


def get_warehouse(kind: Optional[str] = None) -> PerfWarehouse:
    """Return the configured backend (``PERF_WAREHOUSE`` env, default ``duckdb``)."""
    kind = (kind or os.environ.get("PERF_WAREHOUSE", "duckdb")).lower()
    backends = {"duckdb": DuckDBWarehouse, "snowflake": SnowflakeWarehouse}
    if kind not in backends:
        raise ValueError(
            f"unknown PERF_WAREHOUSE={kind!r}; use one of {list(backends)}"
        )
    return backends[kind]()
