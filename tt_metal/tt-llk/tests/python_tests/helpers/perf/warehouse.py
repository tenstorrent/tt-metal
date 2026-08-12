# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Storage seam for LLK perf data.

Downstream code (publish, compare, dashboard) depends only on the ``PerfWarehouse``
interface and ``get_warehouse()`` — never on a concrete database. Two backends:

  ``SnowflakeWarehouse``  the real target (this file).
  ``DuckDBWarehouse``     a local stand-in for building/testing before the table
                          exists. It lives in ``warehouse_duckdb.py`` and is
                          imported lazily by the factory.

    wh = get_warehouse()             # duckdb (default now) | snowflake
    wh.load("run-wormhole.parquet")  # ingest one run's typed Parquet batch
    df = wh.query("SELECT ...")      # analytical read (dashboard / compare use this)

Retiring the stand-in once Snowflake is live is a three-line change: delete
``warehouse_duckdb.py``, drop the ``duckdb`` branch in ``get_warehouse``, and
remove ``duckdb`` from requirements. Nothing downstream changes.
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


class SnowflakeWarehouse(PerfWarehouse):
    """Real backend, using ``snowflake-connector-python`` (imported lazily, so this
    module loads without it).

    Connection parameters come from the data team (env ``SNOWFLAKE_ACCOUNT`` /
    ``USER`` / ``PASSWORD`` / ``ROLE`` / ``WAREHOUSE`` / ``DATABASE`` / ``SCHEMA``,
    or kwargs). ``load`` uses ``write_pandas`` into the pre-created ``llk_perf``
    table; if the data team prefers a stage + ``COPY INTO`` / Snowpipe, only this
    one method changes. ``query`` runs analytical SQL against the live table — the
    same SQL the DuckDB stand-in was validated on.
    """

    def __init__(self, *, table: str = DEFAULT_TABLE, **creds):
        self.table = table
        merged = creds or _snowflake_creds_from_env()
        self.creds = {k: v for k, v in merged.items() if v is not None}

    def _connect(self):
        from snowflake.connector import connect  # lazy: only when actually used

        return connect(**self.creds)

    def load(self, parquet_path: str) -> int:
        import pyarrow.parquet as pq
        from snowflake.connector.pandas_tools import write_pandas

        df = pq.read_table(parquet_path).to_pandas()
        with self._connect() as con:
            _, _, nrows, _ = write_pandas(
                con,
                df,
                self.table.upper(),
                quote_identifiers=True,
                auto_create_table=False,
            )
        return nrows

    def query(self, sql: str) -> pd.DataFrame:
        with self._connect() as con:
            return con.cursor().execute(sql).fetch_pandas_all()


def _snowflake_creds_from_env() -> dict:
    keys = ("account", "user", "password", "role", "warehouse", "database", "schema")
    return {k: os.environ.get(f"SNOWFLAKE_{k.upper()}") for k in keys}


def get_warehouse(kind: Optional[str] = None) -> PerfWarehouse:
    """Return the configured backend (``PERF_WAREHOUSE`` env, default ``duckdb``)."""
    kind = (kind or os.environ.get("PERF_WAREHOUSE", "duckdb")).lower()
    if kind == "snowflake":
        return SnowflakeWarehouse()
    if kind == "duckdb":
        # Lazy import so the deletable stand-in is only loaded when selected.
        from .warehouse_duckdb import DuckDBWarehouse

        return DuckDBWarehouse()
    raise ValueError(f"unknown PERF_WAREHOUSE={kind!r}; use 'snowflake' or 'duckdb'")
