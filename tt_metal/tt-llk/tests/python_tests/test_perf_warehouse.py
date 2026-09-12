# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the storage seam: the factory and the real Snowflake backend.

The Snowflake round-trip runs the *actual* SnowflakeWarehouse.load/query code
against a local Snowflake emulator (fakesnow, backed by DuckDB) — so the real
code path is exercised with no Snowflake account. Backend-specific DuckDB tests
live in test_perf_warehouse_duckdb.py.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.perf.warehouse import PerfWarehouse, SnowflakeWarehouse, get_warehouse


def test_factory_selects_snowflake(monkeypatch):
    monkeypatch.setenv("PERF_WAREHOUSE", "snowflake")
    wh = get_warehouse()
    assert isinstance(wh, SnowflakeWarehouse)
    assert isinstance(wh, PerfWarehouse)


def test_factory_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("PERF_WAREHOUSE", "postgres")
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="unknown"
    ):
        get_warehouse()


def test_snowflake_creds_read_from_env(monkeypatch):
    for key in (
        "ACCOUNT",
        "USER",
        "PASSWORD",
        "ROLE",
        "WAREHOUSE",
        "DATABASE",
        "SCHEMA",
    ):
        monkeypatch.setenv(f"SNOWFLAKE_{key}", key.lower())
    wh = SnowflakeWarehouse()
    assert wh.creds["account"] == "account"
    assert wh.creds["warehouse"] == "warehouse"


def test_snowflake_round_trip_on_emulator(tmp_path):
    """Real SnowflakeWarehouse.load/query, exercised on the fakesnow emulator."""
    fakesnow = pytest.importorskip("fakesnow")

    run = tmp_path / "run.parquet"
    pq.write_table(
        pa.table(
            {
                "test_name": ["perf_a", "perf_a"],
                "marker": ["INIT", "KERNEL"],
                "mean(MATH_ISOLATE)": [10.0, 20.0],
            }
        ),
        run,
    )

    with fakesnow.patch():
        wh = SnowflakeWarehouse(database="LLK", schema="PERF")
        # In prod the data team creates the table from our DDL; here create the
        # matching shape so write_pandas(auto_create_table=False) has a target.
        with wh._connect() as con:
            cur = con.cursor()
            cur.execute("CREATE DATABASE IF NOT EXISTS LLK")
            cur.execute("CREATE SCHEMA IF NOT EXISTS LLK.PERF")
            cur.execute(
                "CREATE TABLE IF NOT EXISTS LLK.PERF.LLK_PERF "
                '("test_name" VARCHAR, "marker" VARCHAR, "mean(MATH_ISOLATE)" FLOAT)'
            )

        nrows = wh.load(str(run))
        assert nrows == 2

        # Note: Snowflake folds UNquoted identifiers to upper-case (DuckDB keeps
        # them as written), so the alias is quoted to stay stable across backends.
        df = wh.query(
            'SELECT "test_name", avg("mean(MATH_ISOLATE)") AS "m" '
            "FROM LLK.PERF.LLK_PERF GROUP BY 1"
        )
        assert df.iloc[0]["m"] == 15.0  # avg(10, 20) — the real load+query path
