# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DuckDB stand-in backend — DELETABLE with warehouse_duckdb.py.

Proves the local stand-in: load a run's Parquet, accumulate across runs, and read
it back with analytical SQL (incl. the window-function shape compare uses). Needs
duckdb + pyarrow, no chip.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from helpers.perf.warehouse import PerfWarehouse, get_warehouse
from helpers.perf.warehouse_duckdb import DuckDBWarehouse


def _write_run(path, *, test_name, math):
    """A tiny run Parquet: a couple of columns incl. a special-char timing name."""
    tbl = pa.table(
        {
            "test_name": [test_name, test_name],
            "marker": ["INIT", "KERNEL"],
            "arch": ["wormhole", "wormhole"],
            "mean(MATH_ISOLATE)": [math, math + 10.0],
        }
    )
    pq.write_table(tbl, path)
    return str(path)


def test_load_returns_rowcount_and_accumulates(tmp_path):
    wh = DuckDBWarehouse(path=":memory:")
    a = _write_run(tmp_path / "run_a.parquet", test_name="perf_a", math=100.0)
    b = _write_run(tmp_path / "run_b.parquet", test_name="perf_b", math=200.0)

    assert wh.load(a) == 2  # first run
    assert wh.load(b) == 4  # second run appended, not replaced


def test_query_returns_dataframe_with_analytical_sql(tmp_path):
    wh = DuckDBWarehouse(path=":memory:")
    wh.load(_write_run(tmp_path / "r.parquet", test_name="perf_a", math=100.0))

    df = wh.query(
        'SELECT test_name, avg("mean(MATH_ISOLATE)") AS m FROM llk_perf GROUP BY test_name'
    )
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["test_name"] == "perf_a"
    assert df.iloc[0]["m"] == 105.0  # avg(100, 110)


def test_window_function_qualify_runs(tmp_path):
    # QUALIFY + window fn is the shape compare uses; must run on the stand-in.
    wh = DuckDBWarehouse(path=":memory:")
    wh.load(_write_run(tmp_path / "r.parquet", test_name="perf_a", math=100.0))
    df = wh.query(
        "SELECT marker FROM llk_perf "
        'QUALIFY row_number() OVER (ORDER BY "mean(MATH_ISOLATE)" DESC) = 1'
    )
    assert list(df["marker"]) == ["KERNEL"]  # the higher value


def test_factory_defaults_to_duckdb(monkeypatch):
    monkeypatch.delenv("PERF_WAREHOUSE", raising=False)
    wh = get_warehouse()
    assert isinstance(wh, DuckDBWarehouse)
    assert isinstance(wh, PerfWarehouse)
