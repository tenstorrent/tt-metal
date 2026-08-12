# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end pipeline over the storage seam (no hardware, no real database).

Proves the whole downstream flow works through a PerfWarehouse (DuckDB stand-in):
load run Parquets -> compare-to-history detects a seeded regression -> dashboard
renders from the warehouse. When Snowflake is ready, only the backend swaps.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.perf_compare import compare_run_to_history
from helpers.perf_dashboard import dashboard_from_warehouse
from helpers.perf_warehouse_duckdb import DuckDBWarehouse


def _run(path, run_id, math):
    """A minimal nightly run Parquet: one test, two markers, one timing metric."""
    pq.write_table(
        pa.table(
            {
                "test_name": ["perf_a", "perf_a"],
                "marker": ["INIT", "KERNEL"],
                "arch": ["wormhole", "wormhole"],
                "pipeline": ["nightly", "nightly"],
                "run_id": [run_id, run_id],
                "tile_cnt": [4, 4],  # a sweep column -> part of the point key
                "mean(MATH_ISOLATE)": [math, math + 10.0],
            }
        ),
        path,
    )
    return str(path)


def test_load_then_compare_flags_regression(tmp_path):
    wh = DuckDBWarehouse(path=":memory:")
    wh.load(_run(tmp_path / "n1.parquet", "n1", 100.0))
    wh.load(_run(tmp_path / "n2.parquet", "n2", 120.0))  # +20% vs baseline

    result = compare_run_to_history(wh, "n2", pipeline="nightly", threshold=0.05)

    assert result["regressions"], "the +20% run should be flagged"
    assert result["regressions"][0]["test"] == "perf_a"
    assert result["regressions"][0]["delta"] == pytest.approx(0.20)


def test_no_regression_when_within_threshold(tmp_path):
    wh = DuckDBWarehouse(path=":memory:")
    wh.load(_run(tmp_path / "n1.parquet", "n1", 100.0))
    wh.load(_run(tmp_path / "n2.parquet", "n2", 102.0))  # +2%, under 5% threshold

    result = compare_run_to_history(wh, "n2", pipeline="nightly", threshold=0.05)

    assert not result["regressions"]
    assert result["records"], "still compared, just not flagged"


def test_dashboard_renders_from_warehouse(tmp_path):
    wh = DuckDBWarehouse(path=":memory:")
    wh.load(_run(tmp_path / "n1.parquet", "n1", 100.0))

    written = dashboard_from_warehouse(wh, tmp_path / "dash")

    assert "perf_a" in written
    assert written["perf_a"].exists()
