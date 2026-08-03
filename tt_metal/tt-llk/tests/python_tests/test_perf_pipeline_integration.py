# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration test: the full pipeline the way a real run uses it.

Chains every piece together on synthetic data (no chip):

    history runs -> Parquet batches
    current run  -> CSV  ->  Parquet  ->  CSV (reverse)  ->  dashboard
                                       ->  compare to history  ->  flag regression

This is the single "everything works together" check on the tip of the stack.
"""

import pandas as pd
import pyarrow.parquet as pq
from helpers.perf_compare import compare_to_history, summarize_comparison
from helpers.perf_dashboard import dashboard_from_parquet
from helpers.perf_parquet import (
    convert_csvs_to_parquet,
    parquet_to_csvs,
    write_run_batch,
)
from helpers.perf_wide_schema import DB_SCHEMA

_PROV = dict(
    commit_sha="cur_commit",
    arch="wormhole",
    run_id="current",
    timestamp="t",
    pipeline="PR",
    pr_number="7",
)


def _report(mean_math):
    """What one perf test emits: a config (tile_cnt=4) with MATH_ISOLATE timings."""
    return pd.DataFrame(
        {
            "marker": ["INIT", "TILE_LOOP"],
            "tile_cnt": [4, 4],
            "mean(MATH_ISOLATE)": list(mean_math),
        }
    )


def test_full_pipeline_run_to_regression(tmp_path):
    # 1. HISTORY: several past runs, published as Parquet batches.
    history = []
    for i, vals in enumerate([[100.0, 200.0], [101.0, 199.0], [99.0, 201.0]]):
        path = tmp_path / f"hist_{i}.parquet"
        write_run_batch(
            {"perf_matmul": _report(vals)}, path, **dict(_PROV, run_id=f"h{i}")
        )
        history.append(path)

    # 2. CURRENT run: produces a CSV, as a real perf test would. INIT regressed +30%.
    perf_data = tmp_path / "perf_data"
    perf_data.mkdir()
    _report([130.0, 205.0]).to_csv(perf_data / "perf_matmul.csv", index=False)

    # 3. CSV -> Parquet (the live conversion / migration path).
    current = tmp_path / "current.parquet"
    convert_csvs_to_parquet([str(perf_data / "perf_matmul.csv")], current, **_PROV)
    assert pq.read_table(current).schema.names == [c.name for c in DB_SCHEMA]

    # 4. Parquet -> CSV round-trips (reverse converter).
    back = parquet_to_csvs(current, tmp_path / "back")
    assert "perf_matmul" in back

    # 5. Dashboard straight from Parquet (no database).
    html = dashboard_from_parquet(current, tmp_path / "dash")
    assert (tmp_path / "dash" / "perf_matmul.html").exists()
    assert "perf_matmul" in html

    # 6. Compare to history: the regressed point is flagged, the stable one is not.
    result = compare_to_history(current, history, threshold=0.05)
    flagged = {(r["test"], r["marker"], r["run_type"]) for r in result["regressions"]}
    assert ("perf_matmul", "INIT", "MATH_ISOLATE") in flagged
    assert ("perf_matmul", "TILE_LOOP", "MATH_ISOLATE") not in flagged
    assert "REGRESSION" in summarize_comparison(result)
