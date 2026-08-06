# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests for the Parquet -> HTML dashboard."""

import pandas as pd
from helpers.perf_dashboard import dashboard_from_parquet
from helpers.perf_parquet import write_run_batch

_RUN_PROV = dict(
    commit_sha="abc123",
    arch="wormhole",
    run_id="42",
    timestamp="2026-01-01T00:00:00",
    pipeline="PR",
    pr_number="7",
)


def _rows_math():
    return pd.DataFrame(
        {
            "marker": ["INIT", "TILE_LOOP"],
            "tile_cnt": [4, 8],
            "mean(MATH_ISOLATE)": [10.5, 20.0],
        }
    )


def _rows_pack():
    return pd.DataFrame(
        {
            "marker": ["INIT", "TILE_LOOP"],
            "tile_cnt": [2, 4],
            "mean(PACK_ISOLATE)": [5.0, 6.0],
        }
    )


def test_dashboard_from_parquet_one_html_per_test(tmp_path):
    batch = tmp_path / "batch.parquet"
    write_run_batch(
        {"perf_math": _rows_math(), "perf_pack": _rows_pack()}, batch, **_RUN_PROV
    )

    written = dashboard_from_parquet(batch, tmp_path / "html")

    assert set(written) == {"perf_math", "perf_pack"}
    for name in ("perf_math", "perf_pack"):
        assert (tmp_path / "html" / f"{name}.html").exists()

    html = (tmp_path / "html" / "perf_math.html").read_text()
    assert "Performance: perf_math" in html
    assert "mean(MATH_ISOLATE)" in html
    assert "tile_cnt" in html  # sweep config shows up in the hover data


def test_dashboard_skips_test_without_mean(tmp_path):
    # A test with only a marker (no mean column) produces no plot.
    batch = tmp_path / "batch.parquet"
    only_marker = pd.DataFrame({"marker": ["INIT"], "tile_cnt": [1]})
    write_run_batch({"perf_empty": only_marker}, batch, **_RUN_PROV)

    written = dashboard_from_parquet(batch, tmp_path / "html")

    assert written == {}
