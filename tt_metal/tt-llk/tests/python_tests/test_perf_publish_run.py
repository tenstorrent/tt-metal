# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests for the publish_run CLI (CSVs -> one typed run.parquet).

This is the front of the pipeline (what the CI publish-parquet job runs). Needs
pandas + pyarrow, no chip.
"""

import pandas as pd
import pyarrow.parquet as pq
import pytest
from helpers.perf.publish_run import _run_csvs, publish


def _write_csv(path, df):
    df.to_csv(path, index=False)


def test_run_csvs_excludes_post_and_counters(tmp_path):
    (tmp_path / "perf_a.csv").write_text("marker\nINIT\n")
    (tmp_path / "perf_a.post.csv").write_text("marker\nINIT\n")
    (tmp_path / "perf_a.counters.csv").write_text("marker\nINIT\n")

    names = [p.rsplit("/", 1)[-1] for p in _run_csvs(str(tmp_path))]
    assert names == ["perf_a.csv"]  # the .post/.counters side files are excluded


def test_publish_stamps_provenance(tmp_path, monkeypatch):
    _write_csv(
        tmp_path / "perf_a.csv",
        pd.DataFrame(
            {
                "marker": ["INIT", "KERNEL"],
                "tile_cnt": [4, 4],
                "mean(MATH_ISOLATE)": [10.0, 20.0],
            }
        ),
    )
    monkeypatch.setenv("COMMIT_SHA", "deadbeef")
    monkeypatch.setenv("RUN_ID", "r1")
    monkeypatch.setenv("PIPELINE", "nightly")

    out = tmp_path / "run.parquet"
    publish(str(tmp_path), str(out), "wormhole")

    df = pq.read_table(out).to_pandas()
    assert set(df["arch"]) == {"wormhole"}
    assert set(df["commit_sha"]) == {"deadbeef"}
    assert set(df["pipeline"]) == {"nightly"}
    assert set(df["test_name"]) == {"perf_a"}  # test_name derived from the filename


def test_publish_raises_on_empty_dir(tmp_path):
    with pytest.raises(SystemExit):  # allow-pytest.raises: no expect_error in LLK suite
        publish(str(tmp_path), str(tmp_path / "x.parquet"), "wormhole")
