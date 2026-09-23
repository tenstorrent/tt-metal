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
from helpers.perf.publish_run import _run_csvs, main, publish


def _write_csv(path, df):
    df.to_csv(path, index=False)


def _set_provenance(monkeypatch):
    monkeypatch.setenv("COMMIT_SHA", "deadbeef")
    monkeypatch.setenv("RUN_ID", "r1")
    monkeypatch.setenv("PIPELINE", "nightly")


def test_run_csvs_excludes_post_and_counters(tmp_path):
    # Real runs nest one dir per test (perf_data/<base>/<base>.csv); mirror that
    # so the recursive glob is actually exercised (flat files pass even without it).
    sub = tmp_path / "perf_a"
    sub.mkdir()
    (sub / "perf_a.csv").write_text("marker\nINIT\n")
    (sub / "perf_a.post.csv").write_text("marker\nINIT\n")
    (sub / "perf_a.counters.csv").write_text("marker\nINIT\n")

    names = [p.rsplit("/", 1)[-1] for p in _run_csvs(str(tmp_path))]
    assert names == ["perf_a.csv"]  # the .post/.counters side files are excluded


def test_publish_stamps_provenance(tmp_path, monkeypatch):
    sub = tmp_path / "perf_a"
    sub.mkdir()
    _write_csv(
        sub / "perf_a.csv",
        pd.DataFrame(
            {
                "marker": ["INIT", "KERNEL"],
                "tile_cnt": [4, 4],
                "mean(MATH_ISOLATE)": [10.0, 20.0],
            }
        ),
    )
    _set_provenance(monkeypatch)

    out = tmp_path / "run.parquet"
    publish(str(tmp_path), str(out), "wormhole")

    df = pq.read_table(out).to_pandas()
    assert set(df["arch"]) == {"wormhole"}
    assert set(df["commit_sha"]) == {"deadbeef"}
    assert set(df["pipeline"]) == {"nightly"}
    assert set(df["test_name"]) == {"perf_a"}  # test_name derived from the filename


def test_publish_raises_on_empty_dir(tmp_path, monkeypatch):
    _set_provenance(monkeypatch)
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="no CSVs"
    ):
        publish(str(tmp_path), str(tmp_path / "x.parquet"), "wormhole")


def test_publish_rejects_empty_commit_sha(tmp_path, monkeypatch):
    sub = tmp_path / "perf_a"
    sub.mkdir()
    _write_csv(sub / "perf_a.csv", pd.DataFrame({"marker": ["INIT"], "tile_cnt": [4]}))
    _set_provenance(monkeypatch)
    monkeypatch.setenv("COMMIT_SHA", "")  # defined-but-empty — the Actions failure mode
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="COMMIT_SHA"
    ):
        publish(str(tmp_path), str(tmp_path / "x.parquet"), "wormhole")


def test_publish_rejects_unknown_pipeline(tmp_path, monkeypatch):
    sub = tmp_path / "perf_a"
    sub.mkdir()
    _write_csv(sub / "perf_a.csv", pd.DataFrame({"marker": ["INIT"], "tile_cnt": [4]}))
    _set_provenance(monkeypatch)
    monkeypatch.setenv("PIPELINE", "staging")  # not PR / nightly
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="PIPELINE"
    ):
        publish(str(tmp_path), str(tmp_path / "x.parquet"), "wormhole")


def test_publish_quasar_uses_quasar_schema(tmp_path, monkeypatch):
    sub = tmp_path / "perf_unpack_tilize_quasar"
    sub.mkdir()
    _write_csv(
        sub / "perf_unpack_tilize_quasar.csv",
        pd.DataFrame(
            {
                "marker": ["INIT"],
                "face_c_dim": [16],
                "implied_math_format": ["ImpliedMathFormat.Yes"],
                "unpacker_engine_sel": ["UnpackerEngine.UnpA"],
                "tile_cnt": [8],
            }
        ),
    )
    _set_provenance(monkeypatch)

    out = tmp_path / "run.parquet"
    diag = publish(str(tmp_path), str(out), "quasar")

    assert diag["unknown_columns"] == {}
    df = pq.read_table(out).to_pandas()
    assert set(df["arch"]) == {"quasar"}
    assert list(df["face_c_dim"].dropna()) == [16]
    assert "face_c_dim" in pq.read_table(out).schema.names


def test_main_rejects_bad_arch(tmp_path, monkeypatch):
    _set_provenance(monkeypatch)
    # argparse choices reject wormhole_b0 before publish() runs.
    with pytest.raises(SystemExit):  # allow-pytest.raises: no expect_error in LLK suite
        main(
            [
                "--csv-dir",
                str(tmp_path),
                "--out",
                str(tmp_path / "x.parquet"),
                "--arch",
                "wormhole_b0",
            ]
        )
