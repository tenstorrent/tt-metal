# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests for publish_manual (hand-made runs -> the warehouse).

Needs pandas + pyarrow and git, no chip. Nothing here opens an SFTP session.
"""

import json
import subprocess

import pandas as pd
import pyarrow.parquet as pq
import pytest
from helpers.perf import publish_manual as pm


def _csv(dir_, name, df):
    dir_.mkdir(parents=True, exist_ok=True)
    df.to_csv(dir_ / f"{name}.csv", index=False)


def _quasar_rows():
    return pd.DataFrame(
        {
            "marker": ["INIT", "TILE_LOOP"],
            "face_c_dim": [16, 16],
            "tile_cnt": [8, 8],
            "mean(SFPU_ISOLATE)": [100.0, 800.0],
            "mean(L1_TO_L1[FPU])": [120.0, 900.0],
        }
    )


def test_manual_run_id_has_the_ci_shape():
    assert (
        pm.manual_run_id("2026-10-05T14:30:12+00:00", "quasar")
        == "manual-20261005-20261005T143012Z-quasar"
    )


def test_manual_run_id_converts_to_utc():
    assert (
        pm.manual_run_id("2026-10-05T01:30:12+02:00", "quasar")
        == "manual-20261004-20261004T233012Z-quasar"
    )


def test_manual_run_id_stamp_is_not_a_workflow_run_number():
    # The dashboard reads an all-digit component before the arch as a GitHub
    # workflow run and links to it. A manual run has none.
    stamp = pm.manual_run_id("2026-10-05T14:30:12Z", "quasar").split("-")[-2]
    assert not stamp.isdigit()


def test_run_start_reads_the_local_run_tag(tmp_path):
    run = tmp_path / "local-20261005T143012Z"
    run.mkdir()
    assert pm.run_start(run).isoformat() == "2026-10-05T14:30:12+00:00"


def test_run_start_follows_the_latest_symlink(tmp_path):
    run = tmp_path / "runs" / "local-20261005T143012Z"
    run.mkdir(parents=True)
    (tmp_path / "latest").symlink_to(run)
    assert pm.run_start(tmp_path / "latest").isoformat() == "2026-10-05T14:30:12+00:00"


def test_publish_writes_a_manual_quasar_file(tmp_path):
    run = tmp_path / "local-20261005T143012Z"
    _csv(run / "perf_pack_quasar", "perf_pack_quasar", _quasar_rows())

    out, n = pm.publish(run, tmp_path / "out", "quasar", commit_sha="abc123")

    assert n == 1
    assert out.name == "llk_perf_manual-20261005-20261005T143012Z-quasar.parquet"
    df = pq.read_table(out).to_pandas()
    assert set(df["pipeline"]) == {"manual"}
    assert set(df["run_id"]) == {"manual-20261005-20261005T143012Z-quasar"}
    assert set(df["arch"]) == {"quasar"}
    assert set(df["commit_sha"]) == {"abc123"}
    assert df["pr_number"].isna().all()
    # The Quasar-only metrics survive into the file the warehouse reads.
    assert "mean(SFPU_ISOLATE)" in df.columns
    assert "mean(L1_TO_L1[FPU])" in df.columns


def test_publish_twice_gives_the_same_run_id(tmp_path):
    # The loader replays by RUN_ID: a second publish must replace, not add.
    run = tmp_path / "local-20261005T143012Z"
    _csv(run / "perf_pack_quasar", "perf_pack_quasar", _quasar_rows())
    first, _ = pm.publish(run, tmp_path / "a", "quasar", commit_sha="abc123")
    second, _ = pm.publish(run, tmp_path / "b", "quasar", commit_sha="abc123")
    assert first.name == second.name


def test_publish_is_strict_about_unknown_columns(tmp_path):
    run = tmp_path / "local-20261005T143012Z"
    rows = _quasar_rows().assign(not_a_schema_column=[1, 2])
    _csv(run / "perf_pack_quasar", "perf_pack_quasar", rows)
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="not_a_schema_column"
    ):
        pm.publish(run, tmp_path / "out", "quasar", commit_sha="abc123")
    assert not list((tmp_path / "out").glob("*.parquet"))


def test_publish_rejects_an_empty_run(tmp_path):
    run = tmp_path / "local-20261005T143012Z"
    run.mkdir()
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="no CSVs"
    ):
        pm.publish(run, tmp_path / "out", "quasar", commit_sha="abc123")


def _archive_run(root, name, meta):
    run = root / name
    _csv(run / "perf_pack_quasar", "perf_pack_quasar", _quasar_rows())
    if meta is not None:
        (run / "run_meta.json").write_text(json.dumps(meta))
    return run


def test_plan_backfill_names_runs_from_their_timestamp(tmp_path):
    _archive_run(
        tmp_path,
        "batch_2026_08_14",
        {"timestamp": "2026-08-14T09:00:00Z", "commit_sha": "c1"},
    )
    (run,) = pm.plan_backfill(tmp_path, "quasar")
    assert run.run_id == "manual-20260814-20260814T090000Z-quasar"
    assert run.pipeline == "manual"
    assert run.arch == "quasar"  # neither the sidecar nor the name says
    assert run.commit_sha == "c1"


def test_plan_backfill_needs_a_timestamp(tmp_path):
    _archive_run(tmp_path, "batch_a", {"commit_sha": "c1"})
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="no timestamp"
    ):
        pm.plan_backfill(tmp_path, "quasar")


def test_plan_backfill_rejects_two_runs_with_one_timestamp(tmp_path):
    meta = {"timestamp": "2026-08-14T09:00:00Z"}
    _archive_run(tmp_path, "batch_a", meta)
    _archive_run(tmp_path, "batch_b", meta)
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="share a timestamp"
    ):
        pm.plan_backfill(tmp_path, "quasar")


def test_backfill_cli_writes_one_file_per_run(tmp_path):
    archive = tmp_path / "archive"
    _archive_run(
        archive, "batch_a", {"timestamp": "2026-08-14T09:00:00Z", "commit_sha": "c1"}
    )
    _archive_run(
        archive, "batch_b", {"timestamp": "2026-08-21T09:00:00Z", "commit_sha": "c2"}
    )

    rc = pm.main(
        ["backfill", "--archive", str(archive), "--out-dir", str(tmp_path / "out")]
    )

    assert rc == 0
    names = sorted(p.name for p in (tmp_path / "out").glob("*.parquet"))
    assert names == [
        "manual-20260814-20260814T090000Z-quasar.parquet",
        "manual-20260821-20260821T090000Z-quasar.parquet",
    ]


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "f").write_text("1")
    _git(repo, "add", "f")
    _git(repo, "commit", "-q", "-m", "c")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def test_checkout_state_clean_main_has_no_problems(tmp_path):
    repo = _repo(tmp_path)
    sha, problems = pm.checkout_state(cwd=repo)
    assert len(sha) == 40
    assert problems == []


def test_checkout_state_flags_uncommitted_changes(tmp_path):
    repo = _repo(tmp_path)
    (repo / "f").write_text("2")
    _, problems = pm.checkout_state(cwd=repo)
    assert any("uncommitted" in p for p in problems)


def test_checkout_state_flags_a_commit_main_does_not_have(tmp_path):
    repo = _repo(tmp_path)
    (repo / "f").write_text("2")
    _git(repo, "commit", "-q", "-am", "branch work")
    _, problems = pm.checkout_state(cwd=repo)
    assert any("origin/main" in p for p in problems)


def test_upload_needs_a_key(monkeypatch, tmp_path):
    monkeypatch.delenv(pm.KEY_ENV, raising=False)
    f = tmp_path / "x.parquet"
    f.write_bytes(b"PAR1")
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        SystemExit, match="--key"
    ):
        pm.main(["upload", str(f)])


def test_upload_dry_run_sends_nothing(tmp_path, capsys, monkeypatch):
    key = tmp_path / "key"
    key.write_text("k")
    f = tmp_path / "llk_perf_manual-20261005-20261005T143012Z-quasar.parquet"
    f.write_bytes(b"PAR1")
    monkeypatch.setattr(pm.subprocess, "run", pytest.fail)

    assert pm.main(["upload", str(f), "--key", str(key), "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert f"{pm.SFTP_USER}@{pm.SFTP_HOST}" in out
    assert f'put "{f}"' in out


def test_upload_refuses_a_file_that_is_not_parquet(tmp_path):
    f = tmp_path / "run.csv"
    f.write_text("marker\n")
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="not a Parquet file"
    ):
        pm.upload([f], tmp_path / "key", dry_run=True)
