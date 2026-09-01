# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests for the backfill CLI (nightly Parquets -> warehouse SFTP).

Everything here runs offline: no chip, no GitHub, no SFTP server. The two
external legs (``gh`` and ``sftp``) are exercised through ``--from-dir`` and
``--dry-run``, which are in the tool for exactly this reason.
"""

import csv
import os

import pandas as pd
from helpers.perf.backfill import (
    bare_run_id,
    batchfile_lines,
    collect_parquets,
    main,
    remote_name,
    stage,
    tag_of,
    write_manifest,
)
from helpers.perf.parquet import write_run_batch


def _write_run_parquet(path, *, run_id="42", arch="wormhole", commit_sha="deadbeef"):
    """Write a valid one-shard run Parquet, the way a nightly shard does."""
    frames = {
        "perf_a": pd.DataFrame(
            {
                "marker": ["INIT", "KERNEL"],
                "tile_cnt": [4, 4],
                "mean(MATH_ISOLATE)": [10.0, 20.0],
            }
        )
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_run_batch(
        frames,
        str(path),
        commit_sha=commit_sha,
        arch=arch,
        run_id=run_id,
        timestamp="2026-09-01T03:09:31+00:00",
        pipeline="nightly",
    )
    return path


def _shard(root, run_id, arch, shard):
    """The real artifact layout: runs/<tag>/<tag>.parquet."""
    tag = f"{run_id}-{arch}-{shard}"
    return _write_run_parquet(root / tag / f"{tag}.parquet", run_id=run_id, arch=arch)


def test_collect_parquets_finds_nested_and_ignores_csvs(tmp_path):
    _shard(tmp_path, "42", "wormhole", 0)
    # The artifact carries the per-test CSVs beside the Parquet; they are not ours.
    (tmp_path / "42-wormhole-0" / "perf_a.csv").write_text("marker\nINIT\n")

    found = collect_parquets(str(tmp_path))
    assert [os.path.basename(p) for p in found] == ["42-wormhole-0.parquet"]


def test_remote_name_is_flat_and_prefixed(tmp_path):
    path = _shard(tmp_path, "42", "blackhole", 3)
    provenance = {"run_id": "42", "arch": "blackhole"}
    assert tag_of(str(path)) == "42-blackhole-3"
    assert (
        remote_name("42-blackhole-3", provenance) == "llk_perf_42-blackhole-3.parquet"
    )
    assert remote_name("42-blackhole-3", provenance, "x_") == "x_42-blackhole-3.parquet"


def test_remote_name_keeps_re_run_attempts_apart():
    # The workflow builds the tag from github.run_id, which drops the attempt,
    # so attempt 2 of a shard arrives under attempt 1's tag. The rows know
    # better, and the object name follows the rows.
    assert bare_run_id("42-2") == "42"
    assert (
        remote_name("42-blackhole-3", {"run_id": "42-2", "arch": "blackhole"})
        == "llk_perf_42-2-blackhole-3.parquet"
    )


def test_stage_accepts_a_re_run_shard(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "stage"
    # Attempt 2: rows say 42-2, the file is still named after the bare run id.
    _write_run_parquet(runs / "42-wormhole-0" / "42-wormhole-0.parquet", run_id="42-2")

    staged, rejected = stage(collect_parquets(str(runs)), str(out))

    assert rejected == []
    assert [row["file"] for row in staged] == ["llk_perf_42-2-wormhole-0.parquet"]


def test_stage_copies_verified_files_and_records_provenance(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "stage"
    _shard(runs, "42", "wormhole", 0)
    _shard(runs, "42", "blackhole", 1)

    staged, rejected = stage(collect_parquets(str(runs)), str(out))

    assert rejected == []
    assert {row["arch"] for row in staged} == {"wormhole", "blackhole"}
    assert {row["run_id"] for row in staged} == {"42"}
    assert all(row["rows"] == 2 and row["bytes"] > 0 for row in staged)
    assert sorted(os.listdir(out)) == [
        "llk_perf_42-blackhole-1.parquet",
        "llk_perf_42-wormhole-0.parquet",
    ]


def test_stage_rejects_a_file_whose_name_disagrees_with_its_rows(tmp_path):
    # A renamed file would upload under a tag that points at the wrong night.
    runs, out = tmp_path / "runs", tmp_path / "stage"
    _write_run_parquet(runs / "99-wormhole-0.parquet", run_id="42", arch="wormhole")

    staged, rejected = stage(collect_parquets(str(runs)), str(out))

    assert staged == []
    assert len(rejected) == 1 and "disagrees with its rows" in rejected[0][1]


def test_stage_rejects_unreadable_and_keeps_the_good_shard(tmp_path):
    # One corrupt shard must not block the other nine.
    runs, out = tmp_path / "runs", tmp_path / "stage"
    _shard(runs, "42", "wormhole", 0)
    (runs / "42-wormhole-1").mkdir()
    (runs / "42-wormhole-1" / "42-wormhole-1.parquet").write_text("not a parquet")

    staged, rejected = stage(collect_parquets(str(runs)), str(out))

    assert [row["tag"] for row in staged] == ["42-wormhole-0"]
    assert len(rejected) == 1 and "cannot read Parquet" in rejected[0][1]


def test_stage_rejects_an_empty_batch(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "stage"
    tag = "42-wormhole-0"
    (runs / tag).mkdir(parents=True)
    write_run_batch(
        {},
        str(runs / tag / f"{tag}.parquet"),
        commit_sha="deadbeef",
        arch="wormhole",
        run_id="42",
        timestamp="2026-09-01T03:09:31+00:00",
        pipeline="nightly",
    )

    staged, rejected = stage(collect_parquets(str(runs)), str(out))

    assert staged == []
    assert "no rows" in rejected[0][1]


def test_manifest_lists_every_staged_file(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "stage"
    _shard(runs, "42", "wormhole", 0)
    staged, _ = stage(collect_parquets(str(runs)), str(out))

    with open(write_manifest(staged, str(out))) as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    assert rows[0]["file"] == "llk_perf_42-wormhole-0.parquet"
    assert rows[0]["commit_sha"] == "deadbeef"
    assert rows[0]["rows"] == "2"


def test_batchfile_puts_then_lists():
    lines = batchfile_lines(["a.parquet", "b.parquet"])
    # ls last, so a listing in the CI log proves both puts succeeded.
    assert lines == ["put a.parquet", "put b.parquet", "ls -hal"]
    assert batchfile_lines(["a.parquet"], remote_dir="inbox")[0] == "cd inbox"


def test_dry_run_stages_and_writes_a_batchfile_without_connecting(tmp_path, capsys):
    runs, out = tmp_path / "runs", tmp_path / "stage"
    _shard(runs, "42", "wormhole", 0)

    code = main(
        [
            "--from-dir",
            str(runs),
            "--stage-dir",
            str(out),
            "--dry-run",
            "--host",
            "example.invalid",
            "--user",
            "llk-perf-run",
            "--key",
            "/dev/null",
        ]
    )

    assert code == 0
    printed = capsys.readouterr().out
    assert "would run:" in printed and "sftp" in printed
    assert (out / "sftp_batchfile.txt").read_text().splitlines() == [
        "put llk_perf_42-wormhole-0.parquet",
        "ls -hal",
    ]


def test_upload_without_credentials_fails_before_touching_the_network(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "stage"
    _shard(runs, "42", "wormhole", 0)

    code = main(["--from-dir", str(runs), "--stage-dir", str(out), "--upload"])

    assert code == 1  # --host/--user/--key are missing


def test_no_parquets_is_an_error(tmp_path):
    assert main(["--from-dir", str(tmp_path), "--stage-dir", str(tmp_path / "s")]) == 1
