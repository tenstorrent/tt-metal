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
import subprocess

import pandas as pd
import pyarrow.parquet as pq
from helpers.perf import backfill
from helpers.perf.backfill import (
    bare_run_id,
    batchfile_lines,
    collect_parquets,
    main,
    remote_name,
    stage,
    tag_of,
    warehouse_run_id,
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
    assert tag_of(str(path)) == "42-blackhole-3"
    assert remote_name("42-blackhole-3") == "llk_perf_42-blackhole-3.parquet"
    assert remote_name("42-blackhole-3", "x_") == "x_42-blackhole-3.parquet"


def test_warehouse_run_id_recovers_the_attempt_without_guessing():
    # A shard index is a trailing number, so the attempt cannot be found by
    # looking for one. It is whatever is left after the known prefix.
    tag = "42-wormhole-0"
    assert warehouse_run_id(tag, {"run_id": "42"}) == tag  # old style, attempt 1
    assert warehouse_run_id(tag, {"run_id": "42-2"}) == "42-wormhole-0-2"  # old, re-run
    assert warehouse_run_id(tag, {"run_id": tag}) == tag  # what the producer writes now
    assert warehouse_run_id(tag, {"run_id": f"{tag}-2"}) == f"{tag}-2"  # ...its re-run
    assert bare_run_id("42-2") == "42"


def test_stage_rewrites_a_shared_run_id_to_the_run_tag(tmp_path):
    # Nights archived before the producer fix carry run_id = <workflow run id>,
    # one value shared by every shard. The warehouse replays by RUN_ID, so left
    # alone the second file to load would erase the first.
    runs, out = tmp_path / "runs", tmp_path / "stage"
    _shard(runs, "42", "wormhole", 0)
    _shard(runs, "42", "wormhole", 1)

    staged, rejected = stage(collect_parquets(str(runs)), str(out))

    assert rejected == []
    assert [row["run_id"] for row in staged] == ["42-wormhole-0", "42-wormhole-1"]
    assert {row["source_run_id"] for row in staged} == {"42"}
    # The rewrite must reach the rows, not only the file name.
    for row in staged:
        table = pq.read_table(out / row["file"])
        assert set(table.column("run_id").to_pylist()) == {row["run_id"]}


def test_stage_keeps_an_already_correct_run_id(tmp_path):
    # What the fixed producer writes. Nothing to rewrite.
    runs, out = tmp_path / "runs", tmp_path / "stage"
    _write_run_parquet(
        runs / "42-wormhole-0" / "42-wormhole-0.parquet", run_id="42-wormhole-0"
    )

    staged, _ = stage(collect_parquets(str(runs)), str(out))

    assert staged[0]["run_id"] == staged[0]["source_run_id"] == "42-wormhole-0"


def test_stage_keeps_a_re_run_attempt_in_the_run_id(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "stage"
    # Attempt 2: rows say 42-2, the file is still named after the bare run id.
    _write_run_parquet(runs / "42-wormhole-0" / "42-wormhole-0.parquet", run_id="42-2")

    staged, rejected = stage(collect_parquets(str(runs)), str(out))

    assert rejected == []
    assert staged[0]["run_id"] == "42-wormhole-0-2"
    assert staged[0]["file"] == "llk_perf_42-wormhole-0-2.parquet"


def test_stage_copies_verified_files_and_records_provenance(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "stage"
    _shard(runs, "42", "wormhole", 0)
    _shard(runs, "42", "blackhole", 1)

    staged, rejected = stage(collect_parquets(str(runs)), str(out))

    assert rejected == []
    assert {row["arch"] for row in staged} == {"wormhole", "blackhole"}
    assert {row["source_run_id"] for row in staged} == {"42"}
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
    assert rows[0]["source_run_id"] == "42"
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


def test_check_runs_sftp_and_needs_no_stage_dir(monkeypatch, capsys):
    # --check is the first thing anyone runs, so it must not require the rest
    # of the pipeline's arguments.
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["batchfile"] = open(command[command.index("-b") + 1]).read()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(backfill.subprocess, "run", fake_run)

    code = main(
        [
            "--check",
            "--host",
            "h",
            "--user",
            "u",
            "--key",
            "/dev/null",
            "--remote-dir",
            "inbox",
        ]
    )

    assert code == 0
    assert "login OK" in capsys.readouterr().out
    assert seen["command"][0] == "sftp"
    assert "BatchMode=yes" in seen["command"]
    assert seen["batchfile"].splitlines() == ["cd inbox", "pwd", "ls -hal"]


def test_check_failure_reports_the_key_fingerprint(monkeypatch, capsys):
    # The fingerprint is what the warehouse owner needs to answer "is the key
    # you installed the key I am sending?".
    monkeypatch.setattr(
        backfill.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 255),
    )
    monkeypatch.setattr(
        backfill, "key_fingerprint", lambda key: "SHA256:abc user (ED25519)"
    )

    code = main(["--check", "--host", "h", "--user", "u", "--key", "/dev/null"])

    assert code == 1
    assert "SHA256:abc" in capsys.readouterr().err


def test_check_without_credentials_is_an_error():
    assert main(["--check"]) == 1


def test_stage_dir_is_required_without_check(tmp_path):
    assert main(["--from-dir", str(tmp_path)]) == 1


def test_no_parquets_is_an_error(tmp_path):
    assert main(["--from-dir", str(tmp_path), "--stage-dir", str(tmp_path / "s")]) == 1
