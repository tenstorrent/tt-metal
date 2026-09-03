# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests for the backfill CLI (archived nights -> the warehouse).

Everything here runs offline: no chip, no GitHub, no SFTP server. The two
external legs are exercised through ``--from-dir`` and ``--dry-run``, which is
what those flags are for.

The tests that matter most are the two-era ones: the archive spans a change in
how CI packaged its artefacts, and the whole point of this tool is that both
eras come out looking like tonight's nightly.
"""

import csv
import os
import subprocess
import zipfile

import pandas as pd
import pyarrow.parquet as pq
from helpers.perf import backfill
from helpers.perf.backfill import (
    BackfillError,
    batchfile_lines,
    collect_shards,
    main,
    stage,
    verify_shard,
    write_manifest,
)
from helpers.perf.parquet import write_run_batch


def _write_parquet(path, *, run_id, arch="wormhole", commit_sha="deadbeef", ts=None):
    """Write one shard's Parquet the way a nightly job does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_run_batch(
        {
            "perf_a": pd.DataFrame(
                {
                    "marker": ["INIT", "KERNEL"],
                    "tile_cnt": [4, 4],
                    "mean(MATH_ISOLATE)": [10.0, 20.0],
                }
            )
        },
        str(path),
        commit_sha=commit_sha,
        arch=arch,
        run_id=run_id,
        timestamp=ts or "2026-09-01T03:10:00+00:00",
        pipeline="nightly",
    )
    return path


def _modern_shard(run_dir, run, arch, shard):
    """2026-08-28 on: perf-data-<arch>-<n>/<tag>/<tag>.parquet."""
    tag = f"{run}-{arch}-{shard}"
    return _write_parquet(
        run_dir / f"perf-data-{arch}-{shard}" / tag / f"{tag}.parquet",
        run_id=tag,
        arch=arch,
        ts=f"2026-09-01T0{3 + shard}:10:00+00:00",
    )


def _legacy_shard(run_dir, run, arch, shard):
    """Before 2026-08-28: an inner zip holding the same rows twice."""
    staging = run_dir / f"build-{arch}-{shard}"
    both = [
        _write_parquet(staging / "perf_data" / f"{run}.parquet", run_id=run, arch=arch),
        _write_parquet(
            staging / "perf_data" / f"run-{arch}-{shard}.parquet",
            run_id=run,
            arch=arch,
        ),
    ]
    archive = run_dir / f"perf-data-{arch}-{shard}" / f"perf_data-{arch}-{shard}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as zf:
        for p in both:
            zf.write(p, arcname=os.path.join("perf_data", p.name))
    for p in both:
        p.unlink()
    return archive


def test_collect_unzips_the_legacy_inner_archive(tmp_path):
    # Before 2026-08-28 the Parquets are invisible to a plain glob.
    run_dir = tmp_path / "runs" / "42"
    _legacy_shard(run_dir, "42", "wormhole", 0)

    shards = collect_shards(str(run_dir))

    assert [os.path.basename(p) for p in shards] == ["run-wormhole-0.parquet"]


def test_collect_resolves_the_legacy_double_write(tmp_path):
    # <run id>.parquet and run-<arch>-<n>.parquet hold identical rows. Loading
    # both would double every measurement in that shard.
    run_dir = tmp_path / "runs" / "42"
    _legacy_shard(run_dir, "42", "wormhole", 0)
    _legacy_shard(run_dir, "42", "wormhole", 1)

    shards = collect_shards(str(run_dir))

    assert sorted(os.path.basename(p) for p in shards) == [
        "run-wormhole-0.parquet",
        "run-wormhole-1.parquet",
    ]


def test_collect_refuses_an_ambiguous_shard_directory(tmp_path):
    # Two Parquets and no run-<arch>-<n> copy: guessing which is authoritative
    # is how a shard gets counted twice or dropped.
    run_dir = tmp_path / "runs" / "42"
    _write_parquet(run_dir / "d" / "a.parquet", run_id="42")
    _write_parquet(run_dir / "d" / "b.parquet", run_id="42")

    try:
        collect_shards(str(run_dir))
    except BackfillError as e:
        assert "no single" in str(e)
    else:
        raise AssertionError("expected a BackfillError")


def test_collect_takes_the_modern_layout_as_is(tmp_path):
    run_dir = tmp_path / "runs" / "42"
    _modern_shard(run_dir, "42", "wormhole", 0)

    assert [os.path.basename(p) for p in collect_shards(str(run_dir))] == [
        "42-wormhole-0.parquet"
    ]


def test_both_eras_stage_to_the_same_shape(tmp_path):
    # The point of the tool: a backfilled night is indistinguishable from a
    # live one, whichever way CI happened to package it.
    modern, legacy = tmp_path / "runs" / "42", tmp_path / "runs" / "43"
    for shard in range(2):
        _modern_shard(modern, "42", "wormhole", shard)
        _legacy_shard(legacy, "43", "wormhole", shard)

    staged, rejected = stage(str(tmp_path / "runs"), str(tmp_path / "out"))

    assert rejected == []
    assert [row["file"] for row in staged] == [
        "llk_perf_nightly-20260901-42-wormhole.parquet",
        "llk_perf_nightly-20260901-43-wormhole.parquet",
    ]
    assert {row["shards"] for row in staged} == {2}


def test_stage_merges_one_file_per_arch_per_run(tmp_path):
    run_dir = tmp_path / "runs" / "42"
    for arch in ("wormhole", "blackhole"):
        for shard in range(5):
            _modern_shard(run_dir, "42", arch, shard)

    staged, _ = stage(str(tmp_path / "runs"), str(tmp_path / "out"))

    assert [(r["arch"], r["shards"], r["rows"]) for r in staged] == [
        ("blackhole", 5, 10),
        ("wormhole", 5, 10),
    ]


def test_stage_drops_a_corrupt_shard_and_merges_the_rest(tmp_path):
    # One unreadable shard costs its own rows, not the night.
    run_dir = tmp_path / "runs" / "42"
    _modern_shard(run_dir, "42", "wormhole", 0)
    bad = run_dir / "perf-data-wormhole-1" / "42-wormhole-1" / "42-wormhole-1.parquet"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("not a parquet")

    staged, rejected = stage(str(tmp_path / "runs"), str(tmp_path / "out"))

    assert [row["shards"] for row in staged] == [1]
    assert len(rejected) == 1 and "cannot read Parquet" in rejected[0][1]


def test_verify_shard_rejects_non_constant_provenance(tmp_path):
    path = _write_parquet(tmp_path / "x" / "x.parquet", run_id="42")
    assert verify_shard(str(path))["run_id"] == "42"

    empty = tmp_path / "y" / "y.parquet"
    empty.parent.mkdir(parents=True)
    write_run_batch(
        {},
        str(empty),
        commit_sha="deadbeef",
        arch="wormhole",
        run_id="42",
        timestamp="2026-09-01T03:10:00+00:00",
        pipeline="nightly",
    )
    try:
        verify_shard(str(empty))
    except BackfillError as e:
        assert "no rows" in str(e)
    else:
        raise AssertionError("expected a BackfillError")


def test_manifest_records_what_was_sent(tmp_path):
    run_dir = tmp_path / "runs" / "42"
    _modern_shard(run_dir, "42", "wormhole", 0)
    staged, _ = stage(str(tmp_path / "runs"), str(tmp_path / "out"))

    with open(write_manifest(staged, str(tmp_path / "out"))) as fh:
        rows = list(csv.DictReader(fh))

    assert rows[0]["run_id"] == "nightly-20260901-42-wormhole"
    assert rows[0]["commit_sha"] == "deadbeef"
    assert rows[0]["pipeline"] == "nightly"
    assert rows[0]["rows"] == "2"


def test_merged_file_carries_one_run_id_and_one_timestamp(tmp_path):
    run_dir = tmp_path / "runs" / "42"
    for shard in range(3):
        _modern_shard(run_dir, "42", "wormhole", shard)

    staged, _ = stage(str(tmp_path / "runs"), str(tmp_path / "out"))
    table = pq.read_table(tmp_path / "out" / staged[0]["file"])

    assert set(table.column("run_id").to_pylist()) == {"nightly-20260901-42-wormhole"}
    assert len(set(table.column("timestamp").to_pylist())) == 1


def test_batchfile_puts_then_lists():
    lines = batchfile_lines(["a.parquet", "b.parquet"])
    assert lines == ["put a.parquet", "put b.parquet", "ls -hal"]
    assert batchfile_lines(["a.parquet"], remote_dir="inbox")[0] == "cd inbox"


def test_check_runs_sftp_and_needs_no_stage_dir(monkeypatch, capsys):
    seen = {}

    def fake_run(command, **kwargs):
        seen["batchfile"] = open(command[command.index("-b") + 1]).read()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(backfill.subprocess, "run", fake_run)

    code = main(["--check", "--host", "h", "--user", "u", "--key", "/dev/null"])

    assert code == 0
    assert "login OK" in capsys.readouterr().out
    assert seen["batchfile"].splitlines() == ["pwd", "ls -hal"]


def test_check_failure_names_the_user_convention_and_the_fingerprint(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        backfill.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 255),
    )
    monkeypatch.setattr(
        backfill, "key_fingerprint", lambda key: "SHA256:abc u (ED25519)"
    )

    assert main(["--check", "--host", "h", "--user", "u", "--key", "/dev/null"]) == 1
    err = capsys.readouterr().err
    assert "SHA256:abc" in err
    assert "llk-perf-run-writer" in err  # the mistake that cost us a day


def test_dry_run_stages_without_connecting(tmp_path, capsys):
    run_dir = tmp_path / "runs" / "42"
    _modern_shard(run_dir, "42", "wormhole", 0)

    code = main(
        [
            "--from-dir",
            str(tmp_path / "runs"),
            "--stage-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--host",
            "example.invalid",
            "--user",
            "llk-perf-run-writer",
            "--key",
            "/dev/null",
        ]
    )

    assert code == 0
    assert "would run:" in capsys.readouterr().out
    assert (tmp_path / "out" / "sftp_batchfile.txt").read_text().splitlines() == [
        "put llk_perf_nightly-20260901-42-wormhole.parquet",
        "ls -hal",
    ]


def test_upload_without_credentials_fails_before_the_network(tmp_path):
    run_dir = tmp_path / "runs" / "42"
    _modern_shard(run_dir, "42", "wormhole", 0)

    assert (
        main(
            [
                "--from-dir",
                str(tmp_path / "runs"),
                "--stage-dir",
                str(tmp_path / "o"),
                "--upload",
            ]
        )
        == 1
    )


def test_nothing_loadable_is_an_error(tmp_path):
    assert main(["--from-dir", str(tmp_path), "--stage-dir", str(tmp_path / "o")]) == 1


def test_stage_dir_is_required_without_check(tmp_path):
    assert main(["--from-dir", str(tmp_path)]) == 1
