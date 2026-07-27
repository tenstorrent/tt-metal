#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3
from pathlib import Path

import pytest

from tracy.visualizer_run import (
    MANIFEST_FILENAME,
    RUN_ID_METADATA_KEY,
    TT_METAL_RUN_ID_ENV,
    find_memory_report_dir,
    get_or_create_run_id,
    inject_run_id_into_env,
    peek_run_id,
    stamp_memory_run_id,
    stamp_report_dir_run_id,
    _safe_manifest_path,
    _write_manifest_json,
    write_performance_manifest,
)


def _make_memory_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE report_metadata (
                key text UNIQUE,
                value text
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
    return path


def test_get_or_create_run_id_mints_and_reuses(monkeypatch):
    monkeypatch.delenv(TT_METAL_RUN_ID_ENV, raising=False)

    first = get_or_create_run_id()
    second = get_or_create_run_id()

    assert first
    assert first == second
    assert get_or_create_run_id() == first


def test_get_or_create_run_id_respects_existing_env(monkeypatch):
    monkeypatch.setenv(TT_METAL_RUN_ID_ENV, "fixed-run-id")
    assert get_or_create_run_id() == "fixed-run-id"


def test_peek_run_id_none_when_unset_or_blank(monkeypatch):
    monkeypatch.delenv(TT_METAL_RUN_ID_ENV, raising=False)
    assert peek_run_id() is None
    monkeypatch.setenv(TT_METAL_RUN_ID_ENV, "   ")
    assert peek_run_id() is None


def test_inject_run_id_into_env_updates_child_snapshot(monkeypatch):
    monkeypatch.delenv(TT_METAL_RUN_ID_ENV, raising=False)
    child_env = {"PATH": "/bin"}

    run_id = inject_run_id_into_env(child_env)

    assert run_id
    assert child_env[TT_METAL_RUN_ID_ENV] == run_id
    assert peek_run_id() == run_id


def test_inject_run_id_into_env_reuses_existing(monkeypatch):
    monkeypatch.setenv(TT_METAL_RUN_ID_ENV, "parent-run-id")
    child_env = {}

    run_id = inject_run_id_into_env(child_env)

    assert run_id == "parent-run-id"
    assert child_env[TT_METAL_RUN_ID_ENV] == "parent-run-id"


def test_write_performance_manifest_noop_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv(TT_METAL_RUN_ID_ENV, raising=False)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "ops_perf_results.csv").write_text("op\n", encoding="utf-8")

    assert write_performance_manifest(report_dir) is None
    assert not (report_dir / "manifest.json").exists()


def test_stamp_memory_run_id_missing_file_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv(TT_METAL_RUN_ID_ENV, "missing-db")
    assert stamp_memory_run_id(tmp_path / "nope.sqlite") is None


def test_stamp_memory_and_manifest_share_run_id(monkeypatch, tmp_path):
    monkeypatch.delenv(TT_METAL_RUN_ID_ENV, raising=False)

    db_path = _make_memory_db(tmp_path / "memory" / "db.sqlite")
    stamped = stamp_memory_run_id(db_path)
    assert stamped is not None

    report_dir = tmp_path / "perf"
    report_dir.mkdir()
    ops_csv = report_dir / "ops_perf_results.csv"
    ops_csv.write_text("op\n", encoding="utf-8")

    manifest_path = write_performance_manifest(report_dir, ops_csv=ops_csv)
    assert manifest_path is not None
    assert manifest_path.is_file()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload[RUN_ID_METADATA_KEY] == stamped
    assert payload["artifact"] == "performance"
    assert payload["report_dir"] == "."
    assert payload["ops_csv"] == "ops_perf_results.csv"
    assert not Path(payload["ops_csv"]).is_absolute()

    found = find_memory_report_dir(stamped, root=tmp_path / "memory")
    assert found == db_path.parent


def test_manifest_paths_are_report_relative(monkeypatch, tmp_path):
    monkeypatch.setenv(TT_METAL_RUN_ID_ENV, "rel-paths")
    report_dir = tmp_path / "perf"
    report_dir.mkdir()
    (report_dir / "ops_perf_results.csv").write_text("op\n", encoding="utf-8")
    (report_dir / "trace.tracy").write_bytes(b"tracy")
    (report_dir / "profile_log_device.csv").write_text("x\n", encoding="utf-8")

    manifest_path = write_performance_manifest(report_dir)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["report_dir"] == "."
    assert payload["ops_csv"] == "ops_perf_results.csv"
    assert payload["tracy_file"] == "trace.tracy"
    assert payload["device_log"] == "profile_log_device.csv"
    for key in ("report_dir", "ops_csv", "tracy_file", "device_log"):
        assert not Path(payload[key]).is_absolute()
        assert str(tmp_path) not in payload[key]


def test_stamp_report_dir_run_id_stamps_primary_rank(monkeypatch, tmp_path):
    monkeypatch.setenv(TT_METAL_RUN_ID_ENV, "fixture-run-id")
    report_path = tmp_path / "memory"
    db_path = _make_memory_db(report_path / "db.sqlite")

    stamped = stamp_report_dir_run_id(report_path, is_primary_rank=True)
    assert stamped == "fixture-run-id"

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT value FROM report_metadata WHERE key = ?",
            (RUN_ID_METADATA_KEY,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == "fixture-run-id"


def test_stamp_report_dir_run_id_skips_non_primary_rank(monkeypatch, tmp_path):
    monkeypatch.setenv(TT_METAL_RUN_ID_ENV, "fixture-run-id")
    report_path = tmp_path / "memory"
    db_path = _make_memory_db(report_path / "db.sqlite")

    assert stamp_report_dir_run_id(report_path, is_primary_rank=False) is None

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT value FROM report_metadata WHERE key = ?",
            (RUN_ID_METADATA_KEY,),
        ).fetchone()
    finally:
        conn.close()
    assert row is None


def test_stamp_report_dir_run_id_noop_without_db(monkeypatch, tmp_path):
    monkeypatch.setenv(TT_METAL_RUN_ID_ENV, "fixture-run-id")
    assert stamp_report_dir_run_id(tmp_path / "missing", is_primary_rank=True) is None


def test_write_manifest_json_writes_fixed_basename(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    written = _write_manifest_json({RUN_ID_METADATA_KEY: "x"}, report_dir=report_dir)

    assert written.name == MANIFEST_FILENAME
    assert written.parent == report_dir.resolve()
    assert written.is_file()
    assert json.loads(written.read_text(encoding="utf-8"))[RUN_ID_METADATA_KEY] == "x"


def test_safe_manifest_path_rejects_symlink_escape(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    outside = tmp_path / "elsewhere" / MANIFEST_FILENAME
    outside.parent.mkdir()
    (report_dir / MANIFEST_FILENAME).symlink_to(outside)

    with pytest.raises(ValueError, match="outside base directory"):
        _safe_manifest_path(report_dir)
    assert not outside.exists()
