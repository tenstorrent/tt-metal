#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3
from pathlib import Path

from tracy.visualizer_run import (
    TT_METAL_RUN_ID_ENV,
    find_memory_report_dir,
    get_or_create_run_id,
    stamp_memory_run_id,
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


def test_write_performance_manifest_noop_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv(TT_METAL_RUN_ID_ENV, raising=False)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "ops_perf_results.csv").write_text("op\n", encoding="utf-8")

    assert write_performance_manifest(report_dir) is None
    assert not (report_dir / "manifest.json").exists()


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
    assert payload["run_id"] == stamped
    assert payload["artifact"] == "performance"
    assert payload["ops_csv"] == str(ops_csv.resolve())

    found = find_memory_report_dir(stamped, root=tmp_path / "memory")
    assert found == db_path.parent
