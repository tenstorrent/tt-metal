# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared run_id helpers for pairing TTNN memory reports with Tracy perf reports."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

TT_METAL_RUN_ID_ENV = "TT_METAL_RUN_ID"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_TTNN_REPORTS_ROOT = Path("generated/ttnn/reports")


def get_or_create_run_id() -> str:
    """Return ``TT_METAL_RUN_ID``, minting and exporting one if unset."""
    existing = os.environ.get(TT_METAL_RUN_ID_ENV, "").strip()
    if existing:
        return existing
    run_id = str(uuid.uuid4())
    os.environ[TT_METAL_RUN_ID_ENV] = run_id
    return run_id


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_performance_manifest(
    report_dir: Path | str,
    *,
    ops_csv: Optional[Path | str] = None,
) -> Optional[Path]:
    """Write ``manifest.json`` beside a Tracy ops report. No-op if run_id unset."""
    run_id = os.environ.get(TT_METAL_RUN_ID_ENV, "").strip()
    if not run_id:
        return None

    report_dir = Path(report_dir)
    ops_csv_path = Path(ops_csv) if ops_csv else None
    if ops_csv_path is None:
        csv_candidates = sorted(report_dir.glob("ops_perf_results*.csv"))
        ops_csv_path = csv_candidates[0] if csv_candidates else None

    payload: dict[str, Any] = {
        "run_id": run_id,
        "artifact": "performance",
        "report_dir": str(report_dir.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if ops_csv_path is not None:
        payload["ops_csv"] = str(Path(ops_csv_path).resolve())

    tracy_files = sorted(report_dir.glob("*.tracy"))
    if tracy_files:
        payload["tracy_file"] = str(tracy_files[0].resolve())

    device_log = report_dir / "profile_log_device.csv"
    if device_log.is_file():
        payload["device_log"] = str(device_log.resolve())

    manifest_path = report_dir / MANIFEST_FILENAME
    write_json(manifest_path, payload)
    logger.info(f"Visualizer run_id={run_id} written to {manifest_path}")
    return manifest_path


def stamp_memory_run_id(db_path: Path | str) -> Optional[str]:
    """Insert ``run_id`` into ``report_metadata`` of a memory ``db.sqlite``."""
    db_path = Path(db_path)
    if not db_path.is_file():
        return None

    run_id = get_or_create_run_id()
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO report_metadata (key, value) VALUES (?, ?)",
            ("run_id", run_id),
        )
        conn.commit()
    finally:
        conn.close()

    logger.info(f"Visualizer run_id={run_id} written to {db_path} report_metadata")
    return run_id


def _read_db_run_id(db_path: Path) -> Optional[str]:
    try:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM report_metadata WHERE key = ?", ("run_id",))
            row = cursor.fetchone()
            return str(row[0]) if row and row[0] else None
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def find_memory_report_dir(
    run_id: str,
    root: Optional[Path | str] = None,
) -> Optional[Path]:
    """Find the newest TTNN report dir whose ``db.sqlite`` has ``run_id``."""
    root_path = Path(root) if root is not None else DEFAULT_TTNN_REPORTS_ROOT
    if not root_path.is_dir():
        return None

    matches: list[tuple[float, Path]] = []
    for db_path in root_path.glob("**/db.sqlite"):
        if _read_db_run_id(db_path) == run_id:
            matches.append((db_path.stat().st_mtime, db_path.parent))

    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]
