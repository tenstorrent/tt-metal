#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for canonical cluster-health label migration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXABOX_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXABOX_DIR))

from migrate_cluster_health_labels import (  # noqa: E402
    MIGRATION_VERSION,
    apply_changes,
    load_snapshot,
    plan_migration,
)


def _host(short_id: str, quad: str, superpod: str, ring: str) -> dict:
    short = f"bh-glx-{short_id}"
    return {
        "short_id": short_id,
        "slurm_hostname": short,
        "bmc_hostname": short,
        "fqdn": f"{short}.exabox.tenstorrent.com",
        "quad": quad,
        "superpod": superpod,
        "ring": ring,
    }


def _write_snapshot(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "generated_at": "2026-08-24T18:00:00Z",
                "hosts": [
                    _host("110-a01u02", "110-A-Quad1", "SC36_5", "SC16"),
                    _host("110-a02u02", "110-A-Quad2", "SC36_5", "SC16"),
                    _host("120-a02u02", "120-A23", "SC16_1", "SC16"),
                ],
            }
        ),
        encoding="utf-8",
    )


def _record(hosts: list[str], labels: dict[str, str]) -> dict:
    return {
        "schema": "exabox.cluster_health.v1",
        "ts": "2026-08-24T17:41:18Z",
        "test_type": "physical",
        "status": "failed",
        "hosts": hosts,
        "analyzer_code": 50,
        "labels": labels,
        "orchestrator_id": "run-1",
    }


def _write_record(root: Path, name: str, record: dict) -> Path:
    date = root / "2026-08-24"
    date.mkdir(parents=True, exist_ok=True)
    path = date / name
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_plan_reclassifies_legacy_110a_group(tmp_path):
    snapshot_path = tmp_path / "snapshot.json"
    _write_snapshot(snapshot_path)
    root = tmp_path / "hot"
    path = _write_record(
        root,
        "legacy.json",
        _record(
            ["bh-glx-110-a01u02", "bh-glx-110-a02u02"],
            {"superpod": "SC16_3", "run_label": "SC16_3"},
        ),
    )
    index, snapshot = load_snapshot(snapshot_path)
    changes, unresolved, scanned = plan_migration([root], index, snapshot)
    assert scanned == 1
    assert unresolved == []
    assert len(changes) == 1
    assert changes[0].path == path
    assert changes[0].after == {"superpod": "SC36_5", "ring": "SC16"}
    assert changes[0].updated["labels"]["run_label"] == "SC16_3"
    assert changes[0].updated["orchestrator_id"] == "run-1"


def test_canonical_standalone_sc16_is_unchanged(tmp_path):
    snapshot_path = tmp_path / "snapshot.json"
    _write_snapshot(snapshot_path)
    root = tmp_path / "hot"
    _write_record(
        root,
        "canonical.json",
        _record(
            ["bh-glx-120-a02u02"],
            {"superpod": "SC16_1", "ring": "SC16", "quad": "120-A23"},
        ),
    )
    index, snapshot = load_snapshot(snapshot_path)
    changes, unresolved, scanned = plan_migration([root], index, snapshot)
    assert scanned == 1
    assert changes == []
    assert unresolved == []


def test_apply_backs_up_and_atomically_updates(tmp_path):
    snapshot_path = tmp_path / "snapshot.json"
    _write_snapshot(snapshot_path)
    root = tmp_path / "hot"
    path = _write_record(
        root,
        "legacy.json",
        _record(["bh-glx-110-a01u02"], {"superpod": "SC16_3", "quad": "110-A-Quad1"}),
    )
    original = path.read_text(encoding="utf-8")
    index, snapshot = load_snapshot(snapshot_path)
    changes, unresolved, _ = plan_migration([root], index, snapshot)
    assert unresolved == []
    backup_root = tmp_path / "backup"
    apply_changes(changes, backup_root)

    backup = backup_root / "hot" / "2026-08-24" / "legacy.json"
    assert backup.read_text(encoding="utf-8") == original
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert migrated["labels"]["superpod"] == "SC36_5"
    assert migrated["labels"]["ring"] == "SC16"
    assert migrated["labels"]["label_migration"] == MIGRATION_VERSION
    assert migrated["ts"] == "2026-08-24T17:41:18Z"


def test_unknown_host_is_unresolved(tmp_path):
    snapshot_path = tmp_path / "snapshot.json"
    _write_snapshot(snapshot_path)
    root = tmp_path / "hot"
    _write_record(root, "unknown.json", _record(["unknown-host"], {"superpod": "legacy"}))
    index, snapshot = load_snapshot(snapshot_path)
    changes, unresolved, _ = plan_migration([root], index, snapshot)
    assert changes == []
    assert len(unresolved) == 1
    assert "unknown hosts" in unresolved[0]["reason"]
