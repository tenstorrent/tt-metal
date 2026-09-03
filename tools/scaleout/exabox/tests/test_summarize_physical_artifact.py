#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for physical artifact inference and pass_pct migration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

EXABOX_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXABOX_DIR))

from migrate_cluster_health_pass_pct import apply_changes, plan_migration  # noqa: E402
from summarize_physical_artifact import (  # noqa: E402
    parse_success_rate,
    summarize_physical_artifact,
)


def _write(path: Path, name: str, text: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    dest = path / name
    dest.write_text(text, encoding="utf-8")
    return dest


class TestSummarizePhysicalArtifact(unittest.TestCase):
    def test_parse_success_rate_strips_ansi(self):
        text = "Success Rate: \x1b[0;32m92.0%\x1b[0m\n"
        self.assertEqual(parse_success_rate(text), 92.0)

    def test_grade_iteration_logs(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            _write(
                tmp_path,
                "cluster_validation_iteration_1.log",
                "Detected Hosts: bh-glx-110-c01u02\nAll Detected Links are healthy\n",
            )
            _write(
                tmp_path,
                "cluster_validation_iteration_2.log",
                "Detected Hosts: bh-glx-110-c01u02\nFound Unhealthy Links\n",
            )
            summary = summarize_physical_artifact(tmp_path)
            self.assertEqual(summary.pass_pct, 50.0)
            self.assertNotEqual(summary.analyzer_code, 0)
            self.assertIn("bh-glx-110-c01u02", summary.hosts)

    def test_wrapper_fallback_when_no_iteration_logs(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            _write(
                tmp_path / "logs",
                "physical_validation-20260819T031200Z.log",
                "=== Physical Validation ===\nSuccess Rate: 72.0%\nAnalysis exit code: 1\n",
            )
            summary = summarize_physical_artifact(tmp_path)
            self.assertEqual(summary.pass_pct, 72.0)

    def test_empty_dir_has_no_rate(self):
        with tempfile.TemporaryDirectory() as raw:
            summary = summarize_physical_artifact(Path(raw))
            self.assertIsNone(summary.pass_pct)
            self.assertIsNone(summary.analyzer_code)


def _physical_record(artifact: str, record_uri: str) -> dict:
    return {
        "schema": "exabox.cluster_health.v1",
        "ts": "2026-08-24T17:41:18Z",
        "test_type": "physical",
        "status": "failed",
        "hosts": ["bh-glx-110-c01u02"],
        "analyzer_code": 1,
        "artifact_uri": artifact,
        "record_id": "abc123",
        "record_uri": record_uri,
    }


class TestMigratePassPct(unittest.TestCase):
    def test_migrate_adds_pass_pct(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            artifact = tmp_path / "run"
            _write(
                artifact,
                "cluster_validation_iteration_1.log",
                "Detected Hosts: bh-glx-110-c01u02\nAll Detected Links are healthy\n",
            )
            root = tmp_path / "hot"
            date = root / "2026-08-24"
            date.mkdir(parents=True)
            path = date / "rec.json"
            record = _physical_record(str(artifact), str(path))
            path.write_text(json.dumps(record), encoding="utf-8")

            changes, skipped, scanned = plan_migration([root])
            self.assertEqual(scanned, 1)
            self.assertEqual(skipped, [])
            self.assertEqual(len(changes), 1)
            self.assertEqual(changes[0].pass_pct, 100.0)

            backup = tmp_path / "backup"
            apply_changes(changes, backup)
            migrated = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(migrated["pass_pct"], 100.0)
            self.assertEqual(migrated["record_id"], "abc123")
            self.assertTrue((backup / "hot" / "2026-08-24" / "rec.json").is_file())

    def test_migrate_skips_when_rate_already_present(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            root = tmp_path / "hot"
            date = root / "2026-08-24"
            date.mkdir(parents=True)
            path = date / "rec.json"
            record = _physical_record("/missing", str(path))
            record["pass_pct"] = 88.0
            path.write_text(json.dumps(record), encoding="utf-8")
            changes, skipped, scanned = plan_migration([root])
            self.assertEqual(scanned, 1)
            self.assertEqual(changes, [])
            self.assertEqual(skipped, [])

    def test_migrate_skips_fabric(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            root = tmp_path / "hot"
            date = root / "2026-08-24"
            date.mkdir(parents=True)
            path = date / "rec.json"
            record = _physical_record("/missing", str(path))
            record["test_type"] = "fabric"
            path.write_text(json.dumps(record), encoding="utf-8")
            changes, skipped, scanned = plan_migration([root])
            self.assertEqual(scanned, 1)
            self.assertEqual(changes, [])
            self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
