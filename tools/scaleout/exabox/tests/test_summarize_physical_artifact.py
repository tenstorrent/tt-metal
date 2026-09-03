#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for physical artifact inference."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

EXABOX_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXABOX_DIR))

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


if __name__ == "__main__":
    unittest.main()
