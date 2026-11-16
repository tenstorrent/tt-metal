#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for physical artifact inference."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

EXABOX_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXABOX_DIR))

from summarize_physical_artifact import (  # noqa: E402
    as_pass_pct,
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

    def test_empty_iteration_counts_in_denominator(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            _write(
                tmp_path,
                "cluster_validation_iteration_1.log",
                "Detected Hosts: bh-glx-110-c01u02\nAll Detected Links are healthy\n",
            )
            _write(tmp_path, "cluster_validation_iteration_2.log", "")
            summary = summarize_physical_artifact(tmp_path)
            self.assertEqual(summary.pass_pct, 50.0)

    def test_wrapper_fallback_when_no_iteration_logs(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            _write(
                tmp_path / "logs",
                "physical_validation-20260819T031200Z.log",
                "=== Physical Validation ===\n"
                "HOSTS=bh-glx-110-c01u02,bh-glx-110-c01u08\n"
                "Success Rate: 72.0%\n"
                "Analysis exit code: 1\n",
            )
            summary = summarize_physical_artifact(tmp_path)
            self.assertEqual(summary.pass_pct, 72.0)
            self.assertEqual(summary.hosts, "bh-glx-110-c01u02,bh-glx-110-c01u08")
            self.assertEqual(summary.analyzer_code, 1)

    def test_wrapper_correlated_by_output_dir(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifact_a = root / "run_a"
            artifact_b = root / "run_b"
            artifact_a.mkdir()
            artifact_b.mkdir()
            logs = root / "logs"
            _write(
                logs,
                "physical_validation-20260819T031200Z.log",
                "=== Physical Validation ===\n"
                f"OUTPUT_DIR={artifact_a}\n"
                "HOSTS=host-a\n"
                "Success Rate: 10.0%\n"
                "Analysis exit code: 1\n",
            )
            _write(
                logs,
                "physical_validation-20260819T041200Z.log",
                "=== Physical Validation ===\n"
                f"OUTPUT_DIR={artifact_b}\n"
                "HOSTS=host-b\n"
                "Success Rate: 90.0%\n"
                "Analysis exit code: 0\n",
            )
            summary_a = summarize_physical_artifact(artifact_a)
            summary_b = summarize_physical_artifact(artifact_b)
            self.assertEqual(summary_a.pass_pct, 10.0)
            self.assertEqual(summary_a.hosts, "host-a")
            self.assertEqual(summary_a.analyzer_code, 1)
            self.assertEqual(summary_b.pass_pct, 90.0)
            self.assertEqual(summary_b.hosts, "host-b")
            self.assertEqual(summary_b.analyzer_code, 0)

    def test_ambiguous_wrappers_without_matching_output_dir_omit_rate(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            logs = tmp_path / "logs"
            _write(
                logs,
                "physical_validation-20260819T031200Z.log",
                "=== Physical Validation ===\nOUTPUT_DIR=/tmp/other-a\nSuccess Rate: 10.0%\n",
            )
            _write(
                logs,
                "physical_validation-20260819T041200Z.log",
                "=== Physical Validation ===\nOUTPUT_DIR=/tmp/other-b\nSuccess Rate: 90.0%\n",
            )
            summary = summarize_physical_artifact(tmp_path)
            self.assertIsNone(summary.pass_pct)
            self.assertIsNone(summary.analyzer_code)

    def test_wrapper_file_artifact_infers_hosts_and_code(self):
        with tempfile.TemporaryDirectory() as raw:
            wrapper = _write(
                Path(raw),
                "physical_validation-20260819T031200Z.log",
                "=== Physical Validation ===\n"
                "HOSTS=bh-glx-110-c01u02\n"
                "Success Rate: 55.0%\n"
                "Analysis exit code: 2\n",
            )
            summary = summarize_physical_artifact(wrapper)
            self.assertEqual(summary.pass_pct, 55.0)
            self.assertEqual(summary.hosts, "bh-glx-110-c01u02")
            self.assertEqual(summary.analyzer_code, 2)

    def test_empty_dir_has_no_rate(self):
        with tempfile.TemporaryDirectory() as raw:
            summary = summarize_physical_artifact(Path(raw))
            self.assertIsNone(summary.pass_pct)
            self.assertIsNone(summary.analyzer_code)

    def test_as_pass_pct_rejects_non_finite_and_out_of_range(self):
        self.assertIsNone(as_pass_pct(120))
        self.assertIsNone(as_pass_pct(-1))
        self.assertIsNone(as_pass_pct(float("nan")))
        self.assertIsNone(as_pass_pct(float("inf")))
        self.assertEqual(as_pass_pct(12.5), 12.5)
        self.assertTrue(math.isfinite(as_pass_pct(0.0)))


if __name__ == "__main__":
    unittest.main()
