# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recovery orchestration tests: no server, HTTP, processes or hardware."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_recover_performance as recovery


def fixture(root):
    config = {
        "model": "synthetic",
        "base_url": "http://localhost:1",
        "budget_seconds": 3600,
        "benchmark_command": ["benchmark"],
        "performance_server_command": ["server"],
        "roofline_command": ["collector"],
        "output_tokens": 128,
    }
    for name, value in [
        ("run_config.json", config),
        (
            "summary.json",
            {"status": "failed", "accuracy": {}, "performance": {}, "error": "Original incomplete accuracy"},
        ),
        ("perf-b32-server.json", {"max_num_seqs": 32}),
    ]:
        (root / name).write_text(json.dumps(value))
    (root / "REPORT.md").write_text("Original failed report")
    return config


class RecoveryTests(unittest.TestCase):
    def test_original_expired_deadline_rejects_without_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            with patch.object(recovery, "command") as command:
                with self.assertRaisesRegex(TimeoutError, "cannot restart clock"):
                    recovery.recover(root, time.monotonic() - 3601)
                command.assert_not_called()

    def test_nonfailed_original_run_cannot_be_recovered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            (root / "summary.json").write_text(json.dumps({"status": "running"}))
            with self.assertRaisesRegex(ValueError, "terminated"):
                recovery.recover(root, time.monotonic() - 100)

    def test_successful_measurements_stay_failed_and_keep_absolute_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            started = time.monotonic() - 100
            calls = []

            def command(argv, log, deadline):
                self.assertEqual(deadline, started + 3600)
                calls.append(Path(log).name)
                if argv[0] == "benchmark":
                    value = lambda flag: argv[argv.index(flag) + 1]
                    requests = int(value("--num-prompts"))
                    capacity = int(value("--max-concurrency"))
                    self.assertEqual(value("--random-input-len"), "4096")
                    self.assertEqual(value("--random-output-len"), "128")
                    raw = {
                        "model_id": "synthetic",
                        "max_concurrency": capacity,
                        "completed": requests,
                        "total_input_tokens": requests * 4096,
                        "total_output_tokens": requests * 128,
                    }
                    (root / value("--result-filename")).write_text(json.dumps(raw))
                elif argv[0] == "server":
                    (root / "perf-b1-server.json").write_text(json.dumps({"max_num_seqs": 1}))

            with (
                patch.object(recovery, "confirm_idle", return_value={"confirmed": True}),
                patch.object(recovery, "validate_server"),
                patch.object(recovery, "validate_performance"),
                patch.object(recovery, "load_roofline"),
                patch.object(recovery, "write_report"),
                patch.object(recovery, "command", side_effect=command),
            ):
                result = recovery.recover(root, started)
            self.assertEqual(
                calls,
                [
                    "perf-b32-warmup.log",
                    "perf-b32.log",
                    "roofline-b32-collect.log",
                    "perf-b1-server.log",
                    "roofline-b1-check.log",
                    "perf-b1-warmup.log",
                    "perf-b1.log",
                    "roofline-b1-collect.log",
                ],
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"], "Original incomplete accuracy")
            self.assertEqual(result["accuracy"], {})
            self.assertEqual(result["performance"]["32"]["completed"], 96)
            self.assertEqual(result["performance"]["1"]["completed"], 8)
            self.assertGreaterEqual(result["elapsed_seconds"], 100)
            self.assertTrue((root / "summary-before-performance-recovery.json").is_file())
            self.assertEqual((root / "REPORT-before-performance-recovery.md").read_text(), "Original failed report")

    def test_live_work_prevents_recovery_and_summary_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            original = (root / "summary.json").read_bytes()
            with (
                patch.object(recovery, "validate_server"),
                patch.object(recovery, "confirm_idle", side_effect=ValueError("server busy")),
                patch.object(recovery, "command") as command,
            ):
                with self.assertRaisesRegex(ValueError, "busy"):
                    recovery.recover(root, time.monotonic() - 100)
                command.assert_not_called()
            self.assertEqual((root / "summary.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
