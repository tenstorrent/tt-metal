#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A run's report is its own, or there is none.

A rerun of run_diag_docker.sh into the directory of an earlier run used to fail
on its first write, since the earlier run's files are handed back to the caller
at exit, and crash on the report with the earlier PASS left standing. The
wrapper now refuses such a directory, and run_diag() removes a stale report
before it starts and fails cleanly when it cannot write its own.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

import diag_runner  # noqa: E402


@unittest.skipUnless(shutil.which("bash"), "the wrapper needs bash")
class TestDockerWrapperOutputDir(unittest.TestCase):
    def test_refuses_a_directory_that_holds_a_run(self):
        for leftover in ("diag_report.json", "console.log"):
            with self.subTest(leftover=leftover), tempfile.TemporaryDirectory() as d:
                (Path(d) / leftover).write_text("earlier run\n")
                cp = subprocess.run(
                    ["bash", str(SUITE_DIR / "run_diag_docker.sh"), "flash", "-o", d],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertNotEqual(cp.returncode, 0)
                self.assertIn("already holds a run", cp.stderr)
                # Refused before the transcript starts, so nothing was added to it.
                self.assertEqual([p.name for p in Path(d).iterdir()], [leftover])
                self.assertEqual((Path(d) / leftover).read_text(), "earlier run\n")


class TestRunDiagReport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.snapshot = self.tmp / "snapshot.json"
        self.snapshot.write_text("{}")
        self.output = self.tmp / "out" / "diag_report.json"

    def tearDown(self):
        self._tmp.cleanup()

    def run_diag(self) -> tuple[int, dict, str]:
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc, report = diag_runner.run_diag(
                "flash",
                dry_run=True,
                skip_eth_status=True,
                input_snapshot=self.snapshot,
                output=self.output,
                snapshot_out=self.tmp / "raw_snapshot.json",
            )
        return rc, report, err.getvalue()

    def test_a_stale_report_is_replaced(self):
        self.output.parent.mkdir(parents=True)
        self.output.write_text(json.dumps({"stale": True, "overall_status": "PASS"}))
        self.run_diag()
        written = json.loads(self.output.read_text())
        self.assertNotIn("stale", written)
        self.assertEqual(written["tier"], "flash")

    def test_an_output_that_cannot_be_replaced_stops_before_any_phase(self):
        self.output.mkdir(parents=True)  # a directory where the report should go
        rc, report, log = self.run_diag()
        self.assertEqual(rc, 1)
        self.assertEqual(report["phases"], {})
        self.assertEqual(report["overall_status"], "FAIL")
        self.assertIn("cannot replace", report["error"])
        self.assertIn("ERROR: cannot replace", log)

    def test_a_report_that_cannot_be_written_is_a_failed_run_not_a_traceback(self):
        # The rerun on the node: an earlier PASS in place, and no way to write over it.
        self.output.parent.mkdir(parents=True)
        self.output.write_text(json.dumps({"stale": True, "overall_status": "PASS"}))
        real_write_text = Path.write_text

        def write_text(path, *args, **kwargs):
            if path == self.output:
                raise PermissionError(13, "Permission denied", str(path))
            return real_write_text(path, *args, **kwargs)

        with mock.patch.object(Path, "write_text", write_text):
            rc, report, log = self.run_diag()
        self.assertEqual(rc, 1)
        self.assertIn("snapshot", report["phases"])
        self.assertIn("ERROR: could not write report", log)
        self.assertFalse(self.output.exists(), "the earlier run's report must not stand in for this one")

    def test_a_phase_the_tier_leaves_empty_is_skip_not_pass(self):
        # flash runs no resets, tests or triage tools: those phases ran nothing.
        _rc, report, _log = self.run_diag()
        for name in ("reset_loop", "tests", "triage", "eth_status"):
            with self.subTest(phase=name):
                self.assertEqual(report["phases"][name]["status"], "SKIP")

    def test_the_snapshot_defaults_to_beside_the_report(self):
        with mock.patch.object(diag_runner, "take_snapshot", return_value={}) as take:
            with mock.patch.object(diag_runner, "resolve_tt_smi", return_value="tt-smi"):
                with mock.patch.object(diag_runner, "detect_tt_smi_version", return_value=None):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        diag_runner.run_diag("flash", dry_run=True, skip_eth_status=True, output=self.output)
        self.assertEqual(take.call_args.args[1], self.output.parent / "diag_snapshot.json")


if __name__ == "__main__":
    unittest.main()
