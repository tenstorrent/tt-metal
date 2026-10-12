#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for summarize_recover_artifact.py and the recover missing-cable and error labels."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
EXABOX_DIR = TESTS_DIR.parent
for _path in (EXABOX_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from report_cluster_health import main  # noqa: E402
from summarize_recover_artifact import (  # noqa: E402
    MAX_LISTED,
    missing_cable_labels,
    missing_cables,
    recover_error_labels,
    recover_errors,
)

START = "[bh-glx-110-c10u02] 2026-10-09 17:51:00.000 | info | Distributed | Running Physical Discovery (cluster_validation_utils.cpp:88)\n"
MISSING_ONE = (
    "[bh-glx-110-c10u02][17:51:02] Physical Discovery found 1 missing port/cable connections:\n"
    "[bh-glx-110-c10u02][17:51:02]   - PhysicalPortEndpoint{hostname='bh-glx-110-c10u08', aisle='C', rack=10, "
    "shelf_u=8, tray_id=3, port_type=QSFP_DD, port_id=9} <-> PhysicalPortEndpoint{hostname='bh-glx-110-c10u08', "
    "aisle='C', rack=10, shelf_u=8, tray_id=4, port_type=QSFP_DD, port_id=9}\n"
    "[bh-glx-110-c10u02][17:51:02] Physical Discovery found 1 missing channel connections:\n"
    "[bh-glx-110-c10u02][17:51:02]   - PhysicalChannelEndpoint{hostname='bh-glx-110-c10u08', tray_id=3, "
    "asic_channel=AsicChannel{asic_location=1, channel_id=9}} <-> PhysicalChannelEndpoint{hostname='bh-glx-110-c10u08', "
    "tray_id=4, asic_channel=AsicChannel{asic_location=1, channel_id=9}}\n"
)
CLEAN = "[bh-glx-110-c10u02] 2026-10-09 17:55:00.000 | info | Distributed | All Detected Links are healthy.\n"
CABLE = "bh-glx-110-c10u08 T3 QSFP_DD/9 <-> bh-glx-110-c10u08 T4 QSFP_DD/9"
P = "[bh-glx-110-d03u02][00:22:44] "
ATTEMPT_1 = P + "Recovery attempt 1 of 2\n"
ATTEMPT_2 = P + "Recovery attempt 2 of 2\n"
PORT_DOWN = P + "WARNING: cross-host port down FAILED (exit code 1); running cleanup reset then retrying port down.\n"
POST_RESET = P + "Error: POST_RESET failed for device 3.\n"
RESET_FAILED = P + "Reset failed | Exit code: 1\n"
UMD = (
    P + "terminate called after throwing an instance of 'tt::umd::error::UmdException<tt::umd::error::RuntimeError>'\n"
)
ABORT = P + "MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD\n"
FAILED_1 = P + "Recovery attempt 1 of 2 failed (exit code 1).\n"
HOSTS = "bh-glx-110-c10u02,bh-glx-110-c10u08,bh-glx-110-c10u14,bh-glx-110-c10u20"


def _write(directory: str, name: str, text: str, mtime: float | None = None) -> None:
    path = Path(directory) / name
    path.write_text(text)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


class TestMissingCables(unittest.TestCase):
    def test_lists_ports_from_last_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "recover_1.log", START + MISSING_ONE)
            self.assertEqual(missing_cables(tmp), [CABLE])

    def test_clean_retry_after_missing_reports_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "recover_1.log", START + MISSING_ONE + START + CLEAN)
            self.assertEqual(missing_cables(tmp), [])
            self.assertEqual(missing_cable_labels(tmp), {"missing_cables": "0"})

    def test_newest_log_with_discovery_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            _write(tmp, "b_old.log", START + CLEAN, now - 60)
            _write(tmp, "a_new.log", START + MISSING_ONE, now)
            self.assertEqual(missing_cables(tmp), [CABLE])

    def test_no_discovery_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "recover_1.log", "Recovery attempt 1 of 1 failed (exit code 1).\n")
            self.assertIsNone(missing_cables(tmp))
            self.assertEqual(missing_cable_labels(tmp), {})
        self.assertIsNone(missing_cables(None))
        self.assertIsNone(missing_cables("/nonexistent/recover-artifact"))

    @unittest.skipIf(os.geteuid() == 0, "root reads everything")
    def test_unreadable_artifact_dir_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            locked = Path(tmp) / "locked"
            locked.mkdir()
            _write(str(locked), "recover_1.log", START + MISSING_ONE)
            locked.chmod(0)
            try:
                self.assertIsNone(missing_cables(str(locked / "run")))
                self.assertIsNone(missing_cables(str(locked)))
            finally:
                locked.chmod(0o700)

    def test_long_list_is_capped(self):
        lines = []
        for port in range(MAX_LISTED + 3):
            lines.append(
                f"[h][00:00:00]   - PhysicalPortEndpoint{{hostname='h1', aisle='A', rack=1, shelf_u=2, tray_id=1, "
                f"port_type=QSFP_DD, port_id={port}}} <-> PhysicalPortEndpoint{{hostname='h2', aisle='A', rack=1, "
                f"shelf_u=8, tray_id=1, port_type=QSFP_DD, port_id={port}}}\n"
            )
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "recover_1.log", START + "".join(lines))
            labels = missing_cable_labels(tmp)
            self.assertEqual(labels["missing_cables"], str(MAX_LISTED + 3))
            listed = labels["missing_cables_list"].split("\n")
            self.assertEqual(len(listed), MAX_LISTED + 1)
            self.assertEqual(listed[-1], "... 3 more")


class TestRecoverErrors(unittest.TestCase):
    def _errors(self, text: str):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "recover_1.log", text)
            return recover_errors(tmp), recover_error_labels(tmp)

    def test_run_ends_on_first_error_of_its_last_failing_attempt(self):
        found, labels = self._errors(ATTEMPT_1 + PORT_DOWN + ATTEMPT_2 + UMD + ABORT + FAILED_1)
        self.assertEqual(found, (["port_down_failed", "chip_init_failed", "mpi_abort"], "chip_init_failed"))
        self.assertEqual(
            labels, {"errors": "port_down_failed,chip_init_failed,mpi_abort", "final_error": "chip_init_failed"}
        )

    def test_retried_port_down_does_not_hide_the_cause(self):
        found, _ = self._errors(ATTEMPT_1 + PORT_DOWN + POST_RESET + RESET_FAILED + FAILED_1)
        self.assertEqual(found[1], "post_reset_failed")
        found, _ = self._errors(ATTEMPT_1 + PORT_DOWN + FAILED_1)
        self.assertEqual(found[1], "port_down_failed")

    def test_specific_cause_wins_over_generic_reset_failure(self):
        found, _ = self._errors(ATTEMPT_1 + POST_RESET + RESET_FAILED + FAILED_1)
        self.assertEqual(found, (["post_reset_failed", "reset_failed"], "post_reset_failed"))

    def test_recovered_run_keeps_the_error_it_recovered_from(self):
        found, _ = self._errors(ATTEMPT_1 + UMD + FAILED_1 + ATTEMPT_2 + CLEAN)
        self.assertEqual(found, (["chip_init_failed"], "chip_init_failed"))

    def test_analyzer_categories_are_reused(self):
        found, _ = self._errors(ATTEMPT_1 + START + MISSING_ONE + FAILED_1)
        self.assertEqual(found, (["missing_connections"], "missing_connections"))

    def test_clean_run_has_no_error_labels(self):
        found, labels = self._errors(ATTEMPT_1 + START + CLEAN)
        self.assertEqual(found, ([], None))
        self.assertEqual(labels, {})

    def test_no_log_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(recover_errors(tmp))
            self.assertEqual(recover_error_labels(tmp), {})


class TestRecoverRecordLabels(unittest.TestCase):
    def _record(self, artifact_dir: str, *extra: str) -> dict:
        argv = [
            "--test-type",
            "recover",
            "--hosts",
            HOSTS,
            "--artifact-dir",
            artifact_dir,
            "--ts",
            "2026-10-09T17:52:00Z",
            "--dry-run",
            *extra,
        ]
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(main(argv), 0, err.getvalue())
        return json.loads(out.getvalue())

    def test_recover_record_carries_missing_cables(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "recover_1.log", START + MISSING_ONE)
            labels = self._record(tmp)["labels"]
            self.assertEqual(labels["missing_cables"], "1")
            self.assertEqual(labels["missing_cables_list"], CABLE)
            self.assertEqual(labels["errors"], "missing_connections")
            self.assertEqual(labels["final_error"], "missing_connections")

    def test_explicit_label_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "recover_1.log", START + MISSING_ONE)
            labels = self._record(tmp, "--label", "missing_cables=7")["labels"]
            self.assertEqual(labels["missing_cables"], "7")

    def test_no_discovery_adds_no_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "recover_1.log", "Recovery attempt 1 of 1 failed (exit code 1).\n")
            self.assertNotIn("missing_cables", self._record(tmp).get("labels", {}))


if __name__ == "__main__":
    unittest.main()
