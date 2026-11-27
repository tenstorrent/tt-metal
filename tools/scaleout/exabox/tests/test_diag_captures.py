#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""capture_events with a faked ipmitool and journalctl: lines of note WARN and
name their chips, a read that cannot run SKIPs with one line, nothing votes."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from diag import captures  # noqa: E402
from diag.model import PASS, SKIP, WARN, Phase  # noqa: E402

SEL = """\
   0 | 03/02/2026 | 08:00:00 | Power Unit #0x03 | AC lost | Asserted
   1 | 11/20/2026 | 10:00:01 | Processor #0x01 | FRB3/Processor Startup Init failure | Asserted
   2 | 11/20/2026 | 10:00:02 | System Event #0x02 | Timestamp Clock Sync | Asserted
"""
KERNEL = """\
2026-11-20T10:01:00+0000 galaxy-host-01 kernel: pcieport 0000:40:01.1: AER: Corrected error received: 0000:46:00.0
2026-11-20T10:01:01 galaxy-host-01 kernel: pcieport 0000:c0:01.1: pciehp: Slot(3): Link Down
2026-11-20T10:01:02 galaxy-host-01 kernel: usb 1-1: new high-speed USB device
"""


def fake_run(outputs: dict[str, tuple[int, str]]):
    def run(cmd, dry_run=False, **kw):
        tool = next(c for c in cmd if c in outputs)
        rc, out = outputs[tool]
        return subprocess.CompletedProcess(cmd, rc, stdout=out if rc == 0 else "", stderr="" if rc == 0 else out)

    return run


class TestCaptureEvents(unittest.TestCase):
    def capture(self, outputs: dict[str, tuple[int, str]]) -> dict:
        phase = Phase(name="captures", gates=False)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(captures, "run", fake_run(outputs)):
            captures.capture_events(phase, False, Path(d), datetime(2026, 11, 20, tzinfo=timezone.utc))
            self.logs = sorted(p.name for p in Path(d).iterdir())
        return {c.name: c for c in phase.checks}

    def test_lines_of_note_warn_and_name_chips(self):
        checks = self.capture({"ipmitool": (0, SEL), "journalctl": (0, KERNEL)})
        sel, kernel = checks["bmc_sel_events"], checks["kernel_events"]
        self.assertEqual(sel.status, WARN)
        # The AC loss in March is from before the run.
        self.assertEqual(len(sel.data["lines"]), 1)
        self.assertIn("FRB3", sel.details)
        self.assertEqual(kernel.status, WARN)
        self.assertEqual(len(kernel.data["lines"]), 2)
        # The chip is named; the root ports are not chips.
        self.assertEqual(kernel.parts, ["0000:46:00.0 (UBB2/U6)"])
        self.assertEqual(self.logs, ["kernel_events.log", "sel.log"])

    def test_quiet_logs_pass(self):
        checks = self.capture({"ipmitool": (0, ""), "journalctl": (0, "")})
        self.assertEqual({c.status for c in checks.values()}, {PASS})

    def test_unreadable_skips_with_one_line(self):
        checks = self.capture(
            {"ipmitool": (1, "Could not open device\nmore"), "journalctl": (1, "sudo: a password is required")}
        )
        self.assertEqual(checks["bmc_sel_events"].status, SKIP)
        self.assertTrue(checks["bmc_sel_events"].details.endswith("rc=1: Could not open device"))
        self.assertIn("password is required", checks["kernel_events"].details)
        self.assertEqual(self.logs, [])

    def test_no_journal_skips(self):
        def run(cmd, dry_run=False, **kw):
            err = "No journal files were found." if "journalctl" in cmd else ""
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=err)

        phase = Phase(name="captures", gates=False)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(captures, "run", run):
            captures.capture_events(phase, False, Path(d), datetime.now(timezone.utc))
        self.assertEqual([c.status for c in phase.checks], [PASS, SKIP])

    def test_missing_tool_skips(self):
        def run(cmd, dry_run=False, **kw):
            raise FileNotFoundError(cmd[0])

        phase = Phase(name="captures", gates=False)
        with mock.patch.object(captures, "run", run):
            captures.capture_events(phase, False, Path("unused"), datetime.now(timezone.utc))
        self.assertEqual([c.status for c in phase.checks], [SKIP, SKIP])


if __name__ == "__main__":
    unittest.main()
