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
    def capture(self, run, now=None) -> Phase:
        phase = Phase(name="captures", gates=False)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(captures, "run", run):
            captures.capture_events(phase, False, Path(d), now or datetime(2026, 11, 20, tzinfo=timezone.utc))
            self.logs = sorted(p.name for p in Path(d).iterdir())
        return phase

    def test_lines_of_note_warn_and_name_chips(self):
        phase = self.capture(fake_run({"ipmitool": (0, SEL), "journalctl": (0, KERNEL)}))
        sel, kernel = phase.checks
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
        phase = self.capture(fake_run({"ipmitool": (0, ""), "journalctl": (0, "")}))
        self.assertEqual({c.status for c in phase.checks}, {PASS})

    def test_a_read_that_cannot_run_skips(self):
        def raising(cmd, dry_run=False, **kw):
            raise FileNotFoundError(cmd[0])

        def no_journal(cmd, dry_run=False, **kw):
            err = "No journal files were found." if "journalctl" in cmd else ""
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=err)

        bad = {"ipmitool": (1, "Could not open device\nmore"), "journalctl": (1, "sudo: a password is required")}
        for name, run, want in (
            ("unreadable", fake_run(bad), [SKIP, SKIP]),
            ("no journal", no_journal, [PASS, SKIP]),
            ("missing tool", raising, [SKIP, SKIP]),
        ):
            with self.subTest(name):
                phase = self.capture(run)
                self.assertEqual([c.status for c in phase.checks], want)
                if name == "unreadable":
                    self.assertTrue(phase.checks[0].details.endswith("rc=1: Could not open device"))
                    self.assertIn("password is required", phase.checks[1].details)
                    self.assertEqual(self.logs, [])


if __name__ == "__main__":
    unittest.main()
