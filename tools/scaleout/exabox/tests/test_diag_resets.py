#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the reset loop against a fake tt-smi that, like the real one on
a Galaxy, prints the CPLD banner before every `-r`."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from diag.galaxy import EXPECTED_CHIP_COUNT  # noqa: E402
from diag.model import PASS, Phase  # noqa: E402
from diag.resets import reset_loop  # noqa: E402

FAKE_TT_SMI = """\
#!/bin/sh
if [ "$1" = "-r" ]; then
    echo "CPLD FW v1.16 or higher is required to use tt-smi -r on Galaxy systems."
    exit "$FAKE_R_RC"
fi
"""


class TestResetLoop(unittest.TestCase):
    def run_fake(self, r_rc):
        with tempfile.TemporaryDirectory() as tmp:
            tt_smi = Path(tmp) / "tt-smi"
            tt_smi.write_text(FAKE_TT_SMI)
            tt_smi.chmod(0o755)
            phase = Phase(name="reset_loop")
            env = mock.patch.dict(os.environ, {"FAKE_R_RC": str(r_rc)})
            chips = mock.patch("diag.resets.count_pcie_chips", return_value=EXPECTED_CHIP_COUNT)
            with env, chips, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                reset_loop(str(tt_smi), ["-r"], phase, dry_run=False)
        phase.rollup()
        return phase

    def test_cpld_banner_is_not_a_warning(self):
        phase = self.run_fake(r_rc=0)
        self.assertEqual([(c.name, c.status) for c in phase.checks], [("reset_1_r", PASS)])
        self.assertEqual(phase.status, PASS)

    def test_failed_r_still_recovers_with_glx_reset(self):
        checks = {c.name: c.status for c in self.run_fake(r_rc=1).checks}
        self.assertEqual(checks.get("cpld_auto_recover"), PASS)


if __name__ == "__main__":
    unittest.main()
