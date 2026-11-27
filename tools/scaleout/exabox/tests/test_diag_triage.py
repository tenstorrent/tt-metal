#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for how the triage scripts' findings are recorded."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from diag.model import PASS, WARN  # noqa: E402
from diag.triage import normalize_triage_check  # noqa: E402

BRIDGE_AER = {
    "name": "hostside_bridge_aer",
    "status": "WARN",
    "details": "31/32 clean; bridge 0000:00:01.1: 27 nonfatal",
    "ip": "pcie",
    "data": {"bridges": 32, "ok": 31, "findings": ["bridge 0000:00:01.1: 27 nonfatal"]},
}


class TestNormalizeTriageCheck(unittest.TestCase):
    def test_bridge_aer_warn_is_recorded_as_pass(self):
        check = normalize_triage_check(BRIDGE_AER, gating=False)
        self.assertEqual((check.name, check.status), ("triage_hostside_bridge_aer", PASS))
        self.assertTrue(check.details.startswith("[advisory: WARN recorded as PASS] 31/32 clean"), check.details)

    def test_other_warns_stay_warn(self):
        check = normalize_triage_check({**BRIDGE_AER, "name": "hostside_pcie_link"}, gating=False)
        self.assertEqual(check.status, WARN)


if __name__ == "__main__":
    unittest.main()
