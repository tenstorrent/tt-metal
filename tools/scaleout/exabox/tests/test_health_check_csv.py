#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""analyze_health_check_results.py rows: a test that ran no cases is not
executed, and the hottest ASIC temperature reaches runs.csv."""

from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))
sys.path.insert(0, str(SUITE_DIR / "test_infrastructure"))

import analyze_health_check_results as A  # noqa: E402
from diag.galaxy import expected_bdf  # noqa: E402
from diag.model import FAIL, PASS, Check, Phase  # noqa: E402

BDFS = [expected_bdf(t, a) for t in range(1, 5) for a in range(1, 9)]
HOST = "galaxy-host-01"


def _analyze(*phases: Phase) -> tuple[dict, dict]:
    for p in phases:
        p.rollup()
    report = {"tier": "weekly", "host": HOST, "phases": {p.name: asdict(p) for p in phases}}
    meta = A.machine_meta(report, HOST, "1", "", A.now_iso())
    rows = A.checks_rows(report, meta)
    return {r["check_name"]: r for r in rows}, A.runs_row(report, meta, rows)


class TestTestsPhaseRows(unittest.TestCase):
    def test_a_test_that_ran_no_cases_is_still_not_executed(self):
        tests = Phase(name="tests")
        tests.add(Check(name="eth_link_up", status=PASS, details="filter=f rc=0 dur=1s passed=1 failed=0 skipped=0"))
        tests.add(Check(name="gddr_full", status=FAIL, details="filter=f rc=0 dur=1s passed=0 failed=0 skipped=0"))
        tests.add(Check(name="deployment_binary_present", status=FAIL, details="none found"))
        rows, _ = _analyze(tests)
        for name in ("gddr_full", "deployment_binary_present"):
            self.assertEqual((rows[name]["executed"], rows[name]["is_covered"]), (0, 0), name)


class TestMaxAsicTemp(unittest.TestCase):
    def test_from_the_thermal_check(self):
        snap = Phase(name="snapshot")
        snap.add(
            Check(
                name="asic_thermal_precheck",
                status=PASS,
                ip="thermal",
                data={"hottest_chip": BDFS[0], "hottest_temp_c": 58.04, "all_temps_c": {BDFS[0]: 58.04}},
            )
        )
        self.assertEqual(_analyze(snap)[1]["max_asic_temp_c"], 58.0)

    def test_none_when_no_reading(self):
        self.assertEqual(_analyze(Phase(name="snapshot"))[1]["max_asic_temp_c"], "")


if __name__ == "__main__":
    unittest.main()
