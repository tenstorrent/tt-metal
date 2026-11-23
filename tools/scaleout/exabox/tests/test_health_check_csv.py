#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""analyze_health_check_results.py on the checks the galaxy load and the reads
around the tests add: each lands in its category, a bus-bar step that ran counts
as executed, the details that name a chip keep their length, and the hottest
ASIC temperature reaches runs.csv."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))
sys.path.insert(0, str(SUITE_DIR / "test_infrastructure"))

import analyze_health_check_results as A  # noqa: E402
from diag import busbar, hsc  # noqa: E402
from diag.galaxy import expected_bdf  # noqa: E402
from diag.model import FAIL, PASS, Check, Phase  # noqa: E402
from diag.snapshot_checks import _thermal_and_telemetry, pcie_enum_check  # noqa: E402

BDFS = [expected_bdf(t, a) for t in range(1, 5) for a in range(1, 9)]
HOST = "bh-glx-110-c03u08"


def _step(name: str, body: str, test: str = "busbar") -> Check:
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        (Path(d) / "prog.py").write_text(body)
        with mock.patch.object(busbar, "GALAXY_DIR", Path(d)):
            return busbar.run_step(busbar.Step(name, "planned load", ("prog.py",), 5), Path(d), test=test)


def _analyze(*phases: Phase) -> tuple[dict, dict]:
    for p in phases:
        p.rollup()
    report = {"tier": "weekly", "host": HOST, "phases": {p.name: asdict(p) for p in phases}}
    meta = A.machine_meta(report, HOST, "1", "", A.now_iso())
    rows = A.checks_rows(report, meta)
    return {r["check_name"]: r for r in rows}, A.runs_row(report, meta, rows)


class TestTestsPhaseRows(unittest.TestCase):
    def setUp(self):
        tests = Phase(name="tests")
        tests.add(Check(name="eth_link_up", status=PASS, details="filter=f rc=0 dur=1s passed=1 failed=0 skipped=0"))
        tests.add(Check(name="gddr_full", status=FAIL, details="filter=f rc=0 dur=1s passed=0 failed=0 skipped=0"))
        tests.add(Check(name="deployment_binary_present", status=FAIL, details="none found"))
        tests.add(_step("qb", "print('ok')"))
        tests.add(_step("mix", f"print('NUMERIC FAIL {'x' * 150} 0000:41:00.0')"))
        tests.add(_step("fabric", "print('ok')", test="links"))
        tests.add(Check(name="links_pair", status=PASS, ip="eth", details="96 linked pairs, 5 rounds"))
        clean = {(t, a): (0, 0) for t in hsc.HSC_BUS for a in hsc.HSC_ADDRS}
        hsc.judge_hsc(clean, clean, tests)
        tests.add(pcie_enum_check([{"board_info": {"bus_id": b}} for b in BDFS[1:]], "pcie_enum_count_after_tests"))
        self.rows, self.run = _analyze(tests)

    def test_bus_bar_steps_that_ran_are_executed_stress_tests(self):
        for name, status in (("busbar_qb", PASS), ("busbar_mix", FAIL)):
            r = self.rows[name]
            self.assertEqual(
                (r["category"], r["status"], r["executed"], r["is_covered"]), ("stress_test", status, 1, 1)
            )

    def test_a_test_that_ran_no_cases_is_still_not_executed(self):
        for name in ("gddr_full", "deployment_binary_present"):
            self.assertEqual((self.rows[name]["executed"], self.rows[name]["is_covered"]), (0, 0), name)

    def test_the_checks_around_the_tests_keep_their_own_category(self):
        got = {n: self.rows[n]["category"] for n in ("hsc_input_status", "pcie_enum_count_after_tests")}
        self.assertEqual(got, {"hsc_input_status": "board", "pcie_enum_count_after_tests": "pcie"})
        self.assertEqual(self.rows["pcie_enum_count_after_tests"]["executed"], 1)

    def test_links_checks_are_eth(self):
        self.assertEqual({self.rows[n]["category"] for n in ("links_pair", "links_fabric")}, {"eth"})

    def test_a_failed_step_keeps_the_chip_it_names(self):
        self.assertIn("0000:41:00.0", self.rows["busbar_mix"]["details_short"])

    def test_a_chip_lost_under_the_load_is_a_pcie_failure(self):
        tests = Phase(name="tests")
        tests.add(Check(name="eth_link_up", status=PASS, details="filter=f rc=0 dur=1s passed=1 failed=0 skipped=0"))
        tests.add(pcie_enum_check([{"board_info": {"bus_id": b}} for b in BDFS[1:]], "pcie_enum_count_after_tests"))
        self.assertEqual(_analyze(tests)[1]["top_fail_category"], "pcie")


class TestMaxAsicTemp(unittest.TestCase):
    def test_from_the_thermal_check(self):
        snap = Phase(name="snapshot")
        devs = [{"board_info": {"bus_id": b}, "telemetry": {"asic_temperature": "50"}} for b in BDFS]
        devs[3]["telemetry"]["asic_temperature"] = "61.46"
        devs[4]["telemetry"]["asic_temperature"] = "65535"
        _thermal_and_telemetry(devs, snap)
        self.assertEqual(_analyze(snap)[1]["max_asic_temp_c"], 61.5)

    def test_from_a_report_written_before_the_thermal_check_changed(self):
        snap = Phase(name="snapshot")
        snap.add(
            Check(
                name="asic_thermal_precheck",
                status=PASS,
                ip="thermal",
                data={"hottest_chip": BDFS[0], "hottest_temp_c": 58.0, "all_temps_c": {BDFS[0]: 58.0}},
            )
        )
        self.assertEqual(_analyze(snap)[1]["max_asic_temp_c"], 58.0)

    def test_none_when_no_reading(self):
        self.assertEqual(_analyze(Phase(name="snapshot"))[1]["max_asic_temp_c"], "")


if __name__ == "__main__":
    unittest.main()
