#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""verdict_lines: one line per fault with what, where and what to do, evidence
from the captures, and a final line of PASS, WARN, FAIL or INCOMPLETE."""

from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from diag.model import FAIL, PASS, SKIP, WARN, Check, Phase  # noqa: E402
from diag.verdict import verdict_lines  # noqa: E402

CHIP = "0000:46:00.0 (UBB2/U6)"


def report(*phases: Phase) -> dict:
    for p in phases:
        p.rollup()
    return {"phases": {p.name: asdict(p) for p in phases}}


def phase(name: str, *checks: Check, gates: bool = True, error: str = "") -> Phase:
    p = Phase(name=name, gates=gates, error=error)
    for c in checks:
        p.add(c)
    return p


class TestVerdict(unittest.TestCase):
    def test_pass(self):
        lines = verdict_lines(report(phase("snapshot", Check(name="pcie_enum_count", status=PASS, details="32/32"))))
        self.assertEqual(lines, ["Final verdict: PASS"])

    def test_missing_chip_is_one_fault_across_snapshots(self):
        lost = Check(name="pcie_enum_count", status=FAIL, details="31/32", ip="pcie", parts=[CHIP])
        lines = verdict_lines(
            report(
                phase("snapshot", lost),
                phase("snapshot_after_r1_glx_reset", lost),
                phase("tests", Check(name="dram", status=SKIP, details="skipped")),
            )
        )
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith(f"Fault 1: FAIL pcie_enum_count: 31/32 | {CHIP} | tt-smi -glx_reset"))
        self.assertEqual(lines[-1], "Final verdict: FAIL: 1 fault(s)")

    def test_reset_eth_and_dram_faults_say_what_to_do(self):
        lines = verdict_lines(
            report(
                phase("reset_loop", Check(name="reset_1_glx_reset", status=FAIL, details="rc=1", parts=[CHIP])),
                phase("eth_status", Check(name="eth_status_links_down", status=FAIL, details="2 down", ip="eth")),
                phase("snapshot", Check(name="dram_status", status=WARN, details="ch3", ip="gddr", parts=[CHIP])),
            )
        )
        self.assertIn("glx_reset and recount", lines[0])
        self.assertIn("| host |", lines[1])
        self.assertIn("reseat or replace the cable", lines[1])
        self.assertIn("re-run", lines[2])
        self.assertEqual(lines[-1], "Final verdict: FAIL: 2 fault(s), 1 warning(s)")

    def test_non_gating_phases_are_evidence_only(self):
        lines = verdict_lines(
            report(
                phase("snapshot", Check(name="pcie_enum_count", status=PASS, details="32/32")),
                phase("captures", Check(name="kernel_events", status=WARN, details="AER on 46:00.0"), gates=False),
                phase("sys_triage", Check(name="sys_triage", status=WARN, details="1 section"), gates=False),
            )
        )
        self.assertEqual(lines, ["Evidence: kernel_events: AER on 46:00.0", "Final verdict: PASS"])

    def test_errored_phase_is_incomplete_not_a_fault(self):
        crash = Check(name="eth_status_read", status=WARN, details="Boom()")
        lines = verdict_lines(report(phase("eth_status", crash, error="Boom()")))
        self.assertEqual(lines, ["Final verdict: INCOMPLETE (not a fault): eth_status did not finish"])

    def test_warn_only(self):
        lines = verdict_lines(
            report(phase("snapshot", Check(name="asic_thermal_precheck", status=WARN, details="hot", ip="thermal")))
        )
        self.assertIn("fans and airflow", lines[0])
        self.assertEqual(lines[-1], "Final verdict: WARN: 1 warning(s)")


if __name__ == "__main__":
    unittest.main()
