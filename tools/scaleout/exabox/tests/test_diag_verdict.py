#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""verdict_lines: per fault what, where, its cause and repair, evidence from the
captures, the final verdict and the repair recommendation. Every row's checks
are emitted by some stage."""

from __future__ import annotations

import ast
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from diag.model import FAIL, PASS, SKIP, WARN, Check, Phase  # noqa: E402
from diag.triage import name_root_ports  # noqa: E402
from diag.verdict import ROWS, verdict_lines  # noqa: E402

CHIP = "0000:46:00.0 (UBB2/U6)"
TRAY2 = [f"0000:4{u}:00.0 (UBB2/U{u})" for u in range(1, 9)]


def report(*phases: Phase) -> dict:
    for p in phases:
        p.rollup()
    return {"phases": {p.name: asdict(p) for p in phases}}


def phase(name: str, *checks: Check, gates: bool = True, error: str = "") -> Phase:
    p = Phase(name=name, gates=gates, error=error)
    for c in checks:
        p.add(c)
    return p


def check(name: str, status: str = FAIL, details: str = "bad", **kw) -> Check:
    return Check(name=name, status=status, details=details, **kw)


# Row id -> (a check that reports it, text its repair must hold).
CASES = {
    "F01": (check("tray_power", cause="POWER", parts=TRAY2), "replace UBB2's bus bar set"),
    "F02": (check("tray_power", WARN, cause="POWER", parts=TRAY2), "watch UBB2's 12 V feed"),
    "F04": (check("pcie_enum_count", cause="POWER", parts=[CHIP]), "fix UBB2's 12 V feed"),
    "F05": (check("pcie_enum_count", parts=TRAY2), "reseat UBB2, then replace it"),
    "F06": (check("reset_1_glx_reset", parts=[CHIP]), "lost on reset again, replace UBB2's board"),
    "F07": (check("pcie_enum_count", cause="POWER-DOWNSTREAM", parts=[CHIP]), "power-good is low"),
    "F08": (check("pcie_enum_count", parts=[CHIP]), "swap UBB2's host PCIe cable"),
    "F09": (check("pcie_aer_diff", WARN, parts=[CHIP]), "if the errors follow the tray"),
    "F10": (check("pcie_gen", WARN, "Gen4", parts=[CHIP]), "to see if the fault follows"),
    "F13": (check("eth_status_links_down", details=f"{CHIP} ch4, TRACE port 2 down"), "a trace port stays"),
    "F14": (check("eth_status_links_down", details=f"{CHIP} ch8, QSFP-DD port 5 down"), "QSFP-DD cable on those"),
    "F19": (check("didt_matmul_galaxy_after_reset", parts=[CHIP]), "vdd_min to 760 mV"),
    "F21": (check("gddr_act_c3", parts=[CHIP]), "replace UBB2's board"),
    "F27": (check("asic_thermal_precheck", WARN, "hot", parts=[CHIP]), "UBB2's fans and airflow"),
}


def _literals(path: Path) -> set[str]:
    """Every string literal in a module, f-string pieces included."""
    return {
        n.value for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Constant) and type(n.value) is str
    }


class TestRows(unittest.TestCase):
    def test_every_row_has_a_case(self):
        self.assertEqual(set(CASES), set(ROWS))

    def test_each_row_prints_its_fault_cause_and_repair(self):
        for rid, (c, repair) in CASES.items():
            with self.subTest(rid):
                lines = verdict_lines(report(phase("p", c)))
                self.assertTrue(lines[0].startswith(f"Fault 1: {c.status} {rid} {ROWS[rid].fault}: "), lines[0])
                cause = c.cause or ROWS[rid].cause
                self.assertEqual(lines[1:-3], [f"  Cause: {cause}"] * bool(cause))
                self.assertIn(repair, lines[-3])

    def test_every_row_check_has_an_emitter(self):
        """A row may only name a check some stage emits: an exact name needs that
        literal, a prefix "x*" a literal starting with x (f-strings count)."""
        mods = [*sorted((SUITE_DIR / "diag").glob("*.py")), SUITE_DIR / "diag_runner.py"]
        found = set().union(*(_literals(m) for m in mods if m.name != "verdict.py"))
        for rid, row in ROWS.items():
            for name in row.checks:
                with self.subTest(rid=rid, check=name):
                    if name.endswith("*"):
                        self.assertTrue(any(s.startswith(name[:-1]) for s in found))
                    else:
                        self.assertIn(name, found)


class TestVerdict(unittest.TestCase):
    def test_pass(self):
        lines = verdict_lines(report(phase("snapshot", check("pcie_enum_count", PASS, "32/32"))))
        self.assertEqual(lines, ["Final verdict: PASS", "Repair recommendation: none"])

    def test_missing_chip_is_one_fault_across_snapshots(self):
        lost = check("pcie_enum_count", details="31/32", ip="pcie", parts=[CHIP])
        lines = verdict_lines(
            report(phase("snapshot", lost), phase("snapshot_after_r1", lost), phase("tests", check("gddr_full", SKIP)))
        )
        self.assertEqual(
            lines[0], f"Fault 1: FAIL F08 chip missing: PCIe link never trains: pcie_enum_count: 31/32 | {CHIP}"
        )
        self.assertEqual(
            lines[3:], ["Final verdict: FAIL: 1 fault(s)", f"Repair recommendation: Fault 1: {lines[2][10:]}"]
        )

    def test_non_gating_phases_are_evidence_only(self):
        lines = verdict_lines(
            report(
                phase("snapshot", check("pcie_enum_count", PASS, "32/32")),
                phase("captures", check("kernel_events", WARN, "AER on 46:00.0"), gates=False),
                phase("sys_triage", check("sys_triage", WARN, "1 section"), gates=False),
            )
        )
        self.assertEqual(
            lines, ["Evidence: kernel_events: AER on 46:00.0", "Final verdict: PASS", "Repair recommendation: none"]
        )

    def test_errored_phase_is_incomplete_not_a_fault(self):
        lines = verdict_lines(report(phase("eth_status", check("eth_status_read", WARN, "Boom()"), error="Boom()")))
        self.assertEqual(
            lines, ["Final verdict: INCOMPLETE (not a fault): eth_status did not finish", "Repair recommendation: none"]
        )

    def test_a_check_no_row_names_falls_back_to_its_ip(self):
        lines = verdict_lines(report(phase("eth", check("eth_status_heartbeat", details="2 stale", ip="eth"))))
        self.assertEqual(lines[0], "Fault 1: FAIL eth_status_heartbeat: 2 stale | host")
        self.assertIn("reseat or replace the cable", lines[1])

    def test_root_port_aer_names_the_chip_and_its_tray(self):
        bridge = check("triage_hostside_bridge_aer", WARN, "31/32 clean; bridge 0000:40:01.4: 1 correctable", ip="pcie")
        name_root_ports(bridge, {"0000:40:01.4": "0000:44:00.0", "0000:40:01.1": "0000:41:00.0"})
        self.assertEqual(
            bridge.details, "31/32 clean; bridge 0000:40:01.4 (root port of 0000:44:00.0 (UBB2/U4)): 1 correctable"
        )
        lines = verdict_lines(report(phase("triage", bridge)))
        self.assertTrue(lines[0].endswith("| 0000:44:00.0 (UBB2/U4)"))
        self.assertTrue(lines[1].startswith("  Repair: reseat UBB2's host PCIe cable;"))

    def test_a_passing_finding_is_left_as_written(self):
        clean = check("triage_hostside_bridge_aer", PASS, "32/32 bridge(s) clean 0000:40:01.4")
        name_root_ports(clean, {"0000:40:01.4": "0000:44:00.0"})
        self.assertEqual((clean.details, clean.parts), ("32/32 bridge(s) clean 0000:40:01.4", []))

    def test_chips_on_several_trays_keep_that_tray(self):
        two = check("pcie_gen", WARN, "Gen4", ip="pcie", parts=[CHIP, "0000:c3:00.0 (UBB3/U3)"])
        self.assertIn("reseat that tray's host PCIe cable", verdict_lines(report(phase("snapshot", two)))[2])

    def test_recommendation_puts_faults_before_warnings_and_merges_repeats(self):
        lines = verdict_lines(
            report(
                phase(
                    "p",
                    check("asic_thermal_precheck", WARN, "hot", parts=[CHIP]),
                    check("gddr_act_c3", parts=[CHIP]),
                    check("gddr_full", parts=[CHIP]),
                )
            )
        )
        self.assertEqual(
            lines[-1],
            "Repair recommendation: Fault 2, 3: re-run; if it repeats on the same chip, replace UBB2's board; "
            "Fault 1: check UBB2's fans and airflow",
        )
        self.assertEqual(lines[-2], "Final verdict: FAIL: 2 fault(s), 1 warning(s)")


if __name__ == "__main__":
    unittest.main()
