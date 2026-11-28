#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""verdict_lines: each fault's cause and repair, evidence, the verdict and the repair line."""

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
from diag.snapshot_checks import fw_bundles  # noqa: E402
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
    "F02": (
        check("tray_power", WARN, cause="POWER", parts=TRAY2),
        "watch UBB2's 12 V feed; if its under-voltage latch sets again",
    ),
    "F03": (check("busbar_brick_vout", WARN, "PSU1 12.00 V"), "12.24 V or 12.35 V"),
    "F04": (check("pcie_enum_count", cause="POWER", parts=[CHIP]), "fix UBB2's 12 V feed"),
    "F05": (check("pcie_enum_count", parts=TRAY2), "reseat UBB2, then replace it"),
    "F06": (check("reset_1_glx_reset", parts=[CHIP]), "lost on reset again, replace UBB2's board"),
    "F07": (check("pcie_enum_count", cause="POWER-DOWNSTREAM", parts=[CHIP]), "power-good is low"),
    "F08": (check("pcie_enum_count", parts=[CHIP]), "swap UBB2's host PCIe cable"),
    "F09": (check("pcie_aer_diff", WARN, parts=[CHIP]), "if the errors follow the tray"),
    "F10": (check("pcie_gen", WARN, "Gen4", parts=[CHIP]), "to see if the fault follows"),
    "F11": (check("pcie_read", parts=[CHIP]), "swap UBB2's host PCIe cable"),
    "F13": (check("eth_status_links_down", details=f"{CHIP} ch4, TRACE port 2 down"), "a trace port stays"),
    "F14": (check("eth_status_links_down", details=f"{CHIP} ch8, QSFP-DD port 5 down"), "QSFP-DD cable on those"),
    "F18": (check("busbar_numeric", data={"vdd_min_mv": {CHIP: 760}}, parts=[CHIP]), "board"),
    "F19": (check("didt_matmul_galaxy_after_reset", parts=[CHIP]), "vdd_min to 760 mV"),
    "F20": (check("busbar_numeric", data={"aiclk_mhz": {CHIP: 1350}, "vdd_min_mv": {CHIP: 700}}), "vdd_min to 760 mV"),
    "F21": (check("gddr_act_c3", parts=[CHIP]), "replace UBB2's board"),
    "F22": (check("gddr_dbi_fw", WARN, "FW bundle 19.16.0.0 is older than 19.17.1"), "flash firmware 19.17.1"),
    "F24": (check("busbar_hang", parts=[CHIP]), "if that chip hangs again, replace UBB2's board"),
    "F25": (check("busbar_reboot", cause="POWER"), "PSUs and power feed"),
    "F26": (check("busbar_quiet_clock", WARN, parts=[CHIP]), "same firmware bundle"),
    "F27": (check("asic_thermal_precheck", WARN, "hot", parts=[CHIP]), "UBB2's fans and airflow"),
    "F28": (check("psu_ac_lost", WARN, cause="POWER"), "AC feed and cord of the PSU"),
    "F29": (check("noc_integrity_after_reset", parts=[CHIP]), "replace UBB2's board"),
}


def _emitted(sources: list[str]) -> set[str]:
    """Strings the stages use and the str constants they read; a module-level definition alone does not count."""
    consts, found, read = {}, set(), set()
    for tree in map(ast.parse, sources):
        skip = set()
        for a in tree.body:  # A = "x", A = B = "x", A, B = "x", "y" and A: str = "x"
            if not isinstance(a, (ast.Assign, ast.AnnAssign)) or a.value is None:
                continue
            for t in a.targets if isinstance(a, ast.Assign) else [a.target]:
                tuples = isinstance(t, ast.Tuple) and isinstance(a.value, ast.Tuple)
                for n, v in zip(t.elts, a.value.elts) if tuples else [(t, a.value)]:
                    if isinstance(n, ast.Name) and isinstance(v, ast.Constant) and type(v.value) is str:
                        consts[n.id] = v.value
                        skip.add(id(v))
        for n in ast.walk(tree):
            if type(getattr(n, "value", 0)) is str and isinstance(n, ast.Constant) and id(n) not in skip:
                found.add(n.value)
            if isinstance(getattr(n, "ctx", None), ast.Load):
                read.add(getattr(n, "id", getattr(n, "attr", "")))
    return found | {v for k, v in consts.items() if k in read}


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
                if rid == "F02":  # a latch alone is watched, not reset
                    self.assertNotIn("glx_reset", lines[-3])

    def test_every_row_check_has_an_emitter(self):
        """A row's check must be emitted: an exact name, or for "x*" a name starting with x."""
        mods = [*sorted((SUITE_DIR / "diag").glob("*.py")), SUITE_DIR / "diag_runner.py"]
        found = _emitted([m.read_text() for m in mods if m.name != "verdict.py"])
        for rid, row in ROWS.items():
            for name in row.checks:
                with self.subTest(rid=rid, check=name):
                    if name.endswith("*"):
                        self.assertTrue(any(s.startswith(name[:-1]) for s in found))
                    else:
                        self.assertIn(name, found)

    def test_a_definition_alone_is_not_an_emit(self):
        for src, want in {
            'A = "x"\nB = C = "y"\nD, E = "z", "w"\nF: str = "v"': set(),
            'A = "x"\nD, E = "z", "w"\nF: str = "v"\nf(A, E, name=F)': {"x", "w", "v"},
            'f("x")': {"x"},
        }.items():
            with self.subTest(src):
                self.assertEqual(_emitted([src]), want)


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

    def test_links_stage_checks_land_on_their_rows(self):
        def lines(name, ip, details="link a <-> b"):
            check = Check(name=name, status=FAIL, details=details, ip=ip, parts=[CHIP])
            return "\n".join(verdict_lines(report(phase("tests", check))))

        self.assertIn("reseat or replace the cable", lines("eth_bandwidth", "eth", "QSFP-DD port 3"))
        self.assertIn("reseat or replace the cable", lines("eth_data_integrity_after_reset", "eth", "TRACE port 7"))
        self.assertIn("FAIL F11 ", lines("pcie_read", "pcie"))
        self.assertIn("FAIL F29 ", lines("noc_integrity", "asic"))

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

    def test_warn_only(self):
        lines = verdict_lines(report(phase("snapshot", check("asic_thermal_precheck", WARN, "hot", ip="thermal"))))
        self.assertEqual(lines[-2], "Final verdict: WARN: 1 warning(s)")
        dram = check("dram_status", WARN, "ch3", ip="gddr", parts=[CHIP])
        self.assertIn("Repair: re-run", verdict_lines(report(phase("snapshot", dram)))[2])

    def test_one_chip_below_the_raised_vdd_min_still_gets_the_raise(self):
        c = check("busbar_numeric", data={"vdd_min_mv": {CHIP: 760, "0000:c3:00.0 (UBB3/U3)": 700}}, parts=[CHIP])
        self.assertIn("F19", verdict_lines(report(phase("busbar", c)))[0])

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


def test_only_parsed_fw_bundles_older_than_19_17_1_are_old():
    assert fw_bundles(["19.16.0.0", "19.17.1", "N/A", "", None, 19]) == (["19.16.0.0", "19.17.1"], ["19.16.0.0"])


if __name__ == "__main__":
    unittest.main()
