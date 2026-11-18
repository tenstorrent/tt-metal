#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The ETH status phase without hardware: triage --llm-output parsing, the
judgement of two reads against the BH Galaxy port map, the sysfs PCIe gate, and
one end-to-end run with a stand-in triage.py."""

from __future__ import annotations

import csv
import io
import sys
import tempfile
import textwrap
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from diag import eth_status  # noqa: E402
from diag.eth_status import (  # noqa: E402
    judge_eth_status,
    parse_chip_bdfs,
    parse_eth_links,
    parse_triage_tables,
    pcie_gate_check,
    read_pcie_functions,
    run_eth_status,
    sysfs_chip_bdfs,
)
from diag.eth_summary import SUMMARY_FILE, render_eth_summary  # noqa: E402
from diag.galaxy import UBB_ETH_PORTS, expected_bdf  # noqa: E402
from diag.model import FAIL, PASS, SKIP, WARN, Check, Phase  # noqa: E402

ETH_HEADER = (
    "Dev,Loc,Port Status,Link,Heartbeat,Retrain Count,CRC Errors,Corr CW,Uncorr CW,RX Link Up,"
    "ERR_STAT,TXQ Resends,RXQ Drops,ETH FW,FW Stage"
)
ETH_COLS = ETH_HEADER.split(",")
DEVICE_INFO_HEADER = "Tray/Board,Chip,ASIC,Arch,Board,Bus ID,PCI BDF,Unique ID,ARC FW,Postcode"

# Chip ids as UMD hands them out: in /dev/tenstorrent order, which is PCI order.
SLOTS = sorted(((t, a) for t in range(1, 5) for a in range(1, 9)), key=lambda ta: expected_bdf(*ta))
CHIP_OF = {slot: chip for chip, slot in enumerate(SLOTS)}
CHIP_BDFS = {chip: expected_bdf(*slot) for slot, chip in CHIP_OF.items()}
XY_TORUS = (1, 2, 3, 4, 5, 6)


def galaxy_rows(qsfp=XY_TORUS, drop=(), only_on=None, overrides=None) -> list[dict]:
    """check_eth_status rows of a healthy Galaxy: every built-in lane, plus the QSFP
    ports in ``qsfp`` on every UBB. ``drop`` holds (tray, asic, ch) lanes that are not
    listed; ``only_on`` maps a QSFP port id to the UBBs it is linked on; ``overrides``
    maps (tray, asic, ch) to column values."""
    only_on = only_on or {}
    overrides = overrides or {}
    rows = []
    for (ptype, pid), lanes in UBB_ETH_PORTS.items():
        if ptype == "QSFP_DD" and pid not in qsfp and pid not in only_on:
            continue
        for tray in range(1, 5):
            if pid in only_on and ptype == "QSFP_DD" and tray not in only_on[pid]:
                continue
            for asic, ch in lanes:
                if (tray, asic, ch) in drop:
                    continue
                row = {
                    "Dev": str(CHIP_OF[tray, asic]),
                    "Loc": f"{ch + 1}-1 (e0,{ch})",
                    "Port Status": "Up",
                    "Link": "Up",
                    "Heartbeat": "Up",
                    "Retrain Count": "0",
                    "CRC Errors": "0",
                    "Corr CW": "12",
                    "Uncorr CW": "0",
                    "RX Link Up": "Up",
                    "ERR_STAT": "0x0",
                    "TXQ Resends": "0",
                    "RXQ Drops": "0",
                    "ETH FW": "1.7.0",
                    "FW Stage": "Base FW",
                }
                row.update(overrides.get((tray, asic, ch), {}))
                rows.append(row)
    return rows


def csv_line(cells: list[str]) -> str:
    # How triage's CsvSerializer writes a row: a Loc cell such as "1-1 (e0,0)" is quoted.
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(cells)
    return buf.getvalue()


def triage_text(rows: list[dict], with_device_info: bool = True) -> str:
    out = []
    if with_device_info:
        out += ["", "device_info.py:", DEVICE_INFO_HEADER]
        for chip, bdf in sorted(CHIP_BDFS.items()):
            tray, asic = SLOTS[chip]
            bus = f"0x{int(bdf[5:7], 16):02x}"
            out.append(
                csv_line(
                    [f"Tray {tray}", str(chip), str(asic), "blackhole", "ubb", bus, bdf, hex(chip), "1.2.3", "N/A"]
                )
            )
    out += ["", "check_eth_status.py:", "  Eth port is not trained", ETH_HEADER]
    out += [csv_line([r[c] for c in ETH_COLS]) for r in rows]
    return "\n".join(out) + "\n"


def links(rows: list[dict]):
    return parse_eth_links(parse_triage_tables(triage_text(rows)))


def by_name(checks) -> dict:
    return {c.name: c for c in checks}


def judge(first_rows, second_rows=None, interval=30.0, chips_on_pcie=32) -> dict:
    second = links(second_rows) if second_rows is not None else None
    return by_name(judge_eth_status(links(first_rows), second, interval if second else None, CHIP_BDFS, chips_on_pcie))


class TestParsing(unittest.TestCase):
    def test_reads_both_tables_and_skips_indented_lines(self):
        tables = parse_triage_tables(triage_text(galaxy_rows()))
        self.assertEqual(set(tables), {"device_info", "check_eth_status"})
        self.assertEqual(tables["check_eth_status"][0], ETH_COLS)
        self.assertEqual(parse_chip_bdfs(tables), CHIP_BDFS)

    def test_link_fields(self):
        parsed = links(galaxy_rows(overrides={(2, 6, 4): {"ERR_STAT": "0x10", "Uncorr CW": "0x20"}}))
        link = parsed[CHIP_OF[2, 6], "5-1"]
        self.assertEqual((link.channel, link.port, link.err_stat, link.uncorr), (4, "Up", 0x10, 0x20))

    def test_stray_unindented_line_is_dropped(self):
        text = triage_text(galaxy_rows()) + "WARNING: something triage printed\n"
        self.assertEqual(len(parse_eth_links(parse_triage_tables(text))), len(galaxy_rows()))

    def test_script_times_suffix(self):
        text = triage_text(galaxy_rows()).replace("check_eth_status.py:", "check_eth_status.py [3.21s]:")
        self.assertEqual(len(parse_eth_links(parse_triage_tables(text))), len(galaxy_rows()))


class TestHealthy(unittest.TestCase):
    def test_xy_torus_passes_and_is_named(self):
        checks = judge(galaxy_rows(), galaxy_rows())
        self.assertEqual({n: c.status for n, c in checks.items() if c.status != PASS}, {})
        self.assertIn("as in an XY torus", checks["eth_status_cabling"].details)
        self.assertIn("208/208 built-in", checks["eth_status_cabling"].details)
        self.assertIn("256 on-host links: 208/208 built-in + 48 on cables", checks["eth_status_cabling"].details)

    def test_mesh(self):
        checks = judge(galaxy_rows(qsfp=()), galaxy_rows(qsfp=()))
        self.assertIn("No QSFP port linked inside this host, as in a mesh", checks["eth_status_cabling"].details)

    def test_cabling_without_a_name_is_described_not_judged(self):
        rows = galaxy_rows(qsfp=(1, 2, 7))
        checks = judge(rows, rows)
        self.assertEqual(checks["eth_status_cabling"].status, PASS)
        self.assertIn(
            "QSFP 1,2,7 linked on all 4 UBBs: not a cabling with a name here", checks["eth_status_cabling"].details
        )

    def test_idle_port_that_is_down_is_not_a_fault(self):
        rows = galaxy_rows(overrides={(1, 5, 0): {"Port Status": "Unused", "Link": "Down"}})
        self.assertEqual(judge(rows, rows)["eth_status_links_down"].status, PASS)


class TestLinkFaults(unittest.TestCase):
    def test_link_down_names_the_port(self):
        rows = galaxy_rows(overrides={(1, 5, 0): {"Link": "Down"}})
        c = judge(rows, rows)["eth_status_links_down"]
        self.assertEqual(c.status, FAIL)
        self.assertIn(f"UBB1/U5 ch0 (TRACE P1, dev {CHIP_OF[1, 5]}): link down", c.details)

    def test_link_gone_between_the_reads(self):
        c = judge(galaxy_rows(), galaxy_rows(drop={(4, 8, 2)}))["eth_status_links_down"]
        self.assertEqual(c.status, FAIL)
        self.assertIn("UBB4/U8 ch2 (TRACE P16", c.details)
        self.assertIn("gone between the reads", c.details)

    def test_unreadable_core(self):
        rows = galaxy_rows(overrides={(3, 1, 4): {"Port Status": "Unreadable", "Link": "Unreadable"}})
        self.assertIn("eth core unreadable", judge(rows, rows)["eth_status_links_down"].details)

    def test_heartbeat_down_fails_released_does_not(self):
        rows = galaxy_rows(overrides={(2, 2, 2): {"Heartbeat": "Down"}, (2, 3, 2): {"Heartbeat": "Released"}})
        c = judge(rows, rows)["eth_status_heartbeat"]
        self.assertEqual(c.status, FAIL)
        self.assertEqual(len(c.data["stopped"]), 1)

    def test_odd_port_state_warns(self):
        rows = galaxy_rows(overrides={(1, 1, 4): {"Port Status": "Unknown"}})
        self.assertEqual(judge(rows, rows)["eth_status_port_state"].status, WARN)


class TestCounters(unittest.TestCase):
    def test_uncorrected_fec_rate(self):
        for per_s, expected in ((150, FAIL), (15, WARN), (1, PASS)):
            second = galaxy_rows(overrides={(1, 2, 2): {"Uncorr CW": str(per_s * 30)}})
            self.assertEqual(judge(galaxy_rows(), second)["eth_status_fec_uncorrected"].status, expected, per_s)

    def test_counter_that_went_down_is_a_reset_not_a_rate(self):
        first = galaxy_rows(overrides={(1, 2, 2): {"Uncorr CW": "900000", "Retrain Count": "2"}})
        c = judge(first, galaxy_rows())
        self.assertEqual(c["eth_status_fec_uncorrected"].status, PASS)
        self.assertIn("1 link(s) reset between the reads", c["eth_status_fec_uncorrected"].details)
        self.assertEqual(c["eth_status_retrains"].status, PASS)

    def test_crc_rate_warns(self):
        second = galaxy_rows(overrides={(3, 3, 0): {"CRC Errors": "60"}})
        self.assertEqual(judge(galaxy_rows(), second)["eth_status_crc_errors"].status, WARN)

    def test_retrain_during_the_check_fails(self):
        second = galaxy_rows(overrides={(1, 7, 2): {"Retrain Count": "1"}})
        c = judge(galaxy_rows(), second)["eth_status_retrains"]
        self.assertEqual(c.status, FAIL)
        self.assertIn("flapping", c.details)

    def test_retrains_since_reset_warn(self):
        rows = galaxy_rows(overrides={(1, 7, 2): {"Retrain Count": "5"}})
        self.assertEqual(judge(rows, rows)["eth_status_retrains"].status, WARN)

    def test_err_stat_warns(self):
        rows = galaxy_rows(overrides={(1, 7, 2): {"ERR_STAT": "0x4"}})
        self.assertEqual(judge(rows, rows)["eth_status_err_stat"].status, WARN)

    def test_one_read_only(self):
        checks = judge(galaxy_rows(overrides={(1, 2, 2): {"Uncorr CW": "2000000"}}))
        self.assertEqual(checks["eth_status_crc_errors"].status, SKIP)
        self.assertEqual(checks["eth_status_fec_uncorrected"].status, WARN)
        self.assertIn("one read only", checks["eth_status_fec_uncorrected"].details)
        self.assertIn("one read only", checks["eth_status_retrains"].details)


class TestTopology(unittest.TestCase):
    def test_builtin_link_not_linked_fails(self):
        rows = galaxy_rows(drop={(2, 5, 0)})
        checks = judge(rows, rows)
        c = checks["eth_status_builtin_links"]
        self.assertEqual(c.status, FAIL)
        self.assertIn("UBB2 TRACE P1: U5 ch0 (on-board trace, nothing to reseat)", c.details)
        # A link that never came up is not listed, so it is not "down".
        self.assertEqual(checks["eth_status_links_down"].status, PASS)

    def test_linking_board(self):
        rows = galaxy_rows(drop={(4, 8, 4)})
        self.assertIn("(linking board)", judge(rows, rows)["eth_status_builtin_links"].details)

    def test_cable_linked_in_part_fails(self):
        rows = galaxy_rows(drop={(3, 2, 1)})
        checks = judge(rows, rows)
        self.assertEqual(checks["eth_status_cable_lanes"].status, FAIL)
        self.assertIn("UBB3 QSFP P4: 1 of 2 lanes", checks["eth_status_cable_lanes"].details)
        self.assertEqual(checks["eth_status_cable_trays_alike"].status, PASS)
        self.assertIn("not linked alike on every UBB", checks["eth_status_cabling"].details)

    def test_port_linked_on_some_ubbs_warns(self):
        rows = galaxy_rows(only_on={7: (1, 2, 3)})
        c = judge(rows, rows)["eth_status_cable_trays_alike"]
        self.assertEqual(c.status, WARN)
        self.assertIn("QSFP P7: linked on UBB1, UBB2, UBB3, not on UBB4", c.details)

    def test_missing_chip_leaves_topology_unjudged(self):
        checks = judge(galaxy_rows(), galaxy_rows(), chips_on_pcie=31)
        for name in eth_status.TOPOLOGY_CHECKS:
            self.assertEqual(checks[name].status, SKIP, name)
            self.assertIn("only 31 of 32 chips on PCIe", checks[name].details)


def fake_sysfs(root: Path, bad: dict | None = None, count: int = 32) -> None:
    """32 Tenstorrent functions; ``bad`` maps a BDF to (vendor, command) read from config."""
    bad = bad or {}
    for chip, bdf in list(sorted(CHIP_BDFS.items()))[:count]:
        dev = root / "bus" / "pci" / "devices" / bdf
        dev.mkdir(parents=True)
        (dev / "vendor").write_text("0x1e52\n")
        vendor, command = bad.get(bdf, (0x1E52, 0x0006))
        (dev / "config").write_bytes(vendor.to_bytes(2, "little") + b"\x40\xb1" + command.to_bytes(2, "little"))
        node = root / "class" / "tenstorrent" / f"tenstorrent!{chip}"
        node.mkdir(parents=True)
        (node / "device").symlink_to(dev)


class TestPcieGate(unittest.TestCase):
    def test_all_answering(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_sysfs(Path(tmp))
            gate = pcie_gate_check(read_pcie_functions(Path(tmp)))
        self.assertEqual(gate.status, PASS)
        self.assertIn("32/32 chips on PCIe", gate.details)

    def test_dropped_chip_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_sysfs(Path(tmp), {"0000:46:00.0": (0xFFFF, 0xFFFF), "0000:c3:00.0": (0x1E52, 0x0004)})
            gate = pcie_gate_check(read_pcie_functions(Path(tmp)))
        self.assertEqual(gate.status, FAIL)
        self.assertIn("0000:46:00.0 (UBB2/U6) not answering", gate.details)
        self.assertIn("0000:c3:00.0 (UBB3/U3) memory decoding off (Command=0x0004)", gate.details)

    def test_no_chip_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(pcie_gate_check(read_pcie_functions(Path(tmp))).status, SKIP)

    def test_missing_chip_is_noted_not_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_sysfs(Path(tmp), count=31)
            gate = pcie_gate_check(read_pcie_functions(Path(tmp)))
        self.assertEqual(gate.status, PASS)
        self.assertIn("31/32", gate.details)

    def test_sysfs_chip_order_skips_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_sysfs(Path(tmp))
            (Path(tmp) / "class" / "tenstorrent" / "tenstorrent!0" / "device").unlink()
            (Path(tmp) / "class" / "tenstorrent" / "tenstorrent!0").rmdir()
            order = sysfs_chip_bdfs(Path(tmp))
        self.assertEqual(len(order), 31)
        self.assertEqual(order[0], CHIP_BDFS[1])


# Stands in for tools/triage/triage.py: prints the read beside it, and leaves a
# mark so a test can tell whether it ran at all.
FAKE_TRIAGE = textwrap.dedent(
    """\
    import sys
    from pathlib import Path
    here = Path(__file__).parent
    (here / "ran").write_text("yes")
    name = "read1.txt" if "--run=device_info" in sys.argv else "read2.txt"
    print((here / name).read_text(), end="")
    """
)


class TestRunEthStatus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.tt_metal, self.sys_root, self.logs = root / "tt-metal", root / "sys", root / "out" / "logs"
        triage_dir = self.tt_metal / "tools" / "triage"
        triage_dir.mkdir(parents=True)
        (triage_dir / "triage.py").write_text(FAKE_TRIAGE)
        (triage_dir / "read1.txt").write_text(triage_text(galaxy_rows()))
        second = galaxy_rows(overrides={(2, 4, 6): {"Retrain Count": "1"}})
        (triage_dir / "read2.txt").write_text(triage_text(second, with_device_info=False))
        self.ran = triage_dir / "ran"
        # The stand-in imports nothing; the real probe would need tt-exalens here.
        patcher = mock.patch.object(eth_status, "TRIAGE_IMPORTS", "import json")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def run_phase(self) -> dict:
        phase = Phase(name="eth_status")
        with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            run_eth_status(self.tt_metal, phase, False, self.logs, sys_root=self.sys_root, interval_s=0)
        return by_name(phase.checks)

    def test_two_reads_judged(self):
        fake_sysfs(self.sys_root)
        checks = self.run_phase()
        self.assertEqual(checks["eth_status_read"].status, PASS, checks["eth_status_read"].details)
        self.assertEqual(checks["eth_status_read"].data["chips_mapped_by"], "device_info")
        self.assertEqual(checks["eth_status_retrains"].status, FAIL)
        self.assertIn("UBB2/U4 ch6 (TRACE P20", checks["eth_status_retrains"].details)
        self.assertEqual(checks["eth_status_builtin_links"].status, PASS)
        self.assertTrue((self.logs / "eth_status_read_1.csv").is_file())
        self.assertTrue((self.logs / "eth_status_read_2.log").is_file())

    def test_dropped_chip_stops_before_opening_any(self):
        fake_sysfs(self.sys_root, {"0000:46:00.0": (0xFFFF, 0xFFFF)})
        checks = self.run_phase()
        self.assertEqual(checks["eth_status_pcie_gate"].status, FAIL)
        self.assertEqual(checks["eth_status_read"].status, SKIP)
        self.assertFalse(self.ran.exists())

    def test_missing_triage_requirements_skip(self):
        fake_sysfs(self.sys_root)
        with mock.patch.object(eth_status, "TRIAGE_IMPORTS", "import ttexalens_not_installed_here"):
            checks = self.run_phase()
        self.assertEqual(checks["eth_status_read"].status, SKIP)
        self.assertIn("ttexalens_not_installed_here", checks["eth_status_read"].details)
        self.assertFalse(self.ran.exists())

    def test_falls_back_to_dev_tenstorrent_order_without_device_info(self):
        fake_sysfs(self.sys_root)
        (self.tt_metal / "tools" / "triage" / "read1.txt").write_text(
            triage_text(galaxy_rows(), with_device_info=False)
        )
        checks = self.run_phase()
        self.assertEqual(checks["eth_status_read"].data["chips_mapped_by"], "/dev/tenstorrent order")
        self.assertEqual(checks["eth_status_builtin_links"].status, PASS)


GATE_OK = Check(
    name="eth_status_pcie_gate", status=PASS, details="", data={"present": 32, "expected": 32, "dropped": []}, ip="pcie"
)
READ_OK = Check(name="eth_status_read", status=PASS, details="2 reads 30s apart", data={"interval_s": 30.0})


def report_of(eth_checks, snapshot_checks=()) -> dict:
    """A report as run_diag() writes it, with the snapshot and ETH status phases only."""
    phases = {}
    for name, checks in (("snapshot", snapshot_checks), ("eth_status", eth_checks)):
        phase = Phase(name=name, checks=list(checks))
        phase.rollup()
        phases[name] = asdict(phase)
    statuses = [p["status"] for p in phases.values()]
    overall = FAIL if FAIL in statuses else WARN if WARN in statuses else PASS
    return {
        "host": "bh-glx-test",
        "tier": "flash",
        "started_utc": "2026-11-18T10:50:23.412+00:00",
        "overall_status": overall,
        "phases": phases,
    }


def judged_report(first_rows, second_rows=None, snapshot_checks=()) -> str:
    second = links(second_rows if second_rows is not None else first_rows)
    checks = judge_eth_status(links(first_rows), second, 30.0, CHIP_BDFS, 32)
    return render_eth_summary(report_of([GATE_OK, READ_OK, *checks], snapshot_checks))


class TestSummary(unittest.TestCase):
    def test_healthy(self):
        text = judged_report(galaxy_rows())
        self.assertIn("VERDICT: HEALTHY - no fault found, no errors rising", text)
        self.assertIn("2 reads 30 s apart", text)
        self.assertIn("QSFP ports 1-2   (Y torus)   linked on all 4 UBBs", text)
        self.assertIn("QSFP ports 7-14  (Z ports)   not linked inside this host", text)
        self.assertIn("as in an XY torus", text)
        self.assertNotIn("** WARNING **", text)
        self.assertNotIn("** PROBLEM **", text)

    def test_warning_lists_each_link(self):
        rows = galaxy_rows(overrides={(1, 7, 2): {"Retrain Count": "5"}})
        text = judged_report(rows)
        self.assertIn("VERDICT: WARNING - nothing proven broken, worth a look (1 found)", text)
        self.assertIn(f"- UBB1/U7 ch2 (TRACE P10, dev {CHIP_OF[1, 7]}): 5 retrains since the last reset", text)
        self.assertRegex(text, r"retrains since last reset\s+1 link\(s\) at >= 3\s+< 3 per link\s+\*\* WARNING \*\*")

    def test_problems_first_then_warnings(self):
        rows = galaxy_rows(overrides={(1, 5, 0): {"Link": "Down"}, (1, 7, 2): {"Retrain Count": "5"}})
        text = judged_report(rows)
        self.assertIn("VERDICT: PROBLEM - links broken or corrupting data (1 found)", text)
        self.assertLess(text.index("link down"), text.index("also worth a look:"))
        self.assertLess(text.index("also worth a look:"), text.index("5 retrains since the last reset"))

    def test_long_lists_are_cut_and_counted(self):
        rows = galaxy_rows(overrides={(1, 5, ch): {"Link": "Down"} for ch in range(10)})
        text = judged_report(rows)
        self.assertIn("(8 found)", text)
        rows = galaxy_rows(overrides={(t, 5, ch): {"Link": "Down"} for t in (1, 2) for ch in range(10)})
        self.assertIn("... and 8 more", judged_report(rows))

    def test_partial_cable_in_the_cabling_lines(self):
        text = judged_report(galaxy_rows(drop={(3, 2, 1)}))
        self.assertIn("QSFP port  4     (X torus)   linked on UBB1, UBB2, UBB4, in part on UBB3", text)
        self.assertIn("only 1 of 2 lanes linked (U2 ch1 not), the cable is there and part of it is dead", text)

    def test_snapshot_fault_without_findings_is_one_line(self):
        snap = [Check(name="pcie_gen", status=WARN, details="U6 chips: 0000:46:00.0=Gen1(!)", ip="pcie")]
        text = judged_report(galaxy_rows(), snapshot_checks=snap)
        self.assertIn("- pcie_gen: U6 chips: 0000:46:00.0=Gen1(!)", text)
        self.assertRegex(text, r"other tt-smi snapshot checks\s+0 fail, 1 warn")

    def test_snapshot_failure_with_clean_links(self):
        snap = [Check(name="dram_status", status=FAIL, details="31/32 dram_status=True", ip="gddr")]
        text = judged_report(galaxy_rows(), snapshot_checks=snap)
        self.assertIn("VERDICT: PROBLEM - no ETH link broken, but other checks failed (1 found)", text)

    def test_dropped_chip(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_sysfs(Path(tmp), {"0000:46:00.0": (0xFFFF, 0xFFFF)})
            gate = pcie_gate_check(read_pcie_functions(Path(tmp)))
        read = Check(name="eth_status_read", status=SKIP, details="links not read: a chip dropped off PCIe")
        text = render_eth_summary(report_of([gate, read]))
        self.assertIn("VERDICT: PROBLEM - a chip dropped off PCIe, the links were not read (1 found)", text)
        self.assertIn("- 0000:46:00.0 (UBB2/U6) not answering on PCIe: do NOT reset, report it", text)
        self.assertIn("Do NOT reset yet", text)
        self.assertRegex(text, r"chips on PCIe\s+31 of 32 ok\s+32\s+\*\* PROBLEM \*\*")

    def test_links_not_read(self):
        text = render_eth_summary(report_of([Check(name="eth_status_read", status=SKIP, details="--skip-eth-status")]))
        self.assertIn("VERDICT: NOT CHECKED - the links were not read: --skip-eth-status", text)
        self.assertNotIn("HEALTHY", text)

    def test_runner_prints_and_keeps_it(self):
        import diag_runner

        report = report_of([GATE_OK, READ_OK, *judge_eth_status(links(galaxy_rows()), None, None, CHIP_BDFS, 32)])
        with tempfile.TemporaryDirectory() as tmp, mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            diag_runner._print_eth_summary(report, Path(tmp) / "diag_report.json", Path(tmp) / "logs")
            kept = (Path(tmp) / "logs" / SUMMARY_FILE).read_text()
        self.assertIn("VERDICT: HEALTHY - no fault found", out.getvalue())
        self.assertIn(kept, out.getvalue())


if __name__ == "__main__":
    unittest.main()
