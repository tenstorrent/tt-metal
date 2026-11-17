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
from diag.galaxy import UBB_ETH_PORTS, expected_bdf  # noqa: E402
from diag.model import FAIL, PASS, SKIP, WARN, Phase  # noqa: E402

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
        for name in eth_status.FIRST_READ_SCRIPTS + eth_status.SECOND_READ_SCRIPTS:
            (triage_dir / f"{name}.py").write_text("")
        self.triage_dir = triage_dir
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

    def test_missing_triage_requirements_warn(self):
        fake_sysfs(self.sys_root)
        with mock.patch.object(eth_status, "TRIAGE_IMPORTS", "import ttexalens_not_installed_here"):
            phase = Phase(name="eth_status")
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                run_eth_status(self.tt_metal, phase, False, self.logs, sys_root=self.sys_root, interval_s=0)
        checks = by_name(phase.checks)
        self.assertEqual(checks["eth_status_read"].status, WARN)
        self.assertIn("ttexalens_not_installed_here", checks["eth_status_read"].details)
        self.assertFalse(self.ran.exists())
        phase.rollup()
        self.assertEqual(phase.status, WARN)

    def test_triage_script_that_does_not_import_is_named(self):
        # As tools/triage/inspector.capnp does in a sparse checkout without its
        # symlink target: triage's excepthook prints the traceback on stdout.
        fake_sysfs(self.sys_root)
        (self.triage_dir / "check_eth_status.py").write_text(
            "import sys\n"
            "print('inspector_capnp.py:13  SchemaParser.load  OSError: File not found: /x/inspector.capnp')\n"
            "sys.exit(1)\n"
        )
        checks = self.run_phase()
        self.assertEqual(checks["eth_status_read"].status, WARN)
        self.assertIn("OSError: File not found: /x/inspector.capnp", checks["eth_status_read"].details)
        self.assertNotIn("Install", checks["eth_status_read"].details)
        self.assertFalse(self.ran.exists())

    def test_missing_triage_script_warns(self):
        fake_sysfs(self.sys_root)
        (self.tt_metal / "tools" / "triage" / "triage.py").unlink()
        checks = self.run_phase()
        self.assertEqual(checks["eth_status_read"].status, WARN)
        self.assertIn("not present", checks["eth_status_read"].details)

    def test_falls_back_to_dev_tenstorrent_order_without_device_info(self):
        fake_sysfs(self.sys_root)
        (self.tt_metal / "tools" / "triage" / "read1.txt").write_text(
            triage_text(galaxy_rows(), with_device_info=False)
        )
        checks = self.run_phase()
        self.assertEqual(checks["eth_status_read"].data["chips_mapped_by"], "/dev/tenstorrent order")
        self.assertEqual(checks["eth_status_builtin_links"].status, PASS)


if __name__ == "__main__":
    unittest.main()
