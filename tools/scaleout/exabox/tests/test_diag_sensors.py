#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tray power and AER: the HSC reads against a fake in-band BMC, the latch clear
and when it is withheld, the AER diff on a fake sysfs, and a lost tray's cause."""

from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))
sys.path.insert(0, str(TESTS_DIR))

from diag import aer, sensors  # noqa: E402
from diag.galaxy import expected_bdf  # noqa: E402
from diag.model import FAIL, PASS, SKIP, WARN, Check, Phase  # noqa: E402
from diag.verdict import attribute_power, verdict_lines  # noqa: E402
from test_diag_resets import fake_sysfs, root_port  # noqa: E402

CHIP1 = expected_bdf(1, 1)
GOOD_VIN = 0x0180  # 12.0 V at 0.03125 V/count


class FakeBmc:
    """The I2C devices behind ``ipmitool raw 0x06 0x52``: each HSC's registers, each
    tray CPLD's VR power-good, and a brick mux with two answering bricks.
    CLEAR_FAULTS clears the latches unless ``sticky``."""

    def __init__(self):
        good = {0x79: 0x0000, 0x7C: 0x00, 0x88: GOOD_VIN, 0x8C: 0x0100}
        self.regs = {
            (bus, a, r): v for bus in sensors.HSC_BUS.values() for a in sensors.HSC_ADDRS for r, v in good.items()
        }
        self.regs.update({(bus, sensors.TRAY_CPLD, sensors.CPLD_VR_PG): 0xFF for bus in sensors.HSC_BUS.values()})
        self.mux, self.sticky, self.cleared, self.calls = 0x00, set(), [], []
        self.refuse = ""  # stderr of a call that fails before reaching the bus

    def run(self, cmd, **_kw):
        self.calls.append(cmd)
        if self.refuse:
            return subprocess.CompletedProcess(cmd, 1, "", self.refuse)
        args = [int(x, 16) for x in cmd[cmd.index("raw") + 1 :]]
        bus, addr, n, *code = args[2:]
        out = None
        if bus == sensors.BRICK_BUS and addr == sensors.BRICK_MUX:
            if n != 1:
                self.mux = code[0]
            out = [self.mux] if n == 1 else []
        elif bus == sensors.BRICK_BUS and self.mux == 1 << 1 and addr in (0xC0, 0xC2):
            out = {0x20: [0x17], 0x8B: [0x00, 0x18], 0x8C: [0x20, 0xD3]}[code[0]][:n]  # VOUT_MODE, 0x1800, 0xD320
        elif n == 0 and code == [sensors.CLEAR_FAULTS]:
            self.cleared.append((bus, addr))
            if (bus, addr) not in self.sticky:
                self.regs[(bus, addr, 0x7C)] = 0
                self.regs[(bus, addr, 0x79)] &= ~0x2000
            out = []
        elif (bus, addr, code[0]) in self.regs:
            out = list(self.regs[(bus, addr, code[0])].to_bytes(n, "little"))
        if out is None:
            return subprocess.CompletedProcess(cmd, 1, "", "Unable to send RAW command\n")
        return subprocess.CompletedProcess(cmd, 0, " ".join(f"{b:02x}" for b in out), "")

    def set(self, tray, addr, reg, value):
        self.regs[(sensors.HSC_BUS[tray], addr, reg)] = value


class SensorCase(unittest.TestCase):
    def setUp(self):
        self.fake = FakeBmc()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        tmp = stack.enter_context(tempfile.TemporaryDirectory())
        for p in (
            mock.patch.object(sensors.subprocess, "run", self.fake.run),
            mock.patch.object(sensors.shutil, "which", lambda _n: "/usr/bin/ipmitool"),
            mock.patch.object(sensors.os, "geteuid", lambda: 1000),
            mock.patch.dict(os.environ, {"HSC_I2C_LOCK": str(Path(tmp) / "i2c.lock")}),
        ):
            stack.enter_context(p)

    def power(self):
        return sensors.read_power(sensors.Bmc())

    def judge(self, before, keep=""):
        """The tray_power checks, the other checks by name, and the causes."""
        phase = Phase(name="tray_power")
        causes = sensors.check_tray_power(phase, sensors.Bmc(), before, keep)
        return (
            [c for c in phase.checks if c.name == "tray_power"],
            {c.name: c for c in phase.checks if c.name != "tray_power"},
            causes,
        )


class TestTrayPower(SensorCase):
    def test_clean_trays_pass_and_read_the_bricks(self):
        before = self.power()
        self.assertEqual(before.hsc[(2, 0x40)].vin, 12.0)
        self.assertEqual(len(before.bricks), 8)
        self.assertAlmostEqual(before.bricks["U3"][0], 12.0)  # mux channel 1 only answers here
        self.assertEqual(before.bricks["U1"], (None, None))
        self.assertEqual(self.fake.mux, 0x00, "the brick mux is written back")
        self.assertTrue(all(c[:3] == ["sudo", "-n", "ipmitool"] for c in self.fake.calls))
        tps, rest, causes = self.judge(before)
        self.assertEqual([c.status for c in tps], [PASS])
        self.assertEqual(rest["hsc_latch_clear"].status, PASS)
        self.assertEqual(causes, {})

    def test_uv_latch_set_during_the_run_is_reported_then_cleared(self):
        before = self.power()
        self.fake.set(3, 0x42, 0x7C, 0x20)  # VIN_UV_WARN
        self.fake.set(3, 0x42, 0x79, 0x2000)
        (tp,), rest, causes = self.judge(before)
        self.assertEqual((tp.status, tp.cause), (WARN, "POWER"))
        self.assertIn("UBB3: HSC 0x42: VIN_UV_WARN set during the run", tp.details)
        self.assertEqual(len(tp.parts), 8)
        self.assertTrue(all("(UBB3/U" in p for p in tp.parts))
        self.assertEqual(causes[3][0], "POWER")
        self.assertEqual(rest["hsc_latch_clear"].status, PASS)
        self.assertEqual(len(self.fake.cleared), 8, "CLEAR_FAULTS once per HSC")
        self.assertEqual(self.fake.regs[(sensors.HSC_BUS[3], 0x42, 0x7C)], 0)

    def test_latch_already_set_before_the_run_says_so(self):
        self.fake.set(1, 0x40, 0x7C, 0x20)
        before = self.power()
        (tp,), _, causes = self.judge(before)
        self.assertIn("VIN_UV_WARN already set before the run", tp.details)
        self.assertEqual((tp.status, tp.cause), (WARN, ""))
        self.assertEqual(causes, {}, "a latch older than the run explains no lost chip")

    def test_latch_that_survives_the_clear_warns(self):
        before = self.power()
        self.fake.set(4, 0x40, 0x7C, 0x10)
        self.fake.sticky.add((sensors.HSC_BUS[4], 0x40))
        _, rest, _ = self.judge(before)
        self.assertEqual(rest["hsc_latch_clear"].status, WARN)
        self.assertIn("still set: UBB4 0x40 VIN_UV_FAULT", rest["hsc_latch_clear"].details)

    def test_hsc_off_withholds_every_clear(self):
        before = self.power()
        self.fake.set(2, 0x40, 0x79, 0x0040)
        self.fake.set(2, 0x40, 0x88, 0x0000)
        self.fake.set(1, 0x42, 0x7C, 0x20)
        tps, rest, causes = self.judge(before)
        self.assertEqual({c.details[:4]: c.status for c in tps}["UBB2"], FAIL)
        self.assertEqual(causes[2][0], "POWER")
        self.assertEqual(rest["hsc_latch_clear"].status, SKIP)
        self.assertIn("skipped: HSC OFF (UBB2 0x40)", rest["hsc_latch_clear"].details)
        self.assertEqual(self.fake.cleared, [])

    def test_no_hsc_clear_keeps_the_latches(self):
        before = self.power()
        self.fake.set(1, 0x42, 0x7C, 0x20)
        _, rest, _ = self.judge(before, keep="--no-hsc-clear")
        self.assertEqual(rest["hsc_latch_clear"].details, "skipped: --no-hsc-clear")
        self.assertEqual(self.fake.cleared, [])

    def test_vr_power_good_low_with_12v_good_is_downstream(self):
        before = self.power()
        self.fake.regs[(sensors.HSC_BUS[1], sensors.TRAY_CPLD, sensors.CPLD_VR_PG)] = 0xF7
        (tp,), _, causes = self.judge(before)
        self.assertEqual((tp.status, tp.cause), (FAIL, "POWER-DOWNSTREAM"))
        self.assertIn("VR power-good 0xf7", causes[1][1])

    def test_no_route_is_one_skipped_row(self):
        real = os.environ["HSC_I2C_LOCK"]
        cases = [  # stderr of the failing call, lock path (None: the real one), hold the lock, SKIP details, calls sent
            ("sudo: a password is required\n", None, False, "skipped: no root, and sudo -n ipmitool was refused", 1),
            ("Could not open device at /dev/ipmi0\n", None, False, "skipped: no BMC device", 1),
            ("", "/nonexistent-dir/i2c.lock", False, "skipped: cannot open the I2C lock /nonexistent-dir/i2c.lock", 0),
            ("", None, True, "skipped: I2C lock busy after 0.1 s", 0),
        ]
        for refuse, lock, hold, details, calls in cases:
            with self.subTest(details), contextlib.ExitStack() as stack:
                self.fake.refuse = refuse
                before = self.power() if hold else None  # a good first read, then the lock goes
                self.fake.calls = []
                stack.enter_context(mock.patch.object(sensors, "LOCK_WAIT_S", 0.1))
                stack.enter_context(mock.patch.dict(os.environ, {"HSC_I2C_LOCK": lock or real}))
                if hold:
                    fcntl.flock(stack.enter_context(open(real, "a")), fcntl.LOCK_EX)
                (tp,), rest, causes = self.judge(before)
                self.assertEqual(tp.status, SKIP)
                self.assertTrue(tp.details.startswith(details), tp.details)
                self.assertEqual((len(self.fake.calls), causes, rest), (calls, {}, {}))


class TestLostTrayCause(SensorCase):
    @staticmethod
    def snapshot(details, parts):
        snap = Phase(name="snapshot")
        snap.add(Check(name="pcie_enum_count", status=FAIL, details=details, ip="pcie", parts=parts))
        snap.rollup()
        return snap

    def test_trip_and_new_flags_come_with_the_cause(self):
        before = self.power()
        self.fake.set(3, 0x42, 0x7C, 0x20)  # VIN_UV_WARN alone
        self.fake.set(3, 0x42, 0x79, 0x2000)
        self.fake.set(2, 0x40, 0x79, 0x0050)  # OFF + IOUT_OC_FAULT
        causes = sensors.tray_causes(self.power(), before)
        self.assertEqual((causes[3].trip, causes[3].new), (False, {0x42: ["VIN_UV_WARN"]}))
        self.assertTrue(causes[2].trip)
        self.assertEqual(causes[2].new[0x40], sensors.read_hsc(sensors.Bmc(), 2, 0x40).flags)

    def test_lost_tray_is_judged_power(self):
        before = self.power()
        self.fake.set(2, 0x40, 0x79, 0x0050)  # OFF + IOUT_OC_FAULT
        power = Phase(name="tray_power")
        causes = sensors.check_tray_power(power, sensors.Bmc(), before, "")
        snap = self.snapshot("24/32 chips on the bus", [f"0000:4{a}:00.0 (UBB2/U{a})" for a in range(1, 9)])
        power.rollup()
        report = {"phases": {p.name: asdict(p) for p in (snap, power)}}
        attribute_power(report, causes)
        row = report["phases"]["snapshot"]["checks"][0]
        self.assertEqual(row["cause"], "POWER")
        self.assertTrue(row["details"].startswith("UBB2 12 V: HSC 0x40"))
        lines = verdict_lines(report)
        i = next(i for i, ln in enumerate(lines) if "pcie_enum_count" in ln)
        self.assertIn("FAIL F04 bad bus bar: pcie_enum_count: UBB2 12 V", lines[i])
        self.assertEqual(lines[i + 1], "  Cause: POWER")
        self.assertTrue(lines[i + 2].startswith("  Repair: fix UBB2's 12 V feed"))

    def test_lost_tray_with_no_power_reading_stays_unattributed(self):
        report = {"phases": {"snapshot": asdict(self.snapshot("24/32", ["0000:46:00.0 (UBB2/U6)"]))}}
        attribute_power(report, {})
        self.assertEqual(report["phases"]["snapshot"]["checks"][0]["cause"], "")


class TestAerDiff(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        fake_sysfs(self.root)

    def counts(self, bdf, sev, **counts):
        lines = [f"{k} {v}" for k, v in counts.items()] + [f"TOTAL_ERR_{sev.upper()} {sum(counts.values())}"]
        (self.root / "bus" / "pci" / "devices" / bdf / f"aer_dev_{sev}").write_text("\n".join(lines) + "\n")

    def diff(self, before):
        """The check_aer rows from *before* to the counters now."""
        phase = Phase(name="pcie_aer")
        aer.check_aer(phase, before, aer.read_aer(self.root))
        return phase.checks

    def test_correctable_rise_at_the_root_port_warns_naming_the_chip(self):
        for bdf in aer.chip_root_ports(self.root):
            self.counts(bdf, "correctable", RxErr=0, BadTLP=0)
        self.counts(root_port("0000:46:00.0"), "correctable", RxErr=3)
        before = aer.read_aer(self.root)
        self.counts(root_port("0000:46:00.0"), "correctable", RxErr=15)
        (c,) = self.diff(before)
        self.assertEqual(c.status, WARN)
        self.assertEqual(c.details, "0000:46:00.0 (UBB2/U6): root port correctable RxErr +12")
        self.assertEqual(c.parts, ["0000:46:00.0 (UBB2/U6)"])

    def test_uncorrectable_rise_fails(self):
        self.counts(CHIP1, "nonfatal", DLP=0)
        before = aer.read_aer(self.root)
        self.counts(CHIP1, "nonfatal", DLP=1)
        (c,) = self.diff(before)
        self.assertEqual(c.status, FAIL)
        self.assertIn("chip nonfatal DLP +1", c.details)

    def test_unchanged_passes_and_no_counters_skips(self):
        self.assertEqual(self.diff(aer.read_aer(self.root))[0].status, SKIP)
        self.counts(CHIP1, "correctable", RxErr=2)
        self.assertEqual(self.diff(aer.read_aer(self.root))[0].status, PASS)


if __name__ == "__main__":
    unittest.main()
