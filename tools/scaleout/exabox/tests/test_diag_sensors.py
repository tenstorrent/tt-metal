#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tray power and AER: the HSC reads against a fake in-band BMC, the latch clear
and when it is withheld, the AER diff on a fake sysfs, and a lost tray's cause."""

from __future__ import annotations

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
        self.regs = {}
        for t, bus in sensors.HSC_BUS.items():
            for a in sensors.HSC_ADDRS:
                self.regs[(bus, a, 0x79)] = 0x0000
                self.regs[(bus, a, 0x7C)] = 0x00
                self.regs[(bus, a, 0x88)] = GOOD_VIN
                self.regs[(bus, a, 0x8C)] = 0x0100
            self.regs[(bus, sensors.TRAY_CPLD, sensors.CPLD_VR_PG)] = 0xFF
        self.mux = 0x00
        self.sticky = set()
        self.cleared = []
        self.calls = []
        self.refuse = ""  # stderr of a call that fails before reaching the bus

    def run(self, cmd, **_kw):
        self.calls.append(cmd)
        if self.refuse:
            return subprocess.CompletedProcess(cmd, 1, "", self.refuse)
        args = [int(x, 16) for x in cmd[cmd.index("raw") + 1 :]]
        bus, addr, n, *code = args[2:]
        out = None
        if bus == sensors.BRICK_BUS and addr == sensors.BRICK_MUX:
            if n == 1:
                out = [self.mux]
            else:
                self.mux, out = code[0], []
        elif bus == sensors.BRICK_BUS and self.mux == 1 << 1 and addr in (0xC0, 0xC2):
            out = {
                0x20: [0x17],
                0x8B: list((0x1800).to_bytes(2, "little")),
                0x8C: list((0xD320).to_bytes(2, "little")),
            }[code[0]][:n]
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
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for p in (
            mock.patch.object(sensors.subprocess, "run", self.fake.run),
            mock.patch.object(sensors.shutil, "which", lambda _n: "/usr/bin/ipmitool"),
            mock.patch.object(sensors.os, "geteuid", lambda: 1000),
            mock.patch.dict(os.environ, {"HSC_I2C_LOCK": str(Path(tmp.name) / "i2c.lock")}),
        ):
            p.start()
            self.addCleanup(p.stop)

    def judge(self, before, keep=""):
        phase = Phase(name="tray_power")
        causes = sensors.check_tray_power(phase, sensors.Bmc(), before, keep)
        return phase, {c.name: c for c in phase.checks if c.name != "tray_power"}, causes


class TestTrayPower(SensorCase):
    def test_clean_trays_pass_and_read_the_bricks(self):
        bmc = sensors.Bmc()
        before = sensors.read_power(bmc)
        self.assertEqual(before.hsc[(2, 0x40)].vin, 12.0)
        self.assertEqual(len(before.bricks), 8)
        self.assertAlmostEqual(before.bricks["U3"][0], 12.0)  # mux channel 1 only answers here
        self.assertEqual(before.bricks["U1"], (None, None))
        self.assertEqual(self.fake.mux, 0x00, "the brick mux is written back")
        self.assertTrue(all(c[:3] == ["sudo", "-n", "ipmitool"] for c in self.fake.calls))
        phase, rest, causes = self.judge(before)
        self.assertEqual([c.status for c in phase.checks if c.name == "tray_power"], [PASS])
        self.assertEqual(rest["hsc_latch_clear"].status, PASS)
        self.assertEqual(causes, {})

    def test_uv_latch_set_during_the_run_is_reported_then_cleared(self):
        before = sensors.read_power(sensors.Bmc())
        self.fake.set(3, 0x42, 0x7C, 0x20)  # VIN_UV_WARN
        self.fake.set(3, 0x42, 0x79, 0x2000)
        phase, rest, causes = self.judge(before)
        (tp,) = [c for c in phase.checks if c.name == "tray_power"]
        self.assertEqual((tp.status, tp.cause), (WARN, "POWER"))
        self.assertIn("UBB3: HSC 0x42: VIN_UV_WARN set during the run", tp.details)
        self.assertEqual(len(tp.parts), 8)
        self.assertTrue(all("(UBB3/U" in p for p in tp.parts))
        self.assertEqual(causes[3][0], "POWER")
        clear = rest["hsc_latch_clear"]
        self.assertEqual(clear.status, PASS)
        self.assertEqual(len(self.fake.cleared), 8, "CLEAR_FAULTS once per HSC")
        self.assertEqual(self.fake.regs[(sensors.HSC_BUS[3], 0x42, 0x7C)], 0)

    def test_latch_already_set_before_the_run_says_so(self):
        self.fake.set(1, 0x40, 0x7C, 0x20)
        before = sensors.read_power(sensors.Bmc())
        phase, _, causes = self.judge(before)
        (tp,) = [c for c in phase.checks if c.name == "tray_power"]
        self.assertIn("VIN_UV_WARN already set before the run", tp.details)
        self.assertEqual((tp.status, tp.cause), (WARN, ""))
        self.assertEqual(causes, {}, "a latch older than the run explains no lost chip")

    def test_latch_that_survives_the_clear_warns(self):
        before = sensors.read_power(sensors.Bmc())
        self.fake.set(4, 0x40, 0x7C, 0x10)
        self.fake.sticky.add((sensors.HSC_BUS[4], 0x40))
        _, rest, _ = self.judge(before)
        self.assertEqual(rest["hsc_latch_clear"].status, WARN)
        self.assertIn("still set: UBB4 0x40 VIN_UV_FAULT", rest["hsc_latch_clear"].details)

    def test_hsc_off_withholds_every_clear(self):
        before = sensors.read_power(sensors.Bmc())
        self.fake.set(2, 0x40, 0x79, 0x0040)
        self.fake.set(2, 0x40, 0x88, 0x0000)
        self.fake.set(1, 0x42, 0x7C, 0x20)
        phase, rest, causes = self.judge(before)
        tps = {c.details[:4]: c for c in phase.checks if c.name == "tray_power"}
        self.assertEqual(tps["UBB2"].status, FAIL)
        self.assertEqual(causes[2][0], "POWER")
        self.assertEqual(rest["hsc_latch_clear"].status, SKIP)
        self.assertIn("skipped: HSC OFF (UBB2 0x40)", rest["hsc_latch_clear"].details)
        self.assertEqual(self.fake.cleared, [])

    def test_no_hsc_clear_keeps_the_latches(self):
        before = sensors.read_power(sensors.Bmc())
        self.fake.set(1, 0x42, 0x7C, 0x20)
        _, rest, _ = self.judge(before, keep="--no-hsc-clear")
        self.assertEqual(rest["hsc_latch_clear"].details, "skipped: --no-hsc-clear")
        self.assertEqual(self.fake.cleared, [])

    def test_vr_power_good_low_with_12v_good_is_downstream(self):
        before = sensors.read_power(sensors.Bmc())
        self.fake.regs[(sensors.HSC_BUS[1], sensors.TRAY_CPLD, sensors.CPLD_VR_PG)] = 0xF7
        phase, _, causes = self.judge(before)
        (tp,) = [c for c in phase.checks if c.name == "tray_power"]
        self.assertEqual((tp.status, tp.cause), (FAIL, "POWER-DOWNSTREAM"))
        self.assertIn("VR power-good 0xf7", causes[1][1])

    def test_sudo_refused_is_one_skipped_line(self):
        self.fake.refuse = "sudo: a password is required\n"
        bmc = sensors.Bmc()
        self.assertIsNone(sensors.read_power(bmc))
        self.assertEqual(len(self.fake.calls), 1, "a refusal ends the route")
        phase, _, causes = self.judge(None)
        (tp,) = phase.checks
        self.assertEqual(tp.status, SKIP)
        self.assertTrue(tp.details.startswith("skipped: no root, and sudo -n ipmitool was refused"))
        self.assertEqual(causes, {})

    def test_no_bmc_device_is_one_skipped_line(self):
        self.fake.refuse = "Could not open device at /dev/ipmi0 or /dev/ipmi/0 or /dev/ipmidev/0\n"
        self.assertIsNone(sensors.read_power(sensors.Bmc()))
        self.assertEqual(len(self.fake.calls), 1)
        phase, _, _ = self.judge(None)
        self.assertTrue(phase.checks[0].details.startswith("skipped: no BMC device"))

    def test_unopenable_lock_is_one_skipped_line(self):
        with mock.patch.dict(os.environ, {"HSC_I2C_LOCK": "/nonexistent-dir/i2c.lock"}):
            bmc = sensors.Bmc()
            self.assertIsNone(sensors.read_power(bmc))
        self.assertEqual(self.fake.calls, [])
        self.assertTrue(bmc.why.startswith("cannot open the I2C lock /nonexistent-dir/i2c.lock"))

    def test_lock_held_elsewhere_skips_the_read(self):
        import fcntl

        with open(os.environ["HSC_I2C_LOCK"], "a") as f, mock.patch.object(sensors, "LOCK_WAIT_S", 0.1):
            fcntl.flock(f, fcntl.LOCK_EX)
            self.assertIsNone(sensors.Bmc().read(0x11, 0x40, sensors.STATUS_WORD))
        self.assertEqual(self.fake.calls, [])

    def test_lock_held_elsewhere_says_why_in_the_skipped_row(self):
        import fcntl

        before = sensors.read_power(sensors.Bmc())
        with open(os.environ["HSC_I2C_LOCK"], "a") as f, mock.patch.object(sensors, "LOCK_WAIT_S", 0.1):
            fcntl.flock(f, fcntl.LOCK_EX)
            phase, _, _ = self.judge(before)
        check = phase.checks[0]
        self.assertEqual((check.status, check.details), (SKIP, "skipped: I2C lock busy after 0.1 s"))


class TestLostTrayCause(SensorCase):
    def test_trip_and_new_flags_come_with_the_cause(self):
        before = sensors.read_power(sensors.Bmc())
        self.fake.set(3, 0x42, 0x7C, 0x20)  # VIN_UV_WARN alone
        self.fake.set(3, 0x42, 0x79, 0x2000)
        self.fake.set(2, 0x40, 0x79, 0x0050)  # OFF + IOUT_OC_FAULT
        causes = sensors.tray_causes(sensors.read_power(sensors.Bmc()), before)
        self.assertEqual((causes[3].trip, causes[3].new), (False, {0x42: ["VIN_UV_WARN"]}))
        self.assertTrue(causes[2].trip)
        self.assertEqual(causes[2].new[0x40], sensors.read_hsc(sensors.Bmc(), 2, 0x40).flags)

    def test_lost_tray_is_judged_power(self):
        before = sensors.read_power(sensors.Bmc())
        self.fake.set(2, 0x40, 0x79, 0x0050)  # OFF + IOUT_OC_FAULT
        power = Phase(name="tray_power")
        causes = sensors.check_tray_power(power, sensors.Bmc(), before, "")
        lost = [f"0000:4{a}:00.0 (UBB2/U{a})" for a in range(1, 9)]
        snap = Phase(name="snapshot")
        snap.add(Check(name="pcie_enum_count", status=FAIL, details="24/32 chips on the bus", ip="pcie", parts=lost))
        for p in (snap, power):
            p.rollup()
        report = {"phases": {p.name: asdict(p) for p in (snap, power)}}
        attribute_power(report, causes)
        row = report["phases"]["snapshot"]["checks"][0]
        self.assertEqual(row["cause"], "POWER")
        self.assertTrue(row["details"].startswith("UBB2 12 V: HSC 0x40"))
        line = next(ln for ln in verdict_lines(report) if "pcie_enum_count" in ln)
        self.assertIn("FAIL CAUSE POWER: pcie_enum_count: UBB2 12 V", line)
        self.assertIn("fix UBB2's 12 V feed", line)

    def test_lost_tray_with_no_power_reading_stays_unattributed(self):
        snap = Phase(name="snapshot")
        snap.add(Check(name="pcie_enum_count", status=FAIL, details="24/32", parts=["0000:46:00.0 (UBB2/U6)"]))
        snap.rollup()
        report = {"phases": {"snapshot": asdict(snap)}}
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

    def test_correctable_rise_at_the_root_port_warns_naming_the_chip(self):
        for bdf in aer.chip_root_ports(self.root):
            self.counts(bdf, "correctable", RxErr=0, BadTLP=0)
        self.counts(root_port("0000:46:00.0"), "correctable", RxErr=3)
        before = aer.read_aer(self.root)
        self.counts(root_port("0000:46:00.0"), "correctable", RxErr=15)
        phase = Phase(name="pcie_aer")
        aer.check_aer(phase, before, aer.read_aer(self.root))
        (c,) = phase.checks
        self.assertEqual(c.status, WARN)
        self.assertEqual(c.details, "0000:46:00.0 (UBB2/U6): root port correctable RxErr +12")
        self.assertEqual(c.parts, ["0000:46:00.0 (UBB2/U6)"])

    def test_uncorrectable_rise_fails(self):
        self.counts(CHIP1, "nonfatal", DLP=0)
        before = aer.read_aer(self.root)
        self.counts(CHIP1, "nonfatal", DLP=1)
        phase = Phase(name="pcie_aer")
        aer.check_aer(phase, before, aer.read_aer(self.root))
        self.assertEqual(phase.checks[0].status, FAIL)
        self.assertIn("chip nonfatal DLP +1", phase.checks[0].details)

    def test_unchanged_passes_and_no_counters_skips(self):
        phase = Phase(name="pcie_aer")
        aer.check_aer(phase, aer.read_aer(self.root), aer.read_aer(self.root))
        self.assertEqual(phase.checks[0].status, SKIP)
        self.counts(CHIP1, "correctable", RxErr=2)
        phase = Phase(name="pcie_aer")
        aer.check_aer(phase, aer.read_aer(self.root), aer.read_aer(self.root))
        self.assertEqual(phase.checks[0].status, PASS)


if __name__ == "__main__":
    unittest.main()
