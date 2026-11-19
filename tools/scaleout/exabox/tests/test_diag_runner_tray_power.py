#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""diag/tray_power.py's tray 12 V feed phase against a fake BMC: HSC latch decode, CPLD tray-alive
verdict, the read-only allowlist, time and no-route bounds, and the wiring into run_diag(). The HSC
vectors are latch values read off Blackhole Galaxy trays, except VIN_UV_FAULT, which is synthetic."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from itertools import cycle
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "health_check_test_suite"))

import diag.tray_power as tray_power  # noqa: E402
import diag_runner  # noqa: E402
from diag.model import FAIL, PASS, SKIP, WARN, Check, Phase  # noqa: E402
from diag.tray_power import CALL_TIMEOUT_S, PHASE_TIMEOUT_S, READS, TRAY_BUS, TrayPowerReader, hsc_faults  # noqa: E402
from tiers import TIERS  # noqa: E402

CLEAN = {"word": 0x0002, "input": 0x00, "iout": 0x00}  # CML alone is not a fault
UV_WARN = {"word": 0x2002, "input": 0x20, "iout": 0x00}
OC_TRIP = {"word": 0x4852, "input": 0x00, "iout": 0x80}  # OFF + IOUT_OC after a hot-swap trip
CALLS_PER_TRAY = 8  # 2 HSCs x 3 status reads, then CPLD reg 0x1E until two reads agree
ALL_DEVS = (0x40, 0x42, 0x80)
NACK = "Unable to send RAW command"
STATUS = {"P": PASS, "F": FAIL, "W": WARN}


def dead(*trays):
    return {(t, a) for t in trays for a in ALL_DEVS}


def txn_of(argv):
    """(bus, addr, nbytes, code) of one `ipmitool raw 0x06 0x52 ...` argv."""
    i = argv.index("raw")
    assert argv[i + 1 : i + 3] == ["0x06", "0x52"], argv
    return tuple(int(x, 16) for x in argv[i + 3 :])


class FakeBmc:
    """Answers the reads from per-tray HSC and CPLD values and records every call.
    A CPLD value may be a list, read back in turn."""

    def __init__(self, hsc=None, cpld=None, fail=(), stderr_prefix=""):
        self.hsc = hsc or {}  # {(tray, addr): {"word", "input", "iout"}}
        self.cpld = {t: cycle(v if isinstance(v, list) else [v]) for t, v in (cpld or {}).items()}
        self.fail = set(fail)  # (tray, addr) whose reads all error out
        self.stderr_prefix = stderr_prefix  # e.g. a sudo warning line before ipmitool's error
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)
        bus, addr, _, code = txn_of(argv)
        tray = next(t for t, b in TRAY_BUS.items() if b == bus)
        if (tray, addr) in self.fail:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr=f"{self.stderr_prefix}{NACK}\n")
        if addr == 0x80:
            out = f" {next(self.cpld.get(tray, iter([0xFF]))):02x}"
        else:
            st = self.hsc.get((tray, addr), CLEAN)
            v = {0x79: st["word"], 0x7C: st["input"], 0x7B: st["iout"]}[code]
            out = f" {v & 0xFF:02x} {v >> 8:02x}" if code == 0x79 else f" {v:02x}"
        return subprocess.CompletedProcess(argv, 0, stdout=out + "\n", stderr="")


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def run_phase(runner, dry_run=False, clock=None) -> tuple[Phase, str]:
    """The phase and what it logged."""
    phase = Phase(name="tray_power", gates=False)
    with contextlib.redirect_stderr(io.StringIO()) as err:
        tray_power.run_tray_power(phase, dry_run, runner=runner, clock=clock or Clock())
    return phase, err.getvalue()


def by_tray(phase):
    checks = {c.data["tray"]: c for c in phase.checks}
    return [checks[t].status for t in (1, 2, 3, 4)], checks


class TestTrayPower(unittest.TestCase):
    def test_hsc_faults(self):
        for word, inp, iout, want in (
            (0x0000, 0x00, 0x00, []),
            (0x0002, 0x00, 0x00, []),
            (0x2002, 0x20, 0x00, ["VIN_UV_WARN"]),
            (0x2008, 0x10, 0x00, ["VIN_UV_FAULT"]),
            (0x4852, 0x00, 0x80, ["OFF", "IOUT_OC"]),
            (0x0000, 0x00, 0x80, ["IOUT_OC"]),  # STATUS_IOUT alone
            (0x0010, 0x00, None, ["IOUT_OC"]),  # STATUS_IOUT not read
        ):
            self.assertEqual(hsc_faults(word, inp, iout), want, hex(word))

    def test_verdicts(self):
        # (case, FakeBmc kwargs, statuses of trays 1-4, tray to look at, its cause, strings in its details)
        sudo_warn = "sudo: unable to resolve host h: Name or service not known\n"
        unread = "not read: HSC 0x40, HSC 0x42, CPLD"
        # fmt: off
        cases = [
            ("clean", {}, "PPPP", 1, None, ["tray 1 (01:00.0-08:00.0): no 12 V latch set, tray up"]),
            ("under-voltage warning", {"hsc": {(2, 0x42): UV_WARN}}, "PFPP", 2, "POWER",
             ["tray 2 (41:00.0-48:00.0): POWER: check bus bar / tray 12 V feed", "HSC 0x42: VIN_UV_WARN latched",
              "STATUS_WORD 0x2002, STATUS_INPUT 0x20, STATUS_IOUT 0x00"]),
            ("over-current trip, tray down", {"hsc": {(3, 0x40): OC_TRIP}, "cpld": {3: 0x00}}, "PPFP", 3, "POWER",
             ["tray 3 (c1:00.0-c8:00.0)", "HSC 0x40: OFF, IOUT_OC latched", "CPLD reg 0x1E=0x00: tray down"]),
            ("tray down, clean latches", {"cpld": {4: 0x00}}, "PPPW", 4, "CHIP/BOARD",
             ["tray 4 (81:00.0-88:00.0): CHIP/BOARD: tray down with no 12 V feed latch set, not the feed"]),
            ("CPLD never settles", {"cpld": {1: [0xFF, 0x00]}}, "WPPP", 1, None, ["not read: CPLD"]),
            ("tray down, an HSC unread", {"cpld": {2: 0x00}, "fail": {(2, 0x42)}}, "PWPP", 2, None,
             ["tray 2 (41:00.0-48:00.0): CPLD reg 0x1E=0x00: tray down", f"not read: HSC 0x42 ({NACK})"]),
            ("fault beside a read error", {"hsc": {(2, 0x40): UV_WARN}, "fail": {(2, 0x42)}}, "PFPP", 2, "POWER",
             [f"not read: HSC 0x42 ({NACK})"]),
            ("dead tray 1", {"hsc": {(3, 0x40): UV_WARN}, "fail": dead(1)}, "WPFP", 1, None, [unread]),
            ("three dead trays", {"hsc": {(4, 0x40): UV_WARN}, "fail": dead(1, 2, 3)}, "WWWF", 3, None, [unread]),
            ("sudo warning line", {"hsc": {(3, 0x40): UV_WARN}, "fail": dead(1), "stderr_prefix": sudo_warn},
             "WPFP", 1, None, [NACK]),
        ]
        # fmt: on
        for name, kw, want, tray, cause, texts in cases:
            with self.subTest(name):
                bmc = FakeBmc(**kw)
                phase, log = run_phase(bmc)
                got, checks = by_tray(phase)
                self.assertEqual(got, [STATUS[c] for c in want])
                self.assertEqual(checks[tray].data["cause"], cause)
                for s in texts:
                    self.assertIn(s, checks[tray].details)
                self.assertNotIn("sudo:", checks[tray].details)
                self.assertNotIn("stopped reading", log)
                for argv in bmc.calls:  # reads only, from the allowlist
                    bus, addr, nbytes, code = txn_of(argv)
                    self.assertIn(bus, TRAY_BUS.values())
                    self.assertEqual(READS.get((addr, code)), nbytes, argv)
                if name == "clean":
                    self.assertEqual(len(bmc.calls), 4 * CALLS_PER_TRAY)

    def test_no_route_is_one_skip(self):
        def raising(exc):
            def runner(argv):
                raise exc(argv)

            return runner

        def refusing(err):
            return lambda argv: subprocess.CompletedProcess(argv, 1, stdout="", stderr=err)

        cases = [  # (case, runner, calls sent, text in the SKIP's details)
            ("every read NACKed", FakeBmc(fail=dead(1, 2, 3, 4)), 4 * (6 * 2 + 4), NACK),
            ("ipmitool missing", raising(FileNotFoundError), 1, "FileNotFoundError"),
            ("every bus hangs", raising(lambda a: subprocess.TimeoutExpired(a, CALL_TIMEOUT_S)), 2 * 4, "no reply"),
        ] + [
            (err, refusing(err + "\n"), 1, err.splitlines()[-1])
            for err in (
                "sudo: a password is required",
                "sudo: unable to resolve host h: Name or service not known\nsudo: a password is required",
                "sudo: a terminal is required to read the password",
                "user is not in the sudoers file.  This incident will be reported.",
                "Sorry, user user is not allowed to execute '/usr/bin/ipmitool raw' as root on h.",
                "sudo: ipmitool: command not found",
            )
        ]
        for name, runner, ncalls, text in cases:
            with self.subTest(name):
                calls = []
                phase, _ = run_phase(lambda argv: calls.append(argv) or runner(argv))
                self.assertEqual([(c.name, c.status) for c in phase.checks], [("tray_12v_feed", SKIP)])
                self.assertIn(text, phase.checks[0].details)
                self.assertEqual(len(calls), ncalls)

        bmc = FakeBmc()
        self.assertEqual([c.status for c in run_phase(bmc, dry_run=True)[0].checks], [SKIP])
        self.assertEqual(bmc.calls, [])

    def test_time_bounds(self):
        # A bus that hangs costs one call timeout and hides no other tray's fault.
        clock, bmc = Clock(), FakeBmc(hsc={(3, 0x40): UV_WARN})

        def hang_tray_1(argv):
            if txn_of(argv)[0] == TRAY_BUS[1]:
                clock.t += CALL_TIMEOUT_S
                raise subprocess.TimeoutExpired(argv, CALL_TIMEOUT_S)
            return bmc(argv)

        got, checks = by_tray(run_phase(hang_tray_1, clock=clock)[0])
        self.assertEqual(got, [WARN, PASS, FAIL, PASS])
        self.assertIn(f"no reply in {CALL_TIMEOUT_S:g} s", checks[1].details)

        # Once the phase time is spent, the trays left are reported not read.
        clock, bmc = Clock(), FakeBmc()

        def spend_after_tray_2(argv):
            cp = bmc(argv)
            if len(bmc.calls) == 2 * CALLS_PER_TRAY:
                clock.t = PHASE_TIMEOUT_S
            return cp

        phase, log = run_phase(spend_after_tray_2, clock=clock)
        self.assertIn(f"stopped reading: phase time limit of {PHASE_TIMEOUT_S:g} s spent", log)
        self.assertEqual(len(bmc.calls), 2 * CALLS_PER_TRAY)
        got, checks = by_tray(phase)
        self.assertEqual(got, [PASS, PASS, WARN, WARN])
        self.assertIn("not read: HSC 0x40, HSC 0x42, CPLD (phase time limit", checks[4].details)

        # A call never gets more time than the phase has left.
        clock, seen = Clock(), {}
        reader = TrayPowerReader(clock=clock)  # the phase clock starts here
        clock.t = PHASE_TIMEOUT_S - 4
        with mock.patch.object(tray_power, "run", lambda argv, **kw: seen.update(kw) or FakeBmc()(argv)):
            self.assertEqual(reader.txn(TRAY_BUS[1], 0x40, 0x79), 0x0002)
        self.assertEqual(seen["timeout"], 4)

    def test_allowlist_refuses_everything_else(self):
        bmc = FakeBmc()
        reader = TrayPowerReader(bmc)
        for bus, addr, code in (
            (0x11, 0x40, 0x03),  # CLEAR_FAULTS
            (0x11, 0x40, 0x01),  # OPERATION
            (0x11, 0x42, 0xA6),  # READ_IOUT_PEAK, clears on read
            (0x11, 0x40, 0x46),  # IOUT_OC_FAULT_LIMIT
            (0x11, 0x80, 0x50),  # CPLD debug index select
            (0x11, 0x44, 0x79),  # not an HSC
            (0x12, 0x40, 0x79),  # not a tray bus
        ):
            with self.assertRaises(ValueError):
                reader.txn(bus, addr, code)
        self.assertEqual(bmc.calls, [])


class TestRunDiagWiring(unittest.TestCase):
    """run_diag() with every other phase mocked out."""

    def run_diag(self, tier, reader=None, **kwargs):
        noop = mock.Mock(return_value=None)
        others = "collect_fru_info reset_loop run_post_test_reset run_tests run_triage run_qsfp_tests"
        others += " run_sys_triage run_eth_status validate_snapshot"
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            for name in others.split():
                stack.enter_context(mock.patch.object(diag_runner, name, noop))
            tray = stack.enter_context(mock.patch.object(diag_runner, "run_tray_power", side_effect=reader))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            snap = Path(tmp) / "snap.json"
            snap.write_text("{}")
            code, report = diag_runner.run_diag(
                tier, input_snapshot=snap, output=Path(tmp) / "report.json", dry_run=True, **kwargs
            )
            written = json.loads((Path(tmp) / "report.json").read_text())
            self.assertEqual(written["phases"].keys(), report["phases"].keys())
        return code, report, tray

    def test_runs_right_after_snapshot_on_the_bmc_reboot_tiers_only(self):
        def reader(phase, dry_run):
            phase.add(Check(name="tray1_12v_feed", status=PASS, details="clean", ip="board"))

        for tier in TIERS:
            with self.subTest(tier):
                _, report, tray = self.run_diag(tier, reader)
                names = list(report["phases"])
                if tier not in ("pre_bmc_reboot", "post_bmc_reboot"):
                    self.assertNotIn("tray_power", names)
                    tray.assert_not_called()
                    continue
                self.assertEqual(names[:2], ["snapshot", "tray_power"])
                self.assertLess(names.index("tray_power"), names.index("reset_loop"))
                self.assertTrue(tray.call_args.args[1])  # dry_run is passed through
                phase = report["phases"]["tray_power"]
                self.assertEqual((phase["error"], phase["status"], phase["gates"]), ("", PASS, False))

    def test_skip_flag_and_a_crash_each_leave_one_skip(self):
        args = diag_runner.build_diag_arg_parser().parse_args(["--tier", "pre_bmc_reboot", "--skip-tray-power"])
        self.assertTrue(args.skip_tray_power)
        _, report, tray = self.run_diag("pre_bmc_reboot", skip_tray_power=True)
        tray.assert_not_called()
        phase = report["phases"]["tray_power"]
        self.assertEqual(
            [(c["name"], c["status"], c["details"]) for c in phase["checks"]],
            [("tray_12v_feed", SKIP, "--skip-tray-power")],
        )
        self.assertEqual(phase["status"], SKIP)

        code, report, _ = self.run_diag("pre_bmc_reboot", RuntimeError("bmc went away"))
        phase = report["phases"]["tray_power"]
        self.assertEqual([(c["name"], c["status"]) for c in phase["checks"]], [("tray_12v_feed", SKIP)])
        self.assertIn("bmc went away", phase["error"])
        self.assertEqual((phase["status"], code), (SKIP, 0))


if __name__ == "__main__":
    unittest.main()
