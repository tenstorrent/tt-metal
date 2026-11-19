#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for diag/tray_power.py's tray 12 V feed phase: the hot-swap
controller (HSC) latch decode, the CPLD tray-alive verdict, the read-only gate
on every ipmitool transaction, and the phase's time and no-route bounds. A fake
BMC stands in for ipmitool, so no hardware is needed.

The HSC vectors are latch values read off Blackhole Galaxy trays: a tray that
dropped on a hot-swap over-current trip, a feed that sagged below the warning
limit without dropping the tray, and clean feeds. VIN_UV_FAULT has not been
seen latched, so its vector is synthetic."""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

import diag.tray_power as tray_power  # noqa: E402
from diag.model import FAIL, PASS, SKIP, WARN, Check, Phase  # noqa: E402
from diag.tray_power import (  # noqa: E402
    CALL_TIMEOUT_S,
    PHASE_TIMEOUT_S,
    READS,
    TRAY_BUS,
    TrayPowerReader,
    hsc_faults,
    run_tray_power,
    tray_bdf_range,
)
import diag_runner  # noqa: E402
from diag_runner import build_diag_arg_parser  # noqa: E402
from tiers import TIERS  # noqa: E402

CLEAN_HSC = {"word": 0x0000, "input": 0x00, "iout": 0x00}
UV_WARN = {"word": 0x2002, "input": 0x20, "iout": 0x00}
OC_TRIP = {"word": 0x4852, "input": 0x00, "iout": 0x80}  # OFF + IOUT_OC after a hot-swap trip
CALLS_PER_TRAY = 8  # 2 HSCs x 3 status reads, then CPLD reg 0x1E until two reads agree
ALL_DEVS = (0x40, 0x42, 0x80)


def txn_of(argv: list[str]) -> tuple[int, int, int, int]:
    """(bus, addr, nbytes, code) of one `ipmitool raw 0x06 0x52 ...` argv."""
    i = argv.index("raw")
    assert argv[i + 1 : i + 3] == ["0x06", "0x52"], argv
    bus, addr, nbytes, code = (int(x, 16) for x in argv[i + 3 :])
    return bus, addr, nbytes, code


class FakeBmc:
    """Answers the phase's reads from per-tray HSC and CPLD values; records every call."""

    def __init__(self, hsc=None, cpld=None, fail=(), stderr_prefix=""):
        self.hsc = hsc or {}  # {(tray, addr): {"word", "input", "iout"}}
        self.cpld = cpld or {}  # {tray: reg 0x1E}
        self.fail = set(fail)  # (tray, addr) whose reads all error out
        self.stderr_prefix = stderr_prefix  # e.g. a sudo warning line before ipmitool's error
        self.calls: list[list[str]] = []
        self.tray_of_bus = {bus: tray for tray, bus in TRAY_BUS.items()}

    def __call__(self, argv):
        self.calls.append(list(argv))
        bus, addr, _, code = txn_of(argv)
        tray = self.tray_of_bus[bus]
        if (tray, addr) in self.fail:
            err = self.stderr_prefix + "Unable to send RAW command\n"
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr=err)
        if addr == 0x80:
            out = f" {self.cpld.get(tray, 0xFF):02x}\n"
        else:
            st = self.hsc.get((tray, addr), CLEAN_HSC)
            v = {0x79: st["word"], 0x7C: st["input"], 0x7B: st["iout"]}[code]
            out = f" {v & 0xFF:02x} {v >> 8:02x}\n" if code == 0x79 else f" {v:02x}\n"
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")


def run_phase(bmc, dry_run=False, clock=None) -> Phase:
    phase = Phase(name="tray_power", gates=False)
    kw = {"clock": clock} if clock else {}
    run_tray_power(phase, dry_run, runner=bmc, **kw)
    return phase


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def by_tray(phase: Phase) -> dict[int, Check]:
    return {c.data["tray"]: c for c in phase.checks}


def statuses(checks: dict[int, Check]) -> list[str]:
    return [checks[t].status for t in (1, 2, 3, 4)]


def refusing(err: str, calls: list):
    def runner(argv):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr=err)

    return runner


class TestDecodeHsc(unittest.TestCase):
    def test_faults(self):
        self.assertEqual(hsc_faults(0x0000, 0x00, 0x00), [])
        self.assertEqual(hsc_faults(0x0002, 0x00, 0x00), [])  # CML alone is not a fault
        self.assertEqual(hsc_faults(0x2002, 0x20, 0x00), ["VIN_UV_WARN"])
        self.assertEqual(hsc_faults(0x2008, 0x10, 0x00), ["VIN_UV_FAULT"])
        self.assertEqual(hsc_faults(0x4852, 0x00, 0x80), ["OFF", "IOUT_OC"])

    def test_iout_oc_from_status_iout_alone(self):
        self.assertEqual(hsc_faults(0x0000, 0x00, 0x80), ["IOUT_OC"])
        self.assertEqual(hsc_faults(0x0010, 0x00, None), ["IOUT_OC"])


class TestTrayPowerPhase(unittest.TestCase):
    def assert_read_only(self, bmc: FakeBmc):
        """Every call is one allowed read on a tray bus; nothing is written."""
        self.assertTrue(bmc.calls)
        for argv in bmc.calls:
            bus, addr, nbytes, code = txn_of(argv)
            self.assertIn(bus, TRAY_BUS.values())
            self.assertEqual(READS.get((addr, code)), nbytes, argv)

    def test_clean_passes(self):
        bmc = FakeBmc(hsc={(t, 0x42): dict(CLEAN_HSC, word=0x0002) for t in TRAY_BUS})
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            phase = run_phase(bmc)
        self.assertNotIn("tray_power: stopped", err.getvalue())
        phase.rollup()
        self.assertEqual([c.status for c in phase.checks], [PASS] * 4)
        self.assertEqual(phase.status, PASS)
        self.assertEqual(len(bmc.calls), 4 * CALLS_PER_TRAY)
        self.assert_read_only(bmc)

    def test_vin_uv_warn_fails_naming_tray_bdf_hsc_and_hint(self):
        checks = by_tray(run_phase(FakeBmc(hsc={(2, 0x42): UV_WARN})))
        self.assertEqual(statuses(checks), [PASS, FAIL, PASS, PASS])
        for s in ("tray 2 (41:00.0-48:00.0)", "POWER", "bus bar / tray 12 V feed", "HSC 0x42: VIN_UV_WARN"):
            self.assertIn(s, checks[2].details)
        self.assertIn("STATUS_WORD 0x2002, STATUS_INPUT 0x20, STATUS_IOUT 0x00", checks[2].details)
        self.assertEqual(checks[2].data["cause"], "POWER")
        self.assertEqual(checks[2].data["hsc"]["0x42"]["faults"], ["VIN_UV_WARN"])

    def test_tray_down_with_oc_trip_is_power(self):
        checks = by_tray(run_phase(FakeBmc(hsc={(3, 0x40): OC_TRIP}, cpld={3: 0x00})))
        c = checks[3]
        self.assertEqual(c.status, FAIL)
        self.assertEqual(c.data["cause"], "POWER")
        for s in ("tray 3 (c1:00.0-c8:00.0)", "HSC 0x40: OFF, IOUT_OC latched", "CPLD reg 0x1E=0x00: tray down"):
            self.assertIn(s, c.details)
        self.assertEqual(c.data["cpld_reg1e"], 0x00)

    def test_tray_down_with_clean_latches_is_chip_board(self):
        checks = by_tray(run_phase(FakeBmc(cpld={4: 0x00})))
        c = checks[4]
        self.assertEqual(c.status, WARN)
        self.assertEqual(c.data["cause"], "CHIP/BOARD")
        self.assertIn("CHIP/BOARD", c.details)
        self.assertNotIn("bus bar", c.details)
        self.assertEqual(statuses(checks)[:3], [PASS] * 3)

    def test_cpld_not_settled_is_not_read(self):
        # Reg 0x1E never reads the same twice in a row: not a verdict either way.
        bmc = FakeBmc()
        flip = iter([0xFF, 0x00] * 2)

        def runner(argv):
            cp = bmc(argv)
            if txn_of(argv)[:2] == (TRAY_BUS[1], 0x80):
                return subprocess.CompletedProcess(argv, 0, stdout=f" {next(flip):02x}\n", stderr="")
            return cp

        checks = by_tray(run_phase(runner))
        self.assertEqual(checks[1].status, WARN)
        self.assertIn("not read: CPLD", checks[1].details)
        self.assertIsNone(checks[1].data["cause"])

    def test_tray_down_with_an_hsc_unread_names_the_tray_and_no_cause(self):
        checks = by_tray(run_phase(FakeBmc(cpld={2: 0x00}, fail={(2, 0x42)})))
        c = checks[2]
        self.assertEqual((c.status, c.data["cause"]), (WARN, None))
        self.assertTrue(c.details.startswith("tray 2 (41:00.0-48:00.0): CPLD reg 0x1E=0x00: tray down"), c.details)
        self.assertIn("not read: HSC 0x42", c.details)

    def test_read_error_beside_a_fault_still_fails(self):
        checks = by_tray(run_phase(FakeBmc(hsc={(2, 0x40): UV_WARN}, fail={(2, 0x42)})))
        self.assertEqual(checks[2].status, FAIL)
        self.assertIn("not read: HSC 0x42 (Unable to send RAW command)", checks[2].details)

    def test_no_route_is_one_skip(self):
        phase = run_phase(FakeBmc(fail={(t, a) for t in TRAY_BUS for a in ALL_DEVS}))
        self.assertEqual([(c.name, c.status) for c in phase.checks], [("tray_12v_feed", SKIP)])
        self.assertIn("Unable to send RAW command", phase.checks[0].details)

    def test_dead_trays_do_not_hide_the_others(self):
        for dead, faulty in (((1,), 3), ((1, 2, 3), 4)):
            bmc = FakeBmc(hsc={(faulty, 0x40): UV_WARN}, fail={(t, a) for t in dead for a in ALL_DEVS})
            checks = by_tray(run_phase(bmc))
            self.assertEqual(checks[faulty].status, FAIL)
            for t in dead:
                self.assertEqual(checks[t].status, WARN)
                self.assertIn("not read: HSC 0x40, HSC 0x42, CPLD", checks[t].details)
            self.assert_read_only(bmc)

    def test_sudo_refusal_is_one_skip_after_one_call(self):
        for err in (
            "sudo: a password is required\n",
            "sudo: unable to resolve host h: Name or service not known\nsudo: a password is required\n",
            "sudo: a terminal is required to read the password\n",
            "user is not in the sudoers file.  This incident will be reported.\n",
            "Sorry, user user is not allowed to execute '/usr/bin/ipmitool raw' as root on h.\n",
            "sudo: ipmitool: command not found\n",
        ):
            calls = []
            phase = run_phase(refusing(err, calls))
            self.assertEqual([c.status for c in phase.checks], [SKIP], err)
            self.assertEqual(len(calls), 1, err)

    def test_missing_ipmitool_is_one_skip_after_one_call(self):
        calls = []

        def runner(argv):
            calls.append(argv)
            raise FileNotFoundError("ipmitool")

        self.assertEqual([c.status for c in run_phase(runner).checks], [SKIP])
        self.assertEqual(len(calls), 1)

    def test_sudo_warning_line_does_not_hide_other_trays(self):
        # sudo warns on the first line and still runs ipmitool; tray 1's bus NACKs.
        for warn in (
            "sudo: unable to resolve host h: Name or service not known\n",
            "sudo: setrlimit(RLIMIT_CORE): Operation not permitted\n",
        ):
            fail = {(1, a) for a in ALL_DEVS}
            checks = by_tray(run_phase(FakeBmc(hsc={(3, 0x40): UV_WARN}, fail=fail, stderr_prefix=warn)))
            self.assertEqual(statuses(checks), [WARN, PASS, FAIL, PASS], warn)
            self.assertIn("Unable to send RAW command", checks[1].details, warn)
            self.assertNotIn("sudo:", checks[1].details, warn)

    def test_hung_tray_1_does_not_hide_tray_3_fault(self):
        clock = FakeClock()
        bmc = FakeBmc(hsc={(3, 0x40): UV_WARN})

        def runner(argv):
            if txn_of(argv)[0] == TRAY_BUS[1]:
                clock.t += CALL_TIMEOUT_S
                raise subprocess.TimeoutExpired(argv, CALL_TIMEOUT_S)
            return bmc(argv)

        checks = by_tray(run_phase(runner, clock=clock))
        self.assertEqual(statuses(checks), [WARN, PASS, FAIL, PASS])
        self.assertIn(f"no reply in {CALL_TIMEOUT_S:g} s", checks[1].details)

    def test_only_the_bmc_reboot_tiers_run_the_phase_and_dry_run_sends_nothing(self):
        self.assertEqual(
            [name for name, tier in TIERS.items() if tier.tray_power], ["pre_bmc_reboot", "post_bmc_reboot"]
        )
        bmc = FakeBmc()
        self.assertEqual([c.status for c in run_phase(bmc, dry_run=True).checks], [SKIP])
        self.assertEqual(bmc.calls, [])

    def test_timeouts_everywhere_stop_at_once_as_one_skip(self):
        calls = []

        def runner(argv):
            calls.append(argv)
            raise subprocess.TimeoutExpired(argv, CALL_TIMEOUT_S)

        phase = run_phase(runner)
        self.assertEqual(len(calls), 2 * len(TRAY_BUS))
        self.assertEqual([(c.name, c.status) for c in phase.checks], [("tray_12v_feed", SKIP)])
        self.assertIn("no reply", phase.checks[0].details)

    def test_deadline_mid_run_leaves_the_rest_not_read(self):
        clock = FakeClock()
        bmc = FakeBmc()

        def runner(argv):
            cp = bmc(argv)
            if len(bmc.calls) == 2 * CALLS_PER_TRAY:  # trays 1 and 2 read
                clock.t = PHASE_TIMEOUT_S
            return cp

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            checks = by_tray(run_phase(runner, clock=clock))
        stop = f"tray_power: stopped reading: phase time limit of {PHASE_TIMEOUT_S:g} s spent"
        self.assertIn(stop, err.getvalue())
        self.assertEqual(len(bmc.calls), 2 * CALLS_PER_TRAY)
        self.assertEqual(statuses(checks), [PASS, PASS, WARN, WARN])
        for t in (3, 4):
            self.assertIn("not read: HSC 0x40, HSC 0x42, CPLD (phase time limit", checks[t].details)

    def test_call_timeout_never_outlasts_the_phase(self):
        clock = FakeClock()
        reader = TrayPowerReader(clock=clock)
        seen = {}

        def fake_run(argv, **kw):
            seen.update(kw)
            return subprocess.CompletedProcess(argv, 0, stdout=" 00 00\n", stderr="")

        clock.t = PHASE_TIMEOUT_S - 4
        with mock.patch.object(tray_power, "run", fake_run):
            self.assertEqual(reader.txn(TRAY_BUS[1], 0x40, 0x79), 0)
        self.assertEqual(seen["timeout"], 4)

    def test_gate_refuses_anything_but_the_status_reads(self):
        bmc = FakeBmc()
        reader = TrayPowerReader(bmc)
        refused = [
            (0x11, 0x40, 0x03),  # CLEAR_FAULTS
            (0x11, 0x40, 0x01),  # OPERATION
            (0x11, 0x42, 0xA6),  # READ_IOUT_PEAK, clears on read
            (0x11, 0x42, 0xAF),  # READ_TEMPERATURE_PEAK, clears on read
            (0x11, 0x40, 0x8B),  # READ_VOUT
            (0x11, 0x40, 0x46),  # IOUT_OC_FAULT_LIMIT
            (0x11, 0x40, 0xFA),  # manufacturer fault record
            (0x11, 0x80, 0x50),  # CPLD debug index select
            (0x11, 0x80, 0x5F),  # any other CPLD register
            (0x11, 0x44, 0x79),  # not an HSC
            (0x12, 0x40, 0x79),  # not a tray bus
        ]
        for bus, addr, code in refused:
            with self.assertRaises(ValueError):
                reader.txn(bus, addr, code)
        self.assertEqual(bmc.calls, [])


class TestPlumbing(unittest.TestCase):
    def test_bdf_ranges(self):
        self.assertEqual(
            [tray_bdf_range(t) for t in (1, 2, 3, 4)],
            ["01:00.0-08:00.0", "41:00.0-48:00.0", "c1:00.0-c8:00.0", "81:00.0-88:00.0"],
        )

    def test_skip_flag(self):
        args = build_diag_arg_parser().parse_args(["--tier", "pre_bmc_reboot", "--skip-tray-power"])
        self.assertTrue(args.skip_tray_power)


class TestRunDiagWiring(unittest.TestCase):
    """diag_runner.run_diag() with every other phase mocked out: where the
    tray_power phase sits in the report, what --skip-tray-power and a crashing
    reader leave behind, and which tiers run it."""

    def run_diag(self, tier, reader=None, **kwargs):
        import json
        import tempfile

        def noop(*_a, **_k):
            return None

        reader = reader or noop
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / "snap.json"
            snap.write_text("{}")
            patches = [
                mock.patch.object(diag_runner, "validate_snapshot", return_value=None),
                mock.patch.object(diag_runner, "collect_fru_info", noop),
                mock.patch.object(diag_runner, "resolve_tt_smi", return_value="tt-smi"),
                mock.patch.object(diag_runner, "reset_loop", noop),
                mock.patch.object(diag_runner, "run_post_test_reset", noop),
                mock.patch.object(diag_runner, "run_tests", noop),
                mock.patch.object(diag_runner, "run_triage", noop),
                mock.patch.object(diag_runner, "run_qsfp_tests", noop),
                mock.patch.object(diag_runner, "run_sys_triage", noop),
                mock.patch.object(diag_runner, "run_eth_status", noop),
                mock.patch.object(diag_runner, "run_tray_power", side_effect=reader),
            ]
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                tray_mock = diag_runner.run_tray_power
                code, report = diag_runner.run_diag(
                    tier, input_snapshot=snap, output=Path(tmp) / "report.json", dry_run=True, **kwargs
                )
            self.assertEqual(
                json.loads((Path(tmp) / "report.json").read_text())["phases"].keys(), report["phases"].keys()
            )
        return code, report, tray_mock

    def test_pre_bmc_reboot_runs_tray_power_right_after_snapshot_before_resets(self):
        def reader(phase, dry_run):
            phase.add(Check(name="tray1_12v_feed", status=PASS, details="clean", ip="board"))

        code, report, tray_mock = self.run_diag("pre_bmc_reboot", reader)
        names = list(report["phases"])
        self.assertEqual(names[:2], ["snapshot", "tray_power"])
        self.assertLess(names.index("tray_power"), names.index("reset_loop"))
        tray_mock.assert_called_once()
        self.assertTrue(tray_mock.call_args.args[1])  # dry_run is passed through
        phase = report["phases"]["tray_power"]
        self.assertEqual((phase["error"], phase["status"], phase["gates"]), ("", PASS, False))

    def test_skip_flag_gives_one_skip_check_and_the_phase_stays_skip(self):
        code, report, tray_mock = self.run_diag("pre_bmc_reboot", skip_tray_power=True)
        tray_mock.assert_not_called()
        phase = report["phases"]["tray_power"]
        self.assertEqual(
            [(c["name"], c["status"], c["details"]) for c in phase["checks"]],
            [("tray_12v_feed", SKIP, "--skip-tray-power")],
        )
        self.assertEqual(phase["status"], SKIP)

    def test_reader_exception_becomes_one_skip_check_with_error_set(self):
        def boom(phase, dry_run):
            raise RuntimeError("bmc went away")

        code, report, _ = self.run_diag("pre_bmc_reboot", boom)
        phase = report["phases"]["tray_power"]
        self.assertEqual([(c["name"], c["status"]) for c in phase["checks"]], [("tray_12v_feed", SKIP)])
        self.assertIn("bmc went away", phase["error"])
        self.assertIn("bmc went away", phase["checks"][0]["details"])
        self.assertEqual((phase["status"], code), (SKIP, 0))

    def test_other_tiers_have_no_tray_power_phase(self):
        for tier in TIERS:
            if TIERS[tier].tray_power:
                continue
            with self.subTest(tier=tier):
                _, report, tray_mock = self.run_diag(tier)
                self.assertNotIn("tray_power", report["phases"])
                tray_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
