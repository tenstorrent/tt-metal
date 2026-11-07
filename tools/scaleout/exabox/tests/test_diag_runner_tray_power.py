#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for diag_runner.py's tray 12 V feed phase: the hot-swap
controller (HSC) and tray CPLD latch decode, the per-tray verdict, and the
read-only gate on every ipmitool transaction. A fake BMC stands in for
ipmitool, so no hardware is needed.

The HSC vectors are latch values read off Blackhole Galaxy trays: a tray that
dropped on a hot-swap over-current trip, a tray that lost its 12 V input, a
feed that sagged below the warning limit without dropping the tray, and clean
feeds. VIN_UV_FAULT has not been seen latched, so its vector is synthetic."""

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

import diag_runner  # noqa: E402
from diag_runner import (  # noqa: E402
    FAIL,
    PASS,
    SKIP,
    TRAY_POWER_ALLOWED,
    TRAY_POWER_BUS,
    TRAY_POWER_CALL_S,
    TRAY_POWER_PHASE_S,
    WARN,
    Phase,
    TrayPowerReader,
    build_diag_arg_parser,
    decode_cpld_power,
    decode_hsc_record,
    decode_hsc_status,
    hsc_record_warnings,
    run_tray_power,
    tray_bdf_range,
)

CLEAN_HSC = {"word": 0x0000, "input": 0x00, "iout": 0x00}
# SYS-5505 record words of a healthy feed, as hand-read on Blackhole Galaxy trays: no
# FAULT_RECORD, FAULT_RECORD_LAST 0x0012, 12.22 V, IOUT peak 184.0 A, 43 C,
# IOUT OC fault/warn limits 228 A / 219 A.
CLEAN_REC = {0xFA: 0x0000, 0xFB: 0x0012, 0x8B: 0x0187, 0xA6: 2944, 0xAF: 43, 0x46: 3648, 0x4A: 3504}
CLEAN_CPLD = {"reg1e": "ff", "dbg5f": "7c", "dbg58": "fa"}


def txn_of(argv: list[str]) -> tuple[int, int, int, tuple[int, ...]]:
    """(bus, addr, nbytes, payload) of one `ipmitool raw 0x06 0x52 ...` argv."""
    i = argv.index("raw")
    assert argv[i + 1 : i + 3] == ["0x06", "0x52"], argv
    vals = [int(x, 16) for x in argv[i + 3 :]]
    return vals[0], vals[1], vals[2], tuple(vals[3:])


class FakeBmc:
    """Answers the phase's reads from per-tray HSC and CPLD values; records every call."""

    def __init__(self, hsc=None, cpld=None, fail=(), stderr_prefix=""):
        # hsc: {(tray, addr): {"word", "input", "iout"[, "rec": {code: word}]}},
        # cpld: {tray: {"reg1e", "dbg5f", "dbg58"}}
        self.hsc = hsc or {}
        self.cpld = cpld or {}
        self.fail = set(fail)  # (tray, addr) whose reads all error out
        self.stderr_prefix = stderr_prefix  # e.g. a sudo warning line before ipmitool's error
        self.calls: list[list[str]] = []
        self.index: dict[int, int] = {}
        self.tray_of_bus = {bus: tray for tray, bus in TRAY_POWER_BUS.items()}

    def __call__(self, argv):
        self.calls.append(list(argv))
        bus, addr, nbytes, payload = txn_of(argv)
        tray = self.tray_of_bus[bus]
        if (tray, addr) in self.fail:
            err = self.stderr_prefix + "Unable to send RAW command\n"
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr=err)
        if addr == 0x80:
            cp = self.cpld.get(tray, CLEAN_CPLD)
            if nbytes == 0:
                self.index[bus] = payload[1]
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            key = {0x1E: "reg1e"}.get(payload[0]) or {0x5F: "dbg5f", 0x58: "dbg58"}[self.index[bus]]
            return subprocess.CompletedProcess(argv, 0, stdout=f" {cp[key]}\n", stderr="")
        st = self.hsc.get((tray, addr), CLEAN_HSC)
        code = payload[0]
        if code == 0x79 or code in CLEAN_REC:
            v = st["word"] if code == 0x79 else st.get("rec", {}).get(code, CLEAN_REC[code])
            out = f" {v & 0xFF:02x} {v >> 8:02x}\n"
        else:
            out = f" {st['input' if code == 0x7C else 'iout']:02x}\n"
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")


def run_phase(bmc, tier="pre_reboot", dry_run=False, clock=None) -> Phase:
    phase = Phase(name="tray_power", gates=False)
    kw = {"clock": clock} if clock else {}
    run_tray_power(tier, phase, dry_run, runner=bmc, **kw)
    return phase


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def by_tray(phase: Phase) -> dict[int, object]:
    return {c.data["tray"]: c for c in phase.checks}


class TestDecodeHsc(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(decode_hsc_status(0x0000, 0x00, 0x00)[0], [])

    def test_cml_alone_is_not_a_fault(self):
        faults, bits = decode_hsc_status(0x0002, 0x00)
        self.assertEqual(faults, [])
        self.assertEqual(bits, ["CML"])

    def test_vin_uv_warn_without_drop(self):
        self.assertEqual(decode_hsc_status(0x2002, 0x20)[0], ["VIN_UV_WARN"])
        self.assertEqual(decode_hsc_status(0x2000, 0x20)[0], ["VIN_UV_WARN"])

    def test_vin_uv_fault_synthetic(self):
        self.assertEqual(decode_hsc_status(0x2008, 0x10)[0], ["VIN_UV_FAULT"])

    def test_12v_loss_off(self):
        faults, bits = decode_hsc_status(0x2A42, 0x20)
        self.assertEqual(faults, ["OFF", "VIN_UV_WARN"])
        for b in ("CML", "OFF", "OTHER", "POWER_GOOD#", "INPUT", "VIN_UV_WARN"):
            self.assertIn(b, bits)

    def test_hotswap_overcurrent_drop(self):
        faults, bits = decode_hsc_status(0x6852, 0x20, 0x80)
        self.assertEqual(faults, ["OFF", "IOUT_OC", "VIN_UV_WARN"])
        self.assertIn("IOUT_OC_FAULT", bits)

    def test_iout_oc_from_status_iout_alone(self):
        self.assertEqual(decode_hsc_status(0x0000, 0x00, 0x80)[0], ["IOUT_OC"])


class TestDecodeCpld(unittest.TestCase):
    def test_states(self):
        self.assertEqual(decode_cpld_power("ff", "7c", "fa")[0], "HEALTHY")
        self.assertEqual(decode_cpld_power("00", "74", "f8")[0], "TRAY_12V_LOST")
        self.assertEqual(decode_cpld_power("00", "78", "fa")[0], "BUSBAR_SIG")
        self.assertEqual(decode_cpld_power("00", "74", "f0")[0], "HOTSWAP_OC")
        self.assertEqual(decode_cpld_power("00", "7c", "fa")[0], "CHIPS_DOWN_12V_GOOD")
        self.assertEqual(decode_cpld_power(None, "7c")[0], "UNREAD")
        self.assertEqual(decode_cpld_power("ff", "70", "fa")[0], "OTHER")


class TestTrayPowerPhase(unittest.TestCase):
    def assert_read_only(self, bmc: FakeBmc):
        """Every call is an allowed transaction, nothing is ever written to an HSC,
        and the clear-on-read peak registers are read at most once per HSC."""
        self.assertTrue(bmc.calls)
        peaks: dict = {}
        for argv in bmc.calls:
            bus, addr, nbytes, payload = txn_of(argv)
            self.assertIn(bus, TRAY_POWER_BUS.values())
            self.assertIn((addr, nbytes, payload), TRAY_POWER_ALLOWED)
            if addr != 0x80:
                self.assertEqual(len(payload), 1, argv)
                self.assertIn(payload[0], (0x79, 0x7B, 0x7C, 0xFA, 0xFB, 0x8B, 0x46, 0x4A, 0xA6, 0xAF), argv)
                self.assertGreater(nbytes, 0, argv)
                if payload[0] in (0xA6, 0xAF):
                    k = (bus, addr, payload[0])
                    peaks[k] = peaks.get(k, 0) + 1
                    self.assertEqual(peaks[k], 1, argv)
            elif nbytes == 0:
                # The CPLD debug-index select, the only transaction that reads nothing back.
                self.assertEqual(payload[0], 0x50, argv)

    def test_clean_passes(self):
        bmc = FakeBmc(hsc={(t, 0x42): {"word": 0x0002, "input": 0x00, "iout": 0x00} for t in TRAY_POWER_BUS})
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            phase = run_phase(bmc)
        self.assertNotIn("tray_power: stopped", err.getvalue())
        phase.rollup()
        self.assertEqual([c.status for c in phase.checks], [PASS] * 4)
        self.assertEqual(phase.status, PASS)
        self.assert_read_only(bmc)

    def test_vin_uv_warn_fails_naming_tray_bdf_and_hsc(self):
        bmc = FakeBmc(hsc={(2, 0x42): {"word": 0x2002, "input": 0x20, "iout": 0x00}})
        checks = by_tray(run_phase(bmc))
        c = checks[2]
        self.assertEqual(c.status, FAIL)
        self.assertEqual(c.name, "tray2_12v_feed")
        for s in ("tray 2", "41:00.0-48:00.0", "HSC 0x42", "VIN_UV_WARN", "bus bar / tray 12 V feed"):
            self.assertIn(s, c.details)
        self.assertNotIn("HSC 0x40", c.details)
        self.assertEqual([checks[t].status for t in (1, 3, 4)], [PASS] * 3)
        self.assert_read_only(bmc)

    def test_vin_uv_fault_fails(self):
        bmc = FakeBmc(hsc={(1, 0x40): {"word": 0x2008, "input": 0x10, "iout": 0x00}})
        c = by_tray(run_phase(bmc))[1]
        self.assertEqual(c.status, FAIL)
        self.assertIn("01:00.0-08:00.0", c.details)
        self.assertIn("VIN_UV_FAULT", c.details)

    def test_off_and_cpld_12v_lost(self):
        bmc = FakeBmc(
            hsc={(4, 0x40): {"word": 0x2A42, "input": 0x20, "iout": 0x00}},
            cpld={4: {"reg1e": "00", "dbg5f": "74", "dbg58": "f8"}},
        )
        c = by_tray(run_phase(bmc))[4]
        self.assertEqual(c.status, FAIL)
        for s in ("tray 4", "81:00.0-88:00.0", "HSC 0x40", "OFF", "STATUS_WORD 0x2a42", "TRAY_12V_LOST"):
            self.assertIn(s, c.details)
        self.assertEqual(c.data["hsc"]["0x40"]["faults"], ["OFF", "VIN_UV_WARN"])
        self.assert_read_only(bmc)

    def test_iout_oc_fails(self):
        bmc = FakeBmc(hsc={(4, 0x40): {"word": 0x6852, "input": 0x20, "iout": 0x80}})
        c = by_tray(run_phase(bmc))[4]
        self.assertEqual(c.status, FAIL)
        self.assertIn("IOUT_OC", c.details)

    def test_cpld_tray_12v_lost_alone_fails(self):
        bmc = FakeBmc(cpld={3: {"reg1e": "00", "dbg5f": "74", "dbg58": "f8"}})
        c = by_tray(run_phase(bmc))[3]
        self.assertEqual(c.status, FAIL)
        for s in ("tray 3", "c1:00.0-c8:00.0", "TRAY_12V_LOST", "bus bar / tray 12 V feed"):
            self.assertIn(s, c.details)

    def test_cpld_busbar_signature_fails(self):
        bmc = FakeBmc(cpld={1: {"reg1e": "00", "dbg5f": "78", "dbg58": "fa"}})
        self.assertEqual(by_tray(run_phase(bmc))[1].status, FAIL)

    def test_read_error_is_not_a_pass(self):
        bmc = FakeBmc(fail={(3, 0x42)})
        checks = by_tray(run_phase(bmc))
        self.assertEqual(checks[3].status, WARN)
        self.assertIn("not read: HSC 0x42", checks[3].details)
        self.assertEqual(checks[1].status, PASS)

    def test_read_error_beside_a_fault_still_fails(self):
        bmc = FakeBmc(hsc={(3, 0x40): {"word": 0x2002, "input": 0x20, "iout": 0x00}}, fail={(3, 0x42)})
        c = by_tray(run_phase(bmc))[3]
        self.assertEqual(c.status, FAIL)
        self.assertIn("not read: HSC 0x42", c.details)

    def test_no_route_is_one_skip(self):
        bmc = FakeBmc(fail={(t, a) for t in TRAY_POWER_BUS for a in (0x40, 0x42, 0x80)})
        phase = run_phase(bmc)
        self.assertEqual([(c.name, c.status) for c in phase.checks], [("tray_12v_feed", SKIP)])
        self.assertIn("Unable to send RAW command", phase.checks[0].details)

    def test_dead_tray_1_does_not_hide_trays_2_to_4(self):
        # Tray 1's HSCs and CPLD NACK at once (a dead tray); trays 2-4 answer.
        bmc = FakeBmc(
            hsc={(3, 0x40): {"word": 0x2002, "input": 0x20, "iout": 0x00}},
            fail={(1, a) for a in (0x40, 0x42, 0x80)},
        )
        checks = by_tray(run_phase(bmc))
        self.assertEqual([checks[t].status for t in (1, 2, 3, 4)], [WARN, PASS, FAIL, PASS])
        self.assertIn("not read: HSC 0x40, HSC 0x42, CPLD", checks[1].details)
        for s in ("tray 3", "HSC 0x40", "VIN_UV_WARN", "bus bar / tray 12 V feed"):
            self.assertIn(s, checks[3].details)
        self.assert_read_only(bmc)

    def test_dead_trays_1_to_3_do_not_hide_tray_4(self):
        bmc = FakeBmc(
            hsc={(4, 0x42): {"word": 0x2002, "input": 0x20, "iout": 0x00}},
            fail={(t, a) for t in (1, 2, 3) for a in (0x40, 0x42, 0x80)},
        )
        checks = by_tray(run_phase(bmc))
        self.assertEqual([checks[t].status for t in (1, 2, 3, 4)], [WARN, WARN, WARN, FAIL])
        self.assertIn("bus bar / tray 12 V feed", checks[4].details)

    def test_missing_ipmitool_is_one_skip(self):
        def runner(argv):
            raise FileNotFoundError("ipmitool")

        calls = []

        def counting(argv):
            calls.append(argv)
            return runner(argv)

        phase = run_phase(counting)
        self.assertEqual([c.status for c in phase.checks], [SKIP])
        self.assertEqual(len(calls), 1)

    def test_sudo_refusal_is_one_skip(self):
        calls = []

        def runner(argv):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="sudo: a password is required\n")

        phase = run_phase(runner)
        self.assertEqual([c.status for c in phase.checks], [SKIP])
        self.assertEqual(len(calls), 1)

    def test_sudo_refusal_anywhere_in_stderr_is_one_skip(self):
        for err in (
            "sudo: unable to resolve host h: Name or service not known\nsudo: a password is required\n",
            "sudo: a terminal is required to read the password\n",
            "smarton is not in the sudoers file.  This incident will be reported.\n",
            "Sorry, user smarton is not allowed to execute '/usr/bin/ipmitool raw' as root on h.\n",
            "sudo: ipmitool: command not found\n",
        ):
            calls = []

            def runner(argv, err=err):
                calls.append(argv)
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr=err)

            phase = run_phase(runner)
            self.assertEqual([c.status for c in phase.checks], [SKIP], err)
            self.assertEqual(len(calls), 1, err)

    def test_sudo_warning_line_does_not_hide_other_trays(self):
        # sudo warns on the first line and still runs ipmitool; tray 1's bus NACKs.
        fail = {(1, a) for a in (0x40, 0x42, 0x80)}
        for warn in (
            "sudo: unable to resolve host h: Name or service not known\n",
            "sudo: setrlimit(RLIMIT_CORE): Operation not permitted\n",
        ):
            bmc = FakeBmc(hsc={(3, 0x40): dict(CLEAN_HSC, word=0x2000, input=0x20)}, fail=fail, stderr_prefix=warn)
            checks = by_tray(run_phase(bmc))
            self.assertEqual([checks[t].status for t in (1, 2, 3, 4)], [WARN, PASS, FAIL, PASS], warn)
            self.assertIn("Unable to send RAW command", checks[1].details, warn)
            self.assertNotIn("resolve host", checks[1].details, warn)
            self.assertNotIn("setrlimit", checks[1].details, warn)

    def test_hung_tray_1_does_not_hide_tray_3_fault(self):
        clock = FakeClock()
        bmc = FakeBmc(hsc={(3, 0x40): dict(CLEAN_HSC, word=0x2000, input=0x20)})

        def runner(argv):
            if txn_of(argv)[0] == TRAY_POWER_BUS[1]:
                clock.t += TRAY_POWER_CALL_S
                raise subprocess.TimeoutExpired(argv, TRAY_POWER_CALL_S)
            return bmc(argv)

        checks = by_tray(run_phase(runner, clock=clock))
        self.assertEqual([checks[t].status for t in (1, 2, 3, 4)], [WARN, PASS, FAIL, PASS])
        self.assertIn("not read", checks[1].details)
        self.assertIn(f"no reply in {TRAY_POWER_CALL_S:g} s", checks[1].details)

    def test_other_tiers_and_dry_run_send_nothing(self):
        for tier in ("light", "medium", "deploy"):
            bmc = FakeBmc()
            self.assertEqual([c.status for c in run_phase(bmc, tier=tier).checks], [SKIP])
            self.assertEqual(bmc.calls, [])
        bmc = FakeBmc()
        self.assertEqual([c.status for c in run_phase(bmc, dry_run=True).checks], [SKIP])
        self.assertEqual(bmc.calls, [])

    def test_timeouts_everywhere_stop_at_once_as_one_skip(self):
        calls = []

        def runner(argv):
            calls.append(argv)
            raise subprocess.TimeoutExpired(argv, TRAY_POWER_CALL_S)

        phase = run_phase(runner)
        self.assertLessEqual(len(calls), 12)
        self.assertEqual([(c.name, c.status) for c in phase.checks], [("tray_12v_feed", SKIP)])
        self.assertIn("no reply", phase.checks[0].details)

    def test_deadline_mid_run_leaves_the_rest_not_read(self):
        clock = FakeClock()
        clean = FakeBmc()

        def runner(argv):
            cp = clean(argv)
            if len(clean.calls) == 32:  # the latches and CPLD of trays 1 and 2
                clock.t = TRAY_POWER_PHASE_S
            return cp

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            checks = by_tray(run_phase(runner, clock=clock))
        lines = err.getvalue().splitlines()
        stop = f"[diag] tray_power: stopped reading: phase time limit of {TRAY_POWER_PHASE_S:g} s spent"
        self.assertEqual([l for l in lines if not l.startswith("[diag] tray_power: tray ")], [stop])
        self.assertEqual(len(lines), 5)  # and one record line per HSC of trays 1 and 2
        self.assertEqual(len(clean.calls), 32)
        self.assertEqual([checks[t].status for t in (1, 2, 3, 4)], [PASS, PASS, WARN, WARN])
        for t in (3, 4):
            self.assertIn("not read: HSC 0x40, HSC 0x42, CPLD (phase time limit", checks[t].details)

    def test_deadline_in_the_record_pass_keeps_every_tray_status_verdict(self):
        # Every tray's latches and CPLD are read before any record word: a slow BMC
        # that spends the phase limit on trays 1-2 record words cannot cost trays 3-4
        # the STATUS_WORD / STATUS_INPUT reads that decide FAIL.
        clock = FakeClock()
        hsc = {
            (3, 0x42): dict(CLEAN_HSC, word=0x2002, input=0x20),  # VIN_UV_WARN
            (4, 0x40): dict(CLEAN_HSC, word=0x0010, iout=0x80),  # IOUT_OC
        }
        bmc = FakeBmc(hsc=hsc)
        status = {(0x79,), (0x7C,), (0x7B,)}

        def runner(argv):
            cp = bmc(argv)
            if len(bmc.calls) == 4 * 16 + 14:  # pass 1 of 4 trays, then tray 1's record words
                clock.t = TRAY_POWER_PHASE_S
            return cp

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            checks = by_tray(run_phase(runner, clock=clock))
        self.assertEqual(len(bmc.calls), 78)
        first_record = next(i for i, a in enumerate(bmc.calls) if txn_of(a)[1] != 0x80 and txn_of(a)[3] not in status)
        self.assertEqual(first_record, 64)
        self.assertEqual([checks[t].status for t in (1, 2, 3, 4)], [PASS, PASS, FAIL, FAIL])
        self.assertIn("HSC 0x42: VIN_UV_WARN latched", checks[3].details)
        self.assertIn("HSC 0x40: IOUT_OC latched", checks[4].details)
        for t in (1, 2, 3, 4):
            self.assertNotIn("not read", checks[t].details)
        lines = err.getvalue().splitlines()
        rec = {(t, a): next(ln for ln in lines if f"tray_power: tray {t} HSC 0x{a:02x}:" in ln)
               for t in (1, 2, 3, 4) for a in (0x40, 0x42)}  # fmt: skip
        for a in (0x40, 0x42):
            self.assertIn("FAULT_RECORD 0x0000, FAULT_RECORD_LAST 0x0012, STATUS_IOUT 0x00, VOUT 12.22 V", rec[1, a])
            for t in (2, 3, 4):
                self.assertIn("FAULT_RECORD not read, FAULT_RECORD_LAST not read, STATUS_IOUT 0x", rec[t, a])
                self.assertIn("VOUT not read, IOUT peak not read, TEMP peak not read", rec[t, a])
                self.assertIsNone(checks[t].data["hsc"][f"0x{a:02x}"]["fault_record"])
        self.assertIn(f"stopped reading: phase time limit of {TRAY_POWER_PHASE_S:g} s spent", err.getvalue())

    def test_bmc_hanging_mid_run_is_bounded_by_the_phase_limit(self):
        clock = FakeClock()
        clean = FakeBmc()

        def runner(argv):
            if txn_of(argv)[0] == TRAY_POWER_BUS[1]:
                clock.t += 1
                return clean(argv)
            clock.t += TRAY_POWER_CALL_S
            raise subprocess.TimeoutExpired(argv, TRAY_POWER_CALL_S)

        checks = by_tray(run_phase(runner, clock=clock))
        self.assertLessEqual(clock.t, TRAY_POWER_PHASE_S + TRAY_POWER_CALL_S)
        self.assertEqual(checks[1].status, PASS)
        for t in (2, 3, 4):
            self.assertEqual(checks[t].status, WARN)
            self.assertIn("not read: HSC 0x40, HSC 0x42, CPLD", checks[t].details)

    def test_call_timeout_never_outlasts_the_phase(self):
        clock = FakeClock()
        reader = TrayPowerReader(clock=clock)
        seen = {}

        def fake_run(argv, **kw):
            seen.update(kw)
            return subprocess.CompletedProcess(argv, 0, stdout=" 00 00\n", stderr="")

        clock.t = TRAY_POWER_PHASE_S - 4
        with mock.patch.object(diag_runner, "run", fake_run):
            reader.txn(TRAY_POWER_BUS[1], 0x40, 2, 0x79)
        self.assertEqual(seen["timeout"], 4)

    def test_powered_off_tray_is_a_warn_without_the_bus_bar_hint(self):
        bmc = FakeBmc(
            hsc={(2, a): {"word": 0x0840, "input": 0x00, "iout": 0x00} for a in (0x40, 0x42)},
            cpld={2: {"reg1e": "00", "dbg5f": "70", "dbg58": "78"}},
        )
        checks = by_tray(run_phase(bmc))
        self.assertEqual(checks[2].status, WARN)
        self.assertIn("powered off", checks[2].details)
        self.assertNotIn("bus bar", checks[2].details)
        self.assertIs(checks[2].data["cpld"]["power_enable"], False)
        self.assertEqual(checks[1].status, PASS)

    def test_gate_refuses_writes_before_ipmitool(self):
        bmc = FakeBmc()
        reader = TrayPowerReader(bmc)
        refused = [
            (0x11, 0x40, 0, (0x03,)),  # CLEAR_FAULTS
            (0x11, 0x40, 1, (0x01, 0x00)),  # OPERATION write
            (0x11, 0x42, 2, (0xA1,)),  # peak register, clears on read
            (0x11, 0x42, 2, (0xA0,)),  # READ_VIN_MIN, not read
            (0x11, 0x42, 2, (0xA7,)),  # peak register not in the whitelist
            (0x11, 0x40, 2, (0xFC,)),  # manufacturer register
            (0x11, 0x40, 0, (0xE0,)),  # manufacturer command, writes
            (0x11, 0x40, 1, (0xA6,)),  # READ_IOUT_PEAK with the wrong length
            (0x11, 0x40, 2, (0x46, 0x00)),  # IOUT_OC_FAULT_LIMIT write
            (0x11, 0x40, 2, (0x4A, 0x10, 0x0E)),  # IOUT_OC_WARN_LIMIT write
            (0x11, 0x80, 0, (0x50, 0x10)),  # CPLD index other than 0x5F / 0x58
            (0x11, 0x80, 0, (0x30, 0x8B)),  # any other CPLD write
            (0x12, 0x40, 2, (0x79,)),  # not a tray bus
        ]
        for bus, addr, nbytes, payload in refused:
            with self.assertRaises(ValueError):
                reader.txn(bus, addr, nbytes, *payload)
        self.assertEqual(bmc.calls, [])


# Hand reads of one galaxy (SYS-5505): tray 2 HSC 0x40 latched OFF and
# IOUT_OC with FAULT_RECORD 0x0022 and an IOUT peak of 235.5 A; the other seven
# feeds peaked at 180.5-185.5 A. Raw IOUT peaks in 62.5 mA steps.
OC_TRAY_PEAKS = {
    (1, 0x40): 2920, (1, 0x42): 2968, (2, 0x40): 3768, (2, 0x42): 2944,
    (3, 0x40): 2912, (3, 0x42): 2888, (4, 0x40): 2896, (4, 0x42): 2904,
}  # fmt: skip


def oc_tray_hsc(tray2_status=True):
    hsc = {}
    for (t, a), peak in OC_TRAY_PEAKS.items():
        hsc[(t, a)] = dict(CLEAN_HSC, rec={0xA6: peak, 0x46: 3712, 0x4A: 3552} if a == 0x42 else {0xA6: peak})
    if tray2_status:
        hsc[(2, 0x40)].update(word=0x6852, input=0x20, iout=0x80)
        hsc[(2, 0x40)]["rec"].update({0xFA: 0x0022})
    return hsc


class TestHscRecord(unittest.TestCase):
    def test_decode(self):
        d = decode_hsc_record(
            {
                "read_vout": 0x0187,
                "iout_peak": 3768,
                "temp_peak": 43,
                "iout_oc_fault_limit": 3648,
                "iout_oc_warn_limit": 3504,
            }
        )
        self.assertEqual(d, {"vout_v": 12.219, "iout_peak_a": 235.5, "temp_peak_c": 43,
                             "iout_oc_fault_limit_a": 228.0, "iout_oc_warn_limit_a": 219.0})  # fmt: skip
        self.assertEqual(set(decode_hsc_record({}).values()), {None})

    def test_oc_tray_2_warns_and_fails_on_its_latches(self):
        bmc = FakeBmc(hsc=oc_tray_hsc())
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            checks = by_tray(run_phase(bmc))
        c = checks[2]
        self.assertEqual(c.status, FAIL)  # the latches decide the FAIL, as before
        for s in (
            "IOUT_OC",
            "HSC 0x40: FAULT_RECORD 0x0022 (FAh) set; WARN only",
            "HSC 0x40: IOUT peak 235.5 A (A6h, since the last peak read) above its own IOUT_OC_WARN_LIMIT 219.0 A",
            "HSC 0x40: IOUT peak 235.5 A, 29% above the median 182.0 A of the other 7 feeds (WARN over 20%)",
        ):
            self.assertIn(s, c.details)
        self.assertNotIn("HSC 0x42: IOUT peak", c.details)
        self.assertEqual(c.data["hsc"]["0x40"]["iout_peak_a"], 235.5)
        self.assertEqual(c.data["hsc"]["0x40"]["fault_record"], 0x0022)
        self.assertEqual(c.data["hsc"]["0x42"]["iout_oc_warn_limit_a"], 222.0)
        self.assertEqual([checks[t].status for t in (1, 3, 4)], [PASS] * 3)
        self.assertIn(
            "[diag] tray_power: tray 2 HSC 0x40: FAULT_RECORD 0x0022, FAULT_RECORD_LAST 0x0012, STATUS_IOUT 0x80, "
            "VOUT 12.22 V, IOUT peak 235.5 A, TEMP peak 43 C, IOUT OC fault/warn limit 228.0/219.0 A",
            err.getvalue(),
        )
        self.assertEqual(err.getvalue().count("[diag] tray_power: tray "), 8)
        self.assert_read_only = TestTrayPowerPhase.assert_read_only.__get__(self)
        self.assert_read_only(bmc)

    def test_peak_alone_turns_a_pass_into_a_warn_never_a_fail(self):
        checks = by_tray(run_phase(FakeBmc(hsc=oc_tray_hsc(tray2_status=False))))
        self.assertEqual(checks[2].status, WARN)
        self.assertIn("no 12 V latch set", checks[2].details)
        self.assertIn("29% above the median", checks[2].details)
        self.assertEqual([checks[t].status for t in (1, 3, 4)], [PASS] * 3)

    def test_fault_record_alone_is_a_warn(self):
        hsc = {(3, 0x42): dict(CLEAN_HSC, rec={0xFA: 0x0004})}
        checks = by_tray(run_phase(FakeBmc(hsc=hsc)))
        self.assertEqual(checks[3].status, WARN)
        self.assertIn("HSC 0x42: FAULT_RECORD 0x0004 (FAh) set; WARN only", checks[3].details)
        self.assertEqual([checks[t].status for t in (1, 2, 4)], [PASS] * 3)

    def test_fault_record_last_baseline_and_even_peaks_pass(self):
        checks = by_tray(run_phase(FakeBmc()))
        self.assertEqual([checks[t].status for t in (1, 2, 3, 4)], [PASS] * 4)

    def test_peak_rule_needs_three_other_feeds(self):
        hscs = {1: {0x40: {"iout_peak": 3768}, 0x42: {"iout_peak": 2944}}, 2: {0x40: {"iout_peak": 2944}}}
        self.assertEqual(hsc_record_warnings(hscs), {1: [], 2: []})

    def test_peaks_are_read_once_even_when_the_read_fails(self):
        clean = FakeBmc()

        def runner(argv):
            if txn_of(argv)[3] in ((0xA6,), (0xAF,)):
                clean.calls.append(list(argv))
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr="Unable to send RAW command\n")
            return clean(argv)

        checks = by_tray(run_phase(runner))
        n = sum(1 for argv in clean.calls if txn_of(argv)[3] in ((0xA6,), (0xAF,)))
        self.assertEqual(n, 16)  # 2 codes x 2 HSCs x 4 trays, no retry
        self.assertEqual([checks[t].status for t in (1, 2, 3, 4)], [PASS] * 4)
        self.assertIsNone(checks[1].data["hsc"]["0x40"]["iout_peak_a"])

    def test_failed_record_reads_log_not_read_never_zero(self):
        clean = FakeBmc()
        codes = [(c,) for c in (0xFA, 0xFB, 0x7B, 0x8B, 0xA6, 0xAF, 0x46)]

        def runner(argv):
            if txn_of(argv)[3] in codes:
                clean.calls.append(list(argv))
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr="Unable to send RAW command\n")
            return clean(argv)

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            checks = by_tray(run_phase(runner))
        line = next(ln for ln in err.getvalue().splitlines() if "tray_power: tray 1 HSC 0x40:" in ln)
        self.assertIn(
            "FAULT_RECORD not read, FAULT_RECORD_LAST not read, STATUS_IOUT not read, VOUT not read, "
            "IOUT peak not read, TEMP peak not read, IOUT OC fault/warn limit not read/",
            line,
        )
        self.assertNotIn("0x0000", line)
        self.assertNotIn(" 0.0", line)
        for key in ("fault_record", "fault_record_last", "status_iout", "read_vout", "iout_peak", "temp_peak"):
            self.assertIsNone(checks[1].data["hsc"]["0x40"][key], key)
        self.assertIsNone(checks[1].data["hsc"]["0x40"]["vout_v"])


class TestPlumbing(unittest.TestCase):
    def test_bdf_ranges(self):
        self.assertEqual(
            [tray_bdf_range(t) for t in (1, 2, 3, 4)],
            ["01:00.0-08:00.0", "41:00.0-48:00.0", "c1:00.0-c8:00.0", "81:00.0-88:00.0"],
        )

    def test_skip_flag(self):
        args = build_diag_arg_parser().parse_args(["--tier", "pre_reboot", "--skip-tray-power"])
        self.assertTrue(args.skip_tray_power)


if __name__ == "__main__":
    unittest.main()
