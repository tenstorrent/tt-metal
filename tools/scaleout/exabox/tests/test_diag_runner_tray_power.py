#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The tray 12 V feed check (diag/hsc.py run_tray_feed) on a fake BMC, and its wiring into diag_runner.run_diag and
the tiers: no BMC, no chip."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SUITE_DIR = Path(__file__).resolve().parent.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

import diag_runner  # noqa: E402
from diag import hsc  # noqa: E402
from diag.model import FAIL, PASS, SKIP, WARN, Check, Phase  # noqa: E402
from tiers import TIERS  # noqa: E402

BUS_TRAY = {b: t for t, b in hsc.HSC_BUS.items()}
UV_WARN = {"STATUS_WORD": 0x2002, "STATUS_INPUT": 0x20}
OFF_OC = {"STATUS_WORD": 0x0050, "STATUS_INPUT": 0x00}


class FakeBmc:
    """hsc.Bmc.raw from registers: hsc[(tray, addr)] overrides the clean STATUS_WORD/STATUS_INPUT, alive[tray] the CPLD
    byte (0xFF up), dead = (tray, addr) HSCs or trays whose bus does not answer."""

    def __init__(self, hsc_regs=None, alive=None, dead=(), no_cpld=()):
        self.hsc, self.alive, self.dead, self.no_cpld, self.calls = hsc_regs or {}, alive or {}, dead, no_cpld, []

    def raw(self, _bmc, *a, lock=True):
        self.calls.append(a)
        assert a[:2] == (0x06, 0x52), a  # reads only: no other command
        bus, addr, nbytes, code = a[2:6]
        tray = BUS_TRAY[bus]
        if addr == hsc.TRAY_CPLD:
            assert code == hsc.CPLD_ALIVE_REG, a
            vals = self.alive.get(tray, [hsc.CPLD_ALIVE])
            v = None if tray in self.no_cpld else vals[min(len(self.calls_to(tray)) - 1, len(vals) - 1)]
            return None if v is None else [v]
        assert addr in hsc.HSC_ADDRS and code in (0x79, 0x7C), a
        if (tray, addr) in self.dead or tray in self.dead:
            return None
        regs = {"STATUS_WORD": 0, "STATUS_INPUT": 0, **self.hsc.get((tray, addr), {})}
        v = regs["STATUS_WORD" if code == 0x79 else "STATUS_INPUT"]
        return list(v.to_bytes(nbytes, "little"))

    def calls_to(self, tray):
        return [a for a in self.calls if a[2] == hsc.HSC_BUS[tray] and a[3] == hsc.TRAY_CPLD]

    def patch(self):
        return mock.patch.object(hsc.Bmc, "raw", lambda b, *x, lock=True: self.raw(b, *x, lock=lock))


def run_feed(fake: FakeBmc) -> Phase:
    phase = Phase(name="tray_power", gates=False)
    bmc = hsc.Bmc("true", lock="/dev/null")
    with fake.patch():
        hsc.run_tray_feed(phase, bmc)
    return phase


def by_tray(phase: Phase) -> dict[str, Check]:
    return {c.name: c for c in phase.checks}


class TestTrayFeedVerdicts(unittest.TestCase):
    def test_clean_box_passes_every_tray(self):
        phase = run_feed(FakeBmc())
        self.assertEqual(
            [(c.name, c.status) for c in phase.checks], [(f"tray{t}_12v_feed", PASS) for t in (1, 2, 3, 4)]
        )
        self.assertEqual(
            by_tray(phase)["tray1_12v_feed"].details, "tray 1 (01:00.0-08:00.0): no 12 V latch set, tray up"
        )

    def test_causes(self):
        cases = [  # (name, fake, tray, status, cause, texts in the details)
            (
                "latch",
                FakeBmc({(3, 0x42): UV_WARN}),
                3,
                FAIL,
                "POWER",
                [
                    "tray 3 (c1:00.0-c8:00.0): POWER: check bus bar / tray 12 V feed",
                    "HSC 0x42: VIN_UV_WARN latched (STATUS_WORD 0x2002, STATUS_INPUT 0x20)",
                ],
            ),
            ("trip", FakeBmc({(1, 0x40): OFF_OC}), 1, FAIL, "POWER", ["HSC 0x40: OFF, IOUT_OC_FAULT latched"]),
            (
                "down, clean latches",
                FakeBmc(alive={2: [0x00]}),
                2,
                WARN,
                "CHIP/BOARD",
                [
                    "tray 2 (41:00.0-48:00.0): CHIP/BOARD: tray down with no 12 V feed latch set, not the feed",
                    "CPLD reg 0x1e=0x00: tray down",
                ],
            ),
            (
                "down and latched: the feed is named, not the chip",
                FakeBmc({(4, 0x40): UV_WARN}, alive={4: [0x00]}),
                4,
                FAIL,
                "POWER",
                ["POWER: check bus bar", "CPLD reg 0x1e=0x00: tray down"],
            ),
        ]
        for name, fake, tray, status, cause, texts in cases:
            with self.subTest(name):
                checks = by_tray(run_feed(fake))
                c = checks[f"tray{tray}_12v_feed"]
                self.assertEqual((c.status, c.data["cause"]), (status, cause))
                for s in texts:
                    self.assertIn(s, c.details)
                others = [x.status for n, x in checks.items() if n != c.name]
                self.assertEqual(others, [PASS] * 3)

    def test_unread_is_not_read_and_no_cause(self):
        checks = by_tray(run_feed(FakeBmc(dead=((2, 0x42),))))
        c = checks["tray2_12v_feed"]
        self.assertEqual((c.status, c.data["cause"]), (WARN, None))
        self.assertEqual(c.details, "tray 2 (41:00.0-48:00.0): not read: HSC 0x42")

        # A tray down whose latches could not all be read is not called CHIP/BOARD.
        c = by_tray(run_feed(FakeBmc(alive={1: [0x00]}, dead=((1, 0x40),))))["tray1_12v_feed"]
        self.assertEqual((c.status, c.data["cause"]), (WARN, None))
        self.assertIn("not read: HSC 0x40", c.details)

        # A latch on the HSC that did answer still fails the tray.
        c = by_tray(run_feed(FakeBmc({(1, 0x42): UV_WARN}, dead=((1, 0x40),))))["tray1_12v_feed"]
        self.assertEqual((c.status, c.data["cause"]), (FAIL, "POWER"))
        self.assertIn("not read: HSC 0x40", c.details)

    def test_cpld_down_needs_two_agreeing_reads(self):
        calls = lambda fake: len(fake.calls_to(1))  # noqa: E731
        fake = FakeBmc(alive={1: [0x00, 0x00]})
        self.assertEqual(by_tray(run_feed(fake))["tray1_12v_feed"].data["cause"], "CHIP/BOARD")
        self.assertEqual(calls(fake), 2)

        fake = FakeBmc(alive={1: [0x00, 0xFF]})  # the reads disagree: not read, no cause
        c = by_tray(run_feed(fake))["tray1_12v_feed"]
        self.assertEqual((c.status, c.data["cause"]), (WARN, None))
        self.assertIn("not read: CPLD", c.details)

        fake = FakeBmc()  # an up tray costs one read
        run_feed(fake)
        self.assertEqual(calls(fake), 1)

    def test_nothing_answers_is_one_skip(self):
        phase = run_feed(FakeBmc(dead=tuple(hsc.HSC_BUS), no_cpld=tuple(hsc.HSC_BUS)))
        self.assertEqual([(c.name, c.status) for c in phase.checks], [("tray_12v_feed", SKIP)])
        self.assertIn("no HSC or CPLD answered", phase.checks[0].details)

        phase = Phase(name="tray_power", gates=False)
        hsc.run_tray_feed(phase, hsc.Bmc("true-not-a-command", lock="/dev/null"))  # no route
        self.assertEqual([(c.name, c.status) for c in phase.checks], [("tray_12v_feed", SKIP)])

    def test_reads_only_and_the_existing_reader(self):
        fake = FakeBmc()
        run_feed(fake)  # FakeBmc.raw asserts each call is an I2C read of STATUS_WORD, STATUS_INPUT or CPLD 0x1E
        self.assertTrue(fake.calls)


class TestTierAndRunDiagWiring(unittest.TestCase):
    def test_only_the_bmc_reboot_tiers_read_the_trays(self):
        self.assertEqual({n for n, t in TIERS.items() if t.tray_power}, {"pre_bmc_reboot", "post_bmc_reboot"})
        for n in ("pre_bmc_reboot", "post_bmc_reboot"):
            self.assertEqual((TIERS[n].resets, TIERS[n].tests), ((), ()))  # nothing resets before the read

    def run_diag(self, tier, side_effect=None, **kwargs):
        others = "collect_fru_info reset_loop run_post_test_reset run_tests run_triage run_qsfp_tests"
        others += " run_sys_triage run_eth_status validate_snapshot resolve_tt_smi detect_tt_smi_version"
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            for name in others.split():
                stack.enter_context(mock.patch.object(diag_runner, name, mock.Mock(return_value=None)))
            stack.enter_context(mock.patch.object(diag_runner, "take_snapshot", mock.Mock(return_value={})))
            feed = stack.enter_context(mock.patch.object(diag_runner, "run_tray_feed", side_effect=side_effect))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            code, report = diag_runner.run_diag(tier, output=Path(tmp) / "report.json", **kwargs)
        return code, report, feed

    def test_runs_right_after_snapshot_on_the_bmc_reboot_tiers(self):
        def feed(phase):
            phase.add(Check(name="tray1_12v_feed", status=PASS, details="clean", ip="board"))

        _, report, _ = self.run_diag("pre_bmc_reboot", feed)
        self.assertEqual(list(report["phases"])[:2], ["snapshot", "tray_power"])
        phase = report["phases"]["tray_power"]
        self.assertEqual((phase["error"], phase["status"], phase["gates"]), ("", PASS, False))

        _, report, run = self.run_diag("flash", dry_run=True)  # a tier without the read has no such phase
        self.assertNotIn("tray_power", report["phases"])
        run.assert_not_called()

    def test_a_latch_fails_the_phase_without_gating_the_run(self):
        def feed(phase):
            phase.add(Check(name="tray3_12v_feed", status=FAIL, details="POWER", ip="board"))

        code, report, _ = self.run_diag("pre_bmc_reboot", feed)
        self.assertEqual(report["phases"]["tray_power"]["status"], FAIL)
        self.assertEqual((report["overall_status"], code), (PASS, 0))

    def test_skip_flag_dry_run_snapshot_replay_and_a_crash_each_leave_one_skip(self):
        snap = Path(tempfile.mkdtemp()) / "snap.json"
        snap.write_text("{}")
        cases = (
            ({"skip_tray_power": True}, "--skip-tray-power"),
            ({"dry_run": True}, "dry"),
            ({"input_snapshot": snap}, "replayed snapshot"),
        )
        for kwargs, text in cases:
            with self.subTest(kwargs):
                _, report, run = self.run_diag("pre_bmc_reboot", **kwargs)
                run.assert_not_called()
                phase = report["phases"]["tray_power"]
                self.assertEqual([(c["name"], c["status"]) for c in phase["checks"]], [("tray_12v_feed", SKIP)])
                self.assertIn(text, phase["checks"][0]["details"])
                self.assertEqual(phase["status"], SKIP)

        code, report, _ = self.run_diag("pre_bmc_reboot", RuntimeError("bmc went away"))
        phase = report["phases"]["tray_power"]
        self.assertEqual([(c["name"], c["status"]) for c in phase["checks"]], [("tray_12v_feed", SKIP)])
        self.assertIn("bmc went away", phase["error"])
        self.assertEqual((phase["status"], code), (SKIP, 0))


if __name__ == "__main__":
    unittest.main()
