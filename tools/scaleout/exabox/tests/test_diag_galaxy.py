#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The galaxy tier without hardware: its reset is -glx_reset and never -r, the
chip recount names missing chips by BDF/tray/ASIC, the thermal and telemetry
checks ignore garbage readings."""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

import tiers  # noqa: E402
from diag import resets  # noqa: E402
from diag.aer import AerWindow, aer_check, aer_new, read_aer  # noqa: E402
from diag.galaxy import TRAY_TO_UBB_NIBBLE, expected_bdf  # noqa: E402
from diag.model import FAIL, PASS, WARN, Phase  # noqa: E402
from diag.naming import chip  # noqa: E402
from diag.snapshot_checks import _thermal_and_telemetry  # noqa: E402


def all_bdfs():
    return [expected_bdf(t, a) for t in sorted(TRAY_TO_UBB_NIBBLE) for a in range(1, 9)]


def _dev(bdf, **telemetry):
    return {"board_info": {"bus_id": bdf}, "telemetry": telemetry, "limits": {"thm_limit": "100"}}


def _good(bdf):
    return _dev(bdf, asic_temperature="45.0", power="120.0", current="90.0", voltage="0.75")


def _checks(devs):
    phase = Phase(name="snapshot")
    _thermal_and_telemetry(devs, phase)
    return {c.name: c for c in phase.checks}


class TestNaming(unittest.TestCase):
    def test_bdf_to_tray_and_asic(self):
        self.assertEqual(chip("0000:02:00.0"), {"bdf": "0000:02:00.0", "tray": 1, "asic": 2})
        self.assertEqual(chip("0000:c8:00.0"), {"bdf": "0000:c8:00.0", "tray": 3, "asic": 8})
        self.assertEqual(chip("0000:41:00.0")["tray"], 2)
        self.assertEqual(chip("0000:81:00.0")["tray"], 4)

    def test_off_map_bdf_has_no_tray(self):
        self.assertEqual(chip("0000:ff:00.0"), {"bdf": "0000:ff:00.0", "tray": None, "asic": None})


class TestGalaxyTier(unittest.TestCase):
    def test_only_glx_reset(self):
        t = tiers.TIERS["galaxy"]
        self.assertEqual(t.resets, ("-glx_reset",))
        self.assertNotIn("-r", t.post_test_resets)
        self.assertTrue(t.glx_reset_only)
        self.assertEqual(t.tests, ())

    def test_only_galaxy_is_glx_reset_only(self):
        self.assertEqual([n for n, t in tiers.TIERS.items() if t.glx_reset_only], ["galaxy"])

    def test_validator_rejects_r(self):
        for bad in (dict(resets=("-r",)), dict(post_test_resets=("-r",)), dict(resets=("-glx_reset", "-r"))):
            with self.subTest(bad):
                broken = {**tiers.TIERS, "galaxy": replace(tiers.TIERS["galaxy"], **bad)}
                with mock.patch.object(tiers, "TIERS", broken):
                    with self.assertRaisesRegex(Exception, "-r is not allowed"):
                        tiers._validate()

    def test_validator_accepts_shipped_tiers(self):
        tiers._validate()


class TestDryRunResets(unittest.TestCase):
    def test_no_r_command_is_reachable(self):
        cmds = []

        def fake_run(cmd, dry_run=False, **kw):
            cmds.append(cmd)
            import subprocess

            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        phase = Phase(name="reset_loop")
        with mock.patch.object(resets, "run", fake_run):
            resets.reset_loop("tt-smi", tiers.TIERS["galaxy"].resets, phase, dry_run=True)
        self.assertEqual(cmds, [["tt-smi", "-glx_reset"]])
        self.assertNotIn("-r", [a for c in cmds for a in c])
        self.assertTrue(all(c.status == PASS for c in phase.checks))


class TestRecount(unittest.TestCase):
    def test_all_back(self):
        c = resets.recount_check("recount_1", all_bdfs())
        self.assertEqual(c.status, PASS)
        self.assertEqual((c.data["actual"], c.data["missing"]), (32, []))

    def test_missing_chips_named(self):
        bdfs = [b for b in all_bdfs() if b not in ("0000:c3:00.0", "0000:84:00.0")]
        c = resets.recount_check("recount_1", bdfs)
        self.assertEqual(c.status, FAIL)
        self.assertEqual(c.data["actual"], 30)
        self.assertEqual(
            sorted(c.data["missing"], key=lambda m: m["bdf"]),
            [{"bdf": "0000:84:00.0", "tray": 4, "asic": 4}, {"bdf": "0000:c3:00.0", "tray": 3, "asic": 3}],
        )
        self.assertIn("0000:c3:00.0 (UBB3/U3)", c.details)
        self.assertIn("0000:84:00.0 (UBB4/U4)", c.details)

    def test_lspci_failure_fails(self):
        self.assertEqual(resets.recount_check("recount_1", None).status, FAIL)


class TestThermalAndTelemetry(unittest.TestCase):
    def test_healthy_unit_passes(self):
        out = _checks([_good(b) for b in all_bdfs()])
        self.assertEqual(out["asic_thermal_precheck"].status, PASS)
        self.assertEqual(out["c8_telemetry"].status, PASS)
        self.assertEqual(out["asic_thermal_precheck"].data["ignored"], [])

    def test_garbage_readings_are_ignored_not_judged(self):
        devs = [_good(b) for b in all_bdfs()]
        devs[0] = _dev(
            devs[0]["board_info"]["bus_id"], asic_temperature="65535", power="0", current="n/a", voltage="-1"
        )
        out = _checks(devs)
        self.assertEqual(out["asic_thermal_precheck"].status, PASS)
        self.assertEqual(out["c8_telemetry"].status, PASS)
        self.assertEqual(len(out["asic_thermal_precheck"].data["ignored"]), 4)
        self.assertIn("4 implausible readings ignored", out["asic_thermal_precheck"].details)
        self.assertFalse(out["asic_thermal_precheck"].console_visible)

    def test_all_garbage_names_no_hottest_chip(self):
        out = _checks([_dev(b, asic_temperature="65535") for b in all_bdfs()])
        self.assertIsNone(out["asic_thermal_precheck"].data["hottest_chip"])
        self.assertEqual(out["asic_thermal_precheck"].status, PASS)

    def test_over_limit_fails_and_names_chip(self):
        devs = [_good(b) for b in all_bdfs()]
        devs[9] = _dev("0000:42:00.0", asic_temperature="101.0", power="120", current="90", voltage="0.75")
        c = _checks(devs)["asic_thermal_precheck"]
        self.assertEqual(c.status, FAIL)
        self.assertIn("0000:42:00.0 (UBB2/U2)", c.details)
        self.assertEqual(c.data["over"], ["0000:42:00.0"])
        self.assertTrue(c.console_visible)

    def test_within_20c_of_limit_warns(self):
        devs = [_good(b) for b in all_bdfs()]
        devs[0] = _dev(
            devs[0]["board_info"]["bus_id"], asic_temperature="85.0", power="120", current="90", voltage="0.75"
        )
        self.assertEqual(_checks(devs)["asic_thermal_precheck"].status, WARN)

    def test_warm_lists_every_warm_chip(self):
        devs = [_good(b) for b in all_bdfs()]
        for i in (0, 9):
            devs[i] = _dev(
                devs[i]["board_info"]["bus_id"], asic_temperature="85.0", power="120", current="90", voltage="0.75"
            )
        c = _checks(devs)["asic_thermal_precheck"]
        self.assertEqual(c.data["warm"], [all_bdfs()[0], all_bdfs()[9]])
        self.assertIn("0000:01:00.0 (UBB1/U1)", c.details)
        self.assertIn("0000:42:00.0 (UBB2/U2)", c.details)

    def test_c8_flags_each_reading_with_tray_and_asic(self):
        devs = [_good(b) for b in all_bdfs()]
        devs[0] = _dev("0000:01:00.0", asic_temperature="76", power="321", current="401", voltage="0.60")
        c = _checks(devs)["c8_telemetry"]
        self.assertEqual(c.status, WARN)
        self.assertEqual(
            [(f["bdf"], f["tray"], f["asic"], f["why"]) for f in c.data["flagged"]],
            [("0000:01:00.0", 1, 1, w) for w in ("temperature", "power", "current", "vcore")],
        )


class TestAer(unittest.TestCase):
    def _sysfs(self, root: Path, bdf: str, vendor="0x1e52", **counts):
        d = root / bdf
        d.mkdir()
        (d / "vendor").write_text(vendor + "\n")
        for kind, fname in (
            ("corr", "aer_dev_correctable"),
            ("nonfatal", "aer_dev_nonfatal"),
            ("fatal", "aer_dev_fatal"),
        ):
            if kind in counts:
                (d / fname).write_text(f"RxErr 0\nTOTAL_ERR_{kind.upper()} {counts[kind]}\n")

    def test_read_aer_only_tenstorrent_chips(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self._sysfs(root, "0000:02:00.0", corr=3, nonfatal=0, fatal=1)
            self._sysfs(root, "0000:05:00.0", vendor="0x8086", corr=9)
            self.assertEqual(read_aer(root), {"0000:02:00.0": {"corr": 3, "nonfatal": 0, "fatal": 1}})

    def test_unreadable_counter_is_left_out(self):
        with tempfile.TemporaryDirectory() as t:
            self._sysfs(Path(t), "0000:02:00.0", corr=3)
            self.assertEqual(read_aer(Path(t)), {"0000:02:00.0": {"corr": 3}})

    def test_no_change_passes(self):
        a = {"0000:02:00.0": {"corr": 5, "nonfatal": 0, "fatal": 0}}
        self.assertEqual(aer_check(aer_new(a, a), 1.0).status, PASS)

    def test_uncorrectable_fails_with_tray_and_asic(self):
        before = {"0000:c2:00.0": {"corr": 0, "nonfatal": 0, "fatal": 0}}
        after = {"0000:c2:00.0": {"corr": 0, "nonfatal": 1, "fatal": 0}}
        c = aer_check(aer_new(before, after), 0.01)
        self.assertEqual(c.status, FAIL)
        self.assertEqual(c.data["flagged"], [{"bdf": "0000:c2:00.0", "tray": 3, "asic": 2, "nonfatal": 1}])

    def test_correctable_needs_both_thresholds(self):
        before = {"0000:02:00.0": {"corr": 0}}
        self.assertEqual(
            aer_check(aer_new(before, {"0000:02:00.0": {"corr": 1}}), 0.1).status, PASS
        )  # under the minimum
        self.assertEqual(
            aer_check(aer_new(before, {"0000:02:00.0": {"corr": 5}}), 1.0).status, PASS
        )  # under 6 per hour
        self.assertEqual(aer_check(aer_new(before, {"0000:02:00.0": {"corr": 6}}), 1.0).status, WARN)
        self.assertEqual(aer_check(aer_new(before, {"0000:02:00.0": {"corr": 2}}), 0.1).status, WARN)

    def test_a_chip_that_reappears_has_its_errors_counted(self):
        c = aer_check(aer_new({}, {"0000:02:00.0": {"nonfatal": 4, "corr": 0}}), 1.0)
        self.assertEqual(c.status, FAIL)
        self.assertEqual(c.data["flagged"], [{**chip("0000:02:00.0"), "nonfatal": 4}])

    def test_a_reset_restarts_the_counts_and_the_window_keeps_what_came_before(self):
        reads = iter(
            [
                {"0000:02:00.0": {"nonfatal": 5}},  # window opens
                {"0000:02:00.0": {"nonfatal": 6}},  # marked before a reset
                {"0000:02:00.0": {"nonfatal": 0}},  # counts restarted by the reset
                {"0000:02:00.0": {"nonfatal": 2}},  # the end
            ]
        )
        w = AerWindow(lambda: next(reads))
        w.mark()
        w.mark()
        self.assertEqual(w.new, {"0000:02:00.0": {"nonfatal": 1}})
        c = w.check()
        self.assertEqual(w.new, {"0000:02:00.0": {"nonfatal": 3}})
        self.assertEqual(c.status, FAIL)


if __name__ == "__main__":
    unittest.main()
