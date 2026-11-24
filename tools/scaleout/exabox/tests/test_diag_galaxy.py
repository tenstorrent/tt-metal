#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The Galaxy chip checks without hardware: a chip is named by BDF/tray/ASIC, a reset is a glx_reset (or says why it
cannot run), the root-port rescan and the chip recount after it name the missing ones, the thermal and telemetry
checks ignore garbage readings, the AER diff counts the errors over a run, and the SEL reads are judged."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

import tiers  # noqa: E402
from diag import firmware, resets, sel  # noqa: E402
from diag.aer import AerWindow, aer_check, aer_new, read_aer  # noqa: E402
from diag.galaxy import TRAY_TO_UBB_NIBBLE, expected_bdf  # noqa: E402
from diag.model import FAIL, PASS, SKIP, WARN, Phase  # noqa: E402
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

    def test_short_trays_counted(self):
        c = resets.recount_check("recount_1", [b for b in all_bdfs() if not b.startswith("0000:4")])
        self.assertEqual(c.data["trays_short"], {2: 8})
        self.assertIn("tray 2: 8 of 8 missing", c.details)


def _ok(*_a, **_k):
    return subprocess.CompletedProcess([], 0, stdout="", stderr="")


class TestResets(unittest.TestCase):
    """glx_reset with tt-smi, sudo, lspci and the rescan writes stubbed: /sys is a temp dir with one bridge."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        rp = Path(tmp.name, "bus/pci/devices/0000:40:01.7")
        rp.mkdir(parents=True)
        (rp / "secondary_bus_number").write_text(f"{0x47}\n")
        self.rescanned = rp / "rescan"
        for target, value in (("SYS", Path(tmp.name)), ("RESCAN_EVERY_S", 0), ("run", _ok)):
            patcher = mock.patch.object(resets, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _reset(self, listings, root=True, gate=None):
        """glx_reset with pcie_bdfs giving *listings* in turn (the last one again after them)."""
        it = iter(listings)
        last = []

        def bdfs():
            last[:] = [next(it, last[0] if last else None)]
            return last[0]

        phase = resets.Phase(name="t")
        why = "" if root else "not root, and sudo -n -l does not list tt-smi -glx_reset"
        with mock.patch.object(resets, "root_cmd", lambda c: (c if root else None, why)), mock.patch.object(
            resets, "pcie_bdfs", bdfs
        ):
            out = resets.glx_reset("tt-smi", phase, "reset_1", "recount_1", False, gate=gate)
        return out, {c.name: c for c in phase.checks}

    def test_a_reset_that_cannot_run_says_why_and_runs_nothing(self):
        with mock.patch.object(resets, "run", side_effect=AssertionError("ran")):
            out, checks = self._reset([], root=False)
        self.assertIsNone(out)
        self.assertEqual(checks["reset_1"].status, SKIP)
        self.assertEqual(
            checks["reset_1"].details, "reset cannot run: not root, and sudo -n -l does not list tt-smi -glx_reset"
        )

    def test_the_reset_gate_can_refuse(self):
        out, checks = self._reset([], gate="sh -c 'echo serving; exit 1'")
        self.assertIsNone(out)
        self.assertEqual(checks["reset_1"].details, "reset cannot run: reset gate: serving")

    def test_all_back_needs_no_rescan(self):
        out, checks = self._reset([all_bdfs()])
        self.assertEqual(len(out), 32)
        self.assertEqual((checks["reset_1"].status, checks["recount_1"].status), (PASS, PASS))
        self.assertNotIn("rescan_reset_1", checks)

    def test_a_chip_back_only_after_the_rescan_is_named(self):
        gone = [b for b in all_bdfs() if b != "0000:47:00.0"]
        out, checks = self._reset([gone, all_bdfs()])
        self.assertEqual(self.rescanned.read_text(), "1")
        self.assertEqual(checks["rescan_reset_1"].status, WARN)
        self.assertIn(
            "0000:47:00.0 back only after a rescan of root port 0000:40:01.7", checks["rescan_reset_1"].details
        )
        self.assertEqual(checks["recount_1"].status, PASS)

    def test_a_chip_still_missing_is_named_with_its_tray(self):
        gone = [b for b in all_bdfs() if b != "0000:47:00.0"]
        with mock.patch.object(resets, "RESCAN_WAIT_S", 0.01):
            out, checks = self._reset([gone])
        self.assertEqual(len(out), 31)
        self.assertEqual(checks["recount_1"].status, FAIL)
        self.assertIn("tray 2: 1 of 8 missing", checks["recount_1"].details)

    def test_the_tiers_never_run_tt_smi_r(self):
        self.assertEqual(resets.RESET_FLAGS, ("-glx_reset",))
        for name, tier in tiers.TIERS.items():
            self.assertNotIn("-r", (*tier.resets, *tier.post_test_resets), name)

    def test_recover_returns_its_finding_lines(self):
        args = argparse.Namespace(skip_reset=False, tt_smi_path=None, dry_run=False, reset_gate=None)
        with mock.patch.object(resets, "resolve_tt_smi", return_value="tt-smi"), mock.patch.object(
            resets, "root_cmd", return_value=(None, "not root")
        ):
            self.assertEqual(resets.recover(args, Path("/nonexistent")), ["reset cannot run: not root"])
        self.assertEqual(
            resets.recover(argparse.Namespace(skip_reset=True), Path(".")), ["reset cannot run: --skip-reset"]
        )

    def test_post_test_reset_that_cannot_run_is_a_skip(self):
        phase = resets.Phase(name="post_test_reset")
        with mock.patch.object(resets, "resolve_tt_smi", return_value="tt-smi"), mock.patch.object(
            resets, "root_cmd", return_value=(None, "not root")
        ):
            self.assertTrue(resets.run_post_test_reset(None, ("-glx_reset",), phase, False))
        self.assertEqual((phase.checks[-1].status, phase.checks[-1].details), (SKIP, "reset cannot run: not root"))


class TestStageOrder(unittest.TestCase):
    def test_stages_in_run_order(self):
        weekly = tiers.TIERS["weekly"]
        cut, steps = tiers.select_stages(weekly, "reset,busbar,links,end")
        self.assertEqual(
            (cut.resets, cut.tests, cut.post_test_resets), (weekly.resets, ("busbar", "links"), ("-glx_reset",))
        )
        cut, steps = tiers.select_stages(weekly, "busbar_heavy")
        self.assertEqual((cut.resets, cut.tests, cut.post_test_resets, steps), ((), ("busbar",), (), ("heavy",)))
        cut, _ = tiers.select_stages(tiers.TIERS["flash"], "reset,end")
        self.assertEqual((cut.resets, cut.tests, cut.post_test_resets), ((), (), ("-glx_reset",)))

    def test_an_order_the_stages_do_not_run_in_is_refused(self):
        for order in ("links,busbar", "end,reset", "busbar_mix,busbar_qb"):
            with self.subTest(order=order), self.assertRaisesRegex(ValueError, "the stages run in the order"):
                tiers.select_stages(tiers.TIERS["weekly"], order)


class TestFirmwareReads(unittest.TestCase):
    def test_reads_that_could_not_be_made_are_a_skip_with_every_chip_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n, bdf in enumerate(("0000:01:00.0", "0000:41:00.0")):
                dev = root / "devices" / bdf
                dev.mkdir(parents=True)
                node = root / "class/tenstorrent" / f"tenstorrent!{n}"
                node.mkdir(parents=True)
                (node / "device").symlink_to(dev)
            no_device = lambda argv, out: out.write_text('{"err": "no device in tests"}\n')  # noqa: E731
            with mock.patch.object(firmware, "SYS", root), mock.patch.object(firmware, "_step", no_device):
                c = firmware.firmware_check(root / "out")
            self.assertIn("C1 INFO SPI images not checked: no device in tests", c.data["not_checked"])
            self.assertEqual((root / "out/chips.txt").read_text(), "0000:01:00.0 0\n0000:41:00.0 1\n")
            self.assertEqual((root / "out/trays.txt").read_text(), "01:00.0 tray 1\n41:00.0 tray 2\n")
        self.assertEqual((c.name, c.status, c.data["findings"]), ("firmware_reads", SKIP, []))


def _sel(when, sensor, event, state="Asserted"):
    return f"   1 | {time.strftime('%m/%d/%Y | %H:%M:%S', time.localtime(when))} | {sensor} | {event} | {state}"


class TestSel(unittest.TestCase):
    def test_history_warns_per_tray(self):
        now = time.time()
        rows = [_sel(now - 3600, "Processor UBB1_FRB3", "FRB3/Processor Startup/Initialization failure")]
        rows += [_sel(now - 60, "Memory #0x1", "Bus Correctable Error")] * 19
        rows += [_sel(now - 8 * 86400, "Processor UBB0_FRB3", "FRB3/Processor Startup/Initialization failure")]
        c = sel.history_check(rows, "", now)
        self.assertEqual(c.status, WARN)
        self.assertEqual(c.details, "tray 2: 1 FRB3 processor init failures in 7 days")

    def test_new_thermal_trip_fails_its_tray(self):
        now = time.time()
        before = [_sel(now - 60, "Processor UBB1_FRB3", "FRB3/Processor Startup/Initialization failure")]
        after = before + [
            _sel(now, "Temperature UBB2_TEMP", "Upper Critical going high"),
            _sel(now, "Power Supply PSU3", "Failure detected"),
            _sel(now, "Temperature UBB0_TEMP", "Upper Critical going high", "Deasserted"),
        ]
        c = sel.new_check(before, after, "")
        self.assertEqual(c.status, FAIL)
        self.assertEqual(
            c.data["findings"],
            [
                "thermal trip tray 3: UBB2_TEMP Upper Critical going high Asserted",
                "host: PSU3 Failure detected Asserted",
            ],
        )

    def test_not_read_is_a_skip_that_says_why(self):
        self.assertEqual(sel.history_check(None, "not root", 0).details, "SEL not read: not root")
        self.assertEqual(sel.new_check([], None, "not root").status, SKIP)


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

    def _window(self, *nonfatal):
        """An AerWindow over reads of one chip's full set of counters, nonfatal taking the given values in turn."""
        reads = iter([{"0000:02:00.0": {"corr": 0, "nonfatal": n, "fatal": 0}} for n in nonfatal])
        return AerWindow(lambda: next(reads))

    def test_a_reset_restarts_the_counts_and_the_window_keeps_what_came_before(self):
        w = self._window(5, 6, 0, 2)  # opens; marked before a reset; as the reset left it; the end
        w.mark()
        w.rebase()
        self.assertEqual(w.new, {"0000:02:00.0": {"nonfatal": 1}})
        c = w.check()
        self.assertEqual(w.new, {"0000:02:00.0": {"nonfatal": 3}})
        self.assertEqual(c.status, FAIL)

    def test_errors_after_a_reset_count_even_when_they_reach_the_old_total(self):
        for end, total in ((6, 7), (9, 10)):
            with self.subTest(end=end):
                w = self._window(5, 6, 0, end)
                w.mark()
                w.rebase()
                self.assertEqual((w.check().status, w.new["0000:02:00.0"]["nonfatal"]), (FAIL, total))

    def test_a_reset_that_keeps_the_counts_does_not_count_them_twice(self):
        w = self._window(5, 6, 6, 8)
        w.mark()
        w.rebase()
        w.check()
        self.assertEqual(w.new, {"0000:02:00.0": {"nonfatal": 3}})

    def test_no_counter_readable_is_skip_not_pass(self):
        for read in ({}, {"0000:02:00.0": {}}):
            with self.subTest(read=read):
                c = AerWindow(lambda: read).check()
                self.assertEqual(c.status, SKIP)
                self.assertIn("no PCIe AER counter readable", c.details)

    def test_counters_not_read_warn_and_are_named(self):
        full = {"corr": 0, "nonfatal": 0, "fatal": 0}
        reads = iter(
            [
                {"0000:02:00.0": full, "0000:03:00.0": {"corr": 0}},
                {"0000:02:00.0": full, "0000:03:00.0": {"corr": 0}},
            ]
        )
        c = AerWindow(lambda: next(reads)).check()
        self.assertEqual(c.status, WARN)
        self.assertIn("0000:03:00.0: nonfatal, fatal not read", c.details)
        self.assertEqual(c.data["missing"], [{**chip("0000:03:00.0"), "counters": ["nonfatal", "fatal"]}])

    def test_a_chip_gone_at_the_end_has_its_counters_not_read(self):
        full = {"corr": 0, "nonfatal": 0, "fatal": 0}
        reads = iter([{"0000:02:00.0": full, "0000:03:00.0": full}, {"0000:02:00.0": full}])
        c = AerWindow(lambda: next(reads)).check()
        self.assertEqual(c.status, WARN)
        self.assertEqual([m["bdf"] for m in c.data["missing"]], ["0000:03:00.0"])

    def test_uncorrectable_errors_still_fail_with_counters_missing(self):
        self.assertEqual(aer_check({"0000:02:00.0": {"fatal": 1}}, 1.0, 2, {"0000:03:00.0": ["corr"]}).status, FAIL)


if __name__ == "__main__":
    unittest.main()
