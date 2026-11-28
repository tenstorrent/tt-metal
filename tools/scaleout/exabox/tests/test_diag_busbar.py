#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The bus-bar stage (diag/busbar.py) against stand-in loads, a fake sysfs and the
fake in-band BMC: its step tables, the HSC poll and its lock, and each finding."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))
sys.path.insert(0, str(TESTS_DIR))

from diag import busbar, sensors  # noqa: E402
from diag.galaxy import expected_bdf, format_bdf  # noqa: E402
from diag.model import FAIL, PASS, SKIP, Phase  # noqa: E402
from diag.verdict import ROWS, verdict_lines  # noqa: E402
from test_diag_resets import fake_sysfs  # noqa: E402
from test_diag_sensors import SensorCase  # noqa: E402
from tiers import BUSBAR_STEPS, TIERS, Step, select_stages  # noqa: E402

REAL_RUN = subprocess.run
BDFS = [expected_bdf(t, a) for t in range(1, 5) for a in range(1, 9)]


class BusbarCase(SensorCase):
    """SensorCase's fake BMC, plus a fake sysfs of 32 chips and stand-in loads.

    ``tray_inject`` sets an HSC flag; ``load_variant`` makes a load a python snippet."""

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.sys = self.dir / "sys"
        fake_sysfs(self.sys)
        for i, bdf in enumerate(BDFS):
            node = self.sys / "class" / "tenstorrent" / f"tenstorrent!{i}"
            node.mkdir(parents=True)
            dev = self.sys / "bus" / "pci" / "devices" / bdf
            (node / "device").symlink_to(dev)
            (node / "tt_aiclk").write_text("800\n")
            (dev.resolve() / "current_link_speed").write_text("32.0 GT/s PCIe\n")
            (dev.resolve() / "current_link_width").write_text("16\n")
            (dev.resolve() / "hwmon" / "hwmon0").mkdir(parents=True)
            (dev.resolve() / "hwmon" / "hwmon0" / "temp1_input").write_text("45000\n")
        # The fake BMC answers ipmitool raw; anything else (the --hook script) really runs.
        run = mock.patch.object(
            sensors.subprocess, "run", lambda cmd, **kw: (self.fake.run if "raw" in cmd else REAL_RUN)(cmd, **kw)
        )
        run.start()
        self.addCleanup(run.stop)
        self.loads = {}
        self.sel = []
        self.captured = []
        for p in (
            mock.patch.object(busbar, "LEG_FILE", self.dir / "leg.json"),
            mock.patch.object(busbar, "LOADS", {**busbar.LOADS, **{k: self._load(k) for k in ("didt", "max_util")}}),
            mock.patch.object(busbar, "_read", lambda cmd, _dry: ("\n".join(self.sel), "")),
            mock.patch.object(busbar, "capture_events", lambda *a: self.captured.append(a)),
            mock.patch.object(busbar, "boot_id", lambda: "boot-a"),
            mock.patch.object(busbar, "QUIET_CAP_S", 0.2),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _load(self, kind):
        def build(_tt_metal, s):
            code = self.loads.get(kind, "print('ok')")
            return [sys.executable, "-c", code], {}

        return build

    def tray_inject(self, tray, flag_reg, value, addr=None):
        self.fake.set(tray, addr or sensors.HSC_ADDRS[0], flag_reg, value)

    def load_variant(self, kind, code):
        self.loads[kind] = code

    def run_stage(self, steps, hook=None):
        phase = Phase(name="busbar")
        busbar.run_busbar(self.dir, steps, phase, sensors.Bmc(), hook, False, self.dir / "logs", self.sys, 0.02)
        return phase, {c.name: c for c in phase.checks}


STEPS = (Step("burst1", 0, "didt", 1), Step("chunk1m", 0, "max_util", 1))


class TestSteps(unittest.TestCase):
    def test_modes_share_load_kinds_and_plain_fits_30_min(self):
        kinds = {m: {s.load for s in sts} for m, sts in BUSBAR_STEPS.items()}
        self.assertEqual(kinds["smoke"], kinds["plain"])
        self.assertEqual(kinds["plain"], kinds["full"])
        total = {m: sum(s.quiet_s + s.load_s for s in sts) for m, sts in BUSBAR_STEPS.items()}
        self.assertLessEqual(total["plain"], 1800)
        self.assertGreater(total["full"], total["plain"])
        self.assertTrue(all(s.load in busbar.LOADS for sts in BUSBAR_STEPS.values() for s in sts))

    def test_tiers_and_stage_order(self):
        self.assertEqual(
            (TIERS["daily"].load_mode, TIERS["weekly"].load_mode, TIERS["full"].load_mode), ("smoke", "plain", "full")
        )
        self.assertEqual(select_stages(TIERS["weekly"], "reset,busbar,end").load_mode, "plain")
        self.assertEqual(select_stages(TIERS["weekly"], "reset,end").load_mode, "")
        with self.assertRaises(ValueError):
            select_stages(TIERS["weekly"], "end,busbar")

    def test_only_the_under_voltage_flags_are_not_trips(self):
        # Row F02's repair covers every flag that is not a trip; a new flag needs its own row.
        flags = set(sensors.WORD_BITS.values()) | set(sensors.INPUT_BITS.values())
        self.assertEqual(flags - sensors.TRIPS, {"VIN_UV_WARN", "VIN_UV_FAULT"})


class TestParse(unittest.TestCase):
    bdfs = dict(enumerate(BDFS))

    def test_numeric_names_the_chip(self):
        text = "Device 0 PCC: 1.0\nDevice 9 output differs from the other chips\nDevice 4 PCC: 0.998\n"
        self.assertEqual(
            sorted(busbar.numeric_faults(text, self.bdfs)), sorted([format_bdf(BDFS[9]), format_bdf(BDFS[4])])
        )

    def test_hung_chip_is_the_open_sync(self):
        text = "Start sync device id: 3\nEnd sync device id: 3\nStart sync device id: 7\n"
        self.assertEqual(busbar.hung_chip(text, self.bdfs), format_bdf(BDFS[7]))
        self.assertEqual(busbar.hung_chip("Start sync device id: 3\nEnd sync device id: 3\n", self.bdfs), "")

    def test_xchip_plugin_fails_and_names_the_odd_chip(self):
        from diag import xchip

        torch = types.SimpleNamespace(equal=lambda a, b: a == b)
        warned = []
        pool = xchip._pool(torch, types.SimpleNamespace(warning=warned.append))(2)
        xchip._device_ids[:] = [10, 11, 12, 13]
        self.assertEqual(list(pool.map(torch.equal, [1, 1, 1, 1], [1, 1, 2, 1])), [True, True, False, True])
        self.assertEqual(warned, ["Device 12 output differs from the other chips"])
        self.assertEqual(list(pool.map(str, [1])), ["1"])

    def test_cross_chip_outliers(self):
        from diag.xchip import cross_chip_outliers

        eq = lambda a, b: a == b
        self.assertEqual(cross_chip_outliers([1, 1, 2, 1], eq), [2])
        self.assertEqual(cross_chip_outliers([1, 1, 1], eq), [])
        self.assertEqual(cross_chip_outliers([1, 3, 1, 3, 3], eq), [0, 2])


class TestStage(BusbarCase):
    def test_clean_steps_pass_with_their_load_line(self):
        _, by = self.run_stage(STEPS)
        self.assertEqual((by["busbar_burst1"].status, by["busbar_chunk1m"].status), (PASS, PASS))
        self.assertTrue(any(k.startswith("tray1/") for k in by["busbar_chunk1m"].data))
        self.assertNotIn("busbar_hsc_trip", by)
        self.assertFalse((self.dir / "leg.json").exists())

    def test_loads_run_from_the_tt_metal_checkout(self):
        self.load_variant("didt", "import os; print(os.getcwd())")
        here = os.getcwd()
        self.addCleanup(os.chdir, here)
        os.chdir(self.sys)
        self.run_stage(STEPS[:1])
        self.assertEqual((self.dir / "logs" / "busbar_burst1.log").read_text().strip(), str(self.dir))

    def test_missing_load_binary_fails_its_step_and_the_stage_goes_on(self):
        with mock.patch.dict(busbar.LOADS, {"didt": lambda _t, _s: (["/nonexistent/didt"], {})}):
            _, by = self.run_stage(STEPS)
        self.assertEqual((by["busbar_burst1"].status, by["busbar_chunk1m"].status), (FAIL, PASS))
        self.assertNotIn("busbar", by)

    def test_numeric_fault_fails_naming_the_chip(self):
        self.load_variant("didt", "print('Device 5 output differs from the other chips')")
        _, by = self.run_stage(STEPS)
        self.assertEqual(by["busbar_numeric"].status, FAIL)
        self.assertEqual(by["busbar_numeric"].parts, [format_bdf(BDFS[5])])

    def test_hsc_trip_fails_on_its_tray(self):
        self.tray_inject(2, 0x79, 0x0040)
        phase, by = self.run_stage(STEPS)
        self.assertEqual(by["busbar_hsc_trip"].status, FAIL)
        self.assertIn("tray 2", by["busbar_hsc_trip"].details)
        from dataclasses import asdict

        report = {"phases": {"busbar": asdict(phase)}}
        self.assertTrue(any(ROWS["F01"].do.replace("that tray", "UBB2") in line for line in verdict_lines(report)))

    def test_hang_is_captured_before_the_load_is_stopped(self):
        self.load_variant("didt", "import time; print('Start sync device id: 6', flush=True); time.sleep(60)")
        alive = []
        with mock.patch.object(busbar, "BRINGUP_S", 0.5), mock.patch.object(
            busbar, "HANG_GRACE_S", 0
        ), mock.patch.object(busbar, "capture_events", lambda *a: alive.append(True)), mock.patch.object(
            busbar, "kill_process_group", wraps=busbar.kill_process_group
        ) as stop:
            _, by = self.run_stage(STEPS)
        self.assertEqual(alive, [True])
        stop.assert_called_once()
        self.assertEqual(by["busbar_hang"].parts, [format_bdf(BDFS[6])])
        self.assertIn("access stopped", by["busbar_hang"].details)
        self.assertNotIn("busbar_chunk1m", by)  # the stage ends at the hang

    def test_lost_chip_ends_the_loads(self):
        lost = format_bdf(BDFS[3])
        counts = iter([(32, []), (31, [lost])])
        with mock.patch.object(busbar, "recount_chips", lambda _r: next(counts)):
            _, by = self.run_stage(STEPS)
        self.assertEqual((by["busbar_chip_lost"].status, by["busbar_chip_lost"].parts), (FAIL, [lost]))
        self.assertNotIn("busbar_chunk1m", by)

    def test_reboot_during_a_leg_is_a_power_fail(self):
        (self.dir / "leg.json").write_text(json.dumps({"leg": "chunk25m", "boot_id": "boot-old"}))
        _, by = self.run_stage(STEPS)
        self.assertEqual(by["busbar_reboot"].details, "host rebooted during chunk25m -> POWER: PSU/feed, see SEL")

    def test_new_sel_power_line_is_a_power_fail(self):
        self.load_variant("didt", "print('ok')")
        orig = busbar._after_step

        def after(*a):
            if not self.sel:
                self.sel.append("1 | Power Supply #0x51 | Power Supply AC lost | Asserted")
            return orig(*a)

        with mock.patch.object(busbar, "_after_step", after):
            phase, by = self.run_stage(STEPS)
        self.assertIn("host power event during burst1 -> POWER", by["busbar_reboot"].details)
        self.assertEqual([c.name for c in phase.checks].count("busbar_reboot"), 1)  # the next step's baseline has it

    def test_quiet_clock_cap_names_the_chips(self):
        (self.sys / "class" / "tenstorrent" / "tenstorrent!2" / "tt_aiclk").write_text("1350\n")
        _, by = self.run_stage(STEPS[:1])
        self.assertEqual(by["busbar_quiet_clock"].parts, [format_bdf(BDFS[2])])

    def test_link_change_is_a_warn(self):
        dev = (self.sys / "bus" / "pci" / "devices" / BDFS[1]).resolve()
        self.load_variant("didt", f"open({str(dev / 'current_link_width')!r}, 'w').write('8\\n')")
        _, by = self.run_stage(STEPS[:1])
        self.assertEqual(by["busbar_pcie_link"].parts, [format_bdf(BDFS[1])])

    def test_power_virus_slot(self):
        pv = (Step("power_virus", 0, "power_virus", 5),)
        _, by = self.run_stage(pv)
        self.assertEqual(by["busbar_power_virus"].status, SKIP)
        hook = self.dir / "hook.sh"
        hook.write_text(f'#!/bin/sh\necho "$@" > {self.dir}/hook.args\n')
        hook.chmod(0o755)
        _, by = self.run_stage(pv, hook=str(hook))
        self.assertEqual(by["busbar_power_virus"].status, PASS)
        self.assertEqual((self.dir / "hook.args").read_text().split()[:3], ["load", "power_virus", "5"])

    def test_sigterm_stops_the_load_cleanly(self):
        self.load_variant("didt", "import time; time.sleep(60)")
        got = []
        before = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, before)
        outer = lambda *_: got.append(1)  # noqa: E731 - the runner's own handler
        signal.signal(signal.SIGTERM, outer)
        timer = threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGTERM))
        timer.daemon = True
        self.addCleanup(timer.cancel)
        timer.start()
        t0 = time.monotonic()
        _, by = self.run_stage(STEPS)
        self.assertLess(time.monotonic() - t0, 30)
        self.assertEqual(by["busbar_burst1"].details, "stopped by SIGTERM at burst1")
        self.assertIs(signal.getsignal(signal.SIGTERM), outer)
        self.assertEqual(got, [1])  # passed on once the load is stopped


class TestPoll(SensorCase):
    def test_poll_gets_the_lock_between_other_holders(self):
        """Others' short holds (the brick reads, another tool) delay the poll by less than a period."""
        bmc, poll, stop = sensors.Bmc(), sensors.Poll(), threading.Event()
        done = threading.Event()

        def other():
            while not done.is_set():
                with bmc.locked(who="other tool"):
                    time.sleep(0.02)  # about one ipmitool raw call
                time.sleep(0.01)

        threads = [
            threading.Thread(target=other),
            threading.Thread(target=sensors.poll_hsc, args=(bmc, poll, stop, 0.1)),
        ]
        for t in threads:
            t.start()
        time.sleep(1.0)
        stop.set()
        done.set()
        for t in threads:
            t.join(5)
        self.assertEqual(poll.gaps, [])
        self.assertTrue(poll.steps["start"].peak_a)

    def test_a_long_hold_is_logged_naming_its_holder(self):
        bmc, poll, stop = sensors.Bmc(), sensors.Poll(), threading.Event()
        lock = os.environ["HSC_I2C_LOCK"]
        poller = threading.Thread(target=sensors.poll_hsc, args=(bmc, poll, stop, 0.05))
        with open(lock, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.truncate(0)
            f.write("tray-test pid 4242\n")
            f.flush()
            poller.start()
            time.sleep(0.5)
        time.sleep(0.3)
        stop.set()
        poller.join(5)
        self.assertTrue(poll.gaps)
        self.assertIn("tray-test pid 4242", poll.gaps[0])


if __name__ == "__main__":
    unittest.main()
