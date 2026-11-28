#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""reset_loop on a fake sysfs: every reset is -glx_reset, followed by a rescan of
the root ports read before it and a recount of the chips by BDF."""

from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from diag import resets  # noqa: E402
from diag.model import FAIL, PASS, SKIP, WARN, Phase  # noqa: E402


def root_port(bdf: str) -> str:
    return f"0000:{bdf[5]}0:0{bdf[6]}.1"


def fake_sysfs(root: Path) -> None:
    """The 32 chips, each behind its own root port."""
    for bdf in resets.EXPECTED_BDFS:
        port = root / "devices" / f"pci0000:{bdf[5]}0" / root_port(bdf)
        chip = port / bdf
        chip.mkdir(parents=True)
        (chip / "vendor").write_text("0x1e52\n")
        (chip / "config").write_bytes(b"\x52\x1e\x40\xb1\x06\x00")
        (port / "rescan").write_text("")
        for dev in (port, chip):
            link = root / "bus" / "pci" / "devices" / dev.name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(dev)


def drop_tray(root: Path, tray_nibble: str) -> None:
    for bdf in resets.EXPECTED_BDFS:
        if bdf[5] == tray_nibble and (root / "bus" / "pci" / "devices" / bdf).is_symlink():
            shutil.rmtree((root / "bus" / "pci" / "devices" / bdf).resolve())
            (root / "bus" / "pci" / "devices" / bdf).unlink()


class TestResetLoop(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        fake_sysfs(self.root)
        self.calls = []

    def _loop(self, plan, on_reset=None, rc=0, **kw) -> Phase:
        def fake_run(cmd, dry_run=False, **kw):
            self.calls.append(cmd)
            if on_reset:
                on_reset()
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")

        phase = Phase(name="reset_loop")
        with mock.patch.object(resets, "run", fake_run), contextlib.redirect_stdout(io.StringIO()):
            resets.reset_loop("tt-smi", plan, phase, dry_run=False, sys_root=self.root, **kw)
        return phase

    def test_every_chip_back_passes_and_every_root_port_is_rescanned(self):
        phase = self._loop(("-glx_reset", "-glx_reset"))
        self.assertEqual([c.status for c in phase.checks], [PASS, PASS])
        self.assertEqual(self.calls, [["tt-smi", "-glx_reset"]] * 2)
        rescans = {(p / "rescan").read_text() for p in (self.root / "bus" / "pci" / "devices").glob("0000:?0:0?.1")}
        self.assertEqual(rescans, {"1"})

    def test_a_tray_lost_in_the_reset_is_named_and_stops_the_loop(self):
        phase = self._loop(("-glx_reset", "-glx_reset"), on_reset=lambda: drop_tray(self.root, "c"))
        check = phase.checks[0]
        self.assertEqual([c.name for c in phase.checks], ["reset_1_glx_reset", "reset_1_glx_reset_round2"])
        self.assertEqual(check.status, FAIL)
        self.assertEqual(check.data["post_pcie_count"], 24)
        self.assertEqual(len(check.parts), 8)
        self.assertIn("0000:c1:00.0 (UBB3/U1)", check.parts)
        self.assertIn("0000:c1:00.0 (UBB3/U1)", check.details)
        # The lost tray's root ports were read before the reset, so they are rescanned too.
        self.assertEqual((self.root / "devices" / "pci0000:c0" / "0000:c0:01.1" / "rescan").read_text(), "1")

    def test_a_rescan_that_cannot_run_is_noted_in_one_line(self):
        port = self.root / "devices" / "pci0000:40" / "0000:40:01.1"
        (port / "rescan").unlink()
        (port / "rescan").mkdir()
        (check,) = self._loop(("-glx_reset",)).checks
        self.assertEqual(check.status, PASS)
        self.assertIn("rescan failed", check.details)
        self.assertIn("0000:40:01.1", check.details)
        with mock.patch.object(Path, "write_text", side_effect=PermissionError(13, "Permission denied")):
            (check,) = self._loop(("-glx_reset",)).checks
        self.assertEqual(check.status, PASS)
        self.assertIn("(root-port rescan needs root (32 port(s) skipped))", check.details)
        self.assertNotIn("Permission denied", check.details)

    def test_each_chip_maps_to_its_root_port(self):
        ports = resets.chip_root_ports(self.root)
        self.assertEqual(len(ports), 32)
        self.assertEqual(ports["0000:44:00.0"], "0000:40:04.1")

    def test_the_only_reset_is_glx_reset(self):
        # tiers.py checks every tier's resets against these flags when it loads.
        self.assertEqual(resets.RESET_FLAGS, ("-glx_reset",))

    def test_a_refusing_gate_skips_the_reset_and_says_why(self):
        phase = self._loop(("-glx_reset", "-glx_reset"), gate="sh -c 'echo busy: someone holds the chips; exit 3'")
        (check,) = phase.checks
        self.assertEqual(
            (check.status, check.details), (SKIP, "reset cannot run: reset gate: busy: someone holds the chips")
        )
        self.assertEqual(self.calls, [])

    def test_a_passing_gate_lets_the_reset_run(self):
        phase = self._loop(("-glx_reset",), gate="true")
        self.assertEqual([c.status for c in phase.checks], [PASS])

    def test_chips_back_on_the_second_round(self):
        lost = []

        def once():
            if not lost:
                lost.append(1)
                drop_tray(self.root, "c")

        stash = Path(tempfile.mkdtemp(dir=self.root.parent))
        self.addCleanup(shutil.rmtree, stash)
        shutil.copytree(self.root, stash / "copy", symlinks=True)

        def on_reset():
            if lost:  # the second reset brings the tray back
                shutil.rmtree(self.root)
                shutil.copytree(stash / "copy", self.root, symlinks=True)
            once()

        phase = self._loop(("-glx_reset",), on_reset=on_reset)
        self.assertEqual(
            [(c.name, c.status) for c in phase.checks],
            [("reset_1_glx_reset", WARN), ("reset_1_glx_reset_round2", PASS)],
        )

    def test_the_hook_runs_after_a_tray_stays_down_then_a_recount(self):
        hook = self.root.parent / f"{self.root.name}-hook.sh"
        hook.write_text('#!/bin/sh\necho "$@" > "$4.args"\n')
        hook.chmod(0o755)
        self.addCleanup(hook.unlink)
        logs = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, logs)
        phase = self._loop(
            ("-glx_reset", "-glx_reset"), on_reset=lambda: drop_tray(self.root, "c"), hook=str(hook), logs_dir=logs
        )
        self.assertEqual([c.status for c in phase.checks], [FAIL, FAIL, FAIL])
        self.assertEqual(phase.checks[-1].name, "recount_reset_1_glx_reset")
        self.assertIn("after the hook (rc 0): still missing 0000:c1:00.0", phase.checks[-1].details)
        args = (logs / "reset_1_glx_reset.args").read_text().split()
        self.assertEqual(args, ["after", "reset_1_glx_reset", FAIL, str(logs / "reset_1_glx_reset")])
        self.assertEqual(len(self.calls), 2)  # the loop stops: no second reset of the plan

    def test_a_failed_reset_with_every_chip_present_stays_a_fail(self):
        self.assertEqual([c.status for c in self._loop(("-glx_reset",), rc=1).checks], [FAIL])

    def test_a_tray_the_hook_brings_back_is_rescanned_and_warns(self):
        stash = Path(tempfile.mkdtemp(dir=self.root.parent))
        self.addCleanup(shutil.rmtree, stash)
        shutil.copytree(self.root, stash / "copy", symlinks=True)
        hook = stash / "hook.sh"
        hook.write_text(f'#!/bin/sh\nrm -rf "{self.root}" && cp -a "{stash}/copy" "{self.root}"\n')
        hook.chmod(0o755)
        phase = self._loop(("-glx_reset",), on_reset=lambda: drop_tray(self.root, "c"), hook=str(hook))
        self.assertEqual([c.status for c in phase.checks], [WARN, WARN, PASS])
        port = self.root / "bus" / "pci" / "devices" / "0000:c0:01.1" / "rescan"
        self.assertEqual(port.read_text(), "1")  # rescanned after the hook, not only after the resets

    def test_a_hung_hook_times_out(self):
        from diag import common

        hook = self.root / "slow-hook.sh"
        hook.write_text("#!/bin/sh\nexec sleep 5\n")
        hook.chmod(0o755)
        with mock.patch.object(common, "HOOK_TIMEOUT_S", 0.2), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(common.call_hook(str(hook), "after", "x", FAIL, self.root / "o", False), 124)

    def test_a_gate_refusing_the_second_round_stops_the_loop(self):
        gate = self.root.parent / f"{self.root.name}-gate.sh"
        gate.write_text(f'#!/bin/sh\n[ -e "{gate}.once" ] && {{ echo busy; exit 1; }}; touch "{gate}.once"\n')
        gate.chmod(0o755)
        self.addCleanup(lambda: [p.unlink(missing_ok=True) for p in (gate, Path(f"{gate}.once"))])
        phase = self._loop(("-glx_reset", "-glx_reset"), on_reset=lambda: drop_tray(self.root, "c"), gate=str(gate))
        self.assertEqual([c.status for c in phase.checks], [FAIL, SKIP])
        self.assertEqual(len(self.calls), 1)

    def test_a_missing_hook_does_not_crash(self):
        from diag.common import call_hook

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                call_hook(str(self.root / "no-such-hook"), "after", "x", FAIL, self.root / "o", False), 127
            )


class TestPostTestResetOk(unittest.TestCase):
    def _ok(self, checks):
        phase = Phase(name="post_test_reset")

        def fake_loop(tt_smi, plan, ph, dry_run, **kw):
            for c in checks:
                ph.add(c)

        with mock.patch.object(resets, "reset_loop", fake_loop), mock.patch.object(
            resets, "resolve_tt_smi", lambda p: "tt-smi"
        ):
            resets.run_post_test_reset(None, ("-glx_reset",), phase, dry_run=False)
        return phase.checks[-1]

    def test_states(self):
        from diag.model import Check

        fail, ok = Check(name="reset_1_glx_reset", status=FAIL), Check(name="reset_1_glx_reset_round2", status=PASS)
        skip = Check(name="reset_1_glx_reset", status=SKIP, details="reset cannot run: reset gate: busy")
        self.assertEqual(self._ok([ok]).status, PASS)
        self.assertEqual(self._ok([fail]).status, FAIL)
        self.assertEqual(self._ok([fail, ok]).status, WARN)
        self.assertEqual(self._ok([fail, skip]).status, FAIL)
        self.assertEqual(
            (self._ok([skip]).status, self._ok([skip]).details),
            (SKIP, "did not run: reset cannot run: reset gate: busy"),
        )


if __name__ == "__main__":
    unittest.main()
