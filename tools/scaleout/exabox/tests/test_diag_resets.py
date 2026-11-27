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
from diag.model import FAIL, PASS, Phase  # noqa: E402
from tiers import TIERS  # noqa: E402


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
        if bdf[5] == tray_nibble:
            shutil.rmtree((root / "bus" / "pci" / "devices" / bdf).resolve())
            (root / "bus" / "pci" / "devices" / bdf).unlink()


class TestResetLoop(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        fake_sysfs(self.root)
        self.calls = []

    def _loop(self, plan, on_reset=None) -> Phase:
        def fake_run(cmd, dry_run=False, **kw):
            self.calls.append(cmd)
            if on_reset:
                on_reset()
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        phase = Phase(name="reset_loop")
        with mock.patch.object(resets, "run", fake_run), contextlib.redirect_stdout(io.StringIO()):
            resets.reset_loop("tt-smi", plan, phase, dry_run=False, sys_root=self.root)
        return phase

    def test_every_chip_back_passes_and_every_root_port_is_rescanned(self):
        phase = self._loop(("-glx_reset", "-glx_reset"))
        self.assertEqual([c.status for c in phase.checks], [PASS, PASS])
        self.assertEqual(self.calls, [["tt-smi", "-glx_reset"]] * 2)
        rescans = {(p / "rescan").read_text() for p in (self.root / "bus" / "pci" / "devices").glob("0000:?0:0?.1")}
        self.assertEqual(rescans, {"1"})

    def test_a_tray_lost_in_the_reset_is_named_and_stops_the_loop(self):
        phase = self._loop(("-glx_reset", "-glx_reset"), on_reset=lambda: drop_tray(self.root, "c"))
        (check,) = phase.checks
        self.assertEqual(check.status, FAIL)
        self.assertEqual(check.data["post_pcie_count"], 24)
        self.assertEqual(len(check.parts), 8)
        self.assertIn("0000:c1:00.0 (UBB3/U1)", check.parts)
        self.assertIn("0000:c1:00.0 (UBB3/U1)", check.details)
        # The lost tray's root ports were read before the reset, so they are rescanned too.
        self.assertEqual((self.root / "devices" / "pci0000:c0" / "0000:c0:01.1" / "rescan").read_text(), "1")

    def test_a_rescan_that_cannot_be_written_is_noted(self):
        port = self.root / "devices" / "pci0000:40" / "0000:40:01.1"
        (port / "rescan").unlink()
        (port / "rescan").mkdir()
        (check,) = self._loop(("-glx_reset",)).checks
        self.assertEqual(check.status, PASS)
        self.assertIn("rescan failed", check.details)
        self.assertIn("0000:40:01.1", check.details)

    def test_no_tier_resets_with_tt_smi_r(self):
        self.assertEqual(resets.RESET_FLAGS, ("-glx_reset",))
        for name, tier in TIERS.items():
            self.assertNotIn("-r", tier.resets + tier.post_test_resets, name)


if __name__ == "__main__":
    unittest.main()
