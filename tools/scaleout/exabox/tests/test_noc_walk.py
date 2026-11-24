#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""noc_walk.py's lines function, without a device: the NUMERIC FAIL / WARN / summary lines of the seeded per-chip
NOC data check, from a check result dict, including the common-fault fold."""

import ast
import inspect
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent / "health_check_test_suite" / "galaxy"))

import noc_walk  # noqa: E402


def chip(bad=0, unfinished=0, examples=None):
    return dict(bad=bad, unfinished=unfinished, noc_bad={0: bad, 1: unfinished}, examples=examples or [])


CHIPS = {i: dict(pci=f"0000:{i:02x}:00.0", tray=(i // 8) + 1, dev=i) for i in range(32)}


def row(chips=None, **kw):
    """A noc_check() result on CHIPS: every chip clean but those in chips (mesh index -> chip()), kw over the rest."""
    r = dict(passes=3, transfers=12, seed=0x5EED, grid=(10, 8), start=1000.0, end=1010.0, error=None)
    return {**r, **kw, "chips": {i: chip() for i in range(32)} | (chips or {})}


class NocWalkLines(unittest.TestCase):
    def test_clean(self):
        out = noc_walk.nocwalk_lines(row(), CHIPS)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].startswith("NUMERIC NOCWALK "))
        self.assertIn("every chip matched", out[0])

    def test_one_chip_flagged(self):
        ex = dict(src=3, dst=7, noc=1, done=True, bad=2, off=8, want=0xABCD1234, got=0xABCD1235, pat="PRBS")
        out = noc_walk.nocwalk_lines(row(chips={5: chip(bad=2, examples=[ex])}), CHIPS)
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0].startswith("NUMERIC FAIL 05:00.0 (tray 1) chip 5"))
        self.assertIn("NOC data check", out[0])
        self.assertTrue(out[1].startswith("NUMERIC NOCWALK "))

    def test_all_chips_flagged_is_a_warn(self):
        ex = dict(src=0, dst=1, noc=0, done=True, bad=1, off=0, want=1, got=2, pat="address")
        out = noc_walk.nocwalk_lines(row(chips={i: chip(bad=1, examples=[ex]) for i in range(32)}), CHIPS)
        self.assertEqual(len(out), 2)  # WARN + summary, no per-chip FAIL
        self.assertTrue(out[0].startswith("NUMERIC WARN "))
        self.assertIn("all 32 chips failed", out[0])
        self.assertIn("every chip mismatched", out[1])

    def test_error_is_a_warn(self):
        out = noc_walk.nocwalk_lines(row(passes=0, transfers=0, error="RuntimeError: boom"), CHIPS)
        self.assertTrue(any(l.startswith("NUMERIC WARN NOC data check") and "not checked" in l for l in out))
        self.assertIn("no chip mismatched", out[-1])

    def test_error_before_the_check_is_a_warn(self):
        """fabric_open.open_check's row when the check raised before it started: error and chips only."""
        out = noc_walk.nocwalk_lines(dict(error="ImportError: torch", chips={}, start=0.0, end=1.0), CHIPS)
        self.assertEqual(out, [f"NUMERIC WARN NOC data check {noc_walk.WHY} not checked: ImportError: torch"])

    def test_tray_order(self):
        # chips ordered by tray then BDF, not mesh index
        ex = dict(src=0, dst=1, noc=0, done=True, bad=1, off=0, want=0, got=1, pat="walking-1")
        out = noc_walk.nocwalk_lines(row(chips={20: chip(bad=1, examples=[ex]), 3: chip(bad=1, examples=[ex])}), CHIPS)
        fails = [l for l in out if l.startswith("NUMERIC FAIL")]
        self.assertEqual(len(fails), 2)
        self.assertEqual(fails[0].split(" ")[2], "03:00.0")  # tray 1 before tray 3
        self.assertEqual(fails[1].split(" ")[2], "14:00.0")

    def test_end_name(self):
        self.assertEqual(noc_walk.end_name(17, 10), "core 7,1")
        self.assertEqual(noc_walk.end_name(-3, 10), "bank 2")

    def test_fabric_open_call_binds(self):
        """fabric_open.py's noc_check call matches noc_check's signature (fabric_open runs a device at import)."""
        src = (TESTS_DIR.parent / "health_check_test_suite" / "galaxy" / "fabric_open.py").read_text()
        call = next(
            n
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "noc_check"
        )
        inspect.signature(noc_walk.noc_check).bind(*call.args, **{k.arg: None for k in call.keywords})


if __name__ == "__main__":
    unittest.main()
