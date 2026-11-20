#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The galaxy tier's load phases without hardware: the bus-bar plan per mode, every planned flag exists in its
program, --plan prints it, a step is judged by its exit, its FAIL lines and its hang, the pair rounds never put a
chip in two pairs, and the UMD index follows the sorted BDFs of a fake sysfs."""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

import tiers  # noqa: E402
from diag import busbar, links  # noqa: E402
from diag.aer import aer_check, aer_new  # noqa: E402
from diag.model import FAIL, PASS, SKIP, Check, Phase  # noqa: E402


def _steps(mode, **kw):
    return {s.name: s for s in busbar.plan(mode, **kw)}


class TestPlan(unittest.TestCase):
    def test_order_is_the_same_in_every_mode(self):
        for mode in busbar.MODES:
            self.assertEqual([s.name for s in busbar.plan(mode)], ["qb", "mix", "step", "heavy", "code"])

    def test_plain_lengths(self):
        p = _steps("plain")
        self.assertEqual(p["qb"].seconds, 180)
        self.assertIn("1,2,3,4", p["mix"].argv)
        self.assertEqual(p["step"].seconds, 100)
        self.assertEqual(p["heavy"].seconds, 100)
        self.assertNotIn("--l1-full", p["mix"].argv)

    def test_full_is_plain_longer(self):
        p, f = _steps("plain"), _steps("full")
        self.assertIn("1,2,3,25", f["mix"].argv)
        self.assertIn("--l1-full", f["mix"].argv)
        self.assertIn("12.5", f["heavy"].argv)
        for name in p:
            self.assertGreaterEqual(f[name].seconds, p[name].seconds, name)

    def test_smoke_caps_each_length(self):
        s = _steps("smoke")
        argv = s["qb"].argv
        self.assertEqual(argv[argv.index("--idle-s") + 1], "2")
        self.assertEqual(argv[argv.index("--burst-s") + 1], "2")
        self.assertLess(sum(x.seconds for x in s.values()), 120)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            busbar.plan("long")

    def test_every_flag_exists_in_its_program(self):
        for mode in busbar.MODES:
            for step in busbar.plan(mode):
                src = (busbar.GALAXY_DIR / step.argv[0]).read_text()
                for flag in (a for a in step.argv[1:] if a.startswith("--")):
                    self.assertIn(f'"{flag}"', src, f"{step.name}: {step.argv[0]} has no {flag}")

    def test_plan_cli(self):
        out = subprocess.run(
            [sys.executable, str(SUITE_DIR / "diag_runner.py"), "--tier", "galaxy", "--mode", "smoke", "--plan"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        self.assertEqual([ln.split()[0] for ln in out], ["qb", "mix", "step", "heavy", "code"])


class TestRunStep(unittest.TestCase):
    def _run(self, body, seconds=5):
        with tempfile.TemporaryDirectory() as d:
            prog = Path(d) / "prog.py"
            prog.write_text(body)
            with mock.patch.object(busbar, "GALAXY_DIR", Path(d)), mock.patch.object(busbar, "SETUP_S", 0):
                return busbar.run_step(busbar.Step("x", "test", ("prog.py", "--out", "{out}"), seconds), Path(d))

    def test_pass(self):
        self.assertEqual(self._run("print('ok')").status, PASS)

    def test_rc(self):
        c = self._run("import sys; print('lost'); sys.exit(3)")
        self.assertEqual((c.status, c.data["rc"]), (FAIL, 3))

    def test_fail_line(self):
        c = self._run("print('chip 3: NUMERIC FAIL 0000:41:00.0')")
        self.assertEqual(c.status, FAIL)
        self.assertIn("NUMERIC FAIL", c.details)

    def test_hang(self):
        c = self._run("import time; time.sleep(30)", seconds=1)
        self.assertEqual(c.status, FAIL)
        self.assertTrue(c.data["hung"])

    def test_dry_run(self):
        c = busbar.run_step(busbar.plan("smoke")[0], Path("/nonexistent"), dry_run=True)
        self.assertEqual(c.status, SKIP)

    def test_a_failed_part_does_not_end_the_phase(self):
        calls = []

        def fake(step, out, dry_run):
            calls.append(step.name)
            return busbar.Check(name=step.name, status=FAIL if step.name == "qb" else PASS)

        phase = Phase(name="busbar")
        with mock.patch.object(busbar, "run_step", fake):
            self.assertTrue(busbar.run_busbar(phase, Path("/tmp"), "smoke"))
        self.assertEqual(calls, ["qb", "mix", "step", "heavy", "code"])


class TestLinks(unittest.TestCase):
    def setUp(self):  # a 2x3 grid, two channels per link
        self.links = {c: {} for c in range(6)}
        for i, (a, b) in enumerate([(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]):
            for ch in (2 * i, 2 * i + 1):
                self.links[a][ch] = [b, ch + 20]
                self.links[b][ch + 20] = [a, ch]

    def test_rounds_cover_every_pair_once(self):
        rounds = links.pairs(self.links)
        got = sorted((a, b) for r in rounds for a, b, _ in r)
        self.assertEqual(got, sorted([(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]))
        for r in rounds:
            chips = [c for a, b, _ in r for c in (a, b)]
            self.assertEqual(len(chips), len(set(chips)))
        self.assertLessEqual(len(rounds), 4)

    def test_channels_are_the_low_chips(self):
        rounds = links.pairs(self.links)
        ch = {(a, b): c for r in rounds for a, b, c in r}
        self.assertEqual(ch[(0, 1)], [0, 1])
        self.assertEqual(ch[(1, 2)], [2, 3])
        self.assertEqual(ch[(0, 3)], [8, 9])

    def test_umd_index_is_the_sorted_rank(self):
        with tempfile.TemporaryDirectory() as d:
            for bdf, n in (("0000:c1:00.0", 16), ("0000:01:00.0", 0), ("0000:81:00.0", 24), ("0000:00:01.0", None)):
                dev = Path(d, "bus/pci/devices", bdf)
                dev.mkdir(parents=True)
                if n is not None:
                    (dev / "tenstorrent").mkdir()
                    (dev / "tenstorrent" / f"tenstorrent!{n}").mkdir()
            self.assertEqual(links.umd_index(["c1:00.0", "0000:81:00.0"], d), [2, 1])
            self.assertIsNone(links.umd_index(["0000:41:00.0"], d))

    def _run_links(self, fault, reset):
        phase = Phase(name="links")
        with mock.patch.object(links, "discover", return_value={"bdfs": {}, "links": {}}), mock.patch.object(
            links, "pair_test", return_value=False
        ) as pt, mock.patch.object(links, "fabric_open"):
            links.run_links(phase, Path("/tmp"), fault, reset)
        return phase, pt

    def test_sweep_only_after_a_fault(self):
        for fault, want in ((False, 0), (True, 1)):
            resets = []
            _, pt = self._run_links(fault, lambda: resets.append(1) or "")
            self.assertEqual(len(resets), want)
            self.assertEqual(pt.call_count, 1 + want)

    def test_no_sweep_when_no_reset_ran(self):
        phase, pt = self._run_links(True, lambda: "no reset (--skip-reset)")
        self.assertEqual(pt.call_count, 1)
        sweep = [c for c in phase.checks if c.name == "galaxy_sweep"]
        self.assertEqual([(c.status, c.details) for c in sweep], [(SKIP, "no reset (--skip-reset)")])

    def _sweep_reset(self, **kw):
        """run_diag's reset callback for the sweep, with a glx_reset that fails or not."""
        import diag_runner

        got = {}

        def fake_reset(tt_smi_path, plan, phase, dry_run):
            phase.add(Check(name="post_test_reset_ok", status=FAIL if kw.pop("fail", False) else PASS, details=""))

        def fake_links(phase, out, fault, reset, dry_run):
            got["why"] = reset()

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            diag_runner, "run_links", fake_links
        ), mock.patch.object(diag_runner, "run_busbar", return_value=True), mock.patch.object(
            diag_runner, "run_post_test_reset", fake_reset
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            d = Path(tmp)
            diag_runner.run_diag(
                "galaxy",
                mode="smoke",
                dry_run=True,
                skip_tests=True,
                skip_triage=True,
                skip_eth_status=True,
                skip_qsfp_tests=True,
                skip_sys_triage=True,
                output=d / "report.json",
                snapshot_out=d / "snapshot.json",
                skip_reset=kw.get("skip_reset", False),
            )
        return got["why"]

    def test_sweep_reset_says_why_it_skips(self):
        self.assertEqual(self._sweep_reset(), "")
        self.assertEqual(self._sweep_reset(skip_reset=True), "no reset (--skip-reset)")
        self.assertEqual(self._sweep_reset(fail=True), "reset failed")


class TestTierAndAer(unittest.TestCase):
    def test_galaxy_has_the_load_and_eth_status(self):
        t = tiers.TIERS["galaxy"]
        self.assertTrue(t.busbar and t.eth_status)
        self.assertEqual([n for n, t in tiers.TIERS.items() if t.busbar], ["galaxy"])

    def test_aer_data_holds_only_the_diff(self):
        c = aer_check(aer_new({"0000:01:00.0": {"corr": 0}}, {"0000:01:00.0": {"corr": 1}}), 1.0)
        self.assertEqual(set(c.data), {"hours", "new", "flagged"})


if __name__ == "__main__":
    unittest.main()
