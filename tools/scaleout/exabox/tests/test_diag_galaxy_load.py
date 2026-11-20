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
import json
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

    def test_steps_subset_keeps_plan_order(self):
        for mode in busbar.MODES:
            self.assertEqual([s.name for s in busbar.plan(mode, "heavy, qb")], ["qb", "heavy"])
            self.assertEqual([s.name for s in busbar.plan(mode, "code")], ["code"])
            self.assertEqual(busbar.plan(mode, "qb,mix,step,heavy,code"), busbar.plan(mode))

    def test_steps_keep_their_own_argv(self):
        for mode in busbar.MODES:
            for s in busbar.plan(mode, "step,mix"):
                self.assertEqual(s, _steps(mode)[s.name])

    def test_unknown_step_is_loud(self):
        for bad in ("qb,nope", "", "heavy,"):
            with self.assertRaisesRegex(ValueError, "--steps: unknown"):
                busbar.plan("plain", bad)

    def test_no_override_changes_nothing(self):
        for mode in busbar.MODES:
            self.assertEqual(busbar.plan(mode, **{k: None for k in busbar.OVERRIDES}), busbar.plan(mode))
            self.assertTrue(all(not s.env for s in busbar.plan(mode)))

    def test_heavy_overrides(self):
        h = _steps("plain", heavy_s=40, ops="16k_hifi4", x_s=15, x_ops="8k_fp32_hifi4")["heavy"]
        argv = h.argv
        self.assertEqual(argv[argv.index("--chunk-min") + 1], f"{40 / 60:g}")
        self.assertEqual(argv[-argv[::-1].index("--heavy") :][0], "16k_hifi4")
        self.assertEqual(argv[argv.index("--heavy-x") + 1], "8k_fp32_hifi4")
        self.assertEqual(argv[argv.index("--heavy-x-s") + 1], "15")
        self.assertEqual(h.seconds, 50)

    def test_x_seconds_are_capped_at_the_heavy_seconds(self):
        argv = _steps("full", heavy_s=20, x_s=300)["heavy"].argv
        self.assertEqual(argv[argv.index("--heavy-x-s") + 1], "20")

    def test_x_ops_may_be_none(self):
        argv = _steps("plain", x_ops=",")["heavy"].argv
        self.assertEqual(argv[argv.index("--heavy-x") + 1], "")

    def test_heavy_seconds_override_wins_over_smoke(self):
        self.assertEqual(_steps("smoke", heavy_s=30)["heavy"].seconds, 32)  # 30 s plus smoke's 2 s gap

    def test_heavy_n_gap_and_trays(self):
        s = _steps("plain", heavy_n=4, gap_ms=0, trays="3, 1")
        for name in ("mix", "heavy"):
            argv = s[name].argv
            self.assertEqual(argv[argv.index("--heavy-n") + 1], "4", name)
            self.assertEqual(argv[argv.index("--burst-gap-ms") + 1], "0", name)
            self.assertEqual(argv[argv.index("--trays") + 1], "3,1", name)
        for name in ("qb", "step"):
            argv = s[name].argv
            self.assertEqual(argv[argv.index("--burst-gap-ms") + 1], "0", name)
            self.assertNotIn("--heavy-n", argv)
            self.assertNotIn("--trays", argv)
        self.assertEqual(s["code"], _steps("plain")["code"])

    def test_matmul_throttle_is_an_environment_setting_of_the_load_steps(self):
        s = _steps("plain", mm_throttle=3)
        for name in ("qb", "mix", "step", "heavy"):
            self.assertEqual(s[name].env, (("TT_MM_THROTTLE_PERF", "3"),), name)
        self.assertEqual(s["code"].env, ())

    def test_bad_overrides_are_loud(self):
        for kw in (
            {"mm_throttle": 0},
            {"mm_throttle": 6},
            {"trays": "5"},
            {"trays": "1,x"},
            {"trays": ","},
            {"ops": ""},
            {"heavy_s": 0},
            {"heavy_n": 0},
            {"gap_ms": -1},
            {"x_s": -1},
            {"nope": 1},
        ):
            with self.assertRaises(ValueError, msg=str(kw)):
                busbar.plan("plain", **kw)

    def test_every_override_flag_exists_in_its_program(self):
        s = busbar.plan(
            "plain",
            **{
                "heavy_s": 9,
                "ops": "16k_hifi4",
                "x_s": 1,
                "x_ops": "16k_hifi4",
                "heavy_n": 2,
                "gap_ms": 1,
                "trays": "1",
            },
        )
        for step in s:
            src = (busbar.GALAXY_DIR / step.argv[0]).read_text()
            for flag in (a for a in step.argv[1:] if a.startswith("--")):
                self.assertIn(f'"{flag}"', src, f"{step.name}: {step.argv[0]} has no {flag}")

    def _plan_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(SUITE_DIR / "diag_runner.py"), "--tier", "galaxy", *args],
            capture_output=True,
            text=True,
        )

    def test_plan_cli_steps_and_overrides(self):
        r = self._plan_cli("--mode", "plain", "--plan", "--steps", "heavy", "--heavy-s", "40")
        self.assertEqual((r.returncode, [ln.split()[0] for ln in r.stdout.splitlines()]), (0, ["heavy"]))
        self.assertIn(" 50 s ", r.stdout)

    def test_cli_unknown_step_and_bad_override_exit_2(self):
        for args in (("--steps", "qb,nope"), ("--mm-throttle", "9"), ("--trays", "7"), ("--heavy-n", "x")):
            r = self._plan_cli("--mode", "plain", "--plan", *args)
            self.assertEqual(r.returncode, 2, args)
            self.assertTrue(r.stderr.strip(), args)

    def test_cli_steps_without_mode_is_an_error(self):
        r = self._plan_cli("--steps", "qb")
        self.assertEqual(r.returncode, 2)
        self.assertIn("need --mode", r.stderr)

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

    def test_step_env_reaches_the_program(self):
        with tempfile.TemporaryDirectory() as d:
            prog = Path(d) / "prog.py"
            prog.write_text("import os; print(os.environ['TT_MM_THROTTLE_PERF'])")
            step = busbar.Step("x", "test", ("prog.py",), 5, (("TT_MM_THROTTLE_PERF", "4"),))
            with mock.patch.object(busbar, "GALAXY_DIR", Path(d)):
                self.assertEqual(busbar.run_step(step, Path(d)).status, PASS)
            self.assertEqual((Path(d) / "x" / "run.log").read_text().strip(), "4")

    def test_dry_run_shows_the_env(self):
        c = busbar.run_step(busbar.plan("smoke", mm_throttle=2)[0], Path("/nonexistent"), dry_run=True)
        self.assertEqual(c.data["env"], {"TT_MM_THROTTLE_PERF": "2"})

    def test_run_busbar_runs_only_the_chosen_steps(self):
        calls = []

        def fake(step, out, dry_run):
            calls.append(step.name)
            return busbar.Check(name=step.name, status=PASS)

        with mock.patch.object(busbar, "run_step", fake):
            self.assertFalse(busbar.run_busbar(Phase(name="busbar"), Path("/tmp"), "smoke", steps="code,qb", gap_ms=1))
        self.assertEqual(calls, ["qb", "code"])

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

    def _sweep_reset(self, reset_fails=False, skip_reset=False, no_reset=False):
        """run_diag's reset callback for the sweep, after a failed pair test, with the real run_post_test_reset
        over a glx_reset that fails or not."""
        import diag_runner
        from diag import resets

        got = {}

        def fake_loop(tt_smi, plan, phase, dry_run):
            phase.add(Check(name="reset_glx", status=FAIL if reset_fails else PASS, details=""))

        def fake_links(phase, out, fault, reset, dry_run):
            phase.add(Check(name="galaxy_pair", status=FAIL, details="pair 00-01 failed", ip="eth"))
            got["why"] = reset()

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            diag_runner, "run_links", fake_links
        ), mock.patch.object(diag_runner, "run_busbar", return_value=True), mock.patch.object(
            resets, "resolve_tt_smi", return_value="tt-smi"
        ), mock.patch.object(
            resets, "reset_loop", fake_loop
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
                skip_reset=skip_reset,
                no_reset=no_reset,
            )
        return got["why"]

    def test_no_reset_sweeps_without_resetting(self):
        import diag_runner

        self.assertEqual(self._sweep_reset(no_reset=True, reset_fails=True), "")
        self.assertEqual(self._sweep_reset(no_reset=True, skip_reset=True), "")
        args = diag_runner.build_diag_arg_parser().parse_args(["--tier", "galaxy", "--mode", "smoke"])
        self.assertFalse(args.no_reset)
        self.assertTrue(diag_runner.build_diag_arg_parser().parse_args(["--tier", "galaxy", "--no-reset"]).no_reset)

    def _disc(self):
        bdfs = {0: "0000:01:00.0", 1: "0000:02:00.0", 2: "0000:c1:00.0"}  # tray 1 U1, tray 1 U2, tray 3 U1
        return {"bdfs": bdfs, "links": {0: {0: [1, 4], 1: [1, 5]}, 1: {4: [0, 0], 5: [0, 1]}}}

    def test_link_ends_are_named_by_tray_port_asic_and_bdf(self):
        got = links.link_ends(self._disc(), 0, 1, [1])
        self.assertEqual(
            got,
            [
                {
                    "a": {"bdf": "0000:01:00.0", "tray": 1, "asic": 1, "channel": 1, "port": got[0]["a"]["port"]},
                    "b": {"bdf": "0000:02:00.0", "tray": 1, "asic": 2, "channel": 5, "port": got[0]["b"]["port"]},
                }
            ],
        )

    def test_report_checks_name_their_chips_and_link_ends(self):
        ends = links.link_ends(self._disc(), 0, 1, [0])
        report = {
            "phases": {
                "links": {
                    "checks": [
                        {
                            "name": "galaxy_pair",
                            "status": "FAIL",
                            "details": "01:00.0-02:00.0 (channels 0): hung",
                            "data": {"failed": [{"pair": "01:00.0-02:00.0", "ends": ends}], "link_ends": ends},
                        },
                        {"name": "x", "status": "PASS", "details": "at 12:00.0 on 0000:c1:00.0", "data": {}},
                        {"name": "none", "status": "PASS", "details": "", "data": {}},
                    ]
                }
            }
        }
        before = json.loads(json.dumps(report))
        links.name_chips(report)
        a, x, none = report["phases"]["links"]["checks"]
        self.assertEqual((a["bdf"], a["tray"], a["asic"]), (["0000:01:00.0", "0000:02:00.0"], [1, 1], [1, 2]))
        self.assertEqual(a["link_ends"], ends)
        self.assertEqual((x["bdf"], x["tray"], x["asic"]), (["0000:c1:00.0"], [3], [1]))
        self.assertEqual((none["bdf"], none["tray"], none["asic"], none["link_ends"]), ([], [], [], []))
        for old, new in zip(before["phases"]["links"]["checks"], report["phases"]["links"]["checks"]):
            self.assertEqual(old, {k: new[k] for k in old})  # only keys were added

    def test_failed_pair_check_carries_the_link_ends(self):
        phase = Phase(name="links")
        with tempfile.TemporaryDirectory() as d:  # no tt-kmd nodes under d: every pair fails to open
            self.assertTrue(links.pair_test(phase, Path(d), "pair", 1, self._disc(), sys_root=d))
        data = phase.checks[0].data
        self.assertEqual([e["a"]["tray"] for e in data["link_ends"]], [1, 1])
        self.assertEqual(data["link_ends"], [e for f in data["failed"] for e in f["ends"]])

    def test_sweep_reset_says_why_it_skips(self):
        self.assertEqual(self._sweep_reset(), "")
        self.assertEqual(self._sweep_reset(skip_reset=True), "no reset (--skip-reset)")
        self.assertEqual(self._sweep_reset(reset_fails=True), "reset failed")


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
