#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The galaxy load without hardware: the bus-bar plan per mode, every planned flag exists in its program, a step
is judged by its exit, its FAIL lines and its hang, the pair rounds never put a chip in two pairs, the UMD index
follows the sorted BDFs of a fake sysfs, and daily, weekly and full run it in the tests phase after the device
tests, each at its own length."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

import tiers  # noqa: E402
from diag import busbar, links  # noqa: E402
from diag.aer import aer_check, aer_new  # noqa: E402
from diag.model import FAIL, PASS, SKIP, Check, Phase  # noqa: E402
from diag.naming import chip, link_end  # noqa: E402


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
            {"ops": "16k_hifi2;reboot"},
            {"x_ops": "16k_hifi4,$(id)"},
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

    def _cli(self, tier, *args):
        return subprocess.run(
            [sys.executable, str(SUITE_DIR / "diag_runner.py"), "--tier", tier, *args],
            capture_output=True,
            text=True,
        )

    def test_cli_bad_override_exits_2_before_the_run(self):
        for tier in ("daily", "weekly", "full"):
            for args in (("--mm-throttle", "9"), ("--trays", "7"), ("--heavy-n", "x"), ("--heavy-ops", "a;b")):
                r = self._cli(tier, *args)
                self.assertEqual(r.returncode, 2, (tier, args))
                self.assertTrue(r.stderr.strip(), (tier, args))

    def test_cli_overrides_on_a_tier_without_the_load_are_an_error(self):
        for tier in ("flash", "pre_bmc_reboot", "pre_reboot"):
            r = self._cli(tier, "--heavy-s", "40")
            self.assertEqual(r.returncode, 2, tier)
            self.assertIn("does not run (tiers: daily, weekly, full)", r.stderr, tier)

    def test_cli_has_no_mode_plan_or_steps(self):
        for args in (("--mode", "plain"), ("--plan",), ("--steps", "qb")):
            r = self._cli("full", *args)
            self.assertEqual(r.returncode, 2, args)
            self.assertIn("unrecognized arguments", r.stderr, args)


class TestRunStep(unittest.TestCase):
    def _run(self, body, seconds=5):
        with tempfile.TemporaryDirectory() as d:
            prog = Path(d) / "prog.py"
            prog.write_text(body)
            with mock.patch.object(busbar, "GALAXY_DIR", Path(d)), mock.patch.object(busbar, "SETUP_S", 0):
                return busbar.run_step(busbar.Step("x", "test", ("prog.py", "--out", "{out}"), seconds), Path(d))

    def test_pass(self):
        self.assertEqual(self._run("print('ok')").status, PASS)

    def test_check_and_log_are_named_after_the_test_and_step(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "prog.py").write_text("print('ok')")
            with mock.patch.object(busbar, "GALAXY_DIR", Path(d)):
                c = busbar.run_step(busbar.Step("qb", "test", ("prog.py",), 5), Path(d))
                f = busbar.run_step(busbar.Step("fabric", "test", ("prog.py",), 5), Path(d), test="links")
        self.assertEqual((c.name, Path(c.data["log"]).name), ("busbar_qb", "busbar_qb.log"))
        self.assertEqual((f.name, Path(f.data["log"]).name), ("links_fabric", "links_fabric.log"))

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
            self.assertEqual((Path(d) / "x" / "busbar_x.log").read_text().strip(), "4")

    def test_dry_run_shows_the_env(self):
        c = busbar.run_step(busbar.plan("smoke", mm_throttle=2)[0], Path("/nonexistent"), dry_run=True)
        self.assertEqual(c.data["env"], {"TT_MM_THROTTLE_PERF": "2"})

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
        sweep = [c for c in phase.checks if c.name == "links_sweep"]
        self.assertEqual([(c.status, c.details) for c in sweep], [(SKIP, "no reset (--skip-reset)")])

    def _sweep_reset(self, reset_fails=False, skip_reset=False):
        return self._run_load(reset_fails=reset_fails, skip_reset=skip_reset)["why"]

    def _run_load(self, tier="full", reset_fails=False, skip_reset=False, busbar_fails=True, skip_tests=False):
        """run_diag on *tier* with its device tests, bus-bar and links tests stubbed: ``order`` is what ran, in
        order, ``device`` the names run_tests got, ``mode`` the length the bus-bar test was given, ``fault`` whether
        the links test was told of a bus-bar fault, ``why`` what its reset callback said after a failed pair test
        (the real run_post_test_reset over a glx_reset that fails or not) and ``report`` the run's report."""
        import diag_runner
        from diag import resets

        got = {"order": [], "device": None, "mode": None, "fault": None, "why": None}

        def fake_tests(tt_metal, names, phase, dry_run, logs_dir):
            got["order"].append("device")
            got["device"] = list(names)

        def fake_busbar(phase, out, mode, dry_run, **over):
            got["order"].append("busbar")
            got["mode"] = mode
            phase.add(Check(name="busbar_qb", status=FAIL if busbar_fails else PASS))
            return busbar_fails

        def fake_loop(tt_smi, plan, phase, dry_run):
            phase.add(Check(name="reset_glx", status=FAIL if reset_fails else PASS, details=""))

        def fake_links(phase, out, fault, reset, dry_run):
            got["order"].append("links")
            phase.add(Check(name="links_pair", status=FAIL, details="pair 00-01 failed", ip="eth"))
            got["fault"], got["why"] = fault, reset()

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            diag_runner, "run_tests", fake_tests
        ), mock.patch.object(diag_runner, "run_links", fake_links), mock.patch.object(
            diag_runner, "run_busbar", fake_busbar
        ), mock.patch.object(
            diag_runner, "resolve_tt_smi", return_value="tt-smi"
        ), mock.patch.object(
            resets, "resolve_tt_smi", return_value="tt-smi"
        ), mock.patch.object(
            resets, "reset_loop", fake_loop
        ), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(
            io.StringIO()
        ):
            d = Path(tmp)
            _rc, got["report"] = diag_runner.run_diag(
                tier,
                dry_run=True,
                input_snapshot=d / "missing.json",
                skip_tests=skip_tests,
                skip_triage=True,
                skip_eth_status=True,
                skip_qsfp_tests=True,
                skip_sys_triage=True,
                output=d / "report.json",
                snapshot_out=d / "snapshot.json",
                skip_reset=skip_reset,
            )
        return got

    def test_load_runs_in_the_tests_phase_after_the_device_tests(self):
        for tier in ("daily", "weekly", "full"):
            with self.subTest(tier=tier):
                got = self._run_load(tier)
                self.assertEqual(got["order"], ["device", "busbar", "links"])
                self.assertEqual(got["device"], [t for t in tiers.TIERS[tier].tests if t not in busbar.LOAD_TESTS])
                names = [c["name"] for c in got["report"]["phases"]["tests"]["checks"]]
                self.assertIn("busbar_qb", names)
                self.assertIn("links_pair", names)
                self.assertFalse({"busbar", "links"} & set(got["report"]["phases"]))

    def test_each_tier_runs_the_load_at_its_own_length(self):
        for tier, mode in (("daily", "smoke"), ("light", "smoke"), ("weekly", "plain"), ("full", "full")):
            with self.subTest(tier=tier):
                self.assertEqual(self._run_load(tier)["mode"], mode)

    def test_a_tier_without_the_load_runs_none_of_it(self):
        for tier in ("flash", "pre_bmc_reboot", "post_bmc_reboot"):
            with self.subTest(tier=tier):
                self.assertEqual(self._run_load(tier)["order"], [])

    def test_a_bus_bar_fault_makes_the_links_test_sweep(self):
        self.assertTrue(self._run_load(busbar_fails=True)["fault"])
        self.assertFalse(self._run_load(busbar_fails=False)["fault"])

    def test_skip_tests_skips_the_load(self):
        got = self._run_load(skip_tests=True)
        self.assertEqual(got["order"], [])
        self.assertEqual([c["name"] for c in got["report"]["phases"]["tests"]["checks"]], ["tests"])

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

    def test_eth_status_links_name_their_ends(self):
        from diag.eth_status import EthLink, judge_eth_status
        from diag.galaxy import UBB_ETH_PORT_OF, expected_bdf

        asic, ch = next(iter(UBB_ETH_PORT_OF))
        bdf = expected_bdf(2, asic)
        row = dict(dev=0, noc="1-1", channel=ch, port="Up", link="Down", heartbeat="Down", rx="Up")
        row.update(retrain=0, crc=0, uncorr=0, err_stat=0)
        checks = {c.name: c for c in judge_eth_status({(0, "1-1"): EthLink(**row)}, None, None, {0: bdf}, 31)}
        report = {
            "phases": {
                "eth": {"checks": [asdict(checks[n]) for n in ("eth_status_links_down", "eth_status_heartbeat")]}
            }
        }
        links.name_chips(report)
        want = [{"a": link_end(bdf, ch)}]
        self.assertEqual(want[0]["a"]["tray"], 2)
        self.assertRegex(want[0]["a"]["port"], r"^(QSFP|[A-Z_]+) P\d+$")
        for c in report["phases"]["eth"]["checks"]:
            self.assertEqual((c["bdf"], c["tray"], c["asic"], c["link_ends"]), ([bdf], [2], [asic], want), c["name"])

    def test_count_fail_names_only_the_missing_chip(self):
        from diag.galaxy import expected_bdf
        from diag.snapshot_checks import pcie_enum_check

        bdfs = [expected_bdf(t, a) for t in range(1, 5) for a in range(1, 9)]
        c = pcie_enum_check([{"board_info": {"bus_id": b}} for b in bdfs[1:]])
        report = {"phases": {"snapshot": {"checks": [asdict(c)]}}}
        links.name_chips(report)
        got = report["phases"]["snapshot"]["checks"][0]
        self.assertEqual((got["status"], got["bdf"], got["tray"], got["asic"]), (FAIL, [bdfs[0]], [1], [1]))

    def test_every_chip_blaming_check_names_exactly_the_bad_chip(self):
        from diag.eth_status import PcieFunction, pcie_gate_check
        from diag.galaxy import expected_bdf, parse_bdf_physical
        from diag.snapshot_checks import validate_snapshot

        bad = "0000:41:00.0"

        def dev(bdf):
            _, asic = parse_bdf_physical(bdf)
            board = {"bus_id": bdf, "dram_status": True, "pcie_width": "8" if asic == 6 else "1"}
            smbus = {"ENABLED_GDDR": "0xff", "HARVESTING_STATE": "0x0", "ASIC_LOCATION": hex(asic)}
            telem = {"asic_temperature": "45.0", "power": "120.0", "current": "90.0", "voltage": "0.75"}
            return {
                "board_info": board,
                "telemetry": telem,
                "limits": {"thm_limit": "100"},
                "smbus_telem": {**smbus, "DDR_STATUS": "0x55555555"},
            }

        devs = [dev(expected_bdf(t, a)) for t in range(1, 5) for a in range(1, 9)]
        i = next(i for i, d in enumerate(devs) if d["board_info"]["bus_id"] == bad)
        devs[i]["board_info"].update(dram_status=False, pcie_width="4")
        devs[i]["telemetry"]["asic_temperature"] = "101.0"
        devs[i]["smbus_telem"].update(
            ENABLED_GDDR="0x7f", HARVESTING_STATE="0x5", ASIC_LOCATION="0x3", DDR_STATUS="0x0"
        )
        phase = Phase(name="snapshot")
        validate_snapshot({"device_info": devs}, phase)
        checks = [asdict(c) for c in phase.checks]
        checks.append(asdict(pcie_gate_check([PcieFunction(bad, 0xFFFF, 0)])))
        checks.append(asdict(pcie_gate_check([PcieFunction(bad, None, None)])))
        checks.append(asdict(aer_check({bad: {"fatal": 1}, "0000:42:00.0": {"corr": 1}}, 1.0)))
        report = {"phases": {"all": {"checks": checks}}}
        links.name_chips(report)
        named = {}
        for c in checks:
            self.assertIn((c["bdf"], c["tray"], c["asic"]), (([], [], []), ([bad], [2], [1])), c["name"])
            if c["bdf"]:
                named.setdefault(c["name"], []).append(c["status"])
        self.assertEqual(
            {n: sorted(set(v)) for n, v in named.items()},
            {
                "dram_status": [FAIL],
                "enabled_gddr_full": [FAIL],
                "harvesting_state": [FAIL],
                "asic_location_per_ubb": [FAIL],
                "physical_vs_fw_location": [FAIL],
                "gddr_training_per_channel": [FAIL],
                "gddr_bist_per_channel": [FAIL],
                "pcie_lane_width": [FAIL],
                "asic_thermal_precheck": [FAIL],
                "c8_telemetry": ["WARN"],
                "eth_status_pcie_gate": [FAIL, "WARN"],
                "pcie_aer_delta": [FAIL],
            },
        )

    def test_report_checks_name_their_chips_and_link_ends(self):
        ends = links.link_ends(self._disc(), 0, 1, [0])
        report = {
            "phases": {
                "links": {
                    "checks": [
                        {
                            "name": "links_pair",
                            "status": "FAIL",
                            "details": "01:00.0-02:00.0 (channels 0): hung",
                            "data": {"failed": [{"pair": "01:00.0-02:00.0", "ends": ends}], "link_ends": ends},
                        },
                        {"name": "x", "status": "PASS", "details": "at 04:00.0 on 0000:c1:00.0", "data": {}},
                        {
                            "name": "dram",
                            "status": "FAIL",
                            "details": "",
                            "data": {"blamed_chips": [chip("0000:c1:00.0")]},
                        },
                        {"name": "old", "status": "FAIL", "details": "", "data": {"failed_chips": ["0000:c1:00.0"]}},
                        {"name": "none", "status": "PASS", "details": "", "data": {}},
                    ]
                }
            }
        }
        before = json.loads(json.dumps(report))
        links.name_chips(report)
        a, x, dram, old, none = report["phases"]["links"]["checks"]
        self.assertEqual((a["bdf"], a["tray"], a["asic"]), (["0000:01:00.0", "0000:02:00.0"], [1, 1], [1, 2]))
        self.assertEqual(a["link_ends"], ends)
        self.assertEqual((x["bdf"], x["link_ends"]), ([], []))  # a BDF or time in the text names nothing
        self.assertEqual((dram["bdf"], dram["tray"], dram["asic"]), (["0000:c1:00.0"], [3], [1]))
        self.assertEqual(old["bdf"], [])  # only blamed_chips and link_ends name chips
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
    def test_daily_weekly_and_full_end_with_the_load_at_their_own_length(self):
        for name, mode in (("daily", "smoke"), ("weekly", "plain"), ("full", "full")):
            t = tiers.TIERS[name]
            self.assertEqual(t.tests[-len(busbar.LOAD_TESTS) :], busbar.LOAD_TESTS, name)
            self.assertEqual(t.load_mode, mode, name)
        self.assertEqual([n for n, t in tiers.TIERS.items() if t.galaxy_load], ["daily", "weekly", "full"])

    def test_full_is_weekly_with_the_full_load_aer_eth_status_and_sys_triage(self):
        full, weekly = tiers.TIERS["full"], tiers.TIERS["weekly"]
        self.assertEqual(replace(full, load_mode="plain", aer_diff=False, eth_status=False, sys_triage=False), weekly)
        self.assertTrue(full.aer_diff and full.eth_status and full.sys_triage)
        self.assertEqual([n for n, t in tiers.TIERS.items() if t.aer_diff], ["full"])

    def _validate_with(self, **changes):
        changed = {**tiers.TIERS, "full": replace(tiers.TIERS["full"], **changes)}
        with mock.patch.object(tiers, "TIERS", changed):
            tiers._validate()

    def test_the_load_runs_last_in_its_own_order(self):
        device = tuple(t for t in tiers.TIERS["full"].tests if t not in busbar.LOAD_TESTS)
        for tests in ((*busbar.LOAD_TESTS, *device), (*device, "links", "busbar"), ("busbar", "gddr_full", "links")):
            with self.subTest(tests=tests):
                with self.assertRaisesRegex(ValueError, "the galaxy load runs last"):
                    self._validate_with(tests=tests)
        for tests in ((*device, "links"), ("busbar",)):
            with self.subTest(tests=tests):
                self._validate_with(tests=tests)

    def test_load_mode_goes_with_the_load(self):
        for mode in (None, "long"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "the galaxy load needs a load_mode"):
                self._validate_with(load_mode=mode)
        with self.assertRaisesRegex(ValueError, "load_mode without the galaxy load"):
            self._validate_with(tests=("gddr_full",))

    def test_aer_data_holds_only_the_diff(self):
        c = aer_check(aer_new({"0000:01:00.0": {"corr": 0}}, {"0000:01:00.0": {"corr": 1}}), 1.0)
        self.assertEqual(set(c.data), {"hours", "new", "flagged", "blamed_chips"})


if __name__ == "__main__":
    unittest.main()
