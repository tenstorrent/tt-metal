#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""tiers.py against the places that repeat what it says.

The shell wrappers validate the tier name themselves, before any Python runs,
so a tier added to tiers.py and not to them is a tier nobody can launch. The
docker wrapper also hardcodes which tiers want the sys-triage mounts and the
syseng package, and run_health_check.py its own --tier choices. These tests
fail until they agree.
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from tiers import DEPRECATED_TIER_ALIASES, TIERS, select_stages  # noqa: E402

ACCEPTED = set(TIERS) | set(DEPRECATED_TIER_ALIASES)

# A case-statement arm listing the tiers, e.g. `flash|daily|weekly|full|pre_bmc_reboot|light|...)`.
TIER_ARM_RE = re.compile(r"^\s*((?:[a-z_]+\|)+[a-z_]+)\)", re.MULTILINE)


def tier_arms(script: str) -> list[set[str]]:
    text = (SUITE_DIR / script).read_text()
    return [set(m.group(1).split("|")) for m in TIER_ARM_RE.finditer(text) if "daily" in m.group(1)]


class TestShellWrappersAgree(unittest.TestCase):
    def test_run_diag_accepts_every_tier(self):
        arms = tier_arms("run_diag.sh")
        self.assertEqual(len(arms), 1, arms)
        self.assertEqual(arms[0], ACCEPTED)

    def test_run_diag_docker_accepts_every_tier(self):
        arms = tier_arms("run_diag_docker.sh")
        # The positional form and the case under -t/--tier.
        self.assertEqual(len(arms), 2, arms)
        for arm in arms:
            self.assertEqual(arm, ACCEPTED)

    def test_docker_sys_triage_tiers(self):
        # run_diag_docker.sh adds the journal mounts and the BMC credentials
        # only for these (and pre_reboot, the old name of pre_bmc_reboot).
        self.assertEqual(
            {name for name, tier in TIERS.items() if tier.sys_triage},
            {"flash", "full", "pre_bmc_reboot", "post_bmc_reboot"},
        )
        text = (SUITE_DIR / "run_diag_docker.sh").read_text()
        self.assertIn("flash|full|pre_bmc_reboot|pre_reboot|post_bmc_reboot)", text)

    def test_docker_qsfp_tiers(self):
        # run_diag_docker.sh sets WANT_QSFP, which installs the syseng package and
        # warns about missing BMC API credentials, on every tier but these.
        self.assertEqual({name for name, tier in TIERS.items() if not tier.qsfp_tests}, {"flash"})
        text = (SUITE_DIR / "run_diag_docker.sh").read_text()
        self.assertIn("[[ ${TIER} != flash ]]", text)

    def test_only_flash_prints_the_eth_summary(self):
        self.assertEqual({name for name, tier in TIERS.items() if tier.eth_summary}, {"flash"})

    def test_run_health_check_tier_choices(self):
        # parse_args() builds and parses in one go, so read the call instead.
        tree = ast.parse((SUITE_DIR / "test_infrastructure" / "run_health_check.py").read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and node.args
            if isinstance(node.args[0], ast.Constant) and node.args[0].value == "--tier"
        ]
        self.assertEqual(len(calls), 1, calls)
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords if kw.arg in ("choices", "default")}
        self.assertEqual(set(kwargs["choices"]), ACCEPTED)
        self.assertIn(kwargs["default"], TIERS)


class TestDiagRunnerCli(unittest.TestCase):
    def test_tier_choices_follow_tiers(self):
        from diag_runner import build_diag_arg_parser

        action = next(a for a in build_diag_arg_parser()._actions if "--tier" in a.option_strings)
        self.assertEqual(list(action.choices), [*TIERS, *DEPRECATED_TIER_ALIASES])

    def test_deprecated_alias_reports_new_name(self):
        from diag_runner import run_diag

        for old, new in DEPRECATED_TIER_ALIASES.items():
            with self.subTest(old), tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    _rc, report = run_diag(
                        old,
                        input_snapshot=d / "missing.json",
                        output=d / "report.json",
                        snapshot_out=d / "snapshot.json",
                        dry_run=True,
                        skip_reset=True,
                        skip_tests=True,
                        skip_triage=True,
                        skip_qsfp_tests=True,
                        skip_sys_triage=True,
                    )
                self.assertEqual(report["tier"], new)

    def test_skip_flags_skip_the_retest(self):
        from diag_runner import run_diag

        for flag in ("skip_reset", "skip_tests"):
            with self.subTest(flag), tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    _rc, report = run_diag(
                        "weekly",
                        input_snapshot=d / "missing.json",
                        output=d / "report.json",
                        snapshot_out=d / "snapshot.json",
                        dry_run=True,
                        skip_triage=True,
                        skip_qsfp_tests=True,
                        skip_sys_triage=True,
                        # Dry run still resolves tt-smi for the resets it prints.
                        tt_smi_path="/usr/bin/true",
                        **{flag: True},
                    )
                retest = report["phases"]["retest"]
                self.assertEqual(retest["status"], "SKIP")
                self.assertEqual([c["name"] for c in retest["checks"]], ["retest"])

    def test_docker_wrapper_feature_probe_still_matches(self):
        # run_diag_docker.sh greps diag_runner.py for this flag to tell an image
        # that predates the sys-triage phase, and drops the phase's flags if it
        # is missing. The parser has to stay in that file for the probe to work.
        self.assertIn("--skip-sys-triage", (SUITE_DIR / "diag_runner.py").read_text())


class TestStageOrder(unittest.TestCase):
    def test_reset_only(self):
        t = select_stages(TIERS["weekly"], "reset")
        self.assertEqual(
            (t.resets, t.tests, t.retest, t.post_test_resets, t.eth_status), (TIERS["weekly"].resets, (), (), (), False)
        )

    def test_tests_and_end(self):
        t = select_stages(TIERS["weekly"], "gddr_full, eth_link_up,end")
        self.assertEqual((t.resets, t.tests, t.post_test_resets), ((), ("gddr_full", "eth_link_up"), ("-glx_reset",)))

    def test_unknown_or_out_of_order_is_refused(self):
        for order in ("busbar", "end,reset", "eth_link_up,gddr_full"):
            with self.subTest(order), self.assertRaises(ValueError):
                select_stages(TIERS["weekly"], order)

    def test_cli_flags_parse(self):
        from diag_runner import build_diag_arg_parser

        a = build_diag_arg_parser().parse_args(
            ["--tier", "weekly", "--stage-order", "reset", "--reset-gate", "gate --x", "--hook", "/h.sh"]
        )
        self.assertEqual((a.stage_order, a.reset_gate, a.hook), ("reset", "gate --x", "/h.sh"))

    def test_empty_input_snapshot_is_not_judged(self):
        from diag_runner import run_diag

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            d = Path(tmp)
            (d / "snap.json").write_text("")
            _rc, report = run_diag(
                "weekly",
                input_snapshot=d / "snap.json",
                output=d / "r.json",
                snapshot_out=d / "s.json",
                dry_run=True,
                stage_order="end",
            )
        snap = report["phases"]["snapshot"]["checks"]
        self.assertEqual([c["status"] for c in snap if c["name"] == "snapshot_capture"], ["SKIP"])
        self.assertIn("post_test_reset", report["phases"])

    def test_run_diag_wires_order_gate_and_hook(self):
        import diag_runner

        from diag.model import Check

        def skipped(_tt_smi, _plan, phase, *_a, **_kw):
            phase.add(Check(name="reset_1_glx_reset", status="SKIP", details="reset cannot run: reset gate: busy"))

        pre, post, hook = mock.Mock(side_effect=skipped), mock.Mock(side_effect=skipped), mock.Mock()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            d = Path(tmp)
            with mock.patch.object(diag_runner, "reset_loop", pre), mock.patch.object(
                diag_runner, "run_post_test_reset", post
            ), mock.patch.object(diag_runner, "call_hook", hook):
                _rc, report = diag_runner.run_diag(
                    "weekly",
                    input_snapshot=d / "missing.json",
                    output=d / "r.json",
                    snapshot_out=d / "s.json",
                    dry_run=True,
                    tt_smi_path="/usr/bin/true",
                    skip_triage=True,  # tray-test passes it with --stage-order and still gets the end reset
                    skip_qsfp_tests=True,
                    skip_sys_triage=True,
                    stage_order="reset,end",
                    reset_gate="G",
                    hook="H",
                )
        self.assertEqual(
            [c.args[:3] for c in hook.call_args_list], [("H", "stage-start", "reset"), ("H", "stage-start", "end")]
        )
        self.assertEqual((pre.call_args.kwargs["gate"], pre.call_args.kwargs["hook"]), ("G", "H"))
        self.assertEqual(post.call_args.args[1:], (("-glx_reset",), mock.ANY, True, "G", "H"))
        self.assertEqual([c["name"] for c in report["phases"]["tests"]["checks"]], ["tests"])  # no tests in the order
        self.assertNotIn("retest", report["phases"])
        # A reset phase that only skipped stays SKIP (all_skip_stays_skip), not PASS.
        self.assertEqual([report["phases"][p]["status"] for p in ("reset_loop", "post_test_reset")], ["SKIP", "SKIP"])


if __name__ == "__main__":
    unittest.main()
