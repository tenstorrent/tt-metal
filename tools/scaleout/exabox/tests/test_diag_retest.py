#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""diag.retest: test -> reset -> test runs only after a clean first run, and a
reset that breaks the unit fails the run instead of being dropped as a reset op."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))
sys.path.insert(0, str(SUITE_DIR / "test_infrastructure"))
sys.path.insert(0, str(TESTS_DIR))

from diag import retest  # noqa: E402
from diag.model import FAIL, PASS, SKIP, Check, Phase  # noqa: E402
from test_diag_tiers import run_weekly  # noqa: E402
from tiers import TIERS  # noqa: E402
from analyze_health_check_results import category_for  # noqa: E402
from utils.report import has_actionable_failure, normalize_health_report  # noqa: E402


def first_run(**statuses: str) -> Phase:
    phase = Phase(name="tests")
    for name, status in statuses.items():
        phase.add(Check(name=name, status=status))
    return phase


class TestRetest(unittest.TestCase):
    def setUp(self):
        self.logs = Path(tempfile.mkdtemp())

    def _run(self, first: Phase, reset_status: str = PASS, dry_run: bool = False, snapshot_error: bool = False):
        phase = Phase(name="retest")

        def fake_reset(tt_smi, plan, ph, dry, logs_dir=None, **kw):
            ph.add(Check(name=f"reset_1_{plan[0].lstrip('-')}", status=reset_status))

        def fake_snapshot(tt_smi, out):
            if snapshot_error:
                raise RuntimeError("tt-smi snapshot failed")
            return {}

        def fake_validate(snap, sp):
            sp.add(Check(name="pcie_enum_count", status=PASS))

        with mock.patch.object(retest, "resolve_tt_smi", return_value="tt-smi"), mock.patch.object(
            retest, "reset_loop", side_effect=fake_reset
        ) as reset, mock.patch.object(retest, "take_snapshot", side_effect=fake_snapshot), mock.patch.object(
            retest, "validate_snapshot", side_effect=fake_validate
        ), mock.patch.object(
            retest, "run_tests"
        ) as tests, redirect_stdout(
            io.StringIO()
        ):
            snap = retest.run_retest(
                None,
                Path("."),
                ("gddr_act_c3", "gddr_act_all"),
                "-glx_reset",
                first,
                phase,
                dry_run,
                self.logs,
                self.logs / "snap.json",
            )
        return phase, snap, reset, tests

    def test_reruns_after_reset_and_snapshot(self):
        phase, snap, reset, tests = self._run(first_run(gddr_act_c3=PASS, gddr_act_all=PASS, gddr_full=FAIL))
        reset.assert_called_once()
        self.assertEqual(reset.call_args.args[1], ("-glx_reset",))
        # Its reset log must not overwrite the pre-test loop's reset_1_r.log.
        self.assertEqual(reset.call_args.kwargs["logs_dir"], self.logs / "retest")
        tests.assert_called_once()
        self.assertEqual(tests.call_args.args[1], ["gddr_act_c3", "gddr_act_all"])
        self.assertEqual(tests.call_args.kwargs["name_suffix"], retest.RETEST_SUFFIX)
        self.assertEqual(snap.name, "snapshot_after_retest_glx_reset")
        self.assertEqual(snap.status, PASS)
        ok = next(c for c in phase.checks if c.name == "retest_reset_ok")
        self.assertEqual(ok.status, PASS)

    def test_only_the_tests_that_passed_rerun(self):
        phase, snap, reset, tests = self._run(first_run(gddr_act_c3=PASS, gddr_act_all=FAIL))
        reset.assert_called_once()
        self.assertEqual(tests.call_args.args[1], ["gddr_act_c3"])

    def test_skipped_when_no_retest_test_passed(self):
        phase, snap, reset, tests = self._run(first_run(gddr_act_all=FAIL))
        reset.assert_not_called()
        tests.assert_not_called()
        self.assertIsNone(snap)
        self.assertEqual([(c.name, c.status) for c in phase.checks], [("retest", SKIP)])
        self.assertIn("gddr_act_all", phase.checks[0].details)

    def test_broken_reset_fails_the_verdict_and_skips_the_rerun(self):
        phase, snap, reset, tests = self._run(first_run(gddr_act_c3=PASS, gddr_act_all=PASS), reset_status=FAIL)
        tests.assert_not_called()
        self.assertIsNone(snap)
        ok = next(c for c in phase.checks if c.name == "retest_reset_ok")
        self.assertEqual(ok.status, FAIL)
        report = {"phases": {"retest": {"checks": [{"name": c.name, "status": c.status} for c in phase.checks]}}}
        self.assertTrue(has_actionable_failure(report))

    def test_a_refusing_reset_gate_skips_the_retest_reset(self):
        from diag import resets

        phase, calls = Phase(name="retest"), []
        with mock.patch.object(retest, "resolve_tt_smi", return_value="tt-smi"), mock.patch.object(
            resets, "run", side_effect=lambda cmd, *a, **kw: calls.append(cmd)
        ), mock.patch.object(retest, "run_tests") as tests, redirect_stdout(io.StringIO()):
            args = (
                None,
                Path("."),
                ("gddr_act_c3",),
                "-glx_reset",
                first_run(gddr_act_c3=PASS),
                phase,
                False,
                self.logs,
                None,
            )
            snap = retest.run_retest(*args, gate="sh -c 'echo busy; exit 3'", hook=None)
        self.assertEqual(calls, [])  # tt-smi never ran
        self.assertIsNone(snap)
        tests.assert_not_called()
        self.assertEqual(
            [(c.name, c.status) for c in phase.checks], [("reset_1_glx_reset", SKIP), ("retest_reset_ok", SKIP)]
        )
        self.assertIn("reset gate: busy", phase.checks[-1].details)

    def test_the_runner_passes_the_reset_gate_and_hook_to_the_retest(self):
        import diag_runner

        with mock.patch.object(diag_runner, "run_retest", return_value=None) as run_retest:
            run_weekly(
                tt_smi_path="/usr/bin/true",
                skip_triage=True,
                skip_qsfp_tests=True,
                skip_sys_triage=True,
                stage_order="gddr_act_c3",  # the weekly retest's test, so the retest runs
                reset_gate="G",
                hook="H",
            )
        self.assertEqual((run_retest.call_args.kwargs["gate"], run_retest.call_args.kwargs["hook"]), ("G", "H"))

    def test_failed_snapshot_keeps_the_pre_test_snapshot_as_verdict(self):
        phase, snap, reset, tests = self._run(first_run(gddr_act_c3=PASS, gddr_act_all=PASS), snapshot_error=True)
        self.assertIsNone(snap)
        tests.assert_called_once()
        self.assertIn("snapshot_capture", [c.name for c in phase.checks])
        report = {
            "phases": {
                "snapshot_after_r3_glx_reset": {"checks": [{"name": "gddr_training", "status": FAIL}]},
                "retest": {"checks": [{"name": c.name, "status": c.status} for c in phase.checks]},
            }
        }
        self.assertTrue(has_actionable_failure(normalize_health_report(report)))

    def test_retest_checks_roll_up_as_stress_and_reset(self):
        self.assertEqual(category_for("retest", {"name": "gddr_act_c3_after_reset", "ip": "other"}), "stress_test")
        self.assertEqual(category_for("retest", {"name": "retest_reset_ok", "ip": "other"}), "reset")

    def test_dry_run_takes_no_snapshot(self):
        phase, snap, reset, tests = self._run(first_run(gddr_act_c3=PASS, gddr_act_all=PASS), dry_run=True)
        self.assertIsNone(snap)
        tests.assert_called_once()


class TestTierRetest(unittest.TestCase):
    def test_retest_names_are_tier_tests(self):
        for name, tier in TIERS.items():
            for t in tier.retest:
                self.assertIn(t, tier.tests, name)

    def test_act_tests_are_retested(self):
        eth = ("eth_link_up", "eth_bandwidth", "eth_data_integrity")
        self.assertEqual(TIERS["weekly"].retest, ("gddr_full", "gddr_act_c3") + eth)
        self.assertEqual(TIERS["full"].retest, ("gddr_full", "gddr_act_all") + eth)
        # gddr_act_all runs both of gddr_act_c3's passes.
        self.assertNotIn("gddr_act_c3", TIERS["full"].tests)


if __name__ == "__main__":
    unittest.main()
