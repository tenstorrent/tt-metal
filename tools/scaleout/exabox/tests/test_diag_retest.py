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

from diag import retest  # noqa: E402
from diag.model import FAIL, PASS, SKIP, Check, Phase  # noqa: E402
from tiers import TIERS  # noqa: E402
from utils.report import has_actionable_failure  # noqa: E402


def first_run(**statuses: str) -> Phase:
    phase = Phase(name="tests")
    for name, status in statuses.items():
        phase.add(Check(name=name, status=status))
    return phase


class TestRetest(unittest.TestCase):
    def setUp(self):
        self.logs = Path(tempfile.mkdtemp())

    def _run(self, first: Phase, reset_status: str = PASS, dry_run: bool = False):
        phase = Phase(name="retest")

        def fake_reset(tt_smi, plan, ph, dry, logs_dir=None):
            ph.add(Check(name=f"reset_1_{plan[0].lstrip('-')}", status=reset_status))

        def fake_snapshot(tt_smi, out):
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
        self.assertEqual(tests.call_args.args[1], ("gddr_act_c3", "gddr_act_all"))
        self.assertEqual(tests.call_args.kwargs["name_suffix"], retest.RETEST_SUFFIX)
        self.assertEqual(snap.name, "snapshot_after_retest_glx_reset")
        self.assertEqual(snap.status, PASS)
        ok = next(c for c in phase.checks if c.name == "retest_reset_ok")
        self.assertEqual(ok.status, PASS)

    def test_skipped_when_a_retest_test_failed(self):
        phase, snap, reset, tests = self._run(first_run(gddr_act_c3=PASS, gddr_act_all=FAIL))
        reset.assert_not_called()
        tests.assert_not_called()
        self.assertIsNone(snap)
        self.assertEqual([(c.name, c.status) for c in phase.checks], [("retest", SKIP)])
        self.assertIn("gddr_act_all", phase.checks[0].details)

    def test_skipped_when_a_retest_test_did_not_run(self):
        phase, snap, reset, tests = self._run(first_run(gddr_act_c3=PASS))
        reset.assert_not_called()
        self.assertEqual(phase.checks[0].status, SKIP)

    def test_broken_reset_fails_the_verdict_and_skips_the_rerun(self):
        phase, snap, reset, tests = self._run(first_run(gddr_act_c3=PASS, gddr_act_all=PASS), reset_status=FAIL)
        tests.assert_not_called()
        self.assertIsNone(snap)
        ok = next(c for c in phase.checks if c.name == "retest_reset_ok")
        self.assertEqual(ok.status, FAIL)
        report = {"phases": {"retest": {"checks": [{"name": c.name, "status": c.status} for c in phase.checks]}}}
        self.assertTrue(has_actionable_failure(report))

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
        self.assertEqual(TIERS["weekly"].retest, ("gddr_act_c3",))
        self.assertEqual(TIERS["full"].retest, ("gddr_act_all",))
        # gddr_act_all runs both of gddr_act_c3's passes.
        self.assertNotIn("gddr_act_c3", TIERS["full"].tests)


if __name__ == "__main__":
    unittest.main()
