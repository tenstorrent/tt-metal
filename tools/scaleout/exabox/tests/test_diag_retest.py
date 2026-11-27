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

    def _run(
        self,
        first: Phase,
        reset_status: str = PASS,
        dry_run: bool = False,
        snapshot_error: bool = False,
        names=("gddr_act_c3", "gddr_act_all"),
        boots=None,
        reset_statuses=(),
        failing=(),
    ):
        """reset_statuses: per round, overriding reset_status. failing: (test, suffix) pairs whose rerun fails."""
        phase = Phase(name="retest")
        resets_done = []

        def fake_reset(tt_smi, plan, ph, dry, logs_dir=None):
            resets_done.append(logs_dir)
            k = len(resets_done)
            status = reset_statuses[k - 1] if k <= len(reset_statuses) else reset_status
            ph.add(Check(name=f"reset_1_{plan[0].lstrip('-')}", status=status))

        def fake_tests(tt_metal, test_names, ph, dry, logs_dir, name_suffix=""):
            for t in test_names:
                ph.add(Check(name=f"{t}{name_suffix}", status=FAIL if (t, name_suffix) in failing else PASS))

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
            retest, "run_tests", side_effect=fake_tests
        ) as tests, redirect_stdout(
            io.StringIO()
        ):
            snaps = retest.run_retest(
                None,
                Path("."),
                names,
                "-glx_reset",
                first,
                phase,
                dry_run,
                self.logs,
                self.logs / "snap.json",
                boots=boots,
            )
        return phase, snaps, reset, tests

    def test_reruns_after_reset_and_snapshot(self):
        phase, snaps, reset, tests = self._run(first_run(gddr_act_c3=PASS, gddr_act_all=PASS, gddr_full=FAIL))
        reset.assert_called_once()
        self.assertEqual(reset.call_args.args[1], ("-glx_reset",))
        # Its reset log must not overwrite the pre-test loop's reset_1_r.log.
        self.assertEqual(reset.call_args.kwargs["logs_dir"], self.logs / "retest")
        tests.assert_called_once()
        self.assertEqual(tests.call_args.args[1], ("gddr_act_c3", "gddr_act_all"))
        self.assertEqual(tests.call_args.kwargs["name_suffix"], retest.RETEST_SUFFIX)
        self.assertEqual([s.name for s in snaps], ["snapshot_after_retest_glx_reset"])
        self.assertEqual(snaps[0].status, PASS)
        ok = next(c for c in phase.checks if c.name == "retest_reset_ok")
        self.assertEqual(ok.status, PASS)

    def test_skipped_when_a_retest_test_failed(self):
        phase, snaps, reset, tests = self._run(first_run(gddr_act_c3=PASS, gddr_act_all=FAIL))
        reset.assert_not_called()
        tests.assert_not_called()
        self.assertEqual(snaps, [])
        self.assertEqual([(c.name, c.status) for c in phase.checks], [("retest", SKIP)])
        self.assertIn("gddr_act_all", phase.checks[0].details)

    def test_skipped_when_a_retest_test_did_not_run(self):
        phase, snaps, reset, tests = self._run(first_run(gddr_act_c3=PASS))
        reset.assert_not_called()
        self.assertEqual(phase.checks[0].status, SKIP)

    def test_broken_reset_fails_the_verdict_and_skips_the_rerun(self):
        phase, snaps, reset, tests = self._run(first_run(gddr_act_c3=PASS, gddr_act_all=PASS), reset_status=FAIL)
        tests.assert_not_called()
        self.assertEqual(snaps, [])
        ok = next(c for c in phase.checks if c.name == "retest_reset_ok")
        self.assertEqual(ok.status, FAIL)
        report = {"phases": {"retest": {"checks": [{"name": c.name, "status": c.status} for c in phase.checks]}}}
        self.assertTrue(has_actionable_failure(report))

    def test_failed_snapshot_keeps_the_pre_test_snapshot_as_verdict(self):
        phase, snaps, reset, tests = self._run(first_run(gddr_act_c3=PASS, gddr_act_all=PASS), snapshot_error=True)
        self.assertEqual(snaps, [])
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
        phase, snaps, reset, tests = self._run(first_run(gddr_act_c3=PASS, gddr_act_all=PASS), dry_run=True)
        self.assertEqual(snaps, [])
        tests.assert_called_once()

    def test_more_boots_rerun_only_the_tests_that_want_them(self):
        phase, snaps, reset, tests = self._run(
            first_run(gddr_act_c3=PASS, gddr_retention=PASS),
            names=("gddr_act_c3", "gddr_retention"),
            boots={"gddr_retention": 5},
        )
        # 5 boots: the first run, then 4 rounds, each with its own reset log dir.
        self.assertEqual(
            [c.kwargs["logs_dir"] for c in reset.call_args_list],
            [self.logs / "retest", self.logs / "retest_2", self.logs / "retest_3", self.logs / "retest_4"],
        )
        self.assertEqual(
            [(c.args[1], c.kwargs["name_suffix"]) for c in tests.call_args_list],
            [
                (("gddr_act_c3", "gddr_retention"), "_after_reset"),
                (("gddr_retention",), "_after_reset_2"),
                (("gddr_retention",), "_after_reset_3"),
                (("gddr_retention",), "_after_reset_4"),
            ],
        )
        self.assertEqual(
            [s.name for s in snaps],
            [
                "snapshot_after_retest_glx_reset",
                "snapshot_after_retest_glx_reset_2",
                "snapshot_after_retest_glx_reset_3",
                "snapshot_after_retest_glx_reset_4",
            ],
        )
        names = [c.name for c in phase.checks]
        # Reset checks are numbered by round, so no two share a name, and keep the reset_ prefix.
        self.assertEqual(
            [n for n in names if n.startswith("reset_")],
            ["reset_1_glx_reset", "reset_2_glx_reset", "reset_3_glx_reset", "reset_4_glx_reset"],
        )
        self.assertEqual(len(names), len(set(names)))
        ok = [c for c in phase.checks if c.name == "retest_reset_ok"]
        self.assertEqual([(c.status, c.details) for c in ok], [(PASS, "unit back after tt-smi -glx_reset, 4 times")])

    def test_a_failing_round_stops_the_rest(self):
        phase, snaps, reset, tests = self._run(
            first_run(gddr_retention=PASS),
            names=("gddr_retention",),
            boots={"gddr_retention": 5},
            failing={("gddr_retention", "_after_reset_2")},
        )
        self.assertEqual(reset.call_count, 2)
        self.assertEqual(tests.call_count, 2)
        self.assertEqual(len(snaps), 2)
        stop = next(c for c in phase.checks if c.name == "retest_more_boots")
        self.assertEqual(stop.status, SKIP)
        self.assertIn("gddr_retention_after_reset_2 failed", stop.details)
        self.assertIn("gddr_retention on 2 more boot(s)", stop.details)

    def test_a_later_reset_that_breaks_the_unit_fails_the_run(self):
        phase, snaps, reset, tests = self._run(
            first_run(gddr_retention=PASS),
            names=("gddr_retention",),
            boots={"gddr_retention": 5},
            reset_statuses=(PASS, PASS, FAIL),
        )
        self.assertEqual(tests.call_count, 2)
        self.assertEqual(len(snaps), 2)
        ok = [c for c in phase.checks if c.name == "retest_reset_ok"]
        self.assertEqual(len(ok), 1)
        self.assertEqual(ok[0].status, FAIL)
        self.assertIn("retest round 3", ok[0].details)
        self.assertIn("reset_3_glx_reset", ok[0].details)


class TestTierRetest(unittest.TestCase):
    def test_retest_names_are_tier_tests(self):
        for name, tier in TIERS.items():
            for t in tier.retest:
                self.assertIn(t, tier.tests, name)

    def test_act_tests_are_retested(self):
        self.assertEqual(TIERS["weekly"].retest, ("gddr_act_c3", "gddr_retention"))
        self.assertEqual(TIERS["full"].retest, ("gddr_act_all", "gddr_retention"))
        # gddr_act_all runs both of gddr_act_c3's passes.
        self.assertNotIn("gddr_act_c3", TIERS["full"].tests)

    def test_retention_runs_on_five_boots(self):
        for name in ("weekly", "full"):
            self.assertEqual(dict(TIERS[name].retest_boots), {"gddr_retention": 5}, name)
        self.assertNotIn("gddr_retention", TIERS["daily"].tests)


if __name__ == "__main__":
    unittest.main()
