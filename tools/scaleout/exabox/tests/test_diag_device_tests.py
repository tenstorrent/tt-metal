#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for how run_tests grades a gtest run, against a fake deployment
binary that behaves like the eth tests: it skips its test case unless slow
dispatch is on; and a DIDT runner run, against a fake tests/didt/didt_runner.py."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from diag.device_tests import DEPLOYMENT_BIN_CANDIDATES, DIDT_RUNNER, DIDT_RUNNERS, TESTS, run_tests  # noqa: E402
from diag.model import FAIL, PASS, Phase  # noqa: E402

# gtest's output for a test that calls GTEST_SKIP(), including the summary that
# lists the skipped test a second time.
FAKE_BINARY = """\
#!/bin/sh
test=MeshDispatchFixture.TensixDeploymentEthernet00LinkUp
echo "[ RUN      ] $test"
if [ -n "$TT_METAL_SLOW_DISPATCH_MODE" ]; then
    echo "[       OK ] $test (10 ms)"
    echo "[  PASSED  ] 1 test."
else
    echo "[  SKIPPED ] $test (10 ms)"
    echo "[  PASSED  ] 0 tests."
    echo "[  SKIPPED ] 1 test, listed below:"
    echo "[  SKIPPED ] $test"
fi
"""


class TestGtestGrading(unittest.TestCase):
    def run_fake(self, names):
        with tempfile.TemporaryDirectory() as tmp:
            tt_metal = Path(tmp)
            binary = tt_metal / DEPLOYMENT_BIN_CANDIDATES[0]
            binary.parent.mkdir(parents=True)
            binary.write_text(FAKE_BINARY)
            binary.chmod(0o755)
            phase = Phase(name="tests")
            with mock.patch.dict(os.environ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                os.environ.pop("TT_METAL_SLOW_DISPATCH_MODE", None)
                run_tests(tt_metal, names, phase, dry_run=False, logs_dir=tt_metal / "logs")
        return {c.name: c for c in phase.checks}

    def test_eth_tests_run_in_slow_dispatch(self):
        for name, check in self.run_fake(["eth_link_up", "eth_bandwidth"]).items():
            with self.subTest(name):
                self.assertEqual(check.status, PASS, check.details)
                self.assertEqual(check.data["env_overrides"]["TT_METAL_SLOW_DISPATCH_MODE"], "1")
                self.assertEqual(check.data["testcases"]["passed"], 1)

    def test_all_skipped_is_fail(self):
        # eth_link_up as it was before it set slow dispatch: its test skips and
        # gtest exits 0, having tested nothing.
        with mock.patch.dict(TESTS, {"eth_link_up": {"filter": "*TensixDeploymentEthernet*LinkUp"}}):
            check = self.run_fake(["eth_link_up"])["eth_link_up"]
        self.assertEqual(check.status, FAIL)
        self.assertIn("passed=0 failed=0 skipped=1", check.details)


# Stands in for tests/didt/didt_runner.py: FAKE_RUNNER picks what it does. It prints the run dir line and writes
# status.json as the real runner does.
FAKE_RUNNER = """\
import json, os, sys, time
from pathlib import Path

mode = os.environ["FAKE_RUNNER"]
out = Path(sys.argv[sys.argv.index("--out") + 1])
Path(os.environ["FAKE_RUNNER_ARGS"]).write_text(json.dumps(sys.argv[1:]))
if mode == "hang_forever":
    time.sleep(600)
if mode == "no_status":
    sys.exit(0)
run_dir = out / "LOG_2026_11_27_10_00_host_abc"
run_dir.mkdir(parents=True)
print(f"run dir {run_dir}  (32 chips)", flush=True)
steps = {
    "pass": [("nd:minimal", "PASS"), ("nd:ff1", "WARN")],
    "fail": [("nd:minimal", "PASS"), ("nd:ff1", "HANG")],
    "empty": [],
}[mode]
doc = {"steps": [{"key": k, "status": st, "reason": "r"} for k, st in steps]}
(run_dir / "status.json").write_text(json.dumps(doc))
sys.exit(1 if mode == "fail" else 0)
"""


class TestDidtRunnerGrading(unittest.TestCase):
    def run_fake(self, mode, name="didt_runner_daily", spec=None):
        with tempfile.TemporaryDirectory() as tmp:
            tt_metal = Path(tmp)
            runner = tt_metal / DIDT_RUNNER
            runner.parent.mkdir(parents=True)
            runner.write_text(FAKE_RUNNER)
            args_file = tt_metal / "args.json"
            phase = Phase(name="tests")
            specs = {name: spec or DIDT_RUNNERS[name]}
            with mock.patch.dict(
                os.environ, {"FAKE_RUNNER": mode, "FAKE_RUNNER_ARGS": str(args_file)}
            ), mock.patch.dict(DIDT_RUNNERS, specs), mock.patch(
                "diag.device_tests.DIDT_RUNNER_TERM_GRACE_S", 1
            ), redirect_stdout(
                io.StringIO()
            ), redirect_stderr(
                io.StringIO()
            ):
                run_tests(tt_metal, [name], phase, dry_run=False, logs_dir=tt_metal / "out" / "logs")
            args = json.loads(args_file.read_text()) if args_file.is_file() else None
            return {c.name: c for c in phase.checks}[name], args, tt_metal

    def test_pass(self):
        check, args, tt_metal = self.run_fake("pass")
        self.assertEqual(check.status, PASS, check.details)
        self.assertEqual(args, ["--config", "daily", "--out", str(tt_metal / "out" / "didt_runner_daily")])
        self.assertIn("passed=2 failed=0", check.details)
        self.assertTrue(check.data["status_json"].endswith("LOG_2026_11_27_10_00_host_abc/status.json"))
        self.assertEqual((check.data["rc"], check.data["testcases"]["passed"]), (0, 2))
        self.assertTrue(check.data["log_file"].endswith("logs/didt_runner_daily.log"))

    def test_weekly_runs_the_weekly_config(self):
        check, args, _ = self.run_fake("pass", name="didt_runner_weekly")
        self.assertEqual(check.status, PASS, check.details)
        self.assertEqual(args[:2], ["--config", "weekly"])

    def test_failed_screen_is_fail(self):
        check, _, _ = self.run_fake("fail")
        self.assertEqual(check.status, FAIL)
        self.assertIn("passed=1 failed=1", check.details)
        self.assertEqual(check.data["testcases"]["failures"], ["nd:ff1: HANG r"])

    def test_no_status_json_is_fail(self):
        check, _, _ = self.run_fake("no_status")
        self.assertEqual(check.status, FAIL)
        self.assertIn("passed=0 failed=0 no status.json", check.details)

    def test_nothing_ran_is_fail(self):
        check, _, _ = self.run_fake("empty")
        self.assertEqual(check.status, FAIL)
        self.assertIn("passed=0 failed=0", check.details)

    def test_timeout_kills_the_runner(self):
        start = time.time()
        check, _, _ = self.run_fake("hang_forever", spec={"config": "daily", "timeout_s": 1})
        self.assertLess(time.time() - start, 60)
        self.assertEqual(check.status, FAIL)
        self.assertIn("timeout after 1 s", check.details)


if __name__ == "__main__":
    unittest.main()
