#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for how run_tests grades a gtest run, against a fake deployment
binary that behaves like the eth tests: it skips its test case unless slow
dispatch is on."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from diag.device_tests import DEPLOYMENT_BIN_CANDIDATES, TESTS, run_tests  # noqa: E402
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

    def test_check_names_are_the_test_names(self):
        self.assertEqual(list(self.run_fake(["eth_link_up", "eth_bandwidth"])), ["eth_link_up", "eth_bandwidth"])

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


# A pytest that fails one test and writes a result file with a summary, as the ACT screen does.
FAKE_PYTEST = """\
#!/bin/sh
[ -n "$WRITE_RESULT" ] && echo '{"summary": {"details": "hits: chip 26", "hits": [{"chip": 26}]}}' > "$DIAG_RESULT_JSON"
echo "FAILED $1 - AssertionError"
echo "==== 1 failed in 1.00s ===="
exit 1
"""


class TestPytestResult(unittest.TestCase):
    def run_fake(self, write_result):
        with tempfile.TemporaryDirectory() as tmp:
            tt_metal = Path(tmp)
            pytest_bin = tt_metal / "python_env/bin/pytest"
            pytest_bin.parent.mkdir(parents=True)
            pytest_bin.write_text(FAKE_PYTEST)
            pytest_bin.chmod(0o755)
            logs = tt_metal / "logs"
            phase = Phase(name="tests")
            with mock.patch.dict(os.environ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                if write_result:
                    os.environ["WRITE_RESULT"] = "1"
                run_tests(tt_metal, ["gddr_act_c3"], phase, dry_run=False, logs_dir=logs, name_suffix="_after_reset")
            [check] = phase.checks
            return check, (logs / "gddr_act_c3_after_reset.json").exists()

    def test_summary_goes_into_the_check(self):
        check, kept = self.run_fake(write_result=True)
        self.assertTrue(kept)
        self.assertEqual(check.name, "gddr_act_c3_after_reset")
        self.assertEqual(check.status, FAIL)
        self.assertEqual(check.data["summary"]["hits"], [{"chip": 26}])
        self.assertTrue(check.data["result_file"].endswith("logs/gddr_act_c3_after_reset.json"))
        self.assertIn("passed=0 failed=1", check.details)
        self.assertTrue(check.details.endswith("hits: chip 26"))

    def test_no_result_file_leaves_the_check_as_is(self):
        check, kept = self.run_fake(write_result=False)
        self.assertFalse(kept)
        self.assertNotIn("summary", check.data)
        self.assertEqual(check.status, FAIL)


if __name__ == "__main__":
    unittest.main()
