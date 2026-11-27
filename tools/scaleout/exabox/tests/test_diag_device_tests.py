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

from diag.device_tests import (  # noqa: E402
    DEPLOYMENT_BIN_CANDIDATES,
    DRAM_TEST_ENV_KEYS,
    GDDR_TESTS,
    TESTS,
    run_tests,
)
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
    def run_fake(self, names, dry_run=False, env=None, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            tt_metal = Path(tmp)
            binary = tt_metal / DEPLOYMENT_BIN_CANDIDATES[0]
            binary.parent.mkdir(parents=True)
            binary.write_text(FAKE_BINARY)
            binary.chmod(0o755)
            phase = Phase(name="tests")
            with mock.patch.dict(os.environ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                os.environ.pop("TT_METAL_SLOW_DISPATCH_MODE", None)
                for key in DRAM_TEST_ENV_KEYS:
                    os.environ.pop(key, None)
                os.environ.update(env or {})
                run_tests(tt_metal, names, phase, dry_run=dry_run, logs_dir=tt_metal / "logs", **kwargs)
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


class TestGddrMbytes(unittest.TestCase):
    run_fake = TestGtestGrading.run_fake

    def test_default_leaves_test_size_unset(self):
        check = self.run_fake(["gddr_fast"], dry_run=True)["gddr_fast"]
        self.assertNotIn("DRAM_TEST_MBYTES", check.data["env_overrides"])
        self.assertEqual(check.data["env_overrides"]["DRAM_TEST_FAST"], "1")

    def test_gddr_mbytes_reaches_gddr_tests_only(self):
        checks = self.run_fake(["gddr_fast", "gddr_full", "eth_link_up"], dry_run=True, gddr_mbytes=1024)
        for name in GDDR_TESTS:
            with self.subTest(name):
                self.assertEqual(checks[name].data["env_overrides"]["DRAM_TEST_MBYTES"], "1024")
                self.assertIn("DRAM_TEST_MBYTES=1024", checks[name].data["command"])
        self.assertNotIn("DRAM_TEST_MBYTES", checks["eth_link_up"].data["env_overrides"])

    def test_ambient_dram_knobs_are_recorded(self):
        env = {"DRAM_TEST_MBYTES": "256", "DRAM_TEST_LOOPS": "3"}
        check = self.run_fake(["gddr_full"], dry_run=True, env=env)["gddr_full"]
        self.assertEqual(check.data["env_overrides"]["DRAM_TEST_MBYTES"], "256")
        self.assertEqual(check.data["env_overrides"]["DRAM_TEST_LOOPS"], "3")

    def test_flag_overrides_ambient(self):
        check = self.run_fake(["gddr_full"], dry_run=True, env={"DRAM_TEST_MBYTES": "256"}, gddr_mbytes=2048)[
            "gddr_full"
        ]
        self.assertEqual(check.data["env_overrides"]["DRAM_TEST_MBYTES"], "2048")


if __name__ == "__main__":
    unittest.main()
