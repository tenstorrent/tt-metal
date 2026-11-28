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
    MEM_BENCH_CANDIDATES,
    MGD,
    TESTS,
    fault_parts,
    name_faults,
    run_tests,
)
from diag.galaxy import format_link_end  # noqa: E402
from diag.model import FAIL, PASS, WARN, Phase  # noqa: E402

# gtest's output for a test that calls GTEST_SKIP(), including the summary that
# lists the skipped test a second time. With no mesh graph descriptor the suite
# setup throws first, as metal's mesh discovery does on a Galaxy.
FAKE_BINARY = """\
#!/bin/sh
test=MeshDispatchFixture.TensixDeploymentEthernet00LinkUp
[ -z "$TT_MESH_GRAPH_DESC_PATH" ] && echo 'C++ exception with description "Resource busy" thrown in SetUpTestSuite().'
echo "[ RUN      ] $test"
if [ -n "$TT_METAL_SLOW_DISPATCH_MODE" ] && [ -n "$TT_MESH_GRAPH_DESC_PATH" ]; then
    echo "[       OK ] $test (10 ms)"
    echo "[  PASSED  ] 1 test."
else
    echo "test_eth_link_up.cpp:104: Skipped"
    echo "fast dispatch to active ethernet has been removed"
    echo "[  SKIPPED ] $test (10 ms)"
    echo "[  PASSED  ] 0 tests."
    echo "[  SKIPPED ] 1 test, listed below:"
    echo "[  SKIPPED ] $test"
fi
"""


class TestGtestGrading(unittest.TestCase):
    def run_fake(self, names, **environ):
        with tempfile.TemporaryDirectory() as tmp:
            tt_metal = Path(tmp)
            binary = tt_metal / DEPLOYMENT_BIN_CANDIDATES[0]
            binary.parent.mkdir(parents=True)
            binary.write_text(FAKE_BINARY)
            binary.chmod(0o755)
            phase = Phase(name="tests")
            with mock.patch.dict(os.environ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                os.environ.pop("TT_METAL_SLOW_DISPATCH_MODE", None)
                os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
                os.environ.update(environ)
                run_tests(tt_metal, names, phase, dry_run=False, logs_dir=tt_metal / "logs")
        return {c.name: c for c in phase.checks}

    def test_check_names_are_the_test_names(self):
        self.assertEqual(list(self.run_fake(["eth_link_up", "eth_bandwidth"])), ["eth_link_up", "eth_bandwidth"])

    def test_eth_tests_run_in_slow_dispatch(self):
        for name, check in self.run_fake(["eth_link_up", "eth_bandwidth"]).items():
            with self.subTest(name):
                self.assertEqual(check.status, PASS, check.details)
                self.assertEqual(check.data["env_overrides"]["TT_METAL_SLOW_DISPATCH_MODE"], "1")
                self.assertTrue(check.data["env_overrides"]["TT_MESH_GRAPH_DESC_PATH"].endswith(MGD))
                self.assertEqual(check.data["testcases"]["passed"], 1)

    def test_all_skipped_is_fail(self):
        # eth_link_up as it was before it set slow dispatch: its test skips and
        # gtest exits 0, having tested nothing.
        with mock.patch.dict(TESTS, {"eth_link_up": {"filter": "*TensixDeploymentEthernet*LinkUp", "ip": "eth"}}):
            check = self.run_fake(["eth_link_up"])["eth_link_up"]
        self.assertEqual((check.status, check.parts), (FAIL, []))
        self.assertTrue(check.details.startswith("not run: skipped: fast dispatch to active ethernet"), check.details)
        self.assertIn("passed=0 failed=0 skipped=1", check.details)

    def test_a_suite_setup_that_throws_is_not_run(self):
        check = self.run_fake(["eth_link_up"], TT_MESH_GRAPH_DESC_PATH="")["eth_link_up"]
        self.assertEqual(check.status, WARN)
        self.assertTrue(check.details.startswith("not run: suite setup failed: Resource busy"), check.details)


# A pytest that fails one test and writes a result file with a summary, as the ACT screen does.
FAKE_PYTEST = """\
#!/bin/sh
if [ -n "$NO_NUMPY" ]; then
    echo "ImportError while loading conftest '/src/conftest.py'."
    echo "E   ModuleNotFoundError: No module named 'numpy'"
    exit 4
fi
[ -n "$WRITE_RESULT" ] && echo '{"summary": {"details": "hits: chip 26", "hits": [{"chip": 26}]}}' > "$DIAG_RESULT_JSON"
echo "FAILED $1 - AssertionError"
echo "==== 1 failed in 1.00s ===="
exit 1
"""


class TestPytestResult(unittest.TestCase):
    def run_fake(self, write_result, **environ):
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
                os.environ.update(environ)
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

    def test_a_conftest_that_fails_to_import_is_not_run(self):
        check, _ = self.run_fake(write_result=False, NO_NUMPY="1")
        self.assertEqual(check.status, WARN)
        self.assertTrue(check.details.startswith("not run: ModuleNotFoundError: No module named 'numpy'"))
        self.assertEqual(check.parts, [])


# Fault lines as the deployment eth tests log them.
ETH_LOG = """\
2026-11-28 01:00:00.000 | info     | Test | sender device id: 3 (0000:46:00.0, ubb: 2, chip: 6), receiver device id: 7 (0000:c1:00.0, ubb: 4, chip: 1)
2026-11-28 01:00:00.000 | info     | Test |   sender core: (x=0,y=4), receiver core: (x=0,y=5) (QSFP)
2026-11-28 01:00:01.000 | critical | Test | Expected at least: 40 Gbps, got 12.5 Gbps
2026-11-28 01:00:02.000 | info     | Test | sender device id: 3 (0000:46:00.0, ubb: 2, chip: 6), receiver device id: 4 (0000:47:00.0, ubb: 2, chip: 7)
2026-11-28 01:00:02.000 | info     | Test |   sender core: (x=0,y=8), receiver core: (x=0,y=9) (TRACE)
2026-11-28 01:00:03.000 | critical | Test | Failing links:
2026-11-28 01:00:03.000 | critical | Test | \tSender device 4, receiver device 3, sender core (x=0,y=9), receiver core (x=0,y=8), processor 0
2026-11-28 01:00:04.000 | critical | Test | missing links: chip[3 (0000:46:00.0), ubb: 2, chip: 6]: expected 14 links, got 13
"""


class TestFaultParts(unittest.TestCase):
    def test_links_and_chips_are_named(self):
        parts, links = fault_parts(ETH_LOG.splitlines())
        self.assertEqual(len(links), 2)
        self.assertEqual(links[0], f"{format_link_end('0000:46:00.0', 4)} <-> {format_link_end('0000:c1:00.0', 5)}")
        self.assertTrue(links[0].startswith("0000:46:00.0 (UBB2/U6) ch4, "), links[0])
        self.assertEqual(links[1], f"{format_link_end('0000:47:00.0', 9)} <-> {format_link_end('0000:46:00.0', 8)}")
        self.assertEqual(parts[:2], ["0000:46:00.0 (UBB2/U6)", "0000:c1:00.0 (UBB3/U1)"])
        self.assertIn("0000:47:00.0 (UBB2/U7)", parts)

    def test_details_lead_with_the_links_and_hits_add_chips(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "t.log"
            log.write_text(ETH_LOG)
            data = {}
            details, parts = name_faults(log, "rc=1", data, hits=[{"bdf": "0000:41:00.0"}, "junk"])
        self.assertTrue(details.startswith("link 0000:46:00.0"), details)
        self.assertTrue(details.endswith(" rc=1"))
        self.assertEqual(len(data["links"]), 2)
        self.assertEqual(parts[-1], "0000:41:00.0 (UBB2/U1)")

    def test_a_clean_log_names_nothing(self):
        self.assertEqual(fault_parts(ETH_LOG.replace("critical", "info").splitlines()), ([], []))


# mem_bench writing its JSON result; FAKE_ERROR set makes the one benchmark fail.
FAKE_MEM_BENCH = """\
#!/bin/sh
for a in "$@"; do case "$a" in --benchmark_out=*) out="${a#--benchmark_out=}";; esac; done
if [ -n "$FAKE_ERROR" ]; then
    echo '{"benchmarks": [{"name": "b", "error_occurred": true, "error_message": "no device"}]}' > "$out"
else
    echo '{"benchmarks": [{"name": "b", "dev_bw": 24500000000}]}' > "$out"
fi
"""


class TestMemBench(unittest.TestCase):
    def run_fake(self, error=""):
        with tempfile.TemporaryDirectory() as tmp:
            tt_metal = Path(tmp)
            binary = tt_metal / MEM_BENCH_CANDIDATES[0]
            binary.parent.mkdir(parents=True)
            binary.write_text(FAKE_MEM_BENCH)
            binary.chmod(0o755)
            phase = Phase(name="tests")
            with mock.patch.dict(os.environ, {"FAKE_ERROR": error}), redirect_stdout(io.StringIO()):
                run_tests(tt_metal, ["pcie_read"], phase, dry_run=False, logs_dir=tt_metal / "logs")
        return phase.checks[0]

    def test_bandwidth_is_recorded(self):
        check = self.run_fake()
        self.assertEqual((check.name, check.status, check.ip), ("pcie_read", PASS, "pcie"))
        self.assertEqual(check.data["gbps"], {"b": 24.5})
        self.assertIn("24.5 GB/s", check.details)

    def test_a_benchmark_error_fails(self):
        check = self.run_fake(error="1")
        self.assertEqual(check.status, FAIL)
        self.assertIn("no device", check.details)

    def test_missing_binary_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            phase = Phase(name="tests")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                run_tests(Path(tmp), ["pcie_write"], phase, dry_run=False, logs_dir=Path(tmp) / "logs")
        self.assertEqual(phase.checks[0].status, FAIL)
        self.assertIn("mem_bench", phase.checks[0].details)


if __name__ == "__main__":
    unittest.main()
