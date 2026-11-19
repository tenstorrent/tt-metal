#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dispatch log analyzer tests for GTest skip reasons."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
EXABOX_DIR = TESTS_DIR.parent
for _path in (EXABOX_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from analyze_dispatch_results import analyze_log_file  # noqa: E402


SKIPPED_LOG = """\
[==========] Running 2 tests from 2 test suites.
[----------] 1 test from UnitMeshCQSingleCardFixture
[ RUN      ] UnitMeshCQSingleCardFixture.TensixTestReadWriteMultipleCoresL1
[       OK ] UnitMeshCQSingleCardFixture.TensixTestReadWriteMultipleCoresL1 (12 ms)
[----------] 1 test from UnitMeshCQMultiCQMultiDeviceEventFixture
[ RUN      ] UnitMeshCQMultiCQMultiDeviceEventFixture.SubDeviceAllocations
Metal | INFO | This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2
/work/tests/tt_metal/tt_metal/dispatch/multi_command_queue_fixture.hpp:34: Skipped
[  SKIPPED ] UnitMeshCQMultiCQMultiDeviceEventFixture.SubDeviceAllocations (0 ms)
[----------] 2 tests from 2 test suites (12 ms total)
[  PASSED  ] 1 test.
[  SKIPPED ] 1 test, listed below:
[  SKIPPED ] UnitMeshCQMultiCQMultiDeviceEventFixture.SubDeviceAllocations
"""


class TestDispatchSkipReasons(unittest.TestCase):
    def _analyze(self, text: str):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "dispatch_tests_20260819_031200.log"
            log.write_text(text, encoding="utf-8")
            return analyze_log_file(str(log))

    def test_gtest_skip_reason_captured(self):
        analysis = self._analyze(SKIPPED_LOG)
        self.assertEqual(len(analysis.tests_skipped), 1)
        self.assertEqual(
            analysis.skip_reasons,
            ["This suite must be run with TT_METAL_GTEST_NUM_HW_CQS=2"],
        )

    def test_no_skip_reason_without_marker(self):
        text = "[ RUN      ] Suite.Test\n[  SKIPPED ] Suite.Test (0 ms)\n"
        self.assertEqual(self._analyze(text).skip_reasons, [])


if __name__ == "__main__":
    unittest.main()
