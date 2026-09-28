#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for diag_runner.py's pure parsing helpers: the ipmitool FRU
output parser and the pytest output markers (summary counts, failing-node
lines). These lock in the text-parsing edge cases without needing hardware."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SUITE_DIR = TESTS_DIR.parent / "health_check_test_suite"
sys.path.insert(0, str(SUITE_DIR))

from diag_runner import (  # noqa: E402
    PYTEST_FAILED_LINE_RE,
    parse_fru_print,
    parse_pytest_summary_counts,
)

# Abridged real `ipmitool fru print` output from a BH Galaxy 6U (Quanta S7T).
FRU_SAMPLE = """\
FRU Device Description : Builtin FRU Device (ID 0)
 Chassis Type          : Rack Mount Chassis
 Chassis Part Number   : ---
 Chassis Extra         : ---
 Chassis Extra         : ---
 Board Mfg Date        : Tue Nov 21 17:00:00 2023
 Board Mfg             : Quanta Computer Inc.
 Board Product         : S7T-MB
 Board Serial          : S7TMBC3A251002007
 Product Name          : ---

FRU Device Description : FP_FRU (ID 1)
 Chassis Serial        : QTWS7TKC260200001
 Product Name          : Galaxy Blackhole 6U Server
 Product Serial        : QTWS7TKC260200001

FRU Device Description : UBB0_FRU (ID 2)
 Board Product         : S7T-UBB-BH
 Board Serial          : T38S7TBA00806010018
 Board Part Number     : 38S7TBA0080
"""


class TestParseFruPrint(unittest.TestCase):
    def test_devices_split_on_description_lines(self):
        devices = parse_fru_print(FRU_SAMPLE)
        self.assertEqual(len(devices), 3)
        self.assertEqual(
            [d["description"] for d in devices],
            ["Builtin FRU Device (ID 0)", "FP_FRU (ID 1)", "UBB0_FRU (ID 2)"],
        )

    def test_colon_containing_values_survive(self):
        devices = parse_fru_print(FRU_SAMPLE)
        self.assertEqual(devices[0]["Board Mfg Date"], "Tue Nov 21 17:00:00 2023")

    def test_placeholder_fields_dropped(self):
        devices = parse_fru_print(FRU_SAMPLE)
        self.assertNotIn("Chassis Part Number", devices[0])
        self.assertNotIn("Product Name", devices[0])
        self.assertNotIn("Chassis Extra", devices[0])
        self.assertEqual(devices[1]["Product Name"], "Galaxy Blackhole 6U Server")

    def test_repeated_key_keeps_last_real_value(self):
        text = (
            "FRU Device Description : X (ID 0)\n" " Board Extra           : first\n" " Board Extra           : second\n"
        )
        self.assertEqual(parse_fru_print(text)[0]["Board Extra"], "second")

    def test_leading_keyless_lines_ignored(self):
        # Key:value lines before any device block, and non-kv lines, are skipped.
        self.assertEqual(parse_fru_print(" Board Serial : X\nnot a kv line\n"), [])

    def test_empty_input(self):
        self.assertEqual(parse_fru_print(""), [])


class TestParsePytestSummaryCounts(unittest.TestCase):
    def test_mixed_summary(self):
        line = "===== 3 passed, 1 failed, 2 deselected in 600.12s =====\n"
        self.assertEqual(parse_pytest_summary_counts(line), (3, 1))

    def test_all_pass(self):
        line = "========== 4 passed, 16 deselected in 4212.55s ==========\n"
        self.assertEqual(parse_pytest_summary_counts(line), (4, 0))

    def test_errors_count_as_failed(self):
        self.assertEqual(parse_pytest_summary_counts("=== 2 passed, 1 error in 50.00s ===\n"), (2, 1))
        self.assertEqual(parse_pytest_summary_counts("=== 16 deselected, 4 errors in 7.63s ===\n"), (0, 4))

    def test_no_tests_ran_is_not_a_summary(self):
        # No counts at all -> None; the caller's zero-initialized counts then
        # record the run as FAIL rather than a silent PASS.
        self.assertIsNone(parse_pytest_summary_counts("========== no tests ran in 0.12s ==========\n"))

    def test_section_headers_and_ordinary_lines_ignored(self):
        self.assertIsNone(parse_pytest_summary_counts("==================== ERRORS ====================\n"))
        self.assertIsNone(
            parse_pytest_summary_counts("==================== test session starts ====================\n")
        )
        self.assertIsNone(parse_pytest_summary_counts("collected 20 items / 16 deselected / 4 selected\n"))


class TestPytestFailedLineRe(unittest.TestCase):
    def test_failed_line_captures_node_id(self):
        m = PYTEST_FAILED_LINE_RE.match(
            "FAILED tests/didt/test_minimal_matmul.py::test_minimal_matmul[bf16_HiFi2-galaxy] - AssertionError\n"
        )
        assert m is not None
        self.assertEqual(m.group(1), "tests/didt/test_minimal_matmul.py::test_minimal_matmul[bf16_HiFi2-galaxy]")

    def test_error_line_captures_node_id(self):
        m = PYTEST_FAILED_LINE_RE.match(
            "ERROR tests/didt/test_minimal_matmul.py::test_minimal_matmul[bf16_HiFi2-galaxy] - RuntimeError: x\n"
        )
        assert m is not None
        self.assertEqual(m.group(1), "tests/didt/test_minimal_matmul.py::test_minimal_matmul[bf16_HiFi2-galaxy]")

    def test_bare_error_and_error_colon_lines_not_matched(self):
        self.assertIsNone(PYTEST_FAILED_LINE_RE.match("ERROR\n"))
        self.assertIsNone(PYTEST_FAILED_LINE_RE.match("ERROR: usage: pytest [options]\n"))


if __name__ == "__main__":
    unittest.main()
