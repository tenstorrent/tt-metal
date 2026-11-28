# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the detect-slow-tests action script.

Run with:
    python -m pytest .github/actions/detect-slow-tests/test_detect_slow_tests.py --noconftest
"""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "detect-slow-tests.py"

SLOW_XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests">
  <testsuite name="pytest" errors="0" failures="0" skipped="0" tests="2" time="42.5">
    <testcase classname="tests.test_slow" name="test_quick" time="0.5"/>
    <testcase classname="tests.test_slow" name="test_slow" time="42.0"/>
  </testsuite>
</testsuites>
"""

FAST_XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests">
  <testsuite name="pytest" errors="0" failures="0" skipped="0" tests="1" time="0.5">
    <testcase classname="tests.test_slow" name="test_quick" time="0.5"/>
  </testsuite>
</testsuites>
"""


def load_module():
    spec = importlib.util.spec_from_file_location("detect_slow_tests_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DetectSlowTestsCliTests(unittest.TestCase):
    def _run(self, xml, timeout=10):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir) / "reports"
            report_dir.mkdir()
            (report_dir / "report.xml").write_text(xml, encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(SCRIPT), str(report_dir), str(timeout)],
                capture_output=True,
                text=True,
            )

    def test_slow_report_does_not_also_claim_no_slow_tests(self):
        result = self._run(SLOW_XML)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("test_slow", result.stdout)
        self.assertNotIn("No slow tests slower than", result.stdout)

    def test_fast_report_prints_no_slow_tests_once(self):
        result = self._run(FAST_XML)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("No slow tests slower than"), 1)
        self.assertNotIn("Detected Slow Tests", result.stdout)


class DetectSlowTestsApiTests(unittest.TestCase):
    def test_returns_the_slow_tests(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir) / "reports"
            report_dir.mkdir()
            (report_dir / "report.xml").write_text(SLOW_XML, encoding="utf-8")
            slow = module.detect_slow_tests(str(report_dir), 10)
        self.assertEqual(len(slow), 1)
        self.assertIn("tests.test_slow.test_slow", slow[0])

    def test_returns_empty_list_when_nothing_is_slow(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir) / "reports"
            report_dir.mkdir()
            (report_dir / "report.xml").write_text(FAST_XML, encoding="utf-8")
            self.assertEqual(module.detect_slow_tests(str(report_dir), 10), [])


if __name__ == "__main__":
    unittest.main()
