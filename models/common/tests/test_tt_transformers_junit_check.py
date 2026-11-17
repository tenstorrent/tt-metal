# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""tests/scripts/check_tt_transformers_junit.py finds the one report for a run and passes
only when a test in it passed.

Host-only: runs the script on hand-written JUnit reports.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CHECK_SCRIPT = REPOSITORY_ROOT / "tests/scripts/check_tt_transformers_junit.py"

PASSED = '<testcase classname="t" name="passed" time="1.0"/>'
SKIPPED = '<testcase classname="t" name="skipped" time="0.0"><skipped message="skip"/></testcase>'
FAILED = '<testcase classname="t" name="failed" time="1.0"><failure message="fail"/></testcase>'
ERROR = '<testcase classname="t" name="error" time="1.0"><error message="error"/></testcase>'


def _write_report(path, *cases):
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">'
        + "".join(cases)
        + "</testsuite></testsuites>",
        encoding="utf-8",
    )


def _check(prefix):
    return subprocess.run([sys.executable, str(CHECK_SCRIPT), str(prefix)], capture_output=True, text=True)


@pytest.mark.parametrize(
    "report_suffix, cases",
    [
        ("_20261008_201748.xml", (PASSED,)),  # renamed by tt-transformers' conftest under CI=true
        (".xml", (PASSED,)),
        ("_20261008_201748.xml", (SKIPPED, PASSED, FAILED)),
    ],
)
def test_report_with_a_passed_test_passes(tmp_path, report_suffix, cases):
    prefix = tmp_path / "tt_transformers_20261008_201741_837538071"
    _write_report(Path(f"{prefix}{report_suffix}"), *cases)

    result = _check(prefix)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "cases",
    [(SKIPPED, SKIPPED), (FAILED,), (ERROR, SKIPPED), ()],
    ids=["all-skipped", "failure-only", "error-and-skipped", "empty"],
)
def test_report_without_a_passed_test_fails(tmp_path, cases):
    prefix = tmp_path / "tt_transformers_20261008_201741_837538071"
    _write_report(Path(f"{prefix}_20261008_201748.xml"), *cases)

    result = _check(prefix)

    assert result.returncode != 0
    assert "No tt-transformers test passed" in result.stderr


def test_missing_report_fails(tmp_path):
    result = _check(tmp_path / "tt_transformers_20261008_201741_837538071")

    assert result.returncode != 0
    assert "found 0" in result.stderr


def test_two_matching_reports_fail(tmp_path):
    prefix = tmp_path / "tt_transformers_20261008_201741_837538071"
    _write_report(Path(f"{prefix}.xml"), PASSED)
    _write_report(Path(f"{prefix}_20261008_201748.xml"), PASSED)

    result = _check(prefix)

    assert result.returncode != 0
    assert "found 2" in result.stderr


def test_reports_of_other_runs_are_ignored(tmp_path):
    prefix = tmp_path / "tt_transformers_20261008_201741_837538071"
    _write_report(Path(f"{prefix}_20261008_201748.xml"), SKIPPED)
    _write_report(Path(f"{prefix}_2_20261008_201748.xml"), PASSED)
    _write_report(tmp_path / "tt_transformers_20261008_201741_837538072_20261008_201748.xml", PASSED)

    result = _check(prefix)

    assert result.returncode != 0
    assert "No tt-transformers test passed" in result.stderr
