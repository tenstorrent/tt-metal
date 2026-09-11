# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Preserve recorded evaluator result semantics without importing its package."""

import pytest

from tools.generic_op_to_factory.classify_failures import parse_junit_xml


@pytest.mark.parametrize(
    "body,status,category",
    [
        ("", "passed", None),
        ('<failure message="severity=precision"/>', "failed", "numerical-precision"),
        ('<failure message="device timeout; severity=bug"/>', "failed", "hang"),
        ('<error message="CompilationError"/>', "error", "compilation"),
        ('<failure message="[XPASS(strict)]"/>', "xpass", None),
        ('<skipped type="pytest.xfail" message="unsupported"/>', "xfail", None),
        ('<skipped message="INFEASIBLE_L1"/>', "skipped", "infeasible"),
        ('<skipped message="earlier case hung"/>', "skipped", "hang"),
    ],
)
def test_recorded_status_and_properties(tmp_path, body, status, category):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuites><testsuite><testcase classname="eval.golden_tests.sample.test_golden" '
        'name="test_op[shape-dtype=FLOAT32]">'
        '<properties><property name="tag" value="recorded_shape"/>'
        '<property name="metric.pcc" value="0.99"/>'
        '<property name="metric.rms" value="invalid"/>'
        '<property name="axis.groups" value="4"/></properties>' + body + "</testcase></testsuite></testsuites>"
    )
    (row,) = parse_junit_xml(junit)
    assert row["status"] == status
    assert row["failure_category"] == category
    assert row["shape"] == "recorded_shape"
    assert row["nodeid"] == "eval/golden_tests/sample/test_golden.py::test_op[shape-dtype=FLOAT32]"
    assert row["observed_axes"] == {"groups": 4}
    assert row["pcc"] == (None if status in ("xfail", "skipped") else 0.99)
    assert row["rms"] is None
