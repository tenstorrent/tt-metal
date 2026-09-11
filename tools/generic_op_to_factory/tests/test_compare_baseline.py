# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Synthetic outcome comparisons; no named production runs or device access."""

import pytest

from tools.generic_op_to_factory.compare_baseline import compare, compare_outcomes
from tools.generic_op_to_factory.export_run import ExportError, export_snapshot
from tools.generic_op_to_factory.tests.test_export_run import ReadConnection, snapshot  # noqa: F401


def row(
    name="test_sample",
    status="passed",
    file="eval.golden_tests.sample_suite.test_golden",
):
    return {"test_file": file, "test_name": name, "status": status}


@pytest.mark.parametrize("status", ["passed", "failed", "error", "skipped", "xfail", "xpass"])
def test_matching_outcomes_do_not_imply_migration_readiness(status):
    result = compare_outcomes([row(status=status)], [row(status=status)])
    assert result["outcomes_match"] and result["same_case_set"]
    assert not result["migration_ready"]
    assert result["observed_failures"] == int(status in ("failed", "error", "xpass"))


def test_case_identity_not_aggregate_counts():
    result = compare_outcomes([row("test_first")], [row("test_second")])
    assert result["recorded_counts"] == result["observed_counts"]
    assert not result["outcomes_match"] and not result["same_case_set"]


def test_changed_status_and_incomplete_report():
    result = compare_outcomes([row("test_first"), row("test_second")], [row("test_first", "skipped")])
    assert len(result["missing"]) == 1
    assert result["changed"][0]["observed"] == "skipped"
    assert not result["outcomes_match"]


@pytest.mark.parametrize("bad", [[], [row(), row()], [row(file=None)], [row(status="unknown")]])
def test_ambiguous_results_rejected(bad):
    with pytest.raises(ExportError):  # allow-pytest.raises: host-only workflow validation
        compare_outcomes(bad, [row()])


@pytest.mark.parametrize("phase", [None, "initial", "refined"])
def test_phase_selection(snapshot, tmp_path, phase):
    connection, run_id = snapshot
    connection.execute("PRAGMA query_only = OFF")
    connection.execute("UPDATE test_results SET test_file=?", (row()["test_file"],))
    connection.execute("PRAGMA query_only = ON")
    export = tmp_path / "export"
    export_snapshot(ReadConnection(connection), run_id, export, database={"host": "fixture"})
    junit = tmp_path / "results.xml"
    junit.write_text(
        '<testsuite><testcase classname="eval.golden_tests.sample_suite.test_golden" name="test_sample[case]"/></testsuite>'
    )
    result = compare(export, junit, phase)
    assert result["selected_phase"] == phase
    assert result["outcomes_match"] and result["observed_counts"] == {"passed": 1}
    with pytest.raises(ExportError, match="empty"):  # allow-pytest.raises: host-only workflow validation
        compare(export, junit, "absent")
