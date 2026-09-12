# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from models.common.readiness_check.check_degenerate_output import Report, check_completion, main


def _check_text(text: str) -> Report:
    report = Report()
    check_completion(
        report,
        artifact=Path("artifact.json"),
        label="completion",
        text=text,
    )
    return report


def test_clean_text_has_no_findings():
    text = " ".join(f"word{index}" for index in range(60))
    report = _check_text(text)

    assert report.exit_code == 0
    assert report.findings == []


def test_adjacent_token_duplication_is_critical():
    text = " ".join(word for index in range(20) for word in (f"word{index}", f"word{index}"))
    report = _check_text(text)

    assert report.exit_code == 2
    assert [(finding.severity, finding.metric) for finding in report.findings] == [("critical", "adjacent_duplication")]


def test_phrase_loop_is_advisory_only():
    report = _check_text("alpha beta gamma " * 20)

    assert report.exit_code == 1
    assert [(finding.severity, finding.metric) for finding in report.findings] == [
        ("advisory", "trigram_loop_fraction")
    ]


@pytest.mark.parametrize(("severity", "expected_exit"), [("advisory", 1), ("critical", 2)])
def test_missing_artifact_policy_controls_exit_code(tmp_path, severity, expected_exit):
    assert main(["--root", str(tmp_path / "missing"), "--missing-artifacts", severity]) == expected_exit
