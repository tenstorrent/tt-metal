"""Tests for output formatters."""

from io import StringIO

from bug_checker.llm import Finding
from bug_checker.output import (
    findings_to_sarif,
    format_pr_comment,
    format_summary_comment,
    print_findings,
)


def _make_finding(**kwargs) -> Finding:
    defaults = dict(
        rule_id="test-rule",
        file="src/foo.cpp",
        line=42,
        message="Something is wrong",
        severity="warning",
        suggested_fix=None,
    )
    defaults.update(kwargs)
    return Finding(**defaults)


def test_sarif_output_structure():
    findings = [_make_finding()]
    sarif = findings_to_sarif(findings, ["test-rule"])
    assert sarif["version"] == "2.1.0"
    assert len(sarif["runs"]) == 1
    run = sarif["runs"][0]
    assert run["tool"]["driver"]["name"] == "bug-checker"
    assert len(run["results"]) == 1
    result = run["results"][0]
    assert result["ruleId"] == "test-rule"
    assert result["level"] == "warning"
    assert result["locations"][0]["physicalLocation"]["region"]["startLine"] == 42


def test_sarif_blocking_severity():
    findings = [_make_finding(severity="blocking")]
    sarif = findings_to_sarif(findings, ["test-rule"])
    assert sarif["runs"][0]["results"][0]["level"] == "error"


def test_sarif_with_suggested_fix():
    findings = [_make_finding(suggested_fix="fixed code")]
    sarif = findings_to_sarif(findings, ["test-rule"])
    result = sarif["runs"][0]["results"][0]
    assert "fixes" in result


def test_print_findings_empty():
    buf = StringIO()
    print_findings([], file=buf)
    assert "No findings" in buf.getvalue()


def test_print_findings_with_results():
    buf = StringIO()
    findings = [
        _make_finding(severity="blocking"),
        _make_finding(severity="warning", file="src/bar.cpp", line=10),
    ]
    print_findings(findings, file=buf)
    output = buf.getvalue()
    assert "[BLOCKING]" in output
    assert "[WARNING]" in output
    assert "1 blocking" in output
    assert "1 warning" in output


def test_format_pr_comment():
    finding = _make_finding()
    comment = format_pr_comment(finding)
    assert "test-rule" in comment
    assert "WARNING" in comment
    assert "Something is wrong" in comment


def test_format_pr_comment_with_fix():
    finding = _make_finding(suggested_fix="auto x = correct();")
    comment = format_pr_comment(finding)
    assert "```suggestion" in comment
    assert "auto x = correct();" in comment


def test_format_summary_comment():
    findings = [
        _make_finding(severity="blocking"),
        _make_finding(severity="warning"),
    ]
    summary = format_summary_comment(findings)
    assert "1 blocking" in summary
    assert "1 warning" in summary


def test_format_summary_no_findings():
    summary = format_summary_comment([])
    assert "No issues found" in summary
