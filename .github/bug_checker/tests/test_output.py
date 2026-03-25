"""Tests for output formatters."""

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
    print_findings([])  # smoke test — loguru writes to stderr


def test_print_findings_with_results():
    findings = [
        _make_finding(severity="blocking"),
        _make_finding(severity="warning", file="src/bar.cpp", line=10),
    ]
    print_findings(findings)  # smoke test — loguru writes to stderr


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


def test_format_summary_comment_with_failures():
    findings = [_make_finding()]
    summary = format_summary_comment(findings, comment_failures=2)
    assert "2 comment(s) could not be posted" in summary


def test_format_summary_comment_with_skipped_rules():
    summary = format_summary_comment([], skipped_rules=["ccl-ring-buffer-mismatch", "reshape-dim-check"])
    assert "2 rule(s) were skipped" in summary
    assert "`ccl-ring-buffer-mismatch`" in summary
    assert "`reshape-dim-check`" in summary
    assert "Results may be incomplete" in summary


def test_format_summary_comment_skipped_rules_shown_even_with_no_findings():
    summary = format_summary_comment([], skipped_rules=["my-rule"])
    assert "No issues found" in summary
    assert "my-rule" in summary


def test_format_summary_comment_no_skipped_rules_no_note():
    summary = format_summary_comment([_make_finding()])
    assert "skipped" not in summary


def test_format_summary_comment_with_truncated_rules():
    summary = format_summary_comment([], truncated_rules=["ccl-ring-buffer-mismatch"])
    assert "truncated" in summary
    assert "`ccl-ring-buffer-mismatch`" in summary
    assert "breaking this PR into smaller pieces" in summary


def test_format_summary_comment_truncated_shown_with_no_findings():
    summary = format_summary_comment([], truncated_rules=["my-rule"])
    assert "No issues found" in summary
    assert "truncated" in summary


def test_format_summary_no_findings():
    summary = format_summary_comment([])
    assert "No issues found" in summary
