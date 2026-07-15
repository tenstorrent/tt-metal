# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for subcommand logic (list-rules, check-rule, dry-run)."""

from unittest.mock import MagicMock, patch

from bug_checker.github_client import PRInfo
from bug_checker.orchestrator import (
    _format_dry_run,
    _format_list_rules,
    check_rule_command,
    dry_run_command,
    list_rules_command,
)
from bug_checker.rules import Rule


def _rule(
    id="test-rule", paths=None, labels=None, severity="warning", content="# Test"
):
    return Rule(
        id=id,
        file="test.md",
        severity=severity,
        suggest_fix=False,
        model=None,
        paths=paths or [],
        labels=labels or [],
        content=content,
    )


def _pr_info(changed_files=None, labels=None, diff=""):
    return PRInfo(
        number=99,
        title="Test PR",
        base_sha="aaa",
        head_sha="bbb",
        diff=diff,
        changed_files=changed_files or [],
        labels=labels or [],
    )


DIFF_ONE_FILE = """\
diff --git a/foo/bar.cpp b/foo/bar.cpp
index abc..def 100644
--- a/foo/bar.cpp
+++ b/foo/bar.cpp
@@ -1,3 +1,4 @@
+// added line
 existing line
"""


# --- _format_list_rules ---


def test_format_list_rules_renders_table():
    rules = [
        _rule(id="rule-a", paths=["src/**"], labels=["area:x"], severity="blocking"),
        _rule(id="rule-b", paths=[], labels=["area:y"], severity="warning"),
    ]
    result = _format_list_rules(rules)
    assert "Available Rules" in result
    assert "`rule-a`" in result
    assert "`rule-b`" in result
    assert "blocking" in result
    assert "| Rule ID |" in result
    assert "**2 rule(s)**" in result


def test_format_list_rules_empty():
    result = _format_list_rules([])
    assert "No rules found in manifest." in result


def test_format_list_rules_no_paths_shows_dash():
    rules = [_rule(id="orphan", paths=[], labels=[])]
    result = _format_list_rules(rules)
    assert "\u2014" in result  # em dash for missing paths/labels


# --- _format_dry_run ---


def test_format_dry_run_matched_rules():
    all_rules = [
        _rule(id="matched", paths=["foo/**"]),
        _rule(id="unmatched", paths=["bar/**"]),
    ]
    matched = [all_rules[0]]
    pr = _pr_info(changed_files=["foo/bar.cpp"], diff=DIFF_ONE_FILE)
    result = _format_dry_run(all_rules, matched, pr)
    assert "### `matched`" in result
    assert "Unmatched Rules" in result
    assert "`unmatched`" in result
    assert "1 of 2" in result


def test_format_dry_run_no_matches():
    all_rules = [_rule(id="nope", paths=["bar/**"])]
    pr = _pr_info(changed_files=["foo/x.cpp"])
    result = _format_dry_run(all_rules, [], pr)
    assert "No rules matched this PR." in result


def test_format_dry_run_shows_diff_files():
    rules = [_rule(id="r", paths=["foo/**"])]
    pr = _pr_info(changed_files=["foo/bar.cpp"], diff=DIFF_ONE_FILE)
    result = _format_dry_run(rules, rules, pr)
    assert "`foo/bar.cpp`" in result
    assert "1 file(s)" in result


# --- list_rules_command ---


@patch("bug_checker.orchestrator.load_rules")
@patch("bug_checker.orchestrator.post_pr_comment")
def test_list_rules_posts_comment(mock_post, mock_load):
    mock_load.return_value = [_rule(id="r1", paths=["x/**"])]
    list_rules_command(pr_number=42, post_comments=True)
    mock_post.assert_called_once()
    assert mock_post.call_args[1]["pr_number"] == 42
    assert "Available Rules" in mock_post.call_args[1]["body"]


@patch("bug_checker.orchestrator.load_rules")
@patch("bug_checker.orchestrator.post_pr_comment")
def test_list_rules_no_comment_without_flag(mock_post, mock_load):
    mock_load.return_value = [_rule()]
    list_rules_command(pr_number=42, post_comments=False)
    mock_post.assert_not_called()


# --- check_rule_command ---


@patch("bug_checker.orchestrator.load_rules")
@patch("bug_checker.orchestrator.post_pr_comment")
def test_check_rule_unknown_id(mock_post, mock_load):
    mock_load.return_value = [_rule(id="real-rule")]
    pr = _pr_info()
    result = check_rule_command(pr, rule_id="fake-rule", post_comments=True)
    assert result == []
    mock_post.assert_called_once()
    assert "not found" in mock_post.call_args[1]["body"]
    assert "real-rule" in mock_post.call_args[1]["body"]


@patch("bug_checker.orchestrator.load_rules")
@patch("bug_checker.orchestrator.LLMSession")
def test_check_rule_runs_named_rule(mock_llm_cls, mock_load):
    rule = _rule(id="my-rule", paths=["foo/**"], content="# Check stuff")
    mock_load.return_value = [rule]

    mock_session = MagicMock()
    mock_session.analyze_rule.return_value = []
    # First call is preflight (no args), second is the actual session
    mock_llm_cls.side_effect = [MagicMock(), mock_session]

    pr = _pr_info(changed_files=["foo/bar.cpp"], diff=DIFF_ONE_FILE)
    result = check_rule_command(pr, rule_id="my-rule")
    mock_session.analyze_rule.assert_called_once()
    assert mock_session.analyze_rule.call_args[1]["rule_id"] == "my-rule"


@patch("bug_checker.orchestrator.load_rules")
@patch("bug_checker.orchestrator.LLMSession")
@patch("bug_checker.orchestrator.post_pr_comment")
def test_check_rule_no_diff_posts_message(mock_post, mock_llm_cls, mock_load):
    rule = _rule(id="my-rule", paths=["other/**"])
    mock_load.return_value = [rule]
    mock_llm_cls.return_value = MagicMock()  # preflight

    # Empty diff means no content to analyze regardless of path matching
    pr = _pr_info(changed_files=["foo/bar.cpp"], diff="")
    result = check_rule_command(pr, rule_id="my-rule", post_comments=True)
    assert result == []
    mock_post.assert_called_once()
    assert "no matching diff sections" in mock_post.call_args[1]["body"]


# --- dry_run_command ---


@patch("bug_checker.orchestrator.load_rules")
@patch("bug_checker.orchestrator.select_rules")
@patch("bug_checker.orchestrator.post_pr_comment")
def test_dry_run_posts_comment(mock_post, mock_select, mock_load):
    rules = [_rule(id="r1", paths=["foo/**"])]
    mock_load.return_value = rules
    mock_select.return_value = rules
    pr = _pr_info(changed_files=["foo/bar.cpp"], diff=DIFF_ONE_FILE)
    dry_run_command(pr, post_comments=True)
    mock_post.assert_called_once()
    assert "Dry Run" in mock_post.call_args[1]["body"]


@patch("bug_checker.orchestrator.load_rules")
@patch("bug_checker.orchestrator.select_rules")
@patch("bug_checker.orchestrator.LLMSession")
def test_dry_run_no_llm_calls(mock_llm_cls, mock_select, mock_load):
    mock_load.return_value = []
    mock_select.return_value = []
    pr = _pr_info()
    dry_run_command(pr)
    mock_llm_cls.assert_not_called()
