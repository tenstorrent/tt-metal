"""Main orchestrator — loads rules, runs LLM analysis, collects findings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .github_client import PRInfo, post_pr_comment
from .llm import Finding, LLMSession
from .logger import logger
from .output import (
    format_pr_comment,
    format_summary_comment,
    print_findings,
    write_sarif,
)
from .rules import group_rules, load_rules, select_rules


def run_bug_check(
    pr_info: PRInfo,
    sarif_path: Optional[Path] = None,
    post_comments: bool = False,
) -> list[Finding]:
    """Run all matching rules against a PR and produce output.

    Args:
        pr_info: PR metadata including diff, changed files, and labels.
        sarif_path: If set, write SARIF output to this path.
        post_comments: If True, post findings as PR comments.

    Returns:
        List of all findings.
    """
    all_rules = load_rules()
    matched_rules = select_rules(all_rules, pr_info.changed_files, pr_info.labels)

    if not matched_rules:
        logger.info("No rules matched this PR.")
        print_findings([])
        return []

    logger.info(f"Matched {len(matched_rules)} rule(s): " f"{', '.join(r.id for r in matched_rules)}")

    rule_groups = group_rules(matched_rules)
    all_findings: list[Finding] = []
    rules_used: list[str] = []

    for group in rule_groups:
        try:
            session = LLMSession(model=group[0].model or "")
        except Exception:
            # Fail open: if session creation fails, skip this entire group
            rule_ids = ", ".join(r.id for r in group)
            logger.exception(f"Failed to create LLM session for rules: {rule_ids}")
            rules_used.extend(r.id for r in group)
            continue

        for rule in group:
            rules_used.append(rule.id)
            try:
                filtered_diff = _filter_diff_for_rule(pr_info.diff, pr_info.changed_files, rule)
                findings = session.analyze_rule(
                    rule_content=rule.content,
                    rule_id=rule.id,
                    severity=rule.severity,
                    suggest_fix=rule.suggest_fix,
                    diff=filtered_diff,
                )
                all_findings.extend(findings)
                logger.info(f"Rule {rule.id}: {len(findings)} finding(s)")
            except Exception:
                # Fail open: log warning with traceback, skip rule, continue
                logger.exception(f"Rule {rule.id} failed")

    # Output: CLI
    print_findings(all_findings)

    # Output: SARIF
    if sarif_path:
        write_sarif(all_findings, rules_used, sarif_path)
        logger.info(f"SARIF output written to {sarif_path}")

    # Output: PR comments
    if post_comments:
        _post_findings_as_comments(pr_info, all_findings)

    return all_findings


def _filter_diff_for_rule(diff: str, changed_files: list[str], rule) -> str:
    """Return only the diff sections for files that match this rule's path patterns."""
    matched = {f for f in changed_files if rule.matches_pr([f], [])}
    sections = re.split(r"(?=^diff --git )", diff, flags=re.MULTILINE)
    kept = []
    for section in sections:
        if not section.startswith("diff --git "):
            kept.append(section)
            continue
        m = re.match(r"^diff --git a/\S+ b/(\S+)", section)
        if m and m.group(1) in matched:
            kept.append(section)
    return "".join(kept)


def _post_findings_as_comments(pr_info: PRInfo, findings: list[Finding]) -> None:
    """Post findings as inline PR comments plus a summary comment."""
    inline_posted = 0
    inline_failed = 0
    for finding in findings or []:
        try:
            body = format_pr_comment(finding)
            post_pr_comment(
                pr_number=pr_info.number,
                body=body,
                path=finding.file,
                line=finding.line,
                commit_sha=pr_info.head_sha,
            )
            inline_posted += 1
        except Exception as e:
            inline_failed += 1
            logger.warning(f"Failed to post inline comment for {finding.rule_id} at {finding.file}:{finding.line}: {e}")

    logger.info(f"Inline comments: {inline_posted} posted, {inline_failed} failed")

    # Post summary comment
    try:
        summary = format_summary_comment(findings, inline_failed=inline_failed)
        post_pr_comment(pr_number=pr_info.number, body=summary)
    except Exception as e:
        logger.warning(f"Failed to post summary comment: {e}")
