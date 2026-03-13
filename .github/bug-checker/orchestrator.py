"""Main orchestrator — loads rules, runs LLM analysis, collects findings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .github_client import PRInfo, fetch_file_content, post_pr_comment
from .llm import Finding, LLMSession
from .output import (
    format_pr_comment,
    format_summary_comment,
    print_findings,
    write_sarif,
)
from .rules import group_rules, load_rules, select_rules

logger = logging.getLogger(__name__)


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
        session = LLMSession(model=group[0].model or "")
        for rule in group:
            rules_used.append(rule.id)
            try:
                findings = session.analyze_rule(
                    rule_content=rule.content,
                    rule_id=rule.id,
                    severity=rule.severity,
                    suggest_fix=rule.suggest_fix,
                    diff=pr_info.diff,
                )
                all_findings.extend(findings)
                logger.info(f"Rule {rule.id}: {len(findings)} finding(s)")
            except Exception as e:
                # Fail open: log warning, skip rule, continue
                logger.warning(f"Rule {rule.id} failed: {e}")

    # Output: CLI
    print_findings(all_findings)

    # Output: SARIF
    if sarif_path:
        write_sarif(all_findings, rules_used, sarif_path)
        logger.info(f"SARIF output written to {sarif_path}")

    # Output: PR comments
    if post_comments and all_findings:
        _post_findings_as_comments(pr_info, all_findings)

    return all_findings


def _post_findings_as_comments(pr_info: PRInfo, findings: list[Finding]) -> None:
    """Post findings as inline PR comments plus a summary comment."""
    for finding in findings:
        try:
            body = format_pr_comment(finding)
            post_pr_comment(
                pr_number=pr_info.number,
                body=body,
                path=finding.file,
                line=finding.line,
                commit_sha=pr_info.head_sha,
            )
        except Exception as e:
            logger.warning(
                f"Failed to post inline comment for {finding.rule_id} " f"at {finding.file}:{finding.line}: {e}"
            )

    # Post summary comment
    try:
        summary = format_summary_comment(findings)
        post_pr_comment(pr_number=pr_info.number, body=summary)
    except Exception as e:
        logger.warning(f"Failed to post summary comment: {e}")
