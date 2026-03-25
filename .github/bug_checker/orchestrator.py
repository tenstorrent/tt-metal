"""Main orchestrator — loads rules, runs LLM analysis, collects findings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .github_client import PRInfo, diff_line_numbers, post_pr_comment
from .llm import Finding, LLMSession
from .logger import logger
from .output import (
    format_pr_comment,
    format_summary_comment,
    print_findings,
    write_sarif,
)
from .rules import load_rules, select_rules


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

    logger.info(
        f"Matched {len(matched_rules)} rule(s): "
        f"{', '.join(r.id for r in matched_rules)}"
    )

    all_findings: list[Finding] = []
    rules_used: list[str] = []
    skipped_rules: list[str] = []
    truncated_rules: list[str] = []
    truncated_file_set = set(pr_info.truncated_files)

    for rule in matched_rules:
        rules_used.append(rule.id)
        try:
            filtered_diff = _filter_diff_for_rule(
                pr_info.diff, pr_info.changed_files, rule
            )
            if not filtered_diff:
                matched_truncated = {
                    f for f in pr_info.changed_files if rule.matches_pr([f], [])
                } & truncated_file_set
                if matched_truncated:
                    truncated_rules.append(rule.id)
                    logger.warning(
                        f"Rule {rule.id}: matched file(s) were truncated from diff — "
                        f"analysis skipped: {', '.join(sorted(matched_truncated))}"
                    )
                else:
                    logger.info(
                        f"Rule {rule.id}: no matching diff sections — skipping LLM call"
                    )
                continue
            session = LLMSession(model=rule.model or "")
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
            skipped_rules.append(rule.id)
            logger.exception(f"Rule {rule.id} failed — skipping")

    if skipped_rules:
        logger.warning(
            f"{len(skipped_rules)} rule(s) skipped due to errors: "
            f"{', '.join(skipped_rules)}. Results may be incomplete."
        )
    if truncated_rules:
        logger.warning(
            f"{len(truncated_rules)} rule(s) not run because matched files were truncated: "
            f"{', '.join(truncated_rules)}."
        )

    # Output: CLI
    print_findings(all_findings)

    # Output: SARIF
    if sarif_path:
        write_sarif(all_findings, rules_used, sarif_path)
        logger.info(f"SARIF output written to {sarif_path}")

    # Output: PR comments
    if post_comments:
        _post_findings_as_comments(
            pr_info, all_findings, skipped_rules, truncated_rules
        )

    return all_findings


def _filter_diff_for_rule(diff: str, changed_files: list[str], rule) -> str:
    """Return only the diff sections for files matching this rule's path patterns.

    Parses the diff line by line, collecting each per-file section and keeping
    only those whose path matches one of the rule's glob patterns. Returns an
    empty string if no sections match.
    """
    matched = {f for f in changed_files if rule.matches_pr([f], [])}
    if not matched:
        # Rule was selected by label only — no path filter, use full diff
        return diff

    kept: list[str] = []
    current_lines: list[str] = []
    current_file: str | None = None

    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            # Flush the completed section if its file was matched
            if current_file in matched:
                kept.extend(current_lines)
            m = re.match(r"^diff --git a/\S+ b/(\S+)", line)
            current_file = m.group(1) if m else None
            current_lines = [line]
        else:
            current_lines.append(line)

    # Flush the final section
    if current_file in matched:
        kept.extend(current_lines)

    return "".join(kept)


def _post_findings_as_comments(
    pr_info: PRInfo,
    findings: list[Finding],
    skipped_rules: list[str],
    truncated_rules: list[str],
) -> None:
    """Post findings as PR comments (inline where valid, general otherwise) plus a summary."""
    valid_diff_lines = diff_line_numbers(pr_info.diff)

    inline_posted = 0
    general_posted = 0
    failed = 0

    for finding in findings or []:
        line_in_diff = finding.line in valid_diff_lines.get(finding.file, set())
        body = format_pr_comment(finding)
        try:
            if line_in_diff:
                post_pr_comment(
                    pr_number=pr_info.number,
                    body=body,
                    path=finding.file,
                    line=finding.line,
                    commit_sha=pr_info.head_sha,
                )
                inline_posted += 1
            else:
                logger.debug(
                    f"Rule {finding.rule_id}: line {finding.line} in {finding.file!r} "
                    "is not in the diff — posting as general comment"
                )
                post_pr_comment(pr_number=pr_info.number, body=body)
                general_posted += 1
        except Exception as e:
            failed += 1
            logger.warning(
                f"Failed to post comment for {finding.rule_id} at {finding.file}:{finding.line}: {e}"
            )

    logger.info(
        f"Comments: {inline_posted} inline, {general_posted} general, {failed} failed"
    )

    # Post summary comment
    try:
        summary = format_summary_comment(
            findings,
            comment_failures=failed,
            skipped_rules=skipped_rules,
            truncated_rules=truncated_rules,
        )
        post_pr_comment(pr_number=pr_info.number, body=summary)
    except Exception as e:
        logger.warning(f"Failed to post summary comment: {e}")
