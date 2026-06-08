"""Main orchestrator — loads rules, runs LLM analysis, collects findings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .github_client import PRInfo, diff_file_paths, diff_line_numbers, post_pr_comment
from .llm import Finding, LLMSession
from .logger import logger
from .output import (
    format_pr_comment,
    format_summary_comment,
    print_failure,
    print_findings,
    write_sarif,
)
from .rules import Rule, load_rules, select_rules


class BugCheckFailed(RuntimeError):
    """Raised when bug checker analysis is incomplete and must fail closed."""


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
    rule_paths = _rule_paths(all_rules)
    matched_rules = select_rules(all_rules, pr_info.changed_files, pr_info.labels)

    if not matched_rules:
        logger.info("No rules matched this PR.")
        print_findings([])
        if post_comments:
            post_pr_comment(
                pr_number=pr_info.number,
                body="## Bug Checker\nNo rules matched the files in this PR — nothing to check.",
            )
        return []

    logger.info(f"Matched {len(matched_rules)} rule(s): " f"{', '.join(r.id for r in matched_rules)}")

    # Preflight: verify LLM is configured before entering the per-rule loop.
    # Hard config errors must fail closed; otherwise a broken setup can look like
    # a clean "no findings" result.
    try:
        LLMSession()
    except Exception as e:
        failed_rules = [rule.id for rule in matched_rules]
        logger.exception("Bug Checker failed during LLM setup")
        print_failure("Bug Checker failed during LLM setup", failed_rules)
        if sarif_path:
            write_sarif([], failed_rules, sarif_path)
            logger.info(f"SARIF output written to {sarif_path}")
        if post_comments:
            _post_findings_as_comments(pr_info, [], failed_rules, [], rule_paths)
        raise BugCheckFailed("Bug Checker failed during LLM setup") from e

    all_findings: list[Finding] = []
    rules_used: list[str] = []
    failed_rules: list[str] = []
    truncated_rules: list[str] = []
    truncated_file_set = set(pr_info.truncated_files)

    for rule in matched_rules:
        rules_used.append(rule.id)
        try:
            filtered_diff = _filter_diff_for_rule(pr_info.diff, pr_info.changed_files, rule)
            if not filtered_diff:
                matched_truncated = {f for f in pr_info.changed_files if rule.matches_pr([f], [])} & truncated_file_set
                if matched_truncated:
                    truncated_rules.append(rule.id)
                    logger.warning(
                        f"Rule {rule.id}: matched file(s) were truncated from diff — "
                        f"analysis skipped: {', '.join(sorted(matched_truncated))}"
                    )
                else:
                    logger.info(f"Rule {rule.id}: no matching diff sections — skipping LLM call")
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
            failed_rules.append(rule.id)
            logger.exception(f"Rule {rule.id} failed")

    if failed_rules:
        logger.error(
            f"{len(failed_rules)} rule(s) failed during LLM analysis: " f"{', '.join(failed_rules)}. Failing the check."
        )
    if truncated_rules:
        logger.warning(
            f"{len(truncated_rules)} rule(s) not run because matched files were truncated: "
            f"{', '.join(truncated_rules)}."
        )

    # Output: CLI
    if all_findings:
        print_findings(all_findings)
    elif failed_rules:
        print_failure("Bug Checker failed because one or more LLM analyses did not complete", failed_rules)
    else:
        print_findings([])

    # Output: SARIF
    if sarif_path:
        write_sarif(all_findings, rules_used, sarif_path)
        logger.info(f"SARIF output written to {sarif_path}")

    # Output: PR comments
    if post_comments:
        _post_findings_as_comments(pr_info, all_findings, failed_rules, truncated_rules, rule_paths)

    if failed_rules:
        raise BugCheckFailed(
            "Bug Checker failed because one or more LLM analyses did not complete: " + ", ".join(failed_rules)
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


def list_rules_command(
    pr_number: int | None = None,
    post_comments: bool = False,
) -> None:
    """List all rules from manifest with their metadata. No LLM call, no diff."""
    all_rules = load_rules()
    body = _format_list_rules(all_rules)
    logger.info(body)

    if post_comments and pr_number:
        post_pr_comment(pr_number=pr_number, body=body)


def check_rule_command(
    pr_info: PRInfo,
    rule_id: str,
    sarif_path: Optional[Path] = None,
    post_comments: bool = False,
) -> list[Finding]:
    """Run a single named rule against the PR. Error if rule not found."""
    all_rules = load_rules()
    rule_paths = _rule_paths(all_rules)
    rule = next((r for r in all_rules if r.id == rule_id), None)
    if rule is None:
        available = ", ".join(r.id for r in all_rules)
        msg = f"Rule '{rule_id}' not found. Available rules: {available}"
        logger.error(msg)
        if post_comments:
            post_pr_comment(
                pr_number=pr_info.number,
                body=f"## Bug Checker\n{msg}",
            )
        return []

    logger.info(f"Running single rule: {rule.id}")

    try:
        LLMSession()  # Preflight check
    except Exception as e:
        logger.exception("Bug Checker failed during LLM setup")
        print_failure("Bug Checker failed during LLM setup", [rule.id])
        if sarif_path:
            write_sarif([], [rule.id], sarif_path)
            logger.info(f"SARIF output written to {sarif_path}")
        if post_comments:
            _post_findings_as_comments(pr_info, [], [rule.id], [], rule_paths)
        raise BugCheckFailed("Bug Checker failed during LLM setup") from e

    filtered_diff = _filter_diff_for_rule(pr_info.diff, pr_info.changed_files, rule)
    if not filtered_diff:
        msg = f"Rule `{rule.id}` has no matching diff sections in this PR."
        logger.info(msg)
        if post_comments:
            post_pr_comment(pr_number=pr_info.number, body=f"## Bug Checker\n{msg}")
        return []

    try:
        session = LLMSession(model=rule.model or "")
        findings = session.analyze_rule(
            rule_content=rule.content,
            rule_id=rule.id,
            severity=rule.severity,
            suggest_fix=rule.suggest_fix,
            diff=filtered_diff,
        )
    except Exception as e:
        logger.exception(f"Rule {rule.id} failed")
        print_failure(f"Bug Checker failed while running rule {rule.id}", [rule.id])
        if sarif_path:
            write_sarif([], [rule.id], sarif_path)
            logger.info(f"SARIF output written to {sarif_path}")
        if post_comments:
            _post_findings_as_comments(pr_info, [], [rule.id], [], rule_paths)
        raise BugCheckFailed(f"Bug Checker failed while running rule {rule.id}") from e

    print_findings(findings)
    if sarif_path:
        write_sarif(findings, [rule.id], sarif_path)
    if post_comments:
        _post_findings_as_comments(pr_info, findings, [], [], rule_paths)

    return findings


def dry_run_command(
    pr_info: PRInfo,
    post_comments: bool = False,
) -> None:
    """Show which rules match and what diff each would see. No LLM calls."""
    all_rules = load_rules()
    matched_rules = select_rules(all_rules, pr_info.changed_files, pr_info.labels)
    body = _format_dry_run(all_rules, matched_rules, pr_info)
    logger.info(body)

    if post_comments:
        post_pr_comment(pr_number=pr_info.number, body=body)


def _format_list_rules(rules: list[Rule]) -> str:
    """Format rule list as a markdown comment."""
    lines = ["## Bug Checker — Available Rules\n"]
    if not rules:
        lines.append("No rules found in manifest.")
        return "\n".join(lines)

    lines.append(f"**{len(rules)} rule(s)** loaded from `manifest.yaml`:\n")
    lines.append("| Rule ID | Severity | Suggest Fix | Paths | Labels |")
    lines.append("|---------|----------|-------------|-------|--------|")
    for r in rules:
        paths = ", ".join(f"`{p}`" for p in r.paths) or "\u2014"
        labels = ", ".join(f"`{l}`" for l in r.labels) or "\u2014"
        fix = "Yes" if r.suggest_fix else "No"
        lines.append(f"| `{r.id}` | {r.severity} | {fix} | {paths} | {labels} |")

    return "\n".join(lines)


def _format_dry_run(
    all_rules: list[Rule],
    matched_rules: list[Rule],
    pr_info: PRInfo,
) -> str:
    """Format dry-run results as a markdown comment."""
    lines = ["## Bug Checker — Dry Run\n"]

    matched_ids = {r.id for r in matched_rules}
    unmatched = [r for r in all_rules if r.id not in matched_ids]

    lines.append(f"**PR:** #{pr_info.number} — {pr_info.title}")
    lines.append(f"**Changed files:** {len(pr_info.changed_files)}")
    lines.append(f"**Labels:** {', '.join(pr_info.labels) or '(none)'}")
    lines.append(f"**Rules matched:** {len(matched_rules)} of {len(all_rules)}\n")

    if not matched_rules:
        lines.append("No rules matched this PR.")
        return "\n".join(lines)

    for rule in matched_rules:
        reason = rule.match_reason(pr_info.changed_files, pr_info.labels)
        filtered_diff = _filter_diff_for_rule(pr_info.diff, pr_info.changed_files, rule)
        diff_files = diff_file_paths(filtered_diff) if filtered_diff else set()
        diff_line_count = len(filtered_diff.splitlines()) if filtered_diff else 0

        lines.append(f"### `{rule.id}` ({rule.severity})")
        lines.append(f"- **Match reason:** {reason}")
        lines.append(f"- **Diff sections:** {len(diff_files)} file(s), {diff_line_count} line(s)")
        if diff_files:
            for f in sorted(diff_files):
                lines.append(f"  - `{f}`")
        else:
            lines.append("  - (no diff sections — rule matched by label only or all matched files were truncated)")
        lines.append("")

    if unmatched:
        lines.append("### Unmatched Rules\n")
        for rule in unmatched:
            paths = ", ".join(f"`{p}`" for p in rule.paths) or "(none)"
            labels = ", ".join(f"`{l}`" for l in rule.labels) or "(none)"
            lines.append(f"- `{rule.id}` — paths: {paths}, labels: {labels}")

    return "\n".join(lines)


def _rule_paths(rules: list[Rule]) -> dict[str, str]:
    """Return rule-id to markdown path links for PR comments."""
    return {rule.id: f".github/bug_checker/rules/{rule.file}" for rule in rules}


def _post_findings_as_comments(
    pr_info: PRInfo,
    findings: list[Finding],
    failed_rules: list[str],
    truncated_rules: list[str],
    rule_paths: dict[str, str],
) -> None:
    """Post findings as PR comments (inline where valid, general otherwise) plus a summary."""
    valid_diff_lines = diff_line_numbers(pr_info.diff)

    inline_posted = 0
    general_posted = 0
    failed = 0

    for finding in findings or []:
        line_in_diff = finding.line in valid_diff_lines.get(finding.file, set())
        body = format_pr_comment(finding, rule_path=rule_paths.get(finding.rule_id))
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
            logger.warning(f"Failed to post comment for {finding.rule_id} at {finding.file}:{finding.line}: {e}")

    logger.info(f"Comments: {inline_posted} inline, {general_posted} general, {failed} failed")

    # Post summary comment
    try:
        summary = format_summary_comment(
            findings,
            comment_failures=failed,
            failed_rules=failed_rules,
            truncated_rules=truncated_rules,
        )
        post_pr_comment(pr_number=pr_info.number, body=summary)
    except Exception as e:
        logger.warning(f"Failed to post summary comment: {e}")
