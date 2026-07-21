"""Output formatters: SARIF, PR comments, CLI stdout."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

from .llm import Finding
from .logger import logger

REPO = os.environ.get("GITHUB_REPOSITORY", "tenstorrent/tt-metal")
DEFAULT_RULE_REF = "main"


# --- SARIF ---


def findings_to_sarif(findings: list[Finding], rules_used: list[str]) -> dict:
    """Convert findings to SARIF format for GitHub Code Scanning."""
    sarif_rules = []
    rule_index_map: dict[str, int] = {}
    for i, rule_id in enumerate(rules_used):
        rule_index_map[rule_id] = i
        sarif_rules.append(
            {
                "id": rule_id,
                "shortDescription": {"text": rule_id.replace("-", " ").title()},
            }
        )

    results = []
    for f in findings:
        result = {
            "ruleId": f.rule_id,
            "ruleIndex": rule_index_map.get(f.rule_id, 0),
            "level": "error" if f.severity == "blocking" else "warning",
            "message": {"text": f.message},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": f.file},
                        "region": {"startLine": max(f.line, 1)},
                    }
                }
            ],
        }
        if f.suggested_fix:
            result["fixes"] = [
                {
                    "description": {"text": "Suggested fix"},
                    "artifactChanges": [
                        {
                            "artifactLocation": {"uri": f.file},
                            "replacements": [
                                {
                                    "deletedRegion": {
                                        "startLine": max(f.line, 1),
                                        "endLine": max(f.line, 1),
                                    },
                                    "insertedContent": {"text": f.suggested_fix + "\n"},
                                }
                            ],
                        }
                    ],
                }
            ]
        results.append(result)

    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "bug-checker",
                        "version": "1.0.0",
                        "informationUri": "https://github.com/tenstorrent/tt-metal",
                        "rules": sarif_rules,
                    }
                },
                "results": results,
            }
        ],
    }


def write_sarif(findings: list[Finding], rules_used: list[str], path: Path) -> None:
    """Write SARIF output to a file."""
    sarif = findings_to_sarif(findings, rules_used)
    with open(path, "w") as f:
        json.dump(sarif, f, indent=2)


# --- CLI stdout ---


def print_findings(findings: list[Finding]) -> None:
    """Print findings in human-readable CLI format with color."""
    if not findings:
        logger.opt(colors=True).info("<green>No findings.</green>")
        return

    blocking = [f for f in findings if f.severity == "blocking"]
    warnings = [f for f in findings if f.severity == "warning"]

    for f in findings:
        if f.severity == "blocking":
            logger.opt(colors=True).error(
                "<red><bold>[BLOCKING]</bold></red> <cyan>{rule}</cyan>\n"
                "  <white>{path}:{line}</white>\n"
                "  {message}",
                rule=f.rule_id,
                path=f.file,
                line=f.line,
                message=f.message,
            )
        else:
            logger.opt(colors=True).warning(
                "<yellow><bold>[WARNING]</bold></yellow> <cyan>{rule}</cyan>\n"
                "  <white>{path}:{line}</white>\n"
                "  {message}",
                rule=f.rule_id,
                path=f.file,
                line=f.line,
                message=f.message,
            )
        if f.suggested_fix:
            logger.opt(colors=True).info("<green>  Suggested fix:</green>")
            for fix_line in f.suggested_fix.splitlines():
                logger.opt(colors=True).info("<green>    {fix_line}</green>", fix_line=fix_line)

    if blocking:
        logger.opt(colors=True).info(
            "\n<bold>Summary:</bold> <red>{b} blocking</red>, <yellow>{w} warning(s)</yellow>",
            b=len(blocking),
            w=len(warnings),
        )
    else:
        logger.opt(colors=True).info(
            "\n<bold>Summary:</bold> <green>{b} blocking</green>, <yellow>{w} warning(s)</yellow>",
            b=len(blocking),
            w=len(warnings),
        )


def print_failure(message: str, failed_rules: list[str] | None = None) -> None:
    """Print a CLI failure message without implying analysis passed."""
    logger.opt(colors=True).error("<red><bold>[FAILED]</bold></red> {message}", message=message)
    if failed_rules:
        logger.opt(colors=True).error(
            "<red>Failed rule(s):</red> {rules}",
            rules=", ".join(failed_rules),
        )


# --- PR Comments ---


def format_pr_comment(finding: Finding, rule_path: str | None = None) -> str:
    """Format a single finding as a PR comment body."""
    severity_emoji = ":red_circle:" if finding.severity == "blocking" else ":black_circle:"
    rule_label = _format_rule_label(finding.rule_id, rule_path)
    parts = [
        f"**Bug Checker [{finding.severity.upper()}]** {severity_emoji} {rule_label}\n",
        finding.message,
    ]
    if finding.suggested_fix:
        parts.append(f"\n**Suggested fix:**\n```suggestion\n{finding.suggested_fix}\n```")
    return "\n".join(parts)


def _format_rule_label(rule_id: str, rule_path: str | None = None) -> str:
    """Format a rule id, linking to its markdown definition when available."""
    if not rule_path:
        return f"`{rule_id}`"

    quoted_path = quote(rule_path, safe="/")
    quoted_ref = quote(_rule_link_ref(), safe="")
    return f"[`{rule_id}`](https://github.com/{REPO}/blob/{quoted_ref}/{quoted_path})"


def _rule_link_ref() -> str:
    """Return the Git ref used in rule definition links."""
    return (
        os.environ.get("BUG_CHECKER_RULE_REF")
        or os.environ.get("GITHUB_SHA")
        or os.environ.get("GITHUB_REF_NAME")
        or DEFAULT_RULE_REF
    )


def format_summary_comment(
    findings: list[Finding],
    comment_failures: int = 0,
    failed_rules: list[str] | None = None,
    truncated_rules: list[str] | None = None,
    skipped_rules: list[str] | None = None,
) -> str:
    """Format a summary comment for the PR."""
    if failed_rules is None:
        failed_rules = skipped_rules
    blocking = [f for f in findings if f.severity == "blocking"]
    warnings = [f for f in findings if f.severity == "warning"]

    lines = ["## Bug Checker Failed\n" if failed_rules else "## Bug Checker Results\n"]
    if failed_rules and not findings:
        lines.append("The analysis did not complete, so this run cannot be treated as a pass.")
    elif not findings:
        lines.append("No issues found.")
    else:
        if failed_rules:
            lines.append("Partial findings were produced before the checker failed. Treat this run as failed.\n")
        if blocking:
            lines.append(f"**{len(blocking)} blocking issue(s) found.**\n")
        if warnings:
            lines.append(f"{len(warnings)} warning(s) found.\n")
        for f in findings:
            tag = "BLOCKING" if f.severity == "blocking" else "WARNING"
            lines.append(f"- [{tag}] `{f.rule_id}` in `{f.file}:{f.line}`: {f.message}")

    if truncated_rules:
        rule_list = ", ".join(f"`{r}`" for r in truncated_rules)
        lines.append(
            f"\n> **Warning:** {len(truncated_rules)} rule(s) could not run because the diff was "
            f"truncated before their matched files: {rule_list}. Consider breaking this PR into smaller pieces."
        )

    if failed_rules:
        rule_list = ", ".join(f"`{r}`" for r in failed_rules)
        lines.append(
            f"\n> **Failure:** {len(failed_rules)} rule(s) failed during LLM analysis: "
            f"{rule_list}. The GitHub check exits non-zero so this is not silently accepted."
        )

    if comment_failures:
        lines.append(f"\n> **Note:** {comment_failures} comment(s) could not be posted due to API errors.")

    return "\n".join(lines)
