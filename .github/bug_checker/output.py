"""Output formatters: SARIF, PR comments, CLI stdout."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TextIO

from .llm import Finding
from .logger import logger


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
                                    "deletedRegion": {"startLine": max(f.line, 1)},
                                    "insertedContent": {"text": f.suggested_fix},
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


def print_findings(findings: list[Finding], file: TextIO = sys.stdout) -> None:
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


# --- PR Comments ---


def format_pr_comment(finding: Finding) -> str:
    """Format a single finding as a PR comment body."""
    severity_emoji = "!!!" if finding.severity == "blocking" else "?"
    parts = [
        f"{severity_emoji} **Bug Checker [{finding.severity.upper()}]** `{finding.rule_id}`\n",
        finding.message,
    ]
    if finding.suggested_fix:
        parts.append(f"\n**Suggested fix:**\n```suggestion\n{finding.suggested_fix}\n```")
    return "\n".join(parts)


def format_summary_comment(findings: list[Finding], inline_failed: int = 0) -> str:
    """Format a summary comment for the PR."""
    blocking = [f for f in findings if f.severity == "blocking"]
    warnings = [f for f in findings if f.severity == "warning"]

    lines = ["## Bug Checker Results\n"]
    if not findings:
        lines.append("No issues found.")
        return "\n".join(lines)

    if blocking:
        lines.append(f"**{len(blocking)} blocking issue(s) found.**\n")
    if warnings:
        lines.append(f"{len(warnings)} warning(s) found.\n")

    for f in findings:
        tag = "BLOCKING" if f.severity == "blocking" else "WARNING"
        lines.append(f"- [{tag}] `{f.rule_id}` in `{f.file}:{f.line}`: {f.message}")

    if inline_failed:
        lines.append(
            f"\n> **Note:** {inline_failed} inline comment(s) could not be posted "
            "(the line may no longer exist in the latest commit). See the list above for all findings."
        )

    return "\n".join(lines)
