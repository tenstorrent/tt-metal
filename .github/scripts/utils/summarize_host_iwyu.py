#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Summarise a native include-what-you-use report as Markdown.

Usage: summarize_host_iwyu.py <iwyu.txt> <analyzer exit code>

Reads the concatenated iwyu_tool.py output written by run_host_iwyu.sh and
prints a step-summary table plus a histogram of the headers IWYU asked for most
often; those are where missing mapping-file entries for third-party umbrella
headers show up first. Warns on stderr if the report has content but none of it
was recognised, so a change in IWYU's output format cannot pass as a clean run.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import re
import sys

ADVICE_BLOCK = re.compile(r"^The full include-list for ", re.M)
CORRECT = re.compile(r" has correct #includes/fwd-decls\)$", re.M)
DIAGNOSTIC = re.compile(r"^.*?:\d+:\d+: (?:fatal )?error: ", re.M)
ADD_SECTION = re.compile(r"^.* should add these lines:\n((?:.*\n)*?)\n", re.M)
ADD_LINE = re.compile(r"\s*(#include\s+\S+|(?:class|struct|namespace|enum)\b.*?;)")

FORWARD_DECLARATION = "forward declaration"


@dataclass
class Summary:
    files_with_advice: int = 0
    clean: int = 0
    errors: int = 0
    additions: collections.Counter = field(default_factory=collections.Counter)

    @property
    def recognised(self) -> int:
        return self.files_with_advice + self.clean + self.errors


def parse_report(text: str) -> Summary:
    summary = Summary(
        files_with_advice=len(ADVICE_BLOCK.findall(text)),
        clean=len(CORRECT.findall(text)),
        errors=len(DIAGNOSTIC.findall(text)),
    )
    for block in ADD_SECTION.findall(text):
        for line in block.splitlines():
            match = ADD_LINE.match(line)
            if match:
                suggestion = match.group(1)
                summary.additions[suggestion if suggestion.startswith("#include") else FORWARD_DECLARATION] += 1
    return summary


def render_markdown(summary: Summary, status: int, top: int = 20) -> str:
    lines = [
        "### Include What You Use (host, report-only)",
        "",
        f"Analyzer exit code: {status} (0 means analysis completed, not that includes are clean).",
        "",
        "| Files (sources and their associated headers) | Count |",
        "| --- | --- |",
        f"| with recommendations | {summary.files_with_advice} |",
        f"| already correct | {summary.clean} |",
        f"| compile/parse errors | {summary.errors} |",
    ]
    if summary.additions:
        lines += ["", "Most-suggested additions:", "", "| Suggestion | Count |", "| --- | --- |"]
        lines += [f"| `{header}` | {count} |" for header, count in summary.additions.most_common(top)]
    lines += ["", "Full recommendations and diagnostics: `iwyu.txt` in the `iwyu-host-report` artifact."]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <iwyu.txt> <analyzer exit code>", file=sys.stderr)
        return 2
    with open(argv[1], errors="replace") as report:
        text = report.read()
    summary = parse_report(text)
    if text.strip() and summary.recognised == 0:
        print(f"::warning::{argv[1]} is not empty but contains no recognisable IWYU output.", file=sys.stderr)
    sys.stdout.write(render_markdown(summary, int(argv[2])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
