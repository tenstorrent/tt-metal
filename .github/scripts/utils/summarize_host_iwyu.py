#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Summarise a native include-what-you-use report as Markdown.

Usage: summarize_host_iwyu.py <iwyu.txt> <analyzer exit code> [<title> <artifact name>]
       summarize_host_iwyu.py --rewrite-c-headers <iwyu.txt>   (rewrites the report in place)

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


# IWYU 0.24 (the release matching the image's LLVM 20) recommends the C spelling
# of the C compatibility headers for symbols such as size_t or uint32_t. Applying
# that verbatim trips clang-tidy's modernize-deprecated-headers, and IWYU's own
# mapping file cannot override it. IWYU 0.27 recommends the <c*> headers in C++
# mode (include-what-you-use#1126); drop this once the image runs IWYU >= 0.27.
# The set is the one modernize-deprecated-headers enforces.
C_COMPAT_HEADERS = (
    "assert ctype errno fenv float inttypes limits locale math setjmp signal "
    "stdarg stddef stdint stdio stdlib string time uchar wchar wctype"
).split()
# Only recommendation lines, which start in column 0 ("- #include ..." removals
# name a line that exists in the source and must stay verbatim).
C_COMPAT_INCLUDE = re.compile(r"^#include <(%s)\.h>( *)" % "|".join(C_COMPAT_HEADERS), re.M)


def rewrite_c_headers(text: str) -> str:
    def cxx(match: re.Match) -> str:
        old, new = f"<{match.group(1)}.h>", f"<c{match.group(1)}>"
        # Keep the "// for ..." comments aligned.
        return f"#include {new}" + " " * max(len(match.group(2)) + len(old) - len(new), 1)

    return C_COMPAT_INCLUDE.sub(cxx, text)


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


def render_markdown(
    summary: Summary,
    status: int,
    top: int = 20,
    title: str = "Include What You Use (host, report-only)",
    artifact: str = "iwyu-host-report",
) -> str:
    lines = [
        f"### {title}",
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
    lines += ["", f"Full recommendations and diagnostics: `iwyu.txt` in the `{artifact}` artifact."]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--rewrite-c-headers":
        with open(argv[2], errors="replace") as report:
            text = report.read()
        with open(argv[2], "w") as report:
            report.write(rewrite_c_headers(text))
        return 0
    if len(argv) not in (3, 5):
        print(f"usage: {argv[0]} <iwyu.txt> <analyzer exit code> [<title> <artifact name>]", file=sys.stderr)
        return 2
    with open(argv[1], errors="replace") as report:
        text = report.read()
    summary = parse_report(text)
    if text.strip() and summary.recognised == 0:
        print(f"::warning::{argv[1]} is not empty but contains no recognisable IWYU output.", file=sys.stderr)
    sys.stdout.write(render_markdown(summary, int(argv[2]), *([] if len(argv) == 3 else [20, argv[3], argv[4]])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
