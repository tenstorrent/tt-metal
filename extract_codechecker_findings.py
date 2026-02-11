#!/usr/bin/env python3
"""
Extract Clang Static Analyzer findings from CodeChecker JSON (e.g. debug3.log).
Reads source from the repo to show the exact code at each step of the bug path,
producing a justification report that explains why each finding was reported.
"""

import argparse
import json
import os
import sys

CONTEXT_LINES = 2


def resolve_path(path: str, repo_root: str, strip_prefix: str) -> str:
    """Convert report path to a path we can open under repo_root."""
    if strip_prefix and path.startswith(strip_prefix):
        path = path[len(strip_prefix) :].lstrip("/")
    return os.path.join(repo_root, path) if repo_root else path


def read_file_lines(path: str, cache: dict) -> list[str] | None:
    """Read file as lines (1-based index). Use cache to avoid re-reading."""
    if path in cache:
        return cache[path]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        cache[path] = lines
        return lines
    except OSError:
        cache[path] = None
        return None


def snippet(
    lines: list[str] | None,
    line_num: int,
    column: int | None,
    path: str,
) -> str:
    """Format a few lines around line_num with optional column caret."""
    if not lines or line_num < 1 or line_num > len(lines):
        return f"  (source not found: {path})"
    start = max(1, line_num - CONTEXT_LINES)
    end = min(len(lines), line_num + CONTEXT_LINES)
    out = []
    for i in range(start, end + 1):
        code = lines[i - 1]
        out.append(f"  {i:4d} | {code}")
        if i == line_num and column is not None and column >= 1:
            col_idx = min(column - 1, len(code))
            spaces = " " * (6 + col_idx)
            out.append(f"       {spaces}^")
    return "\n".join(out)


def is_project_file(path: str) -> bool:
    """Skip system / third-party paths for bug path display."""
    return "/usr/" not in path and "include/" not in path


def get_path_from_file(file_obj: dict) -> str:
    """Extract path string from report file or event file object."""
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("original_path") or file_obj.get("id") or ""
    return str(file_obj)


def build_justification(
    report: dict,
    repo_root: str,
    strip_prefix: str,
    file_cache: dict,
    finding_id: int,
) -> str:
    """Build one finding's justification: bug path + code at each step."""
    r = report
    main_file = r.get("file", {})
    main_path = get_path_from_file(main_file)
    main_line = r.get("line")
    main_col = r.get("column")
    checker = r.get("checker_name", "?")
    message = r.get("message", "?")
    severity = r.get("severity", "?")
    report_hash = r.get("report_hash", "")

    resolved_main = resolve_path(main_path, repo_root, strip_prefix)
    lines_main = read_file_lines(resolved_main, file_cache)

    out = []
    out.append("=" * 72)
    out.append(f"Finding #{finding_id}")
    out.append("=" * 72)
    out.append(f"File:    {main_path}")
    out.append(f"Line:    {main_line}" + (f" (column {main_col})" if main_col is not None else ""))
    out.append(f"Checker: {checker}")
    out.append(f"Severity: {severity}")
    out.append(f"Report hash: {report_hash}")
    out.append("")
    out.append("Summary:")
    out.append(f"  {message}")
    out.append("")
    out.append("Why this was reported (bug path with code):")
    out.append("-" * 72)

    events = r.get("bug_path_events", [])
    seen = set()
    step_num = 0

    for event in events:
        file_obj = event.get("file", {})
        path = get_path_from_file(file_obj)
        if not path or not is_project_file(path):
            continue
        line = event.get("line")
        col = event.get("column")
        msg = event.get("message", "")
        key = (path, line)
        if key in seen:
            continue
        seen.add(key)
        step_num += 1

        resolved = resolve_path(path, repo_root, strip_prefix)
        lines = read_file_lines(resolved, file_cache)

        out.append("")
        out.append(f"  Step {step_num}: {path}:{line}:{col}")
        out.append(f"  → {msg}")
        out.append(snippet(lines, line or 0, col, resolved))

    # Ensure the final flagged line is shown if not already in path
    key_main = (main_path, main_line)
    if (main_line is not None) and key_main not in seen:
        out.append("")
        out.append(f"  Flagged line: {main_path}:{main_line}:{main_col}")
        out.append(f"  → {message}")
        out.append(snippet(lines_main, main_line, main_col, resolved_main))

    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(
        description="Extract CodeChecker JSON findings and produce a justification report with code."
    )
    ap.add_argument(
        "input",
        nargs="?",
        default="debug3.log",
        help="Path to CodeChecker JSON file (default: debug3.log)",
    )
    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file (default: clangsa_justification.txt; use - for stdout)",
    )
    ap.add_argument(
        "--repo-root",
        default=".",
        help="Repository root to resolve source paths (default: current directory)",
    )
    ap.add_argument(
        "--strip-prefix",
        default="/work/",
        help="Prefix to strip from report paths (default: /work/)",
    )
    ap.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Limit number of findings to process (default: all)",
    )
    ap.add_argument(
        "--analyzer",
        default="clangsa",
        help="Filter by analyzer name (default: clangsa)",
    )
    args = ap.parse_args()

    if args.output == "-":
        args.output = None
    elif args.output is None:
        args.output = "clangsa_justification.txt"

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    reports = data.get("reports", [])
    if not reports:
        print("No reports in JSON.", file=sys.stderr)
        sys.exit(0)

    filtered = [r for r in reports if r.get("analyzer_name") == args.analyzer]
    if not filtered:
        print(f"No reports with analyzer_name={args.analyzer!r}.", file=sys.stderr)
        sys.exit(0)

    to_process = filtered[: args.limit] if args.limit else filtered
    file_cache = {}

    sections = []
    sections.append(f"Clang Static Analyzer justification report")
    sections.append(f"Input: {args.input}")
    sections.append(f"Findings: {len(to_process)} (analyzer={args.analyzer})")
    if args.limit:
        sections.append(f"(limited to first {args.limit})")
    sections.append("")

    for i, report in enumerate(to_process, 1):
        sections.append(
            build_justification(
                report,
                args.repo_root,
                args.strip_prefix,
                file_cache,
                i,
            )
        )

    sections.append("=" * 72)
    sections.append(f"Total: {len(to_process)} finding(s)")
    sections.append("=" * 72)

    text = "\n".join(sections)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {len(to_process)} finding(s) to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
