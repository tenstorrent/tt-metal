#!/usr/bin/env python3
"""Extract static analyzer findings from Clang HTML report (e.g. debug.log)."""

import re
import json
import sys

CONTEXT_LINES = 2  # lines of context before/after the flagged line


def unescape_source(text: str) -> str:
    """Unescape HTML entities in embedded report source."""
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("&quot;", '"')


def extract_data(html_path: str) -> dict:
    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    match = re.search(r"var data = (\{.*?\});\s*window\.onload", content, re.DOTALL)
    if not match:
        raise SystemExit("Could not find 'var data = {...}' in file")
    raw = match.group(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raw = raw.replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">")
        raw = raw.replace("&amp;", "&")
        data = json.loads(raw)
    return data


def get_file_lines(files: dict, file_id: str) -> list[str] | None:
    """Return list of source lines (1-based index) for file_id, or None if missing."""
    info = files.get(file_id)
    if not info or "content" not in info:
        return None
    return unescape_source(info["content"]).splitlines()


def snippet(lines: list[str] | None, line_num: int, column: int | None) -> str:
    """Format a few lines around line_num, with optional column caret."""
    if not lines or line_num < 1 or line_num > len(lines):
        return "(source not in report)"
    start = max(1, line_num - CONTEXT_LINES)
    end = min(len(lines), line_num + CONTEXT_LINES)
    out = []
    for i in range(start, end + 1):
        code = lines[i - 1]
        out.append(f"  {i:4d} | {code}")
        if i == line_num and column is not None and column >= 1:
            # caret under the column (0-based column in string)
            col_idx = min(column - 1, len(code))
            spaces = " " * (6 + col_idx)
            out.append(f"       {spaces}^")
    return "\n".join(out)


def is_project_file(file_id: str) -> bool:
    return "/usr/" not in file_id and "include/" not in file_id


def build_finding(files: dict, report: dict, finding_id: int) -> dict:
    """Build a structured finding dict (for JSON or summary)."""
    r = report
    file_id = r.get("fileId", r.get("path", "?"))
    line = r.get("line")
    col = r.get("column")
    checker = r.get("checker", {}) if isinstance(r.get("checker"), dict) else {}
    name = checker.get("name", "?")
    url = checker.get("url", "")
    msg = r.get("message", "?")
    severity = r.get("severity", "?")

    bug_path = []
    seen = set()
    for e in r.get("events", []):
        fid = e.get("fileId", "")
        ln = e.get("line")
        if not ln or not is_project_file(fid):
            continue
        key = (fid, ln)
        if key in seen:
            continue
        seen.add(key)
        file_lines = get_file_lines(files, fid)
        bug_path.append(
            {
                "file": fid,
                "line": ln,
                "column": e.get("column"),
                "message": e.get("message", ""),
                "code_snippet": snippet(file_lines, ln, e.get("column")),
            }
        )
    file_lines = get_file_lines(files, file_id)
    flagged_snippet = snippet(file_lines, line or 0, col) if line else ""

    return {
        "id": finding_id,
        "file": file_id,
        "line": line,
        "column": col,
        "checker": name,
        "checker_url": url,
        "severity": severity,
        "message": msg,
        "bug_path": bug_path,
        "flagged_line_snippet": flagged_snippet,
    }


def main():
    argv = sys.argv[1:]
    json_out = "--json" in argv
    if json_out:
        argv = [a for a in argv if a != "--json"]
    path = argv[0] if argv else "debug.log"
    data = extract_data(path)
    files = data.get("files", {})
    reports = data.get("reports", [])

    findings = [build_finding(files, r, i) for i, r in enumerate(reports, 1)]

    if json_out:
        print(json.dumps({"findings": findings, "total": len(findings)}, indent=2))
        return

    # Summary for LLM / human
    files_summary = {}
    for f in findings:
        key = f["file"]
        if key not in files_summary:
            files_summary[key] = []
        files_summary[key].append(f["line"])
    summary_lines = [f"  {path}: lines {sorted(lines)}" for path, lines in sorted(files_summary.items())]
    print(f"Summary: {len(findings)} finding(s) in {len(files_summary)} file(s)")
    print("\n".join(summary_lines))
    print()

    for f in findings:
        print(f"--- Finding #{f['id']} ---")
        print(f"File:   {f['file']}")
        print(f"Line:   {f['line']}" + (f" (column {f['column']})" if f.get("column") is not None else ""))
        print(f"Checker: {f['checker']}")
        if f.get("checker_url"):
            print(f"Checker URL: {f['checker_url']}")
        print(f"Severity: {f['severity']}")
        print(f"Message: {f['message']}")

        if f["bug_path"]:
            print("Reasoning (bug path) with code:")
            for step in f["bug_path"]:
                loc = f"{step['file']}:{step['line']}:{step.get('column', '')}"
                print(f"  • {loc} — {step['message']}")
                print(step["code_snippet"])
                print()
            # Show flagged line if not already in path
            if not any((s["file"], s["line"]) == (f["file"], f["line"]) for s in f["bug_path"]):
                print(f"  >>> Flagged line: {f['file']}:{f['line']}:{f['column']}")
                print(f["flagged_line_snippet"])
        else:
            print("Code at flagged location:")
            print(f["flagged_line_snippet"])
        print()

    print(f"Total: {len(findings)} report(s)")


if __name__ == "__main__":
    main()
