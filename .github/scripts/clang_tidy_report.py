#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Turn clang-tidy's exported --export-fixes YAML into a legible violation report.

clang-tidy's own diagnostic text does print during the "Analyze code with clang-tidy"
step, but that step is ~5000 lines of ninja build output and the diagnostics land
wherever their translation unit happened to finish, often right at the tail behind
thousands of "[N/M] Building/Linking ..." lines — and that step is green regardless
(it always exits 0 by design, so -k0 keeps exporting fixes for every TU). The
structured, not-buried record of what failed is the per-translation-unit YAML fix
files under .build/clang-tidy/fixes/. This script turns those into something a
human (or the GitHub UI) can read without spelunking: a plain-text list, a
Markdown table for the step summary, and ``::error`` workflow commands so
violations show up as annotations on the PR diff.

Usage:
    clang_tidy_report.py <fixes-dir> --format count
    clang_tidy_report.py <fixes-dir> --format list
    clang_tidy_report.py <fixes-dir> --format summary-md
    clang_tidy_report.py <fixes-dir> --format annotations [--max-annotations N]
    clang_tidy_report.py <fixes-dir> --format rdjson
    clang_tidy_report.py --self-test
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import sys
import unittest
from dataclasses import dataclass

import yaml


@dataclass
class Suggestion:
    """A single-span replacement, suitable for a GitHub "suggested change" block.

    Only built when a diagnostic has exactly one Replacement — GitHub suggestions
    replace one contiguous range, so multiple disjoint replacements for the same
    diagnostic can't be represented as a single suggestion without guessing how
    to merge them.
    """

    start_line: int
    start_col: int
    end_line: int
    end_col: int
    text: str


@dataclass
class Violation:
    check: str
    message: str
    file_path: str
    line: int | None
    col: int | None
    suggestion: Suggestion | None = None


def _offset_to_line_col(file_path: str, offset: int) -> tuple[int | None, int | None]:
    """Convert a byte FileOffset into a 1-based (line, col), or (None, None) if unreadable."""
    try:
        with open(file_path, "rb") as f:
            data = f.read(offset)
    except OSError:
        return None, None
    line = data.count(b"\n") + 1
    last_newline = data.rfind(b"\n")
    col = offset - last_newline
    return line, col


def _build_suggestion(file_path: str, replacements: list) -> Suggestion | None:
    """Build a Suggestion from a diagnostic's Replacements, or None if not representable.

    Only diagnostics with exactly one Replacement produce a suggestion — see the
    Suggestion docstring for why more than one can't be merged safely. Not every
    violation has a Replacement at all: compiler-diagnostic-class findings
    (clang-diagnostic-error and friends) have none, since clang can't auto-fix
    "no such member" the way it can auto-fix a stylistic clang-tidy check.

    A Replacement carries its own FilePath, which is not always the diagnostic's
    file (e.g. a fix in a macro/header pulled in by the diagnostic file). rdjson
    suggestions have no path of their own — they inherit the diagnostic's — so a
    replacement targeting a different file can't be represented as a suggestion;
    doing so anyway would convert its offset against the wrong file's bytes and
    could publish a corrupting one-click "fix".
    """
    if len(replacements) != 1 or not file_path:
        return None
    rep = replacements[0]
    if rep.get("FilePath") != file_path:
        return None
    offset, length = rep.get("Offset"), rep.get("Length")
    if not isinstance(offset, int) or not isinstance(length, int):
        return None
    start_line, start_col = _offset_to_line_col(file_path, offset)
    end_line, end_col = _offset_to_line_col(file_path, offset + length)
    if start_line is None or end_line is None:
        return None
    return Suggestion(start_line, start_col, end_line, end_col, rep.get("ReplacementText", ""))


def parse_fixes_dir(fixes_dir: str, repo_root: str | None = None) -> list[Violation]:
    """Parse every *.yaml export-fixes file in fixes_dir into a flat list of violations.

    One violation per top-level Diagnostics list entry — matches how the workflow's
    prior grep-based counter treated `- DiagnosticName:` lines, so the count doesn't
    change, only its legibility.
    """
    violations: list[Violation] = []
    for yaml_path in sorted(_glob.glob(f"{fixes_dir}/*.yaml")):
        try:
            with open(yaml_path, encoding="utf-8") as f:
                doc = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as e:
            # stderr, deliberately: --format count is captured via `$(...)` by the workflow,
            # and stdout must stay pure numeric there. A warning on stdout would get folded
            # into that captured value and corrupt the CLANG_TIDY_VIOLATIONS env var.
            print(f"::warning::clang_tidy_report: skipping unparseable {yaml_path}: {e}", file=sys.stderr)
            continue
        if not doc or not isinstance(doc.get("Diagnostics"), list):
            continue
        for diag in doc["Diagnostics"]:
            msg = diag.get("DiagnosticMessage") or {}
            file_path = msg.get("FilePath") or doc.get("MainSourceFile") or ""
            offset = msg.get("FileOffset")
            line = col = None
            if file_path and isinstance(offset, int):
                line, col = _offset_to_line_col(file_path, offset)
            display_path = file_path
            if repo_root and file_path.startswith(repo_root):
                display_path = file_path[len(repo_root) :].lstrip("/")
            message = (msg.get("Message") or "").strip().splitlines()[0] if msg.get("Message") else ""
            suggestion = _build_suggestion(file_path, msg.get("Replacements") or [])
            violations.append(
                Violation(
                    check=diag.get("DiagnosticName", "unknown-check"),
                    message=message,
                    file_path=display_path,
                    line=line,
                    col=col,
                    suggestion=suggestion,
                )
            )
    violations.sort(key=lambda v: (v.file_path, v.line or 0, v.col or 0))
    return violations


def _gh_escape_data(s: str) -> str:
    # Escaping for the free-text part (after `::`) of a workflow command.
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _gh_escape_property(s: str) -> str:
    # Escaping for `key=value` properties of a workflow command.
    return _gh_escape_data(s).replace(",", "%2C").replace(":", "%3A")


def format_list(violations: list[Violation]) -> str:
    if not violations:
        return "clang-tidy found no violations"
    lines = [f"{len(violations)} clang-tidy violation(s):", ""]
    for v in violations:
        loc = f"{v.file_path}:{v.line}:{v.col}" if v.line else v.file_path
        lines.append(f"  {loc} [{v.check}]")
        lines.append(f"    {v.message}")
    return "\n".join(lines)


def format_summary_md(violations: list[Violation]) -> str:
    if not violations:
        return "## Clang-Tidy Violations\nFound **0** violation(s)."
    lines = [
        "## Clang-Tidy Violations",
        f"Found **{len(violations)}** violation(s).",
        "",
        "| File | Line | Check | Message |",
        "|---|---|---|---|",
    ]
    for v in violations:
        message = v.message.replace("|", "\\|").replace("\n", " ")
        loc = str(v.line) if v.line else "?"
        lines.append(f"| `{v.file_path}` | {loc} | `{v.check}` | {message} |")
    return "\n".join(lines)


def format_annotations(violations: list[Violation], max_annotations: int) -> str:
    if not violations:
        return ""
    shown = violations[:max_annotations]
    lines = []
    for v in shown:
        props = [f"file={_gh_escape_property(v.file_path)}"]
        if v.line:
            props.append(f"line={v.line}")
        if v.col:
            props.append(f"col={v.col}")
        props.append(f"title={_gh_escape_property('clang-tidy: ' + v.check)}")
        lines.append(f"::error {','.join(props)}::{_gh_escape_data(v.message)}")
    remaining = len(violations) - len(shown)
    if remaining > 0:
        lines.append(f"::warning::clang_tidy_report: {remaining} additional violation(s) not shown as annotations")
    return "\n".join(lines)


def format_rdjson(violations: list[Violation]) -> str:
    """Emit reviewdog's rdjson diagnostic format, for `reviewdog -f=rdjson`.

    Unlike posting a `git diff` (post-`clang-apply-replacements`) through reviewdog's
    diff reporter, this reports every violation directly from the parsed YAML —
    including violations with no machine-applicable fix, which a fix-diff can never
    represent because there's nothing to diff. Violations with a single Replacement
    get a suggested-change block; everything else gets a plain review comment.
    """
    diagnostics = []
    for v in violations:
        location: dict = {"path": v.file_path}
        if v.line:
            location["range"] = {
                "start": {"line": v.line, "column": v.col or 1},
                "end": {"line": v.line, "column": v.col or 1},
            }
        diagnostic = {
            "message": v.message or f"clang-tidy: {v.check}",
            "location": location,
            "severity": "ERROR" if v.check == "clang-diagnostic-error" else "WARNING",
            "code": {"value": v.check},
        }
        if v.suggestion:
            diagnostic["suggestions"] = [
                {
                    "range": {
                        "start": {"line": v.suggestion.start_line, "column": v.suggestion.start_col},
                        "end": {"line": v.suggestion.end_line, "column": v.suggestion.end_col},
                    },
                    "text": v.suggestion.text,
                }
            ]
        diagnostics.append(diagnostic)
    return json.dumps({"source": {"name": "clang-tidy"}, "diagnostics": diagnostics})


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixes_dir", nargs="?", help="Directory containing clang-tidy --export-fixes *.yaml files")
    parser.add_argument("--format", choices=["count", "list", "summary-md", "annotations", "rdjson"], default="list")
    parser.add_argument("--repo-root", default=None, help="Strip this prefix from absolute file paths")
    parser.add_argument("--max-annotations", type=int, default=50)
    parser.add_argument("--self-test", action="store_true", help="Run unit tests and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if unittest.main(argv=[sys.argv[0]], exit=False).result.wasSuccessful() else 1

    if not args.fixes_dir:
        parser.error("fixes_dir is required unless --self-test is given")

    violations = parse_fixes_dir(args.fixes_dir, repo_root=args.repo_root)

    if args.format == "count":
        print(len(violations))
    elif args.format == "list":
        print(format_list(violations))
    elif args.format == "summary-md":
        print(format_summary_md(violations))
    elif args.format == "annotations":
        out = format_annotations(violations, args.max_annotations)
        if out:
            print(out)
    elif args.format == "rdjson":
        print(format_rdjson(violations))
    return 0


class TestOffsetToLineCol(unittest.TestCase):
    def test_first_line(self):
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as f:
            f.write("int main() {\n  return 0;\n}\n")
            path = f.name
        try:
            line, col = _offset_to_line_col(path, 0)
            self.assertEqual((line, col), (1, 1))
            line, col = _offset_to_line_col(path, 13)  # start of second line
            self.assertEqual((line, col), (2, 1))
        finally:
            import os

            os.unlink(path)

    def test_missing_file_returns_none(self):
        self.assertEqual(_offset_to_line_col("/no/such/file", 5), (None, None))


class TestParseFixesDir(unittest.TestCase):
    def _write(self, tmpdir, name, content):
        import os

        path = os.path.join(tmpdir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_parses_violations_and_sorts(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src = self._write(tmp, "foo.cpp", "int x = 1;\nint y = 2;\n")
            self._write(
                tmp,
                "foo.cpp-abc.yaml",
                f"""---
MainSourceFile: '{src}'
Diagnostics:
  - DiagnosticName: bugprone-integer-division
    DiagnosticMessage:
      Message: 'integer division result is silently truncated'
      FilePath: '{src}'
      FileOffset: 11
      Replacements: []
    Level: Warning
...
""",
            )
            violations = parse_fixes_dir(tmp)
            self.assertEqual(len(violations), 1)
            v = violations[0]
            self.assertEqual(v.check, "bugprone-integer-division")
            self.assertEqual(v.line, 2)
            self.assertIn("truncated", v.message)
            self.assertIsNone(v.suggestion)

    def test_single_replacement_becomes_a_suggestion(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src = self._write(tmp, "foo.cpp", "int x = 1;\nint y = 2;\n")
            self._write(
                tmp,
                "foo.cpp-abc.yaml",
                f"""---
MainSourceFile: '{src}'
Diagnostics:
  - DiagnosticName: modernize-use-nullptr
    DiagnosticMessage:
      Message: 'use nullptr'
      FilePath: '{src}'
      FileOffset: 0
      Replacements:
        - FilePath: '{src}'
          Offset: 0
          Length: 3
          ReplacementText: 'long'
    Level: Warning
...
""",
            )
            v = parse_fixes_dir(tmp)[0]
            self.assertIsNotNone(v.suggestion)
            self.assertEqual(v.suggestion.text, "long")
            self.assertEqual((v.suggestion.start_line, v.suggestion.start_col), (1, 1))
            self.assertEqual((v.suggestion.end_line, v.suggestion.end_col), (1, 4))

    def test_replacement_targeting_another_file_does_not_become_a_suggestion(self):
        # Regression (caught by Copilot review on the live demo PR): a Replacement's
        # own FilePath can differ from the diagnostic's file (e.g. a fix inside a
        # header pulled in by the diagnostic file). Converting its offset against
        # the diagnostic file's bytes would compute a bogus line/col and could
        # publish a corrupting one-click suggestion.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src = self._write(tmp, "foo.cpp", "int x = 1;\nint y = 2;\n")
            other = self._write(tmp, "foo.hpp", "int z = 3;\n")
            self._write(
                tmp,
                "foo.cpp-abc.yaml",
                f"""---
MainSourceFile: '{src}'
Diagnostics:
  - DiagnosticName: modernize-use-nullptr
    DiagnosticMessage:
      Message: 'use nullptr'
      FilePath: '{src}'
      FileOffset: 0
      Replacements:
        - FilePath: '{other}'
          Offset: 0
          Length: 3
          ReplacementText: 'long'
    Level: Warning
...
""",
            )
            v = parse_fixes_dir(tmp)[0]
            self.assertIsNone(v.suggestion)

    def test_multiple_replacements_do_not_become_a_suggestion(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src = self._write(tmp, "foo.cpp", "int x = 1;\nint y = 2;\n")
            self._write(
                tmp,
                "foo.cpp-abc.yaml",
                f"""---
MainSourceFile: '{src}'
Diagnostics:
  - DiagnosticName: readability-two-part-fix
    DiagnosticMessage:
      Message: 'needs two edits'
      FilePath: '{src}'
      FileOffset: 0
      Replacements:
        - FilePath: '{src}'
          Offset: 0
          Length: 3
          ReplacementText: 'long'
        - FilePath: '{src}'
          Offset: 11
          Length: 3
          ReplacementText: 'long'
    Level: Warning
...
""",
            )
            v = parse_fixes_dir(tmp)[0]
            self.assertIsNone(v.suggestion)

    def test_skips_unparseable_file_without_crashing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "bad.yaml", "not: valid: yaml: [")
            violations = parse_fixes_dir(tmp)
            self.assertEqual(violations, [])

    def test_count_mode_stdout_stays_numeric_despite_parse_warning(self):
        # Regression: the workflow captures --format count via `$(...)`, so a parse
        # warning must land on stderr, not stdout, or it corrupts CLANG_TIDY_VIOLATIONS.
        import contextlib
        import io
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "bad.yaml", "not: valid: yaml: [")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                main([tmp, "--format", "count"])
            self.assertEqual(out.getvalue().strip(), "0")
            self.assertIn("::warning::", err.getvalue())

    def test_no_yaml_files_is_empty(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(parse_fixes_dir(tmp), [])


class TestFormatters(unittest.TestCase):
    def test_format_annotations_escapes_and_caps(self):
        vs = [Violation(check="c", message="line1\nline2", file_path="a.cpp", line=1, col=2)]
        out = format_annotations(vs, max_annotations=50)
        self.assertIn("::error file=a.cpp,line=1,col=2,title=clang-tidy%3A c::line1%0Aline2", out)

    def test_format_annotations_truncation_note(self):
        vs = [Violation(check="c", message="m", file_path=f"{i}.cpp", line=1, col=1) for i in range(3)]
        out = format_annotations(vs, max_annotations=2)
        self.assertEqual(out.count("::error"), 2)
        self.assertIn("1 additional violation(s) not shown", out)

    def test_format_list_empty(self):
        self.assertEqual(format_list([]), "clang-tidy found no violations")

    def test_format_summary_md_escapes_pipe(self):
        vs = [Violation(check="c", message="a | b", file_path="x.cpp", line=1, col=1)]
        md = format_summary_md(vs)
        self.assertIn("a \\| b", md)

    def test_format_rdjson_reports_violation_with_no_fix(self):
        # The exact case that silently dropped comments: a clang-diagnostic-error
        # has no Replacements, so a post-fix `git diff` would be empty for it.
        vs = [
            Violation(check="clang-diagnostic-error", message="no member named 'X'", file_path="a.cpp", line=5, col=3)
        ]
        doc = json.loads(format_rdjson(vs))
        self.assertEqual(len(doc["diagnostics"]), 1)
        diag = doc["diagnostics"][0]
        self.assertEqual(diag["severity"], "ERROR")
        self.assertEqual(diag["location"]["path"], "a.cpp")
        self.assertEqual(diag["location"]["range"]["start"], {"line": 5, "column": 3})
        self.assertNotIn("suggestions", diag)

    def test_format_rdjson_includes_suggestion_when_present(self):
        suggestion = Suggestion(start_line=1, start_col=1, end_line=1, end_col=4, text="long")
        vs = [
            Violation(
                check="modernize-use-nullptr",
                message="use nullptr",
                file_path="a.cpp",
                line=1,
                col=1,
                suggestion=suggestion,
            )
        ]
        doc = json.loads(format_rdjson(vs))
        diag = doc["diagnostics"][0]
        self.assertEqual(diag["severity"], "WARNING")
        self.assertEqual(
            diag["suggestions"],
            [{"range": {"start": {"line": 1, "column": 1}, "end": {"line": 1, "column": 4}}, "text": "long"}],
        )

    def test_format_rdjson_empty_violations(self):
        self.assertEqual(json.loads(format_rdjson([]))["diagnostics"], [])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
