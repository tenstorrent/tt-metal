#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for run_api_header_include_cleaner.py against real clang-tidy 20.1.8 output.

Run: python3 .github/scripts/utils/test_run_api_header_include_cleaner.py
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_api_header_include_cleaner import (  # noqa: E402
    COVERED,
    Result,
    analyze,
    apply_edits,
    exit_status,
    parse_export_fixes,
    repo_spelling,
    select,
    unquote,
)

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_api_header_include_cleaner.py")
ROOT = subprocess.run(
    ["git", "-C", os.path.dirname(SCRIPT), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
).stdout.strip()

# The header analyzed by clang-tidy 20.1.8 to produce EXPORT_WITH_ERROR below.
HEADER = (
    b"#pragma once\n"  # 0
    b"#include <algorithm>\n"  # 13
    b"#include <vector>\n"  # 34
    b"inline std::size_t g(){ return 1; }\n"  # 52
    b"int h() { return undeclared_thing; }\n"
)

# Verbatim --export-fixes output for HEADER (paths shortened): two unused
# includes, one missing include whose insertion shares offset 34 with the
# removal of <vector>, and a compile error. clang-tidy exited with 1.
EXPORT_WITH_ERROR = """\
---
MainSourceFile:  '/r/findings_and_error.hpp'
Diagnostics:
  - DiagnosticName:  misc-include-cleaner
    DiagnosticMessage:
      Message:         included header algorithm is not used directly
      FilePath:        '/r/findings_and_error.hpp'
      FileOffset:      13
      Replacements:
        - FilePath:        '/r/findings_and_error.hpp'
          Offset:          13
          Length:          21
          ReplacementText: ''
    Level:           Warning
    BuildDirectory:  '/b'
  - DiagnosticName:  misc-include-cleaner
    DiagnosticMessage:
      Message:         included header vector is not used directly
      FilePath:        '/r/findings_and_error.hpp'
      FileOffset:      34
      Replacements:
        - FilePath:        '/r/findings_and_error.hpp'
          Offset:          34
          Length:          18
          ReplacementText: ''
    Level:           Warning
    BuildDirectory:  '/b'
  - DiagnosticName:  misc-include-cleaner
    DiagnosticMessage:
      Message:         'no header providing "std::size_t" is directly included'
      FilePath:        '/r/findings_and_error.hpp'
      FileOffset:      64
      Replacements:
        - FilePath:        '/r/findings_and_error.hpp'
          Offset:          34
          Length:          0
          ReplacementText: "#include <cstddef>\\n"
    Level:           Warning
    BuildDirectory:  '/b'
  - DiagnosticName:  clang-diagnostic-error
    DiagnosticMessage:
      Message:         'use of undeclared identifier ''undeclared_thing'''
      FilePath:        '/r/findings_and_error.hpp'
      FileOffset:      105
      Replacements:    []
    Level:           Error
    BuildDirectory:  '/b'
...
"""
OUTPUT_WITH_ERROR = (
    "3 warnings and 1 error generated.\n"
    "Error while processing /r/findings_and_error.hpp.\n"
    "/r/findings_and_error.hpp:5:18: error: use of undeclared identifier 'undeclared_thing' [clang-diagnostic-error]\n"
)

# The same header without the compile error: findings only, clang-tidy exits 0.
EXPORT_FINDINGS = EXPORT_WITH_ERROR[: EXPORT_WITH_ERROR.index("  - DiagnosticName:  clang-diagnostic-error")] + "...\n"

# Project headers come back quoted, relative to the include directory they were found through.
EXPORT_PROJECT_HEADER = """\
Diagnostics:
  - DiagnosticName:  misc-include-cleaner
    DiagnosticMessage:
      Message:         'no header providing "ttsl::Span" is directly included'
      FileOffset:      900
      Replacements:
        - FilePath:        '/r/mesh_coord.hpp'
          Offset:          133
          Length:          0
          ReplacementText: "#include \\"tt_stl/span.hpp\\"\\n"
    Level:           Warning
  - DiagnosticName:  misc-include-cleaner
    DiagnosticMessage:
      Message:         'no header providing "ttsl::Span" is directly included'
      FileOffset:      950
      Replacements:    []
    Level:           Warning
"""


class ParseExportFixes(unittest.TestCase):
    def test_unquote(self):
        self.assertEqual(unquote("  plain text "), "plain text")
        self.assertEqual(unquote("''"), "")
        self.assertEqual(unquote("'it''s: quoted'"), "it's: quoted")
        self.assertEqual(unquote('"#include <cstddef>\\n"'), "#include <cstddef>\n")
        self.assertEqual(unquote('"#include \\"tt_stl/span.hpp\\"\\n"'), '#include "tt_stl/span.hpp"\n')

    def test_diagnostics(self):
        diagnostics = parse_export_fixes(EXPORT_WITH_ERROR)
        self.assertEqual(
            [(d["name"], d["level"], d["offset"]) for d in diagnostics],
            [
                ("misc-include-cleaner", "Warning", 13),
                ("misc-include-cleaner", "Warning", 34),
                ("misc-include-cleaner", "Warning", 64),
                ("clang-diagnostic-error", "Error", 105),
            ],
        )
        self.assertEqual(diagnostics[2]["message"], 'no header providing "std::size_t" is directly included')
        self.assertEqual(diagnostics[2]["edits"], [[34, 0, "#include <cstddef>\n"]])
        self.assertEqual(diagnostics[0]["edits"], [[13, 21, ""]])
        self.assertEqual(diagnostics[3]["message"], "use of undeclared identifier 'undeclared_thing'")
        self.assertEqual(diagnostics[3].get("edits", []), [])

    def test_empty(self):
        self.assertEqual(parse_export_fixes(""), [])


class Analyze(unittest.TestCase):
    def test_findings(self):
        result = analyze("x.hpp", 0, "", EXPORT_FINDINGS, HEADER, ROOT)
        self.assertEqual(result.missing, {"#include <cstddef>": ["std::size_t"]})
        self.assertEqual(result.unused, [(2, "#include <algorithm>"), (3, "#include <vector>")])
        self.assertEqual(result.errors, [])
        self.assertEqual(sorted(result.edits), [(13, 21, ""), (34, 0, "#include <cstddef>\n"), (34, 18, "")])

    def test_clean(self):
        result = analyze("x.hpp", 0, "", "", HEADER, ROOT)
        self.assertFalse(result.missing or result.unused or result.failed)

    def test_compile_error_is_a_failure_even_with_findings(self):
        result = analyze("x.hpp", 1, OUTPUT_WITH_ERROR, EXPORT_WITH_ERROR, HEADER, ROOT)
        self.assertTrue(result.failed)
        self.assertIn("clang-diagnostic-error: use of undeclared identifier 'undeclared_thing'", result.errors)
        self.assertEqual(len(result.unused), 2)

    def test_unexplained_nonzero_exit_is_a_failure_even_with_findings(self):
        # A crash after a partial report: findings must not hide the exit status.
        result = analyze("x.hpp", 139, "", EXPORT_FINDINGS, HEADER, ROOT)
        self.assertEqual(result.errors, ["clang-tidy exited with 139 without reporting an error"])
        result = analyze("x.hpp", 1, "", "", HEADER, ROOT)
        self.assertEqual(result.errors, ["clang-tidy exited with 1 without reporting an error"])

    def test_signal_is_a_failure(self):
        result = analyze("x.hpp", -11, "", EXPORT_FINDINGS, HEADER, ROOT)
        self.assertEqual(result.errors, ["clang-tidy was killed by signal 11"])

    def test_findings_promoted_to_errors_explain_the_exit(self):
        promoted = EXPORT_FINDINGS.replace("Level:           Warning", "Level:           Error")
        result = analyze("x.hpp", 1, "", promoted, HEADER, ROOT)
        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.unused), 2)

    def test_unrecognised_message_is_a_failure(self):
        changed = EXPORT_FINDINGS.replace("is not used directly", "is unused")
        result = analyze("x.hpp", 0, "", changed, HEADER, ROOT)
        self.assertEqual(len(result.errors), 2)
        self.assertTrue(result.errors[0].startswith("misc-include-cleaner: unrecognised message"))

    def test_project_header_spelling_and_covered_symbols(self):
        result = analyze("x.hpp", 0, "", EXPORT_PROJECT_HEADER, b"", ROOT)
        self.assertEqual(result.missing, {"#include <tt_stl/span.hpp>": ["ttsl::Span"], COVERED: ["ttsl::Span"]})

    def test_repo_spelling(self):
        self.assertEqual(repo_spelling('#include "core_coord.hpp"\n', ROOT), "#include <tt-metalium/core_coord.hpp>\n")
        self.assertEqual(repo_spelling('#include "tt_stl/span.hpp"', ROOT), "#include <tt_stl/span.hpp>")
        self.assertEqual(repo_spelling('#include "no/such/header.hpp"', ROOT), '#include "no/such/header.hpp"')
        self.assertEqual(repo_spelling("#include <cstddef>", ROOT), "#include <cstddef>")


class ApplyEdits(unittest.TestCase):
    def apply(self, content: bytes, edits) -> bytes:
        with tempfile.NamedTemporaryFile("wb", suffix=".hpp", delete=False) as f:
            f.write(content)
        try:
            apply_edits(f.name, edits, ROOT)
            with open(f.name, "rb") as fixed:
                return fixed.read()
        finally:
            os.unlink(f.name)

    def test_real_fixits_in_any_order(self):
        edits = analyze("x.hpp", 0, "", EXPORT_FINDINGS, HEADER, ROOT).edits
        expected = b"#pragma once\n#include <cstddef>\ninline std::size_t g(){ return 1; }\nint h() { return undeclared_thing; }\n"
        self.assertEqual(self.apply(HEADER, edits), expected)
        self.assertEqual(self.apply(HEADER, list(reversed(edits))), expected)

    def test_insertions_at_one_offset_come_out_sorted_and_respelled(self):
        content = b"#pragma once\n#include <vector>\n"
        edits = [
            (13, 0, '#include "tt_stl/span.hpp"\n'),
            (13, 0, "#include <cstdint>\n"),
            (13, 0, "#include <cstdint>\n"),
        ]
        self.assertEqual(
            self.apply(content, edits),
            b"#pragma once\n#include <cstdint>\n#include <tt_stl/span.hpp>\n#include <vector>\n",
        )


class Select(unittest.TestCase):
    def setUp(self):
        self.build = tempfile.TemporaryDirectory()
        stub_dir = os.path.join(self.build.name, "tt_metal", "tt_metal_verify_interface_header_sets", "tt-metalium")
        os.makedirs(stub_dir)
        self.stub = os.path.join(stub_dir, "core_coord.hpp.cxx")
        open(self.stub, "w").close()
        entries = [
            {
                "directory": self.build.name,
                "command": f"/usr/bin/clang++-20 -DX=1 -I{ROOT}/tt_metal/api -x c++ -std=c++20 -o stub.o -c {self.stub}",
                "file": self.stub,
            },
            {"directory": self.build.name, "command": "/usr/bin/clang++-20 -c other.cpp", "file": "other.cpp"},
        ]
        with open(os.path.join(self.build.name, "compile_commands.json"), "w") as db:
            json.dump(entries, db)

    def tearDown(self):
        self.build.cleanup()

    def test_stub_entry_becomes_a_header_entry(self):
        header = os.path.join(ROOT, "tt_metal/api/tt-metalium/core_coord.hpp")
        for request in (["--all"], ["tt_metal/api/tt-metalium/core_coord.hpp"], [header]):
            (entry,), no_stub, _ = select(self.build.name, ROOT, request)
            self.assertEqual(no_stub, [])
            self.assertEqual(entry["file"], header)
            self.assertEqual(
                entry["arguments"],
                ["/usr/bin/clang++-20", "-DX=1", f"-I{ROOT}/tt_metal/api", "-x", "c++-header", "-std=c++20", header],
            )

    def test_requested_headers_without_a_stub_are_returned(self):
        requested = ["tt_metal/api/tt-metalium/no_such.hpp", "README.md"]
        self.assertEqual(select(self.build.name, ROOT, requested), ([], requested, []))

    def test_all_lists_headers_outside_the_public_header_set(self):
        entries, no_stub, outside_set = select(self.build.name, ROOT, ["--all"])
        self.assertEqual((len(entries), no_stub), (1, []))
        self.assertIn("tt_metal/api/tt-metalium/mesh_coord.hpp", outside_set)
        self.assertNotIn("tt_metal/api/tt-metalium/core_coord.hpp", outside_set)


class GateMode(unittest.TestCase):
    def test_exit_status(self):
        clean, finding, failed = Result("a"), Result("b", unused=[(1, "#include <x>")]), Result("c", errors=["e"])
        self.assertEqual(exit_status([clean, finding, failed], gating=False), 0)
        self.assertEqual(exit_status([clean], gating=True), 0)
        self.assertEqual(exit_status([clean, finding], gating=True), 1)
        self.assertEqual(exit_status([clean, failed], gating=True), 1)

    def run_script(self, export: str, returncode: int, *flags: str) -> subprocess.CompletedProcess:
        """The real script against a fake clang-tidy that writes a canned --export-fixes file."""
        # Not under /tmp: CI containers mount it as tmpfs, which Docker makes noexec,
        # so the fake clang-tidy could not run there.
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".test-include-cleaner-") as build:
            stub_dir = os.path.join(build, "tt_metal", "tt_metal_verify_interface_header_sets", "tt-metalium")
            os.makedirs(stub_dir)
            stub = os.path.join(stub_dir, "core_coord.hpp.cxx")
            open(stub, "w").close()
            with open(os.path.join(build, "compile_commands.json"), "w") as db:
                json.dump([{"directory": build, "command": f"clang++ -x c++ -c {stub}", "file": stub}], db)
            fake = os.path.join(build, "fake-clang-tidy")
            with open(fake, "w") as f:
                f.write(
                    textwrap.dedent(
                        f"""\
                        #!{sys.executable}
                        import sys
                        for arg in sys.argv[1:]:
                            if arg.startswith("--export-fixes="):
                                open(arg.split("=", 1)[1], "w").write({export!r})
                        sys.exit({returncode})
                        """
                    )
                )
            os.chmod(fake, 0o755)
            return subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    build,
                    "--clang-tidy",
                    fake,
                    *flags,
                    "tt_metal/api/tt-metalium/core_coord.hpp",
                ],
                capture_output=True,
                text=True,
            )

    def test_without_the_flag_nothing_fails(self):
        for export, returncode in ((EXPORT_FINDINGS, 0), (EXPORT_WITH_ERROR, 1), ("", 139)):
            run = self.run_script(export, returncode)
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_clean(self):
        run = self.run_script("", 0, "--fail-on-findings")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("passed", run.stdout)
        self.assertIn("**PASSED**", run.stdout)

    def test_findings_fail(self):
        run = self.run_script(EXPORT_FINDINGS, 0, "--fail-on-findings")
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertIn("**FAILED**: 1 header(s) with findings, 0 header(s) failed.", run.stdout)
        self.assertIn("| with findings | 1 |", run.stdout)
        self.assertIn("| missing includes | 1 headers, 1 includes to add |", run.stdout)
        self.assertIn("| unused includes | 1 headers, 2 includes to remove |", run.stdout)
        self.assertIn("| `tt-metalium/core_coord.hpp` | `#include <cstddef>` |", run.stdout)
        self.assertIn(
            ".github/scripts/utils/run_api_header_include_cleaner.py .build/iwyu --all --fail-on-findings", run.stdout
        )
        self.assertIn(
            ".github/scripts/utils/run_api_header_include_cleaner.py .build/iwyu --fix "
            "tt_metal/api/tt-metalium/core_coord.hpp",
            run.stdout,
        )
        self.assertIn("::error::misc-include-cleaner: 1 tt_metal public header(s) have include findings", run.stderr)

    def test_parse_failure_fails(self):
        run = self.run_script(EXPORT_WITH_ERROR, 1, "--fail-on-findings")
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertIn("| failed (compile error, crash, no verification TU) | 1 |", run.stdout)
        self.assertIn(
            "| `tt-metalium/core_coord.hpp` | compile error | "
            "`clang-diagnostic-error: use of undeclared identifier 'undeclared_thing'` |",
            run.stdout,
        )

    def test_crash_fails(self):
        run = self.run_script("", 139, "--fail-on-findings")
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertIn(
            "| `tt-metalium/core_coord.hpp` | clang-tidy failed | "
            "`clang-tidy exited with 139 without reporting an error` |",
            run.stdout,
        )
        # A crash after partial findings still fails, and is listed as a failure.
        run = self.run_script(EXPORT_FINDINGS, -11 % 256, "--fail-on-findings")
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertIn("1 header(s) failed", run.stdout)

    def test_missing_stub_fails(self):
        run = self.run_script("", 0, "--fail-on-findings", "tt_metal/api/tt-metalium/no_such.hpp")
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertIn(
            "| `tt_metal/api/tt-metalium/no_such.hpp` | not analyzed (no verification TU) |",
            run.stdout,
        )


if __name__ == "__main__":
    unittest.main()
