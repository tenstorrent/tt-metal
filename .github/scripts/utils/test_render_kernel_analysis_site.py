#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for native IWYU consolidation and static report delivery."""

import json
from pathlib import Path
import tempfile
import unittest

from render_kernel_analysis_site import collect, render_site


def native_output(source: str, symbol: str = "uint32_t", line: int = 12) -> str:
    return f"""warning: a compiler warning
{source} should add these lines:
#include <cstdint>  // for {symbol}
namespace example {{ class Foo; }}

{source} should remove these lines:
- #include "old.h"  // lines {line}-{line}

The full include-list for {source}:
#include <cstdint>  // for {symbol}
#include "already-present.h"
namespace example {{ class Foo; }}
---
({source} has correct #includes/fwd-decls)
"""


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.legs = self.root / "legs"
        self.site = self.root / "site"

    def leg(self, name, output=None, status="0", commands=None):
        path = self.legs / name
        path.mkdir(parents=True)
        (path / "compile_commands.json").write_text(json.dumps([] if commands is None else commands))
        if output is not None:
            (path / "iwyu.txt").write_text(output)
        if status is not None:
            (path / "iwyu-exit-code.txt").write_text(status + "\n")
        return path

    def test_deduplicates_across_legs_and_roots_without_merging_actions(self):
        self.leg("first", native_output("/work/tt_metal/header.h"))
        self.leg(
            "second",
            native_output("/opt/venv/lib/python3.10/site-packages/ttnn/tt_metal/header.h", "uint64_t", 99),
        )
        self.leg("third", '/work/tt_metal/header.h should add these lines:\n#include "old.h"\n\n')
        report = collect(self.legs)
        self.assertEqual(
            set(report.findings),
            {
                ("tt_metal/header.h", "add", "#include <cstdint>"),
                ("tt_metal/header.h", "add", "namespace example { class Foo; }"),
                ("tt_metal/header.h", "add", '#include "old.h"'),
                ("tt_metal/header.h", "remove", '#include "old.h"'),
            },
        )
        self.assertEqual(report.findings[("tt_metal/header.h", "remove", '#include "old.h"')], {(12, 12), (99, 99)})
        self.assertFalse(report.incomplete)

    def test_removal_locations_survive_deduplication_and_rendering(self):
        source = "/work/tt_metal/header.h"
        output = (
            native_output(source)
            + f"""{source} should remove these lines:
- struct Entry;  // lines 30-31
- #include "unlocated.h"

"""
        )
        self.leg("first", output)
        self.leg("duplicate", output)
        self.leg("another-location", native_output(source, line=99))
        report = collect(self.legs)
        self.assertEqual(len(report.findings), 5)
        self.assertEqual(report.findings[("tt_metal/header.h", "remove", "struct Entry;")], {(30, 31)})
        self.assertEqual(report.findings[("tt_metal/header.h", "remove", '#include "unlocated.h"')], set())
        self.assertEqual(report.findings[("tt_metal/header.h", "add", "#include <cstdint>")], set())
        render_site(self.legs, self.site)
        html = (self.site / "iwyu/index.html").read_text()
        self.assertEqual(html.count('&quot;old.h&quot; <span class="muted">// lines 12, 99</span>'), 1)
        self.assertIn('struct Entry; <span class="muted">// lines 30–31</span>', html)
        self.assertNotIn("&quot;unlocated.h&quot; <span", html)
        self.assertNotIn("#include &lt;cstdint&gt; <span", html)
        self.leg("single-line", native_output("/work/another.h", line=7))
        render_site(self.legs, self.site)
        self.assertIn(
            '&quot;old.h&quot; <span class="muted">// line 7</span>', (self.site / "iwyu/index.html").read_text()
        )

    def test_unknown_paths_remain_distinct(self):
        self.leg("first", native_output("/cache/a/header.h") + native_output("/cache/b/header.h"))
        self.assertEqual({f[0] for f in collect(self.legs).findings}, {"/cache/a/header.h", "/cache/b/header.h"})

    def test_errors_are_separate_and_deduplicated(self):
        error = "/work/tt_metal/header.h:4:2: error: ambiguous conversion\n"
        self.leg("first", error + native_output("/work/tt_metal/header.h"), "1")
        self.leg("second", error + "error: unknown target\n", "1")
        report = collect(self.legs)
        self.assertTrue(report.incomplete)
        self.assertEqual(report.errors, {"tt_metal/header.h:4:2: error: ambiguous conversion", "error: unknown target"})
        self.assertEqual(len(report.findings), 3)
        render_site(self.legs, self.site)
        html = (self.site / "iwyu/index.html").read_text()
        self.assertIn("Partial analysis.", html)
        self.assertIn("Parsing errors", html)

    def test_missing_output_or_exit_status_marks_partial_analysis(self):
        self.leg("empty-capture")
        self.assertFalse(collect(self.legs).incomplete)
        self.leg("no-status", native_output("/work/a.h"), status=None)
        self.assertTrue(collect(self.legs).incomplete)
        (self.legs / "no-status/iwyu-exit-code.txt").write_text("0\n")
        self.assertFalse(collect(self.legs).incomplete)
        self.leg("missing-output", commands=[{"file": "kernel.cc"}])
        self.assertTrue(collect(self.legs).incomplete)

    def test_html_escapes_tool_output_and_preserves_codechecker_pages(self):
        source = '/work/ttnn/<script>alert("file")</script>.h'
        self.leg("first", native_output(source))
        tidy = self.site / "clang-tidy"
        tidy.mkdir(parents=True)
        (tidy / "index.html").write_text("CodeChecker index")
        (tidy / "sample.plist.html").write_text("CodeChecker finding")
        self.assertTrue(render_site(self.legs, self.site))
        html = (self.site / "iwyu/index.html").read_text()
        self.assertNotIn('<script>alert("file")</script>', html)
        self.assertIn("#include &lt;cstdint&gt;", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn('href="../index.html"', html)
        index = (self.site / "index.html").read_text()
        self.assertIn('href="clang-tidy/index.html"', index)
        self.assertIn('href="iwyu/index.html"', index)
        self.assertEqual((tidy / "index.html").read_text(), "CodeChecker index")
        self.assertFalse(list(self.site.rglob("*.json")))
        before = html
        render_site(self.legs, self.site)
        self.assertEqual((self.site / "iwyu/index.html").read_text(), before)

    def test_empty_disabled_and_failed_runs_do_not_look_clean(self):
        self.assertFalse(render_site(self.legs, self.site))
        self.assertIn("No IWYU reports were collected.", (self.site / "iwyu/index.html").read_text())
        self.leg("failed", "error: cannot parse input\n", "1")
        # Partial output can still be published even when CodeChecker has no HTML.
        self.assertTrue(render_site(self.legs, self.site))
        self.assertIn("Partial analysis.", (self.site / "iwyu/index.html").read_text())
        self.assertFalse(render_site(self.legs, self.site, iwyu_enabled=False))
        self.assertIn("IWYU was disabled", (self.site / "iwyu/index.html").read_text())
        tidy = self.site / "clang-tidy"
        tidy.mkdir()
        (tidy / "index.html").write_text("An index alone does not mean CodeChecker rendered any findings")
        self.assertFalse(render_site(self.legs, self.site, iwyu_enabled=False))
        (tidy / "sample.plist.html").write_text("CodeChecker finding")
        self.assertTrue(render_site(self.legs, self.site, iwyu_enabled=False))
        (tidy / "index.html").write_text("")
        self.assertFalse(render_site(self.legs, self.site, iwyu_enabled=False))


if __name__ == "__main__":
    unittest.main()
