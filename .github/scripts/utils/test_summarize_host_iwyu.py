#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the host IWYU report summary against real IWYU 0.24 output."""

import unittest

from summarize_host_iwyu import FORWARD_DECLARATION, parse_report, render_markdown, rewrite_c_headers

# Verbatim (trimmed) iwyu_tool.py output from include-what-you-use 0.24 with
# --cxx17ns, as run by run_host_iwyu.sh. Two files with advice, one already
# correct, and one clang diagnostic. IWYU names files by the absolute path in
# the compilation database; /work is the checkout in iwyu-host.yaml.
REPORT = """\

/work/tt_metal/fabric/mesh_graph_descriptor.cpp should add these lines:
#include <cstddef>                                                    // for size_t
#include <utility>                                                    // for pair, get, move, swap
#include "tt_stl/strong_type.hpp"                                     // for StrongType
namespace tt::tt_fabric { class MeshGraph; }

/work/tt_metal/fabric/mesh_graph_descriptor.cpp should remove these lines:
- #include <google/protobuf/io/zero_copy_stream_impl.h>  // lines 29-29
- #include <unistd.h>  // lines 30-30

The full include-list for /work/tt_metal/fabric/mesh_graph_descriptor.cpp:
#include <google/protobuf/text_format.h>                              // for TextFormat
#include <cstddef>                                                    // for size_t
#include <utility>                                                    // for pair, get, move, swap
#include "protobuf/mesh_graph_descriptor.pb.h"                        // for GraphDescriptor, MeshGraphDescriptor
#include "tt_stl/strong_type.hpp"                                     // for StrongType
namespace tt::tt_fabric { class MeshGraph; }
---

/work/tt-train/sources/ttml/datasets/utils.cpp should add these lines:
#include <yaml-cpp/yaml.h>                        // for Node
#include <utility>                                // for move

/work/tt-train/sources/ttml/datasets/utils.cpp should remove these lines:

The full include-list for /work/tt-train/sources/ttml/datasets/utils.cpp:
#include <yaml-cpp/yaml.h>                        // for Node
#include <utility>                                // for move
---

(/work/tt_metal/impl/internal/disaggregation/kv_chunk_address_table_protobuf.hpp has correct #includes/fwd-decls)

/work/tests/tt_metal/broken.cpp:1:10: fatal error: 'nonexistent.hpp' file not found
    1 | #include <nonexistent.hpp>
      |          ^~~~~~~~~~~~~~~~~
"""


class ParseReportTests(unittest.TestCase):
    def test_counts_files_by_outcome(self):
        summary = parse_report(REPORT)
        self.assertEqual(summary.files_with_advice, 2)
        self.assertEqual(summary.clean, 1)
        self.assertEqual(summary.errors, 1)

    def test_histogram_covers_only_the_add_sections(self):
        additions = parse_report(REPORT).additions
        self.assertEqual(additions["#include <utility>"], 2)
        self.assertEqual(additions["#include <yaml-cpp/yaml.h>"], 1)
        self.assertEqual(additions['#include "tt_stl/strong_type.hpp"'], 1)
        self.assertEqual(additions[FORWARD_DECLARATION], 1)
        # Present in the full include-list and remove sections only.
        self.assertNotIn("#include <google/protobuf/text_format.h>", additions)
        self.assertNotIn("#include <unistd.h>", additions)

    def test_empty_report_recognises_nothing(self):
        summary = parse_report("")
        self.assertEqual(summary.recognised, 0)
        self.assertFalse(summary.additions)

    def test_unrecognised_text_recognises_nothing(self):
        self.assertEqual(parse_report("iwyu_tool.py: something unexpected\n").recognised, 0)

    def test_driver_warning_alone_recognises_nothing(self):
        # All iwyu_tool.py prints when a positional path selects no database
        # entry (a header-only directory, for instance); it then exits 0 having
        # analyzed nothing, so this must not read as a clean run.
        report = "warning: '/work/tt_metal/api' not found in compilation database.\n"
        self.assertEqual(parse_report(report).recognised, 0)


class RenderMarkdownTests(unittest.TestCase):
    def test_table_and_histogram(self):
        markdown = render_markdown(parse_report(REPORT), status=0)
        self.assertIn("Analyzer exit code: 0", markdown)
        self.assertIn("| with recommendations | 2 |", markdown)
        self.assertIn("| already correct | 1 |", markdown)
        self.assertIn("| compile/parse errors | 1 |", markdown)
        self.assertIn("| `#include <utility>` | 2 |", markdown)
        self.assertIn("iwyu-host-report", markdown)

    def test_no_histogram_without_additions(self):
        markdown = render_markdown(parse_report(""), status=1)
        self.assertIn("Analyzer exit code: 1", markdown)
        self.assertNotIn("Most-suggested additions", markdown)

    def test_title_and_artifact_are_overridable(self):
        markdown = render_markdown(parse_report(REPORT), status=0, title="API headers", artifact="api-report")
        self.assertIn("### API headers", markdown)
        self.assertIn("`api-report` artifact", markdown)
        self.assertNotIn("iwyu-host-report", markdown)


class RewriteCHeadersTests(unittest.TestCase):
    def test_recommended_c_headers_become_cxx_headers(self):
        report = (
            "a.hpp should add these lines:\n"
            "#include <stddef.h>                    // for size_t\n"
            "#include <stdint.h>                    // for uint32_t\n"
        )
        self.assertEqual(
            rewrite_c_headers(report),
            "a.hpp should add these lines:\n"
            "#include <cstddef>                     // for size_t\n"
            "#include <cstdint>                     // for uint32_t\n",
        )

    def test_only_the_c_compatibility_headers_are_rewritten(self):
        untouched = (
            "#include <sys/types.h>  // for ssize_t\n"
            "#include <unistd.h>  // for read\n"
            '#include "stdint.h"  // quoted\n'
            "#include <fmt/base.h>  // for format\n"
            "#include <cstdint>  // already C++\n"
        )
        self.assertEqual(rewrite_c_headers(untouched), untouched)

    def test_removals_name_existing_lines_and_stay_verbatim(self):
        removal = "- #include <stddef.h>  // lines 10-10\n"
        self.assertEqual(rewrite_c_headers(removal), removal)

    def test_histogram_counts_the_cxx_spelling(self):
        report = "a.hpp should add these lines:\n#include <stddef.h>  // for size_t\n\n"
        self.assertIn("#include <cstddef>", parse_report(rewrite_c_headers(report)).additions)


if __name__ == "__main__":
    unittest.main()
