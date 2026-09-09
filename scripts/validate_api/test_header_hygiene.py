# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from validate_header_hygiene import load_exceptions, source_lines, validate


class HeaderHygieneTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def header(self, name, contents="#pragma once\n"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        return path

    def errors(self, exceptions=None):
        return validate(self.root, exceptions or set())[0]

    def test_dependency_matrix(self):
        paths = {
            "stable": "tt-metalium/stable.hpp",
            "experimental": "tt-metalium/experimental/feature.hpp",
            "internal": "internal/private.hpp",
        }
        forbidden = {("stable", "experimental"), ("stable", "internal"), ("experimental", "internal")}
        for source_tier, source in paths.items():
            for target_tier, target in paths.items():
                with self.subTest(source=source_tier, target=target_tier):
                    self.header(source, f"#pragma once\n#include <{target}>\n")
                    self.assertEqual(bool(self.errors()), (source_tier, target_tier) in forbidden)
                    self.header(source)

    def test_pragma_once_after_license_and_comments(self):
        self.header("tt-metalium/a.hpp", "// License\n/* Another\ncomment */\n # pragma once // guard\n")
        self.assertEqual(self.errors(), [])

    def test_missing_conditional_or_late_pragma_once(self):
        for text in (
            "",
            "// #pragma once\n",
            "#ifndef A\n#define A\n#endif\n",
            "#if ENABLED\n#pragma once\n#endif\n",
            "int x;\n#pragma once\n",
            "#include <vector>\n#pragma once\n",
        ):
            with self.subTest(text=text):
                self.header("tt-metalium/a.hpp", text)
                self.assertTrue(any("unconditional #pragma once" in error for error in self.errors()))

    def test_comments_and_literals_do_not_create_includes(self):
        self.header(
            "tt-metalium/a.hpp",
            "#pragma once\n// #include <internal/a.hpp>\n"
            "/*\n#include <internal/a.hpp>\n*/\n"
            'const char* x = "// not a comment";\n'
            'const char* y = R"tag(\n#include <internal/a.hpp>\n)tag";\n',
        )
        self.assertEqual(self.errors(), [])

    def test_comments_continuations_and_original_line_number(self):
        self.header(
            "tt-metalium/a.hpp",
            "// License\n#pragma once\n#/**/include \\\n<internal/a.hpp> // implementation\n",
        )
        errors = self.errors()
        self.assertEqual(len(errors), 1)
        self.assertIn("a.hpp:3:", errors[0])
        self.assertIn("stable API must not include internal", errors[0])

    def test_continued_line_comment(self):
        self.header("tt-metalium/a.hpp", "#pragma once\n// example \\\n#include <internal/a.hpp>\n")
        self.assertEqual(self.errors(), [])

    def test_all_conditional_branches_are_checked(self):
        self.header("tt-metalium/a.hpp", "#pragma once\n#if OTHER_ARCH\n#include <internal/a.hpp>\n#endif\n")
        self.assertTrue(any("must not include internal" in error for error in self.errors()))

    def test_relative_and_normalized_includes(self):
        self.header("internal/private.hpp")
        for include in ('"../internal/private.hpp"', "<tt-metalium/../internal/private.hpp>"):
            with self.subTest(include=include):
                self.header("tt-metalium/a.hpp", f"#pragma once\n#include {include}\n")
                self.assertTrue(any("must not include internal" in error for error in self.errors()))

    def test_relative_include_uses_local_header_before_api_root(self):
        self.header("tt-metalium/local.hpp")
        self.header("tt-metalium/a.hpp", '#pragma once\n#include "local.hpp"\n')
        self.assertEqual(self.errors(), [])

    def test_symlink_does_not_hide_internal_target(self):
        self.header("internal/private.hpp")
        self.header("tt-metalium/a.hpp", "#pragma once\n#include <tt-metalium/alias.hpp>\n")
        (self.root / "tt-metalium/alias.hpp").symlink_to("../internal/private.hpp")
        self.assertTrue(any("must not include internal" in error for error in self.errors()))

    def test_macro_include_cannot_silently_escape_check(self):
        self.header("tt-metalium/a.hpp", "#pragma once\n#include PRIVATE_HEADER\n")
        self.assertTrue(any("literal #include" in error for error in self.errors()))

    def test_external_includes_are_outside_lexical_boundary_check(self):
        self.header("tt-metalium/a.hpp", "#pragma once\n#include <vector>\n#include <tt_stl/span.hpp>\n")
        self.assertEqual(self.errors(), [])

    def test_wrong_internal_location(self):
        self.header("tt-metalium/internal/private.hpp")
        self.header("tt-metalium/a.hpp", "#pragma once\n#include <tt-metalium/internal/private.hpp>\n")
        errors = self.errors()
        self.assertTrue(any("unsupported API location" in error for error in errors))
        self.assertTrue(any("internal headers belong" in error for error in errors))

    def test_exact_exception_cannot_exempt_another_header_or_target(self):
        self.header("tt-metalium/a.hpp", "#pragma once\n#include <internal/private.hpp>\n")
        exception = {("tt-metalium/a.hpp", "internal/private.hpp")}
        self.assertEqual(self.errors(exception), [])
        self.header("tt-metalium/b.hpp", "#pragma once\n#include <internal/private.hpp>\n")
        self.assertEqual(len(self.errors(exception)), 1)
        self.header(
            "tt-metalium/a.hpp", "#pragma once\n#include <internal/private.hpp>\n#include <internal/other.hpp>\n"
        )
        self.assertEqual(len(self.errors(exception)), 2)

    def test_repaired_or_deleted_include_requires_exception_removal(self):
        path = self.header("tt-metalium/a.hpp")
        self.header("tt-metalium/b.hpp")
        exception = {("tt-metalium/a.hpp", "internal/private.hpp")}
        self.assertTrue(any("stale exception" in error for error in self.errors(exception)))
        path.unlink()
        self.assertTrue(any("stale exception" in error for error in self.errors(exception)))

    def test_empty_or_wrong_root_fails(self):
        self.assertTrue(any("no API headers" in error for error in self.errors()))

    def test_exception_metadata_is_required_and_wildcards_are_rejected(self):
        entry = {
            "source": "tt-metalium/a.hpp",
            "include": "internal/private.hpp",
            "owner": "@owner",
            "reason": "Existing migration shim",
            "remove_when": "Consumers have migrated",
        }
        path = self.root / "exceptions.json"
        path.write_text(json.dumps([entry]))
        self.assertEqual(load_exceptions(path), {("tt-metalium/a.hpp", "internal/private.hpp")})
        for bad in (
            {},
            [dict(entry, owner="")],
            [dict(entry, source="tt-metalium/*.hpp")],
            [dict(entry, source="tt-metalium/../tt-metalium/a.hpp")],
            [dict(entry, include="tt-metalium/stable.hpp")],
            [dict(entry, include=1)],
            [entry, entry],
        ):
            with self.subTest(bad=bad):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_exceptions(path)

    def test_comment_mask_preserves_line_numbers(self):
        self.assertEqual(source_lines("/* hello\nworld */\n#pragma once")[-1], (3, "#pragma once"))

    def test_empty_continued_line_preserves_start_location(self):
        self.assertEqual(source_lines("\\\n#pragma once"), [(1, "#pragma once")])

    def test_digit_separators_do_not_hide_multiline_comments(self):
        self.header(
            "tt-metalium/a.hpp",
            "#pragma once\nconstexpr int n = 1'000;\n/*\n#include <internal/a.hpp>\n*/\n// don't include internals\n",
        )
        self.assertEqual(self.errors(), [])


if __name__ == "__main__":
    unittest.main()
