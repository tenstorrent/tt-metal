# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from validate_api import (
    ALLOWED_PREFIXES,
    ALLOWED_UMD_HEADERS,
    BANNED_HEADERS,
    SKIP_FILES,
    load_exceptions,
    source_lines,
    validate,
)


class ApiValidationTests(unittest.TestCase):
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
        # Small fixtures exercise individual rules without populating the
        # repository-wide prefix allowlist. Test that cleanup rule separately.
        return validate(self.root, exceptions or set(), check_unused_prefixes=False)[0]

    def populate_prefixes(self):
        includes = [
            "umd/device/types/arch.hpp" if prefix == "umd" else f"{prefix}/example.hpp" for prefix in ALLOWED_PREFIXES
        ]
        return self.header(
            "internal/allowlist.hpp", "#pragma once\n" + "".join(f"#include <{name}>\n" for name in includes)
        )

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

    def test_utf8_bom_before_guard_or_license(self):
        for prefix in ("\ufeff", "\ufeff// License\n"):
            with self.subTest(prefix=prefix):
                self.header("tt-metalium/a.hpp", prefix + "#pragma once\n#include <vector>\n")
                self.assertEqual(self.errors(), [])

    def test_non_newline_separators_do_not_end_comments(self):
        for separator in ("\v", "\f", "\u0085", "\u2028", "\u2029"):
            with self.subTest(separator=separator):
                self.header(
                    "tt-metalium/a.hpp",
                    "#pragma once\n// example" + separator + "#include <internal/a.hpp>\n"
                    "#include <internal/b.hpp>\n",
                )
                errors = self.errors()
                self.assertEqual(len(errors), 1)
                self.assertIn("a.hpp:3:", errors[0])
                self.assertIn("<internal/b.hpp>", errors[0])

    def test_horizontal_preprocessor_whitespace(self):
        for whitespace in ("\t", "\v", "\f"):
            with self.subTest(whitespace=whitespace):
                self.header("tt-metalium/a.hpp", f"#pragma{whitespace}once\n#include{whitespace}<vector>\n")
                self.assertEqual(self.errors(), [])

    def test_physical_line_endings_and_final_newline(self):
        for newline in ("\n", "\r\n", "\r"):
            for trailing in ("", newline):
                with self.subTest(newline=newline, trailing=trailing):
                    text = newline.join(("// License", "#pragma once", "#include <internal/a.hpp>")) + trailing
                    self.assertIn((3, "#include <internal/a.hpp>"), source_lines(text))
                    self.header("tt-metalium/a.hpp", text)
                    errors = self.errors()
                    self.assertEqual(len(errors), 1)
                    self.assertIn("a.hpp:3:", errors[0])

    def test_raw_string_splices_do_not_create_closing_delimiters(self):
        for prefix in ("", "u8", "u", "U", "L"):
            for delimiter in ("", "tag"):
                with self.subTest(prefix=prefix, delimiter=delimiter):
                    self.header(
                        "tt-metalium/a.hpp",
                        f'#pragma once\nconstexpr auto example = {prefix}R"{delimiter}(\n'
                        f'){delimiter}\\\n"\n#include <internal/a.hpp>\n){delimiter}";\n'
                        "#include <internal/b.hpp>\n",
                    )
                    errors = self.errors()
                    self.assertEqual(len(errors), 1)
                    self.assertIn("a.hpp:7:", errors[0])
                    self.assertIn("<internal/b.hpp>", errors[0])

    def test_raw_string_splices_do_not_hide_real_includes(self):
        self.header(
            "tt-metalium/a.hpp",
            '#pragma once\nconstexpr auto example = R"tag(\n)ta\\\ng"\n/*\n'
            ')tag";\n#include <internal/a.hpp>\n// */\n',
        )
        errors = self.errors()
        self.assertEqual(len(errors), 1)
        self.assertIn("a.hpp:7:", errors[0])
        self.assertIn("must not include internal", errors[0])

    def test_raw_string_openers_can_span_splices(self):
        for opener in ('R\\\n"', 'u8\\\nR"', 'u8R\\\n"'):
            with self.subTest(opener=opener):
                self.header(
                    "tt-metalium/a.hpp",
                    f"#pragma once\nconstexpr auto example = {opener}tag(\n"
                    '#include <internal/a.hpp>\n)tag";\n#include <internal/b.hpp>\n',
                )
                errors = self.errors()
                self.assertEqual(len(errors), 1)
                self.assertIn("a.hpp:6:", errors[0])
                self.assertIn("<internal/b.hpp>", errors[0])

    def test_raw_openers_inside_comments_and_strings_are_ignored(self):
        self.header(
            "tt-metalium/a.hpp",
            '#pragma once\n// R"tag(\n/* R"tag( */\n'
            'constexpr auto example = "R\\"tag(";\n#include <internal/a.hpp>\n',
        )
        errors = self.errors()
        self.assertEqual(len(errors), 1)
        self.assertIn("a.hpp:5:", errors[0])
        self.assertIn("must not include internal", errors[0])

    def test_raw_strings_preserve_locations_after_comments_and_splices(self):
        self.header(
            "tt-metalium/a.hpp",
            "/* License\ncontinued */\n#pragma once\nconstexpr auto example = \\\n"
            'R"tag(\n)ta\\\ng"\n#include <internal/a.hpp>\n)tag"\n'
            'R"(\n#include <internal/b.hpp>\n)";\n/* trailing\ncomment */ #include <internal/c.hpp>\n',
        )
        errors = self.errors()
        self.assertEqual(len(errors), 1)
        self.assertIn("a.hpp:14:", errors[0])
        self.assertIn("<internal/c.hpp>", errors[0])

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

    def test_forbidden_includes_with_multiline_comments_are_rejected(self):
        for directive in (
            "# /* comment\ncontinued */ include <internal/a.hpp>",
            "#include /* comment\ncontinued */ <internal/a.hpp>",
        ):
            with self.subTest(directive=directive):
                self.header("tt-metalium/a.hpp", f"#pragma once\n{directive}\n")
                errors = self.errors()
                self.assertEqual(len(errors), 1)
                self.assertIn("a.hpp:2:", errors[0])
                self.assertIn("must not include internal", errors[0])

    def test_allowed_directives_with_multiline_comments_are_accepted(self):
        self.header(
            "tt-metalium/a.hpp",
            "#pragma /* comment\ncontinued */ once\n"
            "# /* comment\ncontinued */ include <vector>\n"
            "#include /* comment\ncontinued */ <string>\n",
        )
        self.assertEqual(self.errors(), [])

    def test_multiline_comments_and_continuations_preserve_source_locations(self):
        self.header(
            "tt-metalium/a.hpp",
            "/* license\ncomment */\n#pragma once\n"
            "# /* comment\n*/ include \\\n<internal/a.hpp>\n"
            "/* before directive\n*/ #include <internal/b.hpp>\n",
        )
        errors = self.errors()
        self.assertEqual(len(errors), 2)
        self.assertIn("a.hpp:4:", errors[0])
        self.assertIn("a.hpp:8:", errors[1])
        self.assertTrue(all("must not include internal" in error for error in errors))

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
        errors = self.errors()
        self.assertEqual(len(errors), 1)
        self.assertIn("Quoted includes are not allowed", errors[0])

    def test_missing_quoted_relative_includes_still_enforce_boundaries(self):
        cases = (
            ("tt-metalium/a.hpp", "../internal/private.hpp", "internal"),
            ("tt-metalium/a.hpp", "experimental/feature.hpp", "experimental"),
            ("tt-metalium/experimental/a.hpp", "../../internal/private.hpp", "internal"),
        )
        for source, include, target_tier in cases:
            with self.subTest(source=source, include=include):
                path = self.header(source, f'#pragma once\n#include "{include}"\n')
                errors = self.errors()
                self.assertEqual(len(errors), 2)
                self.assertTrue(any("Quoted includes are not allowed" in error for error in errors))
                self.assertTrue(any(f"must not include {target_tier}" in error for error in errors))
                path.unlink()

    def test_quoted_api_root_includes_enforce_boundaries_with_or_without_target(self):
        cases = (
            ("tt-metalium/a.hpp", "internal/private.hpp", "internal"),
            ("tt-metalium/a.hpp", "tt-metalium/experimental/feature.hpp", "experimental"),
            ("tt-metalium/experimental/a.hpp", "internal/private.hpp", "internal"),
        )
        for source, include, target_tier in cases:
            for exists in (False, True):
                with self.subTest(source=source, include=include, exists=exists):
                    path = self.header(source, f'#pragma once\n#include "{include}"\n')
                    target = self.header(include) if exists else None
                    errors = self.errors()
                    self.assertEqual(len(errors), 2)
                    self.assertTrue(any("Quoted includes are not allowed" in error for error in errors))
                    self.assertTrue(any(f"must not include {target_tier}" in error for error in errors))
                    path.unlink()
                    if target is not None:
                        target.unlink()

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

    def test_migration_exception_only_exempts_the_tier_edge(self):
        self.header("tt-metalium/a.hpp", '#include "internal/private.hpp"\n')
        errors = self.errors({("tt-metalium/a.hpp", "internal/private.hpp")})
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("unconditional #pragma once" in error for error in errors))
        self.assertTrue(any("Quoted includes are not allowed" in error for error in errors))

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

    def test_digit_separators_before_inline_comments_with_apostrophes(self):
        for number in ("1'000", "1'000'000", "0xA'B", "0b1'0", ".1'0", "1e+1'0", "0x1p-1'0"):
            with self.subTest(number=number):
                self.header(
                    "tt-metalium/a.hpp",
                    f"#pragma once\nconstexpr auto n = {number}; /* don't include this example\n"
                    "#include <internal/a.hpp>\n*/\n",
                )
                self.assertEqual(self.errors(), [])

    def test_character_literals_do_not_hide_following_comments(self):
        for literal in ("'/'", "'\"'", "'\\''", "u8'a'", "u'a'", "U'a'", "L'a'"):
            with self.subTest(literal=literal):
                self.header(
                    "tt-metalium/a.hpp",
                    f"#pragma once\nconstexpr auto c = {literal}; /* don't include this example\n"
                    "#include <internal/a.hpp>\n*/\n#include <internal/b.hpp>\n",
                )
                errors = self.errors()
                self.assertEqual(len(errors), 1)
                self.assertIn("a.hpp:5:", errors[0])
                self.assertIn("<internal/b.hpp>", errors[0])

    def test_unapproved_includes_are_rejected(self):
        for include in ("unknown_dependency/header.hpp", "unprefixed.hpp"):
            with self.subTest(include=include):
                self.header("tt-metalium/a.hpp", f"#pragma once\n#include <{include}>\n")
                self.assertTrue(any("Include is not whitelisted" in error for error in self.errors()))

    def test_heavyweight_header_bans_are_preserved(self):
        for include in BANNED_HEADERS:
            with self.subTest(include=include):
                self.header("tt-metalium/a.hpp", f"#pragma once\n#include <{include}>\n")
                self.assertTrue(any("Banned include in public API" in error for error in self.errors()))

    def test_umd_allowlist_is_preserved(self):
        for include in ALLOWED_UMD_HEADERS:
            with self.subTest(include=include):
                self.header("tt-metalium/a.hpp", f"#pragma once\n#include <{include}>\n")
                self.assertEqual(self.errors(), [])
        self.header("tt-metalium/a.hpp", "#pragma once\n#include <umd/device/new_header.hpp>\n")
        self.assertTrue(any("New UMD include not allowed" in error for error in self.errors()))

    def test_legacy_skip_list_only_exempts_include_style(self):
        for name in SKIP_FILES:
            with self.subTest(name=name):
                path = self.header(f"tt-metalium/{name}", '#pragma once\n#include "private.hpp"\n')
                self.assertEqual(self.errors(), [])
                path.write_text('#include "internal/private.hpp"\n')
                errors = self.errors()
                self.assertEqual(len(errors), 2)
                self.assertTrue(any("unconditional #pragma once" in error for error in errors))
                self.assertTrue(any("must not include internal" in error for error in errors))
                path.unlink()

    def test_cpp_sources_keep_include_checks_without_header_only_rules(self):
        path = self.header("tt-metalium/a.cpp", "#include <internal/private.hpp>\n")
        self.assertEqual(self.errors(), [])
        path.write_text('#include "private.hpp"\n')
        self.assertTrue(any("Quoted includes are not allowed" in error for error in self.errors()))

    def test_unused_prefix_check_is_preserved(self):
        self.header("tt-metalium/a.hpp", "#pragma once\n#include <vector>\n")
        self.assertTrue(any("Unused allowed prefixes" in error for error in validate(self.root, set())[0]))
        self.populate_prefixes()
        self.assertEqual(validate(self.root, set())[0], [])

    def test_existing_cli_runs_both_include_and_header_checks(self):
        self.populate_prefixes()
        exceptions = self.root / "exceptions.json"
        exceptions.write_text("[]")
        command = [
            sys.executable,
            str(Path(__file__).with_name("validate_api.py")),
            str(self.root),
            "--exceptions",
            str(exceptions),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.header("tt-metalium/a.hpp", '#include "internal/private.hpp"\n')
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        for diagnostic in (
            "unconditional #pragma once",
            "Quoted includes are not allowed",
            "must not include internal",
        ):
            self.assertIn(diagnostic, result.stdout)


if __name__ == "__main__":
    unittest.main()
