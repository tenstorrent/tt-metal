#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pre-commit script to enforce std:: prefix on uint*_t types and ensure <cstdint> is included.

This script:
1. Checks .h and .cpp files for uint8_t, uint16_t, uint32_t, uint64_t (and int variants)
2. Adds std:: prefix if missing
3. Converts standalone 'uint' (not part of identifiers) to std::uint32_t
4. Adds #include <cstdint> if the types are used but the include is missing

Edge cases handled:
- Multi-line comments /* ... */
- Single-line comments // ...
- Regular string literals "..."
- Raw string literals R"(...)" and R"delim(...)delim"
- Prefixed strings: L"...", u"...", U"...", u8"..."
- Prefixed raw strings: LR"(...)", uR"(...)", etc.
- Character literals (won't match type patterns)
- Commented-out #include directives
- Types inside macros and preprocessor directives
- Standalone 'uint' vs 'uint' as part of identifier names (e.g., my_uint_var)
"""

import argparse
import io
import re
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class FixResult:
    """Result of fixing a single file."""

    file_path: Path
    had_changes: bool
    errors: List[str] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)


class CstdintFixer:
    """Fixes uint*_t types to use std:: prefix, converts standalone uint to std::uint32_t, and ensures <cstdint> is included."""

    # File extensions to check
    CHECK_EXTENSIONS = frozenset({".cpp", ".cc", ".h", ".hpp"})

    # Pattern to find uint*_t and int*_t types without std:: prefix
    # Uses negative lookbehind to exclude those already prefixed with std::
    UINT_TYPES_PATTERN = re.compile(r"(?<!std::)\b(u?int(?:8|16|32|64)_t)\b")

    # Pattern to find standalone 'uint' (without suffix) that needs to be converted to std::uint32_t
    # Uses negative lookbehind to exclude those already prefixed with std::
    # Uses negative lookahead to ensure it's not followed by _ or alphanumeric (not part of identifier)
    # Uses negative lookbehind to ensure it's not preceded by _ or alphanumeric (not part of identifier)
    UINT_STANDALONE_PATTERN = re.compile(
        r"(?<!std::)(?<![_a-zA-Z0-9])\b(uint)\b(?![_a-zA-Z0-9])"
    )

    # Pattern to check if <cstdint> is already included (no MULTILINE needed - applied per line)
    CSTDINT_INCLUDE_PATTERN = re.compile(r"^\s*#\s*include\s*<cstdint>\s*$")

    # Pattern to find std::uint*_t and std::int*_t types (already prefixed)
    STD_UINT_TYPES_PATTERN = re.compile(r"\bstd::(u?int(?:8|16|32|64)_t)\b")

    # Pattern to find std::uint32_t (used to check if standalone uint conversions need cstdint)
    STD_UINT32_PATTERN = re.compile(r"\bstd::uint32_t\b")

    # Pattern for system includes: #include <...>
    SYSTEM_INCLUDE_PATTERN = re.compile(r"^\s*#\s*include\s*<[^>]+>\s*$")

    # Pattern for local includes: #include "..."
    LOCAL_INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s*"[^"]+"\s*$')

    # Patterns for string/comment detection
    # Raw string literals: optional prefix (L/u/U/u8) + R"delimiter(content)delimiter"
    # Delimiter can be 0-16 chars, cannot contain (), \, space, or newline
    RAW_STRING_PATTERN = re.compile(
        r'(?:L|u|U|u8)?R"([^()\\\s]{0,16})\(.*?\)\1"', re.DOTALL
    )

    # Regular string literals with optional prefix (L/u/U/u8)
    # Handles escaped characters including \"
    STRING_PATTERN = re.compile(r'(?:L|u|U|u8)?"(?:[^"\\]|\\.)*"')

    # Character literals (single quotes) - also handle escape sequences and prefixes
    # Excludes newlines to avoid spanning across lines (e.g., apostrophes in comments).
    CHAR_LITERAL_PATTERN = re.compile(r"(?:L|u|U|u8)?'(?:[^'\\\n]|\\.)*'")

    # Multi-line comments /* ... */
    # Avoid matching comment-like patterns inside // line comments (e.g., "//*").
    MULTILINE_COMMENT_PATTERN = re.compile(r"(?<!/)/\*.*?\*/", re.DOTALL)

    # Single-line comments // ... (to end of line)
    SINGLE_LINE_COMMENT_PATTERN = re.compile(r"//[^\n]*")

    def should_check_file(self, file_path: Path) -> bool:
        """Check if file should be processed based on extension."""
        return file_path.suffix.lower() in self.CHECK_EXTENSIONS

    def _is_tests_python_file(self, file_path: Path) -> bool:
        """Check if file is a Python file under tests/."""
        return file_path.suffix.lower() == ".py" and "tests" in file_path.parts

    def _create_safe_content(self, content: str) -> str:
        """
        Create a version of content with comments and strings replaced by spaces.

        This preserves character positions for accurate location tracking while
        ensuring we don't match types inside strings or comments.

        Processing order matters:
        1. Raw strings first (can contain sequences that look like other patterns)
        2. Regular strings
        3. Multi-line comments (can span lines, may contain string-like sequences)
        4. Single-line comments (last, as // could appear in strings)
        5. Character literals (after comments to avoid apostrophes in comments)
        """
        safe = content

        # 1. Raw string literals (can span multiple lines, process first)
        safe = self.RAW_STRING_PATTERN.sub(lambda m: " " * len(m.group(0)), safe)

        # 2. Regular string literals
        safe = self.STRING_PATTERN.sub(lambda m: " " * len(m.group(0)), safe)

        # 3. Multi-line comments /* ... */
        safe = self.MULTILINE_COMMENT_PATTERN.sub(lambda m: " " * len(m.group(0)), safe)

        # 4. Single-line comments // ...
        safe = self.SINGLE_LINE_COMMENT_PATTERN.sub(
            lambda m: " " * len(m.group(0)), safe
        )

        # 5. Character literals
        safe = self.CHAR_LITERAL_PATTERN.sub(lambda m: " " * len(m.group(0)), safe)

        return safe

    def _find_types_needing_prefix(
        self, safe_content: str
    ) -> List[Tuple[int, int, str, str]]:
        """
        Find all uint*_t/int*_t types that need std:: prefix, and standalone uint.

        Args:
            safe_content: Content with comments/strings already removed via _create_safe_content()

        Returns list of (start_pos, end_pos, type_name, replacement) tuples.
        - For uint*_t/int*_t types: replacement is "std::" + type_name
        - For standalone uint: replacement is "std::uint32_t"
        Only finds types in actual code (not in comments or strings).
        """
        types_found = []

        # Find uint*_t and int*_t types
        for match in self.UINT_TYPES_PATTERN.finditer(safe_content):
            type_name = match.group(1)
            replacement = f"std::{type_name}"
            types_found.append((match.start(), match.end(), type_name, replacement))

        # Find standalone uint
        for match in self.UINT_STANDALONE_PATTERN.finditer(safe_content):
            type_name = match.group(1)
            replacement = "std::uint32_t"
            types_found.append((match.start(), match.end(), type_name, replacement))

        return types_found

    def _has_cstdint_include(self, safe_content: str) -> bool:
        """
        Check if <cstdint> is already included (not in comments/strings).

        Args:
            safe_content: Content with comments/strings already removed via _create_safe_content()
        """
        for line in safe_content.split("\n"):
            if self.CSTDINT_INCLUDE_PATTERN.match(line):
                return True
        return False

    def _has_std_cstdint_types(self, safe_content: str) -> bool:
        """
        Check if content has any std::uint*_t or std::int*_t types.

        Args:
            safe_content: Content with comments/strings already removed via _create_safe_content()
        """
        return bool(self.STD_UINT_TYPES_PATTERN.search(safe_content)) or bool(
            self.STD_UINT32_PATTERN.search(safe_content)
        )

    def _replace_cstdint_in_text(self, text: str) -> str:
        """Replace uint types in a text snippet."""
        text = self.UINT_TYPES_PATTERN.sub(r"std::\1", text)
        text = self.UINT_STANDALONE_PATTERN.sub("std::uint32_t", text)
        return text

    def _fix_python_test_file(
        self, file_path: Path, dry_run: bool = False
    ) -> FixResult:
        """
        Fix uint*_t types and standalone uint inside Python string literals
        for files under tests/.
        """
        result = FixResult(file_path=file_path, had_changes=False)

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return result
        except Exception as e:
            result.errors.append(f"Error reading file: {e}")
            return result

        lines = content.splitlines(keepends=True)
        line_offsets = [0]
        for line in lines[:-1]:
            line_offsets.append(line_offsets[-1] + len(line))

        def pos_to_index(pos: Tuple[int, int]) -> int:
            row, col = pos
            return line_offsets[row - 1] + col

        replacements: List[Tuple[int, int, str]] = []
        try:
            tokens = tokenize.generate_tokens(io.StringIO(content).readline)
            for token in tokens:
                if token.type != tokenize.STRING:
                    continue
                start = pos_to_index(token.start)
                end = pos_to_index(token.end)
                original = token.string
                updated = self._replace_cstdint_in_text(original)
                if updated != original:
                    replacements.append((start, end, updated))
        except tokenize.TokenError as e:
            result.errors.append(f"Error tokenizing file: {e}")
            return result

        if replacements:
            new_content = content
            for start, end, updated in sorted(
                replacements, key=lambda x: x[0], reverse=True
            ):
                new_content = new_content[:start] + updated + new_content[end:]
            result.had_changes = new_content != content
            if result.had_changes and not dry_run:
                try:
                    file_path.write_text(new_content, encoding="utf-8")
                except Exception as e:
                    result.errors.append(f"Error writing file: {e}")

        return result

    def _find_include_insert_position(self, content: str, safe_content: str) -> int:
        """
        Find the best position to insert #include <cstdint>.

        Strategy:
        1. If there are existing #include <...> (system includes), add after the last one
        2. If there are only #include "..." (local includes), add before the first one
        3. Otherwise, add after any header comment/copyright block

        Only considers includes that are in actual code (not commented out).

        Args:
            content: Original file content
            safe_content: Content with comments/strings already removed via _create_safe_content()
        """
        last_system_include_end = -1
        first_local_include_start = -1

        original_lines = content.split("\n")
        safe_lines = safe_content.split("\n")
        current_pos = 0

        for i, safe_line in enumerate(safe_lines):
            # Check for system includes in safe content
            if self.SYSTEM_INCLUDE_PATTERN.match(safe_line):
                line_end = current_pos + len(original_lines[i]) + 1
                last_system_include_end = line_end

            # Check for local includes in safe content
            elif self.LOCAL_INCLUDE_PATTERN.match(safe_line):
                if first_local_include_start == -1:
                    first_local_include_start = current_pos

            current_pos += len(original_lines[i]) + 1

        # Prefer inserting after system includes
        if last_system_include_end != -1:
            return last_system_include_end

        # If only local includes, insert before them
        if first_local_include_start != -1:
            return first_local_include_start

        # If no includes at all, find end of header/copyright comments
        return self._find_after_header_comments(content)

    def _find_after_header_comments(self, content: str) -> int:
        """
        Find position after header comments (copyright block etc).

        This is used when no includes exist in the file.
        """
        lines = content.split("\n")
        current_pos = 0
        in_block_comment = False
        found_header_content = False

        for line in lines:
            line_stripped = line.strip()

            # Track block comments
            if not in_block_comment and "/*" in line_stripped:
                in_block_comment = True
                found_header_content = True

            if in_block_comment:
                if "*/" in line_stripped:
                    in_block_comment = False
                current_pos += len(line) + 1
                continue

            # Single-line comments at the top (copyright, license, etc.)
            if line_stripped.startswith("//"):
                found_header_content = True
                current_pos += len(line) + 1
                continue

            # Empty lines after header content
            if found_header_content and line_stripped == "":
                current_pos += len(line) + 1
                continue

            # Pragma once or include guards - these should come before includes
            if (
                line_stripped.startswith("#pragma once")
                or line_stripped.startswith("#ifndef")
                or line_stripped.startswith("#define")
            ):
                current_pos += len(line) + 1
                # Check if next line is also a guard-related define
                continue

            # We've reached actual code/content
            break

        return current_pos

    def fix_file(self, file_path: Path, dry_run: bool = False) -> FixResult:
        """
        Fix a single file.

        Args:
            file_path: Path to the file to fix
            dry_run: If True, don't actually modify the file

        Returns:
            FixResult with details about what was fixed
        """
        result = FixResult(file_path=file_path, had_changes=False)

        if self._is_tests_python_file(file_path):
            return self._fix_python_test_file(file_path, dry_run=dry_run)

        if not self.should_check_file(file_path):
            return result

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                original_content = f.read()
        except UnicodeDecodeError:
            # Skip binary files
            return result
        except Exception as e:
            result.errors.append(f"Error reading file: {e}")
            return result

        content = original_content

        # Create safe content once for all analysis (expensive regex processing)
        safe_content = self._create_safe_content(content)
        types_found = self._find_types_needing_prefix(safe_content)

        # Fix types by adding std:: prefix or converting uint to std::uint32_t
        # Process in reverse to preserve positions
        if types_found:
            types_found.sort(key=lambda x: x[0], reverse=True)
            for start, end, type_name, replacement in types_found:
                content = content[:start] + replacement + content[end:]
                if type_name == "uint" and replacement == "std::uint32_t":
                    result.fixes_applied.append(
                        f"Converted {type_name} to {replacement}"
                    )
                else:
                    result.fixes_applied.append(f"Added std:: prefix to {type_name}")

        # Check if we need to add #include <cstdint>
        # Add if ANY std:: types exist (whether pre-existing or just fixed) and include is missing
        # Re-create safe content after modifications for accurate check
        if types_found:
            safe_content = self._create_safe_content(content)

        if not self._has_cstdint_include(safe_content) and self._has_std_cstdint_types(
            safe_content
        ):
            insert_pos = self._find_include_insert_position(content, safe_content)
            include_line = "#include <cstdint>\n"

            # Check if we need a newline before
            if insert_pos > 0 and content[insert_pos - 1] != "\n":
                include_line = "\n" + include_line

            content = content[:insert_pos] + include_line + content[insert_pos:]
            result.fixes_applied.append("Added #include <cstdint>")

        if content != original_content:
            result.had_changes = True
            if not dry_run:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                except Exception as e:
                    result.errors.append(f"Error writing file: {e}")

        return result

    def check_file(self, file_path: Path) -> FixResult:
        """
        Check a file without modifying it. Returns what would be fixed.
        """
        return self.fix_file(file_path, dry_run=True)

    def process_files(
        self, file_paths: List[Path], fix: bool = False, quiet: bool = False
    ) -> bool:
        """
        Process multiple files.

        Args:
            file_paths: List of file paths to process
            fix: If True, fix issues. If False, just report.
            quiet: If True, only show errors/changes

        Returns:
            True if all files are clean (no fixes needed), False otherwise
        """
        all_clean = True

        for file_path in file_paths:
            if fix:
                result = self.fix_file(file_path)
            else:
                result = self.check_file(file_path)

            if result.errors:
                all_clean = False
                print(f"❌ {file_path}:")
                for error in result.errors:
                    print(f"    {error}")
            elif result.had_changes:
                all_clean = False
                if fix:
                    print(f"🔧 {file_path}:")
                else:
                    print(f"⚠️  {file_path}:")
                for fix_msg in result.fixes_applied:
                    print(f"    {fix_msg}")
            elif not quiet and self.should_check_file(file_path):
                print(f"✅ {file_path}")

        return all_clean


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fix uint*_t types to use std:: prefix, convert standalone uint to std::uint32_t, and ensure <cstdint> is included",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Files to check/fix",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix issues (add std:: prefix and #include <cstdint>)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output errors and fixes, not success messages",
    )

    args = parser.parse_args()

    fixer = CstdintFixer()

    # Get file paths to process (single pass to avoid double exists() check)
    file_paths = []
    missing_files = []
    for f in args.files:
        p = Path(f)
        if p.exists():
            file_paths.append(p)
        else:
            missing_files.append(f)

    if missing_files:
        print(f"Warning: Files not found: {', '.join(missing_files)}")

    if not file_paths:
        print("No files to check")
        return 0

    # Process files
    success = fixer.process_files(file_paths, fix=args.fix, quiet=args.quiet)

    # Print summary
    if success:
        if not args.quiet:
            print(f"\n✅ All {len(file_paths)} files are clean")
        return 0
    else:
        if args.fix:
            print(f"\n🔧 Fixed issues in files (please re-stage)")
        else:
            print(f"\n❌ Issues found. Run with --fix to auto-fix.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
