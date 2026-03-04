# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
C++ kernel source parsing: body extraction, include inlining, collection helpers.
"""

import os
import re
from typing import List, Optional, Set, Tuple


# Matches `void kernel_main() {` with optional ALWI prefix
_KERNEL_MAIN_RE = re.compile(r"\b(?:ALWI\s+)?void\s+kernel_main\s*\(\s*\)\s*\{")


def _is_raw_string_prefix(source: str, quote_pos: int) -> bool:
    """Check if the ``"`` at *quote_pos* opens a C++ raw string literal.

    Raw strings are: ``R"delim(...)delim"`` with optional encoding prefix
    ``L``, ``u``, ``U``, or ``u8`` before ``R``.  The character before the
    entire prefix must not be an identifier character, otherwise ``R`` is
    just part of an identifier like ``myR"..."``.
    """
    # Character immediately before " must be R
    r = quote_pos - 1
    if r < 0 or source[r] != "R":
        return False

    # Check for optional encoding prefix before R: L, u, U, u8
    # Start of the full prefix token (inclusive)
    start = r
    before_r = r - 1
    if before_r >= 0:
        ch = source[before_r]
        if ch in "LU":
            start = before_r  # LR" or UR"
        elif ch == "u":
            start = before_r  # uR"
        elif ch == "8" and before_r >= 1 and source[before_r - 1] == "u":
            start = before_r - 1  # u8R"

    # The character before the prefix must not be an identifier char,
    # otherwise this is something like `someVarR"..."` not a raw string.
    before_prefix = start - 1
    if before_prefix >= 0 and (source[before_prefix].isalnum() or source[before_prefix] == "_"):
        return False
    return True


def _skip_raw_string(source: str, quote_pos: int) -> int:
    """Skip past a raw string literal starting at the ``"`` at *quote_pos*.

    Raw string syntax: ``R"delim(content)delim"`` where *delim* can be
    empty or up to 16 characters (no spaces, backslashes, or parens).

    Returns the index just past the closing ``"``, or ``len(source)`` if
    unterminated (malformed source — treat the rest as consumed).
    """
    n = len(source)
    # Find the opening '(' that ends the delimiter
    paren_pos = source.find("(", quote_pos + 1)
    if paren_pos == -1 or paren_pos - (quote_pos + 1) > 16:
        # Not a valid raw string — fall back to treating as regular string
        return _skip_regular_string(source, quote_pos)
    delim = source[quote_pos + 1 : paren_pos]
    # Find the closing sequence: )delim"
    close_seq = f'){delim}"'
    end = source.find(close_seq, paren_pos + 1)
    if end == -1:
        return n  # unterminated — consume rest
    return end + len(close_seq)


def _skip_quoted(source: str, quote_pos: int, quote_char: str) -> int:
    """Skip past a quoted literal starting at *quote_pos*. Handles backslash escapes."""
    i = quote_pos + 1
    n = len(source)
    while i < n:
        if source[i] == "\\":
            i += 2
            continue
        if source[i] == quote_char:
            return i + 1
        i += 1
    return n


def _skip_regular_string(source: str, quote_pos: int) -> int:
    """Skip past a ``"..."`` string literal."""
    return _skip_quoted(source, quote_pos, '"')


def _skip_char_literal(source: str, quote_pos: int) -> int:
    """Skip past a ``'...'`` character literal."""
    return _skip_quoted(source, quote_pos, "'")


def _find_matching_brace(source: str, open_pos: int) -> Optional[int]:
    """Find closing brace matching the opening brace at *open_pos*.

    Skips braces inside comments, strings, raw strings, and char literals.
    Returns the index of the closing ``}`` or ``None`` if not found.
    """
    depth = 1
    i = open_pos + 1
    n = len(source)

    while i < n:
        c = source[i]

        # Line comment: // to end of line
        if c == "/" and i + 1 < n and source[i + 1] == "/":
            end = source.find("\n", i + 2)
            i = n if end == -1 else end + 1
            continue

        # Block comment: /* ... */
        if c == "/" and i + 1 < n and source[i + 1] == "*":
            end = source.find("*/", i + 2)
            if end == -1:
                return None  # unterminated block comment
            i = end + 2
            continue

        # String literal (regular or raw)
        if c == '"':
            if _is_raw_string_prefix(source, i):
                i = _skip_raw_string(source, i)
            else:
                i = _skip_regular_string(source, i)
            continue

        # Character literal
        if c == "'":
            i = _skip_char_literal(source, i)
            continue

        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return None


def extract_kernel_body(source: str) -> str:
    """Extract the body of ``kernel_main()`` using regex + brace matching.

    Returns the inner body (without outer braces) of the kernel_main
    function definition.  Returns empty string if not found.
    """
    match = _KERNEL_MAIN_RE.search(source)
    if not match:
        return ""
    # The '{' is the last character of the match
    open_brace_pos = match.end() - 1
    close_pos = _find_matching_brace(source, open_brace_pos)
    if close_pos is None:
        return ""
    return source[open_brace_pos + 1 : close_pos]


# =============================================================================
# Include Inlining
# =============================================================================


def inline_local_includes(source: str, kernel_dir: Optional[str]) -> Tuple[List[Tuple[str, str]], str]:
    """Inline local ``#include "..."`` directives, returning ``(headers, remaining_source)``.

    Resolves paths relative to *kernel_dir*. Headers are returned as
    ``(resolved_path, content)`` tuples for file-scope placement.
    """
    if kernel_dir is None:
        return [], source

    lines = source.split("\n")
    result: List[str] = []
    headers: List[Tuple[str, str]] = []
    inlined: Set[str] = set()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#include "'):
            match = re.match(r'#include\s+"([^"]+)"', stripped)
            if match:
                inc_path = match.group(1)
                if inc_path not in inlined:
                    full_inc = os.path.normpath(os.path.join(kernel_dir, inc_path))
                    if os.path.exists(full_inc):
                        with open(full_inc, "r") as f:
                            inc_content = f.read()
                        # Strip #pragma once and nested local includes
                        header_lines: List[str] = []
                        for inc_line in inc_content.split("\n"):
                            stripped_inc = inc_line.strip()
                            if stripped_inc.startswith("#pragma once"):
                                continue
                            if stripped_inc.startswith('#include "'):
                                nested_match = re.match(r'#include\s+"([^"]+)"', stripped_inc)
                                if nested_match:
                                    nested = nested_match.group(1)
                                    nested_full = os.path.normpath(os.path.join(os.path.dirname(full_inc), nested))
                                    # Only skip nested includes that resolve to
                                    # local files.  Non-existent paths are kept
                                    # as-is — they may be system/SDK includes.
                                    if os.path.exists(nested_full):
                                        continue
                            header_lines.append(inc_line)
                        headers.append((full_inc, "\n".join(header_lines)))
                        inlined.add(inc_path)
                        continue  # Remove the #include line from remaining source
        result.append(line)

    return headers, "\n".join(result)


# =============================================================================
# Collection Helpers
# =============================================================================


def collect_includes(sources: List[str]) -> List[str]:
    """Collect unique #include lines from multiple source strings."""
    includes = set()
    for source in sources:
        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#include"):
                includes.add(stripped)
    return sorted(includes)


def collect_defines(sources: List[str]) -> List[str]:
    """Collect unique #define lines from multiple source strings (before kernel_main)."""
    defines: List[str] = []
    seen: Set[str] = set()
    for source in sources:
        # Find where kernel_main starts (line number)
        match = _KERNEL_MAIN_RE.search(source)
        kernel_main_line = None
        if match:
            kernel_main_line = source[: match.start()].count("\n")

        for line_no, line in enumerate(source.split("\n")):
            if kernel_main_line is not None and line_no >= kernel_main_line:
                break
            stripped = line.strip()
            if stripped.startswith("#define") and stripped not in seen:
                defines.append(line)
                seen.add(stripped)
    return defines
