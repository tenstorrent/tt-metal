#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate API includes, header guards, and direct stability boundaries.

This is a lexical check, not a C++ dependency resolver.
See scripts/validate_api/README.md for its scope and the compiler-based follow-up.
"""

import argparse
import json
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Optional

from common import CPP_EXTENSIONS


# Legacy include-style exceptions; guards and tier boundaries still apply.
SKIP_FILES = {
    "fabric_edm_packet_header.hpp",
    "dev_msgs.h",
    "dataflow_buffer.hpp",  # TODO: #37324: remove once dataflow buffer has proper host-dev interface
}

# Headers that must never be included from tt-metalium public API headers.
# These are heavyweight implementation headers whose transitive cost (compile time,
# dependency surface) is too high for the public interface.
BANNED_HEADERS = {
    "tt_stl/reflection.hpp": "reflection.hpp pulls in <reflect> and <nlohmann/json.hpp>; use forward declarations or move usage to .cpp files",
    "tt_stl/concepts.hpp": "concepts.hpp pulls in <reflect>; use sizeof(T)==0 for always_false_v, or move usage to .cpp files",
}

# Exhaustive set of UMD headers allowed in the public API.
# New UMD includes should NOT be added; the goal is to reduce (and eventually remove)
# the UMD surface from public headers.  If you need a UMD type in a public header,
# prefer forward declarations or move the dependency to a .cpp file.
ALLOWED_UMD_HEADERS = {
    "umd/device/types/arch.hpp",
    "umd/device/types/cluster_descriptor_types.hpp",
    "umd/device/types/core_coordinates.hpp",
    "umd/device/types/xy_pair.hpp",
}

ALLOWED_PREFIXES = {
    "hostdevcommon",
    "internal",
    "tt-metalium",
    "tt_stl",
    "umd",
    "fmt",
    "enchantum",
    "nlohmann",
}

STD_HEADERS = {
    # C++ standard headers
    "algorithm",
    "any",
    "array",
    "atomic",
    "barrier",
    "bit",
    "bitset",
    "cassert",
    "cctype",
    "charconv",
    "chrono",
    "climits",
    "cmath",
    "codecvt",
    "compare",
    "complex",
    "concepts",
    "condition_variable",
    "coroutine",
    "deque",
    "exception",
    "execution",
    "expected",
    "filesystem",
    "format",
    "forward_list",
    "fstream",
    "functional",
    "future",
    "initializer_list",
    "iomanip",
    "ios",
    "iosfwd",
    "iostream",
    "istream",
    "iterator",
    "latch",
    "limits",
    "list",
    "locale",
    "map",
    "memory",
    "memory_resource",
    "mutex",
    "new",
    "numbers",
    "numeric",
    "optional",
    "ostream",
    "queue",
    "random",
    "ranges",
    "ratio",
    "regex",
    "scoped_allocator",
    "semaphore",
    "set",
    "shared_mutex",
    "source_location",
    "span",
    "sstream",
    "stack",
    "stdexcept",
    "stop_token",
    "streambuf",
    "string",
    "string_view",
    "strstream",
    "syncstream",
    "system_error",
    "thread",
    "tuple",
    "type_traits",
    "typeindex",
    "typeinfo",
    "unordered_map",
    "unordered_set",
    "utility",
    "valarray",
    "variant",
    "vector",
    "version",
    # C library headers (C++ style)
    "cctype",
    "cerrno",
    "cfenv",
    "cfloat",
    "cinttypes",
    "ciso646",
    "climits",
    "clocale",
    "cmath",
    "csetjmp",
    "csignal",
    "cstdarg",
    "cstdbool",
    "cstddef",
    "cstdint",
    "cstdio",
    "cstdlib",
    "cstring",
    "ctime",
    "cuchar",
    "cwchar",
    "cwctype",
    # Legacy C headers (frequently used)
    "assert.h",
    "ctype.h",
    "errno.h",
    "fenv.h",
    "float.h",
    "inttypes.h",
    "iso646.h",
    "limits.h",
    "locale.h",
    "math.h",
    "setjmp.h",
    "signal.h",
    "stdarg.h",
    "stdbool.h",
    "stddef.h",
    "stdint.h",
    "stdio.h",
    "stdlib.h",
    "string.h",
    "time.h",
    "uchar.h",
    "wchar.h",
    "wctype.h",
    "unistd.h",
}

EXCEPTIONS = Path("scripts/validate_api/header_hygiene_exceptions.json")
HEADER_SUFFIXES = {".h", ".hpp", ".hh", ".hxx"}
ALLOWED_DEPENDENCIES = {
    "stable": {"stable"},
    "experimental": {"stable", "experimental"},
    "internal": {"stable", "experimental", "internal"},
}
# Preserve ordinary string/character literals so comment markers in them are not
# interpreted as comments. Recognize raw-string openers here, but find their
# delimiters in the original text: line splicing is reverted inside raw strings.
# Match preprocessing numbers before their digit separators can start a character
# literal, including separators after exponent signs and in hexadecimal values.
COMMENTS_AND_LITERALS = re.compile(
    r'R"'
    r"|(?<!\w)(?:\d|\.\d)(?:[eEpP][+-]|[\w.]|'\w)*"
    r'|"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|//[^\n]*|/\*.*?\*/",
    re.DOTALL,
)
RAW_STRING = re.compile(r'"(?P<delimiter>[^\s()\\]{0,16})\(.*?\)(?P=delimiter)"', re.DOTALL)
INCLUDE = re.compile(r'^\s*#\s*include\s*(?:<([^>]+)>|"([^"]+)")\s*$')
PRAGMA_ONCE = re.compile(r"^\s*#\s*pragma\s+once\s*$")


def source_lines(text: str) -> list[tuple[int, str]]:
    """Mask comments and raw strings, retaining physical directive locations."""
    text = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    # str.splitlines() would also turn Unicode characters inside comments and
    # literals into newlines that the C++ preprocessor does not see.
    physical_lines = text.split("\n")
    if physical_lines[-1] == "":
        physical_lines.pop()
    lines = []
    pending = []
    start = 1
    original_offsets = []
    original_offset = 0
    for number, line in enumerate(physical_lines, 1):
        if not pending:
            start = number
        continued = line.endswith("\\")
        content = line[:-1] if continued else line
        original_offsets.extend(range(original_offset, original_offset + len(content)))
        if not continued:
            original_offsets.append(original_offset + len(line))
        original_offset += len(line) + 1
        if continued:
            pending.append(content)
            continue
        lines.append((start, "".join(pending) + line))
        pending = []
    if pending:
        lines.append((start, "".join(pending)))
    if not lines:
        return []

    def mask(value: str) -> str:
        if value.startswith("/*"):
            # A block comment is whitespace even when it spans physical lines;
            # its internal newlines do not terminate a preprocessing directive.
            return " " * len(value)
        if value.startswith(("//", 'R"')):
            return "".join("\n" if c == "\n" else " " for c in value)
        return value

    spliced = "\n".join(line for _, line in lines)
    original_offsets = original_offsets[: len(spliced)]
    line_starts = [0] + [match.end() for match in re.finditer("\n", spliced)]
    chunks = []
    position = 0
    while match := COMMENTS_AND_LITERALS.search(spliced, position):
        chunks.append(spliced[position : match.start()])
        end = match.end()
        if match.group() == 'R"':
            # The opener itself may span a splice. Only the text starting at
            # its quote is restored, including the delimiter and closing quote.
            raw = RAW_STRING.match(text, original_offsets[match.end() - 1])
            if raw is not None:
                end = bisect_left(original_offsets, raw.end())
        chunks.append(mask(spliced[match.start() : end]))
        position = end
    chunks.append(spliced[position:])
    cleaned = "".join(chunks)
    # Masks preserve character offsets, so map the first substantive character
    # back to its physical line even after a leading multiline comment.
    result = []
    offset = 0
    for line in cleaned.split("\n"):
        first_token = offset + len(line) - len(line.lstrip())
        source_index = bisect_right(line_starts, first_token) - 1
        result.append((lines[source_index][0], line))
        offset += len(line) + 1
    return result


def tier(path: str) -> str | None:
    if path.startswith("tt-metalium/internal/"):
        return None  # Internal headers belong in api/internal/.
    if path.startswith("tt-metalium/experimental/"):
        return "experimental"
    if path.startswith("tt-metalium/"):
        return "stable"
    if path.startswith("internal/"):
        return "internal"
    return None


def include_target(api_root: Path, source: Path, name: str, quoted: bool) -> str | None:
    """Recognize API-root includes and quoted relative includes, normalizing '..'."""
    candidates = [api_root / name]
    if quoted:
        candidates.insert(0, source.parent / name)
    # Prefer existing headers in include-search order. If none exists, retain
    # the source-relative meaning of quoted paths, except canonical API-root
    # spellings such as "internal/foo.hpp". Compilation checks existence.
    fallback = api_root / name if name.startswith(("tt-metalium/", "internal/")) else candidates[0]
    target = next((p for p in candidates if p.is_file()), fallback)
    try:
        return target.resolve().relative_to(api_root.resolve()).as_posix()
    except ValueError:
        return None


def load_exceptions(path: Path) -> set[tuple[str, str]]:
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError("include exceptions must be a JSON array")
    exceptions = set()
    for entry in entries:
        fields = {"source", "include", "owner", "reason", "remove_when"}
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError(f"each exception must contain exactly {sorted(fields)}")
        if any(not isinstance(value, str) or not value.strip() for value in entry.values()):
            raise ValueError("exception fields must be nonempty strings")
        source, target = entry["source"], entry["include"]
        for name in (source, target):
            if Path(name).as_posix() != name or any(part in name for part in ("..", "*", "?", "[", "\\")):
                raise ValueError(f"exception paths must be exact canonical API-relative paths: {name}")
        source_tier, target_tier = tier(source), tier(target)
        if not source_tier or not target_tier or target_tier in ALLOWED_DEPENDENCIES[source_tier]:
            raise ValueError(f"exception is not a forbidden API include: {source} -> {target}")
        key = (source, target)
        if key in exceptions:
            raise ValueError(f"duplicate include exception: {source} -> {target}")
        exceptions.add(key)
    return exceptions


class Include(NamedTuple):
    source_file: str
    line_num: int
    path: str
    quoted: bool

    def __str__(self) -> str:
        brackets = '""' if self.quoted else "<>"
        return f"#include {brackets[0]}{self.path}{brackets[1]}"

    @staticmethod
    def from_line(source_file: str, line_num: int, line: str) -> Optional["Include"]:
        if match := INCLUDE.fullmatch(line):
            return Include(source_file, line_num, match[1] or match[2], quoted=match[2] is not None)
        return None

    @property
    def prefix(self) -> Optional[str]:
        parts = self.path.split("/", 1)
        return parts[0] if len(parts) > 1 else None

    def check_for_errors(self, prefix_counts: dict[str, int]) -> Optional[str]:
        if self.quoted:
            return f"{self.source_file}:{self.line_num}: Quoted includes are not allowed. Use angle brackets <...> ({self})"

        is_standard = self.path in STD_HEADERS
        has_valid_prefix = self.prefix and self.prefix in ALLOWED_PREFIXES

        if not (is_standard or has_valid_prefix):
            return f"{self.source_file}:{self.line_num}: Include is not whitelisted ({self})"

        if has_valid_prefix:
            prefix_counts[self.prefix] += 1

        return None

    def check_for_banned_header(self) -> Optional[str]:
        """Check if include is a banned header that should not appear in public API."""
        if self.path in BANNED_HEADERS:
            reason = BANNED_HEADERS[self.path]
            return f"{self.source_file}:{self.line_num}: " f"Banned include in public API: <{self.path}> ({reason})"
        return None

    def check_for_umd_header(self) -> Optional[str]:
        """Error if a UMD include is not in the frozen allowlist."""
        if self.prefix == "umd" and self.path not in ALLOWED_UMD_HEADERS:
            return (
                f"{self.source_file}:{self.line_num}: "
                f"New UMD include not allowed in public API: <{self.path}> "
                f"(only {', '.join(sorted(ALLOWED_UMD_HEADERS))} are permitted; "
                f"prefer forward declarations or move usage to .cpp files)"
            )
        return None


def validate(
    api_root: Path, exceptions: set[tuple[str, str]], *, check_unused_prefixes: bool = True
) -> tuple[list[str], int]:
    errors = []
    used_exceptions = set()
    prefix_counts = defaultdict(int)
    suffixes = set(CPP_EXTENSIONS) | HEADER_SUFFIXES
    source_files = sorted(p for p in api_root.rglob("*") if p.is_file() and p.suffix in suffixes)
    if not source_files:
        return [f"{api_root}: no API headers or sources found; check the directory"], 0
    header_count = 0
    for source_file in source_files:
        is_header = source_file.suffix in HEADER_SUFFIXES
        source = source_file.relative_to(api_root).as_posix()
        source_tier = tier(source) if is_header else None
        lines = source_lines(source_file.read_text(encoding="utf-8"))
        if is_header:
            header_count += 1
            if source_tier is None:
                errors.append(
                    f"{source_file}:1: unsupported API location; use tt-metalium/, tt-metalium/experimental/, or internal/"
                )
            substantive = [(number, line) for number, line in lines if line.strip()]
            if not substantive or not PRAGMA_ONCE.fullmatch(substantive[0][1]):
                errors.append(
                    f"{source_file}:1: place unconditional #pragma once before declarations and other directives"
                )
        for number, line in lines:
            if not re.match(r"^\s*#\s*include\b", line):
                continue
            include = Include.from_line(str(source_file), number, line)
            if include is None:
                if is_header:
                    errors.append(f"{source_file}:{number}: use a literal #include so the API boundary can be checked")
                continue
            if source_file.name not in SKIP_FILES:
                errors.extend(
                    error
                    for error in (
                        include.check_for_errors(prefix_counts),
                        include.check_for_banned_header(),
                        include.check_for_umd_header(),
                    )
                    if error is not None
                )
            if not is_header:
                continue
            target = include_target(api_root, source_file, include.path, include.quoted)
            target_tier = tier(target) if target else None
            if target and target.startswith("tt-metalium/internal/"):
                errors.append(
                    f"{source_file}:{number}: internal headers belong in api/internal/, not tt-metalium/internal/"
                )
            elif source_tier and target_tier and target_tier not in ALLOWED_DEPENDENCIES[source_tier]:
                key = (source, target)
                if key in exceptions:
                    used_exceptions.add(key)
                else:
                    errors.append(
                        f"{source_file}:{number}: {source_tier} API must not include {target_tier} header <{target}>; "
                        "move the dependency into a source file or the interface into the appropriate API tier"
                    )
    for source, target in sorted(exceptions - used_exceptions):
        errors.append(f"{EXCEPTIONS}: stale exception {source} -> {target}; remove it with the repaired include")
    unused_prefixes = ALLOWED_PREFIXES - prefix_counts.keys()
    if check_unused_prefixes and unused_prefixes:
        errors.append(f"Unused allowed prefixes (not seen in any #include): {', '.join(sorted(unused_prefixes))}")
    return errors, header_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="API root, normally tt_metal/api")
    parser.add_argument("--exceptions", type=Path, default=Path(__file__).with_name(EXCEPTIONS.name))
    args = parser.parse_args()
    try:
        exceptions = load_exceptions(args.exceptions)
        errors, count = validate(args.directory, exceptions)
    except (OSError, ValueError) as error:
        print(f"API validation: {error}")
        return 1
    for error in errors:
        print(error)
    print(f"API validation: {count} headers, {len(exceptions)} recorded include exceptions, {len(errors)} errors")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
