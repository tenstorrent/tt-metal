#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validate #include directives in C++ source files."""

import re
import sys
from collections import defaultdict
from typing import NamedTuple, Optional

from common import find_cpp_sources


SKIP_FILES = {
    "fabric_edm_packet_header.hpp",
    "dev_msgs.h",
}


ALLOWED_PREFIXES = {
    "hostdevcommon",
    "tt-metalium",
    "tt_stl",
    "umd",
    "fmt",
    "enchantum",
    "nlohmann",
    "tt-logger",
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
    # Inspector RPC auto generated header by capnp
    "inspector_rpc.capnp.h",
}

# Regex patterns
ANGLE_INCLUDE_PATTERN = re.compile(r"^\s*#include\s*<([^>]+)>")
QUOTED_INCLUDE_PATTERN = re.compile(r'^\s*#include\s*"([^"]+)"')


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
        if match := QUOTED_INCLUDE_PATTERN.match(line):
            return Include(source_file, line_num, match.group(1), quoted=True)
        if match := ANGLE_INCLUDE_PATTERN.match(line):
            return Include(source_file, line_num, match.group(1), quoted=False)
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


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        return 1

    directory = sys.argv[1]
    source_files = find_cpp_sources(directory, SKIP_FILES)

    prefix_counts = defaultdict(int)

    def iter_includes(filepath):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                if include := Include.from_line(filepath, line_num, line):
                    yield include

    all_includes = [include for path in source_files for include in iter_includes(path)]
    errors = [err for include in all_includes if (err := include.check_for_errors(prefix_counts)) is not None]
    unused_prefixes = ALLOWED_PREFIXES - prefix_counts.keys()

    for error in errors:
        print(error)

    if unused_prefixes:
        print("\nUnused allowed prefixes (not seen in any #include):")
        for prefix in sorted(unused_prefixes):
            print(f"  - {prefix}")

    print("Done.")

    return 1 if (errors or unused_prefixes) else 0


if __name__ == "__main__":
    sys.exit(main())
