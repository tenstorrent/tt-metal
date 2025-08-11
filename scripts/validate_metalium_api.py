#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import re
import sys
from collections import defaultdict

# List of files or paths (relative or absolute) to skip
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
    # Common system / platform headers
    "cxxabi.h",
    "execinfo.h",
    "dlfcn.h",
    "pthread.h",
    "sys/types.h",
    "sys/stat.h",
    "sys/mman.h",
    "sys/time.h",
    "sys/wait.h",
    "sys/socket.h",
    "netinet/in.h",
    "arpa/inet.h",
    "netdb.h",
    "fcntl.h",
}

# Regex patterns
ANGLE_INCLUDE_PATTERN = re.compile(r"^\s*#include\s*<([^>]+)>")
QUOTED_INCLUDE_PATTERN = re.compile(r'^\s*#include\s*"([^"]+)"')

# Track which prefixes were used
used_prefix_counts = defaultdict(int)


def is_standard_include(include: str) -> bool:
    return "/" not in include and include in STD_HEADERS


def is_valid_include(include: str) -> bool:
    if is_standard_include(include):
        return True
    parts = include.split("/")
    prefix = parts[0]
    if prefix in ALLOWED_PREFIXES:
        used_prefix_counts[prefix] += 1
        return True
    return False


def check_includes_in_file(filepath):
    errors = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            # Disallow quoted includes
            if QUOTED_INCLUDE_PATTERN.match(line):
                include = QUOTED_INCLUDE_PATTERN.match(line).group(1)
                errors.append((line_num, include, "Quoted includes are not allowed. Use angle brackets <...>"))

            # Validate angle-bracket includes
            match = ANGLE_INCLUDE_PATTERN.match(line)
            if match:
                include = match.group(1)
                if not is_valid_include(include):
                    reason = "Invalid or missing namespace prefix"
                    errors.append((line_num, include, reason))
    return errors


def main(directory):
    has_error = False
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith((".hpp", ".h", ".cpp", ".cc", ".cxx")):
                if fname in SKIP_FILES:
                    continue

                path = os.path.join(root, fname)

                errors = check_includes_in_file(path)
                if errors:
                    has_error = True
                    print(f"\nErrors in {path}:")
                    for line_num, include, reason in errors:
                        print(f"  Line {line_num}: {include} — {reason}")

    unused_prefixes = ALLOWED_PREFIXES - used_prefix_counts.keys()
    if unused_prefixes:
        has_error = True
        print("\nUnused allowed prefixes (not seen in any #include):")
        for prefix in sorted(unused_prefixes):
            print(f"  - {prefix}")

    if has_error:
        print("\nInclude check failed.")
        sys.exit(1)
    else:
        print("All includes are valid and all allowed prefixes are used.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(1)
    main(sys.argv[1])
