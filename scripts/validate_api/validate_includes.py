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
    "dataflow_buffer.hpp",  # TODO: #37324: remove once dataflow buffer has proper host-dev interface
}

# Mapping from tt-metalium experimental tensor headers to their TTNN forward headers.
#
# NOTE: experimental/tensor is a staging area for the tensor lowering effort and is short-lived.
# These headers will eventually move to the main tt_metal/api/tt-metalium folder like all other
# host APIs. TTNN code should use the forward headers to avoid duplicated effort for include
# updates when the final migration happens.
#
# This check can be bypassed by committing with the --no-verify flag if necessary.
EXPERIMENTAL_TENSOR_FORWARD_HEADERS = {
    "tt-metalium/experimental/tensor/tensor_types.hpp": "ttnn/tensor/types.hpp",
    "tt-metalium/experimental/tensor/spec/tensor_spec.hpp": "ttnn/tensor/tensor_spec.hpp",
    "tt-metalium/experimental/tensor/spec/layout/alignment.hpp": "ttnn/tensor/layout/alignment.hpp",
    "tt-metalium/experimental/tensor/spec/layout/layout.hpp": "ttnn/tensor/layout/layout.hpp",
    "tt-metalium/experimental/tensor/spec/layout/page_config.hpp": "ttnn/tensor/layout/page_config.hpp",
    "tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp": "ttnn/tensor/layout/tensor_layout.hpp",
    "tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp": "ttnn/tensor/memory_config/memory_config.hpp",
    "tt-metalium/experimental/tensor/topology/distributed_tensor_configs.hpp": "ttnn/distributed/distributed_configs.hpp",
    "tt-metalium/experimental/tensor/topology/tensor_topology.hpp": "ttnn/distributed/tensor_topology.hpp",
}

# Files excluded from tensor forward header check - these are the forward headers themselves.
# Generated from EXPERIMENTAL_TENSOR_FORWARD_HEADERS values, prefixed with "ttnn/api/".
TENSOR_FORWARD_CHECK_EXCLUDE = {f"ttnn/api/{path}" for path in EXPERIMENTAL_TENSOR_FORWARD_HEADERS.values()}


ALLOWED_PREFIXES = {
    "hostdevcommon",
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

    def check_for_forward_header_advice(self) -> Optional[str]:
        """Check if include should use a TTNN forward header instead of experimental tensor headers."""
        # Skip if this file is itself a forward header (it's allowed to include experimental headers)
        if self.source_file in TENSOR_FORWARD_CHECK_EXCLUDE:
            return None
        if self.path in EXPERIMENTAL_TENSOR_FORWARD_HEADERS:
            forward_header = EXPERIMENTAL_TENSOR_FORWARD_HEADERS[self.path]
            return (
                f"{self.source_file}:{self.line_num}: Use forward header "
                f"<{forward_header}> instead of <{self.path}>"
            )
        return None


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory> [--check-ttnn-forwards]")
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

    # Check TTNN files for direct includes of experimental tensor headers
    # TTNN files are defined as files in ttnn/ and tests/ttnn/ directories
    forward_header_warnings = []
    if "--check-ttnn-forwards" in sys.argv:
        ttnn_directories = ["ttnn", "tests/ttnn"]
        ttnn_files = []
        for ttnn_dir in ttnn_directories:
            ttnn_files.extend(find_cpp_sources(ttnn_dir, SKIP_FILES))
        ttnn_includes = [include for path in ttnn_files for include in iter_includes(path)]
        forward_header_warnings = [
            warn for include in ttnn_includes if (warn := include.check_for_forward_header_advice()) is not None
        ]

        if forward_header_warnings:
            print("\n" + "=" * 80)
            print("WARNING: TTNN Tensor users should use forward headers, not experimental headers")
            print("=" * 80)
            print("")
            print("  The experimental/tensor directory is a temporary staging area for the tensor")
            print("  lowering effort. These headers will eventually move to tt_metal/api/tt-metalium")
            print("  alongside other host APIs.")
            print("")
            print("  To avoid duplicated include updates when the final migration happens,")
            print("  please use the TTNN forward headers listed below.")
            print("")
            print("  To bypass this check: git commit --no-verify")
            print("")
            for warning in forward_header_warnings:
                print(f"  {warning}")
            print("")

    print("Done.")

    return 1 if (errors or unused_prefixes or forward_header_warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
