import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from typing import NamedTuple, Optional

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
        """Return the full #include statement."""
        brackets = '""' if self.quoted else "<>"
        return f"#include {brackets[0]}{self.path}{brackets[1]}"

    @staticmethod
    def from_line(source_file: str, line_num: int, line: str) -> Optional["Include"]:
        if match := QUOTED_INCLUDE_PATTERN.match(line):
            return Include(source_file, line_num, match.group(1), quoted=True)
        if match := ANGLE_INCLUDE_PATTERN.match(line):
            return Include(source_file, line_num, match.group(1), quoted=False)

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


class RemovedLine(NamedTuple):
    """Represents a line that was removed in the current changes."""

    file_path: str
    line_number: int
    content: str
    last_modified_timestamp: Optional[int]
    last_modified_date: Optional[str]

    @property
    def age_days(self) -> Optional[float]:
        """Calculate age in days since last modification."""
        if self.last_modified_timestamp is None:
            return None
        return (datetime.now().timestamp() - self.last_modified_timestamp) / 86400


def get_removed_lines(file_path: str, base_ref: str = "origin/main") -> list[RemovedLine]:
    """
    Get all lines removed from a file between base_ref and HEAD.

    For each removed line, retrieves when it was last modified using git blame
    on the base_ref version of the file.

    Args:
        file_path: Path to the file to analyze
        base_ref: Git reference to compare against (default: "origin/main")

    Returns:
        List of RemovedLine objects containing:
        - file_path: Path to the file
        - line_number: Line number in the base_ref version
        - content: The removed line content
        - last_modified_timestamp: Unix timestamp when line was last modified
        - last_modified_date: Human-readable date string (YYYY-MM-DD HH:MM:SS)
        - age_days: Property that calculates age in days

    Example:
        >>> removed = get_removed_lines("tt_metal/api/tt-metalium/bfloat8.hpp")
        >>> for line in removed:
        ...     if line.age_days and line.age_days < 30:
        ...         print(f"Recently removed: {line.content} (age: {line.age_days:.1f} days)")
    """
    try:
        # Get diff showing removed lines
        result = subprocess.run(
            ["git", "diff", "-U0", base_ref, "HEAD", "--", file_path],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []

    removed_lines = []
    current_old_line = 0

    for line in result.stdout.split("\n"):
        # Parse unified diff hunk header: @@ -old_start,old_count +new_start,new_count @@
        if line.startswith("@@"):
            parts = line.split()
            old_range = parts[1][1:]  # Remove '-' prefix
            if "," in old_range:
                current_old_line = int(old_range.split(",")[0])
            else:
                current_old_line = int(old_range)
        elif line.startswith("-") and not line.startswith("---"):
            # This is a removed line
            content = line[1:]  # Remove '-' prefix

            # Get when this line was last modified using git blame
            timestamp, date_str = _get_line_last_modified(file_path, current_old_line, base_ref)

            removed_lines.append(
                RemovedLine(
                    file_path=file_path,
                    line_number=current_old_line,
                    content=content,
                    last_modified_timestamp=timestamp,
                    last_modified_date=date_str,
                )
            )
            current_old_line += 1
        elif line.startswith(" "):
            # Context line (not removed)
            current_old_line += 1

    return removed_lines


def _get_line_last_modified(file_path: str, line_number: int, base_ref: str) -> tuple[Optional[int], Optional[str]]:
    """
    Get the timestamp when a line was last modified using git blame.

    Args:
        file_path: Path to the file
        line_number: Line number to check
        base_ref: Git reference to check against

    Returns:
        Tuple of (unix_timestamp, formatted_date_string) or (None, None) if unavailable
    """
    try:
        result = subprocess.run(
            ["git", "blame", "-t", "-L", f"{line_number},{line_number}", base_ref, "--", file_path],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse timestamp from git blame output
        # Format: commit_hash file (author timestamp tz line_num) content
        parts = result.stdout.split()
        for part in parts:
            if part.isdigit() and len(part) == 10:  # Unix timestamp
                timestamp = int(part)
                date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                return timestamp, date_str

    except subprocess.CalledProcessError:
        pass

    return None, None


def check_includes_correct(files: list[str]) -> bool:
    """Check includes in files and return True if all are correct."""
    prefix_counts = defaultdict(int)

    def iter_includes(filepath):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                if include := Include.from_line(filepath, line_num, line):
                    yield include

    all_includes = [include for path in files for include in iter_includes(path)]

    # Validate all includes and print errors
    errors = [err for include in all_includes if (err := include.check_for_errors(prefix_counts)) is not None]
    unused_prefixes = ALLOWED_PREFIXES - prefix_counts.keys()

    for error in errors:
        print(error)

    if unused_prefixes:
        print("\nUnused allowed prefixes (not seen in any #include):")
        for prefix in sorted(unused_prefixes):
            print(f"  - {prefix}")

    return not (errors or unused_prefixes)


def main(directory):
    CPP_EXTENSIONS = (".hpp", ".h", ".cpp", ".cc", ".cxx")

    # Find all source files
    source_files = [
        os.path.join(root, fname)
        for root, _, files in os.walk(directory)
        for fname in files
        if fname.endswith(CPP_EXTENSIONS) and fname not in SKIP_FILES
    ]

    includes_ok = check_includes_correct(source_files)

    if not includes_ok:
        print("\nInclude check failed.")
        sys.exit(1)
    else:
        print("All includes are valid and all allowed prefixes are used.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(1)
    main(sys.argv[1])
