#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pre-commit hook to verify SPDX license headers are present in source files.

Checks that staged files contain both SPDX-FileCopyrightText and
SPDX-License-Identifier within the first 10 lines. Respects ignore
patterns from .github/spdx_ignore.yaml.
"""

import sys
import os
import re
import fnmatch
import subprocess
from datetime import datetime
from pathlib import Path

CURRENT_YEAR = str(datetime.now().year)

SOURCE_EXTENSIONS = frozenset(
    {
        ".py",
        ".c",
        ".cpp",
        ".cc",
        ".cxx",
        ".h",
        ".hpp",
        ".hxx",
        ".cmake",
        ".sh",
        ".bash",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".rs",
        ".go",
        ".rb",
        ".pl",
        ".r",
    }
)

SOURCE_FILENAMES = frozenset({"CMakeLists.txt", "Makefile", "Dockerfile"})

HEADER_LINES_TO_CHECK = 10

REQUIRED_TAGS = ("SPDX-FileCopyrightText", "SPDX-License-Identifier")

HASH_COMMENT_EXTS = frozenset({".py", ".sh", ".bash", ".cmake", ".rb", ".pl", ".r"})
HASH_COMMENT_NAMES = frozenset({"CMakeLists.txt", "Makefile", "Dockerfile"})
SLASH_COMMENT_EXTS = frozenset(
    {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go"}
)


def make_spdx_header(filepath):
    """Generate the correct SPDX header block for a file based on its extension."""
    name = os.path.basename(filepath)
    _, ext = os.path.splitext(name)
    ext = ext.lower()

    if ext in SLASH_COMMENT_EXTS:
        return (
            f"// SPDX-FileCopyrightText: © {CURRENT_YEAR} Tenstorrent AI ULC\n"
            f"//\n"
            f"// SPDX-License-Identifier: Apache-2.0\n\n"
        )
    else:
        return (
            f"# SPDX-FileCopyrightText: © {CURRENT_YEAR} Tenstorrent AI ULC\n"
            f"#\n"
            f"# SPDX-License-Identifier: Apache-2.0\n\n"
        )


def load_ignore_patterns(repo_root):
    """Load ignore patterns from .github/spdx_ignore.yaml without requiring PyYAML."""
    ignore_file = os.path.join(repo_root, ".github", "spdx_ignore.yaml")
    patterns = []

    if not os.path.isfile(ignore_file):
        return patterns

    in_include = False
    with open(ignore_file, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("include:"):
                in_include = True
                continue
            if in_include:
                if stripped.startswith("- "):
                    pattern = stripped[2:].strip().strip("\"'")
                    patterns.append(pattern)
                elif stripped and not stripped.startswith("#"):
                    in_include = False

    return patterns


def is_ignored(filepath, ignore_patterns):
    """Check if a filepath matches any ignore pattern."""
    while filepath.startswith("./"):
        filepath = filepath[2:]

    parts = filepath.split(os.sep)

    for pattern in ignore_patterns:
        clean = pattern.lstrip("/")

        if clean.startswith("*."):
            if fnmatch.fnmatch(os.path.basename(filepath), clean):
                return True
            continue

        dir_pattern = clean.rstrip("/")

        if filepath.startswith(dir_pattern + "/") or filepath == dir_pattern:
            return True

        if dir_pattern in parts:
            return True

        if fnmatch.fnmatch(filepath, clean):
            return True

    return False


def is_source_file(filepath):
    """Check if a file is a source file that needs an SPDX header."""
    name = os.path.basename(filepath)
    if name in SOURCE_FILENAMES:
        return True
    _, ext = os.path.splitext(name)
    return ext.lower() in SOURCE_EXTENSIONS


def get_newly_added_files(repo_root):
    """Return the set of files that are newly added (not previously tracked by git)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=A"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return set(result.stdout.strip().splitlines())
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return set()


def read_header(filepath):
    """Read the first N lines of a file. Returns the text or None on error."""
    try:
        if os.path.getsize(filepath) == 0:
            return ""
    except OSError:
        return None

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= HEADER_LINES_TO_CHECK:
                    break
                lines.append(line)
            return "".join(lines)
    except (OSError, UnicodeDecodeError):
        return None


YEAR_RE = re.compile(r"(SPDX-FileCopyrightText:.*?)(\d{4})")


def fix_new_file(filepath):
    """
    Auto-fix a new file's SPDX header. Returns True if the file was modified.

    - Missing header entirely: prepend the correct header (preserving shebang).
    - Wrong year: replace with current year.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return False

    head = content[:2048]
    has_copyright = "SPDX-FileCopyrightText" in head
    has_license = "SPDX-License-Identifier" in head

    if has_copyright and has_license:
        match = YEAR_RE.search(head)
        if match and match.group(2) != CURRENT_YEAR:
            new_content = content[: match.start(2)] + CURRENT_YEAR + content[match.end(2) :]
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True
        return False

    header = make_spdx_header(filepath)

    if content.startswith("#!"):
        first_newline = content.index("\n") + 1 if "\n" in content else len(content)
        new_content = content[:first_newline] + "\n" + header + content[first_newline:]
    else:
        new_content = header + content

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    return True


def check_spdx_header(filepath, is_new_file):
    """
    Check SPDX tags in a file's first N lines.
    Returns a list of error strings, empty if all checks pass.

    For all files: both SPDX tags must be present.
    For new files: the year must match the current year.
    """
    head = read_header(filepath)
    if head is None:
        return []

    errors = []

    if head == "":
        errors.extend(f"missing {tag}" for tag in REQUIRED_TAGS)
        if is_new_file:
            errors.append(f"new file must use current year ({CURRENT_YEAR})")
        return errors

    for tag in REQUIRED_TAGS:
        if tag not in head:
            errors.append(f"missing {tag}")

    if is_new_file and "SPDX-FileCopyrightText" in head:
        match = YEAR_RE.search(head)
        if match and match.group(2) != CURRENT_YEAR:
            errors.append(f"new file has year {match.group(2)}, expected {CURRENT_YEAR}")

    return errors


def find_repo_root():
    """Walk up from cwd to find the repo root (contains .github/spdx_ignore.yaml or .git)."""
    path = Path.cwd()
    while path != path.parent:
        if (path / ".git").exists():
            return str(path)
        path = path.parent
    return os.getcwd()


def main():
    repo_root = find_repo_root()
    ignore_patterns = load_ignore_patterns(repo_root)
    newly_added = get_newly_added_files(repo_root)

    files = sys.argv[1:]
    if not files:
        return 0

    fixed_files = []
    failures = []

    for filepath in files:
        rel_path = os.path.relpath(filepath, repo_root) if os.path.isabs(filepath) else filepath

        if is_ignored(rel_path, ignore_patterns):
            continue

        if not is_source_file(rel_path):
            continue

        if not os.path.isfile(filepath):
            continue

        is_new = rel_path in newly_added
        errors = check_spdx_header(filepath, is_new)
        if not errors:
            continue

        if is_new and fix_new_file(filepath):
            fixed_files.append(rel_path)
        else:
            failures.append((rel_path, errors))

    if fixed_files:
        print("SPDX headers auto-fixed in new files:\n")
        for path in fixed_files:
            print(f"  {path}")
        print("\nFiles were modified. Please review the changes and re-commit.")

    if failures:
        print("SPDX license header check failed for the following files:\n")
        for path, errors in failures:
            for error in errors:
                print(f"  {path}: {error}")
        print("\nExpected both of these tags in the first " f"{HEADER_LINES_TO_CHECK} lines:")
        print(f"  SPDX-FileCopyrightText: © {CURRENT_YEAR} Tenstorrent AI ULC")
        print("  SPDX-License-Identifier: Apache-2.0")
        print("\nSee .github/spdx_ignore.yaml for files excluded from this check.")

    if fixed_files or failures:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
