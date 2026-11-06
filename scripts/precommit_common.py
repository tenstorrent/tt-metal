# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the pre-commit checks.

``blank_noncode`` blanks comments, literals and preprocessor lines in place so a
later scan sees only code, while offsets and line numbers stay the same.
``safe_repo_path`` resolves a caller-supplied path and rejects one that leaves
the repository.
``safe_git_rev`` validates a git revision string against an allowlist so option
flags cannot be injected into git commands.
"""

import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Characters a git revision from the environment may contain. Built into a new string so a
# value such as "--output=/tmp/x" never reaches the git command line.
_REV_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./^+~-")


def safe_git_rev(rev):
    """A git revision made only of allowlisted characters, or None."""
    if not rev or len(rev) > 256 or rev[0] in "-." or any(char not in _REV_CHARS for char in rev):
        return None
    return "".join(char for char in rev if char in _REV_CHARS)


def safe_repo_path(path):
    """Absolute path under the repository, or None when it escapes the tree.

    pre-commit passes repo-relative paths, but the path is still caller input.
    """
    if not path:
        return None
    joined = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    resolved = os.path.realpath(joined)
    if resolved != REPO_ROOT and not resolved.startswith(REPO_ROOT + os.sep):
        return None
    return resolved


# u8R"delim( ... )delim" and the other raw-string prefixes.
_RAW_START = re.compile(r'(?:u8|[uUL])?R"([^ ()\\\t\n]{0,16})\(')
# A `'` right after a number literal is a digit separator, not a character literal.
_NUMBER_BEFORE = re.compile(r"(?<![\w.])\d[\w.']*$")


def _line_continuation_end(src, i):
    """End offset of a line starting at ``i``, following trailing ``\\`` continuations."""
    j = i
    n = len(src)
    while j < n:
        k = src.find("\n", j)
        if k < 0:
            return n
        if src[k - 1] == "\\":
            j = k + 1
            continue
        return k
    return n


def blank_noncode(src, *, blank_directives=True):
    """Blank comments, string and character literals, and optionally preprocessor lines.

    Newlines stay, and every other blanked character becomes a space, so a position in
    the result refers to the same position in ``src``. ``blank_directives=False`` leaves
    ``#define`` and other directives visible.
    """
    out, i, n = [], 0, len(src)
    at_line_start = True
    while i < n:
        c = src[i]
        if at_line_start and c in " \t":
            out.append(c)
            i += 1
            continue
        if blank_directives and at_line_start and c == "#":
            j = _line_continuation_end(src, i)
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
            continue
        at_line_start = c == "\n"
        if src.startswith("//", i):
            j = _line_continuation_end(src, i)
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
        elif _RAW_START.match(src, i) and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            match = _RAW_START.match(src, i)
            if not match:
                out.append(c)
                i += 1
                continue
            end = src.find(")" + match.group(1) + '"', match.end())
            j = n if end < 0 else end + len(match.group(1)) + 2
        elif c == "'" and _NUMBER_BEFORE.search(src, max(0, i - 64), i):
            out.append(c)  # digit separator
            i += 1
            continue
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
        else:
            out.append(c)
            i += 1
            continue
        out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
        i = j
    return "".join(out)
