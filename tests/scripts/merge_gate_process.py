# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the matmul/fused/reduce merge-gate process checks.

Two gates use this module:

* Collection-count: the ``-m merge_gate`` selection under the three family dirs
  must stay at a stored number. A parametrize edit that quietly drops marks
  changes the number.
* Changed-test: a PR that adds or edits ``test_*`` functions under those dirs
  must run those functions, even when they are not merge_gate-marked.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

FAMILY_DIRS = (
    "tests/ttnn/unit_tests/operations/matmul",
    "tests/ttnn/unit_tests/operations/fused",
    "tests/ttnn/unit_tests/operations/reduce",
)

COUNT_FILE = Path("tests/pipeline_reorg/ttnn_matmul_fused_reduction_ops.count")

_COLLECTED_RE = re.compile(r"(\d+)(?:/\d+)? tests collected")
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_TEST_DEF_RE = re.compile(r"^def (test_\w+)\s*\(")
_TOP_LEVEL_DEF_RE = re.compile(r"^(def |class |async def )")


def parse_expected_count(text: str) -> int:
    """First non-comment, non-blank token in a count file is the expected integer."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        return int(line.split()[0])
    raise ValueError("count file has no integer")


def parse_collected_count(pytest_output: str) -> int:
    """Read the collected-test count from pytest --collect-only summary text."""
    matches = _COLLECTED_RE.findall(pytest_output)
    if not matches:
        raise ValueError(f"no 'N tests collected' summary in pytest output:\n{pytest_output[-500:]}")
    return int(matches[-1])


def is_family_test_path(path: str) -> bool:
    if not path.endswith(".py"):
        return False
    name = Path(path).name
    if not name.startswith("test_") or name.endswith(" copy.py"):
        return False
    return any(path == directory or path.startswith(directory + "/") for directory in FAMILY_DIRS)


def _new_file_line_ranges(diff_text: str) -> dict[str, set[int]]:
    """Map each path in a unified diff to the 1-based line numbers of the new file that the hunks touch."""
    ranges: dict[str, set[int]] = {}
    current: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            payload = line[4:]
            current = None if payload == "/dev/null" else payload.removeprefix("b/")
            if current is not None:
                ranges.setdefault(current, set())
            continue
        if current is None:
            continue
        match = _HUNK_RE.match(line)
        if match:
            start = int(match.group(1))
            count = int(match.group(2) or 1)
            if count:
                ranges[current].update(range(start, start + count))
    return ranges


def _test_function_spans(source: str) -> dict[str, range]:
    """Map each top-level test function name to an inclusive 1-based line span.

    Decorators stacked immediately above ``def test_*`` belong to that function.
    The span ends at the line before the next top-level def/class.
    """
    lines = source.splitlines()
    starts: list[tuple[int, str]] = []
    pending_decorator_start: int | None = None
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("@") and not line.startswith((" ", "\t")):
            if pending_decorator_start is None:
                pending_decorator_start = index
            continue
        match = _TEST_DEF_RE.match(line)
        if match:
            start = pending_decorator_start if pending_decorator_start is not None else index
            starts.append((start, match.group(1)))
            pending_decorator_start = None
            continue
        if stripped and not stripped.startswith("#"):
            pending_decorator_start = None

    spans: dict[str, range] = {}
    for position, (start, name) in enumerate(starts):
        if position + 1 < len(starts):
            end = starts[position + 1][0] - 1
        else:
            end = len(lines)
            for later in range(start, len(lines) + 1):
                if later == start:
                    continue
                later_line = lines[later - 1]
                if _TOP_LEVEL_DEF_RE.match(later_line) and not _TEST_DEF_RE.match(later_line):
                    end = later - 1
                    break
        spans[name] = range(start, end + 1)
    return spans


def select_pytest_targets(
    name_status: list[tuple[str, str]],
    diff_text: str,
    file_contents: dict[str, str],
) -> list[str]:
    """Return pytest nodeids (file or file::func) for family tests a diff adds or edits.

    An added test file is selected whole. A helper or import edit in a modified
    file selects the whole file. An edit that only touches a test function (and
    its decorators) selects just that function.
    """
    status_by_path = {path: status for status, path in name_status}
    changed_lines = _new_file_line_ranges(diff_text)
    targets: list[str] = []

    for path, status in status_by_path.items():
        if not is_family_test_path(path) or status == "D":
            continue
        if status == "A" or path not in file_contents:
            targets.append(path)
            continue
        spans = _test_function_spans(file_contents[path])
        touched = changed_lines.get(path, set())
        if not touched:
            targets.append(path)
            continue
        hit_functions = {name for name, span in spans.items() if any(line in span for line in touched)}
        outside_test = any(not any(line in span for span in spans.values()) for line in touched)
        if outside_test or not hit_functions:
            targets.append(path)
        else:
            targets.extend(f"{path}::{name}" for name in sorted(hit_functions))

    return targets


def resolve_diff_base(explicit: str | None = None) -> str:
    """Best-effort merge-base revision for a PR or a local branch."""
    if explicit:
        return explicit
    for key in ("GITHUB_EVENT_PULL_REQUEST_BASE_SHA", "GITHUB_BASE_SHA"):
        value = os.environ.get(key)
        if value:
            return value
    base_ref = os.environ.get("GITHUB_BASE_REF") or "main"
    return base_ref if "/" in base_ref else f"origin/{base_ref}"


def _have_commit(rev: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{rev}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def ensure_revision(rev: str) -> str:
    """Fetch a shallow copy of ``rev`` when this checkout does not have it."""
    if _have_commit(rev):
        return rev
    fetch_ref = rev.removeprefix("origin/")
    subprocess.run(["git", "fetch", "--depth=1", "origin", fetch_ref], check=True)
    if _have_commit(rev):
        return rev
    if _have_commit("FETCH_HEAD"):
        return "FETCH_HEAD"
    raise RuntimeError(f"could not resolve diff base {rev!r} after fetching origin {fetch_ref!r}")


def git_name_status(base: str) -> list[tuple[str, str]]:
    result = subprocess.run(
        ["git", "diff", "--name-status", "-M", f"{base}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        status, path = parts[0][0], parts[-1]
        rows.append((status, path))
    return rows


def git_unified_diff(base: str) -> str:
    result = subprocess.run(
        ["git", "diff", "-U0", "-M", f"{base}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout
