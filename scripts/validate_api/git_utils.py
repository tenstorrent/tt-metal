# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Git operations for API validation scripts."""

import subprocess
import sys
from itertools import groupby
from typing import Dict, List, Optional, Tuple


def run_git(cmd: List[str], cwd: str = ".") -> str:
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git command failed: {' '.join(cmd)}\n{e.stderr}")


def get_blame_timestamps(file_path: str, line_ranges: List[Tuple[int, int]], ref: str = "HEAD") -> Dict[int, int]:
    """Get unix timestamps for when lines were last modified.

    Returns: {line_number: unix_timestamp}
    Example: {10: 1699123456, 11: 1699123456, 20: 1701345678}
    """
    timestamps = {}
    for start, end in line_ranges:
        try:
            output = run_git(["git", "blame", "--porcelain", f"-L{start},{end}", ref, "--", file_path])

            # Parse git blame porcelain format
            # Git blame omits full metadata for consecutive lines from the same commit,
            # so we cache commit timestamps and map lines to commits
            commit_timestamps = {}  # {commit_hash: timestamp}
            line_commits = {}  # {line_number: commit_hash}

            lines = output.split("\n")
            for i, line in enumerate(lines):
                parts = line.split()
                # Check if this is a header line: commit_hash original_line final_line [num_lines]
                if len(parts) >= 3 and parts[1].isdigit():
                    commit_hash = parts[0]
                    line_num = int(parts[2])
                    line_commits[line_num] = commit_hash

                    # Look ahead for committer-time (only present in full entries)
                    for j in range(i + 1, min(i + 15, len(lines))):
                        if lines[j].startswith("committer-time "):
                            commit_timestamps[commit_hash] = int(lines[j].split()[1])
                            break

            # Map line numbers to timestamps using cached commit data
            for line_num, commit_hash in line_commits.items():
                if commit_hash in commit_timestamps:
                    timestamps[line_num] = commit_timestamps[commit_hash]

        except (RuntimeError, ValueError) as e:
            print(f"Warning: git blame failed for {file_path}:{start}-{end}: {e}", file=sys.stderr)
            continue
    return timestamps


def parse_diff_for_removed_lines(diff_output: str, pattern_matcher) -> List[Tuple[str, int, str]]:
    """Parse git diff to find removed lines matching a pattern.

    Returns: [(file_path, line_number, content), ...]
    """
    results = []
    current_file = None
    old_line = 0

    for line in diff_output.split("\n"):
        if line.startswith("--- a/"):
            current_file = line[6:]
        elif line.startswith("@@ "):
            # Parse hunk header: @@ -10,3 +9,0 @@
            parts = line.split()
            if len(parts) >= 2:
                old_range = parts[1][1:]  # Remove '-' prefix
                old_line = int(old_range.split(",")[0] if "," in old_range else old_range)
        elif line.startswith("-") and not line.startswith("---"):
            if current_file and pattern_matcher.search(line[1:]):
                results.append((current_file, old_line, line[1:]))
            old_line += 1
        elif line.startswith(" "):
            old_line += 1

    return results


def get_changed_file_paths(base_ref: str, head_ref: str = "HEAD") -> List[str]:
    """List of changed files (paths) (includes deleted ones and staged changes)."""
    committed = run_git(["git", "diff", "--name-only", base_ref, head_ref])
    staged = run_git(["git", "diff", "--name-only", "--cached"])
    all_files = {f for f in (committed + staged).split("\n") if f.strip()}
    return list(all_files)


def get_diff(files: List[str], base_ref: str, head_ref: str = "HEAD") -> str:
    """Get git diff for specified files between base_ref and head_ref."""
    if not files:
        return ""
    return run_git(["git", "diff", "-U0", base_ref, head_ref, "--"] + files)


def get_staged_diff(files: List[str]) -> str:
    """Get git diff for staged changes in specified files."""
    if not files:
        return ""
    return run_git(["git", "diff", "-U0", "--cached", "--"] + files)


def compute_ranges(numbers: List[int]) -> List[Tuple[int, int]]:
    """Convert numbers to contiguous ranges. [1,2,3,7,8] → [(1,3), (7,8)]"""
    return [
        (min(nums), max(nums))
        for _, nums in [
            (key, [item[1] for item in group])
            for key, group in groupby(enumerate(sorted(numbers)), lambda x: x[1] - x[0])
        ]
    ]


def get_timestamps_for_items(items: List[Tuple[str, int, str]], ref: str) -> Dict[Tuple[str, int], Optional[int]]:
    """Get git blame timestamps for items at ref.

    items: [(file_path, line_number, content), ...]
    Returns: {(file_path, line_number): unix_timestamp, ...}
    """
    # Group by file for efficient git blame operations
    files_to_lines = {}
    for file_path, line_num, _ in items:
        if file_path not in files_to_lines:
            files_to_lines[file_path] = []
        files_to_lines[file_path].append(line_num)

    # Get timestamps for each file
    result = {}
    for file_path, line_numbers in files_to_lines.items():
        timestamps = get_blame_timestamps(file_path, compute_ranges(line_numbers), ref)
        for line_num in line_numbers:
            result[(file_path, line_num)] = timestamps.get(line_num)

    return result
