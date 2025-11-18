"""Validate deprecated code removals using git history."""

import subprocess
from datetime import datetime
from typing import NamedTuple, Optional


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
