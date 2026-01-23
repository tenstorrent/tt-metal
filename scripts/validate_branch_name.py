#!/usr/bin/env python3
"""
Pre-commit hook to validate branch names follow lowercase-only convention.
This ensures compatibility with case-insensitive file systems.
"""

import subprocess
import sys


def get_current_branch():
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def has_uppercase(branch_name):
    """Check if branch name contains uppercase characters."""
    return any(c.isupper() for c in branch_name)


def suggest_lowercase_name(branch_name):
    """Convert branch name to lowercase."""
    return branch_name.lower()


def main():
    branch_name = get_current_branch()

    if not branch_name:
        print("Error: Could not determine current branch name", file=sys.stderr)
        return 1

    # Skip check for special branches
    special_branches = ["HEAD"]
    if branch_name in special_branches:
        return 0

    if has_uppercase(branch_name):
        lowercase_name = suggest_lowercase_name(branch_name)
        print(f"Error: Branch name '{branch_name}' contains uppercase characters.", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"Branch names should use only lowercase characters to ensure compatibility", file=sys.stderr)
        print(f"with case-insensitive file systems.", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"Suggested branch name: {lowercase_name}", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"To rename your branch, run:", file=sys.stderr)
        print(f"  git branch -m {lowercase_name}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
