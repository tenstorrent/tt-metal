"""CLI entry point for bug_checker.

Invoked via: python .github/run_bug_checker.py [args]
Or directly if bug_checker is already in sys.modules.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bug_checker.github_client import PRInfo, fetch_pr_info
from bug_checker.orchestrator import run_bug_check


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="bug_checker",
        description="LLM-powered bug pattern checker for tt-metal PRs.",
    )
    parser.add_argument(
        "--pr",
        type=int,
        help="GitHub PR number to analyze.",
    )
    parser.add_argument(
        "--diff",
        type=str,
        help="Path to a local diff file to analyze instead of fetching from GitHub.",
    )
    parser.add_argument(
        "--sarif",
        type=str,
        default=None,
        help="Output path for SARIF file (default: no SARIF output).",
    )
    parser.add_argument(
        "--post-comments",
        action="store_true",
        help="Post findings as PR comments (requires --pr).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=[],
        help="PR labels (used with --diff to simulate label matching).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.pr and not args.diff:
        parser.error("Either --pr or --diff is required.")

    if args.post_comments and not args.pr:
        parser.error("--post-comments requires --pr.")

    if args.pr:
        pr_info = fetch_pr_info(args.pr)
    else:
        diff_path = Path(args.diff)
        if not diff_path.exists():
            print(f"Error: diff file not found: {args.diff}", file=sys.stderr)
            return 1

        diff_text = diff_path.read_text()
        changed_files = _extract_files_from_diff(diff_text)
        pr_info = PRInfo(
            number=0,
            title="Local diff",
            base_sha="",
            head_sha="",
            diff=diff_text,
            changed_files=changed_files,
            labels=args.labels or [],
        )

    sarif_path = Path(args.sarif) if args.sarif else None
    findings = run_bug_check(
        pr_info=pr_info,
        sarif_path=sarif_path,
        post_comments=args.post_comments,
    )

    # Exit code: 1 if any blocking findings, 0 otherwise
    has_blocking = any(f.severity == "blocking" for f in findings)
    return 1 if has_blocking else 0


def _extract_files_from_diff(diff_text: str) -> list[str]:
    """Extract changed file paths from a unified diff."""
    files = []
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:])
    return files


if __name__ == "__main__":
    sys.exit(main())
