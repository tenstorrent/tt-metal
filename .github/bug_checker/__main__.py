"""CLI entry point for bug_checker.

Invoked via: python .github/run_bug_checker.py [args]
Or directly if bug_checker is already in sys.modules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bug_checker.github_client import check_prerequisites, fetch_branch_diff, fetch_pr_info
from bug_checker.logger import set_verbose
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
        "--branch",
        nargs="?",
        const="main",
        metavar="BASE",
        help="Analyze local branch diff against BASE (default: main).",
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
        help="Labels to simulate for rule matching (used with --branch).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        set_verbose()

    if not args.pr and args.branch is None:
        parser.error("Either --pr or --branch is required.")

    if args.post_comments and not args.pr:
        parser.error("--post-comments requires --pr.")

    if args.pr:
        check_prerequisites(need_gh=True)
        pr_info = fetch_pr_info(args.pr)
    else:
        check_prerequisites(need_git=True)
        pr_info = fetch_branch_diff(base=args.branch)
        if args.labels:
            pr_info.labels = args.labels

    sarif_path = Path(args.sarif) if args.sarif else None
    findings = run_bug_check(
        pr_info=pr_info,
        sarif_path=sarif_path,
        post_comments=args.post_comments,
    )

    # Exit code: 1 if any blocking findings, 0 otherwise
    has_blocking = any(f.severity == "blocking" for f in findings)
    return 1 if has_blocking else 0


if __name__ == "__main__":
    sys.exit(main())
