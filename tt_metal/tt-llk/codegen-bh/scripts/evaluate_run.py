#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate the results of a Blackhole issue solver run.

Checks:
1. Were any files actually changed?
2. How meaningful is the diff?

Usage:
    python scripts/evaluate_run.py --repo-root /path/to/tt-llk
    python scripts/evaluate_run.py --repo-root /path/to/tt-llk --output /tmp/eval.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def get_changed_files(repo_root: Path) -> list[str]:
    """Get files changed between origin/main and HEAD."""
    try:
        proc = subprocess.run(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=10,
        )
        return [f.strip() for f in proc.stdout.strip().splitlines() if f.strip()]
    except Exception:
        return []


def analyze_diff(changed_files: list[str], repo_root: Path) -> dict:
    """Analyze the diff for meaningfulness."""
    if not changed_files:
        return {"status": "no_changes", "score": 0, "summary": "No files changed"}

    try:
        proc = subprocess.run(
            ["git", "diff", "--stat", "origin/main...HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=10,
        )
        stat = proc.stdout.strip()
    except Exception:
        stat = ""

    has_kernel = any("tt_llk_" in f for f in changed_files)
    has_test = any("test" in f.lower() for f in changed_files)

    score = 0
    if has_kernel:
        score += 50
    if has_test:
        score += 30
    if has_kernel and has_test:
        score += 20

    parts = [f"{len(changed_files)} files changed"]
    if has_kernel:
        parts.append("includes kernel modifications")
    if has_test:
        parts.append("includes test changes")
    if not has_kernel and not has_test:
        parts.append("no kernel or test files")
        score = max(score, 10)

    return {
        "status": "evaluated",
        "score": min(score, 100),
        "file_count": len(changed_files),
        "has_kernel_changes": has_kernel,
        "has_test_changes": has_test,
        "summary": ", ".join(parts),
        "stat": stat,
    }


INFRA_PREFIXES = (
    ".claude/",
    "codegen-bh/",
    "codegen/",
    "docs/superpowers/",
    "CLAUDE.md",
)


def evaluate(repo_root: Path) -> dict:
    """Run the full evaluation suite."""
    changed_files = [
        f
        for f in get_changed_files(repo_root)
        if not any(f.startswith(p) or f == p for p in INFRA_PREFIXES)
    ]
    diff_analysis = analyze_diff(changed_files, repo_root)

    if not changed_files:
        overall = "no_changes"
    else:
        overall = "has_changes"

    return {
        "overall": overall,
        "diff_analysis": diff_analysis,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BH issue solver run results")
    parser.add_argument(
        "--repo-root", type=Path, required=True, help="tt-llk repo root"
    )
    parser.add_argument("--output", "-o", type=Path, help="Write JSON result to file")
    args = parser.parse_args()

    result = evaluate(args.repo_root)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))
    return 0 if result["overall"] != "no_changes" else 1


if __name__ == "__main__":
    sys.exit(main())
