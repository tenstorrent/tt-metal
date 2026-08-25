#!/usr/bin/env python3
"""
Analyze what percentage of commits to main trigger the LLK PR gate.

The LLK gate runs when commits modify LLK-specific files (wormhole kernels,
blackhole kernels, common code, SFPI config, unit tests, or CI files).

Usage:
  ./analyze-llk-gate-triggers.py [--since YYYY-MM-DD] [--until YYYY-MM-DD]

Examples:
  # Analyze all commits in July 2026
  ./analyze-llk-gate-triggers.py --since 2026-07-01 --until 2026-08-01

  # Analyze the past 30 days
  ./analyze-llk-gate-triggers.py --since "30 days ago"
"""

import argparse
import re
import subprocess


def get_commits(since=None, until=None):
    """Fetch all commits from main in the given date range."""
    cmd = ["git", "log", "main", "--reverse", "--format=%H"]

    if since:
        cmd.extend(["--since", since])
    if until:
        cmd.extend(["--until", until])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git log failed: {result.stderr}")

    commits = [c.strip() for c in result.stdout.strip().split("\n") if c.strip()]
    return commits


def get_files_changed(commit):
    """Get list of files changed in a commit."""
    result = subprocess.run(["git", "show", commit, "--name-only", "--pretty=format:"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git show {commit} failed: {result.stderr}")

    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    return files


def triggers_llk_gate(files):
    """Check if any file in the list triggers the LLK PR gate."""
    # Patterns derived from .github/scripts/utils/find-changed-files.sh
    patterns = [
        # llk-wormhole-changed
        r"^tt_metal/tt-llk/tt_llk_wormhole_b0/",
        r"^tt_metal/hw/ckernels/wormhole_b0/",
        # llk-blackhole-changed
        r"^tt_metal/tt-llk/tt_llk_blackhole/",
        r"^tt_metal/hw/ckernels/blackhole/",
        # llk-common-changed
        r"^tt_metal/tt-llk/common/",
        # llk-sfpi-changed
        r"^tt_metal/(sfpi-info\.sh|sfpi-version)$",
        # llk-unit-tests-changed
        r"^tests/tt_metal/tt_metal/llk/",
        # llk-ci-changed
        r"^tt_metal/tt-llk/\.github/",
        r"^tt_metal/tt-llk/tests/requirements\.txt$",
        r"^\.github/workflows/llk-.*\.yaml$",
        r"^\.github/workflows/build-quasar-perf\.yml$",
        r"^\.github/scripts/llk-.*\.sh$",
        r"^tests/pipeline_reorg/llk_unit_tests\.yaml$",
        r"^tests/pipeline_reorg/llk_merge_gate_tests\.yaml$",
    ]

    for file in files:
        for pattern in patterns:
            if re.match(pattern, file):
                return True
    return False


def calculate_cascade_effect(commits, triggered_positions):
    """Calculate how many LLK test suite runs occur due to cascade effect.

    When an LLK commit is queued in the merge queue, all subsequent commits
    inherit those LLK changes and run the LLK test suite until the LLK PR
    lands (i.e., until the next LLK commit).

    Args:
        commits: List of all commit hashes
        triggered_positions: List of indices where LLK commits appear

    Returns:
        Total number of LLK test suite runs due to cascade effect
    """
    if not triggered_positions:
        return 0

    total_runs = 0
    for i, llk_pos in enumerate(triggered_positions):
        # Find the next LLK commit (or end of list)
        if i + 1 < len(triggered_positions):
            next_llk_pos = triggered_positions[i + 1]
        else:
            next_llk_pos = len(commits)

        # Commits between this LLK and next LLK (inclusive of this LLK, exclusive of next)
        # All of these run the LLK suite
        run_count = next_llk_pos - llk_pos
        total_runs += run_count

    return total_runs


def main():
    parser = argparse.ArgumentParser(description="Analyze what % of main commits trigger the LLK PR gate")
    parser.add_argument("--since", default="2026-07-01", help="Start date for analysis (default: 2026-07-01)")
    parser.add_argument("--until", default="2026-08-01", help="End date for analysis (default: 2026-08-01)")
    parser.add_argument("--verbose", action="store_true", help="Show details for each triggering commit")
    args = parser.parse_args()

    print(f"Analyzing commits to main from {args.since} to {args.until}...\n")

    commits = get_commits(since=args.since, until=args.until)
    triggered_commits = []
    triggered_positions = []

    for i, commit in enumerate(commits):
        files = get_files_changed(commit)
        if triggers_llk_gate(files):
            triggered_commits.append(commit)
            triggered_positions.append(i)
            if args.verbose:
                # Show which files triggered it
                triggering_files = [f for f in files if triggers_llk_gate([f])]
                print(f"{commit[:7]} - triggered by: {', '.join(triggering_files[:3])}")

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(commits)}...", flush=True)

    # Calculate cascade effect
    cascade_total = calculate_cascade_effect(commits, triggered_positions)

    print()
    print("=" * 70)
    print("LLK PR Gate Trigger Analysis")
    print("=" * 70)
    print(f"Date range:              {args.since} to {args.until}")
    print(f"Total commits:           {len(commits)}")
    print()
    print("Trigger Metric (file-based):")
    print(f"  Commits triggering gate: {len(triggered_commits)}")
    print(f"  Percentage:              {len(triggered_commits) / len(commits) * 100:.1f}%")
    print()
    print("Cascade Metric (merge queue effect):")
    print(f"  LLK suite runs (cascade): {cascade_total}")
    print(f"  Percentage of commits:    {cascade_total / len(commits) * 100:.1f}%")
    print(f"  Average runs per LLK:     {cascade_total / len(triggered_commits) if triggered_commits else 0:.1f}")
    print("=" * 70)
    print()
    print("Interpretation:")
    print(
        f"  • {len(triggered_commits)}/{len(commits)} commits ({len(triggered_commits)/len(commits)*100:.1f}%) directly modified"
    )
    print("    LLK-specific code and triggered the LLK PR gate.")
    print()
    print(f"  • Due to cascade effect in the merge queue, the LLK test suite")
    print(f"    actually runs {cascade_total} times ({cascade_total/len(commits)*100:.1f}% of all commits).")
    print()
    print("  • The cascade effect multiplies the gate trigger count because")
    print(f"    subsequent commits inherit LLK changes while the LLK PR is")
    print(f"    queued, waiting to land on main.")


if __name__ == "__main__":
    main()
