#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Compare workflow run results between current branch and main.

Helps identify if failures are new (introduced by your changes) or pre-existing on main.
This is useful for determining whether CI failures are regressions or flaky/pre-existing issues.

How it works:
    1. Fetches the latest workflow run for each workflow on your branch
    2. Fetches the latest workflow run for the same workflow on main
    3. Compares job-level results between the two runs
    4. Categorizes each job as: new failure, fixed, same failure, or passing

Job categories:
    - NEW FAILURES: Failed on branch, passed on main (regression - needs attention!)
    - FIXED: Passed on branch, failed on main (your changes fixed something)
    - SAME FAILURES: Failed on both (pre-existing issue, not your fault)
    - PASSING: Passed on both branches

Workflows checked (by category):
    - core: all-post-commit-workflows
    - single-card: demo-tests, perf-models, perf-device-models
    - t3000: demo, e2e, fast, integration, perf, perplexity, profiler, unit tests
    - galaxy: apc-fast, deepseek, demo, e2e, frequent, model-perf, multi-user, profiler,
              quick, stress, unit, tg-op-perf tests
    - blackhole: demo, multi-card-demo, multi-card-unit, nightly, post-commit tests

Usage examples:
    # Compare all workflows (default)
    python scripts/compare_workflow_runs.py

    # Compare only T3000 workflows
    python scripts/compare_workflow_runs.py --category t3000

    # Compare specific workflows
    python scripts/compare_workflow_runs.py --workflows t3000-unit-tests.yaml

    # Compare a specific branch
    python scripts/compare_workflow_runs.py --branch my-feature-branch

    # Output as JSON for scripting
    python scripts/compare_workflow_runs.py --json

Prerequisites:
    - gh CLI installed and authenticated (https://cli.github.com/)
    - git (for detecting current branch)
"""

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


def check_prerequisites() -> bool:
    """Check that required tools are installed and configured."""
    errors = []

    # Check gh CLI is installed
    if not shutil.which("gh"):
        errors.append("❌ gh CLI not found. Install from: https://cli.github.com/")
    else:
        # Check gh is authenticated
        result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
        if result.returncode != 0:
            errors.append("❌ gh CLI not authenticated. Run: gh auth login")

    # Check git is installed
    if not shutil.which("git"):
        errors.append("❌ git not found. Please install git.")

    if errors:
        print("Prerequisites check failed:\n", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        print("\n", file=sys.stderr)
        return False

    return True


@dataclass
class JobResult:
    name: str
    conclusion: Optional[str]  # success, failure, cancelled, skipped, None (in_progress)
    status: str  # completed, in_progress, queued


@dataclass
class WorkflowRun:
    id: int
    url: str
    workflow_name: str
    branch: str
    conclusion: Optional[str]
    status: str
    jobs: list[JobResult]


def run_gh_command(args: list[str]) -> Optional[dict | list]:
    """Run a gh CLI command and return parsed JSON, or None on error/empty output."""
    cmd = ["gh"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return None
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def get_run_jobs(run_id: int) -> list[JobResult]:
    """Get all jobs for a workflow run."""
    data = run_gh_command(["run", "view", str(run_id), "--json", "jobs"])
    if not data or not isinstance(data, dict):
        return []
    jobs = data.get("jobs", [])
    return [
        JobResult(
            name=job["name"],
            conclusion=job.get("conclusion"),
            status=job["status"],
        )
        for job in jobs
    ]


def get_latest_run(workflow_file: str, branch: str) -> Optional[WorkflowRun]:
    """Get the latest run of a workflow on a specific branch."""
    data = run_gh_command(
        [
            "run",
            "list",
            "--workflow",
            workflow_file,
            "--branch",
            branch,
            "--limit",
            "1",
            "--json",
            "databaseId,url,conclusion,status,displayTitle,workflowName",
        ]
    )

    if not data or not isinstance(data, list) or len(data) == 0:
        return None

    run = data[0]
    jobs = get_run_jobs(run["databaseId"])

    return WorkflowRun(
        id=run["databaseId"],
        url=run["url"],
        workflow_name=run.get("workflowName", workflow_file),
        branch=branch,
        conclusion=run.get("conclusion"),
        status=run["status"],
        jobs=jobs,
    )


def compare_runs(branch_run: WorkflowRun, main_run: Optional[WorkflowRun]) -> dict:
    """Compare two workflow runs and categorize differences."""
    result = {
        "workflow": branch_run.workflow_name,
        "branch_url": branch_run.url,
        "branch_status": branch_run.status,
        "branch_conclusion": branch_run.conclusion,
        "main_url": main_run.url if main_run else None,
        "main_status": main_run.status if main_run else None,
        "main_conclusion": main_run.conclusion if main_run else None,
        "new_failures": [],  # Failed on branch, passed on main
        "fixed": [],  # Passed on branch, failed on main
        "same_failures": [],  # Failed on both
        "same_success": [],  # Passed on both
        "branch_only": [],  # Jobs only on branch
        "main_only": [],  # Jobs only on main
        "in_progress": [],  # Still running
    }

    if not main_run:
        result["main_only"] = []
        result["branch_only"] = [j.name for j in branch_run.jobs]
        return result

    branch_jobs = {j.name: j for j in branch_run.jobs}
    main_jobs = {j.name: j for j in main_run.jobs}

    all_job_names = set(branch_jobs.keys()) | set(main_jobs.keys())

    for name in all_job_names:
        branch_job = branch_jobs.get(name)
        main_job = main_jobs.get(name)

        if branch_job and not main_job:
            result["branch_only"].append(name)
            continue
        if main_job and not branch_job:
            result["main_only"].append(name)
            continue

        # Both exist
        if branch_job.status != "completed":
            result["in_progress"].append(name)
            continue

        branch_passed = branch_job.conclusion == "success"
        if main_job.conclusion is None:
            main_passed = None
        else:
            main_passed = main_job.conclusion == "success"

        # Handle skipped as "neutral" - not a failure
        if branch_job.conclusion == "skipped" or main_job.conclusion == "skipped":
            result["same_success"].append(name)
            continue

        if main_passed is None:  # Main still in progress
            result["in_progress"].append(name)
        elif branch_passed and main_passed:
            result["same_success"].append(name)
        elif not branch_passed and not main_passed:
            result["same_failures"].append(name)
        elif not branch_passed and main_passed:
            result["new_failures"].append(name)
        elif branch_passed and not main_passed:
            result["fixed"].append(name)

    return result


def print_comparison(comparison: dict):
    """Pretty print a comparison result."""
    print(f"\n{'='*60}")
    print(f"📋 {comparison['workflow']}")
    print(f"{'='*60}")

    # Status summary
    branch_icon = {"success": "✅", "failure": "❌", "cancelled": "⚪", "in_progress": "🔄", None: "🔄"}.get(
        comparison["branch_conclusion"], "❓"
    )

    main_icon = {"success": "✅", "failure": "❌", "cancelled": "⚪", "in_progress": "🔄", None: "🔄"}.get(
        comparison["main_conclusion"], "❓"
    )

    print(f"\n  Branch: {branch_icon} {comparison['branch_conclusion'] or comparison['branch_status']}")
    print(f"    └─ {comparison['branch_url']}")

    if comparison["main_url"]:
        print(f"  Main:   {main_icon} {comparison['main_conclusion'] or comparison['main_status']}")
        print(f"    └─ {comparison['main_url']}")
    else:
        print("  Main:   ⚠️  No matching run found")

    # Job-level details
    if comparison["new_failures"]:
        print(f"\n  🚨 NEW FAILURES (regression - needs attention!):")
        for job in comparison["new_failures"]:
            print(f"     ❌ {job}")

    if comparison["fixed"]:
        print(f"\n  🎉 FIXED (passed on branch, failed on main):")
        for job in comparison["fixed"]:
            print(f"     ✅ {job}")

    if comparison["same_failures"]:
        print(f"\n  ⚠️  SAME FAILURES (pre-existing on main):")
        for job in comparison["same_failures"]:
            print(f"     ❌ {job}")

    if comparison["in_progress"]:
        print(f"\n  🔄 STILL RUNNING:")
        for job in comparison["in_progress"]:
            print(f"     ⏳ {job}")

    # Summary counts
    print(
        f"\n  Summary: {len(comparison['same_success'])} passing, "
        f"{len(comparison['same_failures'])} pre-existing failures, "
        f"{len(comparison['new_failures'])} new failures, "
        f"{len(comparison['in_progress'])} in progress"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare workflow runs between branch and main", epilog="Prerequisites: gh CLI (authenticated), git"
    )
    parser.add_argument("--branch", default=None, help="Branch to compare (default: current branch)")
    parser.add_argument("--workflows", nargs="+", help="Specific workflow files to check")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--category",
        choices=["all", "t3000", "galaxy", "blackhole", "single-card", "core"],
        default="all",
        help="Filter workflows by category",
    )
    args = parser.parse_args()

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Get current branch if not specified
    if args.branch:
        branch = args.branch
    else:
        result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
        branch = result.stdout.strip()
        if result.returncode != 0 or not branch:
            print(
                "❌ Failed to determine current git branch. "
                "You may be in a detached HEAD state or not in a git repository.",
                file=sys.stderr,
            )
            print('   Specify the branch explicitly with "--branch <name>".', file=sys.stderr)
            sys.exit(1)

    print(f"🔍 Comparing workflow runs: {branch} vs main\n")

    # Default workflows to check - all board-level workflows with workflow_dispatch
    workflows = args.workflows or [
        # Core post-commit
        "all-post-commit-workflows.yaml",
        # Single-card
        "single-card-demo-tests.yaml",
        "perf-models.yaml",
        "perf-device-models.yaml",
        # T3000 (T3K)
        "t3000-demo-tests.yaml",
        "t3000-e2e-tests.yaml",
        "t3000-fast-tests.yaml",
        "t3000-integration-tests.yaml",
        "t3000-perf-tests.yaml",
        "t3000-perplexity-tests.yaml",
        "t3000-profiler-tests.yaml",
        "t3000-unit-tests.yaml",
        # Galaxy (TG)
        "galaxy-apc-fast-tests.yaml",
        "galaxy-deepseek-tests.yaml",
        "galaxy-demo-tests.yaml",
        "galaxy-e2e-tests.yaml",
        "galaxy-frequent-tests.yaml",
        "galaxy-perf-tests.yaml",
        "galaxy-multi-user-isolation-tests.yaml",
        "galaxy-profiler-tests.yaml",
        "galaxy-quick.yaml",
        "galaxy-stress-tests.yaml",
        "galaxy-unit-tests.yaml",
        "tg-op-perf-tests.yaml",
        # Blackhole
        "blackhole-demo-tests.yaml",
        "blackhole-multi-card-demo-tests.yaml",
        "blackhole-multi-card-unit-tests.yaml",
        "blackhole-nightly-tests.yaml",
        "blackhole-post-commit.yaml",
    ]

    # Filter by category if specified
    if args.category != "all" and not args.workflows:
        category_filters = {
            "core": ["all-post-commit"],
            "single-card": ["single-card", "perf-models", "perf-device-models"],
            "t3000": ["t3000-"],
            "galaxy": ["galaxy-", "tg-"],
            "blackhole": ["blackhole-"],
        }
        prefixes = category_filters.get(args.category, [])
        workflows = [w for w in workflows if any(w.startswith(p) for p in prefixes)]

    comparisons = []
    skipped_workflows = []
    for workflow in workflows:
        print(f"Fetching {workflow}...", end=" ", flush=True)

        branch_run = get_latest_run(workflow, branch)
        if not branch_run:
            print("skipped (no run on branch)")
            skipped_workflows.append(workflow)
            continue

        main_run = get_latest_run(workflow, "main")
        print("done")

        comparison = compare_runs(branch_run, main_run)
        comparisons.append(comparison)

    if args.json:
        print(json.dumps(comparisons, indent=2))
    else:
        # Print summary header
        total_new_failures = sum(len(c["new_failures"]) for c in comparisons)
        total_same_failures = sum(len(c["same_failures"]) for c in comparisons)
        total_in_progress = sum(len(c["in_progress"]) for c in comparisons)

        print("\n" + "=" * 60)
        print("📊 OVERALL SUMMARY")
        print("=" * 60)

        if total_new_failures > 0:
            print(f"🚨 {total_new_failures} NEW FAILURES - These need investigation!")
        elif total_in_progress > 0:
            print(f"🔄 {total_in_progress} jobs still running - check back later")
        elif total_same_failures > 0:
            print(f"✅ No new failures! ({total_same_failures} pre-existing failures on main)")
        else:
            print("🎉 All passing!")

        # Print each comparison
        for comparison in comparisons:
            print_comparison(comparison)

        # Show skipped workflows
        if skipped_workflows:
            print(f"\n  ℹ️  Skipped {len(skipped_workflows)} workflows (no runs on branch)")

        # Final verdict
        print("\n" + "=" * 60)
        if total_new_failures > 0:
            print("❌ VERDICT: New failures detected - review needed")
            sys.exit(1)
        elif total_in_progress > 0:
            print("⏳ VERDICT: Workflows still running - run again later")
            sys.exit(0)
        else:
            print("✅ VERDICT: Safe to merge (no new failures)")
            sys.exit(0)


if __name__ == "__main__":
    main()
