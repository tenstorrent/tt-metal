#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Compare workflow run results between current branch and main.

Helps identify if failures are new (introduced by your changes) or pre-existing on main.
This is useful for determining whether CI failures are regressions or flaky/pre-existing issues.

How it works:
    1. Fetches the latest workflow run for each workflow on your branch
       (or a specific workflow run on your branch, if --branch-run-id is provided)
    2. Fetches the latest completed workflow run for the same workflow on main
    3. Compares job-level results between the two runs
    4. Categorizes each job as: new failure, fixed, same failure, or passing
    5. A failed branch job with no same-named job in main's latest run (job names carry the SKU,
       and workflow_dispatch/schedule runs enable jobs that push runs skip) is looked up in older
       completed main runs (latest few of any event, plus schedule and workflow_dispatch) and
       classified against the newest main run that actually ran that job

Job categories:
    - NEW FAILURES: Failed on branch, passed on main (regression - needs attention!)
    - FIXED: Passed on branch, failed on main (your changes fixed something)
    - SAME FAILURES: Failed on both (pre-existing issue, not your fault)
    - NOT FOUND ON MAIN: Failed on branch, and no recent completed main run ran a job with that name
      (cannot be classified; counts as "review needed")
    - PASSING: Passed on both branches

Workflows checked (by category):
    - core: sanity-tests
    - single-card: demo-tests, perf-models, perf-device-models
    - t3000: demo, e2e, fast, integration, perf, perplexity, profiler, unit tests
    - galaxy: deepseek, deepseek-prefill, demo, e2e, health, integration, model-perf,
              multi-user, profiler, sanity, stress, unit tests
    - blackhole: e2e tests

Usage examples:
    # Compare all workflows (default)
    python scripts/compare_workflow_runs.py

    # Compare only T3000 workflows
    python scripts/compare_workflow_runs.py --category t3000

    # Compare specific workflows
    python scripts/compare_workflow_runs.py --workflows t3000-tests.yaml

    # Compare a specific workflow run ID on your branch
    python scripts/compare_workflow_runs.py --workflows perf-device-models.yaml --branch-run-id 1234567890

    # Compare a specific branch
    python scripts/compare_workflow_runs.py --branch my-feature-branch

    # Output as JSON for scripting (stdout is the JSON array only; progress lines go to stderr)
    python scripts/compare_workflow_runs.py --json

Prerequisites:
    - gh CLI installed and authenticated (https://cli.github.com/)
    - git (for detecting current branch)
    - Run from inside a tt-metal clone, or set GH_REPO=tenstorrent/tt-metal, so gh can find the repository.
      Any failing gh command aborts the script with exit code 1; it is never reported as "no run found".
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

# How many completed main runs to list per query when a failed branch job has no counterpart in
# main's latest run. Jobs are fetched lazily, newest run first, until every such job is resolved.
MAIN_HISTORY_LIMIT_PER_QUERY = 10
MAIN_HISTORY_EVENTS = ("schedule", "workflow_dispatch")


class GhCommandError(RuntimeError):
    """A gh command exited non-zero."""

    def __init__(self, cmd: list[str], stderr: str):
        super().__init__(f"gh command failed: {' '.join(cmd)}\n{stderr.strip()}")
        self.cmd = cmd
        self.stderr = stderr


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
    id: int  # GitHub Actions databaseId (ID of the run)
    url: str
    workflow_name: str
    branch: str
    conclusion: Optional[str]
    status: str
    jobs: list[JobResult]
    workflow_id: Optional[int]  # GitHub Actions workflowDatabaseId (ID of the workflow)
    event: str = ""  # push, schedule, workflow_dispatch, ...
    created_at: str = ""  # ISO 8601 timestamp from the GitHub API


def run_gh_command(args: list[str]) -> Optional[dict | list]:
    """Run a gh CLI command and return parsed JSON, or None on empty output.

    Raises GhCommandError if gh exits non-zero. A gh failure (not a clone, no GH_REPO, auth, network)
    must not look like "no run found", because that would produce a false "Safe to merge" verdict.
    """
    cmd = ["gh"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise GhCommandError(cmd, result.stderr)
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


RUN_LIST_FIELDS = "databaseId,url,conclusion,status,displayTitle,workflowName,workflowDatabaseId,event,createdAt"


def list_runs(workflow_file: str, branch: str, limit: int, completed_only: bool, event: Optional[str] = None) -> list:
    """List runs of a workflow on a branch, newest first, as raw gh JSON dicts."""
    args = ["run", "list", "--workflow", workflow_file, "--branch", branch, "--limit", str(limit), "--json"]
    args.append(RUN_LIST_FIELDS)
    if completed_only:
        args.extend(["--status", "completed"])
    if event:
        args.extend(["--event", event])
    data = run_gh_command(args)
    if not data or not isinstance(data, list):
        return []
    return data


def run_from_gh(run: dict, branch: str, workflow_file: str, jobs: list[JobResult]) -> WorkflowRun:
    return WorkflowRun(
        id=run["databaseId"],
        url=run["url"],
        workflow_name=run.get("workflowName", workflow_file),
        branch=branch,
        conclusion=run.get("conclusion"),
        status=run["status"],
        jobs=jobs,
        workflow_id=run.get("workflowDatabaseId"),
        event=run.get("event", ""),
        created_at=run.get("createdAt", ""),
    )


def get_latest_run(workflow_file: str, branch: str) -> Optional[WorkflowRun]:
    """Get the latest run of a workflow on a specific branch.
    For main, only consider completed runs (skip queued/in_progress).
    """
    data = list_runs(workflow_file, branch, limit=1, completed_only=(branch == "main"))
    if not data:
        return None
    run = data[0]
    return run_from_gh(run, branch, workflow_file, get_run_jobs(run["databaseId"]))


def iter_main_history(workflow_file: str, exclude_run_id: Optional[int]) -> Iterable[WorkflowRun]:
    """Yield older completed main runs of a workflow, newest first, fetching each run's jobs lazily.

    Main's latest completed run is usually a push run with fixed inputs. Jobs enabled only by
    workflow_dispatch inputs or by the schedule (for example the Blackhole multi-card group of
    sanity-tests.yaml) never appear there, so the candidates are the newest completed runs of any
    event plus the newest completed runs of each event in MAIN_HISTORY_EVENTS.
    """
    candidates: dict[int, dict] = {}
    for event in (None,) + MAIN_HISTORY_EVENTS:
        for run in list_runs(workflow_file, "main", MAIN_HISTORY_LIMIT_PER_QUERY, completed_only=True, event=event):
            if run["databaseId"] != exclude_run_id:
                candidates[run["databaseId"]] = run
    for run in sorted(candidates.values(), key=lambda r: r.get("createdAt", ""), reverse=True):
        yield run_from_gh(run, "main", workflow_file, get_run_jobs(run["databaseId"]))


def job_failed(job: JobResult) -> bool:
    """Completed with any conclusion other than success or skipped (same rule as the direct comparison)."""
    return job.status == "completed" and job.conclusion not in ("success", "skipped")


def compare_runs(branch_run: WorkflowRun, main_run: Optional[WorkflowRun]) -> dict:
    """Compare two workflow runs and categorize differences.

    Failed branch jobs with no same-named job in main_run land in "unmatched_failures"; call
    resolve_unmatched_failures() with older main runs to classify them.
    """
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
        "branch_only": [],  # Jobs only on branch (not in main's latest run)
        "main_only": [],  # Jobs only on main
        "in_progress": [],  # Still running
        "unmatched_failures": [],  # Failed on branch, no same-named job on main (yet) - see resolve_unmatched_failures
        "history_baseline": {},  # job name -> older main run used as its baseline (filled by resolve_unmatched_failures)
        "main_history_scanned": 0,  # Older main runs whose jobs were fetched
    }

    branch_jobs = {j.name: j for j in branch_run.jobs}
    main_jobs = {j.name: j for j in main_run.jobs} if main_run else {}

    all_job_names = set(branch_jobs.keys()) | set(main_jobs.keys())

    for name in all_job_names:
        branch_job = branch_jobs.get(name)
        main_job = main_jobs.get(name)

        if branch_job and not main_job:
            result["branch_only"].append(name)
            if branch_job.status != "completed":
                result["in_progress"].append(name)
            elif job_failed(branch_job):
                result["unmatched_failures"].append(name)
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


def resolve_unmatched_failures(comparison: dict, main_history: Iterable[WorkflowRun]) -> None:
    """Classify "unmatched_failures" against older main runs (newest first), in place.

    For each job, the newest main run that ran a job with the same name to a verdict (completed,
    not skipped/cancelled) is its baseline: success there -> new failure, anything else -> same
    failure. Jobs that no scanned run has stay in "unmatched_failures". Stops as soon as every
    job is resolved, so main_history may fetch lazily.
    """
    pending = set(comparison["unmatched_failures"])
    history = iter(main_history)
    # Pull the next run only while something is still unresolved, so a lazy iterator never
    # fetches a run that is not needed.
    while pending and (run := next(history, None)) is not None:
        comparison["main_history_scanned"] += 1
        main_jobs = {j.name: j for j in run.jobs}
        for name in sorted(pending):
            main_job = main_jobs.get(name)
            if (
                main_job is None
                or main_job.status != "completed"
                or main_job.conclusion in (None, "skipped", "cancelled")
            ):
                continue
            pending.discard(name)
            comparison["history_baseline"][name] = {
                "main_url": run.url,
                "main_event": run.event,
                "main_created_at": run.created_at,
                "main_conclusion": main_job.conclusion,
            }
            if main_job.conclusion == "success":
                comparison["new_failures"].append(name)
            else:
                comparison["same_failures"].append(name)
    comparison["unmatched_failures"] = sorted(pending)


def describe_baseline(comparison: dict, job: str) -> str:
    """Suffix for a job whose baseline is an older main run rather than main's latest run."""
    baseline = comparison["history_baseline"].get(job)
    if not baseline:
        return ""
    return (
        f"\n        (not in main's latest run; {baseline['main_conclusion']} on main {baseline['main_event']} run"
        f" {baseline['main_url']} from {baseline['main_created_at']})"
    )


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
    if comparison["main_history_scanned"]:
        print(
            f"  Note:   {len(comparison['history_baseline']) + len(comparison['unmatched_failures'])} failed branch"
            f" job(s) have no same-named job in main's latest run; scanned {comparison['main_history_scanned']}"
            " older completed main run(s) for them"
        )

    if comparison["new_failures"]:
        print(f"\n  🚨 NEW FAILURES (regression - needs attention!):")
        for job in comparison["new_failures"]:
            print(f"     ❌ {job}{describe_baseline(comparison, job)}")

    if comparison["fixed"]:
        print(f"\n  🎉 FIXED (passed on branch, failed on main):")
        for job in comparison["fixed"]:
            print(f"     ✅ {job}")

    if comparison["same_failures"]:
        print(f"\n  ⚠️  SAME FAILURES (pre-existing on main):")
        for job in comparison["same_failures"]:
            print(f"     ❌ {job}{describe_baseline(comparison, job)}")

    if comparison["unmatched_failures"]:
        print("\n  ❓ FAILED ON BRANCH, NOT FOUND ON MAIN (no recent completed main run ran this job):")
        for job in comparison["unmatched_failures"]:
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
        f"{len(comparison['unmatched_failures'])} failures not found on main, "
        f"{len(comparison['in_progress'])} in progress"
    )


def get_run_by_id(run_id: int) -> Optional[WorkflowRun]:
    """Get a specific workflow run by its GitHub run ID."""
    data = run_gh_command(
        [
            "run",
            "view",
            str(run_id),
            "--json",
            "databaseId,url,conclusion,status,workflowName,headBranch,workflowDatabaseId,event,createdAt",
        ]
    )
    if not data or not isinstance(data, dict):
        return None

    db_id = data.get("databaseId")
    if db_id is None:
        return None

    return run_from_gh(data, data.get("headBranch", ""), str(run_id), get_run_jobs(db_id))


def get_workflow_path_by_id(workflow_id: int) -> Optional[str]:
    """
    Return the YAML path for a workflow ID using GitHub REST via gh api.
    Uses the current repository inferred from the local git remote.
    """
    data = run_gh_command(["api", f"repos/:owner/:repo/actions/workflows/{workflow_id}"])
    if not data or not isinstance(data, dict):
        return None
    return data.get("path")


def main():
    try:
        run()
    except GhCommandError as e:
        print(f"\n❌ {e}", file=sys.stderr)
        if "base repo" in e.stderr or "not a git repository" in e.stderr:
            print("   Run from inside a tt-metal clone or set GH_REPO=tenstorrent/tt-metal.", file=sys.stderr)
        print("   No verdict: the comparison could not be completed.", file=sys.stderr)
        sys.exit(1)


def compute_exit_code(comparisons: list[dict]) -> int:
    """Exit status for a set of comparisons; independent of the output format (text or --json).

    1 = new failures, or failed branch jobs not found on main; 0 otherwise (including "still running").
    """
    if any(c["new_failures"] or c["unmatched_failures"] for c in comparisons):
        return 1
    return 0


def run():
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
    parser.add_argument(
        "--branch-run-id",
        type=int,
        help=(
            "Use this workflow run ID on the branch instead of the latest run. "
            "Requires exactly one workflow in --workflows; the run must be for that workflow."
        ),
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

    branch_run_by_id: Optional[WorkflowRun] = None
    if args.branch_run_id is not None:
        if not args.workflows or len(args.workflows) != 1:
            print(
                "❌ Error: --branch-run-id requires exactly one workflow in --workflows.",
                file=sys.stderr,
            )
            sys.exit(1)

        # When using --branch-run-id, fetch the run once and validate it matches the workflow
        branch_run_by_id = get_run_by_id(args.branch_run_id)
        if branch_run_by_id is None:
            print(
                f"❌ Error: No workflow run found with ID {args.branch_run_id}. "
                "Check the ID, repository, and that you're authenticated (gh auth status).",
                file=sys.stderr,
            )
            sys.exit(1)
        if branch_run_by_id.branch != branch:
            print(
                f"❌ Error: Run ID {args.branch_run_id} is from branch '{branch_run_by_id.branch}', "
                f"but you specified branch '{branch}'. Use --branch {branch_run_by_id.branch!r} or a run from {branch!r}.",
                file=sys.stderr,
            )
            sys.exit(1)
        if branch_run_by_id.workflow_id is None:
            print(
                "❌ Error: Could not determine workflow ID for run "
                f"{args.branch_run_id} (missing workflowDatabaseId).",
                file=sys.stderr,
            )
            sys.exit(1)

        yaml_path = get_workflow_path_by_id(branch_run_by_id.workflow_id)
        if yaml_path is None:
            print(
                "❌ Error: Could not resolve workflow YAML path for run "
                f"{args.branch_run_id}, workflow ID {branch_run_by_id.workflow_id}.",
                file=sys.stderr,
            )
            sys.exit(1)

        expected_yaml = os.path.basename(args.workflows[0])
        actual_yaml = os.path.basename(yaml_path)

        if actual_yaml != expected_yaml:
            print(
                "❌ Error: Run ID and workflow file do not match.\n"
                f"    Run ID {args.branch_run_id} → {actual_yaml}\n"
                f"    --workflows value      → {expected_yaml}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Default workflows to check - all board-level workflows with workflow_dispatch
    workflows = args.workflows or [
        # Core post-commit
        "sanity-tests.yaml",
        # Single-card
        "perf-device-models.yaml",
        # T3000 (T3K)
        "t3000-tests.yaml",
        "t3000-dispatch-tests.yaml",
        "t3000-profiler-tests.yaml",
        # Galaxy
        "blaze-models-prefill-tests.yaml",
        "galaxy-tests.yaml",
        "galaxy-sanity.yaml",
        "galaxy-profiler-tests.yaml",
        "galaxy-stress-tests.yaml",
        # Blackhole
        "blackhole-e2e-tests.yaml",
    ]

    # Filter by category if specified
    if args.category != "all" and not args.workflows:
        category_filters = {
            "core": ["sanity-tests"],
            "single-card": ["single-card", "perf-models", "perf-device-models"],
            "t3000": ["t3000-"],
            "galaxy": ["galaxy-"],
            "blackhole": ["blackhole-"],
        }
        prefixes = category_filters.get(args.category, [])
        workflows = [w for w in workflows if any(w.startswith(p) for p in prefixes)]

    # With --json, stdout must be only the JSON array; progress lines go to stderr.
    progress_out = sys.stderr if args.json else sys.stdout
    print(f"🔍 Comparing workflow runs: {branch} vs main\n", file=progress_out)

    comparisons = []
    skipped_workflows = []
    for workflow in workflows:
        print(f"Fetching {workflow}...", end=" ", flush=True, file=progress_out)

        if args.branch_run_id is not None:
            branch_run = branch_run_by_id
        else:
            branch_run = get_latest_run(workflow, branch)
            if not branch_run:
                print("skipped (no run on branch)", file=progress_out)
                skipped_workflows.append(workflow)
                continue

        main_run = get_latest_run(workflow, "main")

        comparison = compare_runs(branch_run, main_run)
        if comparison["unmatched_failures"]:
            print(
                f"{len(comparison['unmatched_failures'])} failed job(s) not in main's latest run, scanning older main runs...",
                end=" ",
                flush=True,
                file=progress_out,
            )
            resolve_unmatched_failures(comparison, iter_main_history(workflow, main_run.id if main_run else None))
        print("done", file=progress_out)
        comparisons.append(comparison)

    if args.json:
        print(json.dumps(comparisons, indent=2))
        sys.exit(compute_exit_code(comparisons))
    else:
        # Print summary header
        total_new_failures = sum(len(c["new_failures"]) for c in comparisons)
        total_same_failures = sum(len(c["same_failures"]) for c in comparisons)
        total_in_progress = sum(len(c["in_progress"]) for c in comparisons)
        total_unmatched = sum(len(c["unmatched_failures"]) for c in comparisons)

        print("\n" + "=" * 60)
        print("📊 OVERALL SUMMARY")
        print("=" * 60)

        if total_new_failures > 0:
            print(f"🚨 {total_new_failures} NEW FAILURES - These need investigation!")
        elif total_unmatched > 0:
            print(f"❓ {total_unmatched} failed jobs not found on main - check them by hand")
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
        elif total_unmatched > 0:
            print("❌ VERDICT: Failed jobs on branch not found in recent main runs - review needed")
        elif total_in_progress > 0:
            print("⏳ VERDICT: Workflows still running - run again later")
        else:
            print("✅ VERDICT: Safe to merge (no new failures)")
        sys.exit(compute_exit_code(comparisons))


if __name__ == "__main__":
    main()
