#!/usr/bin/env python3
"""Scan recent GitHub Actions jobs for runner-failure signatures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from runner_failure_common import (
    JobScanResult,
    RecentJob,
    SIGNATURE_VERSION,
    WorkflowTarget,
    ensure_gh_available,
    format_signature_summary,
    format_utc,
    group_results_by_runner,
    job_state_key,
    job_to_dict,
    list_recent_jobs,
    load_workflows,
    markdown_escape,
    markdown_link,
    parse_github_time,
    result_to_dict,
    runner_name_for_job,
    scan_jobs,
    signature_keys,
    signature_counts,
    write_reports,
)


STATE_SCHEMA_VERSION = 1
STATE_RETENTION_HOURS = 48
NON_FAILED_CONCLUSIONS = {"success", "skipped", "cancelled"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Scan recent jobs from selected GitHub Actions workflows and report " "runner-failure signatures.")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("out/runner_failure_scan_workflows.yaml"),
        help="YAML workflow selector config.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Lookback window in hours (default: 24).",
    )
    parser.add_argument(
        "--state-in",
        type=Path,
        help="Optional checked-job state from a previous workflow artifact.",
    )
    parser.add_argument(
        "--state-out",
        type=Path,
        default=Path("out/runner_failure_state.json"),
        help="Path to write updated checked-job state.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("out/runner_failure_report.json"),
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("out/runner_failure_report.md"),
        help="Path to write the Markdown report.",
    )
    parser.add_argument(
        "--force-fresh",
        action="store_true",
        help="Ignore --state-in and scan all matching failed jobs.",
    )
    parser.add_argument(
        "--gh-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for each gh api request (default: 120).",
    )
    parser.add_argument(
        "--log-workers",
        type=int,
        default=int(os.environ.get("RUNNER_FAILURE_SCAN_LOG_WORKERS", "8")),
        help=(
            "Maximum number of GitHub job logs to scan in parallel " "(default: RUNNER_FAILURE_SCAN_LOG_WORKERS or 8)."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.hours <= 0:
        raise ValueError("--hours must be greater than zero.")
    if args.gh_timeout <= 0:
        raise ValueError("--gh-timeout must be greater than zero.")
    if args.log_workers <= 0:
        raise ValueError("--log-workers must be greater than zero.")


def is_failed_job(job: RecentJob) -> bool:
    conclusion = job.conclusion.lower()
    return bool(conclusion) and conclusion not in NON_FAILED_CONCLUSIONS


def empty_state() -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "signature_version": SIGNATURE_VERSION,
        "signature_keys": signature_keys(),
        "state_retention_hours": STATE_RETENTION_HOURS,
        "checked_jobs": {},
    }


def normalize_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return empty_state()
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        return empty_state()
    if state.get("signature_version") != SIGNATURE_VERSION:
        print("Signature version changed; ignoring previous checked-job state.")
        return empty_state()
    if state.get("signature_keys") != signature_keys():
        print("Signature set changed; ignoring previous checked-job state.")
        return empty_state()
    if not isinstance(state.get("checked_jobs"), dict):
        state["checked_jobs"] = {}
    return state


def load_state(path: Path | None, force_fresh: bool) -> dict[str, Any]:
    if force_fresh or path is None or not path.is_file():
        return empty_state()
    try:
        return normalize_state(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return empty_state()


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def unchecked_jobs(jobs: list[RecentJob], state: dict[str, Any]) -> list[RecentJob]:
    checked_jobs = state.get("checked_jobs", {})
    return [job for job in jobs if job_state_key(job) not in checked_jobs]


def prune_state(
    state: dict[str, Any],
    *,
    now: datetime,
    retention_hours: int = STATE_RETENTION_HOURS,
) -> int:
    checked_jobs = state.get("checked_jobs")
    if not isinstance(checked_jobs, dict):
        state["checked_jobs"] = {}
        return 0

    cutoff = now.astimezone(timezone.utc) - timedelta(hours=retention_hours)
    kept: dict[str, Any] = {}
    pruned_count = 0

    for key, value in checked_jobs.items():
        if not isinstance(value, dict):
            pruned_count += 1
            continue

        checked_at = parse_github_time(str(value.get("checked_at") or ""))
        if checked_at >= cutoff:
            kept[str(key)] = value
        else:
            pruned_count += 1

    state["checked_jobs"] = kept
    state["state_retention_hours"] = retention_hours
    state["pruned_at"] = format_utc(now)
    return pruned_count


def mark_job_checked(state: dict[str, Any], result: JobScanResult) -> None:
    state.setdefault("checked_jobs", {})[job_state_key(result.job)] = {
        "checked_at": format_utc(datetime.now(timezone.utc)),
        "workflow": result.job.workflow,
        "run_id": result.job.run_id,
        "run_attempt": result.job.run_attempt,
        "job_id": result.job.job_id,
        "job_name": result.job.name,
        "runner_name": result.job.runner_name,
        "conclusion": result.job.conclusion,
        "html_url": result.job.html_url,
        "log_checked": result.log_checked,
        "log_status": result.log_status,
        "signatures": list(result.signature_labels),
        "fabric_missing_links": result.fabric_missing_links,
    }


def build_markdown_report(
    *,
    generated_at: datetime,
    since: datetime,
    hours: int,
    workflows: list[WorkflowTarget],
    recent_jobs: list[RecentJob],
    failed_jobs: list[RecentJob],
    jobs_to_scan: list[RecentJob],
    scan_results: list[JobScanResult],
) -> str:
    failures = [result for result in scan_results if result.signature_labels]
    lines = [
        "# Runner Failure Scan",
        "",
        f"- Generated at: `{format_utc(generated_at)}`",
        f"- Window: last `{hours}` hour(s), since `{format_utc(since)}`",
        f"- Workflows: `{', '.join(workflow.name for workflow in workflows)}`",
        f"- Recent selected jobs: `{len(recent_jobs)}`",
        f"- Failed jobs: `{len(failed_jobs)}`",
        f"- Scanned jobs: `{len(jobs_to_scan)}`",
        f"- Runner-failure jobs: `{len(failures)}`",
        "",
    ]

    if not failures:
        lines.append("No runner-failure signatures found in scanned jobs.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Runner Failures",
            "",
            "| Runner | Count | Signatures | Jobs |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for runner, runner_results in sorted(group_results_by_runner(failures).items()):
        jobs_text = "<br>".join(
            markdown_link(
                f"{result.job.workflow} #{result.job.job_id}",
                result.job.html_url,
            )
            for result in runner_results[:10]
        )
        if len(runner_results) > 10:
            jobs_text += f"<br>... {len(runner_results) - 10} more"
        lines.append(
            "| "
            f"{markdown_escape(runner)} | "
            f"{len(runner_results)} | "
            f"{markdown_escape(format_signature_summary(runner_results))} | "
            f"{jobs_text} |"
        )

    lines.extend(["", "## Triggering Jobs", ""])
    for result in failures:
        signature_text = ", ".join(result.signature_labels)
        missing_links_text = f"; missing links: {result.fabric_missing_links}" if result.fabric_missing_links else ""
        lines.append(
            "- "
            f"{markdown_link(result.job.html_url or result.job.job_id, result.job.html_url)} "
            f"`{runner_name_for_job(result.job)}` "
            f"`{result.job.workflow}` "
            f"`{result.job.name}` "
            f"({signature_text}{missing_links_text})"
        )
    return "\n".join(lines) + "\n"


def build_json_report(
    *,
    generated_at: datetime,
    since: datetime,
    hours: int,
    workflows: list[WorkflowTarget],
    recent_jobs: list[RecentJob],
    failed_jobs: list[RecentJob],
    jobs_to_scan: list[RecentJob],
    scan_results: list[JobScanResult],
    runner_log_table_results: list[JobScanResult],
) -> dict[str, Any]:
    failures = [result for result in scan_results if result.signature_labels]
    return {
        "generated_at": format_utc(generated_at),
        "since": format_utc(since),
        "hours": hours,
        "signature_version": SIGNATURE_VERSION,
        "workflows": [
            {
                "name": workflow.name,
                "owner_repo": workflow.owner_repo,
                "workflow_id": workflow.workflow_id,
                "source": workflow.source,
            }
            for workflow in workflows
        ],
        "counts": {
            "recent_jobs": len(recent_jobs),
            "failed_jobs": len(failed_jobs),
            "jobs_to_scan": len(jobs_to_scan),
            "scanned_jobs": len(scan_results),
            "runner_log_table_jobs": len(runner_log_table_results),
            "runner_failure_jobs": len(failures),
        },
        "signature_counts": signature_counts(failures),
        "recent_jobs": [job_to_dict(job) for job in recent_jobs],
        "failed_jobs": [job_to_dict(job) for job in failed_jobs],
        "jobs_to_scan": [job_to_dict(job) for job in jobs_to_scan],
        "runner_failures": {
            runner: [result_to_dict(result) for result in results]
            for runner, results in sorted(group_results_by_runner(failures).items())
        },
        "scan_results": [result_to_dict(result) for result in scan_results],
        "runner_log_table_results": [result_to_dict(result) for result in runner_log_table_results],
    }


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        ensure_gh_available()
        workflows = load_workflows(args.config)
        generated_at = datetime.now(timezone.utc)
        since = generated_at - timedelta(hours=args.hours)
        recent_jobs = list_recent_jobs(workflows, since, gh_timeout=args.gh_timeout)
        state = load_state(args.state_in, force_fresh=args.force_fresh)
        pruned_state_jobs = prune_state(state, now=generated_at)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    failed = [job for job in recent_jobs if is_failed_job(job)]
    jobs_to_scan = unchecked_jobs(failed, state)
    print(f"Recent selected jobs: {len(recent_jobs)} total, {len(failed)} failed, " f"{len(jobs_to_scan)} to scan.")
    print(f"Workflow config: {args.config}")
    if args.state_in:
        state_mode = "ignored" if args.force_fresh else "used if present"
        print(f"State input: {args.state_in} ({state_mode})")
    print(f"State output: {args.state_out}")
    print(f"State retention: {STATE_RETENTION_HOURS}h; " f"pruned {pruned_state_jobs} old checked job(s).")

    scan_results = scan_jobs(
        jobs_to_scan,
        gh_timeout=args.gh_timeout,
        log_workers=args.log_workers,
    )
    for result in scan_results:
        if result.log_checked:
            mark_job_checked(state, result)

    save_state(args.state_out, state)
    failures = [result for result in scan_results if result.signature_labels]

    report_json = build_json_report(
        generated_at=generated_at,
        since=since,
        hours=args.hours,
        workflows=workflows,
        recent_jobs=recent_jobs,
        failed_jobs=failed,
        jobs_to_scan=jobs_to_scan,
        scan_results=scan_results,
        runner_log_table_results=[],
    )
    report_md = build_markdown_report(
        generated_at=generated_at,
        since=since,
        hours=args.hours,
        workflows=workflows,
        recent_jobs=recent_jobs,
        failed_jobs=failed,
        jobs_to_scan=jobs_to_scan,
        scan_results=scan_results,
    )
    write_reports(
        report_json_path=args.report_json,
        report_md_path=args.report_md,
        report_json=report_json,
        report_md=report_md,
    )

    print(f"Runner-failure jobs found: {len(failures)}")
    for runner, runner_results in sorted(group_results_by_runner(failures).items()):
        print(f"runner {runner}: {len(runner_results)} job(s), " f"{format_signature_summary(runner_results)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
