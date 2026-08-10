#!/usr/bin/env python3
"""Build a runner-specific failure report."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from runner_failure_common import (
    JobScanResult,
    RecentJob,
    SIGNATURE_VERSION,
    ensure_gh_available,
    format_signature_summary,
    format_utc,
    gh_api_json,
    job_from_dict,
    job_state_key,
    job_to_dict,
    load_triggering_failures_json,
    markdown_escape,
    markdown_link,
    paginated_items,
    parse_github_time,
    result_to_dict,
    scan_jobs,
    signature_counts,
    write_reports,
)


API_BASE_URL = ""
API_ROUTE = "/api/v1/data_db_main/ci_jobs_by_runner"
AWS_REGION = "us-east-2"
DEFAULT_OWNER_REPO = "tenstorrent/tt-metal"
ACTIVE_RUN_STATUSES = ("in_progress", "queued", "pending", "waiting", "requested")

GITHUB_JOB_LINK_RE = re.compile(
    r"https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/actions/runs/" r"(?P<run_id>\d+)/job/(?P<job_id>\d+)"
)


@dataclass
class LiveGithubPopulationMetrics:
    statuses: tuple[str, ...]
    run_list_api_calls: int = 0
    run_list_api_failures: int = 0
    job_list_api_calls: int = 0
    job_list_api_failures: int = 0
    workflow_runs_seen: int = 0
    workflow_runs_inspected: int = 0
    workflows_inspected: int = 0
    workflow_runs_skipped_stale: int = 0
    jobs_seen: int = 0
    jobs_for_runner: int = 0
    jobs_in_window: int = 0
    jobs_added_to_report: int = 0

    @property
    def api_calls(self) -> int:
        return self.run_list_api_calls + self.job_list_api_calls

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["statuses"] = list(self.statuses)
        value["api_calls"] = self.api_calls
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Build a check-logs-all-style report for one GitHub Actions runner."))
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Lookback window in hours (default: 24).",
    )
    parser.add_argument(
        "--runner-name",
        help="Runner/host name to report on.",
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
        "--triggering-failures-json",
        type=Path,
        help="Optional JSON array of triggering failures from the scan workflow.",
    )
    parser.add_argument(
        "--live-jobs-json",
        type=Path,
        help="Optional JSON cache of live GitHub jobs from the scan workflow.",
    )
    parser.add_argument(
        "--export-live-jobs-json",
        type=Path,
        help="Fetch active GitHub jobs once, write the cache JSON, and exit.",
    )
    parser.add_argument(
        "--run-enrichment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch live GitHub active-run enrichment when no --live-jobs-json is supplied (default: true).",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("RUNNER_FAILURE_API_BASE_URL", API_BASE_URL),
        help=("API Gateway route base URL for data-db-main " "(or RUNNER_FAILURE_API_BASE_URL)."),
    )
    parser.add_argument(
        "--api-route",
        default=os.environ.get("RUNNER_FAILURE_API_ROUTE", API_ROUTE),
        help="API route for runner jobs.",
    )
    parser.add_argument(
        "--aws-region",
        default=os.environ.get("AWS_REGION", AWS_REGION),
        help="AWS region for SigV4 API Gateway signing.",
    )
    parser.add_argument(
        "--api-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for the runner jobs API request (default: 120).",
    )
    parser.add_argument(
        "--owner-repo",
        default=os.environ.get("RUNNER_FAILURE_OWNER_REPO", DEFAULT_OWNER_REPO),
        help="Fallback GitHub owner/repo when the API row has no GitHub URL.",
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
    parser.add_argument(
        "--live-run-workers",
        type=int,
        default=int(os.environ.get("RUNNER_FAILURE_LIVE_RUN_WORKERS", "8")),
        help=(
            "Maximum number of active GitHub workflow runs whose job lists are fetched "
            "in parallel for live enrichment (default: RUNNER_FAILURE_LIVE_RUN_WORKERS or 8)."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.hours <= 0:
        raise ValueError("--hours must be greater than zero.")
    if args.gh_timeout <= 0:
        raise ValueError("--gh-timeout must be greater than zero.")
    if args.live_run_workers <= 0:
        raise ValueError("--live-run-workers must be greater than zero.")
    if not args.owner_repo or not args.owner_repo.strip():
        raise ValueError("--owner-repo is required.")

    if args.export_live_jobs_json:
        return

    if args.api_timeout <= 0:
        raise ValueError("--api-timeout must be greater than zero.")
    if args.log_workers <= 0:
        raise ValueError("--log-workers must be greater than zero.")
    if not args.runner_name or not args.runner_name.strip():
        raise ValueError("--runner-name is required.")
    if not args.api_base_url or not args.api_base_url.strip():
        raise ValueError("--api-base-url or RUNNER_FAILURE_API_BASE_URL is required.")


def build_runner_markdown_report(
    *,
    generated_at: datetime,
    since: datetime,
    hours: int,
    runner_name: str,
    job_source: str,
    runner_jobs: list[RecentJob],
    scan_results: list[JobScanResult],
    triggering_jobs_added: int,
    live_github_metrics: LiveGithubPopulationMetrics,
) -> str:
    failures = [result for result in scan_results if result.signature_labels]
    failure_summary = format_signature_summary(failures) if failures else "none"
    lines = [
        "# Runner Failure Report",
        "",
        f"- Generated at: `{format_utc(generated_at)}`",
        f"- Runner: `{runner_name}`",
        f"- Window: last `{hours}` hour(s), since `{format_utc(since)}`",
        f"- Job source: `{job_source}`",
        f"- Recent jobs on runner: `{len(runner_jobs)}`",
        f"- Triggering jobs added: `{triggering_jobs_added}`",
        f"- Scanned jobs: `{sum(1 for result in scan_results if result.log_checked)}`",
        f"- Runner-failure jobs: `{len(failures)}`",
        f"- Failure summary: `{failure_summary}`",
        (
            "- Live GitHub active-run enrichment: "
            f"`{live_github_metrics.workflow_runs_seen}` workflow run(s) listed, "
            f"`{live_github_metrics.workflow_runs_inspected}` inspected, "
            f"`{live_github_metrics.workflows_inspected}` distinct workflow(s), "
            f"`{live_github_metrics.jobs_seen}` job(s) inspected, "
            f"`{live_github_metrics.jobs_for_runner}` on this runner, "
            f"`{live_github_metrics.jobs_in_window}` in window, "
            f"`{live_github_metrics.jobs_added_to_report}` added; "
            f"`{live_github_metrics.api_calls}` API call(s) "
            f"({live_github_metrics.run_list_api_calls} run-list + "
            f"{live_github_metrics.job_list_api_calls} job-list, "
            f"{live_github_metrics.run_list_api_failures} run-list failure(s), "
            f"{live_github_metrics.job_list_api_failures} job-list failure(s))."
        ),
        "",
    ]

    if not scan_results:
        lines.append("No jobs found for this runner.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| GH job | Failure signatures | Workflow | Job name | Status | Started | Log checked |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in scan_results:
        signatures = ", ".join(result.signature_labels)
        if result.fabric_missing_links:
            signatures += f"; missing links: {result.fabric_missing_links}"
        lines.append(
            "| "
            f"{markdown_link(result.job.job_id or 'open', result.job.html_url)} | "
            f"{markdown_escape(signatures)} | "
            f"{markdown_escape(result.job.workflow)} | "
            f"{markdown_escape(result.job.name)} | "
            f"{markdown_escape(result.job.conclusion or result.job.status or 'unknown')} | "
            f"{markdown_escape(result.job.started_at)} | "
            f"{'YES' if result.log_checked else 'NO'} |"
        )
    return "\n".join(lines) + "\n"


def build_runner_json_report(
    *,
    generated_at: datetime,
    since: datetime,
    hours: int,
    runner_name: str,
    job_source: str,
    runner_jobs: list[RecentJob],
    scan_results: list[JobScanResult],
    triggering_jobs: list[JobScanResult],
    triggering_jobs_added: int,
    live_github_metrics: LiveGithubPopulationMetrics,
) -> dict[str, Any]:
    failures = [result for result in scan_results if result.signature_labels]
    return {
        "type": "runner_report",
        "generated_at": format_utc(generated_at),
        "since": format_utc(since),
        "hours": hours,
        "runner_name": runner_name,
        "signature_version": SIGNATURE_VERSION,
        "job_source": job_source,
        "workflows": [],
        "counts": {
            "runner_jobs": len(runner_jobs),
            "triggering_jobs": len(triggering_jobs),
            "triggering_jobs_added": triggering_jobs_added,
            "live_github_jobs_added": live_github_metrics.jobs_added_to_report,
            "scanned_jobs": len(scan_results),
            "log_checked_jobs": sum(1 for result in scan_results if result.log_checked),
            "runner_failure_jobs": len(failures),
        },
        "live_github_population": live_github_metrics.to_dict(),
        "signature_counts": signature_counts(failures),
        "recent_jobs": [job_to_dict(job) for job in runner_jobs],
        "triggering_jobs": [result_to_dict(result) for result in triggering_jobs],
        "scan_results": [result_to_dict(result) for result in scan_results],
        "runner_log_table_results": [result_to_dict(result) for result in scan_results],
    }


def api_url(base_url: str, route: str) -> str:
    return f"{base_url.rstrip('/')}/{route.lstrip('/')}"


def dict_records_from_list(values: list[Any]) -> list[dict[str, Any]]:
    return [item for item in values if isinstance(item, dict)]


def log_normalized_jobs(jobs: list[RecentJob]) -> None:
    with_job_ids = sum(1 for job in jobs if job.job_id)
    with_links = sum(1 for job in jobs if job.html_url)
    print(
        "Normalized runner jobs: "
        f"{len(jobs)} total, {with_job_ids} with GitHub job id(s), "
        f"{with_links} with GitHub link(s)."
    )


def response_api_call_count(response: Any) -> int:
    if isinstance(response, list):
        return max(1, len(response))
    return 1


def live_github_metrics_from_dict(value: Any) -> LiveGithubPopulationMetrics:
    if not isinstance(value, dict):
        return LiveGithubPopulationMetrics(statuses=ACTIVE_RUN_STATUSES)

    raw_statuses = value.get("statuses")
    statuses = (
        tuple(str(status) for status in raw_statuses if status)
        if isinstance(raw_statuses, list)
        else ACTIVE_RUN_STATUSES
    )
    metrics = LiveGithubPopulationMetrics(statuses=statuses)
    for field_name in (
        "run_list_api_calls",
        "run_list_api_failures",
        "job_list_api_calls",
        "job_list_api_failures",
        "workflow_runs_seen",
        "workflow_runs_inspected",
        "workflows_inspected",
        "workflow_runs_skipped_stale",
        "jobs_seen",
        "jobs_for_runner",
        "jobs_in_window",
        "jobs_added_to_report",
    ):
        try:
            setattr(metrics, field_name, int(value.get(field_name) or 0))
        except (TypeError, ValueError):
            setattr(metrics, field_name, 0)
    return metrics


def api_get_json(
    *,
    base_url: str,
    route: str,
    params: dict[str, str],
    region: str,
    timeout: int,
) -> Any:
    try:
        import boto3
        import requests
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install boto3, botocore, and requests.") from exc

    url = api_url(base_url, route)
    print(
        "Runner jobs API params: "
        f"runner_name={params.get('runner_name')!r}, "
        f"date_start={params.get('date_start')!r}, "
        f"date_end={params.get('date_end')!r}."
    )
    request = AWSRequest(method="GET", url=url, params=params)
    credentials = boto3.Session().get_credentials()
    if credentials is None:
        raise RuntimeError("Unable to find AWS credentials for API Gateway request.")

    SigV4Auth(credentials, "execute-api", region).add_auth(request)
    response = requests.get(
        url,
        params=params,
        headers=dict(request.headers),
        timeout=timeout,
    )
    content_type = response.headers.get("content-type", "unknown")
    print(
        "Runner jobs API response: "
        f"HTTP {response.status_code}, {len(response.content)} byte(s), "
        f"content-type={content_type}."
    )
    if not response.ok:
        raise RuntimeError(
            "Runner jobs API failed with "
            f"HTTP {response.status_code} ({len(response.content)} byte(s), "
            f"content-type={content_type})."
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Runner jobs API returned non-JSON "
            f"(HTTP {response.status_code}, {len(response.content)} byte(s), "
            f"content-type={content_type})."
        ) from exc
    return payload


def api_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("body"), str):
        try:
            return api_records(json.loads(payload["body"]))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Runner jobs API response body was not JSON.") from exc

    if isinstance(payload, list):
        return dict_records_from_list(payload)

    if not isinstance(payload, dict):
        raise RuntimeError("Runner jobs API response must be a JSON object or list.")

    for key in ("data", "rows", "items", "results", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return dict_records_from_list(value)

    if all(isinstance(value, dict) for value in payload.values()):
        return dict_records_from_list(list(payload.values()))

    raise RuntimeError(
        "Runner jobs API response did not contain a supported records list "
        "(expected one of data, rows, items, results, records)."
    )


def row_value(row: dict[str, Any], *field_names: str) -> Any:
    values_by_name = {str(key).casefold(): value for key, value in row.items()}
    for field_name in field_names:
        value = values_by_name.get(field_name.casefold())
        if value is not None:
            return value
    return None


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def parse_github_job_link(value: Any) -> tuple[str, str, str] | None:
    link = clean_string(value)
    if not link:
        return None
    match = GITHUB_JOB_LINK_RE.search(link)
    if not match:
        return None
    return (
        f"{match.group('owner')}/{match.group('repo')}",
        match.group("run_id"),
        match.group("job_id"),
    )


def normalize_timestamp(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return format_utc(value)

    text = clean_string(value)
    if not text:
        return ""

    iso_text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return format_utc(parsed)
    except ValueError:
        pass

    for date_format in (
        "%Y-%m-%d %H:%M:%S.%f %z",
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            parsed = datetime.strptime(text, date_format)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return format_utc(parsed)
        except ValueError:
            continue

    return text


def normalize_conclusion(row: dict[str, Any]) -> str:
    explicit_conclusion = clean_string(row_value(row, "CONCLUSION", "JOB_CONCLUSION", "conclusion")).lower()
    if explicit_conclusion:
        return explicit_conclusion

    job_success = row_value(row, "JOB_SUCCESS", "job_success", "success")
    if job_success is not None:
        if isinstance(job_success, bool):
            return "success" if job_success else "failure"
        normalized = clean_string(job_success).lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return "success"
        if normalized in {"0", "false", "f", "no", "n"}:
            return "failure"

    status = clean_string(row_value(row, "JOB_STATUS", "status", "job_status")).lower()
    if status in {
        "success",
        "failure",
        "failed",
        "cancelled",
        "canceled",
        "skipped",
        "timed_out",
        "timed out",
    }:
        if status == "failed":
            return "failure"
        if status == "canceled":
            return "cancelled"
        if status == "timed out":
            return "timed_out"
        return status
    if "fail" in status:
        return "failure"
    if "success" in status or "pass" in status:
        return "success"
    return status


def recent_job_from_runner_api_row(
    row: dict[str, Any], *, fallback_runner_name: str, fallback_owner_repo: str
) -> RecentJob:
    github_link = row_value(row, "GITHUB_JOB_LINK", "github_job_link", "html_url", "url")
    github_ref = parse_github_job_link(github_link)

    owner_repo = clean_string(row_value(row, "OWNER_REPO", "owner_repo", "REPOSITORY", "repository"))
    run_id = clean_string(row_value(row, "GITHUB_RUN_ID", "github_run_id", "run_id"))
    job_id = clean_string(row_value(row, "GITHUB_JOB_ID", "github_job_id", "job_id"))
    if github_ref is not None:
        owner_repo, run_id, job_id = github_ref
    if not owner_repo:
        owner_repo = fallback_owner_repo

    html_url = clean_string(github_link)
    if not html_url and owner_repo and run_id and job_id:
        html_url = f"https://github.com/{owner_repo}/actions/runs/{run_id}/job/{job_id}"

    workflow = clean_string(
        row_value(
            row,
            "WORKFLOW",
            "WORKFLOW_NAME",
            "workflow",
            "workflow_name",
            "PIPELINE",
            "PIPELINE_NAME",
            "pipeline",
            "pipeline_name",
        )
    )
    workflow_id = clean_string(row_value(row, "WORKFLOW_ID", "workflow_id"))
    job_name = clean_string(row_value(row, "JOB_NAME", "NAME", "job_name", "name"))

    return RecentJob(
        owner_repo=owner_repo,
        workflow=workflow,
        workflow_id=workflow_id,
        run_id=run_id,
        run_attempt=clean_string(row_value(row, "RUN_ATTEMPT", "run_attempt")),
        run_url=(f"https://github.com/{owner_repo}/actions/runs/{run_id}" if owner_repo and run_id else ""),
        job_id=job_id,
        name=job_name,
        runner_name=clean_string(row_value(row, "RUNNER", "RUNNER_NAME", "runner", "runner_name", "HOST_NAME"))
        or fallback_runner_name,
        status=clean_string(row_value(row, "STATUS", "JOB_STATUS", "status", "job_status"))
        or ("completed" if normalize_conclusion(row) else ""),
        conclusion=normalize_conclusion(row),
        html_url=html_url,
        started_at=normalize_timestamp(row_value(row, "JOB_START_TS", "STARTED_AT", "started_at", "start_time")),
        completed_at=normalize_timestamp(row_value(row, "JOB_END_TS", "COMPLETED_AT", "completed_at", "end_time")),
    )


def active_workflow_runs_endpoint(owner_repo: str, status: str) -> str:
    query = urlencode({"status": status, "per_page": "100"})
    return f"repos/{owner_repo}/actions/runs?{query}"


def live_workflow_identity(run: dict[str, Any]) -> str:
    return str(
        run.get("workflow_id")
        or run.get("path")
        or run.get("workflow_url")
        or run.get("name")
        or run.get("display_title")
        or ""
    )


def job_touched_window(job: RecentJob, *, since: datetime, until: datetime) -> bool:
    started_at = parse_github_time(job.started_at)
    completed_at = parse_github_time(job.completed_at)

    if since <= completed_at <= until:
        return True
    if since <= started_at <= until:
        return True
    return job.status == "in_progress" and started_at <= until and not job.completed_at


def recent_job_from_live_github_api(
    *,
    owner_repo: str,
    run: dict[str, Any],
    job: dict[str, Any],
) -> RecentJob:
    run_id = str(run.get("id") or "")
    job_id = str(job.get("id") or "")
    return RecentJob(
        owner_repo=owner_repo,
        workflow=str(run.get("name") or run.get("display_title") or run.get("path") or ""),
        workflow_id=str(run.get("path") or ""),
        run_id=run_id,
        run_attempt=str(run.get("run_attempt") or ""),
        run_url=str(run.get("html_url") or ""),
        job_id=job_id,
        name=str(job.get("name") or ""),
        runner_name=str(job.get("runner_name") or ""),
        status=str(job.get("status") or ""),
        conclusion=str(job.get("conclusion") or ""),
        html_url=str(
            job.get("html_url")
            or (f"https://github.com/{owner_repo}/actions/runs/{run_id}/job/{job_id}" if run_id and job_id else "")
        ),
        started_at=str(job.get("started_at") or ""),
        completed_at=str(job.get("completed_at") or ""),
    )


def fetch_live_run_jobs(
    *,
    owner_repo: str,
    run: dict[str, Any],
    since: datetime,
    until: datetime,
    gh_timeout: int,
) -> tuple[list[RecentJob], int, int]:
    run_id = str(run.get("id") or "")
    if not run_id:
        return [], 0, 0

    response = gh_api_json(
        f"repos/{owner_repo}/actions/runs/{run_id}/jobs?{urlencode({'filter': 'latest', 'per_page': '100'})}",
        paginate=True,
        timeout=gh_timeout,
    )
    api_calls = response_api_call_count(response)
    jobs = paginated_items(response, "jobs")

    live_jobs: list[RecentJob] = []
    for job in jobs:
        live_job = recent_job_from_live_github_api(owner_repo=owner_repo, run=run, job=job)
        if not live_job.runner_name or not job_touched_window(live_job, since=since, until=until):
            continue
        live_jobs.append(live_job)

    return live_jobs, api_calls, len(jobs)


def list_live_jobs_from_github(
    *,
    since: datetime,
    until: datetime,
    owner_repo: str,
    gh_timeout: int,
    live_run_workers: int,
) -> tuple[list[RecentJob], LiveGithubPopulationMetrics]:
    metrics = LiveGithubPopulationMetrics(statuses=ACTIVE_RUN_STATUSES)
    candidate_runs: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()

    for status in ACTIVE_RUN_STATUSES:
        try:
            response = gh_api_json(
                active_workflow_runs_endpoint(owner_repo, status),
                paginate=True,
                timeout=gh_timeout,
            )
        except Exception as exc:
            metrics.run_list_api_calls += 1
            metrics.run_list_api_failures += 1
            print(f"warning: live GitHub run-list lookup failed for status {status!r}: {exc}", file=sys.stderr)
            continue
        metrics.run_list_api_calls += response_api_call_count(response)
        runs = paginated_items(response, "workflow_runs")
        metrics.workflow_runs_seen += len(runs)
        for run in runs:
            run_id = str(run.get("id") or "")
            if not run_id or run_id in seen_run_ids:
                continue
            seen_run_ids.add(run_id)
            candidate_runs.append(run)

    metrics.workflow_runs_inspected = len(candidate_runs)
    metrics.workflows_inspected = len({identity for run in candidate_runs if (identity := live_workflow_identity(run))})
    if not candidate_runs:
        print("Live GitHub enrichment: no active workflow run(s) found to inspect for runner jobs.")
        return [], metrics

    print(
        "Live GitHub enrichment: "
        f"inspecting {len(candidate_runs)} active workflow run(s) with "
        f"{min(live_run_workers, len(candidate_runs))} worker(s)."
    )

    live_jobs: list[RecentJob] = []
    with ThreadPoolExecutor(max_workers=min(live_run_workers, len(candidate_runs))) as executor:
        futures = [
            executor.submit(
                fetch_live_run_jobs,
                owner_repo=owner_repo,
                run=run,
                since=since,
                until=until,
                gh_timeout=gh_timeout,
            )
            for run in candidate_runs
        ]
        for future in as_completed(futures):
            try:
                run_jobs, api_calls, jobs_seen = future.result()
            except Exception as exc:
                metrics.job_list_api_calls += 1
                metrics.job_list_api_failures += 1
                print(f"warning: live GitHub job-list lookup failed: {exc}", file=sys.stderr)
                continue
            metrics.job_list_api_calls += api_calls
            metrics.jobs_seen += jobs_seen
            metrics.jobs_in_window += len(run_jobs)
            live_jobs.extend(run_jobs)

    print(
        "Live GitHub enrichment: "
        f"{metrics.jobs_in_window} runner-assigned job(s) in window from "
        f"{metrics.jobs_seen} inspected job(s); "
        f"{metrics.api_calls} API call(s), "
        f"{metrics.run_list_api_failures} run-list failure(s), "
        f"{metrics.job_list_api_failures} job-list failure(s)."
    )
    return sorted(live_jobs, key=lambda job: parse_github_time(job.started_at), reverse=True), metrics


def filter_live_jobs_for_runner(
    live_jobs: list[RecentJob], *, runner_name: str, since: datetime, until: datetime
) -> list[RecentJob]:
    expected_runner = runner_name.casefold()
    return [
        job
        for job in live_jobs
        if job.runner_name.casefold() == expected_runner and job_touched_window(job, since=since, until=until)
    ]


def load_live_jobs_json(path: Path) -> tuple[list[RecentJob], LiveGithubPopulationMetrics]:
    try:
        raw_value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Unable to read live jobs JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unable to parse live jobs JSON {path}: {exc}") from exc

    metrics = LiveGithubPopulationMetrics(statuses=ACTIVE_RUN_STATUSES)
    raw_jobs: Any
    if isinstance(raw_value, list):
        raw_jobs = raw_value
    elif isinstance(raw_value, dict):
        metrics = live_github_metrics_from_dict(raw_value.get("metrics"))
        raw_jobs = raw_value.get("jobs")
    else:
        raise RuntimeError(f"Live jobs JSON {path} must contain a list or object.")

    if not isinstance(raw_jobs, list):
        raise RuntimeError(f"Live jobs JSON {path} must contain jobs list.")

    jobs = [job_from_dict(value) for value in raw_jobs if isinstance(value, dict)]
    if not metrics.jobs_seen:
        metrics.jobs_seen = len(jobs)
    print(f"Loaded {len(jobs)} precomputed live GitHub job(s) from {path}.")
    return jobs, metrics


def write_live_jobs_json(
    *,
    path: Path,
    generated_at: datetime,
    since: datetime,
    until: datetime,
    owner_repo: str,
    live_jobs: list[RecentJob],
    metrics: LiveGithubPopulationMetrics,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "generated_at": format_utc(generated_at),
                "since": format_utc(since),
                "until": format_utc(until),
                "owner_repo": owner_repo,
                "metrics": metrics.to_dict(),
                "jobs": [job_to_dict(job) for job in live_jobs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def export_live_github_jobs(args: argparse.Namespace) -> int:
    ensure_gh_available()
    generated_at = datetime.now(timezone.utc)
    since = generated_at - timedelta(hours=args.hours)
    live_jobs, metrics = list_live_jobs_from_github(
        since=since,
        until=generated_at,
        owner_repo=args.owner_repo,
        gh_timeout=args.gh_timeout,
        live_run_workers=args.live_run_workers,
    )
    write_live_jobs_json(
        path=args.export_live_jobs_json,
        generated_at=generated_at,
        since=since,
        until=generated_at,
        owner_repo=args.owner_repo,
        live_jobs=live_jobs,
        metrics=metrics,
    )
    print(f"Wrote {len(live_jobs)} live GitHub job(s) to {args.export_live_jobs_json}.")
    return 0


def triggering_results_for_runner(results: list[JobScanResult], runner_name: str) -> list[JobScanResult]:
    expected_runner = runner_name.casefold()
    return [result for result in results if result.job.runner_name.casefold() == expected_runner]


def merge_missing_runner_jobs(
    runner_jobs: list[RecentJob], jobs_to_add: list[RecentJob]
) -> tuple[list[RecentJob], int]:
    merged_jobs = list(runner_jobs)
    seen_job_keys = {job_state_key(job) for job in runner_jobs if job.job_id}
    added_jobs = 0

    for job in jobs_to_add:
        if not job.job_id:
            continue

        key = job_state_key(job)
        if key in seen_job_keys:
            continue

        seen_job_keys.add(key)
        merged_jobs.append(job)
        added_jobs += 1

    return (
        sorted(
            merged_jobs,
            key=lambda runner_job: parse_github_time(runner_job.started_at),
            reverse=True,
        ),
        added_jobs,
    )


def merge_triggering_jobs(
    runner_jobs: list[RecentJob], triggering_results: list[JobScanResult]
) -> tuple[list[RecentJob], int]:
    return merge_missing_runner_jobs(runner_jobs, [result.job for result in triggering_results])


def list_runner_jobs_from_api(
    *,
    runner_name: str,
    since: datetime,
    until: datetime,
    base_url: str,
    route: str,
    region: str,
    timeout: int,
    owner_repo: str,
) -> list[RecentJob]:
    params = {
        "runner_name": runner_name,
        "date_start": format_utc(since),
        "date_end": format_utc(until),
    }
    payload = api_get_json(
        base_url=base_url,
        route=route,
        params=params,
        region=region,
        timeout=timeout,
    )
    records = api_records(payload)
    print(f"Runner jobs API record count: {len(records)}.")
    jobs = [
        recent_job_from_runner_api_row(
            row,
            fallback_runner_name=runner_name,
            fallback_owner_repo=owner_repo,
        )
        for row in records
    ]
    sorted_jobs = sorted(
        jobs,
        key=lambda runner_job: parse_github_time(runner_job.started_at),
        reverse=True,
    )
    log_normalized_jobs(sorted_jobs)
    return sorted_jobs


def should_scan_log_for_full_table(job: RecentJob) -> bool:
    conclusion = job.conclusion.lower()
    return bool(job.job_id) and conclusion not in {"skipped", "cancelled"}


def runner_report_results(
    *,
    runner_jobs: list[RecentJob],
    gh_timeout: int,
    log_workers: int,
    known_results: list[JobScanResult] | None = None,
) -> list[JobScanResult]:
    checked_results_by_key: dict[str, JobScanResult] = {
        job_state_key(result.job): result for result in known_results or [] if result.job.job_id and result.log_checked
    }
    jobs_to_scan = [
        job
        for job in runner_jobs
        if should_scan_log_for_full_table(job) and job_state_key(job) not in checked_results_by_key
    ]

    if jobs_to_scan:
        for result in scan_jobs(
            jobs_to_scan,
            gh_timeout=gh_timeout,
            log_workers=log_workers,
        ):
            checked_results_by_key[job_state_key(result.job)] = result

    ordered_results: list[JobScanResult] = []
    for job in sorted(
        runner_jobs,
        key=lambda runner_job: parse_github_time(runner_job.started_at),
        reverse=True,
    ):
        if job.job_id:
            result = checked_results_by_key.get(job_state_key(job))
            if result is not None:
                ordered_results.append(result)
                continue

        conclusion = job.conclusion or job.status or "unknown"
        ordered_results.append(
            JobScanResult(
                job=job,
                log_status=f"not checked: {conclusion}",
                log_checked=False,
                signature_labels=(),
                fabric_missing_links="",
            )
        )

    return ordered_results


def build_runner_report(args: argparse.Namespace) -> int:
    runner_name = args.runner_name.strip()
    ensure_gh_available()
    generated_at = datetime.now(timezone.utc)
    since = generated_at - timedelta(hours=args.hours)
    runner_jobs = list_runner_jobs_from_api(
        runner_name=runner_name,
        since=since,
        until=generated_at,
        base_url=args.api_base_url,
        route=args.api_route,
        region=args.aws_region,
        timeout=args.api_timeout,
        owner_repo=args.owner_repo,
    )
    live_github_metrics = LiveGithubPopulationMetrics(statuses=ACTIVE_RUN_STATUSES)
    if args.live_jobs_json:
        live_jobs, live_github_metrics = load_live_jobs_json(args.live_jobs_json)
        live_runner_jobs = filter_live_jobs_for_runner(
            live_jobs,
            runner_name=runner_name,
            since=since,
            until=generated_at,
        )
        live_github_metrics.jobs_for_runner = len(live_runner_jobs)
        live_github_metrics.jobs_in_window = len(live_runner_jobs)
    elif args.run_enrichment:
        live_jobs, live_github_metrics = list_live_jobs_from_github(
            since=since,
            until=generated_at,
            owner_repo=args.owner_repo,
            gh_timeout=args.gh_timeout,
            live_run_workers=args.live_run_workers,
        )
        live_runner_jobs = filter_live_jobs_for_runner(
            live_jobs,
            runner_name=runner_name,
            since=since,
            until=generated_at,
        )
        live_github_metrics.jobs_for_runner = len(live_runner_jobs)
        live_github_metrics.jobs_in_window = len(live_runner_jobs)
    else:
        live_runner_jobs = []
        print("Live GitHub enrichment disabled for this runner report.")
    runner_jobs, live_jobs_added = merge_missing_runner_jobs(runner_jobs, live_runner_jobs)
    live_github_metrics.jobs_added_to_report = live_jobs_added
    if live_runner_jobs:
        print(
            f"Live GitHub enrichment found {len(live_runner_jobs)} runner job(s); "
            f"added {live_jobs_added} missing job(s) to the runner report."
        )

    triggering_results = triggering_results_for_runner(
        load_triggering_failures_json(args.triggering_failures_json),
        runner_name,
    )
    runner_jobs, triggering_jobs_added = merge_triggering_jobs(runner_jobs, triggering_results)
    if triggering_results:
        print(
            f"Received {len(triggering_results)} triggering failure(s) for {runner_name!r}; "
            f"added {triggering_jobs_added} missing job(s) to the runner report."
        )
    print(f"Found {len(runner_jobs)} job(s) on runner " f"{runner_name!r} in the last {args.hours} hour(s).")

    scan_results = runner_report_results(
        runner_jobs=runner_jobs,
        gh_timeout=args.gh_timeout,
        log_workers=args.log_workers,
        known_results=triggering_results,
    )
    report_json = build_runner_json_report(
        generated_at=generated_at,
        since=since,
        hours=args.hours,
        runner_name=runner_name,
        job_source=args.api_route,
        runner_jobs=runner_jobs,
        scan_results=scan_results,
        triggering_jobs=triggering_results,
        triggering_jobs_added=triggering_jobs_added,
        live_github_metrics=live_github_metrics,
    )
    report_md = build_runner_markdown_report(
        generated_at=generated_at,
        since=since,
        hours=args.hours,
        runner_name=runner_name,
        job_source=args.api_route,
        runner_jobs=runner_jobs,
        scan_results=scan_results,
        triggering_jobs_added=triggering_jobs_added,
        live_github_metrics=live_github_metrics,
    )
    write_reports(
        report_json_path=args.report_json,
        report_md_path=args.report_md,
        report_json=report_json,
        report_md=report_md,
    )

    failures = [result for result in scan_results if result.signature_labels]
    print(f"Runner-failure jobs found: {len(failures)}")
    if failures:
        print(f"runner {runner_name}: {format_signature_summary(failures)}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        if args.export_live_jobs_json:
            return export_live_github_jobs(args)
        return build_runner_report(args)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
