#!/usr/bin/env python3
"""Build a runner-specific failure report."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from runner_failure_common import (
    JobScanResult,
    RecentJob,
    SIGNATURE_VERSION,
    ensure_gh_available,
    format_signature_summary,
    format_utc,
    job_state_key,
    job_to_dict,
    markdown_escape,
    markdown_link,
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

GITHUB_JOB_LINK_RE = re.compile(
    r"https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/actions/runs/" r"(?P<run_id>\d+)/job/(?P<job_id>\d+)"
)


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
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.hours <= 0:
        raise ValueError("--hours must be greater than zero.")
    if args.api_timeout <= 0:
        raise ValueError("--api-timeout must be greater than zero.")
    if args.gh_timeout <= 0:
        raise ValueError("--gh-timeout must be greater than zero.")
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
) -> str:
    failures = [result for result in scan_results if result.signature_labels]
    lines = [
        "# Runner Failure Report",
        "",
        f"- Generated at: `{format_utc(generated_at)}`",
        f"- Runner: `{runner_name}`",
        f"- Window: last `{hours}` hour(s), since `{format_utc(since)}`",
        f"- Job source: `{job_source}`",
        f"- Recent jobs on runner: `{len(runner_jobs)}`",
        f"- Scanned jobs: `{sum(1 for result in scan_results if result.log_checked)}`",
        f"- Runner-failure jobs: `{len(failures)}`",
        "",
    ]

    if not scan_results:
        lines.append("No jobs found for this runner.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| GH job | Workflow | Job name | Status | Started | Log checked | Signatures |",
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
            f"{markdown_escape(result.job.workflow)} | "
            f"{markdown_escape(result.job.name)} | "
            f"{markdown_escape(result.job.conclusion or result.job.status or 'unknown')} | "
            f"{markdown_escape(result.job.started_at)} | "
            f"{'YES' if result.log_checked else 'NO'} | "
            f"{markdown_escape(signatures)} |"
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
            "scanned_jobs": len(scan_results),
            "log_checked_jobs": sum(1 for result in scan_results if result.log_checked),
            "runner_failure_jobs": len(failures),
        },
        "signature_counts": signature_counts(failures),
        "recent_jobs": [job_to_dict(job) for job in runner_jobs],
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
) -> list[JobScanResult]:
    jobs_to_scan = [job for job in runner_jobs if should_scan_log_for_full_table(job)]
    checked_results_by_key: dict[str, JobScanResult] = {}

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
    print(f"Found {len(runner_jobs)} job(s) on runner " f"{runner_name!r} in the last {args.hours} hour(s).")

    scan_results = runner_report_results(
        runner_jobs=runner_jobs,
        gh_timeout=args.gh_timeout,
        log_workers=args.log_workers,
    )
    report_json = build_runner_json_report(
        generated_at=generated_at,
        since=since,
        hours=args.hours,
        runner_name=runner_name,
        job_source=args.api_route,
        runner_jobs=runner_jobs,
        scan_results=scan_results,
    )
    report_md = build_runner_markdown_report(
        generated_at=generated_at,
        since=since,
        hours=args.hours,
        runner_name=runner_name,
        job_source=args.api_route,
        runner_jobs=runner_jobs,
        scan_results=scan_results,
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
        return build_runner_report(args)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
