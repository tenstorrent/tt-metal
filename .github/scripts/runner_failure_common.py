#!/usr/bin/env python3
"""Shared helpers for GitHub Actions runner-failure scans."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlparse

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - handled in load_config
    yaml = None


SIGNATURE_VERSION = "runner-failure-signatures-2026-07-14-v1"
UNKNOWN_RUNNER = "(unknown runner)"

OSC_SEQUENCE_RE = re.compile(r"\x1b\].*?\x1b\\")
CSI_SEQUENCE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FABRIC_LINK_MISMATCH_RE = re.compile(
    r"target\s+graph\s+edge\s+from\s+node\s+"
    r"\((?P<src_mesh>M\d+),\s*(?P<src_device>D\d+)\)\s+to\s+"
    r"\((?P<dst_mesh>M\d+),\s*(?P<dst_device>D\d+)\)\s+"
    r"requires\s+\d+\s+channels,\s+but\s+physical\s+edge\s+from\s+"
    r"\S+\s+to\s+\S+\s+only\s+has\s+\d+\s+channels",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ErrorSignature:
    key: str
    label: str
    needle: str | None = None
    pattern: str | None = None
    case_sensitive: bool = True


@dataclass(frozen=True)
class WorkflowJobFilter:
    include_exact: tuple[str, ...] = ()
    include_prefixes: tuple[str, ...] = ()
    exclude_exact: tuple[str, ...] = ()
    exclude_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkflowTarget:
    owner_repo: str
    workflow_id: str
    name: str
    source: str
    job_filter: WorkflowJobFilter


@dataclass(frozen=True)
class RecentJob:
    owner_repo: str
    workflow: str
    workflow_id: str
    run_id: str
    run_attempt: str
    run_url: str
    job_id: str
    name: str
    runner_name: str
    status: str
    conclusion: str
    html_url: str
    started_at: str
    completed_at: str


@dataclass(frozen=True)
class LogLookupResult:
    log_text: str | None
    status: str


@dataclass(frozen=True)
class JobScanResult:
    job: RecentJob
    log_status: str
    log_checked: bool
    signature_labels: tuple[str, ...]
    fabric_missing_links: str


ERROR_SIGNATURES = (
    ErrorSignature(
        key="TLB_ERROR_FOUND",
        label="TLB error",
        needle="Failed to allocate TLB window.",
    ),
    ErrorSignature(
        key="MISSING_DEVICES_FOUND",
        label="Missing devices",
        pattern=(
            r"(?:Requested\s+mesh\s+grid\s+shape\s+[^\n]*?"
            r"is\s+larger\s+than\s+number\s+of\s+available\s+devices|"
            r"Requested\s+mesh\s+shape\s+[^\n]*?"
            r"requires\s+\d+\s+devices,\s+but\s+only\s+\d+\s+devices\s+"
            r"(?:are\s+)?available|"
            r"Error\s+in\s+detecting\s+devices|"
            r"Query\s+mappings\s+failed\s+on\s+device\s+\d+)"
        ),
        case_sensitive=False,
    ),
    ErrorSignature(
        key="FABRIC_LINK_DOWN_MGD_TOPOLOGY_FOUND",
        label="Fabric link down (MGD topology)",
        needle="Graph specified in MGD could not fit in the discovered physical topology",
        case_sensitive=False,
    ),
    ErrorSignature(
        key="OUT_OF_DISK_FOUND",
        label="Out of disk",
        pattern=(
            r"(no\s+space\s+left\s+on\s+device|enospc|"
            r"disk\s+usage\s+is\s+(?:9\d|100)\s*%|"
            r"disk\s+usage\s+is\s+high)"
        ),
        case_sensitive=False,
    ),
)


def ensure_gh_available() -> None:
    if shutil.which("gh") is None:
        raise RuntimeError("Missing dependency: GitHub CLI `gh` is not available.")


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_github_time(value: str) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return datetime.fromtimestamp(0, timezone.utc)


def string_tuple_from_config(entry: dict[str, Any], field_name: str, workflow_label: str) -> tuple[str, ...]:
    value = entry.get(field_name)
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, list):
        raise ValueError(f"{workflow_label}: {field_name} must be a list of strings.")

    items: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str):
            raise ValueError(f"{workflow_label}: {field_name}[{index}] must be a string.")
        items.append(item)
    return tuple(items)


def workflow_filter_from_config(entry: dict[str, Any], workflow_label: str) -> WorkflowJobFilter:
    return WorkflowJobFilter(
        include_exact=string_tuple_from_config(entry, "include_exact", workflow_label),
        include_prefixes=string_tuple_from_config(entry, "include_prefixes", workflow_label),
        exclude_exact=string_tuple_from_config(entry, "exclude_exact", workflow_label),
        exclude_prefixes=string_tuple_from_config(entry, "exclude_prefixes", workflow_label),
    )


def parse_workflow_url(value: str) -> tuple[str, str]:
    parsed = urlparse(value)
    path_parts = [part for part in parsed.path.split("/") if part]
    if not parsed.scheme or parsed.netloc != "github.com":
        raise ValueError(f"Expected a GitHub workflow URL, got: {value}")
    try:
        actions_index = path_parts.index("actions")
    except ValueError as exc:
        raise ValueError(f"Workflow URL is missing /actions/workflows/: {value}") from exc
    if actions_index < 2 or actions_index + 2 >= len(path_parts) or path_parts[actions_index + 1] != "workflows":
        raise ValueError(f"Workflow URL is missing /actions/workflows/: {value}")

    owner_repo = f"{path_parts[actions_index - 2]}/{path_parts[actions_index - 1]}"
    workflow_id = "/".join(path_parts[actions_index + 2 :])
    if not workflow_id:
        raise ValueError(f"Workflow URL is missing workflow file name: {value}")
    return owner_repo, workflow_id


def workflow_from_config_entry(
    entry: dict[str, Any], default_owner_repo: str | None, index: int
) -> WorkflowTarget | None:
    workflow_label = f"workflow #{index}"
    enabled = entry.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{workflow_label}: enabled must be true or false.")
    if not enabled:
        return None

    url = entry.get("url")
    workflow_id = entry.get("workflow_id")
    owner_repo = entry.get("repository") or default_owner_repo
    if url:
        if not isinstance(url, str):
            raise ValueError(f"{workflow_label}: url must be a string.")
        owner_repo, workflow_id = parse_workflow_url(url)
    elif not isinstance(workflow_id, str) or not workflow_id:
        raise ValueError(f"{workflow_label}: missing workflow_id or url.")
    elif not isinstance(owner_repo, str) or not owner_repo:
        raise ValueError(f"{workflow_label}: missing repository.")

    name = entry.get("name") or str(workflow_id).rsplit("/", 1)[-1]
    if not isinstance(name, str):
        raise ValueError(f"{workflow_label}: name must be a string.")

    return WorkflowTarget(
        owner_repo=owner_repo,
        workflow_id=str(workflow_id),
        name=name,
        source=str(url or workflow_id),
        job_filter=workflow_filter_from_config(entry, name),
    )


def load_workflows(config_path: Path) -> list[WorkflowTarget]:
    if yaml is None:
        raise RuntimeError("Missing dependency: install PyYAML to read workflow config.")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read workflow config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Unable to parse workflow config {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Workflow config {config_path} must contain a mapping.")

    default_owner_repo = config.get("repository")
    if default_owner_repo is not None and not isinstance(default_owner_repo, str):
        raise ValueError("repository must be a string when set.")

    workflow_entries = config.get("workflows")
    if not isinstance(workflow_entries, list):
        raise ValueError(f"Workflow config {config_path} must contain workflows list.")

    workflows: list[WorkflowTarget] = []
    for index, entry in enumerate(workflow_entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"workflow #{index} must be a mapping.")
        workflow = workflow_from_config_entry(entry, default_owner_repo, index)
        if workflow is None:
            continue
        workflows.append(workflow)

    if not workflows:
        raise ValueError(f"Workflow config {config_path} has no enabled workflows.")
    return workflows


def matches_any_prefix(value: str, prefixes: tuple[str, ...]) -> bool:
    folded_value = value.casefold()
    return any(folded_value.startswith(prefix.casefold()) for prefix in prefixes)


def matches_any_exact(value: str, exact_values: tuple[str, ...]) -> bool:
    folded_value = value.casefold()
    return any(folded_value == exact_value.casefold() for exact_value in exact_values)


def job_allowed_by_filter(job_name: str, job_filter: WorkflowJobFilter) -> bool:
    has_includes = bool(job_filter.include_exact or job_filter.include_prefixes)
    if has_includes and not (
        matches_any_exact(job_name, job_filter.include_exact)
        or matches_any_prefix(job_name, job_filter.include_prefixes)
    ):
        return False
    if matches_any_exact(job_name, job_filter.exclude_exact):
        return False
    if matches_any_prefix(job_name, job_filter.exclude_prefixes):
        return False
    return True


def gh_env() -> dict[str, str]:
    env = os.environ.copy()
    if env.get("GITHUB_TOKEN") and not env.get("GH_TOKEN"):
        env["GH_TOKEN"] = env["GITHUB_TOKEN"]
    return env


def gh_api_json(endpoint: str, *, paginate: bool = False, timeout: int = 120) -> Any:
    cmd = ["gh", "api", "--method", "GET"]
    if paginate:
        cmd.extend(["--paginate", "--slurp"])
    cmd.append(endpoint)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        env=gh_env(),
    )
    if result.returncode != 0:
        details = " ".join((result.stderr or result.stdout or "unknown gh api error").split())
        raise RuntimeError(f"gh api failed for {endpoint}: {details}")
    return json.loads(result.stdout or "{}")


def paginated_items(response: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(response, dict):
        return list(response.get(key, []))

    items: list[dict[str, Any]] = []
    if isinstance(response, list):
        for page in response:
            if isinstance(page, dict):
                items.extend(page.get(key, []))
    return items


def workflow_runs_endpoint(workflow: WorkflowTarget, since: datetime) -> str:
    workflow_id = quote(workflow.workflow_id, safe="")
    query = urlencode(
        {
            "per_page": "100",
            "exclude_pull_requests": "true",
            "created": f">={format_utc(since)}",
        }
    )
    return f"repos/{workflow.owner_repo}/actions/workflows/{workflow_id}/runs?{query}"


def workflow_run_jobs_endpoint(owner_repo: str, run_id: str) -> str:
    query = urlencode({"filter": "all", "per_page": "100"})
    return f"repos/{owner_repo}/actions/runs/{run_id}/jobs?{query}"


def recent_job_from_api(
    *,
    owner_repo: str,
    workflow_name: str,
    workflow_id: str,
    run: dict[str, Any],
    job: dict[str, Any],
) -> RecentJob:
    return RecentJob(
        owner_repo=owner_repo,
        workflow=workflow_name,
        workflow_id=workflow_id,
        run_id=str(run.get("id") or ""),
        run_attempt=str(run.get("run_attempt") or ""),
        run_url=str(run.get("html_url") or ""),
        job_id=str(job.get("id") or ""),
        name=str(job.get("name") or ""),
        runner_name=str(job.get("runner_name") or ""),
        status=str(job.get("status") or ""),
        conclusion=str(job.get("conclusion") or ""),
        html_url=str(job.get("html_url") or ""),
        started_at=str(job.get("started_at") or ""),
        completed_at=str(job.get("completed_at") or ""),
    )


def list_recent_jobs(workflows: list[WorkflowTarget], since: datetime, gh_timeout: int) -> list[RecentJob]:
    recent_jobs: list[RecentJob] = []
    seen_jobs: set[tuple[str, str]] = set()

    for workflow in workflows:
        kept_count = 0
        filtered_count = 0
        runs_response = gh_api_json(
            workflow_runs_endpoint(workflow, since),
            paginate=True,
            timeout=gh_timeout,
        )
        runs = paginated_items(runs_response, "workflow_runs")
        print(f"Found {len(runs)} recent run(s) for {workflow.name} ({workflow.source}).")

        for run in runs:
            run_id = str(run.get("id") or "")
            if not run_id:
                continue

            jobs_response = gh_api_json(
                workflow_run_jobs_endpoint(workflow.owner_repo, run_id),
                paginate=True,
                timeout=gh_timeout,
            )
            for job in paginated_items(jobs_response, "jobs"):
                job_id = str(job.get("id") or "")
                if not job_id:
                    continue

                job_name = str(job.get("name") or "")
                if not job_allowed_by_filter(job_name, workflow.job_filter):
                    filtered_count += 1
                    continue

                job_key = (workflow.owner_repo, job_id)
                if job_key in seen_jobs:
                    continue
                seen_jobs.add(job_key)
                kept_count += 1

                recent_jobs.append(
                    recent_job_from_api(
                        owner_repo=workflow.owner_repo,
                        workflow_name=workflow.name,
                        workflow_id=workflow.workflow_id,
                        run=run,
                        job=job,
                    )
                )

        print(f"Kept {kept_count} job(s) for {workflow.name}; " f"filtered out {filtered_count} job(s).")

    return sorted(
        recent_jobs,
        key=lambda job: parse_github_time(job.started_at),
        reverse=True,
    )


def job_state_key(job: RecentJob) -> str:
    return f"{job.owner_repo}:{job.job_id}"


def signature_keys() -> list[str]:
    return [signature.key for signature in ERROR_SIGNATURES]


def strip_terminal_sequences(value: str) -> str:
    return CSI_SEQUENCE_RE.sub("", OSC_SEQUENCE_RE.sub("", value))


def signature_found(log_text: str, signature: ErrorSignature) -> bool:
    if signature.pattern:
        flags = 0 if signature.case_sensitive else re.IGNORECASE
        return re.search(signature.pattern, log_text, flags=flags) is not None

    if not signature.needle:
        return False

    if signature.case_sensitive:
        return signature.needle in log_text
    return signature.needle.lower() in log_text.lower()


def format_fabric_node(mesh: str, device: str) -> str:
    return f"{mesh.lower()},{device.lower()}"


def extract_fabric_missing_links(log_text: str) -> str:
    links: list[str] = []
    seen_links: set[str] = set()
    plain_log_text = strip_terminal_sequences(log_text)
    for match in FABRIC_LINK_MISMATCH_RE.finditer(plain_log_text):
        source = format_fabric_node(match.group("src_mesh"), match.group("src_device"))
        destination = format_fabric_node(match.group("dst_mesh"), match.group("dst_device"))
        link = f"{source}>{destination}"
        if link not in seen_links:
            seen_links.add(link)
            links.append(link)
    return "; ".join(links)


def matching_signature_labels(log_text: str) -> list[str]:
    return [signature.label for signature in ERROR_SIGNATURES if signature_found(log_text, signature)]


def fetch_github_job_log(job: RecentJob, timeout: int) -> LogLookupResult:
    endpoint = f"repos/{job.owner_repo}/actions/jobs/{job.job_id}/logs"
    try:
        result = subprocess.run(
            ["gh", "api", endpoint],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=gh_env(),
        )
    except subprocess.TimeoutExpired:
        return LogLookupResult(log_text=None, status=f"gh api timed out after {timeout}s")

    if result.returncode != 0:
        details = " ".join((result.stderr or result.stdout or "unknown gh api error").split())
        return LogLookupResult(log_text=None, status=f"gh api failed: {details}")

    return LogLookupResult(log_text=result.stdout, status="fetched")


def scan_job(job: RecentJob, timeout: int) -> JobScanResult:
    log_result = fetch_github_job_log(job, timeout=timeout)
    if log_result.log_text is None:
        return JobScanResult(
            job=job,
            log_status=log_result.status,
            log_checked=False,
            signature_labels=(),
            fabric_missing_links="",
        )

    signature_labels = matching_signature_labels(log_result.log_text)
    fabric_missing_links = ""
    if "Fabric link down (MGD topology)" in signature_labels:
        fabric_missing_links = extract_fabric_missing_links(log_result.log_text)

    return JobScanResult(
        job=job,
        log_status=log_result.status,
        log_checked=True,
        signature_labels=tuple(signature_labels),
        fabric_missing_links=fabric_missing_links,
    )


def scan_jobs(jobs: list[RecentJob], *, gh_timeout: int, log_workers: int) -> list[JobScanResult]:
    if not jobs:
        return []

    worker_count = min(log_workers, len(jobs))
    print(f"Scanning {len(jobs)} job log(s) with {worker_count} worker(s).")
    results: list[JobScanResult] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(scan_job, job, gh_timeout) for job in jobs]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                print(f"warning: job scan failed: {exc}", file=sys.stderr)
                continue
            results.append(result)
            if not result.log_checked:
                print(
                    f"warning: could not check {result.job.html_url}: " f"{result.log_status}",
                    file=sys.stderr,
                )
            elif result.signature_labels:
                print(f"runner failure {result.job.html_url}")
    return sorted(
        results,
        key=lambda result: parse_github_time(result.job.started_at),
        reverse=True,
    )


def job_to_dict(job: RecentJob) -> dict[str, Any]:
    return {
        "owner_repo": job.owner_repo,
        "workflow": job.workflow,
        "workflow_id": job.workflow_id,
        "run_id": job.run_id,
        "run_attempt": job.run_attempt,
        "run_url": job.run_url,
        "job_id": job.job_id,
        "name": job.name,
        "runner_name": job.runner_name,
        "status": job.status,
        "conclusion": job.conclusion,
        "html_url": job.html_url,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }


def result_to_dict(result: JobScanResult) -> dict[str, Any]:
    value = job_to_dict(result.job)
    value.update(
        {
            "log_checked": result.log_checked,
            "log_status": result.log_status,
            "signatures": list(result.signature_labels),
            "fabric_missing_links": result.fabric_missing_links,
        }
    )
    return value


def signature_counts(results: list[JobScanResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for label in result.signature_labels:
            counts[label] = counts.get(label, 0) + 1
    return counts


def format_signature_summary(results: list[JobScanResult]) -> str:
    counts = signature_counts(results)
    return ", ".join(f"{count}x {label}" for label, count in sorted(counts.items()))


def runner_name_for_job(job: RecentJob) -> str:
    return job.runner_name or UNKNOWN_RUNNER


def group_results_by_runner(
    results: list[JobScanResult],
) -> dict[str, list[JobScanResult]]:
    grouped: dict[str, list[JobScanResult]] = {}
    for result in results:
        grouped.setdefault(runner_name_for_job(result.job), []).append(result)
    return grouped


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def markdown_link(label: str, url: str) -> str:
    if not url:
        return markdown_escape(label)
    return f"[{markdown_escape(label)}]({url})"


def write_reports(
    *,
    report_json_path: Path,
    report_md_path: Path,
    report_json: dict[str, Any],
    report_md: str,
) -> None:
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(
        json.dumps(report_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_md_path.write_text(report_md, encoding="utf-8")
