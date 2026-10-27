#!/usr/bin/env python3
"""Shared helpers for GitHub Actions runner-failure scans."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlparse

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - handled in load_config
    yaml = None


SIGNATURE_VERSION = "runner-failure-signatures-2026-10-05-v3"
UNKNOWN_RUNNER = "(unknown runner)"
ACTIVE_JOB_STATUSES = {"queued", "in_progress", "waiting", "pending", "requested"}
NON_FAILED_CONCLUSIONS = {"success", "skipped", "cancelled"}
LOG_SCAN_CHUNK_SIZE = 4 * 1024 * 1024
LOG_SCAN_OVERLAP = 16 * 1024

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
OUT_OF_DISK_HARD_RE = re.compile(r"(no\s+space\s+left\s+on\s+device|enospc)", re.IGNORECASE)
DISK_USAGE_RE = re.compile(r"disk\s+usage\s+is\s+(?P<percent>\d{1,3})\s*%", re.IGNORECASE)
DISK_USAGE_HIGH_RE = re.compile(r"disk\s+usage\s+is\s+high", re.IGNORECASE)


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
    setup_runner_conclusion: str


@dataclass(frozen=True)
class LogLookupResult:
    log_path: Path | None
    status: str
    unavailable: bool = False
    signature_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class JobScanResult:
    job: RecentJob
    log_status: str
    log_checked: bool
    signature_labels: tuple[str, ...]
    fabric_missing_links: str
    log_unavailable: bool = False


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
        pattern=r"Graph\s+specified\s+in\s+MGD\s+could\s+not\s+fit\s+in\s+the\s+discovered\s+physical\s+topology",
        case_sensitive=False,
    ),
    ErrorSignature(
        key="WRONG_MESH_SHAPE_FOUND",
        label="Wrong mesh shape",
        pattern=(
            r"Requested\s+mesh\s+is\s+too\s+big\s+and\s+is\s+not\s+rotatable:\s*"
            r"MeshShape\(\[[^\]]+\]\)\s+and\s+SystemMesh\s+MeshShape\(\[[^\]]+\]\)"
        ),
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
    ErrorSignature(
        key="FAILED_RESET_FOUND",
        label="Failed reset",
        needle="Unable to reset board successfully",
        case_sensitive=False,
    ),
    ErrorSignature(
        key="PHYSICAL_DISCOVERY_FAILURE_FOUND",
        label="Physical discovery failure",
        pattern=r"Physical\s+Discovery\s+found\s+\d+\s+missing\s+channel\s+connections?",
        case_sensitive=False,
    ),
    ErrorSignature(
        key="PHYSICAL_CHIP_NOT_FOUND",
        label="Physical chip not found",
        pattern=r"Physical\s+chip\s+id\s+\d+\s+(?:is\s+)?not\s+found\s+in\s+(?:the\s+)?control\s+plane\s+chip\s+mapping",
        case_sensitive=False,
    ),
    ErrorSignature(
        key="ETH_HEARTBEAT_TIMEOUT_FOUND",
        label="ETH heartbeat timeout",
        pattern=r"Timed\s+out\s+waiting\s+for\s+(?:an?\s+)?ETH(?:\s+core)?\s+heartbeat",
        case_sensitive=False,
    ),
    ErrorSignature(
        key="SETUP_RUNNER_FAILURE_FOUND",
        label="Set up runner failure",
    ),
    ErrorSignature(
        key="RUNNER_DISCONNECTED_FOUND",
        label="Runner disconnected",
        needle="The self-hosted runner lost communication with the server",
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


def step_conclusion(job: dict[str, Any], step_name: str) -> str:
    steps = job.get("steps")
    if not isinstance(steps, list):
        return ""

    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("name") or "").casefold() == step_name.casefold():
            return str(step.get("conclusion") or "")
    return ""


def recent_job_from_api(
    *,
    owner_repo: str,
    workflow_name: str,
    workflow_id: str,
    run: dict[str, Any],
    job: dict[str, Any],
) -> RecentJob:
    setup_runner_conclusion = step_conclusion(job, "Set up runner")
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
        setup_runner_conclusion=setup_runner_conclusion,
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
    if signature.key == "OUT_OF_DISK_FOUND":
        return out_of_disk_signature_found(log_text)

    if signature.pattern:
        flags = 0 if signature.case_sensitive else re.IGNORECASE
        return re.search(signature.pattern, log_text, flags=flags) is not None

    if not signature.needle:
        return False

    if signature.case_sensitive:
        return signature.needle in log_text
    return signature.needle.lower() in log_text.lower()


def out_of_disk_signature_found(log_text: str) -> bool:
    plain_log_text = strip_terminal_sequences(log_text)
    if OUT_OF_DISK_HARD_RE.search(plain_log_text):
        return True
    return latest_disk_pressure(plain_log_text) is True


def latest_disk_pressure(log_text: str) -> bool | None:
    last_position = -1
    high_pressure = None
    for match in DISK_USAGE_RE.finditer(log_text):
        last_position = match.start()
        high_pressure = int(match.group("percent")) >= 90

    for match in DISK_USAGE_HIGH_RE.finditer(log_text):
        if match.start() > last_position:
            last_position = match.start()
            high_pressure = True
    return high_pressure


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
    plain_log_text = strip_terminal_sequences(log_text)
    return [signature.label for signature in ERROR_SIGNATURES if signature_found(plain_log_text, signature)]


def scan_log_file(log_path: Path) -> tuple[list[str], str]:
    labels: set[str] = set()
    links: dict[str, None] = {}
    tail = ""
    hard_disk_failure = False
    disk_pressure = None
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        while chunk := log_file.read(LOG_SCAN_CHUNK_SIZE):
            raw_window = tail + chunk
            plain_window = strip_terminal_sequences(raw_window)
            # Keep 16 KiB of context for signatures and formatting split across reads.
            tail = raw_window[-LOG_SCAN_OVERLAP:]
            for signature in ERROR_SIGNATURES:
                if signature.key != "OUT_OF_DISK_FOUND" and signature.label not in labels:
                    if signature_found(plain_window, signature):
                        labels.add(signature.label)
            hard_disk_failure |= OUT_OF_DISK_HARD_RE.search(plain_window) is not None
            pressure = latest_disk_pressure(plain_window)
            if pressure is not None:
                disk_pressure = pressure
            for match in FABRIC_LINK_MISMATCH_RE.finditer(plain_window):
                source = format_fabric_node(match.group("src_mesh"), match.group("src_device"))
                destination = format_fabric_node(match.group("dst_mesh"), match.group("dst_device"))
                links[f"{source}>{destination}"] = None
    if hard_disk_failure or disk_pressure is True:
        labels.add("Out of disk")
    ordered_labels = [signature.label for signature in ERROR_SIGNATURES if signature.label in labels]
    missing_links = "; ".join(links) if "Fabric link down (MGD topology)" in labels else ""
    return ordered_labels, missing_links


def setup_runner_step_failed(job: RecentJob) -> bool:
    return job.setup_runner_conclusion.casefold() == "failure"


def is_failed_job(job: RecentJob) -> bool:
    conclusion = job.conclusion.casefold()
    return job.status.casefold() == "completed" and bool(conclusion) and conclusion not in NON_FAILED_CONCLUSIONS


def matching_job_metadata_signature_labels(job: RecentJob) -> list[str]:
    if setup_runner_step_failed(job):
        return ["Set up runner failure"]
    return []


def combine_signature_labels(*label_groups: list[str]) -> list[str]:
    """Return unique labels in ERROR_SIGNATURES declaration order."""
    labels_by_name = {label for labels in label_groups for label in labels}
    return [signature.label for signature in ERROR_SIGNATURES if signature.label in labels_by_name]


def runner_disconnection_signature_labels(job: RecentJob, payload: dict[str, Any], timeout: int) -> list[str]:
    check_path = urlparse(str(payload.get("check_run_url") or "")).path
    expected_prefix = f"/repos/{job.owner_repo}/check-runs/"
    if not check_path.startswith(expected_prefix) or not check_path.removeprefix(expected_prefix).isdigit():
        return []
    annotations = gh_api_json(f"{check_path.lstrip('/')}/annotations?per_page=100", paginate=True, timeout=timeout)
    for page in annotations if isinstance(annotations, list) else []:
        for annotation in page if isinstance(page, list) else [page]:
            if (
                isinstance(annotation, dict)
                and "self-hosted runner lost communication" in str(annotation.get("message") or "").casefold()
            ):
                return ["Runner disconnected"]
    return []


def missing_job_log_result(
    job: RecentJob, timeout: int, *, runner_disconnected: bool = False
) -> LogLookupResult | None:
    labels = ["Runner disconnected"] if runner_disconnected else []
    if not labels:
        try:
            payload = gh_api_json(f"repos/{job.owner_repo}/actions/jobs/{job.job_id}", timeout=timeout)
            if not isinstance(payload, dict):
                return None
            status = str(payload.get("status") or "").casefold()
            if status in ACTIVE_JOB_STATUSES:
                return LogLookupResult(
                    log_path=None, status=f"not available: job is {status} (HTTP 404)", unavailable=True
                )
            labels = runner_disconnection_signature_labels(job, payload, timeout)
        except (json.JSONDecodeError, OSError, RuntimeError, subprocess.TimeoutExpired):
            return None

    if labels:
        return LogLookupResult(
            log_path=None,
            status="not available: runner lost communication with GitHub (HTTP 404)",
            unavailable=True,
            signature_labels=tuple(labels),
        )
    return None


def fetch_github_job_log(
    job: RecentJob, timeout: int, log_path: Path, *, runner_disconnected: bool = False
) -> LogLookupResult:
    if job.status.casefold() in ACTIVE_JOB_STATUSES:
        return LogLookupResult(log_path=None, status=f"not available: job is {job.status}", unavailable=True)

    endpoint = f"repos/{job.owner_repo}/actions/jobs/{job.job_id}/logs"
    with log_path.open("wb") as output, tempfile.TemporaryFile() as errors:
        try:
            # Logs go directly to a file and are never printed verbatim. Embedded
            # terminal formatting cannot control this process's terminal.
            result = subprocess.run(
                ["gh", "api", "--allow-escape-sequences", endpoint],
                stdout=output,
                stderr=errors,
                timeout=timeout,
                check=False,
                env=gh_env(),
            )
        except subprocess.TimeoutExpired:
            return LogLookupResult(log_path=None, status=f"gh api timed out after {timeout}s")
        errors.seek(0)
        error_text = errors.read(16384).decode("utf-8", errors="replace")

    if result.returncode != 0:
        if not error_text:
            with log_path.open("rb") as output:
                error_text = output.read(16384).decode("utf-8", errors="replace")
        safe_error_text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", strip_terminal_sequences(error_text))
        details = " ".join((safe_error_text or "unknown gh api error").split())
        if re.search(r"\bHTTP 404\b", details):
            # GitHub can close a disconnected runner's job without publishing its logs.
            missing_log_result = missing_job_log_result(job, timeout=timeout, runner_disconnected=runner_disconnected)
            if missing_log_result is not None:
                return missing_log_result
        return LogLookupResult(log_path=None, status=f"gh api failed: {details}")

    log_size = log_path.stat().st_size
    if log_size >= 100 * 1024 * 1024:
        print(f"Large job log: {job.html_url}, {log_size / 1024 / 1024:.1f} MiB (scanning from disk).", flush=True)
    return LogLookupResult(log_path=log_path, status="fetched")


def should_fetch_setup_runner_metadata(job: RecentJob) -> bool:
    return bool(
        job.owner_repo
        and job.job_id
        and not job.setup_runner_conclusion
        and job.status.casefold() == "completed"
        and job.conclusion.casefold() == "failure"
    )


def enrich_failure_metadata(job: RecentJob, timeout: int) -> tuple[RecentJob, list[str]]:
    fetch_setup = should_fetch_setup_runner_metadata(job)
    check_annotations = is_failed_job(job) and bool(job.owner_repo and job.job_id)
    if not fetch_setup and not check_annotations:
        return job, []

    try:
        payload = gh_api_json(f"repos/{job.owner_repo}/actions/jobs/{job.job_id}", timeout=timeout)
    except (json.JSONDecodeError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"warning: could not fetch job metadata for {job.html_url}: {exc}", file=sys.stderr)
        return job, []

    if not isinstance(payload, dict):
        return job, []

    if fetch_setup:
        setup_runner_conclusion = step_conclusion(payload, "Set up runner")
        if setup_runner_conclusion:
            job = replace(job, setup_runner_conclusion=setup_runner_conclusion)
    labels = []
    if check_annotations:
        try:
            labels = runner_disconnection_signature_labels(job, payload, timeout)
        except (json.JSONDecodeError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"warning: could not fetch failure annotations for {job.html_url}: {exc}", file=sys.stderr)
    return job, labels


def scan_job(job: RecentJob, timeout: int) -> JobScanResult:
    job, annotation_signature_labels = enrich_failure_metadata(job, timeout=timeout)
    metadata_signature_labels = combine_signature_labels(
        matching_job_metadata_signature_labels(job), annotation_signature_labels
    )
    with tempfile.TemporaryDirectory(prefix="runner-failure-log-") as log_dir:
        log_result = fetch_github_job_log(
            job,
            timeout=timeout,
            log_path=Path(log_dir) / "job.log",
            runner_disconnected="Runner disconnected" in metadata_signature_labels,
        )
        log_signature_labels, fabric_missing_links = (
            scan_log_file(log_result.log_path) if log_result.log_path is not None else ([], "")
        )
    metadata_signature_labels = combine_signature_labels(metadata_signature_labels, list(log_result.signature_labels))
    if log_result.log_path is None:
        return JobScanResult(
            job=job,
            log_status=log_result.status,
            log_checked=False,
            signature_labels=tuple(metadata_signature_labels),
            fabric_missing_links="",
            log_unavailable=log_result.unavailable,
        )

    signature_labels = combine_signature_labels(metadata_signature_labels, log_signature_labels)

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
            if result.log_unavailable:
                print(f"Log unavailable for {result.job.html_url}: {result.log_status}.")
            elif not result.log_checked:
                print(
                    f"warning: could not check {result.job.html_url}: " f"{result.log_status}",
                    file=sys.stderr,
                )
            if result.signature_labels:
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
        "setup_runner_conclusion": job.setup_runner_conclusion,
    }


def result_to_dict(result: JobScanResult) -> dict[str, Any]:
    value = job_to_dict(result.job)
    value.update(
        {
            "log_checked": result.log_checked,
            "log_status": result.log_status,
            "signatures": list(result.signature_labels),
            "fabric_missing_links": result.fabric_missing_links,
            "log_unavailable": result.log_unavailable,
        }
    )
    return value


def job_from_dict(value: dict[str, Any]) -> RecentJob:
    return RecentJob(
        owner_repo=str(value.get("owner_repo") or ""),
        workflow=str(value.get("workflow") or ""),
        workflow_id=str(value.get("workflow_id") or ""),
        run_id=str(value.get("run_id") or ""),
        run_attempt=str(value.get("run_attempt") or ""),
        run_url=str(value.get("run_url") or ""),
        job_id=str(value.get("job_id") or ""),
        name=str(value.get("name") or ""),
        runner_name=str(value.get("runner_name") or ""),
        status=str(value.get("status") or ""),
        conclusion=str(value.get("conclusion") or ""),
        html_url=str(value.get("html_url") or ""),
        started_at=str(value.get("started_at") or ""),
        completed_at=str(value.get("completed_at") or ""),
        setup_runner_conclusion=str(value.get("setup_runner_conclusion") or ""),
    )


def scan_result_from_dict(value: dict[str, Any]) -> JobScanResult:
    raw_signatures = value.get("signatures")
    signatures: tuple[str, ...] = ()
    if isinstance(raw_signatures, list):
        signatures = tuple(str(item) for item in raw_signatures if item)
    return JobScanResult(
        job=job_from_dict(value),
        log_status=str(value.get("log_status") or ""),
        log_checked=bool(value.get("log_checked")),
        signature_labels=signatures,
        fabric_missing_links=str(value.get("fabric_missing_links") or ""),
        log_unavailable=bool(value.get("log_unavailable")),
    )


def load_triggering_failures_json(path: Path | None) -> list[JobScanResult]:
    if path is None:
        return []

    try:
        raw_values = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Unable to read triggering failures JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unable to parse triggering failures JSON {path}: {exc}") from exc

    if not isinstance(raw_values, list):
        raise RuntimeError(f"Triggering failures JSON {path} must contain a list.")

    return [scan_result_from_dict(value) for value in raw_values if isinstance(value, dict)]


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
