#!/usr/bin/env python3
"""
Extract jobs that failed in the last N consecutive workflow runs.

The script downloads workflow-data from the latest aggregate-workflow-data run,
filters to workflows with N consecutive failures, then fetches run jobs from
the GitHub API to find which specific jobs failed in all N runs.

Workflows whose display name contains any keyword in
``SKIP_WORKFLOW_NAME_KEYWORDS`` (case-insensitive substring match) are skipped:
no per-run job downloads for those entries. The aggregate artifact is still
downloaded as a single file upstream.

Usage:
  python tools/ci/extract_failing_jobs.py
  python tools/ci/extract_failing_jobs.py t3000-unit-tests

Optional env vars:
  CONSECUTIVE_FAILURES   Number of consecutive failures required (default: 3)
  GITHUB_OWNER           Default: tenstorrent
  GITHUB_REPO            Default: tt-metal
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OWNER = os.environ.get("GITHUB_OWNER", "tenstorrent")
REPO = os.environ.get("GITHUB_REPO", "tt-metal")
WORKFLOW_FILE = "aggregate-workflow-data.yaml"
ARTIFACT_NAME = "workflow-data"
CONSECUTIVE_FAILURES = int(os.environ.get("CONSECUTIVE_FAILURES", "3"))

# Substrings; if any appear in the workflow display name (case-insensitive), skip job fetches.
SKIP_WORKFLOW_NAME_KEYWORDS: tuple[str, ...] = ("sanity",)


def resolve_token() -> str | None:
    if os.environ.get("GITHUB_TOKEN"):
        return os.environ["GITHUB_TOKEN"]
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return None


def download_workflow_data() -> tuple[list, int]:
    print("Finding latest aggregate-workflow-data run...", file=sys.stderr)
    result = subprocess.run(
        [
            "gh",
            "run",
            "list",
            f"--workflow={WORKFLOW_FILE}",
            f"--repo={OWNER}/{REPO}",
            "--status=completed",
            "--limit=1",
            "--json=databaseId,conclusion",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list workflow runs: {result.stderr}")

    runs = json.loads(result.stdout)
    if not runs:
        raise RuntimeError("No completed runs found for aggregate-workflow-data")

    run_id = runs[0]["databaseId"]
    conclusion = runs[0]["conclusion"]
    print(f"  Latest run: {run_id} (conclusion: {conclusion})", file=sys.stderr)
    if conclusion != "success":
        print(
            f"  Warning: latest run did not succeed ({conclusion}); data may be stale.",
            file=sys.stderr,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  Downloading '{ARTIFACT_NAME}' artifact...", file=sys.stderr)
        result = subprocess.run(
            [
                "gh",
                "run",
                "download",
                str(run_id),
                f"--repo={OWNER}/{REPO}",
                "-n",
                ARTIFACT_NAME,
                "-D",
                tmpdir,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download artifact: {result.stderr}")

        json_file = Path(tmpdir) / "workflow-data.json"
        if not json_file.exists():
            json_files = list(Path(tmpdir).rglob("*.json"))
            if not json_files:
                raise RuntimeError("No JSON files found in downloaded artifact")
            json_file = json_files[0]

        print(
            f"  Loaded workflow data ({json_file.stat().st_size / 1_000_000:.1f} MB)",
            file=sys.stderr,
        )
        with open(json_file, encoding="utf-8") as f:
            return json.load(f), int(run_id)


def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            parsed = json.load(f)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def gh_api_request(url: str, token: str | None, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        req = Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        if token:
            req.add_header("Authorization", f"Bearer {token}")

        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except IncompleteRead:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
        except HTTPError as err:
            if err.code == 404:
                return {"jobs": []}
            if err.code in (502, 503, 504) and attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
        except URLError as err:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f"Request failed: {err}") from err

    raise RuntimeError(f"Exhausted retries for {url}")


def fetch_jobs_for_run(run_id: int, token: str | None) -> list[dict]:
    jobs: list[dict] = []
    page = 1
    per_page = 100

    while True:
        url = (
            f"https://api.github.com/repos/{OWNER}/{REPO}"
            f"/actions/runs/{run_id}/jobs?per_page={per_page}&page={page}"
        )
        data = gh_api_request(url, token)
        batch = data.get("jobs", [])
        jobs.extend(batch)

        if len(batch) < per_page:
            break
        page += 1
        time.sleep(0.5)

    return jobs


def extract_failing_jobs(
    workflow_data: list,
    token: str | None,
    aggregate_run_id: int,
    workflow_filter: str | None = None,
    output_path: Path | None = None,
    previous_cache: dict | None = None,
    cache_output_path: Path | None = None,
) -> dict:
    results = []
    api_calls = 0
    cached_reused_workflows = 0
    previous_cache = previous_cache or {}
    previous_workflows = previous_cache.get("workflows", {})
    if not isinstance(previous_workflows, dict):
        previous_workflows = {}

    def write_incremental() -> None:
        if output_path is None:
            return
        payload = {
            "description": ("Jobs that failed in all of the last " f"{CONSECUTIVE_FAILURES} workflow runs"),
            "total_jobs_affected": len(results),
            "jobs": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"    Wrote {len(results)} jobs to {output_path}", file=sys.stderr)

    def write_cache() -> None:
        if cache_output_path is None:
            return
        cache_output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "aggregate_run_id": aggregate_run_id,
            "consecutive_failures": CONSECUTIVE_FAILURES,
            "workflows": workflow_cache,
        }
        with open(cache_output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    write_incremental()
    workflow_cache: dict[str, dict] = {}

    for workflow_name, runs in workflow_data:
        if not runs:
            continue

        name_lower = str(workflow_name).lower()
        skip_kw = next((kw for kw in SKIP_WORKFLOW_NAME_KEYWORDS if kw.lower() in name_lower), None)
        if skip_kw is not None:
            print(
                f'  Skipping workflow "{workflow_name}" (name contains keyword {skip_kw!r}).',
                file=sys.stderr,
            )
            continue

        if workflow_filter:
            yaml_path = runs[0].get("path", "")
            yaml_name = yaml_path.rsplit("/", 1)[-1].removesuffix(".yaml").removesuffix(".yml")
            if yaml_name != workflow_filter:
                continue

        sorted_runs = sorted(
            runs,
            key=lambda run: run.get("created_at", "") or run.get("run_started_at", ""),
            reverse=True,
        )
        last_n = sorted_runs[:CONSECUTIVE_FAILURES]

        if len(last_n) < CONSECUTIVE_FAILURES:
            continue
        if not all(run.get("conclusion") == "failure" for run in last_n):
            continue

        run_id_to_failed_jobs: dict[int, dict[str, str]] = {}
        run_ids_for_cache = [int(r.get("id")) for r in last_n if r.get("id")]
        if len(run_ids_for_cache) < CONSECUTIVE_FAILURES:
            continue
        yaml_path = runs[0].get("path", "")
        workflow_key = f"{yaml_path}|{workflow_name}"
        previous_entry = previous_workflows.get(workflow_key)
        if isinstance(previous_entry, dict):
            prev_run_ids = previous_entry.get("run_ids", [])
            prev_jobs = previous_entry.get("jobs", [])
            if (
                isinstance(prev_run_ids, list)
                and isinstance(prev_jobs, list)
                and [int(x) for x in prev_run_ids] == run_ids_for_cache
            ):
                for item in prev_jobs:
                    if isinstance(item, dict):
                        results.append(item)
                workflow_cache[workflow_key] = {
                    "workflow_name": workflow_name,
                    "yaml_path": yaml_path,
                    "run_ids": run_ids_for_cache,
                    "jobs": [item for item in prev_jobs if isinstance(item, dict)],
                    "reused": True,
                }
                cached_reused_workflows += 1
                print(
                    f'  Reusing cached failing jobs for "{workflow_name}" (run ids unchanged).',
                    file=sys.stderr,
                )
                write_incremental()
                write_cache()
                continue

        print(
            f'  Fetching jobs for "{workflow_name}" ({CONSECUTIVE_FAILURES} runs)...',
            file=sys.stderr,
        )
        for run in last_n:
            run_id = run.get("id")
            if not run_id:
                continue

            jobs = fetch_jobs_for_run(run_id, token)
            api_calls += 1
            failed = {
                job["name"]: job.get("html_url") or job.get("url", "")
                for job in jobs
                if job.get("conclusion") == "failure"
            }
            run_id_to_failed_jobs[run_id] = failed
            time.sleep(0.3)

        if len(run_id_to_failed_jobs) < CONSECUTIVE_FAILURES:
            continue

        run_ids = list(run_id_to_failed_jobs.keys())
        jobs_failed_in_all = set(run_id_to_failed_jobs[run_ids[0]].keys())
        for run_id in run_ids[1:]:
            jobs_failed_in_all &= set(run_id_to_failed_jobs[run_id].keys())

        workflow_jobs: list[dict] = []
        for job_name in jobs_failed_in_all:
            urls = [
                run_id_to_failed_jobs[run_id][job_name]
                for run in last_n
                if (run_id := run.get("id")) in run_id_to_failed_jobs and job_name in run_id_to_failed_jobs[run_id]
            ]
            item = {
                "job_name": job_name,
                "workflow_name": workflow_name,
                "consecutive_failures": CONSECUTIVE_FAILURES,
                "failing_job_urls": urls,
                "workflow_run_urls": [run.get("html_url") or run.get("url", "") for run in last_n],
            }
            results.append(item)
            workflow_jobs.append(item)
        workflow_cache[workflow_key] = {
            "workflow_name": workflow_name,
            "yaml_path": yaml_path,
            "run_ids": run_ids_for_cache,
            "jobs": workflow_jobs,
            "reused": False,
        }

        write_incremental()
        write_cache()

    print(
        (
            f"Done. {api_calls} API calls made, {len(results)} consistently-failing jobs found, "
            f"{cached_reused_workflows} workflows reused from cache."
        ),
        file=sys.stderr,
    )
    payload = {
        "description": (f"Jobs that failed in all of the last {CONSECUTIVE_FAILURES} workflow runs"),
        "total_jobs_affected": len(results),
        "jobs": results,
        "aggregate_run_id": aggregate_run_id,
        "cached_reused_workflows": cached_reused_workflows,
    }
    write_cache()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract jobs failing in last N runs.")
    parser.add_argument("workflow_filter_positional", nargs="?", default=None)
    parser.add_argument("--workflow-filter", default=None)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output path for failing jobs JSON (defaults to build_ci/ci_ticketing/create_tickets/failing_jobs.json)",
    )
    parser.add_argument(
        "--cache-input",
        default=None,
        help="Optional previous cache JSON path or directory containing failing_jobs_cache.json",
    )
    parser.add_argument(
        "--cache-output",
        default=None,
        help="Cache output path (defaults to build_ci/ci_ticketing/create_tickets/failing_jobs_cache.json)",
    )
    args = parser.parse_args()
    workflow_filter = args.workflow_filter or args.workflow_filter_positional
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "build_ci" / "ci_ticketing" / "create_tickets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_json) if args.output_json else (output_dir / "failing_jobs.json")
    cache_output_path = Path(args.cache_output) if args.cache_output else (output_dir / "failing_jobs_cache.json")
    cache_input_path: Path | None = None
    if args.cache_input:
        candidate = Path(args.cache_input)
        if candidate.is_dir():
            found = next(iter(candidate.rglob("failing_jobs_cache.json")), None)
            if found:
                cache_input_path = found
        elif candidate.exists():
            cache_input_path = candidate

    token = resolve_token()
    if not token:
        print(
            "Error: No GitHub token found.\n" "  Run `gh auth login` or set GITHUB_TOKEN.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Authenticated with GitHub.", file=sys.stderr)
    if workflow_filter:
        print(f"Filtering to workflow: {workflow_filter}", file=sys.stderr)

    previous_cache = load_cache(cache_input_path) if cache_input_path else {}
    workflow_data, aggregate_run_id = download_workflow_data()
    result = extract_failing_jobs(
        workflow_data,
        token,
        aggregate_run_id,
        workflow_filter,
        output_path=output_path,
        previous_cache=previous_cache,
        cache_output_path=cache_output_path,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {output_path}", file=sys.stderr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
