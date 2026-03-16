#!/usr/bin/env python3
"""
Extract jobs that failed in the last N consecutive workflow runs.

The script downloads workflow-data from the latest aggregate-workflow-data run,
filters to workflows with N consecutive failures, then fetches run jobs from
the GitHub API to find which specific jobs failed in all N runs.

Usage:
  python tools/ci/extract_failing_jobs.py
  python tools/ci/extract_failing_jobs.py t3000-unit-tests

Optional env vars:
  CONSECUTIVE_FAILURES   Number of consecutive failures required (default: 3)
  GITHUB_OWNER           Default: tenstorrent
  GITHUB_REPO            Default: tt-metal
"""

from __future__ import annotations

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


def download_workflow_data() -> list:
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
            return json.load(f)


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
    workflow_filter: str | None = None,
    output_path: Path | None = None,
) -> dict:
    results = []
    api_calls = 0

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

    write_incremental()

    for workflow_name, runs in workflow_data:
        if not runs:
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

        for job_name in jobs_failed_in_all:
            urls = [
                run_id_to_failed_jobs[run_id][job_name]
                for run in last_n
                if (run_id := run.get("id")) in run_id_to_failed_jobs and job_name in run_id_to_failed_jobs[run_id]
            ]
            results.append(
                {
                    "job_name": job_name,
                    "workflow_name": workflow_name,
                    "consecutive_failures": CONSECUTIVE_FAILURES,
                    "failing_job_urls": urls,
                    "workflow_run_urls": [run.get("html_url") or run.get("url", "") for run in last_n],
                }
            )

        write_incremental()

    print(
        f"Done. {api_calls} API calls made, {len(results)} consistently-failing jobs found.",
        file=sys.stderr,
    )
    return {
        "description": (f"Jobs that failed in all of the last {CONSECUTIVE_FAILURES} workflow runs"),
        "total_jobs_affected": len(results),
        "jobs": results,
    }


def main() -> None:
    workflow_filter = sys.argv[1] if len(sys.argv) > 1 else None
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / ".auto_triage" / "output" / "ci_ticketing" / "create_tickets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "failing_jobs.json"

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

    workflow_data = download_workflow_data()
    result = extract_failing_jobs(
        workflow_data,
        token,
        workflow_filter,
        output_path=output_path,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {output_path}", file=sys.stderr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
