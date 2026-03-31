#!/usr/bin/env python3
"""M4: Create issues from deterministic aggregate failures and notify Slack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PRIMARY_REPO = "tenstorrent/tt-metal"
ISSUE_REPO_TEST = "ebanerjeeTT/issue_dump"
SLACK_CHANNEL_TEST = "C0APK6215B5"
MAX_LOG_CHARS_PER_RUN = 12000
MARKER = "===FINAL_DETERMINISTIC_FAILURE_DECISION==="


def log(message: str) -> None:
    print(f"[m4] {message}", flush=True)


def run(
    cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess[str]:
    proc_env = None
    if env:
        proc_env = os.environ.copy()
        proc_env.update(env)
    proc = subprocess.run(cmd, text=True, capture_output=capture, env=proc_env, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(shlex.quote(x) for x in cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return proc


def run_guarded_gh(tokens: list[str], *, github_token: str) -> subprocess.CompletedProcess[str]:
    command = " ".join(shlex.quote(tok) for tok in tokens)
    return run(
        [sys.executable, "tools/ci/guarded_gh.py", "--command", command],
        env={"GITHUB_TOKEN": github_token},
    )


def parse_job_url(url: str) -> tuple[int, int]:
    m = re.search(r"/actions/runs/(\d+)/job/(\d+)", url)
    if not m:
        raise ValueError(f"unsupported job URL format: {url}")
    return int(m.group(1)), int(m.group(2))


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs", [])
    if not isinstance(jobs, list):
        return []
    return [job for job in jobs if isinstance(job, dict)]


def fetch_job_logs(job_url: str, *, read_token: str) -> str:
    run_id, job_id = parse_job_url(job_url)
    proc = run_guarded_gh(
        ["gh", "run", "view", str(run_id), "--repo", PRIMARY_REPO, "--job", str(job_id), "--log"],
        github_token=read_token,
    )
    text = (proc.stdout or "").strip()
    if len(text) > MAX_LOG_CHARS_PER_RUN:
        return text[-MAX_LOG_CHARS_PER_RUN:]
    return text


def parse_agent_json(text: str) -> dict[str, Any]:
    idx = text.rfind(MARKER)
    if idx < 0:
        raise ValueError(f"marker not found: {MARKER}")
    payload = text[idx + len(MARKER) :].strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload)
        payload = re.sub(r"\s*```$", "", payload)
        payload = payload.strip()
    return json.loads(payload)


def decide_deterministic_failure(
    *,
    workflow_name: str,
    job_name: str,
    job_urls: list[str],
    logs: list[str],
    model: str,
) -> dict[str, Any]:
    log_sections: list[str] = []
    for idx, (url, text) in enumerate(zip(job_urls, logs, strict=False), start=1):
        log_sections.append(f"Run {idx} URL: {url}\n--- BEGIN LOG {idx} ---\n{text}\n--- END LOG {idx} ---")
    prompt = (
        "You are validating deterministic CI failure signatures.\n"
        "Criteria: deterministic only if the same job failed 3 times in a row with semantically identical terminal failure.\n\n"
        f"Workflow: {workflow_name}\n"
        f"Job: {job_name}\n"
        "Decide if all three runs share one identical terminal failure signature.\n"
        "If yes, extract a short normalized signature string and one concrete error excerpt.\n"
        "If uncertain, return deterministic=false and confidence=low.\n\n"
        + "\n\n".join(log_sections)
        + "\n\nOutput marker exactly on its own line:\n"
        + MARKER
        + "\nThen output compact JSON only:\n"
        + '{"deterministic":false,"confidence":"low|medium|high","signature":"","error_excerpt":"","reason":""}'
    )
    cmd = ["agent", "--trust", "-p", prompt]
    if model != "auto":
        cmd[1:1] = ["--model", model]
    proc = run(cmd)
    return parse_agent_json(proc.stdout or "")


def fingerprint_for(workflow_name: str, job_name: str, signature: str) -> str:
    raw = f"{workflow_name}\n{job_name}\n{signature.strip().lower()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def issue_marker(fingerprint: str) -> str:
    return f"Auto-triage-fingerprint: {fingerprint}"


def list_open_issue_bodies(*, issue_token: str) -> list[dict[str, Any]]:
    proc = run_guarded_gh(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            ISSUE_REPO_TEST,
            "--state",
            "open",
            "--limit",
            "200",
            "--json",
            "number,title,body,url",
        ],
        github_token=issue_token,
    )
    parsed = json.loads(proc.stdout or "[]")
    return parsed if isinstance(parsed, list) else []


def find_existing_issue_for_fingerprint(open_issues: list[dict[str, Any]], fingerprint: str) -> str | None:
    marker = issue_marker(fingerprint)
    for issue in open_issues:
        body = str(issue.get("body", ""))
        if marker in body:
            return str(issue.get("url", "")).strip() or None
    return None


def create_issue(
    *,
    issue_token: str,
    title: str,
    body: str,
) -> str:
    proc = run_guarded_gh(
        [
            "gh",
            "issue",
            "create",
            "--repo",
            ISSUE_REPO_TEST,
            "--title",
            title,
            "--body",
            body,
            "--label",
            "CI auto triage",
        ],
        github_token=issue_token,
    )
    return (proc.stdout or "").strip().splitlines()[-1].strip()


def post_slack_message(*, slack_token: str, channel_id: str, text: str) -> str:
    data = urllib.parse.urlencode({"channel": channel_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={
            "Authorization": f"Bearer {slack_token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Slack post failed: http_{exc.code}") from exc
    if not payload.get("ok", False):
        raise RuntimeError(f"Slack post failed: {payload.get('error', 'unknown_error')}")
    return str(payload.get("ts", "")).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 deterministic issue + Slack bootstrap automation.")
    parser.add_argument(
        "--failing-jobs-json",
        default="build_ci/ci_ticketing/create_tickets/failing_jobs.json",
    )
    parser.add_argument(
        "--output-json",
        default="build_ci/ci_ticketing/create_tickets/m4_issue_and_slack_result.json",
    )
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--model", default="auto")
    args = parser.parse_args()

    read_token = os.environ.get("AGGREGATE_READ_TOKEN", "").strip()
    issue_token = os.environ.get("ISSUE_WRITE_TOKEN", "").strip()
    slack_token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not read_token:
        print("AGGREGATE_READ_TOKEN is required", file=sys.stderr)
        return 2
    if not issue_token:
        print("ISSUE_WRITE_TOKEN is required", file=sys.stderr)
        return 2
    if not slack_token:
        print("SLACK_BOT_TOKEN is required", file=sys.stderr)
        return 2
    if not os.environ.get("CURSOR_API_KEY", "").strip():
        print("CURSOR_API_KEY is required", file=sys.stderr)
        return 2

    jobs_path = Path(args.failing_jobs_json)
    if not jobs_path.exists():
        print(f"Missing failing jobs input: {jobs_path}", file=sys.stderr)
        return 2
    candidates = load_candidates(jobs_path)
    if not candidates:
        log("No failing jobs found; nothing to do.")
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps({"created": [], "skipped": []}, indent=2), encoding="utf-8")
        return 0

    open_issues = list_open_issue_bodies(issue_token=issue_token)
    created: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for candidate in candidates[: max(args.max_candidates, 0)]:
        job_name = str(candidate.get("job_name", "")).strip()
        workflow_name = str(candidate.get("workflow_name", "")).strip()
        urls_raw = candidate.get("failing_job_urls", [])
        if not job_name or not workflow_name or not isinstance(urls_raw, list):
            continue
        job_urls = [str(u).strip() for u in urls_raw if str(u).strip()][:3]
        if len(job_urls) < 3:
            skipped.append({"job_name": job_name, "reason": "requires_3_failing_runs"})
            continue
        log(f"Validating deterministic signature for workflow={workflow_name} job={job_name}")
        logs = [fetch_job_logs(url, read_token=read_token) for url in job_urls]
        decision = decide_deterministic_failure(
            workflow_name=workflow_name,
            job_name=job_name,
            job_urls=job_urls,
            logs=logs,
            model=args.model,
        )
        deterministic = bool(decision.get("deterministic", False))
        confidence = str(decision.get("confidence", "low")).strip().lower()
        signature = str(decision.get("signature", "")).strip()
        excerpt = str(decision.get("error_excerpt", "")).strip()
        reason = str(decision.get("reason", "")).strip()
        if not deterministic or confidence != "high" or not signature:
            skipped.append(
                {
                    "job_name": job_name,
                    "workflow_name": workflow_name,
                    "reason": "not_deterministic_or_low_confidence",
                    "decision": decision,
                }
            )
            continue

        fp = fingerprint_for(workflow_name, job_name, signature)
        existing = find_existing_issue_for_fingerprint(open_issues, fp)
        if existing:
            skipped.append(
                {
                    "job_name": job_name,
                    "workflow_name": workflow_name,
                    "reason": f"already_tracked:{existing}",
                    "fingerprint": fp,
                }
            )
            continue

        issue_title = f"CI auto triage: deterministic failure in {job_name}"
        issue_body = (
            f"Workflow: `{workflow_name}`\n"
            f"Job: `{job_name}`\n\n"
            f"Failure signature: `{signature}`\n\n"
            f"Error excerpt:\n"
            f"```\n{excerpt or reason or 'No excerpt captured'}\n```\n\n"
            f"Failing job URLs (last 3):\n"
            + "\n".join(f"- {url}" for url in job_urls)
            + "\n\n"
            + issue_marker(fp)
            + "\n"
        )
        issue_url = create_issue(issue_token=issue_token, title=issue_title, body=issue_body)
        open_issues.append({"url": issue_url, "body": issue_body})
        slack_text = (
            f"CI auto triage detected a deterministic failure (3x in a row) for `{job_name}` in `{workflow_name}`.\n"
            f"Issue: {issue_url}\n"
            f"Fingerprint: `{fp}`\n"
            f"Latest run: {job_urls[0]}"
        )
        slack_ts = post_slack_message(slack_token=slack_token, channel_id=SLACK_CHANNEL_TEST, text=slack_text)
        created.append(
            {
                "workflow_name": workflow_name,
                "job_name": job_name,
                "fingerprint": fp,
                "issue_url": issue_url,
                "slack_channel": SLACK_CHANNEL_TEST,
                "slack_ts": slack_ts,
                "job_urls": job_urls,
            }
        )
        log(f"Created issue and posted Slack bootstrap: {issue_url} (ts={slack_ts})")

    output = {
        "created": created,
        "skipped": skipped,
        "candidate_count": len(candidates),
        "processed_count": min(len(candidates), max(args.max_candidates, 0)),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"created_count": len(created), "skipped_count": len(skipped)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
