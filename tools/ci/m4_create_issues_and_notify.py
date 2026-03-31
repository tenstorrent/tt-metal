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
MARKER = "===FINAL_M4_REVIEW_DECISION==="


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


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


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


def load_command_spec(path: str, fallback: str) -> str:
    p = Path(path)
    if not p.exists():
        return fallback
    return p.read_text(encoding="utf-8")


def fetch_recent_slack_messages(*, slack_token: str, channel_id: str, limit: int = 20) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"channel": channel_id, "limit": str(limit)})
    req = urllib.request.Request(
        f"https://slack.com/api/conversations.history?{query}",
        headers={"Authorization": f"Bearer {slack_token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Slack history fetch failed: http_{exc.code}") from exc
    if not payload.get("ok", False):
        raise RuntimeError(f"Slack history fetch failed: {payload.get('error', 'unknown_error')}")
    messages = payload.get("messages", [])
    return messages if isinstance(messages, list) else []


def build_open_issue_context(open_issues: list[dict[str, Any]], cap: int = 20) -> str:
    lines: list[str] = []
    for issue in open_issues[:cap]:
        number = issue.get("number")
        title = str(issue.get("title", "")).strip()
        url = str(issue.get("url", "")).strip()
        body = str(issue.get("body", "")).strip()
        marker_match = re.search(r"Auto-triage-fingerprint:\s*([A-Za-z0-9_-]+)", body)
        fp = marker_match.group(1) if marker_match else ""
        lines.append(f"- #{number} {title} ({url}) fingerprint={fp}")
    return "\n".join(lines) if lines else "(none)"


def build_recent_slack_context(messages: list[dict[str, Any]], cap: int = 10) -> str:
    lines: list[str] = []
    for msg in messages[:cap]:
        ts = str(msg.get("ts", "")).strip()
        user = str(msg.get("user", "") or msg.get("bot_id", "")).strip()
        text = str(msg.get("text", "")).strip().replace("\n", " ")
        if len(text) > 220:
            text = text[:220] + "..."
        lines.append(f"- ts={ts} user={user} text={text}")
    return "\n".join(lines) if lines else "(none)"


def find_exact_previous_decision(
    review_entries: list[dict[str, Any]],
    *,
    workflow_name: str,
    job_name: str,
    job_urls: list[str],
) -> dict[str, Any] | None:
    for entry in review_entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("workflow_name", "")).strip() != workflow_name:
            continue
        if str(entry.get("job_name", "")).strip() != job_name:
            continue
        prev_decision = entry.get("decision")
        if not isinstance(prev_decision, dict):
            continue
        prev_urls = entry.get("job_urls")
        if isinstance(prev_urls, list) and [str(u).strip() for u in prev_urls] == job_urls:
            return prev_decision
    return None


def agent_review_and_prepare_outputs(
    *,
    workflow_name: str,
    job_name: str,
    job_urls: list[str],
    logs: list[str],
    open_issue_context: str,
    recent_slack_context: str,
    model: str,
) -> dict[str, Any]:
    create_ticket_spec = load_command_spec(
        ".cursor/commands/ci/ci-create-tickets.md",
        "Create deterministic tickets only when the same terminal failure appears in all 3 logs.",
    )
    slack_draft_spec = load_command_spec(
        ".cursor/commands/ci/ci-draft-slack-ci-issue.md",
        "Draft concise Slack incident updates with direct links.",
    )
    log_sections: list[str] = []
    for idx, (url, text) in enumerate(zip(job_urls, logs, strict=False), start=1):
        log_sections.append(f"Run {idx} URL: {url}\n--- BEGIN LOG {idx} ---\n{text}\n--- END LOG {idx} ---")
    prompt = (
        "You are the M4 reviewer for deterministic CI failures.\n"
        "You MUST manually compare all three logs and determine if terminal failure signatures are the same.\n"
        "You MUST also review existing open issues and recent Slack messages before drafting outputs.\n"
        "Criteria: deterministic only if the same job failed 3 times in a row with semantically identical terminal failure.\n\n"
        f"Workflow: {workflow_name}\n"
        f"Job: {job_name}\n"
        "Open issues context:\n"
        f"{open_issue_context}\n\n"
        "Recent Slack messages context:\n"
        f"{recent_slack_context}\n\n"
        "Ticket command guidance:\n"
        f"{create_ticket_spec}\n\n"
        "Slack draft guidance:\n"
        f"{slack_draft_spec}\n\n"
        "Decide if all three runs share one identical terminal failure signature.\n"
        "If yes, draft issue title/body and draft initial Slack message text.\n"
        "If uncertain, set create_issue=false and draft_slack=false.\n\n"
        + "\n\n".join(log_sections)
        + "\n\nOutput marker exactly on its own line:\n"
        + MARKER
        + "\nThen output compact JSON only:\n"
        + (
            '{"deterministic":false,"confidence":"low|medium|high","signature":"","error_excerpt":"","reason":"",'
            '"create_issue":false,"draft_slack":false,"issue_title":"","issue_body":"","slack_text":""}'
        )
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
    parser.add_argument(
        "--review-cache-json",
        default="build_ci/ci_ticketing/create_tickets/m4_review_cache.json",
    )
    parser.add_argument(
        "--previous-artifact-dir",
        default="build_ci/m4_cache_prev",
    )
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-new-issues", type=int, default=1)
    parser.add_argument(
        "--max-agent-reviews",
        type=int,
        default=3,
        help="Maximum number of fresh agent reviews (cache reuses do not count).",
    )
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

    previous_dir = Path(args.previous_artifact_dir)
    prev_review_path = previous_dir / "m4_review_cache.json"
    if not prev_review_path.exists():
        matches = list(previous_dir.rglob("m4_review_cache.json"))
        if matches:
            prev_review_path = matches[0]
    prev_review_payload = load_optional_json(prev_review_path) or {}
    prev_review_entries = prev_review_payload.get("entries", [])
    if not isinstance(prev_review_entries, list):
        prev_review_entries = []
    open_issues = list_open_issue_bodies(issue_token=issue_token)
    recent_slack_messages = fetch_recent_slack_messages(
        slack_token=slack_token, channel_id=SLACK_CHANNEL_TEST, limit=20
    )
    open_issue_context = build_open_issue_context(open_issues)
    recent_slack_context = build_recent_slack_context(recent_slack_messages)
    created: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    review_cache_entries: list[dict[str, Any]] = []
    max_new_issues = max(args.max_new_issues, 0)
    max_agent_reviews = max(args.max_agent_reviews, 0)
    fresh_agent_reviews = 0

    for candidate in candidates[: max(args.max_candidates, 0)]:
        if len(created) >= max_new_issues:
            skipped.append({"reason": f"max_new_issues_reached:{max_new_issues}"})
            break
        job_name = str(candidate.get("job_name", "")).strip()
        workflow_name = str(candidate.get("workflow_name", "")).strip()
        urls_raw = candidate.get("failing_job_urls", [])
        if not job_name or not workflow_name or not isinstance(urls_raw, list):
            continue
        job_urls = [str(u).strip() for u in urls_raw if str(u).strip()][:3]
        if len(job_urls) < 3:
            skipped.append({"job_name": job_name, "reason": "requires_3_failing_runs"})
            continue
        decision: dict[str, Any]
        reused_cache = False
        prior = find_exact_previous_decision(
            prev_review_entries,
            workflow_name=workflow_name,
            job_name=job_name,
            job_urls=job_urls,
        )
        if prior:
            decision = prior
            reused_cache = True
            log(f"Reused cached review decision for workflow={workflow_name} job={job_name}")
        else:
            if fresh_agent_reviews >= max_agent_reviews:
                skipped.append(
                    {
                        "job_name": job_name,
                        "workflow_name": workflow_name,
                        "reason": f"max_agent_reviews_reached:{max_agent_reviews}",
                    }
                )
                break
            log(f"Running agent review for workflow={workflow_name} job={job_name}")
            logs = [fetch_job_logs(url, read_token=read_token) for url in job_urls]
            decision = agent_review_and_prepare_outputs(
                workflow_name=workflow_name,
                job_name=job_name,
                job_urls=job_urls,
                logs=logs,
                open_issue_context=open_issue_context,
                recent_slack_context=recent_slack_context,
                model=args.model,
            )
            fresh_agent_reviews += 1
        review_cache_entries.append(
            {
                "workflow_name": workflow_name,
                "job_name": job_name,
                "job_urls": job_urls,
                "decision": decision,
                "reused_cache": reused_cache,
            }
        )
        deterministic = bool(decision.get("deterministic", False))
        confidence = str(decision.get("confidence", "low")).strip().lower()
        signature = str(decision.get("signature", "")).strip()
        excerpt = str(decision.get("error_excerpt", "")).strip()
        reason = str(decision.get("reason", "")).strip()
        create_issue_flag = bool(decision.get("create_issue", False))
        draft_slack_flag = bool(decision.get("draft_slack", False))
        issue_title = str(decision.get("issue_title", "")).strip()
        issue_body_from_agent = str(decision.get("issue_body", "")).strip()
        slack_text = str(decision.get("slack_text", "")).strip()
        if not deterministic or confidence != "high" or not signature or not create_issue_flag:
            skipped.append(
                {
                    "job_name": job_name,
                    "workflow_name": workflow_name,
                    "reason": "agent_rejected_or_low_confidence",
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

        if not issue_title:
            issue_title = f"CI auto triage: deterministic failure in {job_name}"
        if not issue_body_from_agent:
            issue_body_from_agent = (
                f"Workflow: `{workflow_name}`\n"
                f"Job: `{job_name}`\n\n"
                f"Failure signature: `{signature}`\n\n"
                f"Error excerpt:\n"
                f"```\n{excerpt or reason or 'No excerpt captured'}\n```\n\n"
                f"Failing job URLs (last 3):\n" + "\n".join(f"- {url}" for url in job_urls)
            )
        marker = issue_marker(fp)
        issue_body = issue_body_from_agent
        if marker not in issue_body:
            issue_body = issue_body.rstrip() + "\n\n" + marker + "\n"
        if not draft_slack_flag:
            skipped.append(
                {
                    "job_name": job_name,
                    "workflow_name": workflow_name,
                    "reason": "agent_declined_slack_draft",
                    "fingerprint": fp,
                    "decision": decision,
                }
            )
            continue
        issue_url = create_issue(issue_token=issue_token, title=issue_title, body=issue_body)
        open_issues.append({"url": issue_url, "body": issue_body})
        if not slack_text:
            slack_text = (
                f"CI auto triage detected a deterministic failure (3x in a row) for `{job_name}` in `{workflow_name}`.\n"
                f"Issue: {issue_url}\n"
                f"Fingerprint: `{fp}`\n"
                f"Latest run: {job_urls[0]}"
            )
        if issue_url not in slack_text:
            slack_text = slack_text.rstrip() + f"\nIssue: {issue_url}"
        slack_ts = post_slack_message(slack_token=slack_token, channel_id=SLACK_CHANNEL_TEST, text=slack_text)
        recent_slack_messages.insert(0, {"ts": slack_ts, "user": "m4-auto-triage", "text": slack_text})
        recent_slack_context = build_recent_slack_context(recent_slack_messages)
        open_issue_context = build_open_issue_context(open_issues)
        created.append(
            {
                "workflow_name": workflow_name,
                "job_name": job_name,
                "fingerprint": fp,
                "issue_url": issue_url,
                "slack_channel": SLACK_CHANNEL_TEST,
                "slack_ts": slack_ts,
                "job_urls": job_urls,
                "agent_decision": decision,
            }
        )
        log(f"Created issue and posted Slack bootstrap: {issue_url} (ts={slack_ts})")

    output = {
        "created": created,
        "skipped": skipped,
        "candidate_count": len(candidates),
        "processed_count": min(len(candidates), max(args.max_candidates, 0)),
        "max_new_issues": max_new_issues,
        "max_agent_reviews": max_agent_reviews,
        "fresh_agent_reviews": fresh_agent_reviews,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    review_cache_payload = {
        "version": 1,
        "entries": review_cache_entries,
    }
    review_cache_path = Path(args.review_cache_json)
    review_cache_path.parent.mkdir(parents=True, exist_ok=True)
    review_cache_path.write_text(json.dumps(review_cache_payload, indent=2), encoding="utf-8")
    print(json.dumps({"created_count": len(created), "skipped_count": len(skipped)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
