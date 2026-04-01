#!/usr/bin/env python3
"""Issue lifecycle stage: close/update stale CI auto-triage tickets."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

RUN_JOB_URL_RE = re.compile(r"https://github\.com/tenstorrent/tt-metal/actions/runs/(\d+)/job/(\d+)")
FINAL_MARKER = "===FINAL_ISSUE_LIFECYCLE_DECISION==="


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def now_iso() -> str:
    return now_utc().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso_utc(text: str | None) -> dt.datetime | None:
    if not text:
        return None
    value = text.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = None
    if env:
        merged_env = os.environ.copy()
        merged_env.update(env)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env=merged_env)
    if proc.returncode != 0:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process stale CI auto-triage issues.")
    p.add_argument("--state-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--summary-md", required=True)
    p.add_argument("--issue-repo", default="ebanerjeeTT/issue_dump")
    p.add_argument("--source-repo", default="tenstorrent/tt-metal")
    p.add_argument("--processed-hours", type=float, default=24.0)
    p.add_argument("--max-issues", type=int, default=3)
    p.add_argument("--model", default="auto")
    return p.parse_args()


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "updated_at_utc": now_iso(), "items": [], "issue_lifecycle": {"issues": {}}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        payload = {"version": 1, "updated_at_utc": now_iso(), "items": []}
    issue_lifecycle = payload.get("issue_lifecycle")
    if not isinstance(issue_lifecycle, dict):
        issue_lifecycle = {}
    issues = issue_lifecycle.get("issues")
    if not isinstance(issues, dict):
        issues = {}
    issue_lifecycle["issues"] = issues
    payload["issue_lifecycle"] = issue_lifecycle
    return payload


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = now_iso()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def parse_json_after_marker(text: str, marker: str) -> dict[str, Any]:
    idx = text.find(marker)
    if idx < 0:
        raise ValueError(f"marker not found: {marker}")
    payload = text[idx + len(marker) :].strip()
    if not payload:
        raise ValueError("empty payload after marker")
    obj = json.loads(payload)
    if not isinstance(obj, dict):
        raise ValueError("payload must be object")
    return obj


def list_open_ci_issues(issue_repo: str, issue_token: str) -> list[dict[str, Any]]:
    proc = run_guarded_gh(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            issue_repo,
            "--state",
            "open",
            "--label",
            "CI auto triage",
            "--limit",
            "100",
            "--json",
            "number,title,updatedAt,url",
        ],
        github_token=issue_token,
    )
    payload = json.loads(proc.stdout or "[]")
    return payload if isinstance(payload, list) else []


def issue_view(issue_repo: str, issue_number: int, issue_token: str) -> dict[str, Any]:
    proc = run_guarded_gh(
        [
            "gh",
            "issue",
            "view",
            "--repo",
            issue_repo,
            str(issue_number),
            "--json",
            "number,title,body,url,labels,comments",
        ],
        github_token=issue_token,
    )
    payload = json.loads(proc.stdout or "{}")
    return payload if isinstance(payload, dict) else {}


def extract_latest_run_job_url(issue_payload: dict[str, Any]) -> tuple[int, int] | None:
    corpus = [str(issue_payload.get("body", ""))]
    comments = issue_payload.get("comments", [])
    if isinstance(comments, list):
        for row in comments[-20:]:
            if isinstance(row, dict):
                corpus.append(str(row.get("body", "")))
    matches = []
    for text in corpus:
        matches.extend(RUN_JOB_URL_RE.findall(text))
    if not matches:
        return None
    run_id, job_id = matches[-1]
    return int(run_id), int(job_id)


def run_view(run_id: int, source_repo: str, main_token: str) -> dict[str, Any]:
    proc = run_guarded_gh(
        [
            "gh",
            "run",
            "view",
            "--repo",
            source_repo,
            str(run_id),
            "--json",
            "databaseId,url,workflowName,jobs,conclusion,headBranch,createdAt",
        ],
        github_token=main_token,
    )
    payload = json.loads(proc.stdout or "{}")
    return payload if isinstance(payload, dict) else {}


def find_job_name(run_payload: dict[str, Any], job_id: int) -> str:
    jobs = run_payload.get("jobs", [])
    if not isinstance(jobs, list):
        return ""
    for job in jobs:
        if not isinstance(job, dict):
            continue
        if int(job.get("databaseId", 0) or 0) == job_id:
            return str(job.get("name", "")).strip()
    return ""


def list_recent_runs_for_workflow(source_repo: str, workflow_name: str, main_token: str) -> list[dict[str, Any]]:
    proc = run_guarded_gh(
        [
            "gh",
            "run",
            "list",
            "--repo",
            source_repo,
            "--workflow",
            workflow_name,
            "--branch",
            "main",
            "--status",
            "completed",
            "--limit",
            "30",
            "--json",
            "databaseId,url,conclusion,createdAt",
        ],
        github_token=main_token,
    )
    payload = json.loads(proc.stdout or "[]")
    return payload if isinstance(payload, list) else []


def pick_latest_three_job_instances(
    source_repo: str,
    main_token: str,
    workflow_runs: list[dict[str, Any]],
    target_job_name: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in workflow_runs:
        if not isinstance(row, dict):
            continue
        run_id = int(row.get("databaseId", 0) or 0)
        if run_id <= 0:
            continue
        details = run_view(run_id, source_repo, main_token)
        jobs = details.get("jobs", [])
        if not isinstance(jobs, list):
            continue
        matched = None
        for job in jobs:
            if not isinstance(job, dict):
                continue
            name = str(job.get("name", "")).strip()
            if name == target_job_name:
                matched = job
                break
        if matched is None:
            continue
        selected.append(
            {
                "run_id": run_id,
                "run_url": str(details.get("url", "")).strip(),
                "run_conclusion": str(details.get("conclusion", "")).strip(),
                "job_id": int(matched.get("databaseId", 0) or 0),
                "job_name": str(matched.get("name", "")).strip(),
                "job_conclusion": str(matched.get("conclusion", "")).strip(),
                "job_url": str(matched.get("url", "")).strip(),
            }
        )
        if len(selected) >= 3:
            break
    return selected


def fetch_job_log_excerpt(source_repo: str, main_token: str, run_id: int, job_id: int) -> str:
    proc = run_guarded_gh(
        [
            "gh",
            "run",
            "view",
            "--repo",
            source_repo,
            str(run_id),
            "--job",
            str(job_id),
            "--log",
        ],
        github_token=main_token,
    )
    text = proc.stdout or ""
    lines = text.splitlines()
    tail = "\n".join(lines[-220:])
    if len(tail) > 12000:
        tail = tail[-12000:]
    return tail


def agent_decide_issue_action(
    *,
    issue_payload: dict[str, Any],
    target_job_name: str,
    selected_runs: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    issue_summary = {
        "number": int(issue_payload.get("number", 0) or 0),
        "title": str(issue_payload.get("title", "")).strip(),
        "url": str(issue_payload.get("url", "")).strip(),
        "body": str(issue_payload.get("body", ""))[:4000],
    }
    prompt_payload = {
        "issue": issue_summary,
        "target_job_name": target_job_name,
        "recent_runs": selected_runs,
        "policy": {
            "scope": "CI auto triage lifecycle review",
            "close_when": [
                "problem is resolved/passing",
                "root cause has changed from ticketed issue to a materially new failure",
            ],
            "update_when": [
                "ticket still valid but body/links are stale",
                "deterministic failure appears to continue",
            ],
            "unchanged_when": ["ticket remains valid and already current"],
            "nd_hint": "transient infra flakes like disconnects may be non-deterministic and should not force close",
            "minimal_update_rule": "prefer in-place minimal updates for last updated + refreshed links/signature",
        },
    }
    prompt = (
        "You are deciding CI issue lifecycle actions from ticket text and latest run evidence.\n"
        "Token-efficiency rule: do one short pass and output only final JSON contract.\n"
        "Use provided logs/runs to decide one action: close, update, unchanged.\n"
        "At end print marker exactly on its own line:\n"
        f"{FINAL_MARKER}\n"
        "Then print only JSON object:\n"
        "{\n"
        '  "action": "close|update|unchanged",\n'
        '  "reason": "brief",\n'
        '  "comment": "brief markdown",\n'
        '  "current_signature": "optional short signature",\n'
        '  "most_recent_job_url": "optional url",\n'
        '  "last_three_job_urls": ["url1","url2","url3"]\n'
        "}\n\n"
        f"Input JSON:\n{json.dumps(prompt_payload, ensure_ascii=True)}\n"
    )
    cmd = ["agent", "--trust", "-p", prompt]
    if model and model != "auto":
        cmd.extend(["--model", model])
    proc = run(cmd)
    parsed = parse_json_after_marker(proc.stdout or "", FINAL_MARKER)
    action = str(parsed.get("action", "")).strip().lower()
    if action not in {"close", "update", "unchanged"}:
        raise ValueError(f"invalid action from agent: {action}")
    parsed["action"] = action
    return parsed


def update_issue_body_minimal(
    existing_body: str,
    *,
    most_recent_job_url: str,
    last_three_job_urls: list[str],
    current_signature: str,
) -> str:
    body = existing_body.strip()
    section = [
        "## Auto Triage Refresh",
        f"Last updated: {now_utc().date().isoformat()}",
    ]
    if most_recent_job_url:
        section.append(f"Most recent failing job URL: {most_recent_job_url}")
    if last_three_job_urls:
        section.append("Last 3 reviewed job URLs:")
        for link in last_three_job_urls[:3]:
            section.append(f"- {link}")
    if current_signature:
        section.append(f"Current signature: `{current_signature[:300]}`")
    new_section = "\n".join(section).strip()

    marker = "## Auto Triage Refresh"
    if marker in body:
        head = body.split(marker, 1)[0].rstrip()
        return f"{head}\n\n{new_section}\n"
    if not body:
        return f"{new_section}\n"
    return f"{body}\n\n{new_section}\n"


def comment_issue(issue_repo: str, issue_number: int, issue_token: str, body: str) -> None:
    run_guarded_gh(
        [
            "gh",
            "issue",
            "comment",
            "--repo",
            issue_repo,
            str(issue_number),
            "--body",
            body,
        ],
        github_token=issue_token,
    )


def close_issue(issue_repo: str, issue_number: int, issue_token: str) -> None:
    run_guarded_gh(
        [
            "gh",
            "issue",
            "close",
            "--repo",
            issue_repo,
            str(issue_number),
        ],
        github_token=issue_token,
    )


def edit_issue_body(issue_repo: str, issue_number: int, issue_token: str, body: str) -> None:
    run_guarded_gh(
        [
            "gh",
            "issue",
            "edit",
            "--repo",
            issue_repo,
            str(issue_number),
            "--body",
            body,
        ],
        github_token=issue_token,
    )


def summarize_md(result: dict[str, Any]) -> str:
    lines = [
        "## Issue Lifecycle Summary",
        "",
        f"- Candidates considered: {result.get('candidates_considered', 0)}",
        f"- Processed: {result.get('processed_count', 0)}",
        f"- Closed: {result.get('closed_count', 0)}",
        f"- Updated: {result.get('updated_count', 0)}",
        f"- Unchanged: {result.get('unchanged_count', 0)}",
        "",
        "## Decisions",
    ]
    for row in result.get("decisions", [])[:50]:
        if not isinstance(row, dict):
            continue
        lines.append(f"- #{row.get('issue_number')} action=`{row.get('action')}` reason={row.get('reason','')[:140]}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    issue_token = os.environ.get("ISSUE_WRITE_TOKEN", "").strip() or os.environ.get("GITHUB_TOKEN", "").strip()
    main_token = os.environ.get("MAIN_REPO_READ_TOKEN", "").strip() or os.environ.get("GITHUB_TOKEN", "").strip()
    if not issue_token:
        print("ISSUE_WRITE_TOKEN or GITHUB_TOKEN is required", file=sys.stderr)
        return 2
    if not main_token:
        print("MAIN_REPO_READ_TOKEN or GITHUB_TOKEN is required", file=sys.stderr)
        return 2

    state_path = Path(args.state_json)
    state = load_state(state_path)
    lifecycle_state = state.setdefault("issue_lifecycle", {})
    issues_state = lifecycle_state.setdefault("issues", {})
    if not isinstance(issues_state, dict):
        issues_state = {}
        lifecycle_state["issues"] = issues_state

    open_issues = list_open_ci_issues(args.issue_repo, issue_token)
    now = now_utc()
    threshold = dt.timedelta(hours=max(0.0, args.processed_hours))
    candidates: list[dict[str, Any]] = []
    for row in open_issues:
        if not isinstance(row, dict):
            continue
        issue_number = int(row.get("number", 0) or 0)
        if issue_number <= 0:
            continue
        tracked = issues_state.get(str(issue_number), {})
        last_processed = parse_iso_utc(str(tracked.get("last_processed_utc", "")) if isinstance(tracked, dict) else "")
        if last_processed and now - last_processed < threshold:
            continue
        candidates.append(row)
    candidates.sort(key=lambda x: str(x.get("updatedAt", "")))
    candidates = candidates[: max(0, args.max_issues)]

    result: dict[str, Any] = {
        "generated_at_utc": now_iso(),
        "candidates_considered": len(candidates),
        "processed_count": 0,
        "closed_count": 0,
        "updated_count": 0,
        "unchanged_count": 0,
        "decisions": [],
        "skipped": [],
    }

    for issue_row in candidates:
        issue_number = int(issue_row.get("number", 0) or 0)
        try:
            payload = issue_view(args.issue_repo, issue_number, issue_token)
            target = extract_latest_run_job_url(payload)
            if target is None:
                result["skipped"].append({"issue_number": issue_number, "reason": "missing_run_job_url"})
                continue
            seed_run_id, seed_job_id = target
            seed_run = run_view(seed_run_id, args.source_repo, main_token)
            workflow_name = str(seed_run.get("workflowName", "")).strip()
            target_job_name = find_job_name(seed_run, seed_job_id)
            if not workflow_name or not target_job_name:
                result["skipped"].append({"issue_number": issue_number, "reason": "unable_to_resolve_target_job"})
                continue
            workflow_runs = list_recent_runs_for_workflow(args.source_repo, workflow_name, main_token)
            selected = pick_latest_three_job_instances(args.source_repo, main_token, workflow_runs, target_job_name)
            if len(selected) < 3:
                result["skipped"].append({"issue_number": issue_number, "reason": "fewer_than_three_runs"})
                continue
            for row in selected:
                row["log_excerpt"] = fetch_job_log_excerpt(
                    args.source_repo,
                    main_token,
                    int(row["run_id"]),
                    int(row["job_id"]),
                )
            decision = agent_decide_issue_action(
                issue_payload=payload,
                target_job_name=target_job_name,
                selected_runs=selected,
                model=args.model,
            )
            action = str(decision.get("action", "unchanged"))
            comment = str(decision.get("comment", "")).strip()
            most_recent_job_url = str(decision.get("most_recent_job_url", "")).strip() or str(selected[0]["job_url"])
            last_three_job_urls = [
                str(x)
                for x in (decision.get("last_three_job_urls") or [])
                if isinstance(x, str) and x.startswith("http")
            ] or [str(row["job_url"]) for row in selected[:3]]
            current_signature = str(decision.get("current_signature", "")).strip()
            if action == "close":
                close_comment = comment or (
                    "Auto-triage lifecycle review: closing as latest 3 runs suggest this ticket is no longer the "
                    "right active failure signature.\n\n"
                    f"Most recent reviewed job: {most_recent_job_url}"
                )
                comment_issue(args.issue_repo, issue_number, issue_token, close_comment)
                close_issue(args.issue_repo, issue_number, issue_token)
                result["closed_count"] += 1
            elif action == "update":
                updated_body = update_issue_body_minimal(
                    str(payload.get("body", "")),
                    most_recent_job_url=most_recent_job_url,
                    last_three_job_urls=last_three_job_urls,
                    current_signature=current_signature,
                )
                edit_issue_body(args.issue_repo, issue_number, issue_token, updated_body)
                if comment:
                    comment_issue(args.issue_repo, issue_number, issue_token, comment)
                result["updated_count"] += 1
            else:
                result["unchanged_count"] += 1
            issues_state[str(issue_number)] = {
                "last_processed_utc": now_iso(),
                "last_action": action,
                "last_reason": str(decision.get("reason", ""))[:500],
            }
            result["processed_count"] += 1
            result["decisions"].append(
                {
                    "issue_number": issue_number,
                    "action": action,
                    "reason": str(decision.get("reason", "")),
                    "workflow_name": workflow_name,
                    "target_job_name": target_job_name,
                    "most_recent_job_url": most_recent_job_url,
                }
            )
        except Exception as exc:
            result["skipped"].append({"issue_number": issue_number, "reason": f"error:{exc}"})

    save_state(state_path, state)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_path = Path(args.summary_md)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summarize_md(result), encoding="utf-8")
    print(json.dumps({"processed_count": result["processed_count"], "closed_count": result["closed_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
