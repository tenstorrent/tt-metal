#!/usr/bin/env python3
"""Execute structured stale-disable actions with deterministic git/gh steps."""

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


REPO = "tenstorrent/tt-metal"
GUARDED_GH = [sys.executable, "tools/ci/guarded_gh.py"]
ALLOWED_STATUS = {
    "new",
    "planned",
    "pr_open",
    "kickoff_running",
    "kickoff_failed_new_failure",
    "completed",
    "needs_human",
    "paused",
}


def run(cmd: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )


def run_guarded_gh(tokens: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    command_str = " ".join(shlex.quote(tok) for tok in tokens)
    return run([*GUARDED_GH, "--command", command_str], capture=capture)


def slugify(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9-]+", "-", text).strip("-").lower()
    return clean[:48] if clean else "auto"


def branch_name(action: dict[str, Any]) -> str:
    hint = str(action.get("branch_name_hint", "")).strip()
    if hint:
        return slugify(hint)
    issue = action.get("issue_number")
    ts = str(action.get("source_slack_ts", "")).replace(".", "")
    return f"ci-disable-test-{issue}-{ts[-8:]}"


def ensure_no_duplicate_open_pr(source_ts: str) -> str | None:
    marker = f"Auto-disable-source-ts: {source_ts}"
    prs = run_guarded_gh(["gh", "pr", "list", "--repo", REPO, "--state", "open", "--json", "number,url,body"])
    items = json.loads(prs.stdout)
    for pr in items:
        if marker in (pr.get("body") or ""):
            return pr.get("url")
    return None


def parse_agent_json_after_marker(text: str, marker: str) -> dict[str, Any]:
    idx = text.rfind(marker)
    if idx < 0:
        raise ValueError(f"marker not found: {marker}")
    payload = text[idx + len(marker) :].strip()
    return json.loads(payload)


def run_disable_editor(issue_number: int, issue_url: str, model: str) -> dict[str, Any]:
    prompt = (
        "Follow .cursor/commands/ci/ci-disable-test-ci.md exactly.\n"
        f"Input issue: {issue_url}\n"
        "If evidence is weak, make no code edits and explain in JSON summary.\n"
        "You must emit the marker and JSON contract from the command file."
    )
    cmd = ["agent", "--trust", "-p", prompt]
    if model != "auto":
        cmd[1:1] = ["--model", model]
    result = run(cmd, capture=True)
    return parse_agent_json_after_marker(result.stdout, "===FINAL_DISABLE_EDIT_SUMMARY===")


def invoke_kickoff_agent(pr_url: str, model: str) -> str:
    prompt = (
        "Follow .cursor/commands/ci/ci-kickoff-workflows.md exactly.\n"
        f"Input PR URL: {pr_url}\n"
        "Automatically proceed end-to-end without asking for additional confirmation."
    )
    cmd = ["agent", "--trust", "-p", prompt]
    if model != "auto":
        cmd[1:1] = ["--model", model]
    result = run(cmd, capture=True)
    return result.stdout.strip()[-2000:]


def git_changed_files() -> list[str]:
    out = run(["git", "status", "--porcelain"], capture=True).stdout.strip().splitlines()
    files: list[str] = []
    for line in out:
        if len(line) > 3:
            files.append(line[3:])
    return files


def now_utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def state_key_for_ts(source_ts: str) -> str:
    return f"slack_ts:{source_ts}"


def parse_pr_number(pr_url: str) -> int:
    m = re.search(r"/pull/(\d+)", pr_url)
    if not m:
        return 0
    return int(m.group(1))


def empty_state() -> dict[str, Any]:
    return {"version": 1, "updated_at_utc": now_utc(), "items": []}


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_state()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("state must be a JSON object")
    if not isinstance(data.get("items"), list):
        raise ValueError("state.items must be a list")
    seen: set[str] = set()
    for item in data["items"]:
        if not isinstance(item, dict):
            raise ValueError("state items must be objects")
        key = str(item.get("key", ""))
        if not key:
            raise ValueError("state item missing key")
        if key in seen:
            raise ValueError(f"duplicate state key: {key}")
        seen.add(key)
        status = str(item.get("status", ""))
        if status not in ALLOWED_STATUS:
            raise ValueError(f"invalid state status for {key}: {status}")
        attempts = item.get("attempts", 0)
        if not isinstance(attempts, int) or attempts < 0:
            raise ValueError(f"invalid attempts for {key}")
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = now_utc()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def state_index(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["key"]): item for item in state.get("items", []) if isinstance(item, dict) and "key" in item}


def append_history(item: dict[str, Any], event: str, details: str) -> None:
    history = item.setdefault("history", [])
    if not isinstance(history, list):
        item["history"] = []
        history = item["history"]
    history.append({"ts_utc": now_utc(), "event": event, "details": details})


def ensure_state_item(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    source_ts = str(action["source_slack_ts"])
    key = state_key_for_ts(source_ts)
    idx = state_index(state)
    if key in idx:
        return idx[key]
    issue = int(action["issue_number"])
    item = {
        "key": key,
        "slack_ts": source_ts,
        "issue_numbers": [issue],
        "status": "new",
        "disable_pr": {"number": 0, "url": "", "branch": "", "head_sha": ""},
        "attempts": 0,
        "last_kickoff_runs": [],
        "notification": {"terminal_notified": False, "last_error": ""},
        "terminal_reason": "",
        "history": [],
    }
    append_history(item, "state_created", f"Initialized from issue #{issue}")
    state["items"].append(item)
    return item


def set_status(item: dict[str, Any], status: str, *, event: str, details: str) -> None:
    if status not in ALLOWED_STATUS:
        raise ValueError(f"invalid status transition target: {status}")
    item["status"] = status
    append_history(item, event, details)


def write_summary(path: Path, data: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Auto Disable Actions")
    lines.append("")
    lines.append(f"- Planned actions: {data.get('planned_actions', 0)}")
    lines.append(f"- Executed actions: {len(data.get('executed', []))}")
    lines.append(f"- Skipped actions: {len(data.get('skipped', []))}")
    lines.append(f"- State updates: {data.get('state_updates', 0)}")
    lines.append("")
    lines.append("## State Status Counts")
    for status, count in sorted((data.get("state_status_counts") or {}).items()):
        lines.append(f"- {status}: {count}")
    lines.append("")
    lines.append("## Executed")
    for item in data.get("executed", []):
        lines.append(f"- issue #{item.get('issue_number')}: {item.get('pr_url')}")
    lines.append("")
    lines.append("## Skipped")
    for item in data.get("skipped", []):
        lines.append(f"- ts {item.get('source_slack_ts')}: {item.get('reason')}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute auto-disable action JSON.")
    parser.add_argument("--actions-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--max-attempts-per-item", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        if not os.environ.get("GITHUB_TOKEN"):
            print("GITHUB_TOKEN is required", file=sys.stderr)
            return 2
        if not os.environ.get("CURSOR_API_KEY"):
            print("CURSOR_API_KEY is required", file=sys.stderr)
            return 2

    actions_doc = json.loads(Path(args.actions_json).read_text(encoding="utf-8"))
    actions = actions_doc.get("actions", [])
    if not isinstance(actions, list):
        print("Invalid actions JSON: actions must be a list", file=sys.stderr)
        return 2

    # Validate and dedupe by source ts.
    dedup: dict[str, dict[str, Any]] = {}
    for action in actions:
        if not isinstance(action, dict):
            continue
        source_ts = str(action.get("source_slack_ts", "")).strip()
        issue = action.get("issue_number")
        if not source_ts or not isinstance(issue, int):
            continue
        dedup[source_ts] = action
    validated_actions = list(dedup.values())
    state_path = Path(args.state_json)
    state = load_state(state_path)

    result: dict[str, Any] = {
        "planned_actions": len(validated_actions),
        "executed": [],
        "skipped": [],
        "dry_run": args.dry_run,
        "state_path": str(state_path),
        "state_updates": 0,
    }

    if not args.dry_run:
        run_guarded_gh(["gh", "auth", "status"])
        run(["git", "fetch", "origin", "main"], capture=True)

    for action in validated_actions:
        source_ts = str(action["source_slack_ts"])
        issue_number = int(action["issue_number"])
        item = ensure_state_item(state, action)

        if item["status"] in {"completed", "paused"}:
            result["skipped"].append(
                {
                    "source_slack_ts": source_ts,
                    "issue_number": issue_number,
                    "reason": f"terminal_state:{item['status']}",
                }
            )
            append_history(item, "skip_terminal_state", f"Skipped because status is {item['status']}")
            continue

        if int(item.get("attempts", 0)) >= args.max_attempts_per_item:
            set_status(
                item,
                "needs_human",
                event="max_attempts_exceeded",
                details=f"attempts={item.get('attempts', 0)} threshold={args.max_attempts_per_item}",
            )
            result["state_updates"] += 1
            result["skipped"].append(
                {
                    "source_slack_ts": source_ts,
                    "issue_number": issue_number,
                    "reason": "max_attempts_exceeded",
                }
            )
            continue

        issue_url = f"https://github.com/{REPO}/issues/{issue_number}"
        pr_title = str(action.get("pr_title", "")).strip() or f"ci: disable failing test for #{issue_number}"
        pr_body = str(action.get("pr_body", "")).strip()
        if not pr_body:
            pr_body = (
                f"Refs #{issue_number}\n\n"
                f"Auto-disable-source-ts: {source_ts}\n"
                f"Source Slack: {action.get('source_slack_permalink', '')}\n"
            )
        if f"Auto-disable-source-ts: {source_ts}" not in pr_body:
            pr_body += f"\n\nAuto-disable-source-ts: {source_ts}\n"

        existing = None if args.dry_run else ensure_no_duplicate_open_pr(source_ts)
        if existing:
            item["disable_pr"]["url"] = existing
            item["disable_pr"]["number"] = parse_pr_number(existing)
            set_status(item, "pr_open", event="existing_pr_detected", details=f"Found open PR {existing}")
            result["state_updates"] += 1
            result["skipped"].append(
                {"source_slack_ts": source_ts, "issue_number": issue_number, "reason": f"already_open_pr:{existing}"}
            )
            continue

        branch = branch_name(action)
        if args.dry_run:
            set_status(
                item,
                "planned",
                event="dry_run_planned_disable",
                details=f"Would create or update branch {branch}",
            )
            result["state_updates"] += 1
            result["executed"].append(
                {
                    "source_slack_ts": source_ts,
                    "issue_number": issue_number,
                    "branch": branch,
                    "pr_url": "(dry-run)",
                    "kickoff_output_tail": "(dry-run)",
                }
            )
            continue

        try:
            run(["git", "checkout", "-B", branch, "origin/main"], capture=True)
            before = set(git_changed_files())
            if before:
                result["skipped"].append(
                    {
                        "source_slack_ts": source_ts,
                        "issue_number": issue_number,
                        "reason": "working_tree_not_clean_before_action",
                    }
                )
                append_history(item, "skip_dirty_tree", "Working tree not clean before action")
                continue

            item["attempts"] = int(item.get("attempts", 0)) + 1
            set_status(
                item, "planned", event="attempt_started", details=f"Attempt {item['attempts']} on branch {branch}"
            )
            result["state_updates"] += 1

            edit_summary = run_disable_editor(issue_number, issue_url, args.model)
            changed = git_changed_files()
            if not changed:
                set_status(
                    item,
                    "needs_human",
                    event="no_changes_from_disable_editor",
                    details=f"Attempt {item['attempts']} produced no code changes",
                )
                result["state_updates"] += 1
                result["skipped"].append(
                    {"source_slack_ts": source_ts, "issue_number": issue_number, "reason": "no_code_changes_from_agent"}
                )
                continue

            run(["git", "add", "."], capture=True)
            commit_msg = f"ci: disable failing test for #{issue_number}"
            run(["git", "commit", "-m", commit_msg], capture=True)
            run(["git", "push", "-u", "origin", branch], capture=True)

            pr = run_guarded_gh(
                [
                    "gh",
                    "pr",
                    "create",
                    "--repo",
                    REPO,
                    "--draft",
                    "--base",
                    "main",
                    "--head",
                    branch,
                    "--title",
                    pr_title,
                    "--body",
                    pr_body,
                ]
            )
            pr_url = pr.stdout.strip().splitlines()[-1].strip()
            kickoff_tail = invoke_kickoff_agent(pr_url, args.model)
            item["disable_pr"] = {
                "number": parse_pr_number(pr_url),
                "url": pr_url,
                "branch": branch,
                "head_sha": "",
            }
            set_status(item, "kickoff_running", event="pr_created_and_kickoff_started", details=pr_url)
            result["state_updates"] += 1

            result["executed"].append(
                {
                    "source_slack_ts": source_ts,
                    "issue_number": issue_number,
                    "branch": branch,
                    "pr_url": pr_url,
                    "disable_edit_summary": edit_summary,
                    "kickoff_output_tail": kickoff_tail,
                    "attempts": item["attempts"],
                }
            )
        except Exception as exc:
            set_status(item, "needs_human", event="action_failed", details=str(exc))
            result["state_updates"] += 1
            result["skipped"].append(
                {"source_slack_ts": source_ts, "issue_number": issue_number, "reason": f"action_failed:{exc}"}
            )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for item in state.get("items", []):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    result["state_status_counts"] = counts
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_state(state_path, state)
    write_summary(Path(args.summary_md), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
