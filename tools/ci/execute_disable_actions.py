#!/usr/bin/env python3
"""Execute structured stale-disable actions with deterministic git/gh steps."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = "tenstorrent/tt-metal"


def run(cmd: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )


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
    prs = run(
        ["gh", "pr", "list", "--repo", REPO, "--state", "open", "--json", "number,url,body"],
        capture=True,
    )
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


def write_summary(path: Path, data: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Auto Disable Actions")
    lines.append("")
    lines.append(f"- Planned actions: {data.get('planned_actions', 0)}")
    lines.append(f"- Executed actions: {len(data.get('executed', []))}")
    lines.append(f"- Skipped actions: {len(data.get('skipped', []))}")
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
    parser.add_argument("--model", default="auto")
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

    result: dict[str, Any] = {
        "planned_actions": len(validated_actions),
        "executed": [],
        "skipped": [],
        "dry_run": args.dry_run,
    }

    if not args.dry_run:
        run(["gh", "auth", "status"], capture=True)
        run(["git", "fetch", "origin", "main"], capture=True)

    for action in validated_actions:
        source_ts = str(action["source_slack_ts"])
        issue_number = int(action["issue_number"])
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
            result["skipped"].append(
                {"source_slack_ts": source_ts, "issue_number": issue_number, "reason": f"already_open_pr:{existing}"}
            )
            continue

        branch = branch_name(action)
        if args.dry_run:
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
            continue

        edit_summary = run_disable_editor(issue_number, issue_url, args.model)
        changed = git_changed_files()
        if not changed:
            result["skipped"].append(
                {"source_slack_ts": source_ts, "issue_number": issue_number, "reason": "no_code_changes_from_agent"}
            )
            continue

        run(["git", "add", "."], capture=True)
        commit_msg = f"ci: disable failing test for #{issue_number}"
        run(["git", "commit", "-m", commit_msg], capture=True)
        run(["git", "push", "-u", "origin", branch], capture=True)

        pr = run(
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
            ],
            capture=True,
        )
        pr_url = pr.stdout.strip().splitlines()[-1].strip()
        kickoff_tail = invoke_kickoff_agent(pr_url, args.model)
        result["executed"].append(
            {
                "source_slack_ts": source_ts,
                "issue_number": issue_number,
                "branch": branch,
                "pr_url": pr_url,
                "disable_edit_summary": edit_summary,
                "kickoff_output_tail": kickoff_tail,
            }
        )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_summary(Path(args.summary_md), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
