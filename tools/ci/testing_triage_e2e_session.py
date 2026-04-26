#!/usr/bin/env python3
"""Testing-mode end-to-end Slack thread exercises on real CI issues."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Ensure repository root is importable when running as `python tools/ci/<script>.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ci.slack_thread_agent_analysis import analyze_thread_with_agent

ISSUE_REPO_TEST = "ebanerjeeTT/issue_dump"
ISSUE_URL_RE = re.compile(r"https://github\.com/ebanerjeeTT/issue_dump/issues/(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run testing-mode triage E2E thread exercises.")
    parser.add_argument("--slack-channel-id", required=True)
    parser.add_argument("--max-threads", type=int, default=3)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--summary-md", required=True)
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing env var: {name}")
    return value


def slack_api_form(token: str, endpoint: str, fields: dict[str, str]) -> dict[str, Any]:
    data = urllib.parse.urlencode(fields).encode("utf-8")
    req = urllib.request.Request(
        f"https://slack.com/api/{endpoint}",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def slack_api_get(token: str, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"https://slack.com/api/{endpoint}?{query}",
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def gh_api_get(token: str, endpoint: str) -> dict[str, Any]:
    req = urllib.request.Request(
        f"https://api.github.com{endpoint}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "tt-metal-testing-triage-e2e",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_thread_message(*, token: str, channel: str, thread_ts: str, text: str) -> str:
    payload = slack_api_form(
        token,
        "chat.postMessage",
        {"channel": channel, "thread_ts": thread_ts, "text": text},
    )
    if not payload.get("ok"):
        raise RuntimeError(f"chat.postMessage failed: {payload.get('error', 'unknown_error')}")
    return str(payload.get("ts", "")).strip()


def read_channel_messages(token: str, channel_id: str, limit: int = 120) -> list[dict[str, Any]]:
    payload = slack_api_get(token, "conversations.history", {"channel": channel_id, "limit": str(limit)})
    if not payload.get("ok"):
        raise RuntimeError(f"conversations.history failed: {payload.get('error', 'unknown_error')}")
    messages = payload.get("messages", [])
    return [x for x in messages if isinstance(x, dict)] if isinstance(messages, list) else []


def read_thread_messages(token: str, channel_id: str, thread_ts: str) -> list[dict[str, Any]]:
    payload = slack_api_get(token, "conversations.replies", {"channel": channel_id, "ts": thread_ts, "limit": "100"})
    if not payload.get("ok"):
        raise RuntimeError(f"conversations.replies failed: {payload.get('error', 'unknown_error')}")
    messages = payload.get("messages", [])
    return [x for x in messages if isinstance(x, dict)] if isinstance(messages, list) else []


def parse_issue_number(text: str) -> int:
    m = ISSUE_URL_RE.search(text)
    if not m:
        return 0
    return int(m.group(1))


def parse_codeowners_logins(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for tok in line.split()[1:]:
            if not tok.startswith("@"):
                continue
            login = tok[1:].strip()
            if not login or "/" in login:
                continue
            low = login.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(login)
    return out


def find_real_issue_threads(messages: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    threads: list[dict[str, Any]] = []
    for msg in messages:
        text = str(msg.get("text", ""))
        issue_number = parse_issue_number(text)
        if issue_number <= 0:
            continue
        ts = str(msg.get("ts", "")).strip()
        if not ts:
            continue
        threads.append(
            {
                "thread_ts": ts,
                "issue_number": issue_number,
                "top_text": text,
                "top_permalink": f"https://tenstorrent.slack.com/archives/C0APK6215B5/p{ts.replace('.', '')}",
            }
        )
        if len(threads) >= limit:
            break
    return threads


def choose_mock_dev_reply(idx: int, github_login: str) -> str:
    options = [
        f"Testing-mode mock dev reply from @{github_login}: looking now, I will post PR in 2 hours.",
        f"Testing-mode mock dev reply from @{github_login}: can agent fix this and draft a fix PR if possible?",
        f"Testing-mode mock dev reply from @{github_login}: blocked on infra/hardware dependency.",
        f"Testing-mode mock dev reply from @{github_login}: should be fixed by https://github.com/tenstorrent/tt-metal/pull/12346; please verify.",
    ]
    return options[idx % len(options)]


def build_bot_response(
    *, issue_number: int, progress: dict[str, Any], fix_request: dict[str, Any], mock_github_owner: str
) -> str:
    mock_pr = f"https://github.com/ebanerjeeTT/tt-metal/pull/mock-{issue_number}"
    if fix_request.get("requested"):
        return (
            "Testing-mode bot response: fix request detected in thread. "
            f"Mock draft PR: {mock_pr} | Mock assignment note: would assign issue #{issue_number} to @{mock_github_owner}."
        )
    if bool(progress.get("defer_disable", False)):
        return (
            "Testing-mode bot response: high-confidence progress detected, disabling is deferred for now. "
            "Please post concrete PR updates in-thread."
        )
    return (
        "Testing-mode bot response: unresolved/blocked signal detected. "
        "If still unresolved by SLA, disable path would be considered."
    )


def issue_exists(token: str, issue_number: int) -> bool:
    try:
        payload = gh_api_get(token, f"/repos/{ISSUE_REPO_TEST}/issues/{issue_number}")
    except Exception:
        return False
    return int(payload.get("number", 0) or 0) == issue_number


def build_summary(payload: dict[str, Any]) -> str:
    lines = [
        "## Triage E2E Testing Session",
        "",
        f"- Real issue threads selected: {len(payload.get('threads', []))}",
        f"- Mock dev replies posted: {payload.get('mock_dev_reply_count', 0)}",
        f"- Bot follow-up replies posted: {payload.get('bot_followup_count', 0)}",
        "",
        "## Thread Cases",
    ]
    for row in payload.get("threads", []):
        lines.append(
            f"- issue #{row.get('issue_number')} | state `{row.get('progress_state')}` | "
            f"fix_request={row.get('fix_request_requested')} | thread `{row.get('thread_ts')}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    slack_token = require_env("SLACK_BOT_TOKEN")
    github_token = require_env("GITHUB_TOKEN")
    messages = read_channel_messages(slack_token, args.slack_channel_id, limit=160)
    threads = find_real_issue_threads(messages, args.max_threads)
    mock_devs = parse_codeowners_logins(Path(".github/CODEOWNERS"))
    if not mock_devs:
        mock_devs = ["ebanerjeeTT"]

    results: list[dict[str, Any]] = []
    mock_dev_reply_count = 0
    bot_followup_count = 0
    for idx, thread in enumerate(threads):
        issue_number = int(thread["issue_number"])
        if not issue_exists(github_token, issue_number):
            continue
        thread_ts = str(thread["thread_ts"])
        mock_dev_login = mock_devs[idx % len(mock_devs)]
        mock_reply = choose_mock_dev_reply(idx, mock_dev_login)
        mock_ts = post_thread_message(
            token=slack_token,
            channel=args.slack_channel_id,
            thread_ts=thread_ts,
            text=mock_reply,
        )
        mock_dev_reply_count += 1
        thread_messages = read_thread_messages(slack_token, args.slack_channel_id, thread_ts)
        thread_replies = [m for m in thread_messages if str(m.get("ts", "")).strip() != thread_ts]
        analysis = analyze_thread_with_agent(
            top_level_text=thread["top_text"],
            thread_replies=thread_replies,
            include_owner_claim=False,
            model="auto",
        )
        progress = {
            "progress_state": analysis.get("progress_state", "no_progress"),
            "defer_disable": bool(analysis.get("defer_disable", False)),
        }
        fix_request = {"requested": bool(analysis.get("fix_request_requested", False))}
        bot_text = build_bot_response(
            issue_number=issue_number,
            progress=progress,
            fix_request=fix_request,
            mock_github_owner=mock_dev_login,
        )
        bot_ts = post_thread_message(
            token=slack_token,
            channel=args.slack_channel_id,
            thread_ts=thread_ts,
            text=bot_text,
        )
        bot_followup_count += 1
        # Read thread again to ensure response is visible in API readback.
        thread_messages_after = read_thread_messages(slack_token, args.slack_channel_id, thread_ts)
        results.append(
            {
                "thread_ts": thread_ts,
                "issue_number": issue_number,
                "top_permalink": thread["top_permalink"],
                "mock_dev_reply_ts": mock_ts,
                "bot_followup_ts": bot_ts,
                "progress_state": progress.get("progress_state"),
                "defer_disable": bool(progress.get("defer_disable", False)),
                "fix_request_requested": bool(fix_request.get("requested", False)),
                "thread_message_count_after": len(thread_messages_after),
                "mock_dev_login": mock_dev_login,
            }
        )

    output = {
        "threads": results,
        "mock_dev_reply_count": mock_dev_reply_count,
        "bot_followup_count": bot_followup_count,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    summary_md = Path(args.summary_md)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(build_summary(output), encoding="utf-8")
    print(json.dumps({"threads": len(results), "mock_dev_reply_count": mock_dev_reply_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
