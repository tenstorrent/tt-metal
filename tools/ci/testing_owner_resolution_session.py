#!/usr/bin/env python3
"""Testing-mode owner resolution session for triage-ci workflow.

This script:
- extracts individual GitHub usernames from CODEOWNERS,
- dynamically resolves Slack member ids (no hardcoded ids),
- posts a test ping message in a Slack testing channel,
- reads recent channel messages to measure mention quality,
- writes JSON + markdown artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SLACK_API_BASE = "https://slack.com/api"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dynamic GitHub->Slack owner resolution testing session.")
    parser.add_argument("--codeowners-path", required=True)
    parser.add_argument("--slack-channel-id", required=True)
    parser.add_argument("--max-codeowners", type=int, default=10)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--summary-md", required=True)
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required env var: {name}")
    return value


def github_api_get(token: str, endpoint: str) -> dict[str, Any]:
    req = urllib.request.Request(
        f"https://api.github.com{endpoint}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "tt-metal-ci-triage-testing",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"_error": f"http_{exc.code}"}
    except Exception as exc:  # noqa: BLE001
        return {"_error": f"request_failed:{exc}"}


def slack_api_form(token: str, endpoint: str, fields: dict[str, str]) -> dict[str, Any]:
    data = urllib.parse.urlencode(fields).encode("utf-8")
    req = urllib.request.Request(
        f"{SLACK_API_BASE}/{endpoint}",
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
        f"{SLACK_API_BASE}/{endpoint}?{query}",
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_codeowners_users(path: Path) -> list[str]:
    users: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        for tok in parts[1:]:
            if not tok.startswith("@"):
                continue
            handle = tok[1:].strip()
            if not handle or "/" in handle:
                continue
            lowered = handle.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            users.append(handle)
    return users


def slack_list_members(token: str) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    cursor = ""
    while True:
        params = {"limit": "500"}
        if cursor:
            params["cursor"] = cursor
        payload = slack_api_get(token, "users.list", params)
        if not payload.get("ok"):
            break
        batch = payload.get("members", [])
        if isinstance(batch, list):
            members.extend([x for x in batch if isinstance(x, dict)])
        cursor = str(payload.get("response_metadata", {}).get("next_cursor", "")).strip()
        if not cursor:
            break
    return members


def slack_lookup_by_email(token: str, email: str) -> str | None:
    if not email:
        return None
    try:
        payload = slack_api_form(token, "users.lookupByEmail", {"email": email})
    except Exception:  # noqa: BLE001
        return None
    if not payload.get("ok"):
        return None
    uid = str(payload.get("user", {}).get("id", "")).strip()
    return uid or None


def normalize_name(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def slack_lookup_in_members(
    *, members: list[dict[str, Any]], github_login: str, github_name: str
) -> tuple[str | None, str]:
    login = github_login.strip().lower()
    names_to_match = set()
    if github_name.strip():
        names_to_match.add(normalize_name(github_name))
    if login:
        names_to_match.add(normalize_name(login))
        if login.endswith("tt") and len(login) > 2:
            names_to_match.add(normalize_name(login[:-2]))

    exact_matches: list[str] = []
    for member in members:
        if member.get("deleted") or member.get("is_bot"):
            continue
        uid = str(member.get("id", "")).strip()
        if not uid:
            continue
        profile = member.get("profile", {}) if isinstance(member.get("profile"), dict) else {}
        values = [
            normalize_name(str(member.get("name", ""))),
            normalize_name(str(profile.get("display_name", ""))),
            normalize_name(str(profile.get("real_name", ""))),
            normalize_name(str(profile.get("real_name_normalized", ""))),
            normalize_name(str(profile.get("display_name_normalized", ""))),
        ]
        if any(v and v in names_to_match for v in values):
            exact_matches.append(uid)

    if len(set(exact_matches)) == 1:
        return exact_matches[0], "members_exact"

    # Unique token match from full name words (>=3 chars).
    if github_name.strip():
        tokens = [t for t in normalize_name(github_name).split(" ") if len(t) >= 3]
        for token in tokens:
            candidate_ids: set[str] = set()
            for member in members:
                if member.get("deleted") or member.get("is_bot"):
                    continue
                uid = str(member.get("id", "")).strip()
                if not uid:
                    continue
                profile = member.get("profile", {}) if isinstance(member.get("profile"), dict) else {}
                values = [
                    normalize_name(str(member.get("name", ""))),
                    normalize_name(str(profile.get("display_name", ""))),
                    normalize_name(str(profile.get("real_name", ""))),
                ]
                if any(token in v for v in values if v):
                    candidate_ids.add(uid)
            if len(candidate_ids) == 1:
                return next(iter(candidate_ids)), f"members_unique_token:{token}"

    return None, "unresolved"


def resolve_github_to_slack(
    *, github_token: str, slack_token: str, members: list[dict[str, Any]], github_login: str
) -> dict[str, Any]:
    gh = github_api_get(github_token, f"/users/{urllib.parse.quote(github_login)}")
    if gh.get("_error"):
        return {"github_login": github_login, "resolved": False, "reason": str(gh["_error"])}
    email = str(gh.get("email", "")).strip()
    name = str(gh.get("name", "")).strip()
    login = str(gh.get("login", github_login)).strip() or github_login

    by_email = slack_lookup_by_email(slack_token, email)
    if by_email:
        return {
            "github_login": login,
            "github_name": name,
            "github_email": email,
            "slack_user_id": by_email,
            "resolved": True,
            "method": "lookupByEmail",
        }

    by_members, method = slack_lookup_in_members(members=members, github_login=login, github_name=name)
    if by_members:
        return {
            "github_login": login,
            "github_name": name,
            "github_email": email,
            "slack_user_id": by_members,
            "resolved": True,
            "method": method,
        }

    return {
        "github_login": login,
        "github_name": name,
        "github_email": email,
        "resolved": False,
        "reason": method,
    }


def post_test_message(*, slack_token: str, channel_id: str, resolved_user_ids: list[str], unresolved: list[str]) -> str:
    mentions = " ".join(f"<@{uid}>" for uid in resolved_user_ids) if resolved_user_ids else "(none)"
    text_lines = [
        "CI triage testing mode: dynamic CODEOWNERS owner-resolution probe.",
        f"Resolved mentions: {mentions}",
    ]
    if unresolved:
        text_lines.append("Unresolved GitHub handles: " + ", ".join(f"@{u}" for u in unresolved))
    payload = slack_api_form(
        slack_token,
        "chat.postMessage",
        {"channel": channel_id, "text": "\n".join(text_lines)},
    )
    if not payload.get("ok"):
        raise RuntimeError(f"chat.postMessage failed: {payload.get('error', 'unknown_error')}")
    return str(payload.get("ts", "")).strip()


def analyze_recent_channel_messages(*, slack_token: str, channel_id: str, limit: int = 40) -> dict[str, Any]:
    payload = slack_api_get(
        slack_token,
        "conversations.history",
        {"channel": channel_id, "limit": str(limit)},
    )
    if not payload.get("ok"):
        raise RuntimeError(f"conversations.history failed: {payload.get('error', 'unknown_error')}")
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    likely_owner_msgs = 0
    likely_owner_with_mentions = 0
    likely_owner_pending = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        text = str(msg.get("text", ""))
        if "Likely owners:" not in text:
            continue
        likely_owner_msgs += 1
        if "<@" in text:
            likely_owner_with_mentions += 1
        if "(owner resolution pending)" in text:
            likely_owner_pending += 1
    return {
        "recent_messages_scanned": len(messages),
        "likely_owner_messages": likely_owner_msgs,
        "likely_owner_messages_with_mentions": likely_owner_with_mentions,
        "likely_owner_messages_pending": likely_owner_pending,
    }


def build_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "## Owner Resolution Testing Report",
        "",
        f"- CODEOWNERS users scanned: {summary['codeowners_users_scanned']}",
        f"- Resolution attempts: {summary['attempted']}",
        f"- Resolved users: {summary['resolved']}",
        f"- Unresolved users: {summary['unresolved']}",
        f"- Resolution rate: {summary['resolution_rate_percent']:.1f}%",
        f"- Slack test message ts: `{summary['test_message_ts']}`",
        "",
        "## Recent Channel Readback",
        "",
        f"- Messages scanned: {summary['recent_messages_scanned']}",
        f"- `Likely owners:` messages: {summary['likely_owner_messages']}",
        f"- With `<@U...>` mentions: {summary['likely_owner_messages_with_mentions']}",
        f"- With `(owner resolution pending)`: {summary['likely_owner_messages_pending']}",
        "",
    ]
    if result["unresolved_github_logins"]:
        lines.append("## Unresolved GitHub Handles")
        lines.append("")
        for login in result["unresolved_github_logins"]:
            lines.append(f"- `@{login}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    slack_token = require_env("SLACK_BOT_TOKEN")
    github_token = require_env("GITHUB_TOKEN")

    codeowners_path = Path(args.codeowners_path)
    if not codeowners_path.exists():
        raise RuntimeError(f"missing CODEOWNERS path: {codeowners_path}")

    gh_users = parse_codeowners_users(codeowners_path)
    max_users = max(args.max_codeowners, 1)
    selected_users = gh_users[:max_users]

    members = slack_list_members(slack_token)
    attempts: list[dict[str, Any]] = []
    resolved_ids: list[str] = []
    unresolved_logins: list[str] = []

    for login in selected_users:
        item = resolve_github_to_slack(
            github_token=github_token,
            slack_token=slack_token,
            members=members,
            github_login=login,
        )
        attempts.append(item)
        if item.get("resolved"):
            uid = str(item.get("slack_user_id", "")).strip()
            if uid and uid not in resolved_ids:
                resolved_ids.append(uid)
        else:
            unresolved_logins.append(str(item.get("github_login", login)))

    test_message_ts = post_test_message(
        slack_token=slack_token,
        channel_id=args.slack_channel_id,
        resolved_user_ids=resolved_ids,
        unresolved=unresolved_logins,
    )
    readback = analyze_recent_channel_messages(
        slack_token=slack_token,
        channel_id=args.slack_channel_id,
        limit=40,
    )

    attempted = len(selected_users)
    resolved = len(resolved_ids)
    unresolved = len(unresolved_logins)
    resolution_rate_percent = (100.0 * resolved / attempted) if attempted else 0.0

    result = {
        "slack_channel_id": args.slack_channel_id,
        "attempts": attempts,
        "resolved_slack_ids": resolved_ids,
        "unresolved_github_logins": unresolved_logins,
        "summary": {
            "codeowners_users_scanned": len(gh_users),
            "attempted": attempted,
            "resolved": resolved,
            "unresolved": unresolved,
            "resolution_rate_percent": resolution_rate_percent,
            "test_message_ts": test_message_ts,
            **readback,
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary_md = Path(args.summary_md)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(build_markdown(result), encoding="utf-8")
    print(json.dumps({"attempted": attempted, "resolved": resolved, "unresolved": unresolved}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
