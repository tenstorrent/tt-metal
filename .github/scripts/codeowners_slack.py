#!/usr/bin/env python3
"""Slack side effects for `.github/workflows/codeowners-group-analysis.yaml`.

Two subcommands, both best-effort: any Slack failure is logged as a
`::warning::` and the script still exits 0, so the ping itself never fails
because of them.

  invite --channel C --users U1,U2
      Add any of the users who are not yet in the channel. A Slack mention of
      someone outside the channel does not notify them, so without this the
      ping silently misses exactly the owners who do not follow the channel.
      Needs `channels:read` + `channels:manage` (`groups:read` +
      `groups:write` for a private channel) on the bot token.

  report --channel C --report-file F [--thread-ts T] [--pr-url U --pr-title S]
      Post the admin report written by `select_owners_to_ping.py` as a thread
      reply to the ping at T. Without T (no owner could be pinged at all) a
      short parent message is posted first so the report has a thread to go in.
      The admins named in CODEOWNERS_ADMIN_SLACK_IDS are mentioned at the top.

Environment:
  SLACK_BOT_TOKEN              required; the CODEOWNERS_PING_BOT secret
  CODEOWNERS_ADMIN_SLACK_IDS   optional; comma-separated Slack member IDs
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request

SLACK_API = "https://slack.com/api"

# conversations.members pages at up to 1000 entries.
MAX_PAGES = 20


def warn(msg: str) -> None:
    print(f"::warning::{msg}", flush=True)


def slack(method: str, token: str, payload: dict | None = None, query: dict | None = None) -> dict | None:
    """Call a Slack Web API method; return the body when `ok`, else None."""
    url = f"{SLACK_API}/{method}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    request.add_header("Authorization", f"Bearer {token}")
    if data:
        request.add_header("Content-Type", "application/json; charset=utf-8")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.load(response)
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        warn(f"Slack {method} failed: {exc}")
        return None
    if not body.get("ok"):
        warn(f"Slack {method} failed: {body.get('error', body)}")
        return None
    return body


def channel_members(token: str, channel: str) -> set[str] | None:
    members: set[str] = set()
    cursor = ""
    for _ in range(MAX_PAGES):
        query = {"channel": channel, "limit": 1000}
        if cursor:
            query["cursor"] = cursor
        body = slack("conversations.members", token, query=query)
        if body is None:
            return None
        members.update(body.get("members") or [])
        cursor = (body.get("response_metadata") or {}).get("next_cursor") or ""
        if not cursor:
            break
    return members


def invite(token: str, channel: str, users: list[str]) -> None:
    users = list(dict.fromkeys(u for u in users if u))
    if not users:
        return
    present = channel_members(token, channel)
    if present is None:
        return
    missing = [u for u in users if u not in present]
    if not missing:
        print(f"All {len(users)} pinged user(s) are already in {channel}")
        return
    # `force` invites the valid IDs even when one of them cannot be (a guest, a
    # deactivated account) instead of failing the whole call.
    if slack("conversations.invite", token, {"channel": channel, "users": ",".join(missing), "force": True}):
        print(f"Invited {len(missing)} pinged user(s) to {channel}: {', '.join(missing)}")


def post(token: str, channel: str, text: str, thread_ts: str = "") -> str:
    payload = {"channel": channel, "text": text, "unfurl_links": False}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    body = slack("chat.postMessage", token, payload)
    return (body or {}).get("ts") or ""


def report(token: str, channel: str, report_text: str, thread_ts: str, pr_url: str, pr_title: str) -> None:
    report_text = report_text.strip()
    if not report_text:
        return
    if not thread_ts:
        link = f"<{pr_url}|{pr_title or pr_url}>" if pr_url else "this PR"
        thread_ts = post(token, channel, f"⚠️ *CodeOwners Review Request*\n\nNo code owner could be pinged for {link}.")
        if not thread_ts:
            return
    admins = [a.strip() for a in os.environ.get("CODEOWNERS_ADMIN_SLACK_IDS", "").split(",") if a.strip()]
    if admins:
        report_text = " ".join(f"<@{a}>" for a in admins) + "\n" + report_text
    if post(token, channel, report_text, thread_ts):
        print("Posted CODEOWNERS admin report in the ping thread")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_invite = sub.add_parser("invite")
    p_invite.add_argument("--channel", required=True)
    p_invite.add_argument("--users", default="")
    p_report = sub.add_parser("report")
    p_report.add_argument("--channel", required=True)
    p_report.add_argument("--report-file", required=True)
    p_report.add_argument("--thread-ts", default="")
    p_report.add_argument("--pr-url", default="")
    p_report.add_argument("--pr-title", default="")
    args = parser.parse_args()

    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        warn("SLACK_BOT_TOKEN is not set; skipping Slack " + args.command)
        return 0

    if args.command == "invite":
        invite(token, args.channel, args.users.replace(" ", ",").split(","))
    else:
        try:
            with open(args.report_file, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            warn(f"Could not read {args.report_file}: {exc}")
            return 0
        report(token, args.channel, text, args.thread_ts, args.pr_url, args.pr_title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
