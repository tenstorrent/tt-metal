#!/usr/bin/env python3
"""Export top-level messages by one Slack user with nested thread replies."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_BASE = "https://slack.com/api"


def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def slack_api_get(token: str, endpoint: str, params: dict[str, Any], max_retries: int = 5) -> dict[str, Any]:
    url = f"{API_BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})

    attempt = 0
    while True:
        attempt += 1
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt <= max_retries:
                retry_after = int(exc.headers.get("Retry-After", "1"))
                time.sleep(retry_after)
                continue
            raise RuntimeError(f"HTTP error from Slack API ({endpoint}): {exc}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error calling Slack API ({endpoint}): {exc}") from exc

        if payload.get("ok", False):
            return payload

        error = payload.get("error", "unknown_error")
        if error == "ratelimited" and attempt <= max_retries:
            time.sleep(min(2**attempt, 30))
            continue
        raise RuntimeError(f"Slack API error from {endpoint}: {error}")


def fetch_channel_messages(token: str, channel_id: str, oldest_ts: float, latest_ts: float) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    cursor: str | None = None

    while True:
        params: dict[str, Any] = {
            "channel": channel_id,
            "limit": 200,
            "oldest": f"{oldest_ts:.6f}",
            "latest": f"{latest_ts:.6f}",
            "inclusive": "true",
        }
        if cursor:
            params["cursor"] = cursor

        payload = slack_api_get(token, "conversations.history", params)
        messages.extend(payload.get("messages", []))

        cursor = payload.get("response_metadata", {}).get("next_cursor") or None
        if not cursor:
            break

    return messages


def fetch_thread_replies(
    token: str, channel_id: str, thread_ts: str, oldest_ts: float, latest_ts: float
) -> list[dict[str, Any]]:
    replies: list[dict[str, Any]] = []
    cursor: str | None = None

    while True:
        params: dict[str, Any] = {
            "channel": channel_id,
            "ts": thread_ts,
            "limit": 200,
            "oldest": f"{oldest_ts:.6f}",
            "latest": f"{latest_ts:.6f}",
            "inclusive": "true",
        }
        if cursor:
            params["cursor"] = cursor

        payload = slack_api_get(token, "conversations.replies", params)
        batch = payload.get("messages", [])
        # Slack includes the root message in replies; keep only replies.
        replies.extend([m for m in batch if m.get("ts") != thread_ts])

        cursor = payload.get("response_metadata", {}).get("next_cursor") or None
        if not cursor:
            break

    replies.sort(key=lambda r: float(r.get("ts", "0")))
    return replies


def is_top_level(msg: dict[str, Any]) -> bool:
    return msg.get("thread_ts") in (None, "", msg.get("ts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export top-level Slack messages by one author with thread replies.")
    parser.add_argument("--channel-id", required=True, help="Slack channel ID (e.g. C05GRJC4J4A)")
    parser.add_argument("--author-user-id", required=True, help="Slack user ID to filter top-level messages")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days")
    parser.add_argument(
        "--output",
        default="build_ci/raw_data/slack_author_threads.json",
        help="Output JSON file path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.days <= 0:
        print("--days must be positive", file=sys.stderr)
        return 2

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("SLACK_BOT_TOKEN is not set in this shell.", file=sys.stderr)
        return 2

    now_ts = time.time()
    oldest_ts = now_ts - (args.days * 24 * 60 * 60)

    all_messages = fetch_channel_messages(token, args.channel_id, oldest_ts, now_ts)
    top_level_by_author = [m for m in all_messages if is_top_level(m) and m.get("user") == args.author_user_id]
    top_level_by_author.sort(key=lambda m: float(m.get("ts", "0")))

    exported_messages: list[dict[str, Any]] = []
    total_replies = 0

    for msg in top_level_by_author:
        ts = msg.get("ts")
        if not ts:
            continue

        replies = fetch_thread_replies(token, args.channel_id, ts, oldest_ts, now_ts)
        total_replies += len(replies)

        exported_messages.append(
            {
                "ts": ts,
                "thread_ts": msg.get("thread_ts", ts),
                "user": msg.get("user"),
                "bot_id": msg.get("bot_id"),
                "text": msg.get("text", ""),
                "subtype": msg.get("subtype"),
                "reply_count": msg.get("reply_count", 0),
                "latest_reply": msg.get("latest_reply"),
                "raw": msg,
                "thread_replies": replies,
            }
        )

    payload = {
        "exported_at_utc": iso_utc(now_ts),
        "channel_id": args.channel_id,
        "author_user_id": args.author_user_id,
        "window": {
            "days": args.days,
            "oldest_ts": f"{oldest_ts:.6f}",
            "latest_ts": f"{now_ts:.6f}",
            "oldest_utc": iso_utc(oldest_ts),
            "latest_utc": iso_utc(now_ts),
        },
        "counts": {
            "all_messages_in_window": len(all_messages),
            "top_level_messages_by_author": len(exported_messages),
            "thread_replies_total": total_replies,
            "total_exported_items": len(exported_messages) + total_replies,
        },
        "messages": exported_messages,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote Slack export to {output_path}")
    print("Counts:", json.dumps(payload["counts"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
