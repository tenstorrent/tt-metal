#!/usr/bin/env python3
"""Export recent Slack channel activity (including thread replies) to JSON."""

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


DEFAULT_CHANNEL_ID = "C08SJ7MGESY"
DEFAULT_DAYS = 30
API_BASE = "https://slack.com/api"


def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def slack_api_get(token: str, endpoint: str, params: dict[str, Any], max_retries: int = 5) -> dict[str, Any]:
    """Call a Slack Web API GET endpoint with basic rate-limit handling."""
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
        # Respect app-level rate limiting if Slack reports it in payload.
        if error == "ratelimited" and attempt <= max_retries:
            time.sleep(min(2**attempt, 30))
            continue
        raise RuntimeError(f"Slack API error from {endpoint}: {error}")


def fetch_channel_messages(token: str, channel_id: str, oldest_ts: float, latest_ts: float) -> list[dict[str, Any]]:
    """Fetch all top-level channel messages in the requested window."""
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
    """Fetch replies for a thread root within the requested window."""
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
        # Slack includes the root message in conversations.replies results.
        replies.extend([m for m in batch if m.get("ts") != thread_ts])

        cursor = payload.get("response_metadata", {}).get("next_cursor") or None
        if not cursor:
            break

    return replies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Export last N days of Slack channel messages (including thread replies) " "to JSON.")
    )
    parser.add_argument("--channel-id", default=DEFAULT_CHANNEL_ID, help="Slack channel ID")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Lookback window in days")
    parser.add_argument(
        "--output",
        default=f"build_ci/raw_data/slack_{DEFAULT_CHANNEL_ID}_last_{DEFAULT_DAYS}_days.json",
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

    top_level_messages = fetch_channel_messages(token, args.channel_id, oldest_ts, now_ts)
    top_level_messages.sort(key=lambda m: float(m.get("ts", "0")))

    results: list[dict[str, Any]] = []
    total_replies = 0

    for msg in top_level_messages:
        item: dict[str, Any] = {
            "ts": msg.get("ts"),
            "thread_ts": msg.get("thread_ts", msg.get("ts")),
            "user": msg.get("user"),
            "bot_id": msg.get("bot_id"),
            "subtype": msg.get("subtype"),
            "text": msg.get("text", ""),
            "reply_count": msg.get("reply_count", 0),
            "latest_reply": msg.get("latest_reply"),
            "raw": msg,
        }

        if int(msg.get("reply_count", 0)) > 0 and msg.get("ts"):
            replies = fetch_thread_replies(token, args.channel_id, msg["ts"], oldest_ts, now_ts)
            replies.sort(key=lambda r: float(r.get("ts", "0")))
            item["replies"] = replies
            total_replies += len(replies)
        else:
            item["replies"] = []

        results.append(item)

    payload = {
        "exported_at_utc": iso_utc(now_ts),
        "channel_id": args.channel_id,
        "window": {
            "days": args.days,
            "oldest_ts": f"{oldest_ts:.6f}",
            "latest_ts": f"{now_ts:.6f}",
            "oldest_utc": iso_utc(oldest_ts),
            "latest_utc": iso_utc(now_ts),
        },
        "counts": {
            "top_level_messages": len(results),
            "thread_replies": total_replies,
            "total_items": len(results) + total_replies,
        },
        "messages": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote Slack export to {output_path}")
    print(
        "Counts:",
        json.dumps(payload["counts"], separators=(",", ":")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
