#!/usr/bin/env python3
"""Select stale unresolved Slack top-level messages for auto-disable handling."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Slack export to unresolved stale candidates.")
    parser.add_argument("--input", required=True, help="Path to enriched Slack JSON")
    parser.add_argument("--output", required=True, help="Path to candidate JSON")
    parser.add_argument("--stale-hours", type=float, default=32.0, help="Minimum age in hours")
    parser.add_argument("--channel-id", default="C0APK6215B5", help="Slack channel ID for permalink generation")
    parser.add_argument("--max-candidates", type=int, default=20, help="Optional max candidate count")
    return parser.parse_args()


def permalink(channel_id: str, ts: str) -> str:
    return f"https://tenstorrent.slack.com/archives/{channel_id}/p{ts.replace('.', '')}"


def issue_numbers(msg: dict[str, Any]) -> list[int]:
    refs = msg.get("referenced_issue_numbers", [])
    if isinstance(refs, list):
        return [int(n) for n in refs if isinstance(n, int) or (isinstance(n, str) and n.isdigit())]
    return []


def main() -> int:
    args = parse_args()
    now = time.time()
    src = Path(args.input)
    payload = json.loads(src.read_text(encoding="utf-8"))
    messages = payload.get("messages", [])

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for msg in messages:
        ts = str(msg.get("ts", "")).strip()
        if not ts:
            continue
        try:
            ts_float = float(ts)
        except ValueError:
            skipped.append({"ts": ts, "reason": "invalid_ts"})
            continue

        age_hours = (now - ts_float) / 3600.0
        msg_issue_closed = bool(msg.get("issue_closed", False))
        refs = issue_numbers(msg)

        if msg_issue_closed:
            skipped.append({"ts": ts, "reason": "issue_closed_true"})
            continue
        if age_hours <= args.stale_hours:
            skipped.append({"ts": ts, "reason": "not_stale", "age_hours": round(age_hours, 2)})
            continue

        candidates.append(
            {
                "source_slack_ts": ts,
                "source_slack_permalink": permalink(args.channel_id, ts),
                "age_hours": round(age_hours, 2),
                "issue_closed": False,
                "issue_status_lookup_failed": bool(msg.get("issue_status_lookup_failed", False)),
                "issue_numbers": refs,
                "primary_issue_number": refs[0] if refs else None,
                "top_level_text": str(msg.get("text", "")),
                "thread_reply_count": len(msg.get("thread_replies", [])),
            }
        )

    candidates.sort(key=lambda c: c["age_hours"], reverse=True)
    limited = candidates[: args.max_candidates]
    out_payload = {
        "generated_at_unix": now,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "stale_hours_threshold": args.stale_hours,
        "channel_id": args.channel_id,
        "input_message_count": len(messages),
        "candidate_count": len(limited),
        "candidates": limited,
        "skipped_count": len(skipped),
        "skipped": skipped,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_count": len(limited), "skipped_count": len(skipped)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
