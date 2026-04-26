#!/usr/bin/env python3
"""Select stale unresolved Slack top-level messages for auto-disable handling."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is importable when running as `python tools/ci/<script>.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ci.slack_thread_agent_analysis import analyze_thread_with_agent

NON_ACTIONABLE_SUBTYPES = {
    "channel_join",
    "channel_leave",
    "channel_topic",
    "channel_purpose",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Slack export to unresolved stale candidates.")
    parser.add_argument("--input", required=True, help="Path to enriched Slack JSON")
    parser.add_argument("--output", required=True, help="Path to candidate JSON")
    parser.add_argument("--stale-hours", type=float, default=32.0, help="Minimum age in hours")
    parser.add_argument("--channel-id", default="C0APK6215B5", help="Slack channel ID for permalink generation")
    parser.add_argument("--max-candidates", type=int, default=20, help="Optional max candidate count")
    parser.add_argument(
        "--hold-hours-after-progress",
        type=float,
        default=24.0,
        help="Temporary hold window after high-confidence progress signals",
    )
    return parser.parse_args()


def permalink(channel_id: str, ts: str) -> str:
    return f"https://tenstorrent.slack.com/archives/{channel_id}/p{ts.replace('.', '')}"


def issue_numbers(msg: dict[str, Any]) -> list[int]:
    refs = msg.get("referenced_issue_numbers", [])
    if isinstance(refs, list):
        return [int(n) for n in refs if isinstance(n, int) or (isinstance(n, str) and n.isdigit())]
    return []


def primary_issue_detail(msg: dict[str, Any]) -> dict[str, Any] | None:
    refs = msg.get("referenced_issues", [])
    if not isinstance(refs, list) or not refs:
        return None
    first = refs[0]
    if not isinstance(first, dict):
        return None
    return first


def message_replies(msg: dict[str, Any]) -> list[dict[str, Any]]:
    replies = msg.get("thread_replies")
    if not isinstance(replies, list):
        replies = msg.get("replies")
    if not isinstance(replies, list):
        return []
    return [row for row in replies if isinstance(row, dict)]


def main() -> int:
    args = parse_args()
    now = time.time()
    src = Path(args.input)
    payload = json.loads(src.read_text(encoding="utf-8"))
    messages = payload.get("messages", [])

    analysis_model = str(os.environ.get("SLACK_THREAD_ANALYSIS_MODEL", "auto")).strip() or "auto"

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for msg in messages:
        ts = str(msg.get("ts", "")).strip()
        if not ts:
            continue
        subtype = str(msg.get("subtype", "")).strip()
        if subtype in NON_ACTIONABLE_SUBTYPES:
            skipped.append({"ts": ts, "reason": f"non_actionable_subtype:{subtype}"})
            continue
        try:
            ts_float = float(ts)
        except ValueError:
            skipped.append({"ts": ts, "reason": "invalid_ts"})
            continue

        age_hours = (now - ts_float) / 3600.0
        msg_issue_closed = bool(msg.get("issue_closed", False))
        refs = issue_numbers(msg)
        primary = primary_issue_detail(msg)
        primary_repo = str(primary.get("repo", "")).strip() if primary else ""
        primary_url = str(primary.get("url", "")).strip() if primary else ""

        if msg_issue_closed:
            skipped.append({"ts": ts, "reason": "issue_closed_true"})
            continue
        if age_hours <= args.stale_hours:
            skipped.append({"ts": ts, "reason": "not_stale", "age_hours": round(age_hours, 2)})
            continue

        thread_replies = message_replies(msg)
        try:
            analysis = analyze_thread_with_agent(
                top_level_text=str(msg.get("text", "")),
                thread_replies=thread_replies,
                hold_hours_after_progress=args.hold_hours_after_progress,
                include_owner_claim=False,
                model=analysis_model,
            )
        except Exception as exc:
            skipped.append({"ts": ts, "reason": f"agent_thread_analysis_failed:{exc}"})
            continue
        progress = {
            "progress_state": analysis.get("progress_state", "no_progress"),
            "confidence": analysis.get("confidence", "low"),
            "defer_disable": bool(analysis.get("defer_disable", False)),
            "reason": analysis.get("progress_reason", "unspecified"),
        }
        fix_request = {
            "requested": bool(analysis.get("fix_request_requested", False)),
            "reason": analysis.get("fix_request_reason", "unspecified"),
        }
        if bool(progress.get("defer_disable", False)):
            skipped.append(
                {
                    "ts": ts,
                    "reason": "defer_disable_due_to_progress",
                    "progress_state": progress.get("progress_state"),
                    "progress_reason": progress.get("reason"),
                }
            )
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
                "primary_issue_repo": primary_repo or None,
                "primary_issue_url": primary_url or None,
                "top_level_text": str(msg.get("text", "")),
                "thread_reply_count": len(thread_replies),
                "progress_signal": progress,
                "fix_request_signal": fix_request,
            }
        )

    # Prioritize actionable candidates with explicit issue linkage first, then oldest.
    candidates.sort(
        key=lambda c: (
            0 if c.get("primary_issue_number") is not None else 1,
            -float(c["age_hours"]),
        )
    )
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
