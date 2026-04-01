#!/usr/bin/env python3
"""Testing-mode thread persona simulation for CI triage behavior."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Ensure repository root is importable when running as `python tools/ci/<script>.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ci.thread_signal_analysis import classify_thread_progress, detect_dev_fix_request


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post simulated developer thread replies and evaluate triage interpretation."
    )
    p.add_argument("--slack-channel-id", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--summary-md", required=True)
    return p.parse_args()


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


def post_message(*, token: str, channel: str, text: str, thread_ts: str | None = None) -> str:
    fields = {"channel": channel, "text": text}
    if thread_ts:
        fields["thread_ts"] = thread_ts
    payload = slack_api_form(token, "chat.postMessage", fields)
    if not payload.get("ok"):
        raise RuntimeError(f"chat.postMessage failed: {payload.get('error', 'unknown_error')}")
    return str(payload.get("ts", "")).strip()


def build_summary(result: dict[str, Any]) -> str:
    lines = [
        "## Thread Persona Simulation Session",
        "",
        f"- Anchor thread ts: `{result.get('anchor_ts','')}`",
        f"- Personas simulated: {len(result.get('scenarios', []))}",
        "",
        "## Scenario Results",
    ]
    for row in result.get("scenarios", []):
        lines.append(
            f"- `{row['name']}` -> state `{row['progress_state']}`, defer_disable={row['defer_disable']}, "
            f"fix_request={row['fix_request_requested']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    token = require_env("SLACK_BOT_TOKEN")
    anchor_text = (
        "CI triage testing-mode thread persona simulation anchor. "
        "This is synthetic test data and not a real incident."
    )
    anchor_ts = post_message(token=token, channel=args.slack_channel_id, text=anchor_text)

    scenarios = [
        ("active_plan", "Looking now, I will post PR in 2 hours."),
        ("wip_pr", "Fix in progress: https://github.com/tenstorrent/tt-metal/pull/12345"),
        ("blocked", "Blocked on infra/hardware dependency."),
        ("resolved_claim", "Should be fixed by https://github.com/tenstorrent/tt-metal/pull/12346; please verify."),
        ("vague", "Taking a look."),
        ("fix_request", "Can agent fix this and draft a fix PR if possible?"),
    ]

    outputs: list[dict[str, Any]] = []
    for name, text in scenarios:
        ts = post_message(token=token, channel=args.slack_channel_id, text=text, thread_ts=anchor_ts)
        progress = classify_thread_progress(
            top_level_text=anchor_text,
            thread_replies=[{"ts": ts, "text": text}],
        )
        fix_req = detect_dev_fix_request(top_level_text=anchor_text, thread_replies=[{"ts": ts, "text": text}])
        outputs.append(
            {
                "name": name,
                "reply_ts": ts,
                "reply_text": text,
                "progress_state": progress.get("progress_state"),
                "defer_disable": bool(progress.get("defer_disable", False)),
                "progress_reason": progress.get("reason", ""),
                "fix_request_requested": bool(fix_req.get("requested", False)),
                "fix_request_reason": fix_req.get("reason", ""),
            }
        )

    result = {"anchor_ts": anchor_ts, "scenarios": outputs}
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    out_md = Path(args.summary_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_summary(result), encoding="utf-8")
    print(json.dumps({"anchor_ts": anchor_ts, "scenario_count": len(outputs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
