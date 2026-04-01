#!/usr/bin/env python3
"""Agent-driven Slack thread analysis helpers for CI triage flows."""

from __future__ import annotations

import json
import shlex
import subprocess
from typing import Any

FINAL_MARKER = "===FINAL_THREAD_ANALYSIS_JSON==="


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(shlex.quote(x) for x in cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return proc


def parse_json_after_marker(text: str, marker: str) -> dict[str, Any]:
    idx = text.find(marker)
    if idx < 0:
        raise ValueError(f"marker not found: {marker}")
    payload = text[idx + len(marker) :].strip()
    if not payload:
        raise ValueError("empty json payload after marker")
    obj = json.loads(payload)
    if not isinstance(obj, dict):
        raise ValueError("payload after marker is not a JSON object")
    return obj


def analyze_thread_with_agent(
    *,
    top_level_text: str,
    thread_replies: list[dict[str, Any]],
    bot_user_ids: set[str] | None = None,
    hold_hours_after_progress: float = 24.0,
    include_owner_claim: bool = False,
    model: str = "auto",
) -> dict[str, Any]:
    curated_replies: list[dict[str, str]] = []
    for reply in thread_replies[-20:]:
        if not isinstance(reply, dict):
            continue
        curated_replies.append(
            {
                "ts": str(reply.get("ts", "")).strip(),
                "user": str(reply.get("user", "")).strip(),
                "text": str(reply.get("text", "")).strip()[:1200],
            }
        )

    payload = {
        "top_level_text": top_level_text[:2000],
        "thread_replies": curated_replies,
        "bot_user_ids": sorted(bot_user_ids or set()),
        "hold_hours_after_progress": float(max(0.0, hold_hours_after_progress)),
        "include_owner_claim": bool(include_owner_claim),
    }
    prompt = (
        "You are analyzing a CI triage Slack thread.\n"
        "Token-efficiency rule: do one short pass and output only final JSON contract.\n"
        "Rules:\n"
        "- Use only provided text, do not invent context.\n"
        "- progress_state must be one of: resolved_signal, fix_in_progress, blocked, investigating, no_progress.\n"
        "- defer_disable=true only when there is a high-confidence active progress signal within hold window.\n"
        "- fix_request_requested=true only when someone explicitly asks agent/bot to create or draft a fix.\n"
        "- If include_owner_claim=true, detect explicit self-ownership from non-bot replies; otherwise return false/empty.\n"
        "At end print marker exactly on its own line:\n"
        f"{FINAL_MARKER}\n"
        "Then print only JSON object:\n"
        "{\n"
        '  "progress_state": "resolved_signal|fix_in_progress|blocked|investigating|no_progress",\n'
        '  "confidence": "high|medium|low",\n'
        '  "defer_disable": true,\n'
        '  "progress_reason": "short reason",\n'
        '  "fix_request_requested": false,\n'
        '  "fix_request_reason": "short reason",\n'
        '  "owner_claimed": false,\n'
        '  "owner_slack_user_id": "",\n'
        '  "owner_claim_reason": "short reason"\n'
        "}\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=True)}\n"
    )
    cmd = ["agent", "--trust", "-p", prompt]
    if model and model != "auto":
        cmd.extend(["--model", model])
    proc = run(cmd)
    parsed = parse_json_after_marker(proc.stdout or "", FINAL_MARKER)
    return {
        "progress_state": str(parsed.get("progress_state", "no_progress")).strip() or "no_progress",
        "confidence": str(parsed.get("confidence", "low")).strip() or "low",
        "defer_disable": bool(parsed.get("defer_disable", False)),
        "progress_reason": str(parsed.get("progress_reason", "unspecified")).strip() or "unspecified",
        "fix_request_requested": bool(parsed.get("fix_request_requested", False)),
        "fix_request_reason": str(parsed.get("fix_request_reason", "unspecified")).strip() or "unspecified",
        "owner_claimed": bool(parsed.get("owner_claimed", False)),
        "owner_slack_user_id": str(parsed.get("owner_slack_user_id", "")).strip(),
        "owner_claim_reason": str(parsed.get("owner_claim_reason", "unspecified")).strip() or "unspecified",
    }
