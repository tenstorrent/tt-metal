#!/usr/bin/env python3
"""Thread-signal analysis helpers for CI triage lifecycle decisions."""

from __future__ import annotations

import re
import time
from typing import Any


def _to_float_ts(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _normalize(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _reply_texts(thread_replies: list[dict[str, Any]]) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    for reply in thread_replies:
        if not isinstance(reply, dict):
            continue
        text = _normalize(str(reply.get("text", "")))
        ts = _to_float_ts(reply.get("ts"))
        if not text:
            continue
        rows.append((ts if ts is not None else 0.0, text))
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows


def classify_thread_progress(
    *,
    top_level_text: str,
    thread_replies: list[dict[str, Any]],
    now_unix: float | None = None,
    hold_hours_after_progress: float = 24.0,
) -> dict[str, Any]:
    now = time.time() if now_unix is None else now_unix
    replies = _reply_texts(thread_replies)
    if not replies:
        return {
            "progress_state": "no_progress",
            "confidence": "low",
            "recent_progress": False,
            "defer_disable": False,
            "reason": "no_thread_replies",
        }

    resolved_markers = (
        "fixed",
        "resolved",
        "landed",
        "merged",
        "closed by",
        "issue closed",
    )
    in_progress_markers = (
        "working on",
        "looking now",
        "investigating",
        "wip",
        "in progress",
        "opening pr",
        "will send pr",
        "pr soon",
        "have a fix",
    )
    blocked_markers = (
        "blocked",
        "waiting on",
        "needs infra",
        "needs hardware",
        "cannot reproduce",
    )
    stale_noise_markers = (
        "taking a look",
        "will check",
        "looking",
    )

    latest_ts, latest_text = replies[0]
    age_hours = (now - latest_ts) / 3600.0 if latest_ts > 0 else 1e9
    recent = age_hours <= hold_hours_after_progress

    has_pr_link = "github.com/" in latest_text and "/pull/" in latest_text
    has_issue_link = "github.com/" in latest_text and "/issues/" in latest_text
    has_resolved = any(marker in latest_text for marker in resolved_markers)
    has_progress = any(marker in latest_text for marker in in_progress_markers)
    has_blocked = any(marker in latest_text for marker in blocked_markers)
    has_noise = any(marker in latest_text for marker in stale_noise_markers)

    if has_resolved and (has_pr_link or has_issue_link):
        return {
            "progress_state": "resolved_signal",
            "confidence": "high",
            "recent_progress": True,
            "defer_disable": True,
            "reason": "latest_reply_claims_resolution_with_link",
            "latest_reply_age_hours": round(age_hours, 2),
        }
    if has_progress and (has_pr_link or recent):
        return {
            "progress_state": "fix_in_progress",
            "confidence": "high" if has_pr_link else "medium",
            "recent_progress": recent,
            "defer_disable": recent,
            "reason": "latest_reply_indicates_active_fix",
            "latest_reply_age_hours": round(age_hours, 2),
        }
    if has_blocked:
        return {
            "progress_state": "blocked",
            "confidence": "medium",
            "recent_progress": recent,
            "defer_disable": False,
            "reason": "latest_reply_indicates_blocker",
            "latest_reply_age_hours": round(age_hours, 2),
        }
    if has_noise:
        return {
            "progress_state": "investigating",
            "confidence": "low",
            "recent_progress": recent,
            "defer_disable": recent and age_hours <= min(hold_hours_after_progress, 8.0),
            "reason": "latest_reply_is_vague_progress_signal",
            "latest_reply_age_hours": round(age_hours, 2),
        }

    # Check if any recent reply (not just latest) has strong progress.
    for ts, text in replies[:5]:
        age = (now - ts) / 3600.0 if ts > 0 else 1e9
        if age > hold_hours_after_progress:
            continue
        if any(marker in text for marker in in_progress_markers) and ("github.com/" in text or "pr" in text):
            return {
                "progress_state": "fix_in_progress",
                "confidence": "medium",
                "recent_progress": True,
                "defer_disable": True,
                "reason": "recent_reply_indicates_active_fix",
                "latest_reply_age_hours": round(age_hours, 2),
            }

    return {
        "progress_state": "no_progress",
        "confidence": "low",
        "recent_progress": False,
        "defer_disable": False,
        "reason": "no_high_confidence_progress_signal",
        "latest_reply_age_hours": round(age_hours, 2),
    }


def detect_dev_fix_request(*, top_level_text: str, thread_replies: list[dict[str, Any]]) -> dict[str, Any]:
    corpus = [_normalize(top_level_text)]
    corpus.extend(text for _, text in _reply_texts(thread_replies)[:12])
    joined = "\n".join(corpus)
    patterns = (
        r"\b(can|could|please)\s+(you|agent|cursor)\s+(fix|patch|handle)\b",
        r"\bauto\s*fix\b",
        r"\bdraft\s+(a\s+)?fix\s+pr\b",
        r"\bmake\s+a\s+pr\s+to\s+fix\b",
    )
    for pat in patterns:
        if re.search(pat, joined):
            return {"requested": True, "reason": f"matched_pattern:{pat}"}
    return {"requested": False, "reason": "no_fix_request_signal"}
