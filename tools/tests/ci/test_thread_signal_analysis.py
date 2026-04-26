from __future__ import annotations

from tools.ci.thread_signal_analysis import classify_thread_progress, detect_dev_fix_request


def test_classify_thread_progress_detects_high_confidence_fix_in_progress() -> None:
    now = 2000000000.0
    replies = [{"ts": str(now - 3600), "text": "Working on it now, PR soon: https://github.com/foo/bar/pull/123"}]
    out = classify_thread_progress(
        top_level_text="failure",
        thread_replies=replies,
        now_unix=now,
        hold_hours_after_progress=24.0,
    )
    assert out["progress_state"] == "fix_in_progress"
    assert out["defer_disable"] is True
    assert out["confidence"] in {"high", "medium"}


def test_classify_thread_progress_does_not_defer_for_blocked_reply() -> None:
    now = 2000000000.0
    replies = [{"ts": str(now - 1800), "text": "Blocked on hardware infra dependency"}]
    out = classify_thread_progress(
        top_level_text="failure",
        thread_replies=replies,
        now_unix=now,
        hold_hours_after_progress=24.0,
    )
    assert out["progress_state"] == "blocked"
    assert out["defer_disable"] is False


def test_classify_thread_progress_ignores_old_progress_outside_hold_window() -> None:
    now = 2000000000.0
    replies = [{"ts": str(now - 60 * 60 * 30), "text": "working on it, PR soon"}]
    out = classify_thread_progress(
        top_level_text="failure",
        thread_replies=replies,
        now_unix=now,
        hold_hours_after_progress=24.0,
    )
    assert out["defer_disable"] is False


def test_detect_dev_fix_request_matches_explicit_request() -> None:
    out = detect_dev_fix_request(
        top_level_text="Incident details",
        thread_replies=[{"ts": "1", "text": "Can agent fix this and draft a fix PR?"}],
    )
    assert out["requested"] is True
