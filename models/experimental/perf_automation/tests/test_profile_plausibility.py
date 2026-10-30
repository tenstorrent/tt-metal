# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A profile must be BELIEVABLE before it becomes the permanent baseline.

llama3_1_8b_p150, 2026-07-27: one capture reported device_ms=0.0612 with its op buckets totalling
2.657 ms, against a real profile of 2464 ms -- 61 microseconds for an 8B model. It dropped no
markers, so capture_partial (the only quality check) passed it, and because the ORIGINAL baseline is
written once and never refreshed, it became the permanent anchor. The final headline would have read
"0.06 ms -> 714.94 ms" for a run that actually achieved 2464 -> 715.

It also carried a correct depth stamp and a correct method, so BOTH comparability guards accepted it.
Comparability asks "are these the same kind of number"; this asks "can this number be true at all".
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cc_optimize"))

import perf_mcp as P  # noqa: E402


def test_the_real_garbage_profile_is_refused():
    """The exact shape that got pinned: a near-zero total with near-zero buckets."""
    assert P._is_credible_profile({"device_ms": 0.0612, "buckets": [{"device_ms": 2.657}]}) is False


def test_a_real_profile_is_accepted():
    assert (
        P._is_credible_profile({"device_ms": 2464.18, "buckets": [{"device_ms": 1205.58}, {"device_ms": 1010.23}]})
        is True
    )


def test_zero_and_negative_are_not_speeds():
    for dev in (0.0, -1.0, 0.0001):
        assert P._is_credible_profile({"device_ms": dev}) is False


def test_missing_or_unparseable_device_ms_is_refused():
    assert P._is_credible_profile({}) is False
    assert P._is_credible_profile({"device_ms": None}) is False
    assert P._is_credible_profile({"device_ms": "fast"}) is False


def test_total_far_below_its_own_buckets_is_an_incomplete_capture():
    """A capture that recorded a few ops but reports them as the whole model."""
    assert P._is_credible_profile({"device_ms": 10.0, "buckets": [{"device_ms": 900.0}]}) is False


def test_total_slightly_below_buckets_is_still_accepted():
    """Bucket sums do not have to match device_ms exactly -- host gaps and rounding differ. Only an
    order-of-magnitude disagreement indicates a broken capture."""
    assert P._is_credible_profile({"device_ms": 900.0, "buckets": [{"device_ms": 1000.0}]}) is True


def test_no_buckets_falls_back_to_the_floor_check_only():
    assert P._is_credible_profile({"device_ms": 500.0}) is True
    assert P._is_credible_profile({"device_ms": 0.5}) is False


def test_malformed_buckets_do_not_crash_the_guard():
    assert P._is_credible_profile({"device_ms": 500.0, "buckets": [None, "x", {"device_ms": "y"}]}) is True


def test_floor_is_env_overridable(monkeypatch):
    """A genuinely sub-millisecond module could exist; the floor must be tunable rather than absolute."""
    assert P._MIN_CREDIBLE_DEVICE_MS == 1.0


# --- integration: the guard must actually prevent the PIN, not merely return False ---------------


def _orig_path():
    return Path(P._original_baseline_path())


def _write_path():
    return Path(P._baseline_path())


def _simulate_profile_write(prof):
    """Reproduce profile_model's two writes verbatim: the rolling baseline always, the ORIGINAL only
    when it does not exist AND the profile is credible."""
    import json

    _write_path().write_text(json.dumps(prof))
    _orig = _orig_path()
    if not _orig.exists() and P._is_credible_profile(prof):
        stamped = dict(prof)
        stamped["perf_layers"] = "16"
        _orig.write_text(json.dumps(stamped))
