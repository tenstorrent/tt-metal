# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The repair loop must FEED prior failed approaches back to the editor so it doesn't blindly
re-author the same fix (the kernel-lever cycle). Same 'feed accumulated context' pattern as the
off-menu knob path — the repair prompt previously saw only the latest error."""

from agent.edit_agent import build_repair_prompt


def test_repair_prompt_includes_prior_attempts():
    prior = [
        {"attempt": 1, "approach": "fused eltwise into a ttl kernel", "error": "perf run crashed 9.5ms"},
        {"attempt": 2, "approach": "same ttl kernel, smaller tile", "error": "perf run crashed 9.5ms"},
    ]
    p = build_repair_prompt("tt-lang-kernel", "(section)", ["tt/x.py"], "crash: perf failed", prior_attempts=prior)
    assert "ALREADY TRIED" in p  # the do-not-repeat block is present
    assert "attempt 1" in p and "attempt 2" in p  # each prior approach listed
    assert "fused eltwise into a ttl kernel" in p  # the approach text is fed back
    assert "change the APPROACH" in p  # instructs a structurally different attempt


def test_repair_prompt_clean_without_history():
    # back-compat: no prior_attempts -> no prior block, identical to the old single-error prompt
    p = build_repair_prompt("x", "(s)", ["f.py"], "err")
    assert "ALREADY TRIED" not in p
    assert "err" in p


def test_repair_prompt_tolerates_missing_fields():
    # a prior entry with no approach recorded still renders without crashing
    p = build_repair_prompt("x", "(s)", ["f.py"], "err", prior_attempts=[{"attempt": 1, "error": "boom"}])
    assert "ALREADY TRIED" in p and "boom" in p and "approach not recorded" in p
