# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""integrity.Reading — a ms value may only be differenced against one measuring the same thing.

Four headlines from the 2026-07-27 audit, each reporting a change that never happened, each fixed
with its own bespoke check until the pattern became obvious:

    baseline 832.93 -> final 1088.15 (-30.6%)   2-layer profile vs a 16-layer one   (DEPTH)
    before 47.10 -> after 100.00 (-112.3%)      eager wall-clock vs trace+1cq       (MODE)
    714.94 -> 714.94 (+0.0%)                    anchor fell back to CURRENT         (STAGE)
    0.0612 ms pinned as the permanent baseline  an empty capture                    (validity)

Reading carries the three axes with the value so the guard is inherited by any number added to the
report later, instead of each one needing to rediscover the rule.
"""
from __future__ import annotations

from models.experimental.perf_automation.agent.integrity import Reading


def test_same_provenance_gives_a_delta():
    a = Reading(100.0, depth="16", mode="trace+1cq", stage="current")
    b = Reading(200.0, depth="16", mode="trace+1cq", stage="current")
    assert a.comparable_to(b).is_pass
    assert round(a.delta_pct_vs(b), 1) == 50.0


def test_depth_mismatch_refuses_the_delta():
    """832.93 (2 layers) vs 1088.15 (16 layers)."""
    a = Reading(1088.15, depth="16", mode="tracy")
    b = Reading(832.93, depth="2", mode="tracy")
    assert not a.comparable_to(b)
    assert a.delta_pct_vs(b) is None
    assert "depth differs" in a.comparable_to(b).reason


def test_mode_mismatch_refuses_the_delta():
    """47.10 [eager] vs 100.00 [trace+1cq]."""
    a = Reading(100.0, depth="all", mode="trace+1cq")
    b = Reading(47.10, depth="all", mode="eager")
    assert a.delta_pct_vs(b) is None
    assert "mode differs" in a.comparable_to(b).reason


def test_stage_mismatch_refuses_the_delta():
    """An anchor must predate every lever; comparing current against current prints +0.0%."""
    a = Reading(714.94, depth="16", stage="current")
    b = Reading(714.94, depth="16", stage="current")
    assert a.comparable_to(b).is_pass, "same stage IS comparable; the bug was WHICH value was chosen"
    c = Reading(2464.18, depth="16", stage="baseline")
    assert not a.comparable_to(c)


def test_an_axis_known_on_one_side_only_is_not_assumed_to_match():
    """The exact drift case: a value captured before the mode changed underneath it."""
    a = Reading(100.0, mode="trace+1cq")
    b = Reading(200.0, mode="")
    assert a.comparable_to(b).is_unknown
    assert a.delta_pct_vs(b) is None


def test_legacy_readings_with_no_provenance_still_compare():
    """Refusing every historical pair would be noise, not safety."""
    a, b = Reading(100.0), Reading(200.0)
    assert a.comparable_to(b).is_pass and round(a.delta_pct_vs(b), 1) == 50.0


def test_an_unusable_value_is_never_a_measurement():
    for bad in (0, -1, None, "fast", 0.0):
        r = Reading(bad, depth="16")
        assert not r.ok and not r
        assert r.delta_pct_vs(Reading(100.0, depth="16")) is None


def test_axes_are_case_and_whitespace_insensitive():
    assert Reading(1.0, mode=" Trace+1CQ ").comparable_to(Reading(2.0, mode="trace+1cq")).is_pass


def test_label_never_prints_a_bare_number():
    assert Reading(714.94, depth="16", mode="tracy").label() == "714.94 ms [16 layers, tracy]"
    assert Reading(23.89, depth="all", mode="trace+1cq").label() == "23.89 ms [all, trace+1cq]"
    assert Reading(None).label() == "n/a"


def test_comparing_against_a_non_reading_is_unknown_not_true():
    assert Reading(1.0).comparable_to(1.0).is_unknown
