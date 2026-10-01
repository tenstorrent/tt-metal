# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-free tests for the overlap mode's gather-slot bookkeeping (``ttml.fsdp.SlotSchedule``).

The schedule decides which compute release a gather into a reused persistent buffer must wait for.
The invariant under test: with ``2 * lookahead`` slots, a gather into a slot waits for the release
``lookahead`` positions after the slot's last release, or for everything (``None``) when that
release does not exist yet.
"""

from ttml.fsdp import SlotSchedule


def test_lookahead_must_be_positive(expect_error):
    with expect_error(ValueError, "lookahead must be at least 1"):
        SlotSchedule(0)


def test_slot_count_is_twice_the_lookahead():
    assert SlotSchedule(1).num_slots == 2
    assert SlotSchedule(2).num_slots == 4
    assert SlotSchedule(2).slot_of(6) == 2


def test_new_slot_needs_a_full_drain():
    schedule = SlotSchedule(2)
    assert schedule.wait_target(0) is None


def test_forward_pass_waits_lookahead_units_behind():
    lookahead, units = 2, 10
    schedule = SlotSchedule(lookahead)
    releases = {}
    for unit in range(units):
        # Gather of `unit` is issued while `unit - 1` computes; then `unit` computes and releases.
        slot = schedule.slot_of(unit)
        target = schedule.wait_target(slot)
        if unit < schedule.num_slots:
            assert target is None  # first use of every slot: buffer is new
        else:
            # The slot was last read by unit - 2*lookahead; wait for the release of unit - lookahead.
            assert target == releases[unit - lookahead]
        releases[unit] = schedule.release(slot)


def test_target_not_yet_recorded_falls_back_to_drain():
    schedule = SlotSchedule(2)
    schedule.release("a")  # seq 0
    schedule.release("b")  # seq 1
    assert schedule.wait_target("a") is None  # seq 2 does not exist yet
    schedule.release("c")  # seq 2
    assert schedule.wait_target("a") == 2
    assert schedule.wait_target("b") is None


def test_backward_reverses_direction_with_the_same_rule():
    """Forward 0..4 releases in order, backward 4..0 re-releases; the target is always 'lookahead
    releases after the slot's latest release', regardless of which unit that was."""
    schedule = SlotSchedule(1)  # 2 slots
    fwd = [schedule.release(schedule.slot_of(u)) for u in range(5)]  # seqs 0..4
    # Backward: unit 4 (slot 0) has the newest release of slot 0 (seq 4); nothing after it yet.
    assert schedule.wait_target(0) is None
    bwd4 = schedule.release(0)  # seq 5
    # Unit 3 (slot 1): slot 1's last release was unit 3's forward (seq 3); target seq 4 = unit 4's forward.
    assert schedule.wait_target(1) == fwd[4]
    schedule.release(1)  # seq 6
    # Unit 2 (slot 0): last release seq 5 (unit 4's backward); target seq 6 = unit 3's backward.
    assert schedule.wait_target(0) == bwd4 + 1


def test_oldest_needed_keeps_events_a_dedicated_slot_may_still_target():
    schedule = SlotSchedule(2)
    assert schedule.oldest_needed() == 0
    schedule.release("kept-unit")  # seq 0, released once per step
    for unit in range(1, 40):
        schedule.release(schedule.slot_of(unit))
    # The kept unit's slot still points at seq 0, so nothing may be dropped ...
    assert schedule.oldest_needed() == 0
    assert schedule.wait_target("kept-unit") == 2
    # ... until it is released again.
    schedule.release("kept-unit")
    assert schedule.oldest_needed() > 0
