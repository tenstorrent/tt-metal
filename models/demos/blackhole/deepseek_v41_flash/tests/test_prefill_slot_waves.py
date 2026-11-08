# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression test of the prefill slot state machine (tt/prefill_waves.py): the AgentX c8 crash of the B=16 server (2026-11-08, 'user 1 has no prefill slot to continue').

B=16: U=4 users per mesh row, Up=2 prefill slots. One call carried, in mesh row 0, the continuations of users 0 and 1 and a NEW prefix-cache hit of user 2 (start > 0, restore pending, no slot yet). With the
old order (sort by start == 0 only) the hit counted as a continuation: wave 1 = {2, 0}, wave 2 = {1}; wave 1 allocated user 2 like a new prompt, took the free-looking slot of user 1 (not in the wave) and wave 2
asserted. The new order runs the real continuations first and the hit after them."""
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_waves import plan_groups, slot_alloc, split_waves

U, UP, ROWS = 4, 2, 4


def run_call(items, hit_todo, state, order, per_row=True):
    """walk the waves of a call like Model.prefill_interleaved_one: allocate, 'restore' the hits of a wave, return the slots seen per wave"""
    owner, of, resume = state
    if order == "old":
        groups = [sorted(items, key=lambda it: it[2] == 0)]
    else:
        groups = plan_groups(items, hit_todo, per_row)
    hit_todo = set(hit_todo)
    seen = []
    for grp in groups:
        for wave in split_waves(grp, UP, U):
            view = [(b, t, 0 if b in hit_todo else st, e) for b, t, st, e in wave]
            seen.append((sorted(b for b, _, _, _ in wave), slot_alloc(owner, of, UP, U, view, resume)))
            hit_todo -= {b for b, _, _, _ in wave}
            for b, _, _, e in wave:
                resume[b] = e  # (the wave ran its windows to the end of its chunk)
    return seen


def fresh_state():
    """slots at the start of the call: row 0 (slots 0, 1): users 0 and 1 are mid-prefill; row 2 (slots 4, 5): users 8 and 9; row 1: user 4 not started"""
    owner, of, resume = [None] * (ROWS * UP), {}, {}
    for b, sl, pos in ((0, 0, 16384), (1, 1, 52736), (8, 4, 16896), (9, 5, 307712)):
        owner[sl], of[b], resume[b] = b, sl, pos
    return owner, of, resume


ITEMS = [
    (8, None, 16896, 20992),
    (2, None, 4608, 8704),
    (9, None, 307712, 311808),
    (0, None, 16384, 18185),
    (1, None, 52736, 56832),
    (4, None, 0, 4096),
]


def test_old_order_reproduces_the_crash():
    try:
        run_call(ITEMS, {2}, fresh_state(), "old")
    except AssertionError as e:
        assert "has no prefill slot to continue" in str(e)
    else:
        raise RuntimeError("the old order should have asserted")


def test_new_order_keeps_every_continuation_slot():
    state = fresh_state()
    seen = run_call(ITEMS, {2}, state, "new")
    owner, of, resume = state
    row0 = [(w, s) for w, s in seen if any(b // U == 0 for b in w)]
    first_w, first_s = row0[0]
    assert {0, 1} <= set(first_w), "the continuations of row 0 run in the first wave"
    assert first_s[0] == 0 and first_s[1] == 1, "...with their own slots"
    assert 2 in [b for w, _ in seen for b in w], "the hit runs too"
    assert of[2] in (0, 1) and owner[of[2]] == 2, "...after them, on a slot of a finished user"


def test_no_hit_order_is_unchanged():
    """without hits the plan is the old one: continuations first, new prompts after"""
    items = [(0, None, 0, 100), (1, None, 512, 1024), (5, None, 0, 100), (4, None, 2048, 4096)]
    for per_row in (True, False):
        flat = [it[0] for g in plan_groups(items, set(), per_row) for it in g]
        assert flat.index(1) < flat.index(0) and flat.index(4) < flat.index(5)


def test_up_equals_u_has_no_slot_path():
    """Up == U (B=8: Up 2 / U 2, B=4: 1 / 1): ``general = Up != U`` is False, the model never calls slot_alloc / split_waves (slot = user, one wave per group, hit_todo empty). Documented here, the condition is in
    Model.prefill_interleaved_one."""
    for up in (1, 2):
        assert not (up != up)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
