# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of ``BlockIndex.peek_home`` / ``ApcPagedHook.peek_home`` (read-only home-row look for the scheduler's bounded hold)."""
import copy

from test_apc_paged_index import AP, C, mk, run, toks


def test_peek_home_is_read_only_and_reports_the_busy_home_row():
    idx = mk()
    a = toks(1, 6 * C + 300)
    run(idx, "A", a, user=0)
    idx.bind("B", 0, user=0, prompt_len=1000)
    idx.bind("C", 0, user=1, prompt_len=1000)  # row 0 (the home of A's blocks) has no free user
    a2 = list(a) + list(toks(2, 4 * C))
    before = (dict(idx.counters), sorted(idx._reqs), dict(idx._reserved), dict(idx._busy), len(idx._rows[0].free_pages))
    hit, home, free = idx.peek_home(a2, len(a2))
    assert (hit > 0, home, free) == (True, 0, [0, 2])
    assert before == (dict(idx.counters), sorted(idx._reqs), dict(idx._reserved), dict(idx._busy), len(idx._rows[0].free_pages))
    # plan_hit (which does the real work) agrees on the hit, and loses it to the busy row
    assert idx.plan_hit("A2", a2, len(a2)).hit_tokens == 0 and idx.counters["busy_loss_tokens"] == hit


def test_peek_home_free_home_row_and_unknown_prompt():
    idx = mk()
    a = toks(3, 5 * C + 10)
    run(idx, "A", a, user=0)
    hit, home, free = idx.peek_home(a, len(a))
    assert hit > 0 and home == 0 and free[0] == 2
    assert idx.peek_home(toks(4, 3 * C), 3 * C)[:2] == (0, -1)
    assert idx.peek_home(a, 100)[:2] == (0, -1)  # too short for a hit


def test_hook_peek_home():
    idx = mk()
    a = toks(5, 5 * C + 10)
    run(idx, "A", a, user=0)
    hook = AP.ApcPagedHook(idx)
    assert hook.peek_home("X", a, len(a)) == idx.peek_home(a, len(a))
    hook2 = AP.ApcPagedHook(idx, key_tokens_fn=lambda t, mm: None)
    assert hook2.peek_home("X", a, len(a)) == (0, -1, [])
