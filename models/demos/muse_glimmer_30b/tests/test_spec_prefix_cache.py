# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the spec-rail prefix-cache bookkeeping (no device)."""

import torch

from models.demos.muse_glimmer_30b.tt import spec_prefix_cache as sp


def test_drafter_rows_needed_matches_a_from_scratch_prefill():
    assert sp.drafter_rows_needed(100) == (0, 100)
    assert sp.drafter_rows_needed(5000) == (2944, 5000)  # (5000 - 2048) // 32 * 32
    assert sp.drafter_rows_needed(2048) == (0, 2048)


def test_drafter_gap_is_zero_when_the_new_prompt_extends_the_old_one():
    assert sp.drafter_gap(5000, 4864, 8000) == 0  # the window of the new prompt lies in the suffix taps
    assert sp.drafter_gap(5000, 4864, 5100) == 0  # rows 3040..4864 come from the shadow drafter ring (2944..)
    assert sp.drafter_gap(9000, 8960, 9100) == 0


def test_drafter_gap_counts_rows_older_than_the_shadow_ring():
    # new prompt ends at 8000 but the resume point is 8960: not a real plan, but the helper counts rows 5952..6944 (the shadow ring starts at (9000 - 2048) // 32 * 32)
    assert sp.drafter_gap(9000, 8960, 8000) == 992


def _slots(prompts):
    c = sp.SlotPrefixCache(n_slots=8)
    for e, t in prompts.items():
        c.record_prefill(e, t)
    return c


def test_plan_gives_back_the_slot_that_holds_the_prefix():
    a, b = torch.arange(5000), torch.arange(100_000, 105_000)
    c = _slots({2: a, 5: b})
    nxt = torch.cat([a, torch.arange(70_000, 73_000)])
    plan = c.plan(nxt, cached_tokens=4992, idle_slots=[2, 5, 6])
    assert (plan.engine_slot, plan.start, plan.drafter_gap_rows) == (2, 4864, 0)
    assert c.hits == 1


def test_plan_ignores_busy_slots_and_short_cached_lengths():
    a = torch.arange(5000)
    c = _slots({2: a})
    nxt = torch.cat([a, torch.arange(70_000, 71_000)])
    assert c.plan(nxt, 4992, idle_slots=[3, 4]) is None  # slot 2 is live
    assert c.plan(nxt, 4800, idle_slots=[2]) is None  # vLLM cached less than the boundary 4864
    assert c.plan(nxt, 200, idle_slots=[2]) is None
    assert c.misses == 2


def test_plan_rejects_a_diverged_prompt_and_picks_the_longest_boundary():
    a = torch.arange(6000)
    c = _slots({1: a, 4: a[:3000]})
    d = a.clone()
    d[5000:] += 1  # diverges after 5000: boundary of slot 1 (5888) is past the divergence, slot 4 (2816) is before it
    plan = c.plan(torch.cat([d, torch.arange(10)]), 5120, idle_slots=[1, 4])
    assert (plan.engine_slot, plan.start) == (4, 2816)
    same = torch.cat([a, torch.arange(50_000, 50_500)])
    assert c.plan(same, 6016, idle_slots=[1, 4]).engine_slot == 1


def test_victim_prefers_empty_then_least_recently_used():
    c = _slots({0: torch.arange(300), 1: torch.arange(400), 2: torch.arange(500)})
    assert c.victim([0, 1, 2, 3]) == 3
    assert c.victim([0, 1, 2]) == 0
    c.record_prefill(0, torch.arange(300))  # slot 0 touched again
    assert c.victim([0, 1, 2]) == 1


def test_swap_phys_keeps_a_permutation():
    phys = [0, 1, 2, 3]
    assert sp.swap_phys(phys, plugin_slot=0, engine_slot=3) == [3, 1, 2, 0]
    assert sp.swap_phys(phys, plugin_slot=2, engine_slot=2) == phys
    assert sorted(sp.swap_phys([2, 0, 3, 1], 1, 1)) == [0, 1, 2, 3]
