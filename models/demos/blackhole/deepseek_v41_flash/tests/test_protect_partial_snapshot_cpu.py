# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of DSV41_SNAPSHOT_ON_EVICT (T24): the carried halo state of a prompt in progress is saved as a record before its slot is taken (``PagedApc.snapshot``), and the evicting
``Model._slot_alloc`` hook calls it. Fake model / stores as in test_apc_paged_model_cpu."""
from models.demos.blackhole.deepseek_v41_flash.tests.test_apc_paged_model_cpu import C, pa, prefill, toks  # noqa: F401
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_waves import slot_alloc


def halo_blocks(pa_, row=0):
    return sorted(b.idx for b in pa_.idx._rows[row].blocks.values() if b.halo_a >= 0)


def test_snapshot_adds_the_halo_record_at_the_carried_position(pa):
    t = toks(5, 20000)  # 39 blocks: the per-prompt budget thins the records to every 2nd block
    prefill(pa, 0, "A", t, e=7 * C, tot=20000)
    assert halo_blocks(pa) == [2, 4, 6]  # (block 7 has none: the carried state at 3584 would be lost)
    n0 = len([c for c in pa.hs.calls if c[0] == "set_save"])
    assert pa.snapshot(0, 7 * C) is True
    assert halo_blocks(pa) == [2, 4, 6, 7]
    saves = [c for c in pa.hs.calls if c[0] == "set_save"][n0:]
    assert (
        saves and saves[0][1][0] == "A" and saves[0][1][1][pa.lane(0)] is not None
    )  # the lane of the user saved into a fresh slot
    assert pa.stats["snapshots"] == 1
    pa.idx.check()
    # the record is usable: a continuation of the same request planned with max_hit = 3584 finds it
    pa.model.pool.release(0)
    plan = pa.idx.plan_hit("A2", t.numpy(), 20000, max_hit=7 * C)
    assert plan.hit_tokens == 7 * C


def test_snapshot_is_a_noop_where_a_record_exists_or_without_a_binding(pa):
    t = toks(6, 20000)
    prefill(pa, 0, "A", t, e=6 * C, tot=20000)
    n = len(pa.hs.calls)
    assert pa.snapshot(0, 6 * C) is True and len(pa.hs.calls) == n  # block 6 has its record already
    assert pa.snapshot(3, 6 * C) is False  # no request bound to user 3
    assert pa.snapshot(0, 6 * C + 100) is False  # not on the chunk grid


def test_slot_alloc_hook_order_snapshot_before_the_owner_is_forgotten():
    seen = []
    resume = {0: 3584}
    owner, of = [0, 1, None, None], {0: 0, 1: 1}
    resume[1] = 512
    slot_alloc(
        owner,
        of,
        2,
        4,
        [(2, None, 0, 512)],
        resume,
        on_evict=lambda o: seen.append((o, dict(resume), dict(of))),
        cheapest=True,
    )
    assert seen == [
        (1, {0: 3584, 1: 512}, {0: 0, 1: 1})
    ]  # the cheapest owner, its carried state and slot still known when the hook runs
