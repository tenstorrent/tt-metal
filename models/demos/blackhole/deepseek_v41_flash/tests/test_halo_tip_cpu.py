# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of DSV41_APC_PAGED_HALO_TIP (T24): the halo record of a prompt's tip block survives short sessions' record churn."""
import importlib.util
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("apc_halo_tip", os.path.join(_HERE, "../tt/apc_paged.py"))
AP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AP)
C = 512


def toks(seed, n):
    return np.random.default_rng(seed).integers(1, 100000, size=n, dtype=np.int64)


def mk(tip, a=24):
    return AP.BlockIndex(
        rows=1,
        users_per_row=4,
        pages_per_row=20000,
        halo_a=a,
        halo_b=8,
        key_slots=8000,
        zone=2560,
        halo_budget=16,
        halo_tip=tip,
    )


def run(idx, rid, t, user, release=True):
    e = len(t)
    idx.bind(rid, 0, user=user)
    idx.restored(rid)
    idx.page_table(rid, e)
    for st in range(0, e // C * C, C):
        cp = idx.begin_chunk(rid, st, st + C, t, e)
        if cp is not None:
            idx.finish_chunk(cp)
    if release:
        idx.release(rid)


def giant_hit(idx, giant):
    plan = idx.plan_hit("next-turn", giant, len(giant) + 100)
    n = plan.hit_tokens
    idx.cancel("next-turn")
    return n


def scenario(tip):
    idx = mk(tip)
    giant = toks(1, 300_000)
    run(idx, "g", giant, 0)  # a finished giant: ~17 records + tip
    for k in range(30):  # short sessions (24k..40k tokens, ~5 records each) churn the 24-slot A store
        run(idx, f"s{k}", toks(100 + k, 24_000 + 500 * k), 1 + k % 3)
    return giant_hit(idx, giant), idx


def test_tip_record_survives_churn_only_with_the_flag():
    off, _ = scenario(False)
    on, idx = scenario(True)
    assert off < 100_000, off  # old LRU: the giant's records are gone, the next turn restarts (almost) cold
    assert on >= 299_000, on  # the tip record (and its chain) is kept
    assert idx.check()


def test_superseded_tip_is_not_protected():
    """a later turn extends the chain: the older tip has children and becomes evictable again"""
    idx = mk(True, a=40)
    t = toks(2, 60_000)
    run(idx, "t1", t[:30_000], 0)
    run(idx, "t2", t, 0)
    tips = [b for b in idx._rows[0].blocks.values() if b.tipb]
    assert len(tips) == 2
    leaf = [b for b in tips if b.children == 0]
    assert len(leaf) == 1 and leaf[0].idx * C >= 59_000


def test_default_off_identical():
    a = mk(False)
    b = AP.BlockIndex(
        rows=1, users_per_row=4, pages_per_row=20000, halo_a=24, halo_b=8, key_slots=8000, zone=2560, halo_budget=16
    )
    g = toks(3, 120_000)
    for idx in (a, b):
        run(idx, "g", g, 0)
        for k in range(10):
            run(idx, f"s{k}", toks(50 + k, 30_000), 1)
    assert sorted(x.idx for x in a._rows[0].a_lru.values()) == sorted(x.idx for x in b._rows[0].a_lru.values())
