# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the halo record schedule / in-flight protection of the block index (T24, DSV41_APC_PAGED_HALO_SPACING, DSV41_PROTECT_PARTIAL). No device."""
import importlib.util
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("apc_paged_protect", os.path.join(_HERE, "../tt/apc_paged.py"))
AP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AP)
C = 512


def toks(seed, n):
    return np.random.default_rng(seed).integers(1, 100000, size=n, dtype=np.int64)


def mk(**kw):
    d = dict(rows=1, users_per_row=4, pages_per_row=6000, halo_a=400, halo_b=40, key_slots=3000, zone=2560)
    d.update(kw)
    return AP.BlockIndex(**d)


def feed(idx, rid, t, user, upto=None, total=None):
    """a cold request, chunks [0, upto) committed (the whole prompt when upto is None); the request stays bound"""
    e = total or len(t)
    idx.bind(rid, 0, user=user)
    idx.restored(rid)
    idx.page_table(rid, e)
    n = (e // C * C) if upto is None else upto
    for st in range(0, n, C):
        cp = idx.begin_chunk(rid, st, st + C, t, e)
        if cp is not None:
            idx.finish_chunk(cp)


def halo_positions(idx, row=0):
    return sorted(b.idx * C for b in idx._rows[row].a_lru.values())


def test_record_schedule_budget_vs_spacing():
    t = toks(1, 300_000)
    base = mk(halo_budget=16)
    feed(base, "a", t, 0)
    pos0 = halo_positions(base)
    dense = mk(halo_budget=16, halo_spacing=16384, halo_spacing_min=100000, halo_spacing_max=64)
    feed(dense, "a", t, 0)
    pos1 = halo_positions(dense)
    assert 14 <= len(pos0) <= 18 and max(b - a for a, b in zip(pos0, pos0[1:])) <= 19_456 + C
    assert (
        18 <= len(pos1) <= 22 and max(b - a for a, b in zip(pos1, pos1[1:])) <= 16384 + C
    )  # a record every 16 k tokens
    # short prompts keep the budget rule
    s = toks(2, 60_000)
    a = mk(halo_budget=16)
    feed(a, "s", s, 0)
    b = mk(halo_budget=16, halo_spacing=16384)
    feed(b, "s", s, 0)
    assert halo_positions(a) == halo_positions(b)
    # 1 M tokens: at most halo_spacing_max records
    big = mk(halo_budget=16, halo_spacing=16384, halo_spacing_max=64, pages_per_row=20000, key_slots=2100, halo_a=400)
    feed(big, "g", toks(3, 1_000_000), 0)
    assert len(halo_positions(big)) <= 66


def test_default_off_is_identical():
    t = toks(4, 200_000)
    a = mk(halo_budget=16)
    b = mk(halo_budget=16, halo_spacing=0, protect_fn=lambda: False)
    feed(a, "x", t, 0)
    feed(b, "x", t, 0)
    assert halo_positions(a) == halo_positions(b)


def _stress(protect):
    """the in-flight giant (half of its chunks committed) keeps its records while a finished-prompt stream fills the small halo store"""
    idx = mk(
        halo_a=24,
        halo_b=8,
        halo_budget=16,
        halo_spacing=16384,
        protect_fn=lambda: protect,
        pages_per_row=12000,
        key_slots=6000,
    )
    giant = toks(10, 300_000)
    feed(idx, "giant", giant, 0, upto=150_016, total=300_000)  # in flight
    mine = lambda: [b for b in idx._rows[0].a_lru.values() if b.infl or True]
    g0 = sorted(b.idx for b in idx._rows[0].a_lru.values())
    for k in range(6):  # six finished 120 k prompts churn the store (each wants ~7 records)
        feed(idx, f"f{k}", toks(20 + k, 120_000), 1 + (k % 3))
        idx.release(f"f{k}")
    g1 = sorted(
        b.idx
        for b in idx._rows[0].a_lru.values()
        if b.idx * C < 150_016 and b.parent is not None or b.idx * C < 150_016
    )
    kept = [b for b in idx._rows[0].a_lru.values() if b.idx * C <= 150_016]
    # the giant's own records: blocks of the giant chain
    chain = AP.Chain(C).extend(giant, 150_016).digests
    own = [blk for d in chain if (blk := idx._rows[0].blocks.get(d)) is not None and blk.halo_a >= 0]
    return len(g0), len(own)


def test_in_flight_records_survive_a_full_store_only_when_protected():
    n0, own_off = _stress(False)
    n1, own_on = _stress(True)
    assert n0 == n1 and n0 >= 8
    assert own_off < own_on  # the old LRU lets finished prompts wipe the giant's records
    assert own_on >= n0 - 1  # the protected records (all but possibly the oldest) are still there
