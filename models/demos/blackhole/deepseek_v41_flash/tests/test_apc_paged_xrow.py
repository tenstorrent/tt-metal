# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the cross-row hit of ``tt/apc_paged.py`` (``BlockIndex(xrow=True)``, DSV41_APC_PAGED_XROW=1): the donor row has no free user, the receiving row gets fresh pages, the model copies the
blocks (here: only the index calls), the request then continues as a LOCAL hit. python_env/bin/python -m pytest --noconftest tests/test_apc_paged_xrow.py -q"""
import importlib.util
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AP = _load("apc_paged_xrow_under_test", "../tt/apc_paged.py")
C = 512


def toks(seed, n):
    return np.random.default_rng(seed).integers(1, 100000, size=n, dtype=np.int64)


def mk(**kw):
    d = dict(
        rows=2,
        users_per_row=1,
        pages_per_row=200,
        halo_a=16,
        halo_b=8,
        key_slots=64,
        zone=1 << 40,
        xrow=True,
        xrow_min=1024,
    )
    d.update(kw)
    return AP.BlockIndex(**d)


def donor(idx, rid, t, user):
    """a cold request that commits every chunk on ``user``'s row and finishes"""
    e = len(t)
    idx.bind(rid, 0, user=user, prompt_len=e)
    idx.page_table(rid, e)
    for st in range(0, e // C * C, C):
        cp = idx.begin_chunk(rid, st, st + C, t, e)
        if cp is not None:
            idx.finish_chunk(cp)
    idx.release(rid)


def copy_in(idx, rid, t, e, plan):
    """what the model's executor does at the index level: one copy chunk per donor block, the tip with its halos; -> the plan of the receiving row"""
    src = idx.xrow_source(rid)
    assert src is not None and len(src) == plan.hit_tokens // C
    cps = []
    for i, (bid, pages, ks, ha, hb, bz) in enumerate(src):
        last = i == len(src) - 1
        cp = idx.begin_chunk(rid, i * C, (i + 1) * C, t, e, copy=(last and ha >= 0, last and hb >= 0, bz))
        if cp is not None:
            cps.append(cp)
    for cp in cps:
        assert idx.finish_chunk(cp)
    return idx.xrow_finish(rid)


def test_cross_row_plan_copy_and_local_hit_afterwards():
    idx = mk()
    t = toks(1, 5000)
    donor(idx, "d", t, user=0)  # row 0 holds 9 blocks (hit lookups need e - 128 >= ...)
    idx.bind("occupant", 0, user=0)  # row 0's only user is busy now
    idx.check()
    t2 = np.concatenate([t[:4096], toks(2, 3000)])
    plan = idx.plan_hit("new", t2, len(t2))
    assert plan.xrow_src == 0 and plan.row == 1 and plan.user == 1 and plan.src_row == 0
    assert plan.hit_tokens == 4096 and len(plan.page_ids) == 4096 // 128 and len(plan.block_ids) == 8
    idx.check()
    b = idx.bind("new", plan.hit_tokens)
    assert b is not None and b.xrow_src == 0
    new = copy_in(idx, "new", t2, len(t2), b)
    assert new is not None and new.xrow_src == -1 and new.row == 1 and new.hit_tokens == 4096
    assert new.halo_a >= 0 and len(new.key_slots) == 8 and new.page_ids == plan.page_ids
    idx.check()
    assert idx.counters["xrow_plans"] == 1 and idx.counters["xrow_done"] == 1
    idx.restored("new")
    idx.page_table("new", len(t2))
    idx.release("new")
    idx.release("occupant")
    idx.check()
    # the receiving row now holds the blocks: a third request that finds user 1 busy hits LOCALLY on row 1... and row 0 is free again
    t3 = np.concatenate([t[:4096], toks(3, 2000)])
    p3 = idx.plan_hit("third", t3, len(t3))
    assert p3.hit_tokens == 4096 and p3.xrow_src == -1
    idx.cancel("third")
    idx.check()


def test_off_by_default_and_small_gains_stay_local():
    idx = mk(xrow=False)
    t = toks(4, 5000)
    donor(idx, "d", t, user=0)
    idx.bind("occupant", 0, user=0)
    p = idx.plan_hit("new", t, len(t))
    assert p.hit_tokens == 0 and p.xrow_src == -1  # (the old behaviour: busy loss, recompute)
    idx.cancel("new")
    idx2 = mk(xrow_min=1 << 30)
    donor(idx2, "d", t, user=0)
    idx2.bind("occupant", 0, user=0)
    assert idx2.plan_hit("new", t, len(t)).xrow_src == -1
    idx2.check()


def test_reservation_expiry_and_abort_return_the_receiving_pages():
    now = [0.0]
    idx = mk(clock=lambda: now[0], ttl_s=5.0)
    t = toks(5, 5000)
    donor(idx, "d", t, user=0)
    idx.bind("occupant", 0, user=0)
    free0 = idx.stats()["rows"][1]["pages_free"]
    p = idx.plan_hit("new", t, len(t))
    assert p.xrow_src == 0
    assert idx.stats()["rows"][1]["pages_free"] == free0 - len(p.page_ids)
    idx.check()
    now[0] = 100.0  # the reservation expires: private pages and the donor pins come back
    idx.plan_hit("probe", toks(9, 600), 600)
    assert idx.stats()["rows"][1]["pages_free"] == free0
    idx.check()
    # an abort after the bind (the copy never ran)
    now[0] = 0.0
    p = idx.plan_hit("new2", t, len(t))
    idx.bind("new2", p.hit_tokens)
    idx.release("new2")
    assert idx.stats()["rows"][1]["pages_free"] == free0
    idx.release("occupant")
    idx.check()


def test_a_failed_copy_is_reported_and_leaves_a_consistent_index():
    idx = mk(key_slots=6, users_per_row=2)
    t = toks(6, 2700)
    donor(idx, "d", t, user=0)  # row 0 holds 5 blocks
    idx.bind("o0", 0, user=0)
    idx.bind("o1", 0, user=1)  # row 0 has no free user
    # row 1: a request keeps all 6 key slots pinned (its blocks are held, nothing is evictable)
    h = toks(7, 3300)
    idx.bind("hold", 0, user=2, prompt_len=len(h))
    idx.page_table("hold", len(h))
    for st in range(0, 3072, C):
        cp = idx.begin_chunk("hold", st, st + C, h, len(h))
        idx.finish_chunk(cp)
    p = idx.plan_hit("new", t, len(t))
    assert p.xrow_src == 0 and p.row == 1 and p.user == 3
    idx.bind("new", p.hit_tokens)
    assert copy_in(idx, "new", t, len(t), p) is None
    assert idx.counters["xrow_failed"] == 1
    idx.release("new")  # the caller drops the failed request
    idx.release("hold")
    idx.release("o0")
    idx.release("o1")
    idx.check()


def test_receiving_row_has_the_block_without_the_donors_tip_halo():
    """the shared-system-prompt case seen on the server (3 of 111 copies failed with 'a receiving block is missing', all at the same 5632 token hit): the receiving row already holds the
    blocks of the hit (published by an earlier request that ended elsewhere, so the block at the hit end has NO halo there) while the donor's tip carries one. The copy must add the halo.
    """
    idx = mk(users_per_row=2, halo_thin=4)
    t = toks(11, 5700)
    # row 1 (users 2, 3): an earlier request over the same first 4608 tokens but a longer, different tail -> block 9 exists on row 1 without a halo (not its tip)
    t_other = np.concatenate([t[:5632], toks(12, 3000)])
    donor(idx, "early", t_other, user=2)
    # row 0 (users 0, 1): the donor over t: its tip (block 9, 4608 tokens) has the halos
    donor(idx, "d", t, user=0)
    idx.bind("o0", 0, user=0)
    idx.bind("o1", 0, user=1)  # row 0 has no free user
    idx.bind("o2", 0, user=2)  # one user of row 1 busy, one free
    t2 = np.concatenate([t[:5632], toks(13, 2000)])
    p = idx.plan_hit("new", t2, len(t2))
    assert p.xrow_src == 0 and p.row == 1 and p.hit_tokens == 5632
    idx.bind("new", p.hit_tokens)
    new = copy_in(idx, "new", t2, len(t2), p)
    assert new is not None and new.hit_tokens == 5632 and new.halo_a >= 0
    assert idx.counters["xrow_failed"] == 0
    idx.release("new")
    for r in ("o0", "o1", "o2"):
        idx.release(r)
    idx.check()


def test_live_switch_file_turns_cross_row_off_and_on(tmp_path, monkeypatch):
    f = tmp_path / "xrow"
    monkeypatch.setenv("DSV41_APC_PAGED_XROW_FILE", str(f))
    idx = mk()
    t = toks(21, 5000)
    donor(idx, "d", t, user=0)
    idx.bind("occupant", 0, user=0)
    f.write_text("0")
    p = idx.plan_hit("a", t, len(t))
    assert p.xrow_src == -1 and p.hit_tokens == 0  # off: the old behaviour
    idx.cancel("a")
    f.write_text("1")
    p = idx.plan_hit("b", t, len(t))
    assert p.xrow_src == 0
    idx.cancel("b")
    f.unlink()  # a missing file leaves the flag as set
    p = idx.plan_hit("c", t, len(t))
    assert p.xrow_src == 0
    idx.cancel("c")
    idx.release("occupant")
    idx.check()
