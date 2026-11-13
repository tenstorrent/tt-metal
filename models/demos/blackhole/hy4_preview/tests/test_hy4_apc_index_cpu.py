# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device, no ttnn/torch) of ``tt/apc_blocks.py::Hy4BlockIndex``: the DSV4 BlockIndex on the Hy4 geometry (page = chunk = 512 tokens, no halo store, margin 1, 1M cross-row copies).
Adapted from the DeepSeek-V4.1 tests test_apc_paged_index.py / _xrow.py / test_apc_peek_home_cpu.py / test_apc_loss_log_cpu.py (halo/zone tests replaced by the Hy4 'no tail state' tests).
Run (worktree root):  python3 -m pytest --noconftest -c /dev/null --rootdir=/tmp models/demos/blackhole/hy4_preview/tests/test_hy4_apc_index_cpu.py -q"""
import importlib.util
import os
import queue
import random
import threading

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AB = _load("hy4_apc_blocks_under_test", "../tt/apc_blocks.py")
C = 512
IDX = AB.Hy4BlockIndex


def toks(seed, n):
    return np.random.default_rng(seed).integers(1, 120000, size=n, dtype=np.int64)


def mk(**kw):
    d = dict(rows=2, users_per_row=2, pages_per_row=400)
    d.update(kw)
    return IDX(**d)


def run(idx, rid, t, user=None, release=True, max_hit=None):
    """one request the way the model drives it: plan -> bind -> pages -> chunks. -> (hit, binding)"""
    e = len(t)
    plan = idx.plan_hit(rid, t, e, max_hit=max_hit)
    if plan.hit_tokens:
        b = idx.bind(rid, plan.hit_tokens)
        assert b is not None
    else:
        if user is None:
            user = next(u for r in range(idx.rows) for u in idx.free_users(r))
        b = idx.bind(rid, 0, user=user)
    idx.restored(rid)
    idx.page_table(rid, e)
    for st in range(plan.hit_tokens, e // C * C, C):
        cp = idx.begin_chunk(rid, st, st + C, t, e)
        if cp is not None:
            idx.finish_chunk(cp)
    if release:
        idx.release(rid)
    return plan.hit_tokens, b


# ---- Hy4 parametrization ------------------------------------------------------------------------------------------------------------------------------
def test_hy4_constants_and_forbidden_overrides():
    idx = mk(pages_per_row=2147)
    assert (idx.chunk, idx.page, idx.ppc, idx.margin, idx.zone) == (512, 512, 1, 1, 0)
    assert idx.caps == dict(
        pages=2147, key=2147, a=2147, b=0
    )  # one key slot and one hit flag per page: never scarcer than pages
    assert idx.xrow_max == 1 << 20
    for bad in ("halo_a", "halo_b", "zone", "margin", "page", "chunk", "key_slots", "halo_thin"):
        with pytest.raises(TypeError):  # allow-pytest.raises: CPU-only module
            IDX(**{bad: 1})  # allow-pytest.raises: CPU-only module


def test_every_block_is_a_hit_point_and_there_is_no_tail_state():
    idx = mk(rows=1, pages_per_row=100)
    t = toks(1, 10 * C + 5)
    run(idx, "a", t, user=0)
    row = idx._rows[0]
    blocks = sorted(row.blocks.values(), key=lambda b: b.idx)
    assert [b.idx for b in blocks] == list(range(1, 11))
    assert all(b.halo_a >= 0 and b.halo_b == -1 and len(b.pages) == 1 for b in blocks)
    longer = np.concatenate([t[: 7 * C], toks(2, 3 * C)])
    for k in range(1, 8):  # every grid boundary up to the divergence is a hit point (max_hit forces the cut)
        assert idx.peek(longer, len(longer), max_hit=k * C) == (k * C, 0)
    idx.check()


# ---- hashing -------------------------------------------------------------------------------------------------------------------------------------------
def test_chain_prefix_incremental_salt_and_image_keys():
    a, b = toks(2, 3000), toks(3, 3000)
    mix = np.concatenate([a[:1536], b[1536:]])
    da, dm = AB.chain_digests(a), AB.chain_digests(mix)
    assert len(da) == 5 and da[:3] == dm[:3] and da[3] != dm[3] and da[4] != dm[4]
    ch = AB.Chain()
    for top in (600, 2000, 3000):
        ch.extend(a, top)
    assert ch.digests == da
    assert AB.chain_digests(a, salt=b"x")[0] != da[0] and AB.chain_digests(a, salt=b"x") == AB.chain_digests(
        a, salt=b"x"
    )
    img1, img2 = a.copy(), a.copy()
    img1[100:400], img2[100:400] = -12345, -54321
    d1, d2 = AB.chain_digests(img1), AB.chain_digests(img2)
    assert d1[0] != da[0] and d1[0] != d2[0] and AB.chain_digests(a.astype(np.int32)) == da


# ---- hit rule ------------------------------------------------------------------------------------------------------------------------------------------
def test_hit_rule_margin_one_and_grid():
    idx = mk()
    t = toks(10, 4096)
    assert run(idx, "a", t)[0] == 0
    # same prompt again: deepest boundary <= e - 1 = 4095 -> 3584 (the last chunk must be computed: its logits are the first token)
    assert idx.peek(t, 4096) == (3584, 0)
    longer = np.concatenate([t, toks(11, 300)])
    assert idx.peek(longer, len(longer)) == (4096, 0)
    assert idx.peek(t[:512], 512) == (0, -1)  # never a hit at s == 0 / the whole prompt
    assert idx.peek(t[:513], 513)[0] == 512  # one token left to compute is enough (DSV4 needed 128)
    assert idx.peek(t[:1000], 1000)[0] == 512 and idx.peek(t[:1025], 1025)[0] == 1024


def test_max_hit_bounds_the_answer():
    idx = mk()
    t = toks(20, 4096)
    run(idx, "a", t)
    longer = np.concatenate([t, toks(21, 500)])
    assert idx.plan_hit("b", longer, len(longer), max_hit=2000).hit_tokens == 1536
    assert idx.plan_hit("c", longer, len(longer), max_hit=0).hit_tokens == 0


def test_strict_spans_block_boundaries():
    idx = mk(strict_spans=True)
    t = toks(22, 3000)
    spans = [(700, 600, "h")]  # boundary 1024 inside (700, 1300)
    e = len(t)
    idx.bind("a", 0, user=0)
    idx.page_table("a", e)
    ends = []
    for st in range(0, e // C * C, C):
        cp = idx.begin_chunk("a", st, st + C, t, e, spans=spans)
        ends.append(None if cp is None else cp.block_idx)
        if cp:
            idx.finish_chunk(cp)
    assert ends == [1, None, None, None, None]
    idx.release("a")


# ---- hit binding ---------------------------------------------------------------------------------------------------------------------------------------
def test_hit_goes_to_reserved_user_on_owning_row_and_page_ids_are_blocks():
    idx = mk()
    t = toks(30, 3000)
    run(idx, "a", t, user=3)  # row 1
    p = idx.plan_hit("b", t, 3000)
    assert p.hit_tokens == 2560 and p.row == 1 and p.user in (2, 3) and p.user in idx._reserved
    b = idx.bind("b", 2560)
    assert b is p and idx._busy[p.user] == "b"
    assert len(p.page_ids) == 5 == len(p.block_ids) == len(p.key_slots)  # page = block
    idx.release("b")
    assert p.user in idx.free_users(1)


def test_replan_is_idempotent_and_cancel_unpins():
    idx = mk()
    t = toks(34, 3000)
    run(idx, "a", t)
    p1, p2 = idx.plan_hit("b", t, 3000), idx.plan_hit("b", t, 3000)
    assert p1.hit_tokens == p2.hit_tokens and len(idx._reserved) == 1
    idx.check()
    idx.cancel("b")
    assert not idx._reserved and all(b.refs == 0 for r in idx._rows for b in r.lru.values())
    idx.check()


def test_continuation_never_takes_a_hit_and_unknown_start_is_none():
    idx = mk()
    t = toks(35, 3000)
    run(idx, "a", t)
    p = idx.plan_hit("b", t, 3000)
    assert idx.bind("b", p.hit_tokens) is p
    assert idx.bind("b", 4096) is None
    assert idx.bind("zzz", 512, user=0) is None and idx.bind("zzz", 0) is None


def test_lost_reservation_reset_ttl_and_start_mismatch():
    now = [0.0]
    idx = mk(ttl_s=10.0, clock=lambda: now[0])
    t = toks(36, 3000)
    run(idx, "a", t)
    p = idx.plan_hit("b", t, 3000)
    idx.reset()
    assert idx.bind("b", p.hit_tokens) is None and idx.stats()["lost"] == 1 and not idx._reserved
    assert idx.bind("b", 0, user=0).hit_tokens == 0
    idx.release("b")
    run(idx, "a2", t)
    p = idx.plan_hit("c", t, 3000)
    now[0] = 11.0
    assert idx.bind("c", p.hit_tokens) is None and idx.stats()["lost"] == 2 and idx.stats()["expired"] == 1
    p = idx.plan_hit("d", t, 3000)
    assert idx.bind("d", 1024) is None
    idx.plan_hit("e", t, 3000)
    assert idx.bind("e", 0, user=1).hit_tokens == 0
    idx.check()


def test_no_free_user_on_owning_row_falls_back_to_miss_and_free_users_provider():
    idx = mk(users_per_row=1)
    t = toks(37, 3000)
    run(idx, "a", t, user=0)
    idx.bind("busy", 0, user=0)
    assert idx.plan_hit("b", t, 3000).hit_tokens == 0
    idx.release("busy")
    p = idx.plan_hit("b", t, 3000)
    assert (p.hit_tokens, p.row, p.user) == (2560, 0, 0)
    assert idx.plan_hit("c", t, 3000).hit_tokens == 0  # the reserved user is not available twice
    free = {0: [], 1: [3]}
    idx2 = mk(free_users=lambda r: free[r])
    run(idx2, "a", t, user=3)
    assert idx2.plan_hit("b", t, 3000).user == 3
    free[1] = []
    assert idx2.plan_hit("c", t, 3000).hit_tokens == 0


def test_two_requests_get_distinct_users_and_deepest_row_wins():
    free = {0: [0, 1], 1: [2, 3]}
    idx = mk(free_users=lambda r: free[r])
    t = toks(38, 3000)
    run(idx, "a", t, user=0)
    longer = np.concatenate([t, toks(39, 3000)])
    free[0] = []
    run(idx, "b", longer, user=2)
    free[0] = [0, 1]
    p = idx.plan_hit("c", longer, len(longer))
    assert p.row == 1 and p.hit_tokens > 2560
    q = idx.plan_hit("d", longer, len(longer))
    assert q.row == 1 and q.user != p.user
    r = idx.plan_hit("e", longer, len(longer))
    assert r.row == 0 and r.hit_tokens == 2560


def test_finished_requests_free_their_users_for_the_next_reservation():
    idx = mk(rows=1, users_per_row=2)
    t = toks(41, 2000)
    idx.bind("A", 0, user=0)
    idx.page_table("A", 2000)
    for st in range(0, 1536, C):
        idx.finish_chunk(idx.begin_chunk("A", st, st + C, t, 2000))
    idx.bind("B", 0, user=1)
    idx.page_table("B", 700)
    t2 = np.concatenate([t[:1536], toks(42, 600)])
    assert idx.plan_hit("N", t2, len(t2)).hit_tokens == 0
    plan = idx.plan_hit("N", t2, len(t2), finished_ids=["A"])
    assert plan.hit_tokens == 1536 and plan.user == 0
    idx.release("A")
    assert idx.bind("N", 1536, user=0) is not None
    idx.check()


# ---- refcount / eviction -------------------------------------------------------------------------------------------------------------------------------
def test_pages_shared_and_freed_only_at_last_release():
    idx = mk(pages_per_row=100)
    t = toks(50, 3000)
    run(idx, "a", t, user=0, release=False)
    idx.check()
    pa = idx.page_table("a", 3000)
    plan = idx.plan_hit("b", t, 3000)
    assert plan.hit_tokens == 2560 and plan.row == 0
    idx.bind("b", 2560)
    idx.restored("b")
    pb = idx.page_table("b", 3000)
    assert pb[:5] == pa[:5] and not set(pb[5:]) & set(pa)  # shared pages, private tail
    row = idx._rows[0]
    assert all(b.refs == 2 for b in row.lru.values() if b.idx <= 5)
    idx.release("a")
    idx.check()
    assert all(b.refs == 1 for b in row.lru.values() if b.idx <= 5)
    free_before = len(row.free_pages)
    idx.release("b")
    assert len(row.free_pages) == free_before + 1  # b's private tail page only; the cached blocks stay
    assert all(b.refs == 0 for b in row.lru.values())
    idx.check()


def test_never_double_books_pages_under_pressure():
    idx = mk(rows=1, pages_per_row=12)
    rng = random.Random(0)
    heads = [toks(60 + i, 1536) for i in range(4)]
    for k in range(60):
        t = np.concatenate(
            [heads[rng.randrange(4)][: rng.choice([512, 1024, 1536])], toks(1000 + k, rng.choice([300, 700, 1300]))]
        )
        try:
            run(idx, k, t, user=rng.randrange(2))
        except AB.OutOfPages:
            idx.release(k)
        idx.check()


def test_lru_leaf_first_eviction_order():
    idx = mk(rows=1, pages_per_row=8)
    a, b, c = toks(70, 1100), toks(71, 1100), toks(72, 1100)  # 2 blocks each
    for rid, x in (("a", a), ("b", b), ("c", c)):
        run(idx, rid, x, user=0)
    row = idx._rows[0]
    assert len(row.blocks) == 6 and len(row.free_pages) == 2
    ev = []
    idx.add_listener(lambda e: ev.append(e) if e["kind"] == "block" else None)
    assert idx.plan_hit("a2", np.concatenate([a, toks(73, 600)]), 1700).hit_tokens == 1024  # touch a: newest
    idx.cancel("a2")
    run(idx, "d", toks(74, 1700), user=0)  # needs 4 pages, 2 free: evicts 2 blocks, the oldest leaves first (b's chain)
    assert sum(len(e["ids"]) for e in ev) == 2
    keys = {blk.key for blk in row.blocks.values()}
    da, db = AB.chain_digests(a)[:2], AB.chain_digests(b)[:2]
    assert set(da) <= keys and not (set(db) <= keys)
    if db[0] not in keys:
        assert db[1] not in keys
    idx.check()


def test_pinned_blocks_are_never_evicted_and_dead_blocks_free_at_release():
    idx = mk(rows=1, pages_per_row=6)
    a = toks(80, 1100)
    run(idx, "a", a, user=0)
    p = idx.plan_hit("p", np.concatenate([a, toks(81, 300)]), 1400)
    assert p.hit_tokens == 1024
    run(idx, "x", toks(82, 1100), user=1)
    row = idx._rows[0]
    assert {b.id for b in row.lru.values()} >= set(p.block_ids)
    idx.check()
    idx.bind("p", 1024)
    idx.reset()
    assert not row.lru and p.page_ids[0] not in row.free_pages
    idx.check()
    idx.release("p")
    assert all(pg in row.free_pages for pg in p.page_ids)
    idx.check()


def test_out_of_pages_rollback_and_eviction_of_cached_blocks():
    idx = mk(rows=1, pages_per_row=10)
    idx.bind("a", 0, user=0)
    with pytest.raises(AB.OutOfPages):  # allow-pytest.raises: CPU-only module
        idx.page_table("a", 11 * C)
    assert idx.stats()["out_of_pages"] == 1
    idx.page_table("a", 10 * C)
    idx.rollback("a", 3 * C)
    assert len(idx._rows[0].free_pages) == 7
    idx.release("a")
    assert len(idx._rows[0].free_pages) == 10
    run(idx, "b", toks(90, 4 * C + 10), user=0)
    run(idx, "c", toks(91, 4 * C + 10), user=0)  # 8 of 10 pages cached
    idx.bind("d", 0, user=0)
    idx.page_table("d", 9 * C)
    assert idx.stats()["ev_block"] >= 2
    idx.check()


def test_listener_events_reset_and_signature():
    idx = mk(rows=1)
    ev = []
    idx.add_listener(ev.append)
    run(idx, "a", toks(110, 4 * C + 5), user=0)
    idx.reset()
    assert ev[-1]["kind"] == "block" and ev[-1]["why"] == "reset" and len(ev[-1]["ids"]) == 4
    idx.add_listener(lambda e: 1 / 0)
    idx.reset()
    assert idx.stats()["resets"] == 2
    t = toks(111, 2000)
    run(idx, "b", t, user=0)
    idx.set_signature("same")  # first non-empty signature: resets
    run(idx, "c", t, user=0)
    idx.set_signature("same")
    assert idx.peek(t, 2000)[0] > 0
    idx.set_signature("other")
    assert idx.peek(t, 2000)[0] == 0


# ---- commit protocol -----------------------------------------------------------------------------------------------------------------------------------
def test_two_phase_commit_invisible_until_finished_and_abort_returns_slots():
    idx = mk(rows=1)
    t = toks(120, 2 * C + 200)
    e = len(t)
    idx.bind("a", 0, user=0)
    idx.page_table("a", e)
    cp1 = idx.begin_chunk("a", 0, C, t, e)
    cp2 = idx.begin_chunk("a", C, 2 * C, t, e)
    assert cp1.key_slot >= 0 and cp1.halo_a >= 0 and cp1.halo_b == -1 and cp1.block_idx == 1
    assert idx.peek(t, e) == (0, -1)
    assert idx.begin_chunk("a", 2 * C, e, t, e) is None  # the final partial chunk is not a boundary
    assert idx.finish_chunk(cp1) and idx.peek(t, e) == (512, 0)
    idx.abort_chunk(cp2)
    st = idx.stats()["rows"][0]
    assert st["key_free"] == idx.caps["key"] - 1 and st["halo_a_free"] == idx.caps["a"] - 1
    idx.check()
    idx.release("a")
    idx.check()


def test_dup_commit_keeps_pages_private():
    idx = mk(rows=1)
    t = toks(121, 2048 + 52)
    run(idx, "a", t, user=0)
    idx.bind("b", 0, user=1)
    idx.page_table("b", len(t))
    cps = [idx.begin_chunk("b", s, s + C, t, len(t)) for s in range(0, 4 * C, C)]
    assert all(cp is None for cp in cps)  # the blocks exist: nothing to add, pages stay private to b
    assert all(p in idx._reqs["b"].priv for p in idx.page_table("b", len(t))[:4])
    idx.release("b")
    idx.check()


def test_concurrent_identical_prefills_one_block_wins():
    idx = mk(rows=1, users_per_row=2)
    t = toks(124, 1100)
    for r, u in (("a", 0), ("b", 1)):
        idx.bind(r, 0, user=u)
        idx.page_table(r, 1100)
    ca = idx.begin_chunk("a", 0, C, t, 1100)
    cb = idx.begin_chunk("b", 0, C, t, 1100)
    assert ca is not None and cb is None
    assert idx.finish_chunk(ca)
    idx.release("a")
    idx.release("b")
    idx.check()


def test_commit_dropped_when_cache_reset_and_release_with_pending_chunks():
    idx = mk(rows=1)
    t = toks(125, 1600)
    idx.bind("a", 0, user=0)
    idx.page_table("a", 1600)
    c1 = idx.begin_chunk("a", 0, C, t, 1600)
    c2 = idx.begin_chunk("a", C, 2 * C, t, 1600)
    idx.reset()
    assert not idx.finish_chunk(c1) and not idx.finish_chunk(c2)
    idx.check()
    assert idx.stats()["rows"][0]["key_free"] == idx.caps["key"]
    idx.release("a")
    assert len(idx._rows[0].free_pages) == idx.caps["pages"]
    idx.bind("b", 0, user=0)
    idx.page_table("b", 1600)
    idx.begin_chunk("b", 0, C, t, 1600)
    idx.release("b")
    s = idx.stats()["rows"][0]
    assert (
        s["pending"] == 0
        and s["key_free"] == idx.caps["key"]
        and s["pages_free"] == idx.caps["pages"]
        and s["halo_a_free"] == idx.caps["a"]
    )
    idx.check()


def test_release_user_by_physical_id():
    idx = mk()
    idx.bind("a", 0, user=3)
    idx.release_user(3)
    assert "a" not in idx._reqs and 3 in idx.free_users(1)
    idx.release_user(3)


def test_hook_plan_peek_cancel_and_key_tokens_fn():
    idx = mk()
    t = toks(140, 3000)
    run(idx, "a", t)
    hook = AB.AP.ApcPagedHook(idx)
    assert hook.plan_hit("h", t, 3000) == (2560, idx.peek(t, 3000)[1])
    hook.cancel("h")
    assert not idx._reserved
    assert hook.peek_home("X", t, 3000) == idx.peek_home(t, 3000)
    calls = []

    def fn(tokens, mm):
        calls.append(mm)
        return None if mm == "image" else tokens

    hook2 = AB.AP.ApcPagedHook(idx, fn)
    assert hook2.plan_hit("i", t, 3000, mm="image") == (0, -1)
    assert hook2.plan_hit("j", t, 3000, mm=None)[0] == 2560 and calls == ["image", None]


# ---- peek_home / loss log ------------------------------------------------------------------------------------------------------------------------------
def test_peek_home_is_read_only_and_reports_the_busy_home_row():
    idx = mk()
    a = toks(1, 6 * C + 300)
    run(idx, "A", a, user=0)
    idx.bind("B", 0, user=0, prompt_len=1000)
    idx.bind("C", 0, user=1, prompt_len=1000)
    a2 = list(a) + list(toks(2, 4 * C))
    before = (dict(idx.counters), sorted(idx._reqs), dict(idx._reserved), dict(idx._busy), len(idx._rows[0].free_pages))
    hit, home, free = idx.peek_home(a2, len(a2))
    assert (hit > 0, home, free) == (True, 0, [0, 2])
    assert before == (
        dict(idx.counters),
        sorted(idx._reqs),
        dict(idx._reserved),
        dict(idx._busy),
        len(idx._rows[0].free_pages),
    )
    assert idx.plan_hit("A2", a2, len(a2)).hit_tokens == 0 and idx.counters["busy_loss_tokens"] == hit


def test_loss_log_one_line_per_busy_row_loss_and_counters_unchanged():
    def scenario(idx):
        a = toks(1, 6 * C + 300)
        run(idx, "A", a, user=0)
        idx.bind("B", 0, user=0, prompt_len=1000)
        idx.bind("C", 0, user=1, prompt_len=1000)
        a2 = list(a) + list(toks(2, 4 * C))
        return idx.plan_hit("A2", a2, len(a2))

    lines = []
    i1, i0 = mk(loss_log=lines.append), mk()
    assert scenario(i1).hit_tokens == scenario(i0).hit_tokens == 0
    assert i1.counters == i0.counters
    busy = [x for x in lines if "reason=busy_row" in x]
    assert len(busy) == 1 and "lost_halo=0" in busy[0] and "lost_busy=3072" in busy[0]
    assert all("req=A2" in x for x in busy) and sum("reason=no_lookup" in x for x in lines) == 2


# ---- cross-row copy (rows-mode d2d, DSV4 test_apc_paged_xrow.py) -----------------------------------------------------------------------------------------
def donor(idx, rid, t, user):
    e = len(t)
    idx.bind(rid, 0, user=user, prompt_len=e)
    idx.page_table(rid, e)
    for st in range(0, e // C * C, C):
        cp = idx.begin_chunk(rid, st, st + C, t, e)
        if cp is not None:
            idx.finish_chunk(cp)
    idx.release(rid)


def copy_in(idx, rid, t, e, plan):
    """what the model's d2d executor does at index level: one copy chunk per donor block"""
    src = idx.xrow_source(rid)
    assert src is not None and len(src) == plan.hit_tokens // C
    cps = []
    for i, (bid, pages, ks, ha, hb, bz) in enumerate(src):
        cp = idx.begin_chunk(
            rid, i * C, (i + 1) * C, t, e, copy=(False, False, 0)
        )  # the Hy4 index forces (True, False, 0): every copied block is a hit point
        if cp is not None:
            cps.append(cp)
    for cp in cps:
        assert idx.finish_chunk(cp)
    return idx.xrow_finish(rid)


def test_cross_row_plan_copy_and_local_hit_afterwards_beyond_dsv4_cap():
    idx = mk(users_per_row=1, pages_per_row=200, xrow=True, xrow_min=1024)
    t = toks(1, 100 * C + 17)  # 51k tokens (a 100-block donor)
    donor(idx, "d", t, user=0)
    idx.bind("occupant", 0, user=0)
    t2 = np.concatenate([t[: 99 * C], toks(2, 3000)])
    plan = idx.plan_hit("new", t2, len(t2))
    assert plan.xrow_src == 0 and plan.row == 1 and plan.user == 1 and plan.hit_tokens == 99 * C
    assert len(plan.page_ids) == 99 and len(plan.block_ids) == 99
    idx.check()
    b = idx.bind("new", plan.hit_tokens)
    assert b.xrow_src == 0
    new = copy_in(idx, "new", t2, len(t2), b)
    assert (
        new is not None
        and new.xrow_src == -1
        and new.row == 1
        and new.hit_tokens == 99 * C
        and len(new.key_slots) == 99
    )
    idx.check()
    assert idx.counters["xrow_plans"] == 1 and idx.counters["xrow_done"] == 1
    idx.restored("new")
    idx.page_table("new", len(t2))
    idx.release("new")
    idx.release("occupant")
    idx.check()
    p3 = idx.plan_hit("third", np.concatenate([t[: 99 * C], toks(3, 2000)]), 99 * C + 2000)
    assert p3.hit_tokens == 99 * C and p3.xrow_src == -1


def test_cross_row_copy_cap_is_the_context_not_65536():
    assert (
        mk().xrow_max == 1 << 20 and IDX(xrow_max=65536).xrow_max == 65536
    )  # (only an explicit override restores the DSV4 cap)


def test_cross_row_failed_copy_and_expiry_leave_consistent_index():
    now = [0.0]
    idx = mk(users_per_row=2, pages_per_row=12, xrow=True, xrow_min=1024, ttl_s=5.0, clock=lambda: now[0])
    t = toks(6, 5 * C + 9)
    donor(idx, "d", t, user=0)
    idx.bind("o0", 0, user=0)
    idx.bind("o1", 0, user=1)
    p = idx.plan_hit("new", t, len(t))
    assert p.xrow_src == 0 and p.row == 1
    now[0] = 6.0  # reservation expires: the receiving pages come back
    assert idx.bind("new", p.hit_tokens) is None
    idx.check()
    assert len(idx._rows[1].free_pages) == 12
    # receiving row full of held pages: no copy plan (nopages), plan degrades to a miss
    idx.bind("hold", 0, user=2, prompt_len=11 * C)
    idx.page_table("hold", 11 * C)
    p2 = idx.plan_hit("new2", t, len(t))
    assert p2.hit_tokens == 0 and idx.counters["xrow_nopages"] >= 1
    idx.release("hold")
    idx.release("o0")
    idx.release("o1")
    idx.check()


def test_receiving_row_already_holds_some_blocks():
    idx = mk(users_per_row=2, pages_per_row=100, xrow=True, xrow_min=1024)
    t = toks(11, 12 * C + 9)
    donor(
        idx, "early", np.concatenate([t[: 5 * C], toks(12, 3000)]), user=2
    )  # row 1 holds blocks 1..5 of the same head
    donor(idx, "d", t, user=0)
    for r, u in (("o0", 0), ("o1", 1), ("o2", 2)):
        idx.bind(r, 0, user=u)
    t2 = np.concatenate([t[: 11 * C], toks(13, 2000)])
    p = idx.plan_hit("new", t2, len(t2))
    assert p.xrow_src == 0 and p.row == 1 and p.hit_tokens == 11 * C
    idx.bind("new", p.hit_tokens)
    new = copy_in(idx, "new", t2, len(t2), p)
    assert new is not None and new.hit_tokens == 11 * C and idx.counters["xrow_failed"] == 0
    idx.check()


def test_cross_row_live_switch_file(tmp_path, monkeypatch):
    f = tmp_path / "xrow"
    monkeypatch.setenv("DSV41_APC_PAGED_XROW_FILE", str(f))
    idx = mk(users_per_row=1, pages_per_row=100, xrow=True, xrow_min=1024)
    t = toks(21, 10 * C + 5)
    donor(idx, "d", t, user=0)
    idx.bind("o", 0, user=0)
    f.write_text("0")
    assert idx.plan_hit("a", t, len(t)).xrow_src == -1
    f.write_text("1")
    assert idx.plan_hit("b", t, len(t)).xrow_src == 0


# ---- property / stress ---------------------------------------------------------------------------------------------------------------------------------
class Oracle:
    """independent model of the published state (no eviction): digest sets per row; hit = deepest boundary s <= e - 1 on the grid with the whole chain present"""

    def __init__(self, rows):
        self.rows = [set() for _ in range(rows)]

    def commit(self, row, t, e):
        self.rows[row].update(AB.chain_digests(t, top=e // C * C))

    def hit(self, row, t, e, max_hit=None):
        top = max(min(e - 1, max_hit if max_hit is not None else 1 << 60), 0) // C * C
        best = 0
        for i, k in enumerate(AB.chain_digests(t, top=top), 1):
            if k not in self.rows[row]:
                break
            best = i * C
        return best


@pytest.mark.parametrize("seed", range(6))
def test_grid_rule_vs_oracle(seed):
    rng = random.Random(seed)
    idx = IDX(rows=2, users_per_row=1, pages_per_row=3000)
    oracle = Oracle(2)
    fam = [toks(500 + seed * 10 + i, 12000) for i in range(3)]
    for k in range(40):
        base = fam[rng.randrange(3)]
        t = np.concatenate([base[: rng.randrange(0, 9000)], toks(9000 + k + seed * 100, rng.randrange(1, 4000))])
        e = len(t)
        hit, row = idx.peek(t, e)
        want = max((oracle.hit(r, t, e), -r) for r in range(2))
        assert hit == want[0] and hit % C == 0 and hit <= max(e - 1, 0)
        plan = idx.plan_hit(k, t, e)
        if plan.hit_tokens:
            idx.bind(k, plan.hit_tokens)
            tgt = plan.row
        else:
            tgt = rng.randrange(2)
            idx.bind(k, 0, user=tgt)
        idx.page_table(k, e)
        for st in range(plan.hit_tokens, e // C * C, C):
            cp = idx.begin_chunk(k, st, st + C, t, e)
            if cp:
                idx.finish_chunk(cp)
        oracle.commit(tgt, t, e)
        idx.release(k)
        idx.check()


def test_random_op_sequences_keep_invariants():
    for seed in range(8):
        rng = random.Random(seed)
        idx = mk(rows=2, users_per_row=2, pages_per_row=24, xrow=bool(seed % 2), xrow_min=1024)
        fam = [toks(700 + seed * 7 + i, 4000) for i in range(3)]
        live = {}
        for step in range(150):
            op = rng.choice(["plan", "plan", "bind", "chunk", "release", "cancel", "reset", "abort"])
            try:
                if op == "plan":
                    t = np.concatenate(
                        [fam[rng.randrange(3)][: rng.randrange(0, 3500)], toks(5000 + step, rng.randrange(1, 1500))]
                    )
                    live[step] = dict(t=t, plan=idx.plan_hit(step, t, len(t)), state="planned", chunks=[], next=0)
                elif op == "bind" and live:
                    k = rng.choice(list(live))
                    s = live[k]
                    if s["state"] == "planned":
                        if s["plan"].hit_tokens:
                            b = idx.bind(k, s["plan"].hit_tokens)
                        else:
                            fr = [u for r in range(2) for u in idx.free_users(r)]
                            b = idx.bind(k, 0, user=rng.choice(fr)) if fr else None
                        if b is None:
                            live.pop(k)
                            idx.release(k)
                            continue
                        s["state"], s["next"] = "bound", b.hit_tokens
                        if b.xrow_src >= 0:  # a copy plan: the executor is not simulated here, give it up
                            live.pop(k)
                            idx.release(k)
                            continue
                        idx.page_table(k, len(s["t"]))
                elif op == "chunk" and live:
                    k = rng.choice(list(live))
                    s = live[k]
                    e = len(s["t"])
                    if s["state"] == "bound" and s["next"] + C <= e:
                        cp = idx.begin_chunk(k, s["next"], s["next"] + C, s["t"], e)
                        s["next"] += C
                        if cp:
                            s["chunks"].append(cp)
                            if rng.random() < 0.7:
                                idx.finish_chunk(s["chunks"].pop())
                elif op == "abort" and live:
                    k = rng.choice(list(live))
                    for cp in live[k]["chunks"]:
                        idx.abort_chunk(cp)
                    live[k]["chunks"] = []
                elif op == "release" and live:
                    k = rng.choice(list(live))
                    live.pop(k)
                    idx.release(k)
                elif op == "cancel" and live:
                    k = rng.choice(list(live))
                    idx.cancel(k)
                    if live[k]["state"] == "planned":
                        live.pop(k)
                elif op == "reset":
                    idx.reset()
            except AB.OutOfPages:
                pass
            idx.check()
        for k in list(live):
            idx.release(k)
        idx.check()
        s = idx.stats()
        assert all(r["pending"] == 0 for r in s["rows"]) and s["requests"] == 0


def test_thread_safety_scheduler_and_model_threads():
    idx = IDX(rows=2, users_per_row=2, pages_per_row=60)
    fam = [toks(800 + i, 3000) for i in range(3)]
    q = queue.Queue()
    errors = []

    def sched():
        rng = random.Random(1)
        try:
            for k in range(300):
                t = np.concatenate(
                    [fam[rng.randrange(3)][: rng.randrange(0, 2800)], toks(6000 + k, rng.randrange(1, 900))]
                )
                plan = idx.plan_hit(k, t, len(t))
                if rng.random() < 0.15:
                    idx.cancel(k)
                q.put((k, t, plan))
            for _ in range(3):
                q.put(None)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def model(seed):
        rng = random.Random(seed)
        try:
            while True:
                it = q.get()
                if it is None:
                    return
                k, t, plan = it
                e = len(t)
                b = idx.bind(k, plan.hit_tokens) if plan.hit_tokens else None
                if b is None:
                    for _ in range(20):
                        fr = [u for r in range(2) for u in idx.free_users(r)]
                        if not fr:
                            continue
                        try:
                            b = idx.bind(k, 0, user=rng.choice(fr))
                            break
                        except ValueError:
                            continue
                    if b is None:
                        idx.release(k)
                        continue
                try:
                    idx.restored(k)
                    idx.page_table(k, e)
                    for st in range(b.hit_tokens, e // C * C, C):
                        cp = idx.begin_chunk(k, st, st + C, t, e)
                        if cp:
                            (idx.abort_chunk if rng.random() < 0.1 else idx.finish_chunk)(cp)
                except AB.OutOfPages:
                    pass
                idx.release(k)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ths = [threading.Thread(target=sched)] + [threading.Thread(target=model, args=(i,)) for i in range(3)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(timeout=120)
    assert not any(th.is_alive() for th in ths) and not errors, errors[:3]
    for k in range(300):
        idx.release(k)
    idx.check()
    assert idx.stats()["requests"] == 0 and not idx._reserved and not idx._busy


# ---- Hy4 scale: the shipped pool sizes and the contract points ------------------------------------------------------------------------------------------
def test_pool_at_contract_points_fits():
    # B=8 bfp4 KV: 4402 pages per row, two 1M users per row = 4096 pages
    big = IDX(rows=1, users_per_row=2, pages_per_row=4402)
    for u in range(2):
        big.bind(f"u{u}", 0, user=u)
        big.page_table(f"u{u}", 1 << 20)
    assert len(big._rows[0].free_pages) == 4402 - 4096
    big.check()
    # B=4 SCALED_FP8: 2147 pages per row (704 B rows), one 1M user per row = 2048 pages (99 spare = the whole APC room)
    P = 2147
    idx = IDX(rows=1, users_per_row=1, pages_per_row=P)
    idx.bind("g1", 0, user=0)
    idx.page_table("g1", 1 << 20)
    assert len(idx._rows[0].free_pages) == 99
    idx.release("g1")
    idx = IDX(
        rows=1, users_per_row=2, pages_per_row=P
    )  # (a second 1M user on the same row does not fit: B=8 needs bfp4)
    idx.bind("g1", 0, user=0)
    idx.page_table("g1", 1 << 20)
    idx.bind("g2", 0, user=1)
    with pytest.raises(AB.OutOfPages):  # allow-pytest.raises: CPU-only module
        idx.page_table("g2", 1 << 20)
    idx.release("g2")
    idx.release("g1")
    idx.check()


def test_million_token_prompt_round_trip():
    idx = IDX(rows=2, users_per_row=1, pages_per_row=2147)
    t = toks(1234, (1 << 20) - 100)
    e = len(t)
    idx.bind("a", 0, user=0)
    idx.page_table("a", e)
    for st in range(0, e // C * C, C):
        cp = idx.begin_chunk("a", st, st + C, t, e)
        idx.finish_chunk(cp)
    idx.release("a")
    assert idx.peek(t, e) == ((e - 1) // C * C, 0)
    follow = np.concatenate([t, toks(5, 3000)])
    p = idx.plan_hit("b", follow, len(follow))
    assert p.hit_tokens == e // C * C and len(p.page_ids) == 2047 and p.row == 0
    idx.cancel("b")
    idx.check()
