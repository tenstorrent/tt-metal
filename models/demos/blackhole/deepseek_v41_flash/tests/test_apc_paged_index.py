# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of ``tt/apc_paged.py`` (block index of the device-resident paged APC; no device, no ttnn/torch). Run:  python_env/bin/python -m pytest tests/test_apc_paged_index.py -q"""
import hashlib
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


AP = _load("apc_paged_under_test", "../tt/apc_paged.py")
C = 512


def toks(seed, n):
    return np.random.default_rng(seed).integers(1, 100000, size=n, dtype=np.int64)


def mk(**kw):
    d = dict(rows=2, users_per_row=2, pages_per_row=400, halo_a=16, halo_b=8, key_slots=64, zone=2560)
    d.update(kw)
    return AP.BlockIndex(**d)


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


# ---- hashing -------------------------------------------------------------------------------------------------------------------------------------------
def prefixstore_chain(tokens, chunk=512):
    """the host store's chain (PrefixStore._chain in tt/prefix_cache.py), unsalted: one sha256 fed with the int64 bytes of every chunk"""
    t = np.asarray(tokens, dtype=np.int64)
    h, out = hashlib.sha256(), {}
    for s in range(chunk, len(t) + 1, chunk):
        h.update(t[s - chunk : s].tobytes())
        out[s] = h.copy().hexdigest()
    return out


def test_chain_equals_prefixstore_chain():
    t = toks(1, 5000)
    ref = prefixstore_chain(t)
    got = AP.chain_digests(t)
    assert got == [ref[s] for s in sorted(ref)] and len(got) == 9
    assert AP.chain_digests(t, top=1100) == got[:2]  # a chunk is only hashed when it is complete


def test_chain_prefix_and_incremental():
    a, b = toks(2, 3000), toks(3, 3000)
    mix = np.concatenate([a[:1536], b[1536:]])
    da, dm = AP.chain_digests(a), AP.chain_digests(mix)
    assert da[:3] == dm[:3] and da[3] != dm[3] and da[4] != dm[4]  # chained: the divergence propagates
    ch = AP.Chain()
    ch.extend(a, 600)
    ch.extend(a, 2000)
    ch.extend(a, 3000)
    assert ch.digests == da


def test_chain_salt_and_image_keys_separate():
    t = toks(4, 2048)
    assert AP.chain_digests(t, salt=b"x")[0] != AP.chain_digests(t)[0]
    assert AP.chain_digests(t, salt=b"x") == AP.chain_digests(t, salt=b"x")
    img1, img2 = t.copy(), t.copy()
    img1[
        100:400
    ] = -12345  # image span replaced by one value derived from the content hash (key_tokens of the image branch)
    img2[100:400] = -54321
    d, d1, d2 = AP.chain_digests(t), AP.chain_digests(img1), AP.chain_digests(img2)
    assert d1[0] != d[0] and d1[0] != d2[0] and d1[1:] != d[1:]
    assert AP.chain_digests(img1) == AP.chain_digests(img1.copy())


def test_chain_accepts_negative_and_other_dtypes():
    t = toks(5, 1024)
    assert AP.chain_digests(t.astype(np.int32)) == AP.chain_digests(t)
    assert AP.chain_digests(list(map(int, t))) == AP.chain_digests(t)


# ---- hit rule ------------------------------------------------------------------------------------------------------------------------------------------
def test_hit_rule_margin_and_grid():
    idx = mk(zone=10**6)
    t = toks(10, 4096)
    assert run(idx, "a", t)[0] == 0
    # same prompt again: deepest boundary <= e - 128 = 3968 -> 3584 (the chunk end 4096 is a boundary but > e - 128)
    assert idx.peek(t, 4096) == (3584, 0)
    longer = np.concatenate([t, toks(11, 300)])
    assert idx.peek(longer, len(longer)) == (4096, 0)  # the whole old prompt is a prefix: its tip halo is a hit point
    assert idx.peek(t[:600], 600) == (0, -1)  # e - 128 = 472 < 512
    assert idx.peek(t[:1000], 1000)[0] == 512
    assert idx.peek(t[:1152], 1152)[0] == 1024  # 1152 - 128 = 1024


def test_no_hit_without_block_chain_or_halo():
    idx = mk(halo_thin=3)  # halos only at every 3rd chunk end and at the tip
    t = toks(12, 6 * C + 77)
    run(idx, "a", t)
    longer = np.concatenate([t, toks(13, 2000)])
    # halos exist at boundaries 3 and 6 (chunks 3,6) and the tip (6 * C): hits can only be at 1536 and 3072
    assert idx.peek(longer[: 5 * C + 300], 5 * C + 300)[0] == 3 * C
    assert idx.peek(longer, len(longer))[0] == 6 * C


def test_zone_rule_partial_and_full_halos():
    # zone = 2560: the prompt of 8192 runs layers >= 21 only on chunks starting >= p0 = 5632; boundaries <= 5632 have partial halos (A only)
    idx = mk(halo_a=32, halo_b=16, key_slots=64, pages_per_row=600)
    t = toks(14, 8192)
    run(idx, "a", t)
    row = idx._rows[0 if idx.peek(t, 8192)[1] == 0 else 1]
    by = {b.idx: b for b in row.blocks.values()}
    assert all(by[i].halo_a >= 0 for i in range(1, 17))
    assert [i for i in range(1, 17) if by[i].halo_b >= 0] == list(
        range(12, 17)
    )  # chunks 12..16 start at >= 5632 (i - 1 >= 11)
    # a new prompt of the same 8192 head with a different tail and length 9000: p0 = 6144; hits at <= 6144 only need A
    b = np.concatenate([t[:8192], toks(15, 808)])
    assert idx.peek(b, len(b))[0] == 8192  # 8192 > p0 (6144): the halo at 8192 is full (chunk 16 started at 7680)
    b2 = np.concatenate([t[:7000], toks(16, 2000)])  # 9000 tokens, the old data is valid up to 6656
    assert idx.peek(b2, len(b2))[0] == 6656
    # a prompt of 6200 tokens: p0 = max(0, 3584 // 512 * 512) = 3584; boundary 5632 needs B: the old halo at 5632 is A-only
    c = t[:6200]
    assert AP.zone_start(6200, 2560) == 3584
    assert idx.peek(c, 6200)[0] == 5632 if by[11].halo_b >= 0 else idx.peek(c, 6200)[0] == 3584


def test_zone_rule_partial_halo_only_usable_before_p0():
    idx = mk(zone=1024, halo_a=16, halo_b=0 + 8, key_slots=64)  # p0 = floor((e-1024)/512)*512
    t = toks(17, 4096)
    run(idx, "a", t)  # e = 4096: p0 = 3072; chunks 7, 8 (starts 3072, 3584) full; boundaries 512..3072 partial
    row = idx._rows[idx.peek(t, 4096)[1]]
    by = {b.idx: b for b in row.blocks.values()}
    assert [i for i in range(1, 9) if by[i].halo_b >= 0] == [7, 8]
    # new prompt, e = 3200 -> p0 = 2048: a hit at s <= 2048 may use partial halos; s = 2560 needs B (the old halo at 2560 is partial) -> hit 2048
    n = np.concatenate([t[:3000], toks(18, 200)])
    assert idx.peek(n, 3200)[0] == 2048
    # e = 8000 with the same head: p0 = 6912: every boundary up to 4096 is below p0 -> A is enough: deepest = 4096 (tip halo present, full)
    n2 = np.concatenate([t, toks(19, 3904)])
    assert idx.peek(n2, 8000)[0] == 4096


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
    spans = [(700, 600, "h")]  # boundary 1024 is inside (700, 1300); 512 and 1536 are not
    e = len(t)
    idx.bind("a", 0, user=0)
    idx.page_table("a", e)
    ends = []
    for st in range(0, e // C * C, C):
        cp = idx.begin_chunk("a", st, st + C, t, e, spans=spans)
        ends.append(None if cp is None else cp.block_idx)
        if cp:
            idx.finish_chunk(cp)
    assert ends == [1, None, None, None, None]  # block 2 is not a boundary: the chain stops (no parent, no later block)
    idx.release("a")


# ---- hit binding ---------------------------------------------------------------------------------------------------------------------------------------
def test_hit_goes_to_reserved_user_on_owning_row():
    idx = mk()
    t = toks(30, 3000)
    run(idx, "a", t, user=3)  # row 1
    p = idx.plan_hit("b", t, 3000)
    assert p.hit_tokens == 2560 and p.row == 1 and p.user in (2, 3)
    assert p.user in idx._reserved
    assert p.user not in idx.free_users(1) and idx.reserved_users(1) == [p.user]
    b = idx.bind("b", 2560)
    assert b is p and idx._busy[p.user] == "b"
    assert len(p.page_ids) == 2560 // 128 and len(p.key_slots) == 5 and len(p.block_ids) == 5
    assert p.halo_a >= 0 and p.halo_b >= 0  # e = 3000: p0 = 0: 2560 > p0: B needed and the donor (same p0) has it
    idx.release("b")
    assert p.user in idx.free_users(1)


def test_halo_b_in_plan_only_when_needed():
    idx = mk(zone=1024)
    t = toks(31, 4096)
    run(idx, "a", t)
    n = np.concatenate([t[:2100], toks(32, 4000)])  # e = 6100 -> p0 = 5120; hit 2048 <= p0: B not needed
    p = idx.plan_hit("n", n, len(n))
    assert p.hit_tokens == 2048 and p.halo_b == -1 and p.halo_a >= 0
    m = np.concatenate([t, toks(33, 300)])  # e = 4396, p0 = 3328: hit 4096 > p0: B needed and present
    p2 = idx.plan_hit("m", m, len(m))
    assert p2.hit_tokens == 4096 and p2.halo_b >= 0


def test_replan_is_idempotent_and_replaces_reservation():
    idx = mk()
    t = toks(34, 3000)
    run(idx, "a", t)
    p1 = idx.plan_hit("b", t, 3000)
    p2 = idx.plan_hit("b", t, 3000)
    assert p1.hit_tokens == p2.hit_tokens and len(idx._reserved) == 1
    assert all(b.refs == 1 for b in idx._rows[p2.row].lru.values() if b.id in p2.block_ids)
    idx.check()
    idx.cancel("b")
    assert not idx._reserved and all(b.refs == 0 and b.halo_pins == 0 for r in idx._rows for b in r.lru.values())
    idx.check()


def test_continuation_never_takes_a_hit_and_unknown_start_is_none():
    idx = mk()
    t = toks(35, 3000)
    run(idx, "a", t)
    p = idx.plan_hit("b", t, 3000)
    assert idx.bind("b", p.hit_tokens) is p
    assert idx.bind("b", 4096) is None  # a continuation of a bound request is the tracker's business
    assert idx.bind("zzz", 512, user=0) is None  # never planned, start > 0
    assert idx.bind("zzz", 0) is None  # a miss without a user


def test_lost_reservation_reset_ttl_and_start_mismatch():
    now = [0.0]
    idx = mk(ttl_s=10.0, clock=lambda: now[0])
    t = toks(36, 3000)
    run(idx, "a", t)
    # reset between plan and bind
    p = idx.plan_hit("b", t, 3000)
    idx.reset()
    assert idx.bind("b", p.hit_tokens) is None and idx.stats()["lost"] == 1
    assert not idx._reserved
    # the model then recomputes from 0 and registers itself
    assert idx.bind("b", 0, user=0).hit_tokens == 0
    idx.release("b")
    run(idx, "a2", t)
    # TTL
    p = idx.plan_hit("c", t, 3000)
    now[0] = 11.0
    assert idx.bind("c", p.hit_tokens) is None and idx.stats()["lost"] == 2 and idx.stats()["expired"] == 1
    assert not idx._reserved and all(b.refs == 0 for r in idx._rows for b in r.lru.values())
    # start_pos != hit (the plugin passed another hit): lost; start 0 falls to the miss path
    p = idx.plan_hit("d", t, 3000)
    assert idx.bind("d", 1024) is None
    p = idx.plan_hit("e", t, 3000)
    assert idx.bind("e", 0, user=1).hit_tokens == 0
    idx.check()


def test_no_free_user_on_owning_row_falls_back_to_other_row_or_miss():
    idx = mk(users_per_row=1)
    t = toks(37, 3000)
    run(idx, "a", t, user=0)  # row 0 caches
    idx.bind("busy", 0, user=0)  # row 0's only user is busy
    assert idx.plan_hit("b", t, 3000).hit_tokens == 0
    idx.release("busy")
    p = idx.plan_hit("b", t, 3000)
    assert p.hit_tokens == 2560 and p.row == 0 and p.user == 0
    assert idx.plan_hit("c", t, 3000).hit_tokens == 0  # the reserved user is not available to a second request


def test_two_requests_get_distinct_users_and_deepest_row_wins():
    free = {0: [0, 1], 1: [2, 3]}
    idx = mk(free_users=lambda r: free[r])
    t = toks(38, 3000)
    run(idx, "a", t, user=0)  # row 0
    longer = np.concatenate([t, toks(39, 3000)])
    free[0] = []  # row 0 is busy: "b" misses and is computed on row 1
    run(idx, "b", longer, user=2)
    free[0] = [0, 1]
    p = idx.plan_hit("c", longer, len(longer))
    assert p.row == 1 and p.hit_tokens > 2560
    q = idx.plan_hit("d", longer, len(longer))
    assert q.row == 1 and q.user != p.user  # the second user of the deepest row
    r = idx.plan_hit("e", longer, len(longer))
    assert r.row == 0 and r.hit_tokens == 2560  # row 1 has no free user left


def test_free_users_provider_is_the_truth():
    free = {0: [], 1: [3]}
    idx = mk(free_users=lambda r: free[r])
    t = toks(40, 3000)
    run(idx, "a", t, user=3)
    assert idx.plan_hit("b", t, 3000).user == 3
    free[1] = []
    assert idx.plan_hit("c", t, 3000).hit_tokens == 0


# ---- refcount / eviction -------------------------------------------------------------------------------------------------------------------------------
def test_pages_shared_and_freed_only_at_last_release():
    idx = mk(pages_per_row=100, key_slots=32)
    t = toks(50, 3000)
    run(idx, "a", t, user=0, release=False)
    idx.check()
    pa = idx.page_table("a", 3000)
    run_b = idx.plan_hit("b", t, 3000)
    assert run_b.hit_tokens == 2560 and run_b.row == 0
    idx.bind("b", 2560)
    idx.restored("b")
    pb = idx.page_table("b", 3000)
    assert pb[:20] == pa[:20] and not set(pb[20:]) & set(pa)  # shared pages, private tail
    row = idx._rows[0]
    assert all(b.refs == 2 for b in row.lru.values() if b.idx <= 5)
    idx.release("a")
    idx.check()
    assert all(b.refs == 1 for b in row.lru.values() if b.idx <= 5)
    free_before = len(row.free_pages)
    idx.release("b")
    assert len(row.free_pages) == free_before + 4  # only b's private pages (24 - 20): the cached blocks stay
    assert all(b.refs == 0 for b in row.lru.values())
    idx.check()


def test_never_double_books_pages_under_pressure():
    idx = mk(rows=1, pages_per_row=60, key_slots=12, halo_a=6, halo_b=3)
    rng = random.Random(0)
    heads = [toks(60 + i, 1536) for i in range(4)]
    for k in range(40):
        t = np.concatenate(
            [heads[rng.randrange(4)][: rng.choice([512, 1024, 1536])], toks(1000 + k, rng.choice([300, 700, 1300]))]
        )
        try:
            run(idx, k, t, user=rng.randrange(2))
        except AP.OutOfPages:
            idx.release(k)
        idx.check()


def test_lru_leaf_first_eviction_order():
    idx = mk(rows=1, key_slots=6, pages_per_row=200, halo_a=6, halo_b=8)
    a, b = toks(70, 1500), toks(71, 1500)  # 2 full chunks each (+ partial)
    run(idx, "a", a, user=0)
    run(idx, "b", b, user=0)
    run(idx, "c", toks(72, 1500), user=0)  # 6 blocks now
    row = idx._rows[0]
    assert len(row.blocks) == 6 and not row.free_key
    ev = []
    idx.add_listener(lambda e: ev.append(e) if e["kind"] == "block" else None)
    # touch prompt a (a hit): its blocks become the newest
    assert idx.plan_hit("a2", np.concatenate([a, toks(73, 600)]), 2100).hit_tokens == 1024
    idx.cancel("a2")
    run(
        idx, "d", toks(74, 1100), user=0
    )  # needs 2 blocks: evicts the oldest LEAVES: b's chain is older than c's: b2 (leaf) then b1? leaf first: the oldest leaf, then its parent
    gone = [i for e in ev for i in e["ids"]]
    assert len(gone) == 2
    keys = {blk.key for blk in row.blocks.values()}
    da, db, dc = (AP.chain_digests(x)[:2] for x in (a, b, toks(72, 1500)))
    assert set(da) <= keys  # the touched chain survives
    assert not (set(db) <= keys)  # b (the oldest) lost blocks, deepest first
    if db[0] not in keys:
        assert db[1] not in keys  # never a parent without its child gone first
    idx.check()


def test_pinned_blocks_are_never_evicted_and_dead_blocks_free_at_release():
    idx = mk(rows=1, key_slots=3, pages_per_row=40, halo_a=3, halo_b=3)
    a = toks(80, 1100)
    run(idx, "a", a, user=0)
    p = idx.plan_hit("p", np.concatenate([a, toks(81, 300)]), 1400)  # pins a's 2 blocks
    assert p.hit_tokens == 1024
    run(idx, "x", toks(82, 1100), user=1)  # takes the last slot, then wants 2 more: evicts what it can
    row = idx._rows[0]
    assert {b.id for b in row.lru.values()} >= set(p.block_ids)
    idx.check()
    idx.bind("p", 1024)
    idx.reset()  # bound request keeps its pages
    assert not row.lru and p.page_ids[0] not in row.free_pages
    idx.check()
    idx.release("p")
    assert all(pg in row.free_pages for pg in p.page_ids)
    idx.check()


def test_out_of_pages_when_everything_is_held():
    idx = mk(rows=1, pages_per_row=10, key_slots=4)
    idx.bind("a", 0, user=0)
    with pytest.raises(AP.OutOfPages):  # allow-pytest.raises: CPU-only module, no repo fixtures
        idx.page_table("a", 11 * 128)
    assert idx.stats()["out_of_pages"] == 1
    idx.page_table("a", 10 * 128)
    idx.rollback("a", 3 * 128)
    assert len(idx._rows[0].free_pages) == 7
    idx.release("a")
    assert len(idx._rows[0].free_pages) == 10


def test_page_eviction_makes_room_from_cached_blocks():
    idx = mk(rows=1, pages_per_row=20, key_slots=16, halo_a=8, halo_b=0)
    run(idx, "a", toks(90, 1100), user=0)  # 2 cached blocks = 8 pages
    run(idx, "b", toks(91, 1100), user=0)  # 2 more blocks = 8 pages: 16 of 20 cached
    idx.bind("c", 0, user=0)
    idx.page_table("c", 16 * 128)  # needs 16 pages: evicts cached blocks
    assert idx.stats()["ev_block"] >= 2
    idx.check()


def test_halo_slots_lru_and_halo_b_drop_keeps_block():
    idx = mk(rows=1, key_slots=64, pages_per_row=300, halo_a=4, halo_b=2, zone=512)
    t = toks(100, 3 * C + 10)
    run(
        idx, "a", t, user=0
    )  # halos at chunk ends 1,2,3 (zone 512: p0 = 1024: chunk 3 starts at 1024 -> B at 3 only... chunk 3 is full)
    row = idx._rows[0]
    by = {b.idx: b for b in row.blocks.values()}
    assert all(by[i].halo_a >= 0 for i in (1, 2, 3)) and by[3].halo_b >= 0
    t2 = toks(101, 3 * C + 10)
    run(idx, "b", t2, user=0)  # 6 halo A wanted, 4 slots: the oldest of the first prompt go
    assert len(row.free_a) == 0 and len(row.a_lru) == 4
    assert len(row.blocks) == 6  # blocks stay (key slots are plentiful)
    assert by[1].halo_a == -1 and by[1].state == "live"
    idx.check()


def test_listener_events_and_reset_event():
    idx = mk(rows=1, key_slots=64, halo_a=2, halo_b=0)
    ev = []
    idx.add_listener(ev.append)
    run(idx, "a", toks(110, 4 * C + 5), user=0)
    assert any(e["kind"] == "halo_a" and e["why"] == "lru" for e in ev)
    idx.reset()
    assert ev[-1]["kind"] == "block" and ev[-1]["why"] == "reset" and len(ev[-1]["ids"]) == 4
    idx.add_listener(lambda e: 1 / 0)  # a broken listener must not break the cache
    idx.reset()
    assert idx.stats()["resets"] == 2


def test_set_signature_resets():
    idx = mk(signature="s1")
    t = toks(111, 3000)
    run(idx, "a", t)
    assert idx.peek(t, 3000)[0] == 2560
    idx.set_signature("s1")
    assert idx.peek(t, 3000)[0] == 2560
    idx.set_signature("s2")
    assert idx.peek(t, 3000)[0] == 0


# ---- chunk commit protocol -----------------------------------------------------------------------------------------------------------------------------
def test_two_phase_commit_invisible_until_finished_and_abort_returns_slots():
    idx = mk(rows=1)
    t = toks(120, 2 * C + 200)
    e = len(t)
    idx.bind("a", 0, user=0)
    idx.page_table("a", e)
    cp1 = idx.begin_chunk("a", 0, C, t, e)
    cp2 = idx.begin_chunk("a", C, 2 * C, t, e)
    assert cp1.key_slot >= 0 and cp1.halo_a >= 0 and cp1.halo_b >= 0 and cp1.block_idx == 1 and cp1.full
    assert cp2.key_slot != cp1.key_slot and cp2.halo_a != cp1.halo_a
    assert idx.peek(t, e) == (0, -1)  # nothing published yet
    assert idx.begin_chunk("a", 2 * C, e, t, e) is None  # the final partial chunk is not a boundary
    assert idx.finish_chunk(cp1) and idx.peek(t, e) == (512, 0)  # block 1 alone is a hit point (e - 128 = 1096)
    idx.abort_chunk(cp2)
    st = idx.stats()["rows"][0]
    assert st["key_free"] == idx.caps["key"] - 1 and st["halo_a_free"] == idx.caps["a"] - 1
    idx.check()
    idx.release("a")
    idx.check()


def test_dup_commit_keeps_pages_private_and_upgrades_partial_halo():
    idx = mk(rows=1, zone=1024, halo_a=8, halo_b=4, key_slots=32, pages_per_row=200)
    t = toks(121, 2048)
    # request 1 has a prompt of 4100 tokens in the zone sense (e = 4100: p0 = 3072): chunks 1-4 are partial (A only)
    long1 = np.concatenate([t, toks(122, 2052)])
    run(idx, "a", long1, user=0)
    row = idx._rows[0]
    by = {b.idx: b for b in row.blocks.values()}
    assert by[2].halo_b == -1 and by[4].halo_b == -1
    # request 2: the same first 2048 tokens, prompt of 2100 tokens: p0 = 1024 -> chunks 3.. full; chunk 2 (start 512) is partial; here e = 2100: chunks 1..4 with starts 0..1536: chunks 3,4 full
    r2 = np.concatenate([t, toks(123, 52)])
    idx.bind("b", 0, user=1)
    idx.page_table("b", len(r2))
    cps = [idx.begin_chunk("b", s, s + C, r2, len(r2)) for s in range(0, 4 * C, C)]
    assert cps[0] is None or cps[0].dup  # block exists; chunk 1 is partial for r2 too, nothing to add
    up = cps[3]
    assert up is not None and up.dup and up.key_slot == -1 and up.halo_a == -1 and up.halo_b >= 0 and up.full
    for cp in cps:
        if cp:
            idx.finish_chunk(cp)
    assert by[4].halo_b >= 0 and idx.stats()["upgrades"] >= 1
    assert all(p in idx._reqs["b"].priv for p in idx.page_table("b", len(r2))[: 4 * 4])  # dup: pages stay private
    idx.check()
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
    assert ca is not None and cb is None  # same key pending: the second does not commit
    assert idx.finish_chunk(ca)
    idx.release("a")
    idx.release("b")
    idx.check()


def test_commit_dropped_when_parent_is_gone_or_cache_reset():
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
    idx.check()
    assert len(idx._rows[0].free_pages) == idx.caps["pages"]


def test_release_with_pending_chunks_frees_everything():
    idx = mk(rows=1)
    t = toks(126, 1600)
    idx.bind("a", 0, user=0)
    idx.page_table("a", 1600)
    idx.begin_chunk("a", 0, C, t, 1600)
    idx.release("a")
    idx.check()
    s = idx.stats()["rows"][0]
    assert s["pending"] == 0 and s["key_free"] == idx.caps["key"] and s["pages_free"] == idx.caps["pages"]


def test_release_user_by_physical_id():
    idx = mk()
    idx.bind("a", 0, user=3)
    idx.release_user(3)
    assert "a" not in idx._reqs and 3 in idx.free_users(1)
    idx.release_user(3)  # idempotent


def test_halo_pin_blocks_slot_recycling_until_restored():
    idx = mk(rows=1, halo_a=2, halo_b=2, key_slots=32, pages_per_row=200)
    a = toks(130, 1100)
    run(idx, "a", a, user=0)  # halos at 1, 2 (tip = 1024)
    p = idx.plan_hit("p", np.concatenate([a, toks(131, 200)]), 1300)
    assert p.hit_tokens == 1024
    idx.bind("p", 1024)
    t = toks(132, 1100)
    idx.bind("x", 0, user=1)
    idx.page_table("x", 1100)
    cp = idx.begin_chunk("x", 0, C, t, 1100)
    assert cp.halo_a != p.halo_a  # the pinned halo of the hit boundary survived (the other one was recycled)
    idx.finish_chunk(cp)
    idx.restored("p")
    cp2 = idx.begin_chunk("x", C, 2 * C, t, 1100)
    assert cp2.halo_a >= 0
    idx.check()


# ---- hook ----------------------------------------------------------------------------------------------------------------------------------------------
def test_hook_plan_hit_and_key_tokens_fn():
    idx = mk()
    t = toks(140, 3000)
    run(idx, "a", t)
    hook = AP.ApcPagedHook(idx)
    assert hook.plan_hit("h", t, 3000) == (2560, idx.peek(t, 3000)[1])
    hook.cancel("h")
    assert not idx._reserved
    calls = []

    def fn(tokens, mm):
        calls.append(mm)
        return None if mm == "image" else tokens

    hook2 = AP.ApcPagedHook(idx, fn)
    assert hook2.plan_hit("i", t, 3000, mm="image") == (0, -1)
    assert hook2.plan_hit("j", t, 3000, mm=None)[0] == 2560 and calls == ["image", None]


# ---- property tests ------------------------------------------------------------------------------------------------------------------------------------
class Oracle:
    """independent model of the published state (no eviction): digest -> (has A, has B) per row; hit = the deepest valid boundary"""

    def __init__(self, rows):
        self.rows = [dict() for _ in range(rows)]

    def commit(self, row, t, e, s0, zone, thin=1):
        d = AP.chain_digests(t, top=e // C * C)
        p0 = AP.zone_start(e, zone)
        for i, k in enumerate(d, 1):
            self.rows[row].setdefault(k, [False, False])
            if i * C > s0 and (i == e // C or i % thin == 0):
                h = self.rows[row][k]
                h[0] = True
                h[1] = h[1] or (i * C - C >= p0)

    def hit(self, row, t, e, zone, max_hit=None):
        top = max(min(e - 128, max_hit if max_hit is not None else 1 << 60), 0) // C * C
        d = AP.chain_digests(t, top=top)
        p0 = AP.zone_start(e, zone)
        best = 0
        for i, k in enumerate(d, 1):
            h = self.rows[row].get(k)
            if h is None:
                break
            if h[0] and (i * C <= p0 or h[1]):
                best = i * C
        return best


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("zone", [1024, 2560, 8192])
def test_grid_and_replay_rule_vs_oracle(seed, zone):
    rng = random.Random(seed)
    idx = AP.BlockIndex(rows=2, users_per_row=1, pages_per_row=5000, halo_a=400, halo_b=400, key_slots=1000, zone=zone)
    oracle = Oracle(2)
    fam = [toks(500 + seed * 10 + i, 12000) for i in range(3)]
    for k in range(40):
        base = fam[rng.randrange(3)]
        cut = rng.randrange(0, 9000)
        tail = toks(9000 + k + seed * 100, rng.randrange(0, 4000))
        t = np.concatenate([base[:cut], tail])
        e = len(t)
        if e < 1:
            continue
        # property: the answer is on the grid, <= e - 128, and equals the oracle on the best row
        hit, row = idx.peek(t, e)
        want = max((oracle.hit(r, t, e, zone), -r) for r in range(2))
        assert hit == want[0] and hit % C == 0 and hit <= max(e - 128, 0)
        free_user = {0: 0, 1: 1}
        plan = idx.plan_hit(k, t, e)
        if plan.hit_tokens:
            idx.bind(k, plan.hit_tokens)
            tgt = plan.row
        else:
            tgt = rng.randrange(2)
            idx.bind(k, 0, user=free_user[tgt])
        idx.page_table(k, e)
        for st in range(plan.hit_tokens, e // C * C, C):
            cp = idx.begin_chunk(k, st, st + C, t, e)
            if cp:
                idx.finish_chunk(cp)
        oracle.commit(tgt, t, e, plan.hit_tokens, zone)
        idx.release(k)
        idx.check()


def test_random_op_sequences_keep_invariants():
    for seed in range(8):
        rng = random.Random(seed)
        idx = mk(rows=2, users_per_row=2, pages_per_row=70, halo_a=5, halo_b=2, key_slots=14, zone=1536)
        fam = [toks(700 + seed * 7 + i, 4000) for i in range(3)]
        live = {}
        for step in range(150):
            op = rng.choice(["plan", "plan", "bind", "chunk", "release", "cancel", "reset", "abort"])
            try:
                if op == "plan":
                    k = step
                    base = fam[rng.randrange(3)]
                    t = np.concatenate([base[: rng.randrange(0, 3500)], toks(5000 + step, rng.randrange(1, 1500))])
                    plan = idx.plan_hit(k, t, len(t))
                    live[k] = dict(t=t, plan=plan, state="planned", chunks=[], next=0)
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
            except AP.OutOfPages:
                pass
            idx.check()
        for k in list(live):
            idx.release(k)
        idx.check()
        s = idx.stats()
        assert all(r["pending"] == 0 for r in s["rows"]) and s["requests"] == 0


# ---- threads ---------------------------------------------------------------------------------------------------------------------------------------------
def test_thread_safety_scheduler_and_model_threads():
    idx = AP.BlockIndex(rows=2, users_per_row=2, pages_per_row=120, halo_a=8, halo_b=4, key_slots=20, zone=1536)
    fam = [toks(800 + i, 3000) for i in range(3)]
    q = queue.Queue()
    errors = []
    done = threading.Event()

    def sched():
        rng = random.Random(1)
        try:
            for k in range(300):
                base = fam[rng.randrange(3)]
                t = np.concatenate([base[: rng.randrange(0, 2800)], toks(6000 + k, rng.randrange(1, 900))])
                plan = idx.plan_hit(k, t, len(t))
                if rng.random() < 0.15:
                    idx.cancel(k)  # abort before the model binds it
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
                if plan.hit_tokens:
                    b = idx.bind(k, plan.hit_tokens)
                else:
                    b = None
                if b is None:
                    for _ in range(
                        20
                    ):  # claim a free user (another model thread may take it between the query and the bind)
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
                except AP.OutOfPages:
                    pass
                idx.release(k)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ths = [threading.Thread(target=sched)] + [threading.Thread(target=model, args=(i,)) for i in range(3)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(timeout=120)
    assert not any(th.is_alive() for th in ths)
    assert not errors, errors[:3]
    for k in range(300):
        idx.release(k)
    idx.check()
    assert idx.stats()["requests"] == 0 and not idx._reserved and not idx._busy


# ---- simulator comparison ------------------------------------------------------------------------------------------------------------------------------
SIM = _load("apc_paged_sim", "../tools/apc_paged_sim.py")


def _hits_sim(reqs, H, P, p_busy, seed, thin):
    rng = np.random.default_rng(seed)
    rows = [SIM.Row(H, P, thin) for _ in range(4)]
    out = []
    for t in reqs:
        e = len(t)
        free = [i for i in range(4) if rng.random() >= p_busy] or [int(rng.integers(4))]
        best = (0, None, None)
        for i in free:
            b, ch = rows[i].hit(t, e)
            if b > best[0]:
                best = (b, i, ch)
        s = best[0]
        tgt = best[1] if best[1] is not None else free[int(rng.integers(len(free)))]
        if s:
            rows[best[1]].touch(best[2], s)
        rows[tgt].insert(t, e, s)
        out.append(s)
    return out


def _hits_idx(reqs, H, P, p_busy, seed, thin):
    rng = np.random.default_rng(seed)
    state = {"free": set()}
    idx = AP.BlockIndex(
        rows=4, users_per_row=1, pages_per_row=P * 4 + 4000, halo_a=H, halo_b=H, key_slots=P, zone=SIM.ZR, halo_thin=thin,
        free_users=lambda r: [r] if r in state["free"] else [],
    )  # fmt: skip
    out = []
    for k, t in enumerate(reqs):
        e = len(t)
        free = [i for i in range(4) if rng.random() >= p_busy] or [int(rng.integers(4))]
        state["free"] = set(free)
        plan = idx.plan_hit(k, t, e)
        if plan.hit_tokens:
            idx.bind(k, plan.hit_tokens)
            tgt = plan.row
        else:
            tgt = free[int(rng.integers(len(free)))]
            idx.bind(k, 0, user=tgt)
        idx.restored(k)
        idx.page_table(k, e)
        for st in range(plan.hit_tokens, e // C * C, C):
            cp = idx.begin_chunk(k, st, st + C, t, e)
            if cp:
                idx.finish_chunk(cp)
        idx.release(k)
        out.append(plan.hit_tokens)
    idx.check()
    return out


CASES = {
    "agents8": lambda: SIM.agents(8, 6),
    "agents16": lambda: SIM.agents(16, 4),
    "agents32": lambda: SIM.agents(32, 3),
    "agents_8k16k": lambda: SIM.agents(8, 10, sys_len=1500, grow=1600, cap=16400, first=(5000, 6500)),
    "sys6k": lambda: SIM.shared_sys(24, 6000),
    "sys30k": lambda: SIM.shared_sys(10, 30000),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("H,thin,p_busy", [(128, 1, 0.0), (128, 1, 0.5), (24, 1, 0.0), (32, 4, 0.5)])
def test_matches_simulator_hit_for_hit(case, H, thin, p_busy):
    reqs = CASES[case]()
    a = _hits_sim(reqs, H, 4096, p_busy, 0, thin)
    b = _hits_idx(reqs, H, 4096, p_busy, 0, thin)
    assert a == b, [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y][:5]
    assert sum(a) > 0 or case.startswith("sys") is False
