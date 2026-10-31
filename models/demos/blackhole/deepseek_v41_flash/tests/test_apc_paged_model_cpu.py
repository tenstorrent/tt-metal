# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) of the model wiring of the paged APC (``tt/apc_paged_model.py``): the pool allocator on top of the block index, request binding (miss / hit / lost reservation /
continuation / anonymous / image), the slot-id tables handed to the device stores, refcounts of the shared pages, the user placement that dodges reserved users (``vllm_state.SlotTable``).
A fake model + fake stores stand in for the device. Run:  python_env/bin/python -m pytest --noconftest -c /dev/null --rootdir=/tmp tests/test_apc_paged_model_cpu.py -q   (from the worktree root)"""

import numpy as np
import pytest
import torch

from models.demos.blackhole.deepseek_v41_flash.tt import apc_paged_model as APM
from models.demos.blackhole.deepseek_v41_flash.tt.kv_paged import PageAllocator
from models.demos.blackhole.deepseek_v41_flash.tt.vllm_state import SlotTable

C = 512
ROWS, U = 2, 2


class FakePool:
    def __init__(self, rows, users, num_pages):
        self.rows, self.T, self.B, self.num_pages, self.max_pages = rows, users, rows * users, num_pages, 8200
        self.allocs = [PageAllocator(num_pages, 128, self.max_pages) for _ in range(rows)]

    def user_key(self, b):
        return b // self.T, b

    def admit(self, b, tokens, reserve_tokens=0):
        r, k = self.user_key(b)
        return self.allocs[r].admit(k, tokens, reserve_tokens)

    def grow(self, b, tokens, reserve_tokens=0):
        r, k = self.user_key(b)
        return self.allocs[r].grow(k, tokens, reserve_tokens)

    def release(self, b):
        r, k = self.user_key(b)
        self.allocs[r].release(k)

    def sync_page_table(self):
        pass


class FakeModel:
    rows, U, Up, host_rows = ROWS, U, U, None

    def __init__(self, pages=600):
        self.num_pages = pages + APM.IDLE_PAGES * U
        self.pool = FakePool(ROWS, U, self.num_pages)
        self.log_lines = []


class Rec:
    """records the calls of a device store"""

    def __init__(self):
        self.calls = []
        self.store = {"A": 1, "B": 1}

    def __getattr__(self, name):
        def f(*a):
            self.calls.append((name, a))

        return f


@pytest.fixture
def pa(monkeypatch):
    monkeypatch.setenv("DSV41_PF_SINGLE", "1")
    monkeypatch.setenv("DSV41_PF_REPLAY", "2560")
    monkeypatch.setenv("DSV41_PF_PAGED", "read")
    monkeypatch.setenv("DSV41_PF_UMASK", "1")
    m = FakeModel()
    p = APM.PagedApc(m, C, log=m.log_lines.append)
    p.model = m
    p.ks, p.hs = Rec(), Rec()
    p.owners = lambda: ({}, {})
    p.halos = lambda: {}
    p.idx.free_users_fn = None
    p.install()
    return p


def toks(seed, n):
    return torch.from_numpy(np.random.default_rng(seed).integers(1, 100000, size=n, dtype=np.int64))


def prefill(pa, user, rid, t, st=0, hits=None, e=None, tot=None, rep=True):
    """what ``_prefill_interleaved_one`` does for one user: hit binding / new binding, pages, then one commit per 512 window ending <= e"""
    e = e or t.numel()
    items, hits_ = pa.prepare_hits([(user, t, st, e)], {user: rid}, {user: tot or e}, rep=rep)
    (b, _, st, _) = items[0]
    if st > 0 and not hits_ and pa.rid.get(b) != rid:
        st = 0  # (the model's resume check: no carried state for a start > 0 -> recompute from 0)
    if st == 0:
        pa.model.pool.release(b)
        pa.bind_new(b, rid, tot or e)
        pa.model.pool.admit(b, e + 1)
    else:
        pa.model.pool.grow(b, e + 1)
    if hits_:
        pa.restore(hits_)
    for s_end in range(st // C * C + C, e // C * C + 1, C):
        if s_end > st:
            pa.commit([(b, t, s_end, e)])
    return st, hits_


def test_allocator_idle_and_bound(pa):
    pool = pa.model.pool
    for b in range(4):
        assert pool.allocs[b // U].pages[b] == [pa.idx.caps["pages"] + APM.IDLE_PAGES * (b % U)]
    ids = pool.allocs[0].pages[0]
    assert ids == [pa.idx.caps["pages"]]  # idle page of user 0
    pool.grow(0, 1, 128)  # the decode loop's look-ahead: still idle (2 pages)
    assert pool.allocs[0].pages[0] == [pa.idx.caps["pages"], pa.idx.caps["pages"] + 1]
    pool.release(0)
    pa.bind_new(0, "r0", 3000)
    pool.admit(0, 3001)
    pg = pool.allocs[0].pages[0]
    assert len(pg) == 24 and max(pg) < pa.idx.caps["pages"] and len(set(pg)) == 24
    pa.idx.check()
    pool.release(0)
    pa.idx.check()
    assert pa.idx.stats()["rows"][0]["pages_free"] == pa.idx.caps["pages"]


def test_miss_commit_then_hit_tables(pa):
    t = toks(1, 5000)
    prefill(pa, 0, "A", t)
    # chunks 1..9 committed: key slots saved in order, halo A saved at every boundary, halo B only in the zone
    saves = [c for c in pa.ks.calls if c[0] == "set_save"]
    assert len(saves) == 9
    assert [c[1][0][0] for c in saves] == list(range(9))  # user 0, key slots 0..8 (free list order)
    assert all(c[1][0][1:] == [None, None, None] for c in saves)
    # zone p0 = (5000-2560)//512*512 = 2048: halo B at the ends of the chunks starting >= 2048 (ends 2560..4608)
    hb = [c for c in pa.hs.calls if c[0] == "set_save" and c[1][0] == "B"]
    assert len(hb) == 5
    ends_with_b = 512 * np.arange(1, 10)[[i for i in range(9) if i * 512 >= 2048]]
    assert list(ends_with_b) == [2560, 3072, 3584, 4096, 4608]
    pa.idx.release("A") if False else pa.model.pool.release(0)
    pa.idx.check()
    # a second request with the same prefix and 2000 more tokens
    t2 = torch.cat([t, toks(2, 2000)])
    plan = pa.idx.plan_hit("B", t2.numpy(), t2.numel())
    assert plan.hit_tokens == 4608 and plan.user in (0, 1)
    pa.hs.calls.clear()
    pa.ks.calls.clear()
    st, hits = prefill(pa, plan.user, "B", t2, st=plan.hit_tokens)
    assert st == 4608 and plan.user in hits
    ld = [c for c in pa.ks.calls if c[0] == "set_load"][0][1][0]
    assert ld[plan.user] == list(plan.key_slots) and len(ld[plan.user]) == 9
    ra = [c for c in pa.hs.calls if c[0] == "set_restore" and c[1][0] == "A"][0][1][1]
    assert ra[plan.user] == plan.halo_a
    # the table of the hit user starts with the shared pages
    pg = pa.model.pool.allocs[plan.user // U].pages[plan.user]
    assert tuple(pg[: len(plan.page_ids)]) == plan.page_ids and len(pg) == -(-(t2.numel() + 1) // 128)
    pa.idx.check()


def test_shared_pages_refcounted(pa):
    t = toks(3, 4096 + 512)
    prefill(pa, 0, "A", t)
    pa.model.pool.release(0)
    t2 = torch.cat([t, toks(4, 600)])
    users = []
    for i, rid in enumerate(["B", "C"]):
        plan = pa.idx.plan_hit(rid, t2.numpy(), t2.numel())
        assert plan.hit_tokens == 4608, plan
        users.append((rid, plan))
    assert users[0][1].user != users[1][1].user
    shared = set(users[0][1].page_ids)
    assert shared == set(users[1][1].page_ids)
    for rid, plan in users:
        prefill(pa, plan.user, rid, t2, st=plan.hit_tokens)
    pa.idx.check()
    pa.model.pool.release(users[0][1].user)
    pa.idx.check()
    free_after_one = pa.idx.stats()["rows"][users[0][1].row]["pages_free"]
    pa.model.pool.release(users[1][1].user)
    pa.idx.check()
    # the shared pages are cached (not free) after both users left; only their private pages came back
    assert pa.idx.stats()["rows"][users[0][1].row]["pages_free"] > free_after_one
    assert shared.isdisjoint(set(pa.idx._rows[users[0][1].row].free_pages))


def test_lost_reservation_recomputes(pa):
    t = toks(5, 3000)
    prefill(pa, 0, "A", t)
    pa.model.pool.release(0)
    t2 = torch.cat([t, toks(6, 500)])
    plan = pa.idx.plan_hit("B", t2.numpy(), t2.numel())
    assert plan.hit_tokens == 2560
    pa.idx.reset()  # reset_prefix_cache between the scheduler's answer and the prefill
    st, hits = prefill(pa, plan.user, "B", t2, st=plan.hit_tokens)
    assert st == 0 and not hits  # recompute from 0, counted, the request still gets a binding
    assert pa.rid[plan.user] == "B"
    pa.idx.check()


def test_continuation_is_no_hit(pa):
    t = toks(7, 3000)
    items, hits = pa.prepare_hits([(1, t, 0, 1024)], {1: "X"}, {1: 3000}, rep=True)
    pa.model.pool.release(1)
    pa.bind_new(1, "X", 3000)
    pa.model.pool.admit(1, 1025)
    items, hits = pa.prepare_hits([(1, t, 1024, 2048)], {1: "X"}, {1: 3000}, rep=True)
    assert items[0][2] == 1024 and not hits  # same request id, already bound: the model's own continuation
    pa.idx.check()


def test_anonymous_and_image_never_commit(pa):
    t = toks(8, 2048)
    pa.model.pool.release(0)
    pa.model.pool.admit(0, 2049)  # no bind_new: an anonymous request (warm-up, compaction, legacy callers)
    assert 0 in pa.auto
    pa.commit([(0, t, 512, 2048)])
    assert pa.ks.calls == []
    pa.model.pool.release(0)
    pa.model.pool.release(1)
    pa.bind_new(1, "IMG", 2048, nocommit=True)
    pa.model.pool.admit(1, 2049)
    pa.commit([(1, t, 512, 2048)])
    assert pa.ks.calls == [] and pa.idx.stats()["commits"] == 0


def test_rep_needs_total_for_a_hit(pa):
    t = toks(9, 3000)
    prefill(pa, 0, "A", t)
    pa.model.pool.release(0)
    t2 = torch.cat([t, toks(10, 500)])
    plan = pa.idx.plan_hit("B", t2.numpy(), t2.numel())
    items, hits = pa.prepare_hits([(plan.user, t2, plan.hit_tokens, t2.numel())], {plan.user: "B"}, {}, rep=True)
    assert (
        items[0][2] == 0 and not hits
    )  # bounded replay without the total length: the zone of the hit is unknown -> recompute
    pa.idx.check()


def test_slot_table_dodges_reserved_users():
    s = SlotTable(8, 8)
    s.live.update({0, 2})
    # user 1 is reserved for a hit: a new request in logical slot 1 must not take it
    p = s.claim([1], avoid={1})[0]
    assert p != 1 and p not in (0, 2)
    p2 = s.claim_packed(3, {}, 2, avoid={1})
    assert p2 != 1
    p3 = s.claim_balanced(4, {}, 2, avoid={1})
    assert p3 != 1
    # the reserved user is still free for the hit
    assert 1 not in s.live


def test_pool_pages_total_unchanged(pa):
    """every page of the row is either free, cached, held by a request or idle: a long mixed run keeps the invariant"""
    rng = np.random.default_rng(11)
    base = toks(12, 3000)
    for i in range(30):
        t = torch.cat([base[: int(rng.integers(1, 6)) * 512], toks(100 + i, int(rng.integers(600, 2000)))])
        plan = pa.idx.plan_hit(f"R{i}", t.numpy(), t.numel())
        user = plan.user if plan.hit_tokens else None
        if user is None:
            free = [u for u in range(ROWS * U) if u not in pa.rid and u not in pa.idx.reserved_all()]
            if not free:
                for u in list(pa.rid):
                    pa.model.pool.release(u)
                free = [0]
            user = free[0]
        prefill(pa, user, f"R{i}", t, st=plan.hit_tokens)
        pa.idx.check()
        if i % 3 == 0:
            for u in list(pa.rid):
                pa.model.pool.release(u)
            pa.idx.check()
