# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host tests of the KV store manager: no ttnn, no device.

The replay tests need the AgentX model and its recorded run (``generated/hy4_preview/agentx_model``, the export
``workload.EXPORT``); they are skipped where those are missing.

    test_recorded_hour_*   the recorded hour as it happened (request start / end times of the export): every
                           request reserves its conversation, is bound to a decode row while one is free, releases.
    test_simulated_hour    the closed loop of ``agentx_model/sim.py`` (evictions cost prefill time, which moves the
                           later arrivals) with the store in place of the simulator's token counter: the numbers
                           next to ``kv_occupancy.md``'s LRU table.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

from models.experimental.hy4_preview.fused.kv_store import frames as fr
from models.experimental.hy4_preview.fused.kv_store.manager import KvStore, PrefixIndex

F = fr.FRAME_TOKENS
MODEL = Path("generated/hy4_preview/agentx_model")
BUDGETS = [4e6, 5e6, 6e6, 8e6]


# ---------------------------------------------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------------------------------------------
def test_reserve_grow_shrink():
    s = KvStore(num_frames=10)
    assert s.reserve("a", 1) and s.conversation("a").frames == [0]
    assert s.reserve("a", F) and s.conversation("a").frames == [0]
    assert s.reserve("a", F + 1) and s.conversation("a").frames == [0, 1]
    assert s.reserve("b", 3 * F) and s.conversation("b").frames == [2, 3, 4]
    assert s.reserve("a", 4 * F) and s.conversation("a").frames == [0, 1, 5, 6]  # frames need not be a run
    s.conversation("a").cached = 4 * F
    assert s.reserve("a", F + 5) and s.conversation("a").frames == [0, 1]
    assert s.conversation("a").cached == 2 * F  # what the dropped frames held is gone
    assert s.pool.free == 5 and s.used_tokens == 5 * F
    assert s.reserve("c", 5 * F) and s.conversation("c").frames == [5, 6, 7, 8, 9]  # lowest free first
    s.check()
    assert s.stats.frames_allocated == 12 and s.stats.frames_freed == 2 and s.stats.allocations == 5


def test_lru_eviction_by_tokens():
    s = KvStore(num_frames=10)
    for key, n in (("a", 3), ("b", 2), ("c", 4)):
        assert s.begin(key, n * F) is not None
        s.end(key, n * F)
    assert s.begin("b", 2 * F, shared_prefix=2 * F).keep == 2 * F  # b is now the most recently used
    s.end("b", 2 * F)
    # d needs 4 frames, 1 is free: a (oldest) goes, that is enough (3 + 1); c and b stay
    g = s.begin("d", 4 * F)
    assert g is not None and g.keep == 0 and sorted(g.new_frames) == [0, 1, 2, 9]
    assert not s.conversation("a").frames and s.conversation("c").frames and s.conversation("b").frames
    assert s.stats.evictions == 1 and s.stats.evicted_tokens == 3 * F
    s.end("d", 4 * F)
    # a comes back with a prompt that continues its old one: all of it is prefilled again
    g = s.begin("a", 3 * F + 100, shared_prefix=3 * F)
    assert g.keep == 0 and g.restart == 0 and g.reprefill == 3 * F and s.stats.reprefill_tokens == 3 * F
    assert not s.conversation("c").frames  # c was the least recently used
    s.check()


def test_pinned_and_bound_conversations_are_not_evicted_and_nothing_is_evicted_in_vain():
    s = KvStore(num_frames=8, rows=2)
    assert s.begin("a", 3 * F) and s.begin("b", 3 * F)
    s.end("b", 3 * F)
    assert s.begin("c", 6 * F) is None  # free 2 + b's 3 < 6: wait, and b keeps its frames
    assert s.conversation("b").frames and s.stats.evictions == 0 and s.stats.waits == 1
    assert not s.conversation("c").frames and s.conversation("c").pins == 0
    s.bind(0, "b")
    assert s.begin("c", 5 * F) is None  # b is on a decode row
    s.unbind(0)
    g = s.begin("c", 5 * F)
    assert g is not None and not s.conversation("b").frames and s.stats.evictions == 1
    with pytest.raises(ValueError):  # allow-pytest.raises: host-side check
        s.reserve("d", 9 * F)
    s.check()


def test_shrinking_prompt_keeps_the_prefix():
    s = KvStore(num_frames=8)
    s.begin("a", 5 * F)
    s.end("a", 5 * F - 10)
    g = s.begin("a", 2 * F + 300, shared_prefix=2 * F - 212)  # compaction: only the start is the same
    assert g.keep == 2 * F - 212 and g.restart == (2 * F - 212) // 4096 * 4096 and g.reprefill == 0
    assert len(s.conversation("a").frames) == 3 and s.pool.free == 5
    s.end("a", 2 * F + 300 - 50, trim=True)
    assert len(s.conversation("a").frames) == 3
    s.check()


def test_tables_and_binding():
    s = KvStore(num_frames=300, rows=4)
    s.begin("a", 3 * F)
    s.begin("b", 100 * F)
    s.bind(2, "a")
    v = s.table_version
    s.bind(0, "b")
    assert s.table_version == v + 1
    lat, idx = fr.Geometry.pool(fr.LATENT), fr.Geometry.pool(fr.INDEX)
    t = s.tables()
    assert len(t["latent"]) == 4 * fr.TABLE_FRAMES and t["latent"].itemsize == 4
    a, b = s.conversation("a").frames, s.conversation("b").frames
    for row, frames in ((0, b), (2, a)):
        got = t["latent"][row * fr.TABLE_FRAMES : (row + 1) * fr.TABLE_FRAMES]
        assert list(got[: len(frames)]) == [f * 78 * 32 * 704 for f in frames]
        assert all(x == fr.UNMAPPED for x in got[len(frames) :])
        got = t["index"][row * fr.TABLE_FRAMES : (row + 1) * fr.TABLE_FRAMES]
        assert list(got[: len(frames)]) == [f * 21 * 4352 for f in frames] == [idx.frame_base(f) for f in frames]
    assert all(x == fr.UNMAPPED for x in t["latent"][fr.TABLE_FRAMES : 2 * fr.TABLE_FRAMES])
    assert lat.frame_base(299) + lat.slot_bytes(77) + 31 * 704 < 2**32
    # a bound conversation that grows changes the table; an unbound one does not
    v = s.table_version
    s.reserve("a", 4 * F)
    assert s.table_version == v + 1 and s.mapped_tokens(2) == 4 * F
    s.begin("c", F)
    assert s.table_version == v + 1
    s.unbind(2)
    assert s.frame_ids(2) == [fr.UNMAPPED] * fr.TABLE_FRAMES and s.free_row() == 1
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        s.bind(0, "a")  # the row is taken
    s.check()


def test_workbench_copy_plans():
    s = KvStore(num_frames=64)
    s.begin("x", F)  # so that a's frames do not start at 0
    s.begin("a", 5 * F)
    frames = s.conversation("a").frames
    assert frames == [1, 2, 3, 4, 5]
    assert s.copy_in("a", 0) == []
    assert s.copy_in("a", 2 * F) == [(1, 0), (2, 1)]
    assert s.copy_in("a", 2 * F + 4096) == [(1, 0), (2, 1), (3, 2)]  # the half frame before the restart point
    assert s.copy_out("a", 2 * F + 4096, 4 * F + 7) == [(2, 3), (3, 4), (4, 5)]
    assert s.copy_out("a", 0, 1) == [(0, 1)]
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        s.copy_out("a", 0, 5 * F + 1)


def test_fork_copies_frames_and_shares_none():
    s = KvStore(num_frames=16)
    s.begin("parent", 3 * F + 100)
    s.end("parent", 3 * F + 100)
    pairs = s.fork("parent", "child", 2 * F + 5000)  # the shared start, down to the 4096 grid
    child = s.conversation("child")
    assert child.cached == 2 * F + 4096 and len(child.frames) == 3
    assert pairs == list(zip(s.conversation("parent").frames[:3], child.frames))
    assert not set(child.frames) & set(s.conversation("parent").frames)
    s.check()


def test_prefix_index():
    idx = PrefixIndex()
    rng = random.Random(0)
    system = [rng.randrange(1 << 17) for _ in range(3 * 4096)]
    a = system + [rng.randrange(1 << 17) for _ in range(5 * 4096 + 77)]
    b = system + [rng.randrange(1 << 17) for _ in range(2 * 4096)]
    assert idx.match(a) == (None, 0, 0)
    idx.register("a", a)
    assert idx.match(a + [1, 2, 3]) == ("a", 8 * 4096, 8)  # the next turn of a
    assert idx.match(b) == ("a", 3 * 4096, 8)  # a fork: 3 of a's 8 blocks
    idx.register("b", b)
    assert idx.match(b + [5] * 5000) == ("b", 5 * 4096, 5)
    assert idx.match(a) == ("a", 8 * 4096, 8)  # b answers for the shared start, a still for what is its own
    idx.remove("a")
    assert idx.match(a) == ("b", 3 * 4096, 5)


def test_random_operations_keep_the_books():
    rng = random.Random(7)
    s = KvStore(num_frames=60, rows=8)
    keys = [f"c{i}" for i in range(40)]
    busy = {}
    for step in range(6000):
        op = rng.random()
        if op < 0.5:
            key = rng.choice(keys)
            if key in busy:
                continue
            conv = s.conversation(key)
            cached = conv.cached if conv else 0
            tokens = max(1, min(cached + rng.randrange(-3 * F, 4 * F), 30 * F))
            g = s.begin(key, tokens, shared_prefix=min(cached, tokens))
            if g is not None:
                busy[key] = tokens
                row = s.free_row()
                if row is not None and rng.random() < 0.8:
                    s.bind(row, key)
        elif op < 0.8 and busy:
            key = rng.choice(sorted(busy))
            if s.conversation(key).row is not None:
                s.unbind(s.conversation(key).row)
            s.end(key, busy.pop(key), trim=rng.random() < 0.3)
        elif busy:
            key = rng.choice(sorted(busy))
            want = busy[key] + rng.randrange(1, 2 * F)
            if want <= 30 * F and s.reserve(key, want):
                busy[key] = want
        s.check()
    assert s.stats.evictions > 50 and s.stats.waits > 0 and s.stats.peak_frames <= 60


# ---------------------------------------------------------------------------------------------------------------
# The recorded hour
# ---------------------------------------------------------------------------------------------------------------
def _workload():
    if not MODEL.is_dir():
        pytest.skip(f"{MODEL} is not here")
    if str(MODEL) not in sys.path:
        sys.path.insert(0, str(MODEL))
    import workload

    if not Path(workload.EXPORT).is_file():
        pytest.skip(f"{workload.EXPORT} is not here")
    return workload.load()


def replay_recorded(budget: float, rows: int = 8, check_every: int = 1):
    """The requests of the export at their recorded start and end times. A request that finds no room (everything
    else is in flight) is counted and retried at every later end; the recording itself never waits."""
    reqs, _ = _workload()
    store = KvStore.for_budget(budget, rows=rows)
    events = sorted([(q.rec_start, 1, q.rid) for q in reqs] + [(q.rec_end, 0, q.rid) for q in reqs])
    waiting, running, unbound, n = [], set(), [], 0
    peak_live = 0

    def start(q):
        g = store.begin(q.stream, q.isl + q.osl, shared_prefix=q.lcp)
        if g is None:
            return False
        running.add(q.rid)
        row = store.free_row()
        if row is None:
            unbound.append(q)
        else:
            store.bind(row, q.stream)
        return True

    for _, is_start, rid in events:
        q = reqs[rid]
        if is_start:
            if not start(q):
                waiting.append(q)
        else:
            if rid not in running:  # it never got in: its recorded end passed while it waited
                waiting.remove(q)
                continue
            running.discard(rid)
            row = store.conversation(q.stream).row
            if row is not None:
                store.unbind(row)
            else:
                unbound.remove(q)
            store.end(q.stream, q.isl)
            waiting[:] = [w for w in waiting if not start(w)]
            while unbound and store.free_row() is not None:
                store.bind(store.free_row(), unbound.pop(0).stream)
        n += 1
        if n % check_every == 0:
            store.check()
            assert store.used_tokens <= budget
        peak_live = max(peak_live, sum(1 for c in store.conversations.values() if c.frames))
    store.check()
    return store, peak_live, len(reqs)


@pytest.mark.parametrize("budget", BUDGETS, ids=lambda b: f"{b / 1e6:g}M")
def test_recorded_hour(budget):
    store, peak_live, n = replay_recorded(budget)
    st = store.stats
    assert st.peak_frames * F <= budget and store.used_tokens <= budget
    print(
        f"\nHY4_MK10 recorded hour, budget {budget / 1e6:g} M tokens ({store.num_frames} frames): {n} requests, "
        f"allocations {st.allocations} ({st.frames_allocated} frames), freed by shrinking "
        f"{st.frames_freed - st.evicted_frames}, evictions {st.evictions} ({st.evicted_tokens / 1e6:.2f} M valid "
        f"tokens), prefilled again {st.reprefill_tokens / 1e6:.2f} M, waits {st.waits}, peak "
        f"{st.peak_frames * F / 1e6:.2f} M tokens in {peak_live} conversations"
    )
    if budget >= 8e6:  # the export has 111 conversations, never told to end: the old ones go, none comes back
        assert st.reprefill_tokens == 0 and st.waits == 0


# ---------------------------------------------------------------------------------------------------------------
# The simulated hour (closed loop), next to kv_occupancy.md
# ---------------------------------------------------------------------------------------------------------------
def simulate(budget, clients: int = 16, frame_tokens: int = F):
    """agentx_model/sim.py with the store deciding what fits and what is evicted. Returns (result, store)."""
    reqs, trees = _workload()
    import engines
    import sim

    fit = json.loads((MODEL / "prefill_fit_traced.json").read_text())

    class StoreSim(sim.Sim):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.store = KvStore.for_budget(budget, rows=self.p.rows, on_evict=self._evicted)
            self.key_of, self.stream_of = {}, {}
            self.kept_beyond_prompt = 0

        def _key(self, st):
            if id(st) not in self.key_of:
                self.key_of[id(st)] = len(self.key_of)
                self.stream_of[self.key_of[id(st)]] = st
            return self.key_of[id(st)]

        def _evicted(self, conv):
            st = self.stream_of[conv.key]
            st.has_slot, st.cached, st.tokens = False, 0, 0
            self.evictions += 1

        def _room(self, st, need, evict=True):
            key = self._key(st)
            if not evict:  # the warm-up phase's cache: kept only if it fits
                return self.store.reserve(key, need, evict=False)
            r = self.wait_slot[0].req
            lcp = r.lcp or (st.cached if st.has_slot else 0)
            g = self.store.begin(key, need, shared_prefix=lcp)
            if g is None:
                return False
            # sim.py takes all of the cached prompt as shared when the request has no recorded prefix (lcp 0: a
            # first turn after the warm-up phase, and the three compaction turns, whose prompt may be shorter
            # than what is cached); the store cannot keep more than the frames of this request hold
            sim_keep = min(st.cached if st.has_slot else 0, lcp)
            assert g.keep == min(sim_keep, self.store.frames_for(need) * F), (g.keep, sim_keep, need, r.lcp)
            self.kept_beyond_prompt += max(sim_keep - r.isl, 0)
            return True

        def _prewarm(self, skey, tokens):
            super()._prewarm(skey, tokens)
            st = self.streams[skey]
            if st.has_slot:
                self.store.conversation(self._key(st)).cached = tokens

        def _returned(self, run):
            st = self.streams.get(run.skey)
            if st is not None and id(st) in self.key_of:
                self.store.end(self.key_of[id(st)], st.cached)
                self.store.check()
                assert self.store.used_tokens <= budget
            super()._returned(run)

    engine = engines.hy4("fused", 8, fit=fit, kv_tokens=None)
    s = StoreSim(reqs, trees, engine, clients, 3600.0)
    result = s.run()
    s.store.check()
    assert s.store.stats.evictions == result["evictions"]
    assert s.store.stats.reprefill_tokens == result["prefill_tokens_after_eviction"]
    exact = sim.simulate(reqs, trees, engines.hy4("fused", 8, fit=fit, kv_tokens=budget), clients, 3600.0)
    if s.kept_beyond_prompt:
        print(f"\nHY4_MK10 sim.py kept {s.kept_beyond_prompt} cached tokens beyond a prompt's own length (lcp 0)")
    return result, s.store, exact


@pytest.mark.parametrize("budget", BUDGETS, ids=lambda b: f"{b / 1e6:g}M")
def test_simulated_hour(budget):
    result, store, exact = simulate(budget)
    st = store.stats
    assert st.peak_frames * F <= budget
    share = result["prefill_tokens_after_eviction"] / max(result["prefill_tokens"], 1)
    print(
        f"\nHY4_MK10 simulated hour, 16 client slots, budget {budget / 1e6:g} M ({store.num_frames} frames of 8192): "
        f"{result['tokens_per_s']:.0f} tok/s, evictions {st.evictions}, prefilled again "
        f"{st.reprefill_tokens / 1e6:.2f} M ({share:.0%} of all prefill), allocations {st.allocations} "
        f"({st.frames_allocated} frames), waits {st.waits}, peak {st.peak_frames * F / 1e6:.2f} M; "
        f"exact-token LRU of kv_occupancy.md: {exact['tokens_per_s']:.0f} tok/s, evictions {exact['evictions']}, "
        f"prefilled again {exact['prefill_tokens_after_eviction'] / 1e6:.2f} M"
    )
    if budget >= 8e6:
        assert st.evictions == 0 and result["tokens_per_s"] > 230


@pytest.mark.parametrize("budget", [6e6, 8e6], ids=lambda b: f"{b / 1e6:g}M")
def test_simulated_hour_32_clients(budget):
    result, store, exact = simulate(budget, clients=32)
    st = store.stats
    print(
        f"\nHY4_MK10 simulated hour, 32 client slots, budget {budget / 1e6:g} M: {result['tokens_per_s']:.0f} tok/s, "
        f"evictions {st.evictions}, prefilled again {st.reprefill_tokens / 1e6:.2f} M, allocations {st.allocations} "
        f"({st.frames_allocated} frames), waits {st.waits}; exact-token LRU: {exact['tokens_per_s']:.0f} tok/s, "
        f"evictions {exact['evictions']}, prefilled again {exact['prefill_tokens_after_eviction'] / 1e6:.2f} M"
    )
