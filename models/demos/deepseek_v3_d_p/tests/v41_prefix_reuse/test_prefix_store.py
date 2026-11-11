# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill-side prefix reuse H2 (the host-RAM chunk store) decisions and memory, host only (no device, no ttnn): the
``ChunkStore`` / ``SlotArena`` of ``models/demos/common/prefill/prefix_reuse.py`` and ``PrefixRetention`` with a store. GLM-5.3's
G53F-0311 phase-2 suite (``tests/test_prefix_reuse_multi.py``) ported to V4.1's export semantics:

    python -m pytest -c /dev/null --noconftest -p no:cacheprovider models/demos/deepseek_v3_d_p/tests/v41_prefix_reuse

``StoreDevice`` is ``test_prefix_retention.FakeDevice`` (rows that depend on the chunk partition and on the carries a chunk
starts from; V4.1's export: a cold chunk 0 clears it, every chunk writes its full width and the carry at its end into one block
of a two-block window ring) plus the H2 callbacks: a computed full chunk's rows are read into the store, a restore writes them
back into the model's rows AND the export's (and, at chunk 0, clears the export first, as a cold chunk 0 does), a spill reads a
checkpoint slot, a load writes the live carries. Every scenario compares the WHOLE export and the model's rows with a cold prefill.
"""

from __future__ import annotations

import random

import pytest
import test_prefix_retention as T  # noqa: E402 -- the H1 fake device (same directory)
import torch

from models.demos.common.prefill import prefix_reuse as PRM
from models.demos.common.prefill.prefix_reuse import PAGE, ChunkStore, SlotArena, packed_bytes

C = T.C
toks, chunks_of, check_equal = T.toks, T.chunks_of, T.check_equal


def mkstore(chunks: int, tips: int, **kw) -> ChunkStore:
    """A store whose arenas hold ``chunks`` chunk records and ``tips`` tip records of this fake (one page each)."""
    return ChunkStore((chunks + tips) * PAGE, tip_budget_bytes=tips * PAGE, log=lambda m: None, **kw)


class StoreDevice(T.FakeDevice):
    def retention(self, chunk, n_ckpts, logs=None, store=None, spill_tips=2, **kw):
        base = super().retention(chunk, n_ckpts, logs=logs)
        self.spills = 0
        self.restores = []

        def restore_rows(parts, ids, start):
            (rows,) = parts
            if start == 0:
                self.zero_slot()  # a restore at chunk 0 first clears the export, as a cold chunk 0 does
            for i, v in enumerate(rows.tolist()):
                self.rows[start + i] = v
                self.slab[start + i] = v  # the export's rows, repacked from the restored ones
            self.restores.append(start)

        def spill(slot):
            self.spills += 1
            return [torch.tensor([self.slots[slot]], dtype=torch.int64)]

        def load(parts):
            self.carries = int(parts[0][0])

        return PRM.PrefixRetention(
            chunk=chunk,
            n_ckpts=n_ckpts,
            save=base._save,
            restore=base._restore,
            export_carries=base._export,
            set_next_start=base._set_next_start,
            compute_silent=base._compute_silent,
            on_resume=base._on_resume,
            log=(logs.append if logs is not None else (lambda m: None)),
            skip_partial_last=False,
            store=mkstore(64, 64) if store is None else store,
            restore_rows=restore_rows,
            spill=spill,
            load_carries=load,
            spill_tips=spill_tips,
            **kw,
        )


def run(dev, pr, tokens, *, marks=True, cut=None):
    """The runtime's prefill_chunk for every chunk of one prompt, H2 included (rows into the store after a computed chunk,
    spills after the burst's last chunk); returns S (skipped: resident), R (restored), C (computed) per chunk."""
    did = []
    parts = chunks_of(len(tokens), C, cut)
    for k, (s, e) in enumerate(parts):
        ids = tokens[s:e]
        last = (k == len(parts) - 1) if marks else None
        if pr.begin_chunk(torch.tensor(ids), s, e, last):
            did.append("R" if pr.last_restored else "S")
        else:
            if s == 0:
                dev.zero_slot()
            dev.prefill(ids, s)
            pr.end_compute(torch.tensor(ids), s, e)
            dev.export(s, e)
            if pr.wants_rows(s, e):
                pr.store_rows(torch.tensor(ids), s, e, [torch.tensor([dev.rows[p] for p in range(s, e)])])
            did.append("C")
        if last:
            pr.flush_spills()
    return did


def setup(n_ckpts=8, store=None, spill_tips=2, **kw):
    dev = StoreDevice(n_ckpts)
    logs = []
    return dev, dev.retention(C, n_ckpts, logs=logs, store=store, spill_tips=spill_tips, **kw), logs


# ------------------------------------------------------------------------------------------------- the arena
def test_arena_packs_and_unpacks_parts_bit_exactly():
    a = SlotArena(3, 1000)
    assert a.slot_bytes == PAGE and a.size_bytes == 3 * PAGE
    parts = [
        torch.arange(7, dtype=torch.int64),
        torch.randn(3, 5).to(torch.bfloat16),
        torch.tensor([1, -2], dtype=torch.int32),
    ]
    k = a.alloc()
    layout = a.pack(k, parts)
    out = a.unpack(k, layout)
    assert all(x.dtype == y.dtype and torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(parts, out))
    assert packed_bytes(parts) <= a.slot_bytes and a.resident_bytes == PAGE and a.used == 1
    with pytest.raises(AssertionError):  # allow-pytest.raises: no ttnn conftest
        a.pack(a.alloc(), [torch.zeros(PAGE + 1, dtype=torch.uint8)])


def test_arena_release_hands_pages_back_and_slots_are_reused():
    a = SlotArena(2, PAGE)
    k = a.alloc()
    a.pack(k, [torch.full((PAGE // 8,), 7, dtype=torch.int64)])
    assert a.put_back(k, release=True) == PAGE and a.releases == 1 and a.resident_bytes == 0
    k2 = a.alloc()
    assert k2 == k and int(a.unpack(k2, ((0, (PAGE // 8,), torch.int64),))[0].sum()) == 0  # DONTNEED: zero pages again
    assert a.release(k2) == 0  # never written since: nothing to release


# ------------------------------------------------------------------------------------------------- the store
def _ids(n=C, v=0):
    return torch.full((n,), v, dtype=torch.int32)


def _part(v=1):
    return [torch.full((C,), v, dtype=torch.int64)]


def test_store_evicts_lru_leaves_first_and_keeps_shared_prefixes():
    st = mkstore(4, 0)
    st.put(b"s0", b"seed", 0, C, _ids(), _part(), stamp=1)  # a shared first chunk
    st.put(b"a1", b"s0", C, 2 * C, _ids(), _part(), stamp=1)
    st.put(b"b1", b"s0", C, 2 * C, _ids(), _part(), stamp=2)
    st.put(b"b2", b"b1", 2 * C, 3 * C, _ids(), _part(), stamp=2)
    assert len(st.entries) == 4 and st.capacity == 4
    st.touch(b"s0", 3)
    st.put(b"c1", b"s0", C, 2 * C, _ids(), _part(), stamp=3)  # full: the oldest leaf (a1) goes, s0 (shared) stays
    assert set(st.entries) == {b"s0", b"b1", b"b2", b"c1"}
    st.put(b"c2", b"c1", 2 * C, 3 * C, _ids(), _part(), stamp=3)  # next: b2 (leaf of the older conversation), then b1
    assert set(st.entries) == {b"s0", b"b1", b"c1", b"c2"}
    st.put(b"c3", b"c2", 3 * C, 4 * C, _ids(), _part(), stamp=3)
    assert set(st.entries) == {b"s0", b"c1", b"c2", b"c3"}
    assert st.stats.evicted == 3 and st.bytes == 4 * PAGE and st.resident_bytes <= st.budget


def test_store_never_evicts_the_prompt_in_flight_it_refuses_instead():
    st = mkstore(1, 0)
    assert st.put(b"x0", b"seed", 0, C, _ids(), _part(), stamp=5)
    assert not st.put(b"x1", b"x0", C, 2 * C, _ids(), _part(), stamp=5)  # full, and x0 belongs to the prompt in flight
    assert set(st.entries) == {b"x0"} and st.stats.refused == 1 and st.resident_bytes == PAGE


def test_children_counts_stay_exact_when_a_parent_comes_back():
    """A child stored while its parent was evicted, then the parent stored again: the parent counts the child (no negative
    count that would pin it forever; GLM's put did not count existing children)."""
    st = mkstore(3, 0)
    st.put(b"p", b"seed", 0, C, _ids(), _part(), stamp=1)
    st.put(b"c", b"p", C, 2 * C, _ids(), _part(), stamp=2)
    st.put(b"x", b"seed", 0, C, _ids(v=9), _part(), stamp=3)
    del st.entries[b"p"]  # as if evicted while c was not yet a child (orphan)
    st.arena.put_back(0)
    st.entries[b"c"].stamp = 4
    st.put(b"p", b"seed", 0, C, _ids(), _part(), stamp=5)
    assert st.entries[b"p"].children == 1
    st.put(b"y", b"seed", 0, C, _ids(v=8), _part(), stamp=6)  # evicts x (the oldest leaf); p is not a leaf
    st.put(b"z", b"seed", 0, C, _ids(v=7), _part(), stamp=7)  # evicts c, then p becomes a leaf
    assert all(e.children >= 0 for e in st.entries.values())
    st.put(b"w", b"seed", 0, C, _ids(v=6), _part(), stamp=8)
    assert b"p" not in st.entries and all(e.children >= 0 for e in st.entries.values())


def test_tip_arena_is_separate_and_drops_the_oldest_tip_when_full():
    st = mkstore(4, 2)
    for k in range(4):
        st.put(bytes([k]), b"seed", 0, C, _ids(v=k), _part(k), stamp=k + 1)
    assert st.attach_carries(bytes([0]), _part(10), stamp=10)
    assert st.attach_carries(bytes([1]), _part(11), stamp=10)
    assert not st.attach_carries(bytes([1]), _part(11), stamp=10)  # already has carries
    st.touch(bytes([0]), 9)
    assert st.attach_carries(
        bytes([2]), _part(12), stamp=10
    )  # the tip arena is full: the oldest holder's (1) is dropped
    assert st.entries[bytes([1])].tip is None and st.entries[bytes([0])].has_carries and st.stats.tip_drops == 1
    assert int(st.carries(st.entries[bytes([2])])[0][0]) == 12 and len(st.entries) == 4  # rows kept
    assert not st.attach_carries(bytes([3]), _part(13), stamp=1)  # every holder is newer than the caller: refused
    assert st.stats.tips_refused == 1


def test_a_budget_under_one_slot_stores_nothing():
    st = mkstore(0, 0)
    assert not st.put(b"x", b"seed", 0, C, _ids(), _part(), stamp=1) and st.no_chunks and st.arena is None


def test_low_host_memory_releases_pages_by_the_deficit_and_trims(monkeypatch):
    """The host-memory floor: while MemAvailable is under the reserve, the store first hands back the pages of free slots,
    then evicts LRU leaves (pages released) until the deficit is covered, and trims; with enough memory it keeps all."""
    calls = []

    class FakeLibc:
        def malloc_trim(self, pad):
            calls.append(pad)
            return 1

    monkeypatch.setattr(PRM, "_LIBC", FakeLibc())
    avail = [100 * PAGE]
    st = mkstore(10, 0, reserve_bytes=50 * PAGE, available=lambda: avail[0])
    for k in range(6):
        st.put(bytes([k]), b"seed", 0, C, _ids(v=k), _part(k), stamp=k + 1)
    assert len(st.entries) == 6 and st.stats.evicted == 0 and calls == []
    avail[0] = 47 * PAGE  # 3 pages short of the floor
    st.put(bytes([9]), b"seed", 0, C, _ids(v=9), _part(9), stamp=10)
    assert st.stats.pressure_evictions == 3 and st.stats.released_bytes == 3 * PAGE and calls == [0]
    assert bytes([9]) in st.entries and len(st.entries) == 4 and st.resident_bytes == 4 * PAGE
    avail[0] = None  # unreadable meminfo: the arena alone bounds it
    st.put(bytes([10]), b"seed", 0, C, _ids(v=10), _part(10), stamp=11)
    assert len(st.entries) == 5 and st.stats.pressure_evictions == 3


def test_reads_trim_the_heap_every_trim_every_bytes(monkeypatch):
    calls = []

    class FakeLibc:
        def malloc_trim(self, pad):
            calls.append(pad)
            return 1

    monkeypatch.setattr(PRM, "_LIBC", FakeLibc())
    monkeypatch.setattr(PRM, "TRIM_EVERY_BYTES", 3 * C * 8)
    st = mkstore(10, 0)
    for k in range(7):
        st.put(bytes([k]), b"seed", 0, C, _ids(v=k), _part(k), stamp=k + 1)  # 64 B read each
    assert calls == [0, 0] and st.stats.trims == 2 and st.read_since_trim == C * 8


def test_resident_size_is_bounded_by_the_arenas_whatever_the_history():
    r = random.Random(5)
    st = mkstore(6, 2)
    for k in range(400):
        h = bytes([k % 251, k // 251])
        st.put(h, b"seed" if r.random() < 0.3 else bytes([r.randrange(251), 0]), 0, C, _ids(v=k), _part(k), stamp=k)
        if r.random() < 0.3:
            st.attach_carries(h, _part(-k), stamp=k)
        assert st.resident_bytes <= st.budget and st.bytes <= st.budget and len(st.entries) <= 6
    assert st.stats.evicted > 100


# ------------------------------------------------------------------------------------------------------- scenarios
def test_interleaved_conversations_resume_from_the_store():
    """A1, B1, A2, B2 (the AgentX case): the next turn of each conversation finds its chunks in the store although the
    device holds the other one, restores them and resumes at its tip on the carries spilled when the device left it."""
    dev, pr, _ = setup()
    A1 = toks(4 * C + 3, 40)
    B1 = toks(3 * C + 5, 41)
    A2 = A1 + toks(2 * C, 42)
    B2 = B1 + toks(C + 7, 43)
    for P in (A1, B1):
        assert run(dev, pr, P) == ["C"] * len(chunks_of(len(P), C))
        check_equal(dev, P, C)
    did = run(dev, pr, A2)
    assert did == ["R"] * 4 + ["C"] * 3, did  # A1's 4 full chunks restored; resume at 4C on the spilled carries
    check_equal(dev, A2, C)
    did = run(dev, pr, B2)
    assert did == ["R"] * 3 + ["C"] * 3, did  # 5C + 4 tokens: B1's 3 full chunks restored, the rest computed
    check_equal(dev, B2, C)
    assert pr.stats.silent_chunks == 0 and dev.spills >= 2 and pr.stats.host_carries_loaded >= 2


def test_exact_repeat_restores_the_full_chunks_and_computes_the_partial_last():
    dev, pr, _ = setup()
    A1 = toks(3 * C + 5, 44)
    run(dev, pr, A1)
    run(dev, pr, toks(2 * C, 45))
    did = run(dev, pr, A1)
    assert did == ["R", "R", "R", "C"], did  # V4.1: a partial last chunk is computed, resumed on the tip at 3C
    check_equal(dev, A1, C)


def test_strict_prefix_on_a_boundary_restores_its_last_chunk_with_the_stored_carries():
    """A prompt ending ON a stored boundary: its last chunk is restored only if the store holds the carries there (the
    export's window ring at the burst's end comes from them), and the export past it is cleared; otherwise computed."""
    dev, pr, _ = setup(spill_tips=2)
    A1 = toks(3 * C + 1, 46)
    run(dev, pr, A1)
    run(dev, pr, toks(C, 47))  # leaves A1: its 2 newest boundaries' (3C, 2C) carries are spilled
    did = run(dev, pr, A1[: 2 * C])
    assert did == ["R", "R"], did
    check_equal(dev, A1[: 2 * C], C)
    dev2, pr2, _ = setup(spill_tips=0)
    run(dev2, pr2, A1)
    run(dev2, pr2, toks(C, 47))
    did = run(dev2, pr2, A1[: 2 * C])
    assert did[-1] == "C", did
    check_equal(dev2, A1[: 2 * C], C)


def test_shared_system_prompt_across_conversations():
    dev, pr, _ = setup()
    sysp = toks(2 * C + 3, 48)
    A, B, D = sysp + toks(2 * C, 49), sysp + toks(C, 50), toks(3 * C, 51)
    run(dev, pr, A)
    run(dev, pr, D)  # an unrelated prompt overwrites the device
    did = run(dev, pr, B)
    assert did[:2] == ["R", "R"], did  # the system prompt's 2 full chunks come from the store
    check_equal(dev, B, C)


def test_without_spilled_tips_the_carries_are_rebuilt_silently():
    dev, pr, logs = setup(spill_tips=0)
    A1 = toks(3 * C + 1, 52)
    run(dev, pr, A1)
    run(dev, pr, toks(2 * C, 53))
    A2 = A1 + toks(C, 54)
    did = run(dev, pr, A2)
    assert did[:3] == ["R"] * 3 and pr.stats.silent_chunks == 3, (did, pr.stats.silent_chunks)
    check_equal(dev, A2, C)


def test_unmarked_headers_spill_at_once_and_stay_correct():
    dev, pr, _ = setup()
    A1 = toks(3 * C + 2, 55)
    run(dev, pr, A1, marks=False)
    run(dev, pr, toks(2 * C, 56), marks=False)
    assert dev.spills >= 1  # read when the device left A1, not deferred
    A2 = A1 + toks(C, 57)
    did = run(
        dev, pr, A2, marks=False
    )  # an unmarked chunk may be the last: restored only where its end carries are stored
    check_equal(dev, A2, C)
    # the tips are at 3C and 2C, not at C: chunk 0 cannot be restored, and a computed chunk ends the match (GLM's rule)
    assert did == ["C"] * 5, did


def test_a_restore_at_chunk_0_clears_the_export_like_a_cold_prompt():
    """Unmarked headers cannot clear the export past a restored last chunk; a restore at chunk 0 first clears it (the
    cold chunk 0's zero), so an older, longer prompt's rows never survive."""
    dev, pr, _ = setup()
    A1 = toks(2 * C, 58)
    run(dev, pr, A1, marks=False)
    run(dev, pr, toks(5 * C, 59), marks=False)  # a longer prompt: its rows reach 5C
    assert run(dev, pr, A1, marks=False) == ["R", "R"]
    check_equal(dev, A1, C, whole_export=True)


def test_budget_evicts_the_least_recent_conversation():
    dev, pr, _ = setup(store=mkstore(8, 8))
    convs = [toks(4 * C + 1, 60 + k) for k in range(3)]
    for P in convs:
        run(dev, pr, P)  # 4 full chunks each: the first conversation's are evicted to fit the third's
    assert pr.store.stats.evicted >= 4
    P0 = convs[0] + toks(C, 70)
    did = run(dev, pr, P0)
    check_equal(dev, P0, C)
    assert did[0] == "C", did  # evicted: computed again
    P2 = convs[2] + toks(C, 71)
    did = run(dev, pr, P2)
    check_equal(dev, P2, C)


def test_colliding_hash_is_caught_by_the_ids(monkeypatch):
    """Equal chain hashes with different ids must not restore the wrong rows."""
    monkeypatch.setattr(PRM, "chain_hash", lambda parent, ids: b"same")
    dev, pr, _ = setup()
    run(dev, pr, toks(2 * C, 72))
    run(dev, pr, toks(C, 73))
    Q = toks(2 * C, 74)
    did = run(dev, pr, Q)
    check_equal(dev, Q, C)
    assert "R" not in did, did


def test_same_chunk_ids_under_another_prefix_are_not_reused():
    """Rows depend on everything before them: a chunk is stored under the hash of ids [0, its end)."""
    dev, pr, _ = setup()
    a, b, x, y = toks(C, 75), toks(C, 76), toks(C, 77), toks(C, 78)
    run(dev, pr, a + x + toks(3, 79))
    run(dev, pr, y + b + toks(3, 80))
    run(dev, pr, toks(C, 81))
    P = a + b + toks(3, 82)
    did = run(dev, pr, P)
    assert did == ["R", "C", "C"], did
    check_equal(dev, P, C)


def test_mutation_chain_hash_without_the_prefix_is_caught(monkeypatch):
    """The parent hash is what makes a stored chunk's rows valid only under its own prefix: hashing the chunk's ids alone
    restores rows computed after another prefix (the per-entry id check only sees the chunk itself)."""
    real = PRM.chain_hash
    monkeypatch.setattr(PRM, "chain_hash", lambda parent, ids: real(b"no-parent", ids))
    dev, pr, _ = setup()
    a, b, x, y = toks(C, 75), toks(C, 76), toks(C, 77), toks(C, 78)
    run(dev, pr, a + x + toks(3, 79))
    run(dev, pr, y + b + toks(3, 80))
    run(dev, pr, toks(C, 81))
    P = a + b + toks(3, 82)
    assert run(dev, pr, P)[:2] == ["R", "R"]
    with pytest.raises(AssertionError):  # allow-pytest.raises: no ttnn conftest
        check_equal(dev, P, C)


def test_mutation_restore_at_chunk_0_without_the_clear_is_caught():
    dev = StoreDevice(8)
    pr = dev.retention(C, 8)
    real = pr._restore_rows

    def no_clear(parts, ids, start):
        z, dev.zero_slot = dev.zero_slot, lambda: None
        try:
            real(parts, ids, start)
        finally:
            dev.zero_slot = z

    pr._restore_rows = no_clear
    A1 = toks(2 * C, 58)
    run(dev, pr, A1, marks=False)
    run(dev, pr, toks(5 * C, 59), marks=False)
    run(dev, pr, A1, marks=False)
    with pytest.raises(AssertionError):  # allow-pytest.raises: no ttnn conftest
        check_equal(dev, A1, C, whole_export=True)


def test_mutation_carries_not_loaded_is_caught():
    dev = StoreDevice(8)
    pr = dev.retention(C, 8)
    pr._load_carries = lambda parts: None
    A1 = toks(3 * C + 1, 40)
    run(dev, pr, A1)
    run(dev, pr, toks(2 * C, 41))
    A2 = A1 + toks(C, 42)
    assert run(dev, pr, A2)[:3] == ["R"] * 3
    with pytest.raises(AssertionError):  # allow-pytest.raises: no ttnn conftest
        check_equal(dev, A2, C)


def test_off_is_h1():
    dev = T.FakeDevice(8)
    pr = dev.retention(C, 8)
    assert pr.store is None and not pr.wants_rows(0, C) and pr.flush_spills() == 0


@pytest.mark.parametrize("seed", range(60))
def test_fuzz_interleaved_conversations_equal_cold(seed):
    """Several conversations interleaved (next turns, siblings sharing a prefix, rewritten ends, truncations, cuts, empty
    chunks), random checkpoint pools, store sizes (chunk and tip arenas), spill counts and engines with / without the header
    mark: the export (whole, marked; the shipped rows, unmarked or cut) and the model's rows always equal a cold prefill's.
    """
    r = random.Random(seed)
    n_ckpts = r.choice([0, 1, 2, 3, 5, 8])
    store = mkstore(r.choice([0, 3, 8, 30, 200]), r.choice([0, 1, 4, 64]))
    dev, pr, _ = setup(n_ckpts=n_ckpts, store=store, spill_tips=r.choice([0, 1, 2, 3]))
    sysp = toks(r.randrange(1, 2 * C), 2000 + seed)
    convs = {
        k: (sysp if r.random() < 0.5 else []) + toks(r.randrange(1, 3 * C), 3000 + 10 * seed + k) for k in range(4)
    }
    for _ in range(30):
        k = r.randrange(4)
        P = convs[k]
        roll = r.random()
        if roll < 0.6:
            P = P[: max(1, len(P) - r.randrange(0, C // 2))] + toks(r.randrange(0, 2 * C), r.random())  # next turn
        elif roll < 0.75:
            P = P[: r.randrange(1, len(P) + 1)]  # a shorter prompt (a retry, a sibling)
        convs[k] = P
        if r.random() < 0.1:
            pr.note_empty()
        marks = r.random() < 0.75
        cut = r.randrange(1, len(chunks_of(len(P), C)) + 1) if r.random() < 0.1 else None
        run(dev, pr, P, marks=marks, cut=cut)
        check_equal(dev, P, C, cut, whole_export=marks and cut is None)
        assert store.resident_bytes <= store.budget
    assert pr.stats.computed_chunks > 0
