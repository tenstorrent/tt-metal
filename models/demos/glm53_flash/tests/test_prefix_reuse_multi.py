# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt-blaze G53F-0311 phase 2: several retained prompts (host chunk store), no device, no ttnn.

    pytest --noconftest models/demos/glm53_flash/tests/test_prefix_reuse_multi.py

The fake device of ``test_prefix_reuse.py`` (rows and carries that depend on the chunk partition and on the carries a
chunk starts from) plus a host store: a computed full chunk's rows are captured as the export reads them, a restore
writes them back into the caches AND the slab, a spill reads a checkpoint slot, a load writes the live carries. Every
scenario compares what the KV Manager would ship and the model's own rows with a cold prefill on a fresh device.
"""

from __future__ import annotations

import random

import pytest
import torch

from models.demos.glm53_flash.tests.test_prefix_reuse import C, FakeDevice, check_equal, chunks_of, toks
from models.demos.glm53_flash.tt.runners import prefix_reuse
from models.demos.glm53_flash.tt.runners.prefix_reuse import ChunkStore, PrefixRetention


class StoreDevice(FakeDevice):
    def retention(self, chunk, n_ckpts, logs=None, budget=10**9, spill_tips=2):
        base = super().retention(chunk, n_ckpts, logs=logs)
        self.spills = 0

        def restore_rows(rows, start):  # rows are chunk-relative, as the runtime's tensors are: start places them
            for i, v in enumerate(rows):
                self.rows[start + i] = v
                self.slab[start + i] = v

        def spill(slot):
            self.spills += 1
            return self.slots[slot], 1

        def load(c):
            self.carries = c

        return PrefixRetention(
            chunk=chunk,
            n_ckpts=n_ckpts,
            save=base._save,
            restore=base._restore,
            export_carries=base._export,
            set_next_start=base._set_next_start,
            compute_silent=base._compute_silent,
            log=(logs.append if logs is not None else (lambda m: None)),
            store=ChunkStore(budget, log=lambda m: None),
            restore_rows=restore_rows,
            spill=spill,
            load_carries=load,
            spill_tips=spill_tips,
        )


def run(dev, pr, tokens, *, marks=True, cut=None):
    """The runtime's prefill_chunk for every chunk of one prompt, phase 2 included (rows to the store after the export,
    spills after a marked last chunk); returns S (skipped: resident), R (restored), C (computed) per chunk."""
    did = []
    parts = chunks_of(len(tokens), C, cut)
    for k, (s, e) in enumerate(parts):
        ids = tokens[s:e]
        last = (k == len(parts) - 1) if marks else None
        n_restored = pr.stats.restored_chunks
        if pr.begin_chunk(torch.tensor(ids), s, e, last):
            did.append("R" if pr.stats.restored_chunks > n_restored else "S")
        else:
            dev.prefill(ids, s)
            pr.end_compute(torch.tensor(ids), s, e)
            dev.export(s, e)
            if pr.wants_rows(s, e):
                pr.store_rows(torch.tensor(ids), s, e, [dev.rows[p] for p in range(s, e)], 1)
            did.append("C")
        if last:
            pr.flush_spills()
    return did


def setup(n_ckpts=8, budget=10**9, spill_tips=2):
    dev = StoreDevice(n_ckpts)
    logs = []
    return dev, dev.retention(C, n_ckpts, logs=logs, budget=budget, spill_tips=spill_tips), logs


# ------------------------------------------------------------------------------------------------- the store itself
def test_chain_hash_depends_on_the_whole_prefix():
    a = torch.tensor(toks(C, 1))
    b = torch.tensor(toks(C, 2))
    h1 = prefix_reuse.chain_hash(prefix_reuse.CHAIN_SEED, a)
    assert prefix_reuse.chain_hash(h1, b) != prefix_reuse.chain_hash(
        prefix_reuse.chain_hash(prefix_reuse.CHAIN_SEED, b), b
    )
    assert prefix_reuse.chain_hash(prefix_reuse.CHAIN_SEED, a) == h1


def test_store_evicts_lru_leaves_first_and_keeps_shared_prefixes():
    st = ChunkStore(4)
    t = torch.zeros(C, dtype=torch.int32)
    st.put(b"s0", b"seed", 0, C, t, "s0", 1, stamp=1)  # a shared first chunk
    st.put(b"a1", b"s0", C, 2 * C, t, "a1", 1, stamp=1)
    st.put(b"b1", b"s0", C, 2 * C, t, "b1", 1, stamp=2)
    st.put(b"b2", b"b1", 2 * C, 3 * C, t, "b2", 1, stamp=2)
    assert len(st.entries) == 4
    st.touch(b"s0", 3)
    st.put(b"c1", b"s0", C, 2 * C, t, "c1", 1, stamp=3)  # over budget: the oldest leaf (a1) goes, s0 (shared) stays
    assert set(st.entries) == {b"s0", b"b1", b"b2", b"c1"}
    st.put(b"c2", b"c1", 2 * C, 3 * C, t, "c2", 1, stamp=3)  # next: b2 (leaf of the older conversation), then b1
    assert set(st.entries) == {b"s0", b"b1", b"c1", b"c2"}
    st.put(b"c3", b"c2", 3 * C, 4 * C, t, "c3", 1, stamp=3)
    assert set(st.entries) == {b"s0", b"c1", b"c2", b"c3"}


def test_store_never_evicts_the_prompt_in_flight():
    st = ChunkStore(1)
    t = torch.zeros(C, dtype=torch.int32)
    st.put(b"x0", b"seed", 0, C, t, "x", 1, stamp=5)
    st.put(b"x1", b"x0", C, 2 * C, t, "x", 1, stamp=5)
    assert set(st.entries) == {b"x0", b"x1"}  # over budget, but both belong to stamp 5 (the prompt in flight)


# ------------------------------------------------------------------------------------------------------- scenarios
def test_interleaved_conversations_resume_from_the_store():
    """A1, B1, A2, B2: the next turn of each conversation finds its chunks in the store although the device holds the
    other one, restores them, and resumes at its tip with the carries spilled when the device left it."""
    dev, pr, _ = setup()
    A1 = toks(4 * C + 3, 40)
    B1 = toks(3 * C + 5, 41)
    A2 = A1 + toks(2 * C, 42)
    B2 = B1 + toks(C + 7, 43)  # 5C + 4: chunks 0-2 stored (B1's full chunks), 3-5 new
    for P in (A1, B1):
        assert run(dev, pr, P) == ["C"] * len(chunks_of(len(P), C))
        check_equal(dev, P, C)
    did = run(dev, pr, A2)
    assert did == ["R"] * 4 + ["C"] * 3, did  # A1's 4 full chunks restored; resume at 4C on the spilled carries
    check_equal(dev, A2, C)
    did = run(dev, pr, B2)
    assert did == ["R"] * 3 + ["C"] * 3, did
    check_equal(dev, B2, C)
    assert pr.stats.silent_chunks == 0 and dev.spills >= 2 and pr.stats.host_carries_loaded >= 2


def test_shared_system_prompt_across_conversations():
    dev, pr, _ = setup()
    sysp = toks(2 * C + 9, 44)
    A, B, D = sysp + toks(2 * C, 45), sysp + toks(C, 46), toks(3 * C, 47)
    run(dev, pr, A)
    run(dev, pr, D)  # an unrelated prompt overwrites the device
    did = run(dev, pr, B)
    assert did[:2] == ["R", "R"], did  # the system prompt's 2 full chunks come from the store
    check_equal(dev, B, C)


def test_without_spilled_tips_the_carries_are_rebuilt():
    dev, pr, logs = setup(spill_tips=0)
    A1 = toks(3 * C + 1, 48)
    run(dev, pr, A1)
    run(dev, pr, toks(2 * C, 49))
    A2 = A1 + toks(C, 50)
    did = run(dev, pr, A2)
    assert did[:3] == ["R"] * 3 and pr.stats.silent_chunks == 3, (did, pr.stats.silent_chunks)
    check_equal(dev, A2, C)


def test_unmarked_headers_spill_at_once_and_stay_correct():
    dev, pr, _ = setup()
    A1 = toks(3 * C + 2, 51)
    run(dev, pr, A1, marks=False)
    run(dev, pr, toks(2 * C, 52), marks=False)
    assert dev.spills >= 1  # read when the device left A1, not deferred
    A2 = A1 + toks(C, 53)
    run(dev, pr, A2, marks=False)  # an unmarked chunk may be the last: restored only where its end carries are stored
    check_equal(dev, A2, C)


def test_a_restored_last_chunk_needs_its_stored_carries():
    """A prompt that ends ON a stored boundary: its last chunk is restored only if the store holds the carries there
    (the KV Manager ships the KDA window after it); otherwise it is computed."""
    dev, pr, _ = setup(spill_tips=1)
    A1 = toks(3 * C, 54)
    run(dev, pr, A1)
    run(dev, pr, toks(C, 55))  # leaves A1: its tip (3C) carries are spilled
    did = run(dev, pr, A1)
    assert did == ["R", "R", "R"], did
    check_equal(dev, A1, C)
    dev2, pr2, _ = setup(spill_tips=0)
    run(dev2, pr2, A1)
    run(dev2, pr2, toks(C, 55))
    did = run(dev2, pr2, A1)
    assert did[-1] == "C", did
    check_equal(dev2, A1, C)


def test_budget_evicts_the_least_recent_conversation():
    dev, pr, _ = setup(budget=8)  # 8 stored chunks
    convs = [toks(4 * C + 1, 60 + k) for k in range(3)]
    for P in convs:
        run(dev, pr, P)  # 4 full chunks each: the first conversation's are evicted to fit the third's
    assert pr.store.evicted >= 4
    did = run(dev, pr, convs[0] + toks(C, 70))
    check_equal(dev, convs[0] + toks(C, 70), C)
    assert did[0] == "C", did  # evicted: computed again
    did = run(dev, pr, convs[2] + toks(C, 71))
    check_equal(dev, convs[2] + toks(C, 71), C)


def test_colliding_hash_is_caught_by_the_ids(monkeypatch):
    """Equal chain hashes with different ids must not restore the wrong rows."""
    monkeypatch.setattr(prefix_reuse, "chain_hash", lambda parent, ids: b"same")
    dev, pr, _ = setup()
    run(dev, pr, toks(2 * C, 72))
    run(dev, pr, toks(C, 73))
    Q = toks(2 * C, 74)
    did = run(dev, pr, Q)
    check_equal(dev, Q, C)
    assert "R" not in did, did


def test_same_chunk_ids_under_another_prefix_are_not_reused():
    """Rows depend on everything before them: a chunk is stored under the hash of ids [0, its end), so the same chunk
    after a different first chunk is a different entry."""
    dev, pr, _ = setup()
    shared = toks(C, 75)
    X = toks(C, 76) + shared + toks(3, 77)
    Y = toks(C, 78) + shared + toks(3, 79)
    run(dev, pr, X)
    run(dev, pr, toks(C, 80))
    did = run(dev, pr, Y)
    assert did == ["C", "C", "C"], did
    check_equal(dev, Y, C)


def test_off_is_h1():
    dev = FakeDevice(8)
    pr = dev.retention(C, 8)
    assert pr.store is None and not pr.wants_rows(0, C)


@pytest.mark.parametrize("seed", range(60))
def test_fuzz_interleaved_conversations_equal_cold(seed):
    """Several conversations interleaved (next turns, siblings sharing a prefix, rewritten ends, truncations, cuts,
    empty chunks), random pool sizes, store budgets, spill counts and engines with / without the header mark: the KV
    Manager always ships what a cold prefill would."""
    r = random.Random(seed)
    n_ckpts = r.choice([0, 1, 2, 3, 5, 8])
    dev, pr, _ = setup(n_ckpts=n_ckpts, budget=r.choice([0, 3, 8, 30, 10**9]), spill_tips=r.choice([0, 1, 2, 3]))
    sysp = toks(r.randrange(1, 2 * C), 2000 + seed)
    convs = {
        k: (sysp if r.random() < 0.5 else []) + toks(r.randrange(1, 3 * C), 3000 + 10 * seed + k) for k in range(4)
    }
    restored = 0
    for step in range(30):
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
        did = run(dev, pr, P, marks=marks, cut=cut)
        restored += did.count("R")
        check_equal(dev, P, C, cut)
    assert pr.stats.computed_chunks > 0


def test_evictions_trim_the_heap_every_trim_every_bytes(monkeypatch):
    """Freed chunk rows stay resident in glibc's brk heap until malloc_trim (G53F-0256 11-09: 130 GB heap at a 64 GiB
    budget). The store trims once per TRIM_EVERY_BYTES of evicted bytes, and not before."""
    calls = []

    class FakeLibc:
        def malloc_trim(self, pad):
            calls.append(pad)
            return 1

    monkeypatch.setattr(prefix_reuse, "_LIBC", FakeLibc())
    monkeypatch.setattr(prefix_reuse, "TRIM_EVERY_BYTES", 3)
    st = ChunkStore(2)
    ids = torch.zeros(4, dtype=torch.int32)
    for k in range(6):  # each put of 1 byte past the 2-byte budget evicts one leaf (1 byte freed)
        st.put(bytes([k]), b"root", 4 * k, 4 * k + 4, ids, None, 1, stamp=k + 1)
    assert st.evicted == 4
    assert calls == [0] and st.trims == 1 and st.freed_since_trim == 1


def test_low_host_memory_shrinks_the_store_by_the_deficit_and_trims(monkeypatch):
    """The host-memory floor: while MemAvailable is under the reserve, the store evicts down by the deficit (within its
    budget) and trims at once; with enough memory it keeps everything its budget allows."""
    calls = []

    class FakeLibc:
        def malloc_trim(self, pad):
            calls.append(pad)
            return 1

    monkeypatch.setattr(prefix_reuse, "_LIBC", FakeLibc())
    avail = [100]
    st = ChunkStore(10, reserve_bytes=50, available=lambda: avail[0])
    ids = torch.zeros(4, dtype=torch.int32)
    for k in range(6):
        st.put(bytes([k]), b"root", 4 * k, 4 * k + 4, ids, None, 1, stamp=k + 1)
    assert st.bytes == 6 and st.evicted == 0 and calls == []  # under budget, plenty of memory
    avail[0] = 47  # 3 bytes short of the floor
    st.put(bytes([9]), b"root", 40, 44, ids, None, 1, stamp=10)
    assert st.bytes == 4 and st.evicted == 3 and st.pressure_evictions == 3
    assert calls == [0]  # trimmed at once so the freed pages reach MemAvailable
    assert bytes([9]) in st.entries  # the prompt in flight stays
    avail[0] = None  # unreadable meminfo: the budget alone
    st.put(bytes([10]), b"root", 44, 48, ids, None, 1, stamp=11)
    assert st.bytes == 5 and st.evicted == 3
