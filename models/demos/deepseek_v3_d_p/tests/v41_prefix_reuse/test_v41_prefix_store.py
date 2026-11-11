# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""V4.1 prefill-side prefix reuse H2 (the host-RAM chunk store, ``V41_PREFILL_PREFIX_STORE_GB``) end to end on a HOST fake device
(``fake_v41.py``; no chips, no real ttnn): the real ``V41PrefillRuntime`` chunk path, the real ``kv_export.py`` (all three export
layouts) and ``tt/v41/prefix_reuse.py`` with a store.

    python -m pytest -c /dev/null --noconftest -p no:cacheprovider models/demos/deepseek_v3_d_p/tests/v41_prefix_reuse

* a prompt whose chunks the device no longer holds (it moved to another prompt: the AgentX case) restores them from the store
  and resumes on the spilled tip carries: the WHOLE export (7 tensors), the model's rows (entries, keys, the path-A slab), the
  Engram history and -- when its last chunk was computed -- the counters and carries equal a cold prefill's;
* exact repeats, a strict prefix ending on a stored tip (restored as the burst's last chunk), a sibling sharing chunk 0 (no
  tip there: rebuilt silently), unmarked headers, two slots, an image prompt (never stored), a failed restore;
* capacity: the store's slots, LRU eviction, the in-flight prompt's refusals, the resident size within the budget; the floor;
* H2 off issues H1's device work (``test_v41_prefix_reuse.test_store_off_issues_the_h1_commits_device_work``), and an extra
  device call with the store off is caught by it;
* each piece of a restore is needed: removing it makes a scenario fail (mutations).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytest  # noqa: E402
import test_v41_prefix_reuse as T  # noqa: E402 -- installs the fake ttnn first (fake_v41), then the H1 harness
import torch  # noqa: E402

F, PR, RT, KX, CFG = T.F, T.PR, T.RT, T.KX, T.CFG
C, MAXSEQ, MARKED, LAST, LAYOUTS = T.C, T.MAXSEQ, T.MARKED, T.LAST, T.LAYOUTS
toks, X, A, A2, AP, LONG = T.toks, T.X, T.A, T.A2, T.AP, T.LONG
U = toks(2 * C + 128, 33)  # an unrelated prompt: moves the device away from the conversation
SIB = A[: C + 256] + toks(2 * C - 128, 3)  # shares A's chunk 0 (and 256 tokens of chunk 1): no tip at C
assert not hasattr(sys.modules["ttnn"], "open_mesh_device"), "the REAL ttnn got imported"

# the fake's records (int64 rows): a chunk = per source C / ratio rows x (512 + 128); the tip = 21 carries [128, 512]
CHUNK_REC = sum((C // (2 if L in (2, 8, 14) else 1)) * (512 + 128) * 8 for L in (2, 8, 14, 20))
TIP_REC = 21 * 128 * 512 * 8


def _gb(nbytes: int) -> str:
    return repr(nbytes / 2**30)  # the record sizes are page multiples: exact in binary


def store_env(layout, *, chunks=40, tips=16, spill=2, ckpts=8, reserve="0", flag=True) -> dict:
    env = T._base_env(layout, flag=flag, ckpts=ckpts)
    env.update(
        {
            PR.ENV_STORE_GB: _gb(chunks * CHUNK_REC + tips * TIP_REC),
            PR.ENV_STORE_TIP_GB: _gb(tips * TIP_REC),
            PR.ENV_RESERVE_GB: reserve,
            PR.ENV_SPILL_TIPS: str(spill),
        }
    )
    return env


class StoreSystem(T.System):
    """The H1 harness's runner, reporting R for a chunk restored from the host store (S = skipped, C = computed)."""

    def prompt(self, tokens, *, slot=0, marks=True, cut=None) -> list:
        S = len(tokens)
        parts = [(s, min(S, s + C)) for s in range(0, S, C)]
        parts = parts[:cut] if cut else parts
        did = []
        for k, (s, e) in enumerate(parts):
            word = slot | ((MARKED | (LAST if k == len(parts) - 1 else 0)) if marks else 0)
            n0, a0 = len(F.SYS.trace), len(self.acks)
            self.rt.prefill_chunk(
                list(tokens[s:e]) + [1] * (C - (e - s)),
                self.kv,
                slot_id=word,
                actual_start=s,
                actual_end=e,
                request_id=self.request,
            )
            self.request += 1
            assert len(self.acks) - a0 == CFG.n_layers, "every chunk acks every contract layer"
            n = sum(1 for op in F.SYS.trace[n0:] if op[0] == "pf._chunk")
            if n == 0:
                did.append("R" if self.rt._prefix is not None and self.rt._prefix.last_restored else "S")
            else:
                did.append("C" if n == 1 else f"C+{n - 1}silent")
        self.last_did = did
        return did

    def store(self, slot=0):
        return self.rt._prefix.stores[slot]


_REFS: dict = {}


def cold(layout, tokens, *, users=1, slot=0):
    """Today's runner (flag off): an unrelated prompt, then ``tokens``; (export, live state, rows, Engram history)."""
    key = (layout, tuple(tokens), users, slot)
    if key not in _REFS:
        with T._isolated_sys(), T._env(**T._base_env(layout, flag=False)):
            s = T.System(users=users)
            s.prompt(X, slot=slot)
            s.prompt(tokens, slot=slot)
            S = len(tokens)
            _REFS[key] = (
                F.export_bytes(s.kv),
                F.model_state(s.rt.pf, slot, S),
                F.model_state(s.rt.pf, slot, S, live=False),
                s.rt.engram.cache[slot][:S].clone(),
            )
    return _REFS[key]


def check(sys_, layout, tokens, *, users=1, slot=0, what="", whole_export=True):
    ref_exp, ref_live, ref_rows, ref_eng = cold(layout, tokens, users=users, slot=slot)
    S = len(tokens)
    exp = F.export_bytes(sys_.kv)
    bad = [k for k in ref_exp if not F.same(exp[k], ref_exp[k])]
    assert not bad, f"{what}: export tensors {bad} differ from a cold prefill's"
    assert F.same(F.model_state(sys_.rt.pf, slot, S, live=False), ref_rows), f"{what}: model rows (or the slab) differ"
    assert torch.equal(sys_.rt.engram.cache[slot][:S], ref_eng), f"{what}: the Engram history differs"
    if sys_.last_did and sys_.last_did[-1][0] == "C":  # the live state is the prompt's end only when its last chunk ran
        assert F.same(F.model_state(sys_.rt.pf, slot, S), ref_live), f"{what}: counters / carries differ"


# ------------------------------------------------------------------------------------------------------- scenarios
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_store_sequence_equals_cold(layout):
    with T._env(**store_env(layout)):
        s = StoreSystem(flag=True)
        assert s.store().capacity == 0  # the arena is made at the first stored chunk
        s.prompt(X)
        assert s.prompt(A) == ["C"] * 4
        check(s, layout, A, what="A computed")
        assert s.store().capacity == 40 and len(s.store().entries) == 5  # X's 2 + A's 3 full chunks
        assert s.prompt(U) == ["C"] * 3  # the device leaves A: A's tips (3C, 2C) are spilled after U's last chunk
        assert s.prompt(A2) == [
            "R",
            "R",
            "R",
            "C",
            "C",
            "C",
        ]  # A's full chunks from the store, resumed on A's tip at 3C
        check(s, layout, A2, what="A2 after U (store restore + host tip carries)")
        assert s.prompt(U) == ["R", "R", "C"]
        check(s, layout, U, what="U again (restored, its partial last chunk computed)")
        assert s.prompt(A) == ["R", "R", "R", "C"]  # an exact repeat after another prompt
        check(s, layout, A, what="A again after U")
        s.prompt(U)
        assert s.prompt(AP) == ["R", "R"]  # a strict prefix ending on A's stored tip at 2C: its last chunk restored
        check(s, layout, AP, what="AP: restored last chunk (windows from the stored carries, export cleared past 2C)")
        s.prompt(U)
        assert s.prompt(SIB) == ["R", "C+1silent", "C", "C"]  # chunk 0 restored; no tip at C: rebuilt from 0
        check(s, layout, SIB, what="sibling sharing chunk 0 after U")
        assert F.SYS.late_allocs == [], f"zero blocks built after the island capture: {F.SYS.late_allocs}"
        pr = s.rt._prefix.ret[0]
        assert pr.stats.restored_chunks == 15 and pr.stats.host_carries_loaded >= 4 and pr.stats.spills >= 4
        assert s.store().resident_bytes <= s.store().budget


def test_device_held_prefix_then_store_restore_then_compute():
    """H1 skips and H2 restores in one prompt: the device holds A2[:3C] (A sent last), the store holds A2's later chunks."""
    layout = "cp2"
    with T._env(**store_env(layout)):
        s = StoreSystem(flag=True)
        s.prompt(X)
        s.prompt(A2)
        s.prompt(
            A
        )  # the device now holds A (chunks 0..2 skipped, its partial chunk computed): A2's chunks 3..5 only in the store
        P = A2 + toks(C + 256, 34)
        assert s.prompt(P) == ["S", "S", "S", "R", "R", "R", "C", "C"]
        check(s, layout, P, what="skip [0, 3C), restore [3C, 6C), compute on A2's tip at 6C")


def test_unmarked_headers_restore_where_end_carries_are_stored():
    """Without the header mark every chunk may be the burst's last, so it is restored only with its END carries stored:
    spilling every boundary (V41_PREFILL_PREFIX_SPILL_TIPS=8) makes each of A's full chunks restorable."""
    layout = "cp2"
    with T._env(**store_env(layout, spill=8)):
        s = StoreSystem(flag=True)
        s.prompt(X, marks=False)
        s.prompt(A, marks=False)
        s.prompt(U, marks=False)  # unmarked: the tips are read at once (no last chunk to wait for)
        assert s.prompt(A2, marks=False) == ["R", "R", "R", "C", "C", "C"]
        check(s, layout, A2, what="A2 unmarked")
        s.prompt(LONG, marks=False)  # a longer prompt: its rows reach 8C
        assert s.prompt(A[: 2 * C], marks=False) == ["R", "R"]
        check(s, layout, A[: 2 * C], what="A[:2C] unmarked after LONG (the chunk-0 restore cleared LONG's rows)")


def test_two_slots_have_separate_stores():
    layout = "cp2"
    with T._env(**store_env(layout)):
        s = StoreSystem(flag=True, users=2)
        s.prompt(A, slot=0)
        s.prompt(U, slot=1)
        s.prompt(U, slot=0)
        assert s.prompt(A2, slot=0) == ["R", "R", "R", "C", "C", "C"]
        assert s.prompt(U, slot=1) == ["S", "S", "C"]  # slot 1 still holds U (H1)
        exp, st = F.export_bytes(s.kv), F.model_state(s.rt.pf, 0, len(A2))
        assert len(s.store(1).entries) == 2 and s.store(0) is not s.store(1)
    with T._isolated_sys(), T._env(**T._base_env(layout, flag=False)):
        r = T.System(users=2)
        r.prompt(A, slot=0)
        r.prompt(U, slot=1)
        r.prompt(U, slot=0)
        r.prompt(A2, slot=0)
        r.prompt(U, slot=1)
        ref_exp, ref_st = F.export_bytes(r.kv), F.model_state(r.rt.pf, 0, len(A2))
    assert all(F.same(exp[k], ref_exp[k]) for k in ref_exp), "two-slot export differs from today's runner"
    assert F.same(st, ref_st)


def test_capacity_lru_eviction_and_in_flight_refusals():
    """5 chunk slots: three 2-chunk conversations evict the oldest's chunks (leaves first); a 7-chunk prompt fills the store
    with its own chunks and is refused the rest (its chunks are never evicted while it runs); results stay cold-equal.
    """
    layout = "cp2"
    P = [toks(2 * C + 128, 40 + k) for k in range(3)]
    with T._env(**store_env(layout, chunks=5, tips=4)):
        s = StoreSystem(flag=True)
        for p in P:
            s.prompt(p)
        st = s.store()
        assert st.capacity == 5 and len(st.entries) == 5 and st.stats.evicted == 1
        assert s.prompt(P[0]) == ["R", "C", "C"]  # its LRU leaf (chunk 1) was evicted, chunk 0 survived
        check(s, layout, P[0], what="P0 after eviction")
        assert s.prompt(P[2]) == ["R", "R", "C"]
        check(s, layout, P[2], what="P2 (most recent) restored")
        s.prompt(LONG)
        assert st.stats.refused >= 2 and len(st.entries) == 5 and st.resident_bytes <= st.budget
        assert s.prompt(LONG) == ["S"] * 7 + ["C"]  # the device holds it (H1)
        check(s, layout, LONG, what="LONG again")


def test_floor_releases_pages_and_stays_correct():
    """MemAvailable under V41_PREFILL_PREFIX_HOST_RESERVE_GB: the store evicts and hands slot pages back by the deficit."""
    layout = "cp2"
    with T._env(**store_env(layout, reserve="1")):
        s = StoreSystem(flag=True)
        st = s.store()
        assert st.reserve == 2**30
        avail = [2**31]
        st.available = lambda: avail[0]
        for p in (X, A, U):
            s.prompt(p)
        assert st.stats.pressure_evictions == 0 and len(st.entries) == 7
        avail[0] = 2**30 - 2 * CHUNK_REC  # two chunk records short of the floor
        s.prompt(LONG)
        assert st.stats.pressure_evictions >= 2 and st.stats.released_bytes >= 2 * CHUNK_REC
        avail[0] = 2**31
        did = s.prompt(A2)
        check(s, layout, A2, what="A2 after pressure evictions")
        assert st.resident_bytes <= st.budget, did


IMG = A[:C] + [CFG.vocab_size + 3] * 128 + A[C + 128 :]  # an image span (virtual ids) in chunk 1; chunk 0 is A's text


def test_image_chunks_are_never_stored_or_restored():
    """Only the text chunk before the span is stored (its rows depend on its own ids); the image chunk and everything after
    it is computed every time, resumed first after a restored chunk 0 (no tip there: rebuilt silently)."""
    layout = "cp2"
    with T._env(**store_env(layout)):
        s = StoreSystem(flag=True)
        assert s.prompt(IMG) == ["C"] * 4
        assert len(s.store().entries) == 1  # chunk 0 (text)
        s.prompt(U)
        assert s.prompt(IMG) == ["R", "C+1silent", "C", "C"]
        check(s, layout, IMG, what="image prompt again")
        assert len(s.store().entries) == 3  # + U's 2: nothing of the image prompt's chunks 1.. was stored
        assert s.prompt(A) == ["R", "C+1silent", "C", "C"]  # nothing of the image prompt was retained on the device
        check(s, layout, A, what="A after the image prompt")


@pytest.mark.parametrize("store", [False, True])
def test_image_chunk_after_a_reused_prefix_is_resumed_first(store, tmp_path):
    """A, then an image prompt sharing A's chunk 0: the chunk-0 skip (H1) left the model at A's end; the image chunk must be
    resumed at C before it is computed. a7830ff1d74 tripped the runtime's ``kv_actual == actual_start`` assert here."""
    layout = "cp2"
    env = store_env(layout) if store else T._base_env(layout, flag=True)
    with T._env(**env):
        s = StoreSystem(flag=True)
        s.prompt(A)
        assert s.prompt(IMG) == ["S", "C", "C", "C"]  # resumed at C from the checkpoint there
        check(s, layout, IMG, what="the image prompt after A")
        # H1: nothing of the image prompt retained (chunk 0 recomputed); H2: A's chunks from the store, resumed on A's tip at
        # 3C (spilled when the image chunk moved the device off A)
        assert s.prompt(A) == (["R", "R", "R", "C"] if store else ["C"] * 4)
        check(s, layout, A, what="A after the image prompt")
    if store:
        return
    rt0, kx0 = T._base_modules(tmp_path, T.H1_COMMIT, prefix=True)
    with T._isolated_sys(), T._env(**env):
        b = T.System(rt0, kx0, flag=True)
        b.prompt(A)
        with pytest.raises(AssertionError):  # allow-pytest.raises: no ttnn conftest
            b.prompt(IMG)  # the H1 commit: pf.kv_actual (A's end) != the image chunk's start


def test_failed_restore_invalidates_the_device_retention_not_the_store(monkeypatch):
    layout = "cp2"
    with T._env(**store_env(layout)):
        s = StoreSystem(flag=True)
        s.prompt(A)
        s.prompt(U)
        real = PR.export_entry_rows
        calls = [0]

        def flaky(*a, **k):
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError("device fault")
            return real(*a, **k)

        monkeypatch.setattr(PR, "export_entry_rows", flaky)
        with pytest.raises(RuntimeError):  # allow-pytest.raises: no ttnn conftest
            s.prompt(A2)
        assert s.rt._prefix.ret[0].length == 0 and len(s.store().entries) > 0
        monkeypatch.setattr(PR, "export_entry_rows", real)
        assert s.prompt(A2) == ["R", "R", "R", "C", "C", "C"]
        check(s, layout, A2, what="A2 after a failed restore")


# ---------------------------------------------------------------------------------------------- off == H1 (sanity)
def test_an_extra_device_call_with_the_store_off_is_caught(tmp_path, monkeypatch):
    """The identity test is not vacuous: one synchronize per computed chunk with the store off makes it fail."""
    real = PR.V41PrefixReuse.capture

    def noisy(self, *a, **k):
        F.TTNN.synchronize_device(self.rt.mesh_device)
        return real(self, *a, **k)

    monkeypatch.setattr(PR.V41PrefixReuse, "capture", noisy)
    with pytest.raises(AssertionError, match="first difference"):  # allow-pytest.raises: no ttnn conftest
        T.test_store_off_issues_the_h1_commits_device_work(tmp_path, "cp2", 1, True)


# ------------------------------------------------------------------------------------------------------- mutations
def _no_slab(fn):
    def wrapped(pf):
        return [(L, attn, st, ratio, False) for L, attn, st, ratio, _blk in fn(pf)]

    return wrapped


def _no_zero_on_restore(real):
    def wrapped(self, slot, parts, ids, start):
        z = self.exp.zero_slot
        self.exp.zero_slot = lambda s: None
        try:
            return real(self, slot, parts, ids, start)
        finally:
            self.exp.zero_slot = z

    return wrapped


MUTATIONS = {
    "export rows not repacked on restore": (PR, "export_entry_rows", lambda real: (lambda *a, **k: None)),
    "path-A slab not written on restore": (PR, "entry_sources", _no_slab),
    "Engram history not advanced on restore": (F.FakeEngram, "hash_ids", lambda real: (lambda self, *a, **k: None)),
    "host carries not loaded": (PR.V41PrefixReuse, "_load_carries", lambda real: (lambda self, *a, **k: None)),
    "no export clear at a chunk-0 restore": (PR.V41PrefixReuse, "_restore_rows", _no_zero_on_restore),
}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_mutation_is_caught(name, monkeypatch):
    owner, attr, make = MUTATIONS[name]
    monkeypatch.setattr(owner, attr, make(getattr(owner, attr)))
    layout = "cp2"
    failures = []
    with T._env(**store_env(layout, spill=8)):
        s = StoreSystem(flag=True)
        for p, marks in ((X, True), (A, True), (U, True), (A2, True), (U, True), (A, True), (U, True), (AP, True),
                         (LONG, False), (U, False), (A[: 2 * C], False)):  # fmt: skip
            s.prompt(p, marks=marks)
            try:
                check(s, layout, p, what=f"{len(p)} tokens")
            except AssertionError as e:
                failures.append(str(e)[:100])
    assert failures, f"mutation '{name}' went unnoticed"
