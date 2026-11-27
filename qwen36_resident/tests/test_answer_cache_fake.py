# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of Strata S4-1 (vllm_rows RESIDENT_ANSWER_CACHE): a finished request leaves a prefix-cache entry
through its answer — its state snapshotted as it crossed each boundary (plain launches; inside a verify's commit for a
spec user: the row that holds it), the KV read from the pool at the finish past the entry its KV came from — and the
next turn, whose prompt carries the answer, restores through it.

The fakes (test_vllm_rows_fake's FakeModel; a fake PP with load_prefix / run_from; test_adaptive_fake's spec build with
the verify rows' ring / slot bookkeeping) hold each slot's consumed tokens as its "state": a boundary state taken a step
early or late, an entry whose KV is not its tokens', a restore into the wrong blocks, all fail their checks. Small
blocks (16 tokens) keep the sequences short."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm_tt_plugin.spec_decode")

import qwen36_resident.tests.test_adaptive_fake as AF
import qwen36_resident.tests.test_vllm_rows_fake as F
from qwen36_resident import vllm_adaptive as A
from qwen36_resident import vllm_rows as VR
from qwen36_resident.prefix_cache import PrefixCache
from qwen36_resident.vllm_rows import BLOCK, CHUNK_UNIT

BLK = 16  # (the prefix cache's / answer cache's block in these tests)
V = F.V


class AnswerModel(F.FakeModel):
    """FakeModel + snapshot_state / entry_from"""

    entries = 0

    def snapshot_state(self, s):
        h = self.hist[s]
        assert h is not None and self.conv_ok[s], f"state snapshot of slot {s} without a state"
        return dict(hist=list(h), kv=None, n_tokens=len(h), device=True)

    def entry_from(self, state_host, table, n_tokens, parent=None):
        assert state_host["host"], "entry from a host state"
        h = list(state_host["hist"])
        assert len(h) == n_tokens, f"the boundary state consumed {len(h)} tokens, the entry is at {n_tokens}"
        kv, start = [], 0
        if parent is not None and 0 < parent["n_tokens"] <= n_tokens:
            pn = parent["n_tokens"]
            assert list(parent["hist"]) == h[:pn], "the parent entry is not a prefix of the state"
            kv, start = list(parent["kv"]), pn
            self.parent_tokens = getattr(self, "parent_tokens", 0) + pn
        owners = set()
        for q in range(start, n_tokens):
            got = self.pool.get((table[q // BLOCK], q % BLOCK))
            assert got is not None and got[1] == h[q], f"KV at {q} through the entry's table is {got}, want {h[q]}"
            owners.add(got[0])
            kv.append(h[q])
        assert len(owners) <= 1, f"the entry's KV pages belong to several slots {owners}"
        AnswerModel.entries += 1
        self.read_tokens = getattr(self, "read_tokens", 0) + n_tokens - start
        return dict(hist=h, kv=kv, n_tokens=n_tokens, host=True)


class PrefixPP(F.FakePP):
    """FakePP + PPPrefill.load_prefix / run_from (a prefix-cache hit resumes the pipeline from the slot's restored
    state); a handoff may come at any length (a boundary on the way)"""

    def load_prefix(self, model, h, slot=None, block_table=None):
        m = self.model
        hist = m.hist[slot]
        assert hist is not None and len(hist) == h, f"load_prefix({h}) but slot {slot} consumed {len(hist or [])}"
        for q in range(h):
            got = m.pool.get((block_table[q // BLOCK], q % BLOCK))
            assert got == (slot, hist[q]), f"load_prefix: KV at {q} is {got}, want {(slot, hist[q])}"
        self.ids, self.done, self.chunk, self.prompt_spans = list(hist), False, None, 0

    def run_from(self, ids, h, chunk=None, drain=True):
        ids = [int(t) for t in ids]
        assert len(self.ids) == h and ids[:h] == self.ids, "run_from past a prefix the pipeline does not hold"
        self.ids, self.chunk, self.done = ids, chunk, drain

    def handoff(self, model, n_tokens, block_table=None, slot=None):
        assert self.done and len(self.ids) == n_tokens, (self.done, len(self.ids), n_tokens)
        assert isinstance(model, F.Contract) and model._obj is self.model, "handoff into another model"
        m = self.model
        m.hist[slot], m.conv_ok[slot] = list(self.ids), True
        m.tables[slot] = list(block_table)
        for q in range(n_tokens):
            m.pool[(block_table[q // BLOCK], q % BLOCK)] = (slot, self.ids[q])
        self.handoffs.append((n_tokens, self.chunk, self.prompt_spans))


class RingSpec(AF.FakeSpec):
    """the fake spec session with the verify rows' (ring, slot) bookkeeping of ResidentDFlashRows / SpecRowsSession
    (start: ring 1, slot 0; a verify after a commit of a: ring + 1 + a, slot + a) and the KV its commits write to the
    pool (the spec build shares the plain build's pool)"""

    def __init__(self, ae):
        super().__init__()
        U = A.SPEC_USERS
        self.ae, self.last, self.committed, self.base = ae, [None] * U, [1] * U, [None] * U

    def start_many(self, items, prefilled=()):
        out = super().start_many(items, prefilled)
        for u, _, _ in items:
            self.ae.spec.ring[u], self.ae.spec.slot[u], self.last[u] = 1, 0, None
        return out

    def verify(self, tokens, positions, active):
        e = self.ae.spec
        for u, a in enumerate(active):
            if a and not self.started[u]:
                acc = self.committed[u] - 1
                e.ring[u], e.slot[u] = e.ring[u] + 1 + acc, (e.slot[u] + acc) % 4
        g = super().verify(tokens, positions, active)
        for u, a in enumerate(active):
            if a:
                self.last[u], self.base[u] = int(positions[u]), list(self.seq[u])
        return g

    def propose(self, anchors, positions, accepted, active):
        for u, a in enumerate(active):
            if a and not self.started[u] and self.pending[u] is not None:
                c = int(accepted[u])
                self.committed[u] = c
                j, table, pool = self.ae.owner[u], self.ae.spec_tables[u], self.ae._pf.pool
                n = len(self.seq[u])
                for k, t in enumerate(self.pending[u][:c]):
                    pool[(table[(n + k) // BLOCK], (n + k) % BLOCK)] = (j, int(t))
        return super().propose(anchors, positions, accepted, active)


class RingSpecModel(AF.FakeSpecModel):
    def __init__(self, spec, e):
        super().__init__(spec)
        self.e, self.snaps = e, []

    def vrows_snapshot(self, u, ring, slot):
        s, e = self.spec, self.e
        if s.started[u]:
            assert (ring, slot) == (e.ring[u], e.slot[u]), ((ring, slot), (e.ring[u], e.slot[u]))
            hist = list(s.seq[u])
        else:
            j = ring - e.ring[u] - 1
            assert 0 <= j < 4 and slot == (e.slot[u] + j) % 4, (u, ring, slot, e.ring[u], e.slot[u])
            hist = s.base[u] + [int(t) for t in s.pending[u][: j + 1]]
        self.snaps.append((u, len(hist)))
        return dict(hist=hist, kv=None, n_tokens=len(hist), device=True)


def make(monkeypatch, spec=False):
    monkeypatch.setattr(VR, "ANSWER_BLOCK", BLK)
    monkeypatch.setattr(A, "ANSWER_BLOCK", BLK)
    monkeypatch.setattr(VR, "PREFIX_CACHE_STEP", 0)
    m, pf0 = AF.make(monkeypatch)
    pf = AnswerModel(AF.B, sample=True)
    pf.launch_rows = AF.B
    pf.row_order = lambda: list(range(AF.B))
    pp = PrefixPP(CHUNK_UNIT, pf)
    view = F.Contract(pf, F.SWAP_API + ("launch_rows", "snapshot_state", "entry_from"))
    pp_view = F.Contract(pp, F.PP_API + ("load_prefix", "run_from"))
    m.ae = AF.FakeAE(pf, view, pp)
    m.ae.owner, m.ae.R = {}, 4
    if spec:
        e = m.ae.spec
        e.ring, e.slot = [0] * A.SPEC_USERS, [0] * A.SPEC_USERS
        m.ae.sess = RingSpec(m.ae)
        e.m = RingSpecModel(m.ae.sess, e)
        old = m.ae.move_to_spec

        def move_to_spec(j, u, table, tokens):
            old(j, u, table, tokens)
            e.ring[u], e.slot[u], m.ae.owner[u] = 0, 0, j
            m.ae.spec_tables[u] = list(table)

        m.ae.move_to_spec = move_to_spec
    m.sess = m.ae.sess
    m.ae.plain_tables = m.tables
    m.engine = A._PlainRowsEngine(view, pp_view, torch.eye(V), V, V, AF.B, CHUNK_UNIT, None)
    m.engine.prefix_cache = PrefixCache(1 << 40, block=BLK)
    m.engine.answer = True
    return m, pf


def _turn(run, row, prompt, steps, plain=False):
    run.arrive([row], [prompt])
    if plain:  # (an idle server's prefill sets SPEC: plain users only from here, PLAIN_HOLD keeps it)
        run.m.mode = "PLAIN"
    for _ in range(steps):
        run.step()
    run.check()
    return list(run.req[row]["tokens"])


def test_plain_answer_entry(monkeypatch):
    """plain launches: boundary states at 48 and 64 as the answer decodes; the finish's entry at 64 takes the prompt's
    entry (32) KV pieces and reads only 32 .. 64 from the pool; the next turn hits 64 and decodes the greedy chain"""
    m, pf = make(monkeypatch)
    monkeypatch.setattr(A, "PLAIN_HOLD", 10**6)
    run = AF.Runner(m)
    prompt = [(7 * i + 3) % V for i in range(40)]
    toks = _turn(run, 0, prompt, 30, plain=True)
    assert not m.spec_of
    eng, pc = m.engine, m.engine.prefix_cache
    assert pc.peek(prompt + [0]) == 32, "the prompt's boundary stored at its prefill"
    assert eng.cross[m.slots.res_of[0]][0] == 64 and eng.answer_stats["crossings"] == 2, eng.answer_stats
    run.finish(0)
    assert eng.answer_stats["entries"] == 1 and pf.read_tokens == 32 and pf.parent_tokens == 32, eng.answer_stats
    assert not eng.cross and not eng.entry_of
    # the next turn: the answer carried back + a question
    turn2 = toks[:-1] + [(5 * i + 1) % V for i in range(9)]
    restores = pf.restores
    _turn(run, 0, turn2, 6, plain=True)
    assert pf.restores == restores + 1 and pc.stats["hit_tokens"] >= 64, pc.stats
    r = m.slots.res_of[0]
    assert eng.entry_of[r][0] == 64, "the slot's KV is the answer entry's below 64 (the next finish's parent)"


@pytest.mark.parametrize("prompt_len", [40, 47], ids=["boundary-in-a-commit", "near-the-prompt"])
def test_spec_answer_entry(monkeypatch, prompt_len):
    """spec users: a boundary inside a verify's commit is snapshotted from the row that holds it (the fake checks the
    row's (ring, slot) and that its state consumed exactly the boundary's tokens), the entry and the next turn's hit
    as for plain users"""
    m, pf = make(monkeypatch, spec=True)
    monkeypatch.setattr(A, "PAY_MIN", 10**6)
    rings = []
    m.ae.spec.ring_part = lambda u, hb: rings.append((u, hb)) or dict(s0=max(0, hb - 2048) // 64 * 64, hb=hb, ring=[])
    run = AF.Runner(m)
    prompt = [(11 * i + 5) % V for i in range(prompt_len)]
    toks = _turn(run, 0, prompt, 24)
    r = m.slots.res_of[0]
    assert r in m.spec_of, "promoted into the spec build"
    eng = m.engine
    snaps = m.ae.spec.m.snaps
    assert snaps and all(n % BLK == 0 for _, n in snaps), snaps
    B = eng.cross[r][0]
    assert B == (len(toks) - 2) // BLK * BLK or B == (len(toks) - 1) // BLK * BLK, (B, len(toks))
    run.finish(0)
    assert eng.answer_stats["entries"] == 1, eng.answer_stats
    # (S4-1b: a boundary crossed at a propose keeps the drafter ring read after that propose's extension)
    assert rings and rings[-1][1] == B, (rings, B)
    _, entry = eng.prefix_cache.lookup(toks[: B + 1])
    assert [p["hb"] for p in entry["taps"]] == [B], entry.get("taps")
    turn2 = toks[:-1] + [(3 * i + 2) % V for i in range(7)]
    restores = pf.restores
    _turn(run, 0, turn2, 8)
    assert pf.restores == restores + 1 and eng.prefix_cache.stats["hit_tokens"] >= B, eng.prefix_cache.stats


def test_no_entry_on_preemption_or_without_a_boundary(monkeypatch):
    """a preempted request leaves no answer entry (its swap store entry restores it); an answer that crossed no
    boundary past the prompt's entry leaves none either"""
    m, pf = make(monkeypatch)
    monkeypatch.setattr(A, "PLAIN_HOLD", 10**6)
    run = AF.Runner(m)
    toks = _turn(run, 0, [(7 * i + 3) % V for i in range(40)], 30, plain=True)
    run.req.pop(0)
    m.preempt_request(0, token_ids=toks)
    assert m.engine.answer_stats["entries"] == 0 and not m.engine.cross
    _turn(run, 0, [(2 * i + 9) % V for i in range(35)], 5, plain=True)  # (35 .. 40: no boundary past 32)
    run.finish(0)
    assert m.engine.answer_stats["entries"] == 0
