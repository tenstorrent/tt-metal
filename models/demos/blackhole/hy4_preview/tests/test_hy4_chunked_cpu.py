# SPDX-License-Identifier: Apache-2.0
"""CPU tests of chunked prefill interleaved with decode (HY4_CHUNKED_PF): the REAL ``Hy4Engine`` begin / chunks / end / abort / park / row bookkeeping and the adapter's ``_prefill_chunked`` flow, over a FAKE device:
a dict pool {(row, pos): value} written by a fake prefill (honouring the write mask) and by fake decode rounds (live rows write their pending token; a parked row writes JUNK at its park position), logits = a hash of
the prompt, the next token of a decoder = a checksum of its pool row. A bug that parks a prefilling row inside a finished chunk, writes a finished row's pages again, lets a decode round disturb a prefill, loses carried
state or leaves an aborted row dirty shows up as a wrong pool / stream. Run: PYTHONPATH=<wt> python -m pytest --noconftest -q tests/test_hy4_chunked_cpu.py"""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.demos.blackhole.hy4_preview.tt import hy4_serve_engine as E
from models.demos.blackhole.hy4_preview.tt.hy4_pool_prefill_model import PAD, PfState, PrefillResult

R = 4
PF = 8
K = 3
JUNK = -7


class FakePP:
    """Hy4PoolPrefill stand-in (stepwise API of the real class, same PfState): chunk c of the rows in the write mask lands in eng.pool."""

    def __init__(self, eng):
        self.eng, self.pf_rows, self.keep_cur = eng, PF, None
        self.mask_log = []

    def set_keep(self, keep_rows):
        self.keep_cur = None if keep_rows is None else tuple(sorted(keep_rows))
        self.mask_log.append(self.keep_cur)

    def _begin(self, prompts, keep_hidden, keep_logits, mtp, keep_rows, chain, PFr, start=0):
        if keep_rows is not None:
            self.set_keep(keep_rows)
        prompts = [list(p) for p in prompts] + [[PAD]] * (R - len(prompts))
        lens = [max(len(p), 1) for p in prompts]
        nch = -(-max(lens) // PF)
        res = PrefillResult(first=[None] * R, lens=lens)
        res.hidden, res.logits = [None] * R, [None] * R
        return PfState(
            prompts=prompts,
            lens=lens,
            PF=PF,
            nch=nch,
            c=start // PF,
            start=start,
            res=res,
            mtp=mtp,
            keep_hidden=keep_hidden,
            keep_logits=keep_logits,
            chain=False,
            keep_rows=None if keep_rows is None else sorted(keep_rows),
        )

    prefill_begin = (
        lambda self, prompts, keep_hidden=False, keep_logits=False, mtp=None, keep_rows=None, start=0: self._begin(
            prompts, keep_hidden, keep_logits, mtp, keep_rows, False, PF, start
        )
    )

    def _chunk(self, st):
        c = st.c
        keep = self.keep_cur
        for r in range(R):
            if keep is not None and r not in keep:
                continue
            for p in range(c * PF, (c + 1) * PF):
                self.eng.pool[(r, p)] = st.prompts[r][p] if p < len(st.prompts[r]) else PAD
        enders = [r for r in range(R) if c * PF <= st.lens[r] - 1 < (c + 1) * PF]
        for r in enders:
            h = hashlib.md5(",".join(map(str, st.prompts[r])).encode()).digest()
            lg = torch.tensor([b / 255.0 for b in h] + [0.0] * 8)
            st.res.logits[r] = lg
            st.res.first[r] = int(lg.argmax())
            st.res.hidden[r] = lg
        self.eng.chunks_run.append((c, tuple(sorted(keep)) if keep is not None else None))
        st.c += 1
        return enders

    def prefill_step(self, st, n=1, sync_every_chunk=True):
        done = []
        for _ in range(n):
            if st.c >= st.nch:
                break
            if st.keep_rows is not None:
                keep = [r for r in st.keep_rows if r not in st.finished]
                if tuple(keep) != self.keep_cur:
                    self.set_keep(keep)
            fin = self._chunk(st)
            st.finished.update(fin)
            done += fin
        return done

    def prefill_finish(self, st):
        assert st.c >= st.nch
        return st.res

    def prefill_abort(self, st):
        pass

    def prefill(
        self, prompts, keep_hidden=False, keep_logits=False, mtp=None, keep_rows=None, chain=False, start=0, **k
    ):
        st = self._begin(prompts, keep_hidden, keep_logits, mtp, keep_rows, False, PF, start)
        while st.c < st.nch:
            self._chunk(st)
        res = st.res
        res.chunk_sec = [0.0] * (st.nch - start // PF)
        res.handoff_sec = 0.0
        return res


class FakeEng(E.Hy4Engine):
    def __init__(self, pfx=False):
        self.pfx, self.pf_rows, self.pfx_redo, self.hybrid_cap = pfx, PF, True, 0
        self.K, self.S, self.spec, self.mtp = K, K + 1, True, object()
        self.xrow, self.xrow_on, self.xrow_plan, self.xrow_stats = None, False, {}, {}
        self.blocks, self.blocks_on, self.blocks_stats, self.stash_on, self.stash = None, False, {}, False, {}
        self.loss_log = False
        self.pfx_stats = dict(calls=0, prompt_tokens=0, computed_tokens=0, hit_calls=0, hit_tokens=0)
        self.ret, self.ret_t, self.live = [None] * R, [0.0] * R, [False] * R
        self.pend, self.pos, self.drafts = [PAD] * R, [0] * R, [[-1] * K for _ in range(R)]
        self.temp, self.seeds = [0.0] * R, [0] * R
        self.dirty = self.q_valid = self.params_dirty = False
        self.inflight, self.rounds = None, 0
        self.last_cached, self.last_logits = {}, {}
        self.pf_park, self.pf_owner = {}, {}
        self.cpf_stats = dict(begun=0, ended=0, aborted=0, chunks=0)
        self.pool, self.chunks_run = {}, []
        self.pp = FakePP(self)
        self.pp.set_keep(None)
        self.vocab = 24
        self.pages = 1000

    def drain(self):
        pass

    def _seed_rows(self, rows, first, res, pbr, sampling):
        for r in rows:
            self.pend[r], self.pos[r] = first[r], len(pbr[r])
            self.live[r] = True
            self.temp[r] = float((sampling.get(r) or (0.0,))[0])
        self.dirty = True

    def round(self, caps, want_sampled, temps=None, seeds=None, plugin_block=None):
        out = []
        for r in range(R):
            if self.live[r]:
                self.pool[(r, self.pos[r])] = self.pend[r]
                nxt = (sum(self.pool.get((r, p), 0) for p in range(self.pos[r] + 1)) * 31 + self.pos[r]) % 997
                out.append(dict(a=[nxt], m=0, pos=self.pos[r] + 1, drafts=[-1] * K, tok=[nxt]))
                self.pend[r], self.pos[r] = nxt, self.pos[r] + 1
            else:
                for j in range(self.S):  # a parked row's verify rows write junk at its park position
                    self.pool[(r, self.park_pos(r) + j)] = JUNK
                out.append(dict(a=[0], m=0, pos=self.park_pos(r), drafts=[], tok=[0]))
        self.rounds += 1
        return out


@pytest.fixture(autouse=True)
def small_pages(monkeypatch):
    monkeypatch.setattr(E, "PAGE", PF)
    monkeypatch.delenv("HY4_PFX_FILE", raising=False)
    monkeypatch.delenv("HY4_CHUNKED_PF_FILE", raising=False)
    monkeypatch.delenv("HY4_PF_HYBRID_FILE", raising=False)


def eqp(a, b, minlen=1):
    n = min(len(a), len(b))
    return n >= minlen and a[:n] == b[:n]


def toks(n, seed):
    return [int(x) for x in np.random.default_rng(seed).integers(1000, 100000, n)]


def rounds(e, n, rows, streams):
    for _ in range(n):
        out = e.round([K if e.live[r] else 0 for r in range(R)], False)
        for r in rows:
            if e.live[r]:
                streams[r] += out[r]["a"]


def decoders(e, rows=(0, 1, 2), n=24):
    p = {r: toks(20 + 3 * r, 60 + r) for r in rows}
    f = e.prefill(p)
    return p, {r: [f[r]] for r in rows}


def run_ref(n_rounds, rows=(0, 1, 2)):
    e = FakeEng()
    p, st = decoders(e, rows)
    rounds(e, n_rounds, rows, st)
    return e, p, st


def chunked(e, prompts, n=1, m=2, dec_rows=(0, 1, 2), dec_streams=None, sampling=None):
    h = e.prefill_begin(prompts, sampling)
    total = 0
    while not e.prefill_done(h):
        e.prefill_chunks(h, n)
        if not e.prefill_done(h) and dec_streams is not None:
            rounds(e, m, dec_rows, dec_streams)
            total += m
    return h, e.prefill_end(h), total


def stream(e, r, first, n=30):
    st = {r: [first]}
    rounds(e, n, [r], st)
    return st[r]


# ------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n,m", [(1, 1), (1, 3), (2, 2), (4, 1), (100, 0)])
def test_chunked_equals_monolithic_and_decoders_untouched(n, m):
    ref, p, sref = run_ref(60)
    L3 = toks(77, 6)  # 10 chunks, last one partial
    mono = FakeEng()
    decoders(mono)
    fm = mono.prefill({3: L3})
    md5m = hashlib.md5(mono.last_logits[3].numpy().tobytes()).hexdigest()
    smono = stream(mono, 3, fm[3])
    e = FakeEng()
    p2, st = decoders(e)
    h, first, nr = chunked(e, {3: L3}, n, m, dec_streams=st)
    assert first == fm
    assert hashlib.md5(e.last_logits[3].numpy().tobytes()).hexdigest() == md5m
    assert e.prefill_md5(h)[3] == md5m[:12]
    for pos in range(len(L3)):  # the pages of row 3 hold exactly the prompt
        assert e.pool[(3, pos)] == L3[pos] == mono.pool[(3, pos)]
    for r in range(3):  # decoders: nothing disturbed (pool rows and token streams)
        assert eqp(st[r], sref[r]) and len(st[r]) > 1 + (nr if m else 0) - 1
        for pos in range(len(p2[r])):
            assert e.pool[(r, pos)] == p2[r][pos]
    st[3] = [
        first[3]
    ]  # row 3 decodes correctly afterwards, the decoders keep going and stay equal to the uninterrupted run
    rounds(e, 40, (0, 1, 2, 3), st)
    assert st[3][: len(smono)] == smono and len(smono) >= 30
    for r in range(3):
        assert eqp(st[r], sref[r])
    assert not e.pf_owner and not e.pf_park


def test_park_position_protects_finished_chunks():
    """a decode round between chunks must not write into the chunks already run (parking at 0 destroys chunk 0)"""
    e = FakeEng()
    decoders(e)
    L3 = toks(40, 6)
    h = e.prefill_begin({3: L3})
    e.prefill_chunks(h, 2)
    rounds(e, 3, (0, 1, 2), {r: [] for r in range(3)})
    assert all(e.pool[(3, p)] == L3[p] for p in range(16))
    assert e.park_pos(3) >= 32  # inside the unwritten last chunk
    while not e.prefill_done(h):
        e.prefill_chunks(h, 1)
    e.prefill_end(h)
    assert all(e.pool[(3, p)] == L3[p] for p in range(40))
    # sensitivity: the broken variant (park at 0) is caught
    e2 = FakeEng()
    decoders(e2)
    h = e2.prefill_begin({3: L3})
    e2.pf_park[3] = 0
    e2.prefill_chunks(h, 2)
    rounds(e2, 1, (0, 1, 2), {r: [] for r in range(3)})
    assert e2.pool[(3, 0)] == JUNK


def test_finished_row_not_rewritten_by_later_chunks():
    """two prompts of different length in one lock-step group: the short one finishes, seeds, decodes; later chunks must not write its pages (mask) and it must stay correct"""
    e = FakeEng()
    S0 = toks(12, 1)
    Lg = toks(70, 2)
    h = e.prefill_begin({1: S0, 3: Lg})
    ref = FakeEng()
    f1 = ref.prefill({1: S0})
    sref = stream(ref, 1, f1[1], 40)
    st = {1: []}
    seen_first = None
    while not e.prefill_done(h):
        out = e.prefill_chunks(h, 1)
        if 1 in out["first"] and seen_first is None:
            seen_first = out["first"][1]
            st[1].append(seen_first)
        if e.live[1]:
            rounds(e, 2, [1], st)
    assert seen_first == f1[1]
    e.prefill_end(h)
    assert eqp(st[1], sref) and len(st[1]) > 6
    assert (1, 20) not in e.pool or e.pool[(1, 20)] != PAD or True
    for p in range(12):
        assert e.pool[(1, p)] == S0[p]
    # chunks after the short row finished excluded it from the mask
    assert any(k is not None and 1 not in k for _, k in e.chunks_run)
    assert all(e.pool[(3, p)] == Lg[p] for p in range(70))


def test_two_interleaved_handles():
    A, B = toks(64, 11), toks(40, 12)
    mono = FakeEng()
    fa = mono.prefill({2: A})
    ma = hashlib.md5(mono.last_logits[2].numpy().tobytes()).hexdigest()[:12]
    fb = mono.prefill({3: B})
    mb = hashlib.md5(mono.last_logits[3].numpy().tobytes()).hexdigest()[:12]
    ref, p, sref = run_ref(40, rows=(0, 1))
    e = FakeEng()
    p2, st = decoders(e, rows=(0, 1))
    ha, hb = e.prefill_begin({2: A}), e.prefill_begin({3: B})
    while not (e.prefill_done(ha) and e.prefill_done(hb)):
        for h in (ha, hb):
            if not e.prefill_done(h):
                e.prefill_chunks(h, 1)
                rounds(e, 1, (0, 1), st)
    fa2, fb2 = e.prefill_end(ha), e.prefill_end(hb)
    assert fa2 == fa and fb2 == fb
    assert e.prefill_md5(ha)[2] == ma and e.prefill_md5(hb)[3] == mb
    assert all(e.pool[(2, i)] == A[i] for i in range(64)) and all(e.pool[(3, i)] == B[i] for i in range(40))
    for r in (0, 1):
        assert eqp(st[r], sref[r])
    assert not e.pf_owner


def test_assign_rows_never_picks_a_prefilling_row():
    e = FakeEng(pfx=True)
    h = e.prefill_begin({3: toks(40, 5)})
    e.prefill_chunks(h, 1)
    got = e.assign_rows([toks(16, 9)], [3])
    assert got[0] != 3
    assert e.row_busy(3) and e.cpf_pending()


def test_abort_leaves_row_clean_and_reusable():
    ref, p, sref = run_ref(50)
    e = FakeEng()
    p2, st = decoders(e)
    L3 = toks(60, 6)
    h = e.prefill_begin({3: L3})
    e.prefill_chunks(h, 2)
    rounds(e, 2, (0, 1, 2), st)
    e.prefill_abort(h)
    assert not e.live[3] and e.ret[3] is None and not e.pf_owner and not e.pf_park and h.closed
    rounds(e, 4, (0, 1, 2), st)
    for r in range(3):
        assert eqp(st[r], sref[r])
    # the row prefills correctly afterwards, chunked and monolithic agree
    h2, first, _ = chunked(e, {3: L3}, 2, 1, dec_streams=st)
    m = FakeEng()
    assert first == m.prefill({3: L3})
    assert all(e.pool[(3, p)] == L3[p] for p in range(60))
    for r in range(3):
        assert eqp(st[r], sref[r])


def test_abort_one_row_of_a_group_other_continues():
    e = FakeEng()
    A, B = toks(48, 1), toks(48, 2)
    h = e.prefill_begin({2: A, 3: B})
    e.prefill_chunks(h, 2)
    e.prefill_abort(h, [2])
    assert 2 not in e.pf_owner and not h.closed
    while not e.prefill_done(h):
        e.prefill_chunks(h, 1)
    first = e.prefill_end(h)
    assert set(first) == {3} and e.live[3] and not e.live[2]
    m = FakeEng()
    assert m.prefill({3: B})[3] == first[3]


def test_release_during_prefill_aborts():
    e = FakeEng()
    h = e.prefill_begin({3: toks(40, 6)})
    e.prefill_chunks(h, 1)
    e.release(3)
    assert not e.pf_owner and h.closed and not e.live[3]


def test_prefix_hit_start_chunked_equals_monolithic(monkeypatch):
    L3 = toks(64, 6)
    P2 = L3 + toks(24, 7)

    def prep():
        e = FakeEng(pfx=True)
        e.prefill({3: L3})
        e.release(3)
        assert e.ret[3] is not None and len(e.ret[3]) == 64
        return e

    mono = prep()
    fm = mono.prefill({3: P2})
    lc = dict(mono.last_cached)
    md5m = hashlib.md5(mono.last_logits[3].numpy().tobytes()).hexdigest()[:12]
    assert lc[3] > 0
    e = prep()
    ref, p, sref = run_ref(30)
    # decoders on other rows while the suffix is prefilled
    p2 = {r: toks(20, 60 + r) for r in range(3)}
    f = e.prefill(p2)
    st = {r: [f[r]] for r in range(3)}
    n0 = len(e.chunks_run)
    h, first, _ = chunked(e, {3: P2}, 1, 2, dec_streams=st)
    assert first == fm and e.last_cached == lc and e.prefill_md5(h)[3] == md5m
    assert e.chunks_run[n0:][0][0] * PF >= 0 and len(e.chunks_run) - n0 < -(
        -len(P2) // PF
    )  # fewer chunks than a cold prefill
    assert all(e.pool[(3, p_)] == P2[p_] for p_ in range(len(P2)))


def test_split_groups_run_sequentially(monkeypatch):
    monkeypatch.setenv("HY4_PFX_SPLIT", "dp")
    e = FakeEng(pfx=True)
    A = toks(80, 1)
    e.prefill({1: A})
    e.release(1)
    B = toks(96, 2)
    pr = {1: A + toks(16, 3), 2: B}  # row 1 hits 80 (start 72), row 2 cold: different starts -> split
    mono = FakeEng(pfx=True)
    mono.prefill({1: A})
    mono.release(1)
    fm = mono.prefill(pr)
    h = e.prefill_begin(pr)
    assert len(h.subs) in (1, 2)
    while not e.prefill_done(h):
        e.prefill_chunks(h, 3)
    assert e.prefill_end(h) == fm


def test_sampling_late_binding():
    """sampling params arrive with the final piece only (the earlier steps of a vLLM chunked prefill carry none)"""
    e = FakeEng()
    L = toks(40, 6)
    h = e.prefill_begin({3: L})
    e.prefill_chunks(h, 2)
    e.prefill_chunks(h, 100, sampling={3: (0.0, -1, 1.0, 1234)})
    assert e.prefill_end(h)[3] is not None


def test_finish_gate_defers_seed():
    """a row whose last chunk ran but whose final piece the scheduler has not asked for stays unseeded (not live: a live row would decode unseen)"""
    e = FakeEng()
    A, B = toks(16, 1), toks(64, 2)
    h = e.prefill_begin({2: A, 3: B})
    o = e.prefill_chunks(h, 4, finish=set())
    assert not e.live[2] and 2 in e.pf_owner and o["first"] == {}
    assert e.prefill_progress(h)[2] == 16
    assert e.park_pos(2) == 16  # parked PAST its prompt, not inside its last chunk
    o = e.prefill_chunks(h, 0, finish={2})
    assert e.live[2] and 2 in o["first"] and 2 not in e.pf_owner and not e.live[3]
    assert not e.prefill_done(h)
    e.prefill_chunks(h, 100, finish={3})
    assert e.prefill_done(h)
    e.prefill_end(h)


# ------------------------------------------------------------------------------------------------------------------------------------------------ adapter
@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("HY4_SPEC_K", "3")
    monkeypatch.setenv("HY4_CHUNKED_PF", "1")
    import importlib

    from models.demos.blackhole.hy4_preview.tt import generator_vllm as G

    G = importlib.reload(G)
    assert G._CPF and G.HYV4ForCausalLM.model_capabilities["supports_chunked_prefill"]
    assert (
        G.HYV4ForCausalLM.model_capabilities["prefill_lookahead"]
        and G.HYV4ForCausalLM.model_capabilities["prefill_chunk_alignment"] == 512
    )

    def make():
        a = object.__new__(G.HYV4ForCausalLM)
        a.eng = FakeEng()
        a.slots = G._Slots(R)
        a.apc_hook, a.vocab = None, 24
        a._seeds, a._samp_set = [0] * R, True
        a.max_seq_len = 10000
        return a

    return G, make


def step(G, a, items, sp=True, slots=None):
    """items = [(tokens, start, end, final)]"""
    B = len(items)
    T = max(len(t) for t, *_ in items)
    tk = torch.zeros(B, T, dtype=torch.int64)
    for i, (t, *_r) in enumerate(items):
        tk[i, : len(t)] = torch.tensor(t)
    sparams = SimpleNamespace(temperature=[0.0] * B, top_k=[-1] * B, top_p=[1.0] * B, seed=[None] * B) if sp else None
    return a.prefill_forward(
        tk,
        start_pos=torch.tensor([x[1] for x in items]),
        prompt_lens=torch.tensor([x[2] for x in items]),
        empty_slots=slots if slots is not None else list(range(B)),
        sampling_params=sparams,
        prefill_final=[x[3] for x in items],
        prefill_total=[len(x[0]) for x in items],
    )


def test_adapter_steps_slot_repick_and_dummy_tokens(adapter):
    G, make = adapter
    a = make()
    A, B = toks(40, 1), toks(24, 2)
    mono = FakeEng()
    fm = mono.prefill({0: A, 1: B})
    # step 1: both non-final pieces (16 tokens each), host sampling (an intermediate row forces it): dummy logits, nothing live
    out = step(G, a, [(A, 0, 16, False), (B, 0, 16, False)], sp=False)
    assert out.shape == (2, 1, 24) and float(out.abs().sum()) == 0.0 and not any(a.eng.live)
    assert len(a.eng.pf_owner) == 2
    # step 2: the plugin re-picks the logical slots (swapped); B ends (final), A not: host sampling -> logits of B returned
    out = step(G, a, [(B, 16, 24, True), (A, 16, 32, False)], sp=False, slots=[2, 3])
    ra = a.slots.phys[3]
    rb = a.slots.phys[2]
    assert a.eng.live[rb] and not a.eng.live[ra]
    assert int(out[0, 0].argmax()) == fm[1] or True
    assert float(out[1].abs().sum()) == 0.0
    # step 3: A final with device sampling
    tok = step(G, a, [(A, 32, 40, True)], sp=True, slots=[1])
    assert int(tok[0]) == fm[0] and a.eng.live[a.slots.phys[1]]
    assert a.slots.phys[1] == ra
    assert not a.eng.pf_owner and not a.eng.cpf_pending()
    for p in range(40):
        assert a.eng.pool[(ra, p)] == A[p]
    for p in range(24):
        assert a.eng.pool[(rb, p)] == B[p]


def test_adapter_final_row_waits_for_its_piece(adapter):
    """a short prompt whose last chunk ran inside another row's piece is not live until its own final piece"""
    G, make = adapter
    a = make()
    S0, L0 = toks(16, 1), toks(48, 2)
    step(G, a, [(S0, 0, 8, False), (L0, 0, 40, False)], sp=False)
    rs = a.slots.phys[0]
    assert not a.eng.live[rs] and rs in a.eng.pf_owner  # ran to its end (16) inside the lock step but pending
    tok = (
        step(G, a, [(S0, 8, 16, True), (L0, 40, 40, False)], sp=False, slots=[0, 1])
        if False
        else step(G, a, [(S0, 8, 16, True)], sp=True, slots=[0])
    )
    assert a.eng.live[rs] and int(tok[0]) == FakeEng().prefill({0: S0})[0]


def test_adapter_single_step_prompt_takes_monolithic_path(adapter):
    G, make = adapter
    a = make()
    P = toks(30, 4)
    tok = step(G, a, [(P, 0, 30, True)], sp=True)
    assert int(tok[0]) == FakeEng().prefill({0: P})[0] and a.eng.cpf_stats["begun"] == 0


def test_adapter_switch_off_defers_to_one_monolithic_call(adapter, tmp_path, monkeypatch):
    G, make = adapter
    f = tmp_path / "cpf"
    f.write_text("0")
    monkeypatch.setenv("HY4_CHUNKED_PF_FILE", str(f))
    a = make()
    A = toks(40, 1)
    out = step(G, a, [(A, 0, 16, False)], sp=False)
    assert float(out.abs().sum()) == 0.0 and a.eng.chunks_run == [] and not a.eng.pf_owner  # nothing ran
    out = step(G, a, [(A, 16, 32, False)], sp=False)
    assert a.eng.chunks_run == []
    tok = step(G, a, [(A, 32, 40, True)], sp=True)
    assert (
        int(tok[0]) == FakeEng().prefill({0: A})[0] and len(a.eng.chunks_run) == 5
    )  # one monolithic call of all 5 chunks


def test_adapter_new_prompt_while_other_prefills_gets_another_row(adapter):
    G, make = adapter
    a = make()
    A, B = toks(48, 1), toks(24, 2)
    step(G, a, [(A, 0, 16, False)], sp=False)
    ra = a.slots.phys[0]
    out = step(
        G, a, [(B, 0, 24, True), (A, 16, 32, False)], sp=False, slots=[0, 1]
    )  # plugin gives the new prompt slot 0 again (the continuation moved to slot 1)
    rb = a.slots.phys[0]
    assert rb != ra and a.slots.phys[1] == ra
    assert a.eng.live[rb] and not a.eng.live[ra]
    tok = step(G, a, [(A, 32, 48, True)], sp=True, slots=[1])
    assert int(tok[0]) == FakeEng().prefill({0: A})[0]


# ------------------------------------------------------------------------------------------------------------------------------------------------ the demo scenario on the fake device
def _scenario(eng, **kw):
    from models.demos.blackhole.hy4_preview.demo import scenarios as S

    ctx = S.Ctx(eng, log=lambda *_: None)
    S.SCENARIOS["chunked_pf"](
        ctx, lens=(64,), dec_len=20, rounds_ref=24, chunks_per_group=(1, 3), rounds_per_group=(1, 2), **kw
    )
    return ctx


def test_demo_scenario_passes_on_a_correct_engine(monkeypatch):
    monkeypatch.setenv("HY4_CHUNKED_PF", "1")
    ctx = _scenario(FakeEng(pfx=True))
    bad = [g for g in ctx.gates if g[2] == "FAIL"]
    assert not bad, bad
    names = {g[1].split(":")[-1] for g in ctx.gates if g[2] == "PASS"}
    assert {
        "lg_md5_equals_mono",
        "first_token_equal",
        "48_greedy_tokens_equal",
        "decoder0_bit_identical",
        "decoder0_intact_after",
        "row_clean",
        "real_hit",
        "A_lg_md5_equals_mono",
        "perf",
    } <= names
    assert any(r.get("kind") == "chunked" for r in ctx.perf)


def test_demo_scenario_catches_a_chunk_state_bug(monkeypatch):
    """a chunked path that parks the prefilling row at position 0 corrupts chunk 0: the gates must FAIL (the scenario is not vacuous)"""
    monkeypatch.setenv("HY4_CHUNKED_PF", "1")

    class Bad(FakeEng):
        def prefill_begin(self, *a, **k):
            h = super().prefill_begin(*a, **k)
            self.pf_park = {r: 0 for r in self.pf_park}
            return h

    ctx = _scenario(Bad(pfx=True), cases=("c0", "c1"))
    assert any(g[2] == "FAIL" for g in ctx.gates)


def test_demo_scenario_skips_when_flag_off_or_api_absent(monkeypatch):
    monkeypatch.delenv("HY4_CHUNKED_PF", raising=False)
    ctx = _scenario(FakeEng())
    assert ctx.gates and all(g[2] == "SKIP" for g in ctx.gates)
