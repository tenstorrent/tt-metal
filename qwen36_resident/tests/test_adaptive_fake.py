# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of the DB3 phase-2 class (qwen36_resident/vllm_adaptive.py) on fakes: the plain build is
test_vllm_rows_fake's FakeModel (one 32-row launch) behind the rows adapter; the spec build a fake SpecRowsSession with
the same greedy function of the sequence; the moves carry each user's sequence (and its KV ownership in the fake pool;
the move to spec checks the class's token history against the plain row's tokens). A small runner drives the
runner-model spec contract as vllm-tt-plugin does: a verify call when a row carries drafts or a multi-token commit,
an ordinary (narrow) call otherwise, propose after every decode step. Scenario: 4 users (SPEC) -> 6 arrive (PLAIN,
mixed steps, a resolving verify) -> 7 finish -> the 3 left return to SPEC after PLAIN_HOLD steps. Gate: every
request's tokens == its greedy chain, and the modes / moves happen."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm_tt_plugin.spec_decode")

from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID, DraftOutput, VerifyOutput

import qwen36_resident.tests.test_vllm_rows_fake as F
from qwen36_resident import vllm_adaptive as A
from qwen36_resident.vllm_rows import BLOCK, CHUNK_UNIT, ResidentRowsAdapter

V, K, B = F.V, 3, 32
ref_next = F.ref_next


class FakeSpec:
    """SpecRowsSession's interface; verify's rows follow ref_next of the user's sequence; drafts: the greedy chain with
    every third draft wrong"""

    def __init__(self, miss_all=False):
        U = A.SPEC_USERS
        self.e = SimpleNamespace(U=U, R=4)
        self.miss_all = miss_all  # (every draft wrong: every verify commits one token, no resolving verify)
        self.seq, self.pending = [None] * U, [None] * U
        self.started, self.pos = [False] * U, [0] * U

    def start_many(self, items, prefilled=()):
        out = []
        for u, ids, _ in items:
            if u in prefilled:  # (a prefix-cache restore moved in: its state is the prompt but the last token)
                assert self.seq[u] == list(ids[:-1]), (u, len(self.seq[u]), len(ids))
            self.seq[u], self.started[u], self.pending[u] = list(ids), True, None
            self.pos[u] = len(ids)
            lg = torch.zeros(V)
            lg[ref_next(ids)] = 1.0
            out.append(lg)
        return out

    def verify(self, tokens, positions, active):
        g = [[0] * 4 for _ in range(len(active))]
        for u, a in enumerate(active):
            if not a:
                continue
            assert int(positions[u]) == len(self.seq[u]), (u, positions[u], len(self.seq[u]))
            row = [int(t) for t in tokens[u]]
            g[u] = [ref_next(self.seq[u] + row[: k + 1]) for k in range(4)]
            self.pending[u], self.started[u] = row, False
        return g

    def propose(self, anchors, positions, accepted, active):
        out = []
        for u, a in enumerate(active):
            if a and not self.started[u] and self.pending[u] is not None:
                self.seq[u] += self.pending[u][: int(accepted[u])]
                self.pending[u] = None
            if not a:
                out.append([0] * K)
                continue
            assert int(positions[u]) == len(self.seq[u]), (u, positions[u], len(self.seq[u]))
            d, s = [], self.seq[u] + [int(anchors[u])]
            for k in range(K):
                t = ref_next(s)
                t = (t + 1) % V if (self.miss_all or (len(s) + k) % 3 == 0) else t
                d.append(t)
                s = s + [t]
            out.append(d)
        return out


class FakeAE:
    def __init__(self, pf, plain, pp):
        self._pf, self.plain, self.pp = pf, plain, pp
        self.spec = SimpleNamespace(window=2048)
        self.sess, self.spec_tables, self.plain_tables, self.moves = FakeSpec(), {}, None, []

    def _set_spec_table(self, u, table):
        self.spec_tables[u] = list(table)

    def move_to_plain(self, u, j, table, acc):
        s, pf = self.sess, self._pf
        if not s.started[u] and s.pending[u] is not None:
            s.seq[u] += s.pending[u][: int(acc)]
            s.pending[u] = None
        seq = s.seq[u]
        pf.hist[j], pf.conv_ok[j], pf.tables[j] = list(seq), True, list(table)
        for q in range(len(seq)):
            pf.pool[(table[q // BLOCK], q % BLOCK)] = (j, seq[q])
        self.moves.append(("to_plain", u, j))

    def move_to_spec_many(self, items):
        for j, u, table, tokens in items:
            self.move_to_spec(j, u, table, tokens)

    def move_to_spec(self, j, u, table, tokens):
        pf, s = self._pf, self.sess
        assert list(tokens[:-1]) == pf.hist[j], "the class's token history != the plain row's tokens"
        s.seq[u], s.started[u], s.pending[u], s.pos[u] = list(pf.hist[j]), True, None, len(pf.hist[j])
        self.moves.append(("to_spec", j, u))


def make(monkeypatch):
    monkeypatch.setattr(F, "ROWS", B)
    pf = F.FakeModel(B, sample=True)
    pf.launch_rows = B
    pf.row_order = lambda: list(range(B))
    plain = F.Contract(pf, F.MODEL_API + ("launch_rows",))
    pp = F.FakePP(CHUNK_UNIT, pf)
    m = object.__new__(A.Qwen36ResidentAdaptiveForCausalLM)
    ResidentRowsAdapter.__init__(m, None, V, 8192, B, CHUNK_UNIT, False, rows=B)
    m.mesh, m.ck = None, None
    m.ae = FakeAE(pf, plain, pp)
    m.sess = m.ae.sess
    m.ae.plain_tables = m.tables
    m.spec_of, m.history, m.last_acc, m.mode, m.low_calls = {}, {}, {}, "SPEC", 0
    m.sampled, m._t, m._plain_stale, m.greedy_ok, m.prompt_len = set(), {}, False, set(), {}
    m._pay = dict(acc=None, spec_ms={}, plain_ms={}, verifies=0, last_t=None, kind=None, n=0, hold=0, excl=0.0)
    m.engine = A._PlainRowsEngine(plain, pp, torch.eye(V), V, V, B, CHUNK_UNIT, None)
    return m, pf


class Runner:
    def __init__(self, m, steady=False):
        self.m, self.req = m, {}  # row -> dict(prompt, tokens, drafts, acc, table)
        self.nb = 40
        # steady (async scheduling, DECODE_RELOAD_CONTRACT.md): a narrow call right after a narrow call over the same
        # rows comes without reload_inputs, its tokens / positions the previous call's (a step behind)
        self.steady, self.last_narrow = steady, None

    def table(self, row):
        return list(range(row * self.nb, (row + 1) * self.nb))

    def pt(self):
        t = torch.zeros(B, self.nb, dtype=torch.int32)
        for row in self.req:
            t[row] = torch.tensor(self.table(row))
        return t

    def sp(self, rows):
        """the device-sampling parameters per wire row (a sampled request: temperature 0.8, its seed)"""
        temp, seed = [0.0] * B, [None] * B
        for r in rows:
            if self.req[r].get("seed") is not None:
                temp[r], seed[r] = 0.8, self.req[r]["seed"]
        return SimpleNamespace(temperature=temp, top_k=[20] * B, top_p=[0.95] * B, seed=seed)

    def arrive(self, rows, prompts):
        n, T = len(rows), max(len(p) for p in prompts)
        tok = torch.zeros(n, T, dtype=torch.long)
        pt = torch.zeros(n, self.nb, dtype=torch.int32)
        for j, (row, p) in enumerate(zip(rows, prompts)):
            tok[j, : len(p)] = torch.tensor(p)
            pt[j] = torch.tensor(self.table(row))
        lg = self.m.prefill_forward(tok, pt, None, [len(p) for p in prompts], empty_slots=list(rows))
        for j, (row, p) in enumerate(zip(rows, prompts)):
            self.req[row] = dict(prompt=list(p), tokens=list(p) + [int(lg[j, 0].argmax())], drafts=[], acc=1)

    def finish(self, row):
        self.req.pop(row)
        self.m.release_request(row)

    def step(self):
        rows = sorted(self.req)
        assert rows == list(range(len(rows))), "the scenario keeps the rows a prefix"
        verify = any(self.req[r]["drafts"] for r in rows) or any(self.req[r]["acc"] > 1 for r in rows)
        if verify:
            tok = torch.zeros(B, 1 + K, dtype=torch.long)
            pos = torch.full((B, 1 + K), -1, dtype=torch.int32)
            nvd, acc = torch.zeros(B, dtype=torch.int32), torch.ones(B, dtype=torch.int32)
            for r in rows:
                q = self.req[r]
                p = len(q["tokens"]) - 1
                tok[r, 0], pos[r, 0] = q["tokens"][-1], p
                tok[r, 1:] = PLACEHOLDER_TOKEN_ID
                for k, d in enumerate(q["drafts"]):
                    tok[r, 1 + k], pos[r, 1 + k] = d, p + 1 + k
                nvd[r], acc[r] = len(q["drafts"]), q["acc"]
            out = self.m.decode_forward(
                tok,
                pos,
                self.pt(),
                num_valid_drafts=nvd,
                accepted_counts=acc,
                spec_mode="argmax_ids",
                sampling_params=self.sp(rows),
            )
            assert isinstance(out, VerifyOutput)
            committed = {}
            for r in rows:
                q, g = self.req[r], out.argmax_ids[r].tolist()
                k = 0
                while k < len(q["drafts"]) and q["drafts"][k] == g[k]:
                    k += 1
                committed[r] = q["drafts"][:k] + [g[k]]
            first = pos[:, 0]
            # (the plugin may send the next narrow call steady after a verify too: its inputs the verify's column 0)
            self.last_narrow = (rows, tok[:, :1].clone(), pos[:, 0].clone())
        else:
            tok = torch.zeros(B, 1, dtype=torch.long)
            pos = torch.full((B,), -1, dtype=torch.int32)
            for r in rows:
                tok[r, 0], pos[r] = self.req[r]["tokens"][-1], len(self.req[r]["tokens"]) - 1
            sp = self.sp(rows)
            steady = self.steady and self.last_narrow is not None and self.last_narrow[0] == rows
            send_tok, send_pos = (self.last_narrow[1], self.last_narrow[2]) if steady else (tok, pos)
            res = self.m.decode_forward(
                send_tok,
                send_pos,
                self.pt(),
                reload_inputs=not steady,
                reload_page_table=False,
                reload_sampling_params=not steady,
                reset_sampling_state=not steady,
                read_from_device=True,
                sampling_params=sp,
            )
            self.last_narrow = (rows, tok.clone(), pos.clone())
            res = self.m.process_decode_output_host(res)
            committed = {r: [int(res[r, 0])] for r in rows}
            first = pos
        ct = torch.full((B, 1 + K), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        cp = torch.full((B, 1 + K), -1, dtype=torch.int32)
        ca = torch.ones(B, dtype=torch.int32)
        for r in rows:
            q, c = self.req[r], committed[r]
            q["tokens"] += c
            q["acc"] = len(c)
            ct[r, : len(c)] = torch.tensor(c)
            cp[r] = int(first[r]) + 1 + torch.arange(1 + K)
            ca[r] = len(c)
        d = self.m.propose_draft_tokens(K, ct, cp, ca)
        assert isinstance(d, DraftOutput)
        for r in rows:
            self.req[r]["drafts"] = d.draft_token_ids[r].tolist()[: int(d.num_valid[r])]
        return verify

    def check(self):
        for r, q in self.req.items():
            s, seed = list(q["prompt"]), q.get("seed")
            for k, t in enumerate(q["tokens"][len(q["prompt"]) :]):
                # (a sampled request: its first token is the runner's host sample of the prefill logits; then the hub's)
                want = ref_next(s, (seed, len(s) - 1) if seed is not None else None) if (seed is None or k > 0) else t
                assert t == want, f"row {r}: token {k} is {t}, want {want}"
                s.append(t)


def test_adaptive_ramp(monkeypatch):
    monkeypatch.setattr(A, "PLAIN_HOLD", 4)
    m, pf = make(monkeypatch)
    run = Runner(m)
    prompts = {r: [(r * 17 + i * 5) % V for i in range(30 + 3 * r)] for r in range(10)}
    run.arrive(range(4), [prompts[r] for r in range(4)])
    assert m.mode == "SPEC" and not m.spec_of, "prefilled into the plain build"
    kinds = [run.step() for _ in range(8)]
    run.check()
    assert len(m.spec_of) == 4, "promoted after their first (narrow) decode showed them greedy"
    assert kinds[0] is False and all(kinds[1:]), "first a narrow step (no drafts yet), then verify steps"
    run.arrive(range(4, 10), [prompts[r] for r in range(4, 10)])
    assert m.mode == "PLAIN"
    for _ in range(8):
        run.step()
    run.check()
    assert not m.spec_of, "every user moved to the plain build"
    for r in range(9, 2, -1):
        run.finish(r)
    for _ in range(8):
        run.step()
    run.check()
    assert m.mode == "SPEC" and len(m.spec_of) == 3, (m.mode, m.spec_of)
    for _ in range(6):
        run.step()
    run.check()
    kinds_moves = [mv[0] for mv in m.ae.moves]
    assert kinds_moves.count("to_plain") == 4 and kinds_moves.count("to_spec") == 4 + 3, m.ae.moves
    assert sum(len(q["tokens"]) - len(q["prompt"]) for q in run.req.values()) > 3 * 20


def test_sampled_request_joins_spec(monkeypatch):
    """a sampled request arrives while 3 greedy users speculate: PLAIN from its first decode call, its tokens the hub's
    samples (never the argmax), the greedy users' tokens their greedy chains; when it leaves, back to SPEC"""
    monkeypatch.setattr(A, "PLAIN_HOLD", 3)
    m, pf = make(monkeypatch)
    run = Runner(m)
    prompts = {r: [(r * 23 + i * 7) % V for i in range(28 + 2 * r)] for r in range(4)}
    run.arrive(range(3), [prompts[r] for r in range(3)])
    for _ in range(5):
        run.step()
    run.arrive([3], [prompts[3]])
    run.req[3]["seed"] = 1234  # (the runner sampled its first token on the host; from here the device samples)
    assert m.slots.res_of[3] not in m.spec_of, "prefilled into the plain build (its temperature unknown there)"
    for _ in range(6):
        run.step()
    run.check()
    assert m.mode == "PLAIN" and not m.spec_of and m.slots.res_of[3] in m.sampled
    run.finish(3)
    for _ in range(6):
        run.step()
    run.check()
    assert m.mode == "SPEC" and len(m.spec_of) == 3, (m.mode, m.spec_of)


@pytest.mark.parametrize("scenario", ["ramp", "sampled", "sampled_miss_all"])
def test_steady_calls(monkeypatch, scenario):
    """the same scenarios with async scheduling's steady narrow calls (no reload_inputs, inputs a step behind): after
    a move into the plain build or a spec-served step the class rebuilds the call's inputs from its history
    (miss_all: every draft rejected, so the switch to PLAIN is followed by a narrow call, the move happens there)"""
    monkeypatch.setattr(A, "PLAIN_HOLD", 3)
    m, pf = make(monkeypatch)
    if scenario.endswith("miss_all"):
        m.ae.sess.miss_all = True
    run = Runner(m, steady=True)
    prompts = {r: [(r * 29 + i * 3) % V for i in range(26 + 2 * r)] for r in range(10)}
    if scenario.startswith("ramp"):
        run.arrive(range(4), [prompts[r] for r in range(4)])
        for _ in range(6):
            run.step()
        run.arrive(range(4, 10), [prompts[r] for r in range(4, 10)])
        for _ in range(8):
            run.step()
        run.check()
        assert m.mode == "PLAIN" and not m.spec_of
        for r in range(9, 2, -1):
            run.finish(r)
        for _ in range(8):
            run.step()
    else:
        run.arrive(range(3), [prompts[r] for r in range(3)])
        for _ in range(4):
            run.step()
        run.arrive([3], [prompts[3]])
        run.req[3]["seed"] = 99
        for _ in range(8):
            run.step()
        run.check()
        assert m.mode == "PLAIN"
        run.finish(3)
        for _ in range(8):
            run.step()
    run.check()
    assert m.mode == "SPEC" and len(m.spec_of) == 3, (m.mode, m.spec_of)


@pytest.mark.parametrize("acc,expect", [(1.0, False), (3.2, True)])
def test_pay_policy(monkeypatch, acc, expect):
    """speculation pays when its tokens per user-ms reach a plain step's (x PAY_MARGIN): 8 users, 40 ms verify steps
    vs the default 20 ms plain step -> 1.0 token per step does not pay, 3.2 does"""
    import time as _time

    m, pf = make(monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(_time, "perf_counter", lambda: clock[0])
    m._pay["last_t"] = clock[0]
    pays = None
    for _ in range(A.PAY_MIN + 4):
        clock[0] += 0.040
        m._pay["kind"], m._pay["n"] = "verify", 8
        pays = m._measure([acc] * 8)
    assert pays is expect, (m._pay, pays)


def test_promotion_waits_on_long_contexts(monkeypatch):
    """a greedy user with a 1,500-token prompt stays in the plain build until it has generated 1500 / PROMOTE_DIV
    tokens (its refill would cost more than a short answer gains), then moves; its tokens stay its greedy chain"""
    m, pf = make(monkeypatch)
    run = Runner(m)
    run.nb = 40  # (2,560 positions a table)
    prompt = [(i * 7 + 3) % V for i in range(1500)]
    run.arrive([0], [prompt])
    need = 1500 // A.PROMOTE_DIV
    while len(run.req[0]["tokens"]) - 1500 < need - 1:
        run.step()
        assert not m.spec_of, f"promoted after {len(run.req[0]['tokens']) - 1500} tokens, before {need}"
    for _ in range(4):
        run.step()
    assert m.spec_of, "promoted once it has generated enough"
    for _ in range(6):
        run.step()
    run.check()
