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
            # (sampled speculation: a row with temperature > 0 draws at its own position with its user's seed)
            prm = getattr(self, "params", None)
            key = lambda k: (  # noqa: E731
                (prm[u * 4 + k][3], int(positions[u]) + k) if prm is not None and prm[u * 4 + k][0] > 0 else None
            )
            g[u] = [ref_next(self.seq[u] + row[: k + 1], key(k)) for k in range(4)]
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


class FakeSpecModel:
    """the spec build's set_sampling (sampled speculation): per verify row (temperature, top_k, top_p, seed)"""

    def __init__(self, spec):
        self.spec = spec

    def set_sampling(self, rows):
        assert len(rows) == A.SPEC_USERS * 4
        self.spec.params = list(rows)


class FakeAE:
    def __init__(self, pf, plain, pp):
        self._pf, self.plain, self.pp = pf, plain, pp
        self.sess, self.spec_tables, self.plain_tables, self.moves = FakeSpec(), {}, None, []
        self.spec = SimpleNamespace(m=FakeSpecModel(self.sess), window=2048)

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

    def move_to_spec_many(self, items, taps=None):
        self.taps_used = getattr(self, "taps_used", []) + [j for j, _, _, _ in items if j in (taps or {})]
        for j, u, table, tokens in items:
            self.move_to_spec(j, u, table, tokens)

    def move_to_spec(self, j, u, table, tokens):
        pf, s = self._pf, self.sess
        assert list(tokens[:-1]) == pf.hist[j], "the class's token history != the plain row's tokens"
        s.seq[u], s.started[u], s.pending[u], s.pos[u] = list(pf.hist[j]), True, None, len(pf.hist[j])
        self.moves.append(("to_spec", j, u))


def make(monkeypatch, gate=False):
    monkeypatch.setattr(F, "ROWS", B)
    monkeypatch.setattr(A, "PROMOTE_GATE", gate)  # (fake steps take microseconds: the gate has its own tests)
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
    m.sampled, m._t, m._plain_stale, m.greedy_ok, m._calls = set(), {}, False, set(), [0, 0]
    m._pay, m.prompt_len, m.taps_of = m._new_pay(), {}, {}
    m._spec_seed, m._spec_params = {}, None
    m._prior, m._req_stat, m.temp_reqs = {"g": A.PROMOTE_ACC, "s": A.PROMOTE_ACC_SAMPLED}, {}, set()
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
        self._propose(rows, committed, first)
        return verify

    def host_step(self):
        """an ordinary decode the runner samples on the host (a live request needs logits: logprobs, structured
        output, ...): narrow even right after a multi-token commit (plugin narrow_decode_resolves_commits), no
        sampling_params, logits back (greedy here)"""
        rows = sorted(self.req)
        tok = torch.zeros(B, 1, dtype=torch.long)
        pos = torch.full((B,), -1, dtype=torch.int32)
        for r in rows:
            tok[r, 0], pos[r] = self.req[r]["tokens"][-1], len(self.req[r]["tokens"]) - 1
        res = self.m.decode_forward(
            tok, pos, self.pt(), reload_inputs=True, reload_page_table=False, read_from_device=True
        )
        lg = torch.as_tensor(self.m.process_decode_output_host(res, is_tokens=False))
        lg = lg.reshape(lg.shape[0], -1, lg.shape[-1])[:, 0, :]
        committed = {r: [int(lg[r].argmax())] for r in rows}
        self.last_narrow = (rows, tok.clone(), pos.clone())
        self._propose(rows, committed, pos)

    def _propose(self, rows, committed, first):
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
    monkeypatch.setattr(
        A, "PAY_MIN", 10**6
    )  # (the fake drafts' acceptance would not pay: the pay policy's own tests)
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


def test_host_sampled_steps_right_after_a_multi_token_commit(monkeypatch):
    """a request needing host-only sampling (logprobs, structured output, ...) makes the plugin send ordinary
    host-sampled decodes (no sampling_params, logits back) even right after a verify that committed several tokens
    (narrow_decode_resolves_commits): the class moves its spec users to plain from the counts propose recorded, every
    user's tokens stay the reference's, and nobody speculates while such steps come"""
    m, pf = make(monkeypatch)
    run = Runner(m)
    prompts = [[(r * 13 + i * 7) % V for i in range(30 + 2 * r)] for r in range(3)]
    run.arrive(range(3), prompts)
    for _ in range(12):
        run.step()
        if m.spec_of and any(q["acc"] > 1 for q in run.req.values()):
            break
    assert m.spec_of and any(q["acc"] > 1 for q in run.req.values()), "a multi-token commit to resolve"
    for _ in range(5):
        run.host_step()
        assert not m.spec_of and m.mode == "PLAIN"
        assert not any(q["drafts"] for q in run.req.values()), "no drafts while host steps come"
    run.check()


def test_the_spec_plan_declares_commit_resolving_narrow_decodes():
    plan = A.Qwen36ResidentAdaptiveForCausalLM.spec_plan(None, 32, 3)
    assert plan.supports_narrow_decode and plan.narrow_decode_resolves_commits


def test_sampled_requests_speculate_with_sampled_verify_rows(monkeypatch):
    """(RESIDENT_SPEC_SAMPLE) a temperature > 0 request is promoted like a greedy one; its verify rows draw at their
    own positions with its seed, a draft accepted while the draw equals it, so its tokens stay its seeded reference
    (the plain build's draws) through the moves; greedy users alongside keep theirs"""
    monkeypatch.setattr(A, "SPEC_SAMPLE", True)
    monkeypatch.setattr(A, "SEEDED_PLAIN", False)  # (the fake's sampled references need seeds)
    m, pf = make(monkeypatch)
    run = Runner(m)
    prompts = {r: [(r * 23 + i * 7) % V for i in range(28 + 3 * r)] for r in range(3)}
    run.arrive(range(3), [prompts[r] for r in range(3)])
    run.req[1]["seed"], run.req[2]["seed"] = 77, 99
    for _ in range(12):
        run.step()
    run.check()
    assert set(m.spec_of) == {0, 1, 2} and m.mode == "SPEC", (m.spec_of, m.mode)
    prm = m.ae.sess.params
    assert prm[m.spec_of[1] * 4][3] == 77 and prm[m.spec_of[1] * 4][0] > 0 and prm[m.spec_of[0] * 4][0] == 0
    plan = A.Qwen36ResidentAdaptiveForCausalLM.spec_plan(None, 32, 3)
    assert getattr(plan, "verify_rows_sample", False)


def test_acceptance_priors_per_kind(monkeypatch):
    """a sampled request is expected at the sampled prior (2.2 tokens a verify): promoted at 1 user, not at 4 (the
    4-user break-even ~2.4); a finished request moves its kind's prior toward its own acceptance (>= PRIOR_MIN verifies)
    or, if it never speculated, back toward the default"""
    monkeypatch.setattr(A, "SPEC_SAMPLE", True)
    m = _gate_case(monkeypatch, 4, 200, 1)
    m.temp_reqs.update(range(4))
    assert not m._promotion_pays(sorted(m.live))[0], m._promotion_pays(sorted(m.live))[1]
    m = _gate_case(monkeypatch, 1, 200, 1)
    m.temp_reqs.add(0)
    assert m._promotion_pays([0])[0], m._promotion_pays([0])[1]
    m._req_stat[0] = [15, 10]  # (1.5 tokens a verify over 10 verifies)
    m._note_finished(0)
    assert abs(m._prior["s"] - (2.2 + 0.3 * (1.5 - 2.2))) < 1e-9 and m._prior["g"] == A.PROMOTE_ACC
    m.temp_reqs.add(1)
    m._note_finished(1)  # (never speculated: back toward 2.2)
    assert abs(m._prior["s"] - (1.99 + A.PRIOR_RECOVER * (2.2 - 1.99))) < 1e-9, m._prior
    for _ in range(6):  # (a run of hard requests: the prior falls under the 1-user break-even, no more probes)
        m.temp_reqs.add(2)
        m._req_stat[2] = [13, 10]
        m._note_finished(2)
    m.temp_reqs.add(0)
    assert not m._promotion_pays([0])[0], m._prior


def test_penalized_requests_stay_plain_with_sampled_speculation(monkeypatch):
    """penalties reshape the target per accepted token: non-speculable even with sampled verify rows"""
    monkeypatch.setattr(A, "SPEC_SAMPLE", True)
    sp = SimpleNamespace(temperature=[0.8], presence_penalty=[0.5], frequency_penalty=[0.0], repetition_penalty=[1.0])
    assert A.Qwen36ResidentAdaptiveForCausalLM._non_speculable(sp, 0)
    sp.presence_penalty = [0.0]
    assert not A.Qwen36ResidentAdaptiveForCausalLM._non_speculable(sp, 0)
    sp.seed = [1234]  # (a seeded sampled request decodes plainly: the same seed gives the same text)
    assert A.Qwen36ResidentAdaptiveForCausalLM._non_speculable(sp, 0)


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


@pytest.mark.parametrize("n,expect", [(1, True), (4, True), (8, False)])
def test_pay_policy_decides_an_unmeasured_live_count_at_once(monkeypatch, n, expect):
    """before any verify step at a live count, the default step-time model with the acceptance at least PROMOTE_ACC
    decides: 1-4 users speculate (a probe), 8 do not — 8 users arriving together were promoted one by one and the
    8-user probe's PAY_MIN ~60 ms verify steps ran before PLAIN (rc-20261124j natural c8 41.7 vs 50.7 tok/s per user)"""
    m = _gate_case(monkeypatch, n, 200, 1)
    m._pay["kind"], m._pay["n"], m._pay["ctx"] = "plain", n, 200 * n
    assert m._measure([]) is expect, m._pay_line()


def test_pay_policy_excludes_prefills(monkeypatch):
    """a prefill between two decode steps (another user's prompt) is not the step's time: 40 ms verify steps with 3.2
    tokens each pay at 8 users even when a 300 ms prefill runs between every other two of them (the step across a
    prefill is not sampled: its device time outlasts the call)"""
    import time as _time

    m, pf = make(monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(_time, "perf_counter", lambda: clock[0])

    def slow_prefill(self, *a, **k):
        clock[0] += 0.300
        return None

    monkeypatch.setattr(ResidentRowsAdapter, "prefill_forward", slow_prefill)
    m._pay["last_t"] = clock[0]
    pays = None
    for i in range(A.PAY_MIN + 4):
        clock[0] += 0.040
        if i % 2:
            m.prefill_forward(None, None, None, [1], empty_slots=[0], prefill_final_mask=[False])
            clock[0] += 0.015  # (the prefill's device time past its call: in the step across it, not sampled)
        m._pay["kind"], m._pay["n"] = "verify", 8
        pays = m._measure([3.2] * 8)
    assert pays is True and abs(m._pay["spec_ms"][8] - 40.0) < 1e-6, (m._pay, pays)


def _gate_case(monkeypatch, n_live, ctx, generated, out=None):
    """a class with n_live greedy plain users of ctx-token histories, `generated` of them output"""
    m, pf = make(monkeypatch, gate=True)
    for r in range(n_live):
        m.history[r], m.prompt_len[r] = [1] * ctx, ctx - generated
    m.greedy_ok.update(range(n_live))
    m.live.clear()
    m.live.update(range(n_live))
    m._pay["out"] = out
    return m


@pytest.mark.parametrize(
    "n_live,ctx,generated,out,expect",
    [
        (1, 8192, 1, None, True),  # one user, 8K: ~255 tokens x ~6 ms saved >> a ~290 ms refill
        (1, 300, 1, None, True),  # short prompt: refill ~60 ms
        (8, 2048, 1, None, False),  # 8 users arriving with 2K prompts: ~2.1 s refill x 8 users vs ~1 ms saved a token
        (8, 2048, 1, 128, False),
        (4, 256, 1, None, True),  # 4 short chats
        (8, 128, 1, None, False),  # 8 short chats: a 150 ms refill x 8 waiting users vs ~0.6 ms saved a token
        (1, 12000, 1, 16, False),  # short answers learned (out 16 tokens) on a long context: the refill does not pay
        (1, 12000, 240, 256, True),  # (at least a quarter of the expected output left: 64 tokens)
    ],
)
def test_promotion_gate(monkeypatch, n_live, ctx, generated, out, expect):
    """promote only when the candidates' expected remaining tokens x the saving per token pay for the refill every
    live user waits for (defaults: no measurement yet)"""
    m = _gate_case(monkeypatch, n_live, ctx, generated, out)
    ok, line = m._promotion_pays(sorted(m.live))
    assert ok is expect, line


def test_promotion_gate_estimates_are_robust(monkeypatch):
    """a warm-up's two 16-token answers do not stop a single user's promotion (the output EMA starts at PROMOTE_OUT
    and moves slowly); a stray step / refill time (a trace capture, a prefill's device time) does not set an estimate"""
    m = _gate_case(monkeypatch, 1, 8192, 1)
    m._note_output(16)
    m._note_output(16)
    assert m._promotion_pays([0])[0], m._promotion_pays([0])[1]
    m.history[0], m.prompt_len[0] = [1] * 128, 127
    assert A.robust_ema(None, 270.0, A.PLAIN_MS_DEFAULT(1)) is None  # (first sample 16x the default: skipped)
    assert A.robust_ema(None, 18.0, A.PLAIN_MS_DEFAULT(1)) == 18.0
    assert A.robust_ema(18.0, 201.0, A.PLAIN_MS_DEFAULT(1)) == 18.0
    assert A.robust_ema(None, 4.4, A.REFILL_MS_PER_TOKEN, a=0.3) is None
    for _ in range(80):
        m._note_output(16)
    assert m._promotion_pays([0])[0], "a short prompt's refill is cheap: still pays for short answers"
    m.history[0], m.prompt_len[0] = [1] * 12000, 11999
    assert not m._promotion_pays([0])[0], "short answers learned: a long context's refill does not pay"


def test_a_skipped_first_sample_keeps_the_default(monkeypatch):
    """a first step-time sample that robust_ema skips (an outlier) leaves no estimate behind — a None stored there
    crashed the promotion gate served (TypeError, 2026-11-24): the default model stays in use"""
    import time as _time

    m = _gate_case(monkeypatch, 4, 2048, 1)
    clock = [100.0]
    monkeypatch.setattr(_time, "perf_counter", lambda: clock[0])
    m._pay["last_t"] = clock[0]
    for kind in ("plain", "verify"):
        clock[0] += 1.0  # (1,000 ms: over 3 x either default model at 4 users)
        m._pay["kind"], m._pay["n"] = kind, 4
        m._measure([2] * 4)
    assert 4 not in m._pay["plain_ms"] and 4 not in m._pay["spec_ms"], m._pay
    ok, line = m._promotion_pays(sorted(m.live))  # (the gate still computes)
    assert isinstance(ok, bool), line


def test_long_context_steps_do_not_close_the_gate(monkeypatch):
    """a 250K request's verify steps (~44 ms) and plain steps (~23 ms) are its context's cost, not the live count's:
    afterwards a short chat at the same live count still promotes (rc-20261124i: they closed the gate, natural c1 ran
    plain at 61 vs 97 tok/s), and speculation at 250K itself pays at 2.4 tokens a verify"""
    import time as _time

    m = _gate_case(monkeypatch, 1, 250000, 1)
    clock = [100.0]
    monkeypatch.setattr(_time, "perf_counter", lambda: clock[0])
    m._pay["last_t"] = clock[0]
    for kind, ms in (("plain", 23.3), ("verify", 44.4)):
        for _ in range(A.PAY_MIN + 48):
            clock[0] += ms / 1000
            m._pay["kind"], m._pay["n"], m._pay["ctx"] = kind, 1, 250000
            pays = m._measure([2.4])
    assert pays is True, m._pay_line()
    assert abs(m._pay["spec_ms"][1] - 28.15) < 0.5 and abs(m._pay["plain_ms"][1] - 16.55) < 0.5, m._pay
    m.history[0], m.prompt_len[0] = [1] * 300, 299
    ok, line = m._promotion_pays([0])
    assert ok, line


def test_promotion_gate_blocks_moves(monkeypatch):
    """a gate that says no: nobody moves, the mode stays SPEC (steps run plain); a measured spec step that saves
    nothing keeps it closed for a single short prompt too"""
    m = _gate_case(monkeypatch, 8, 2048, 1)
    m._promote()
    assert not m.spec_of and not m.ae.moves and m.mode == "SPEC"
    m = _gate_case(monkeypatch, 1, 300, 1)
    m._pay["spec_ms"][1] = 2.7 * 17.0  # (no faster than the 16.5 ms plain step at 2.7 tokens a verify)
    m._promote()
    assert not m.spec_of and not m.ae.moves


def _with_swap(monkeypatch, m, pf):
    """the class's plain engine with the snapshot calls and a swap store (Strata S3)"""
    from qwen36_resident.prefix_cache import SwapStore

    old = m.engine
    m.engine = A._PlainRowsEngine(
        F.Contract(pf, F.SWAP_API + ("launch_rows",)), old.pp, torch.eye(V), V, V, B, CHUNK_UNIT, None
    )
    m.engine.swap = SwapStore(1 << 40)
    return m.engine.swap


@pytest.mark.parametrize("while_spec", [True, False], ids=["spec-user", "plain-user"])
def test_preempted_user_swaps_and_resumes(monkeypatch, while_spec):
    """Strata S3 on batch-dflash2's class: a preempted greedy user (in the spec build, or plain) keeps its state in the
    swap store keyed by the class's exact history (the runner's record lags), and its replay restores it into a plain
    row — no prefill; its tokens stay the reference's (the fakes check the restored history and KV)"""
    m, pf = make(monkeypatch)
    sw = _with_swap(monkeypatch, m, pf)
    run = Runner(m)
    run.arrive([0, 1], [[3, 1, 4, 1, 5, 9, 2, 6], [2, 7, 1, 8, 2, 8]])
    for _ in range(6):
        run.step()
    if not while_spec:
        m.mode = "PLAIN"
        run.step()
        run.step()
    assert (0 in m.spec_of) is while_spec, (m.spec_of, m.mode)
    tokens = list(run.req[0]["tokens"])
    run.req.pop(0)
    m.preempt_request(0, token_ids=tokens[:-2])  # (the runner's record two draws behind)
    assert sw.stats["inserts"] == 1 and 0 not in m.spec_of
    restores = pf.restores
    lg = m.prefill_forward(torch.tensor([tokens]), torch.tensor([run.table(0)]), None, [len(tokens)], empty_slots=[0])
    assert sw.stats["hits"] == 1 and pf.restores == restores + 1, sw.stats
    run.req[0] = dict(prompt=list(tokens), tokens=tokens + [int(lg[0, 0].argmax())], drafts=[], acc=1)
    for _ in range(4):
        run.step()
    run.req[0]["prompt"] = run.req[0]["prompt"][:8]  # (the reference from the original prompt)
    run.check()


def _fake_spec(caps=None, window=2048):
    """ResidentDFlashRows' P2d captures as bookkeeping only (parts without tensors)"""

    def capture_taps(n, s0=None):
        s0 = max(0, n - window) // 64 * 64 if s0 is None else s0
        if caps is not None:
            caps.append(n)
        hb = n // 64 * 64
        return dict(parts=[dict(s0=s0, hb=hb, main=[])] if hb > s0 else [], n=n, tail=None if n == hb else "rows")

    return SimpleNamespace(
        window=window, capture_taps=capture_taps, capture_part=lambda s0, n: dict(s0=s0, hb=n, main=[])
    )


def test_p2d_prefix_cache_entry_taps(monkeypatch):
    """P2d-2: an entry at boundary H keeps its window's taps (the ring's when the pass covered it, else the restored
    entry's parts + the ring's); a hit's request record joins the entry's parts to the pass's, usable by a promotion;
    parts that do not reach the window start make no record"""
    from qwen36_resident.adaptive import taps_usable

    m, pf = make(monkeypatch)
    m.ae.spec = _fake_spec()
    # turn 1 (a miss, the pass from 0) stores at 11264: its window 9216 .. 11264 from the ring
    e1 = m._entry_taps(None, 0, 11264)
    assert [(p["s0"], p["hb"]) for p in e1] == [(9216, 11264)]
    # turn 2 hits 11264, prefills 11264 .. 12125: its record joins the entry's part and the ring's
    m._on_taps(3, 11264, 12125, e1)
    rec = m.taps_of[3]
    assert [(p["s0"], p["hb"]) for p in rec["parts"]] == [(9216, 11264), (11264, 12096)], rec
    assert taps_usable(rec, 12126, 2048) and taps_usable(rec, 12125 + 8, 2048) and not taps_usable(rec, 12125 + 9, 2048)
    # an entry stored at 12288 after that pass: the old part reaching its window (10240 ..) + the ring's
    e2 = m._entry_taps(e1, 11264, 12288)
    assert [(p["s0"], p["hb"]) for p in e2] == [(9216, 11264), (11264, 12288)]
    # a hit whose entry kept no taps (or a gap): no record, the promotion refills
    m._on_taps(4, 11264, 12125, None)
    assert 4 not in m.taps_of
    m._on_taps(4, 11264, 14000, e1)  # (the window starts at 11904: e1 does not reach it, but the ring does not either)
    assert 4 not in m.taps_of or taps_usable(m.taps_of[4], 14001, 2048)


def test_p2d_promotions_use_the_prefill_taps(monkeypatch):
    """P2d: a prompt's prefill reports its taps (RowsEngine.on_taps); the class keeps the window's capture and the
    promotion hands it to move_to_spec_many (no PP refill for that user); the gate counts it as a fill; a released
    slot drops it; a pass that did not cover the window (a restored prefix past its start) keeps none"""
    m, pf = make(monkeypatch)
    caps = []
    m.ae.spec = _fake_spec(caps)
    m.engine.on_taps = m._on_taps
    run = Runner(m)
    run.arrive([0], [[3, 1, 4, 1, 5, 9, 2, 6]])  # (one prompt a call: a batched gang's pass keeps no taps)
    run.arrive([1], [[2, 7, 1, 8, 2, 8]])
    assert sorted(m.taps_of) == [0, 1] and caps == [7, 5], (m.taps_of, caps)
    ok, line = m._promotion_pays([0, 1])
    assert "(0 tokens)" in line, line  # (both quick: no PP-pass tokens in the refill estimate)
    for _ in range(3):
        run.step()
    assert set(m.spec_of) == {0, 1} and sorted(m.ae.taps_used) == [0, 1] and not m.taps_of, (m.spec_of, m.ae.taps_used)
    m._on_taps(5, 7168, 8192)  # (the pass started past the window's start, 6144: nothing kept)
    assert 5 not in m.taps_of
    run.arrive([2], [[1, 6, 1, 8, 0, 3, 3]])
    assert 2 in m.taps_of
    run.finish(2)
    assert 2 not in m.taps_of


def test_no_promotion_with_a_step_in_flight(monkeypatch):
    """async scheduling submits ordinary step N + 1 before step N's propose: the plain build is a token ahead of the
    history there, so a promotion (which moves the plain state with the history) waits for a propose with no step in
    flight"""
    m, pf = make(monkeypatch)
    run = Runner(m)
    prompts = [[(r * 11 + i * 5) % V for i in range(40)] for r in range(2)]
    run.arrive([0, 1], prompts)
    m._calls[0] += 1  # (a step submitted ahead of the next propose)
    for _ in range(4):
        run.step()
        assert not m.spec_of, "promoted with a step in flight"
    m._calls[0] -= 1
    run.step()
    assert len(m.spec_of) == 2
    for _ in range(4):
        run.step()
    run.check()
