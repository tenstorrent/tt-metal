# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of the DB3 spec rows vLLM class (qwen36_resident/vllm_spec_rows.py) on a fake SpecRowsSession: the
runner-model spec contract's calls (prefill -> verify [B, 1+K] -> host greedy accept -> propose) as vllm-tt-plugin
drives them, padding rows / placeholder columns, a draftless row joining a speculating batch, release and slot reuse,
a slot remap. The fake keeps SpecRowsSession's position bookkeeping (and asserts it), its "model" is a fixed greedy
chain next(token, position), its drafter that chain with a planted miss."""
import pytest
import torch

pytest.importorskip("vllm_tt_plugin.spec_decode")

from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID, DraftOutput, SpecPlan, SpecReject, VerifyOutput

from qwen36_resident import vllm_spec_rows as V
from qwen36_resident.vllm_rows import ResidentRowsAdapter

VOCAB = 1000
U, R = V.SPEC_USERS, V.SPEC_R
K = R - 1


def nxt(t, p):
    return (int(t) * 31 + int(p) * 7 + 3) % VOCAB


class FakeEngine:
    def __init__(self):
        self.U, self.R = U, R
        self.m = self
        self.block_tables = [[0] for _ in range(U * R)]
        self.writes = 0

    def set_page_table(self, tables):
        self.block_tables = [list(t) for t in tables]
        self.writes += 1


class FakeSession:
    """SpecRowsSession's interface and bookkeeping; verify / propose follow nxt (the drafter misses every 5th draft)"""

    def __init__(self):
        self.e = FakeEngine()
        self.pos, self.started, self.last, self.committed = [0] * U, [False] * U, [None] * U, [1] * U
        self.calls = []

    def start(self, u, ids, table=None):
        if table is not None:
            for r in range(u * R, (u + 1) * R):
                self.e.block_tables[r] = list(table)
        self.pos[u], self.started[u], self.last[u], self.committed[u] = len(ids), True, None, 1
        lg = torch.zeros(VOCAB)
        lg[nxt(ids[-1], len(ids) - 1)] = 5.0
        return lg

    def start_many(self, items):
        return [self.start(u, ids, table) for u, ids, table in items]

    def verify(self, tokens, positions, active):
        self.calls.append(("verify", [list(t) for t in tokens], list(positions), list(active)))
        g = [[0] * R for _ in range(U)]
        for u in range(U):
            if not active[u]:
                continue
            assert int(positions[u]) == self.pos[u], (u, positions[u], self.pos[u])
            g[u] = [nxt(tokens[u][k], positions[u] + k) for k in range(R)]
            self.last[u], self.started[u] = int(positions[u]), False
        return g

    def propose(self, anchors, positions, accepted, active):
        self.calls.append(("propose", list(anchors), list(positions), list(accepted), list(active)))
        out = []
        for u in range(U):
            if active[u] and not self.started[u] and self.last[u] is not None:
                self.committed[u] = int(accepted[u])
                self.pos[u] = self.last[u] + self.committed[u]
            if active[u]:
                assert int(positions[u]) == self.pos[u], (u, positions[u], self.pos[u])
            d, t, p = [], anchors[u], positions[u]
            for k in range(K):
                t = nxt(t, p + k) if (p + k) % 5 else (nxt(t, p + k) + 1) % VOCAB
                d.append(t)
            out.append(d)
        return out


def make(max_batch=U):
    m = object.__new__(V.Qwen36ResidentSpecRowsForCausalLM)
    ResidentRowsAdapter.__init__(m, None, VOCAB, 4096, U, 1024, False, rows=U)
    m.sess = FakeSession()
    m.engine = V._SpecEngine(m.sess.e)
    return m


def greedy_ref(prompt, n):
    out, t, p = [], prompt[-1], len(prompt) - 1
    for _ in range(n):
        t = nxt(t, p)
        p += 1
        out.append(t)
    return out


class Runner:
    """the plugin's side: requests in vLLM slots, a padded batch of B rows, host greedy accept, the committed block"""

    def __init__(self, m, B=U):
        self.m, self.B = m, B
        self.req = {}  # vslot -> dict(prompt, gen, pos, acc, drafts)
        self.block = {}  # vslot -> (committed tokens, positions)

    def prefill(self, vslot, prompt, blocks):
        n = len(prompt)
        tok = torch.zeros(1, n, dtype=torch.long)
        tok[0] = torch.tensor(prompt)
        pt = torch.zeros(1, 64, dtype=torch.int32)
        pt[0, : len(blocks)] = torch.tensor(blocks)
        lg = self.m.prefill_forward(tok, pt, None, [n], empty_slots=[vslot])
        t = int(lg[0, 0].argmax())
        self.req[vslot] = dict(prompt=prompt, gen=[t], pos=n, acc=1, drafts=None, blocks=blocks)
        self.block[vslot] = ([t], [n])

    def propose(self):
        ct = torch.zeros(self.B, R, dtype=torch.int32)
        cp = torch.full((self.B, R), -1, dtype=torch.int32)
        acc = torch.ones(self.B, dtype=torch.int32)
        for v, (ts, ps) in self.block.items():
            ct[v, : len(ts)], cp[v, : len(ps)], acc[v] = torch.tensor(ts), torch.tensor(ps), len(ts)
        out = self.m.propose_draft_tokens(K, ct, cp, acc)
        assert isinstance(out, DraftOutput) and out.draft_token_ids.shape == (self.B, K)
        for v in self.req:
            self.req[v]["drafts"] = out.draft_token_ids[v].tolist()[: int(out.num_valid[v])]

    def verify(self, rows=None, drop_drafts=()):
        rows = sorted(self.req) if rows is None else rows
        tok = torch.zeros(self.B, R, dtype=torch.long)
        pos = torch.full((self.B, R), -1, dtype=torch.int32)
        nvd = torch.zeros(self.B, dtype=torch.int32)
        acc = torch.ones(self.B, dtype=torch.int32)
        pt = torch.zeros(self.B, 64, dtype=torch.int32)
        for v in rows:
            q = self.req[v]
            d = [] if v in drop_drafts or q["drafts"] is None else q["drafts"]
            tok[v, 0], pos[v, 0] = q["gen"][-1], q["pos"]
            tok[v, 1:] = PLACEHOLDER_TOKEN_ID
            for k, x in enumerate(d):
                tok[v, 1 + k], pos[v, 1 + k] = x, q["pos"] + 1 + k
            nvd[v], acc[v] = len(d), q["acc"]
            pt[v, : len(q["blocks"])] = torch.tensor(q["blocks"])
        out = self.m.decode_forward(tok, pos, pt, num_valid_drafts=nvd, accepted_counts=acc, spec_mode="argmax_ids")
        assert isinstance(out, VerifyOutput) and out.argmax_ids.shape == (self.B, R)
        self.block = {}
        for v in rows:
            q, g = self.req[v], out.argmax_ids[v].tolist()
            d = tok[v, 1 : 1 + int(nvd[v])].tolist()
            k = 0
            while k < len(d) and d[k] == g[k]:
                k += 1
            new = d[:k] + [g[k]]
            self.block[v] = (new, [q["pos"] + 1 + j for j in range(len(new))])
            q["gen"] += new
            q["pos"] += len(new)
            q["acc"] = len(new)
        return out


def test_spec_plan():
    p = V.Qwen36ResidentSpecRowsForCausalLM.spec_plan(None, U, 7)
    assert isinstance(p, SpecPlan) and p.effective_k == K and p.accept_modes == ("argmax_ids",)
    assert V.Qwen36ResidentSpecRowsForCausalLM.spec_plan(None, 2, 1).effective_k == 1
    assert isinstance(V.Qwen36ResidentSpecRowsForCausalLM.spec_plan(None, U + 1, K), SpecReject)
    caps = V.Qwen36ResidentSpecRowsForCausalLM.model_capabilities
    assert caps["supports_spec_decode"] and caps["spec_requirements"] == ["device_propose"]


def test_protocol_matches_greedy():
    m = make()
    run = Runner(m)
    prompts = {v: [(v * 97 + i * 13) % VOCAB for i in range(20 + 7 * v)] for v in range(U)}
    for v in range(U - 2):
        run.prefill(v, prompts[v], [1 + 4 * v, 2 + 4 * v])
    for step in range(30):
        if step == 3:  # two requests join a speculating batch: their first verify carries no drafts
            for v in range(U - 2, U):
                run.prefill(v, prompts[v], [1 + 4 * v, 2 + 4 * v])
        run.verify()
        run.propose()
    for v in range(U):
        g = run.req[v]["gen"]
        assert g == greedy_ref(prompts[v], len(g)), v
    # some drafts were accepted (more than one token per step on average) and some missed
    assert sum(len(run.req[v]["gen"]) for v in range(U)) > 30 * U


def test_padding_rows_and_placeholders():
    m = make()
    run = Runner(m)
    run.prefill(2, [5, 6, 7], [3])
    run.verify()  # (no drafts yet: columns 1..K are placeholders, every other row padding)
    v = m.sess.calls[-1]
    assert v[3] == [u == m.slots.res_of[2] for u in range(U)]
    assert all(0 <= t < VOCAB for t in v[1][m.slots.res_of[2]])
    run.propose()
    run.verify(drop_drafts=(2,))  # (a row whose drafts the runner dropped: num_valid_drafts 0)
    run.propose()
    run.verify()
    g = run.req[2]["gen"]
    assert g == greedy_ref([5, 6, 7], len(g))


def test_release_and_reuse():
    m = make()
    run = Runner(m)
    run.prefill(0, [1, 2, 3], [4])
    run.prefill(1, [9, 8], [5])
    run.verify()
    run.propose()
    r0 = m.slots.res_of[0]
    m.release_request(0)
    del run.req[0]
    run.block.pop(0, None)
    assert r0 not in m.live
    run.prefill(0, [4, 4, 4, 4], [6])
    for _ in range(4):
        run.verify()
        run.propose()
    for v, p in ((0, [4, 4, 4, 4]), (1, [9, 8])):
        assert run.req[v]["gen"] == greedy_ref(p, len(run.req[v]["gen"]))


def test_page_table_update():
    m = make()
    run = Runner(m)
    run.prefill(0, [1, 2, 3], [4])
    w = m.sess.e.writes
    run.verify()
    assert m.sess.e.writes == w  # (same table: no rewrite)
    run.req[0]["blocks"] = [4, 7]
    run.propose()
    run.verify()
    assert m.sess.e.writes == w + 1
    u = m.slots.res_of[0]
    assert all(m.sess.e.block_tables[r][:2] == [4, 7] for r in range(u * R, (u + 1) * R))


def test_slot_remap():
    m = make()
    run = Runner(m)
    run.prefill(0, [1, 2, 3], [4])
    run.prefill(1, [7, 7], [5])
    run.verify()
    run.propose()
    r0, r1 = m.slots.res_of[0], m.slots.res_of[1]
    # the runner swaps the two requests' rows: row i reads the state in slot remap[i]
    remap = list(range(U))
    remap[0], remap[1] = 1, 0
    run.req = {0: run.req[1], 1: run.req[0]}
    run.block = {0: run.block[1], 1: run.block[0]}
    tok = torch.zeros(U, R, dtype=torch.long)
    pos = torch.full((U, R), -1, dtype=torch.int32)
    for v in (0, 1):
        q = run.req[v]
        tok[v, 0], pos[v, 0] = q["gen"][-1], q["pos"]
        for k, x in enumerate(q["drafts"]):
            tok[v, 1 + k], pos[v, 1 + k] = x, q["pos"] + 1 + k
    nvd = torch.tensor([K, K] + [0] * (U - 2), dtype=torch.int32)
    acc = torch.tensor([run.req[0]["acc"], run.req[1]["acc"]] + [1] * (U - 2), dtype=torch.int32)
    m.decode_forward(
        tok, pos, None, num_valid_drafts=nvd, accepted_counts=acc, spec_mode="argmax_ids", slot_remap=remap
    )
    assert m.slots.res_of[0] == r1 and m.slots.res_of[1] == r0
    m.note_state_slots_moved({1: 0, 0: 1})
