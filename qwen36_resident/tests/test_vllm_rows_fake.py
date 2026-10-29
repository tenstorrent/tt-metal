# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU test of the multi-user vLLM adapter (vllm_rows.py) on a fake rows-mode ResidentModel and PP prefill.

The fakes follow the host-facing contract of .omc/research/resident-multiuser-api.md and check what the device
would get wrong silently:
- each resident slot's state is the token history it consumed; a step at position p must come at p == len(history);
- the KV pool is written and read through the page tables: every step re-reads positions 0..p of its slot through
  its table and compares them with the slot's history (a wrong block table or a shared block shows up here);
- a launch without preserve clobbers the conv history of its inactive rows; a live slot whose history was clobbered
  fails its next step (the adapter must keep live slots active, or set preserve);
- the PP prefill holds one prompt; a resumed span must continue exactly where the last one stopped;
- the adapter sees each fake through Contract: the contract's public names only, read-only (MODEL_API, PP_API).
Each row's next token is a function of its whole history (and, sampled, of its seed and position), so a slot mix-up
anywhere changes the tokens the driver checks against its own reference.

The driver plays vLLM's part: arrivals of mixed prompt lengths, chunked prefill (exclusive, aligned chunks), decode
steps over a condensing persistent batch, aborts, preemptions with replay, skipped decoders, a block pool with
random physical blocks. Slot bookkeeping is the real plugin's (TTModelRunner._alloc_prefill_state_slots,
_decode_state_slot_remap, note_decode_state_slots_settled, _release_model_request, _release_dead_state_slots).

usage: PYTHONPATH=<vllm-tt-plugin resident-mu src>:<tt-metal> python -m pytest --noconftest -c /dev/null \
    qwen36_resident/tests/test_vllm_rows_fake.py   (a venv with vLLM; no device is opened)
"""
import random
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from qwen36_resident.vllm_rows import BLOCK, ROWS, ResidentRowsAdapter, RowsEngine, SlotMap

model_runner = pytest.importorskip("vllm_tt_plugin.model_runner")
TTModelRunner = model_runner.TTModelRunner
raises = pytest.raises  # allow-pytest.raises: run with --noconftest, there is no expect_error fixture

V = 61  # vocab (= hidden: the fake embedding is the identity, so x0 decodes back to its token)


def ref_next(seq, sampled=None, vocab=V):
    """the token after seq: greedy, or (temperature > 0) mixed with the row's seed and position"""
    h = (sum((i * 7 + 3) * int(t) for i, t in enumerate(seq)) + 11 * len(seq)) % vocab
    if sampled is not None:
        seed, pos = sampled
        h = (h + 13 * int(seed) + 5 * int(pos)) % vocab
    return h


class FakeTok:
    def __init__(self, x0, pos, active, preserve):
        self.x0, self.pos, self.active, self.preserve = x0.clone(), list(pos), list(active), bool(preserve)


class FakeModel:
    """a rows-mode ResidentModel: launches of ROWS slots, paged KV pool, hub feed when `sample`"""

    def __init__(self, n_slots, sample, vocab=V):
        self.S, self.sample, self.V = n_slots, sample, vocab
        self.hist = [None] * n_slots  # token history per slot (None: never prefilled)
        self.conv_ok = [False] * n_slots
        self.pool = {}  # (physical block, offset) -> (slot, token)
        self.tables = [[0] for _ in range(n_slots)]
        self.tok = [None] * (n_slots // ROWS)
        self.lg = torch.zeros(n_slots, 1, vocab)
        self.out = torch.zeros(n_slots, 4, dtype=torch.int64)  # per slot its launch row's out word
        self.params = [(0.0, 1, 1.0, 0)] * n_slots
        self.steps = 0

    def token_state_rows(self, rows, preserve=False):
        """one launch's token state: rows {u: (x0, position)} active, the others inactive (x0 0, position 0)"""
        assert rows, "a launch needs at least one active user"
        x0, pos, active = torch.zeros(ROWS, self.V), [0] * ROWS, [False] * ROWS
        for u, (x, p) in rows.items():
            x0[u], pos[u], active[u] = x, int(p), True
        return FakeTok(x0, pos, active, preserve)

    def set_page_table(self, tables):
        assert len(tables) == self.S
        self.tables = [list(t) for t in tables]

    def clear_slot(self, r):
        self.hist[r], self.conv_ok[r] = [], True

    def _kv(self, r, p):
        t = self.tables[r]
        assert p // BLOCK < len(t), f"slot {r}: position {p} past its block table {t}"
        return t[p // BLOCK], p % BLOCK

    def step(self, tok_hosts, launches):
        if tok_hosts is not None:
            assert len(tok_hosts) == self.S // ROWS
            for k, t in enumerate(tok_hosts):
                if t is not None:
                    self.tok[k] = t
        for k in launches:
            t = self.tok[k]
            assert t is not None and any(t.active), f"launch {k} replayed with no active row"
            for u in range(ROWS):
                r = k * ROWS + u
                if not t.active[u]:
                    if not t.preserve:
                        self.conv_ok[r] = False  # a non-preserve launch rewrites every row's conv history
                    self.out[r] = torch.tensor([0, t.pos[u], 0, 0])
                    continue
                tok, p = int(t.x0[u].argmax()), t.pos[u]
                assert float(t.x0[u].sum()) == 1.0, f"slot {r}: x0 is not one embedding row"
                h = self.hist[r]
                assert h is not None and self.conv_ok[r], f"slot {r}: no state or clobbered conv history"
                assert p == len(h), f"slot {r}: step at position {p}, history has {len(h)} tokens"
                self.pool[self._kv(r, p)] = (r, tok)
                seq = h + [tok]
                for q in range(p + 1):
                    got = self.pool.get(self._kv(r, q))
                    assert got == (r, seq[q]), f"slot {r}: KV at position {q} is {got}, want {(r, seq[q])}"
                self.hist[r] = seq
                temp, _, _, seed = self.params[r]
                nxt = ref_next(seq, (seed, p) if temp > 0 else None, self.V)
                self.lg[r] = 0
                self.lg[r, 0, ref_next(seq, None, self.V)] = 100.0  # logits: the greedy token, sharply (host sampling)
                if self.sample:
                    self.out[r] = torch.tensor([nxt, p + 1, 1, 0])
                    t.x0[u] = torch.nn.functional.one_hot(torch.tensor(nxt), self.V).float()  # the hub's feed
                    t.pos[u] = p + 1
        self.steps += 1

    def logits(self, launches):
        """[8 * len(launches), 1, vocab] in the order given"""
        return torch.cat([self.lg[k * ROWS : (k + 1) * ROWS] for k in launches]).clone()

    def out_tensor(self, k):
        """launch k's out tensor (8 rows x (token, position + 1, active, 0)); a fake's is already on the host"""
        return self.out[k * ROWS : (k + 1) * ROWS]

    def set_sampling(self, rows):
        assert len(rows) == self.S
        self.params = list(rows)


class FakePP:
    """the PP prefill: one prompt; spans must continue where the last stopped. Counts the resumed spans with model
    launches since the previous span (the interleave whose device safety split_prefill_gate.py checks)"""

    def __init__(self, unit, model):
        self.unit, self.ids, self.done = unit, None, False
        self.model, self.spans, self.interleaved, self.mark = model, 0, 0, None

    def reset(self):
        self.ids, self.done = [], False

    def run(self, ids):
        assert self.ids == [], "run() after a reset only"
        self.ids, self.done = [int(t) for t in ids], True

    def run_span(self, ids, first_chunk, n_tokens, drain=True, chunk=None):
        assert chunk == self.unit and self.ids is not None and not self.done
        assert first_chunk * chunk == len(self.ids), f"span from chunk {first_chunk}, the PP holds {len(self.ids)}"
        self.spans += 1
        self.interleaved += first_chunk > 0 and self.model.steps != self.mark
        self.mark = self.model.steps
        self.ids += [int(t) for t in ids]
        assert len(self.ids) == n_tokens
        assert drain or len(ids) % chunk == 0
        self.done = drain

    def handoff(self, model, n_tokens, block_table=None, slot=None):
        assert self.done and len(self.ids) == n_tokens, (self.done, len(self.ids), n_tokens)
        assert isinstance(model, Contract) and model._obj is self.model, "handoff into another model"
        model = self.model
        assert len(block_table) * BLOCK >= n_tokens
        model.hist[slot], model.conv_ok[slot] = list(self.ids), True
        model.tables[slot] = list(block_table)
        for q in range(n_tokens):
            model.pool[(block_table[q // BLOCK], q % BLOCK)] = (slot, self.ids[q])


MODEL_API = (
    "sample",
    "token_state_rows",
    "set_page_table",
    "clear_slot",
    "step",
    "logits",
    "out_tensor",
    "set_sampling",
)
PP_API = ("reset", "run", "run_span", "handoff")


class Contract:
    """what the adapter sees of a fake: the contract's public API only, read-only (the fake's bookkeeping - histories,
    the KV pool, conv flags - is out of its reach, so the adapter cannot pass by reading or patching it)"""

    def __init__(self, obj, api):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_api", frozenset(api))

    def __getattr__(self, name):
        if name not in self._api:
            raise AttributeError(f"{type(self._obj).__name__}.{name} is not in the contract")
        return getattr(self._obj, name)

    def __setattr__(self, name, value):
        raise AttributeError(f"the adapter set {name} on {type(self._obj).__name__}")


def make_adapter(max_seqs, unit, sample, vocab=V, max_len=4096, cls=ResidentRowsAdapter):
    """the adapter on contract views of a fresh FakeModel / FakePP; returns (adapter, the raw FakeModel)"""
    n_slots = -(-max_seqs // ROWS) * ROWS
    model = FakeModel(n_slots, sample, vocab)
    pp = FakePP(unit, model)
    engine = RowsEngine(Contract(model, MODEL_API), Contract(pp, PP_API), torch.eye(vocab), vocab, vocab, n_slots, unit)
    return cls(engine, vocab, max_len, max_seqs, unit), model


class Req:
    def __init__(self, rid, prompt, max_tokens, sampled):
        self.rid, self.tokens, self.n_prompt = rid, list(prompt), len(prompt)
        self.max_tokens, self.out, self.computed, self.blocks = max_tokens, 0, 0, []
        self.sampled = sampled  # (temperature, seed) or None


class Sim:
    """vLLM's side of the contract, with the plugin's own slot bookkeeping"""

    def __init__(self, seed, max_seqs=12, unit=128, nb=600, sample=False, max_len=2048, max_tokens=(1, 40), sync=False):
        self.rng = random.Random(seed)
        # sync: the runner without async scheduling hands decode_forward's handle straight to
        # process_decode_output_host (finalize_decode); async: read_decode_output(async_read=True) first
        self.sync = sync
        self.max_tokens = max_tokens
        self.adapter, self.model = make_adapter(max_seqs, unit, sample)
        self.sample, self.unit, self.max_seqs, self.max_len = sample, unit, max_seqs, max_len
        self.runner = SimpleNamespace(
            tt_per_lane_max_num_seqs=max_seqs,
            _req_state_slot={},
            _pending_state_slot_settle=None,
            _pending_state_slot_moves=None,
            requests={},
            model=self.adapter,
        )
        free = list(range(1, nb))  # vLLM's block 0 is the null block
        self.rng.shuffle(free)
        self.free_blocks = free
        self.waiting, self.rows, self.reqs = deque(), [], {}
        self.partial, self.next_id, self.chunk_due = None, 0, True
        self.last_rows, self.last_tokens, self.last_table, self.layout_changed = None, None, None, True
        self.stats = dict(prefill=0, chunks=0, decode=0, tokens=0, finished=0, preempted=0, aborted=0, skipped=0)

    # ---- requests
    def arrive(self):
        n = self.rng.choice([1, 2, 5, 63, 64, 65, 127, 128, 129, 200, 300, 513, 700, 1100, 1500])
        prompt = [self.rng.randrange(V) for _ in range(n)]
        sampled = (0.8, self.rng.randrange(1000)) if self.sample and self.rng.random() < 0.4 else None
        r = Req(f"r{self.next_id}", prompt, self.rng.randint(*self.max_tokens), sampled)
        self.next_id += 1
        self.reqs[r.rid] = r
        self.waiting.append(r)

    def _free(self, r):
        self.free_blocks += r.blocks
        r.blocks = []

    def _leave_rows(self, rid):
        if rid in self.rows:
            i = self.rows.index(rid)
            last = self.rows.pop()
            if last != rid:
                self.rows[i] = last  # condense: the last row fills the hole
            self.layout_changed = True

    def finish(self, rid, aborted=False):
        r = self.reqs.pop(rid)
        TTModelRunner._release_model_request(self.runner, rid)
        TTModelRunner._release_dead_state_slots(
            self.runner, SimpleNamespace(finished_req_ids=[rid], preempted_req_ids=[])
        )
        self.runner.requests.pop(rid, None)
        self._leave_rows(rid)
        if self.partial == rid:
            self.partial = None
        self._free(r)
        self.stats["aborted" if aborted else "finished"] += 1

    def preempt(self, rid):
        r = self.reqs[rid]
        TTModelRunner._release_model_request(self.runner, rid)
        TTModelRunner._release_dead_state_slots(
            self.runner, SimpleNamespace(finished_req_ids=[], preempted_req_ids=[rid])
        )
        self._leave_rows(rid)
        if self.partial == rid:
            self.partial = None
        self._free(r)
        r.computed = 0  # replays prompt + outputs through prefill
        self.waiting.appendleft(r)
        self.stats["preempted"] += 1

    def _blocks(self, r, n_pos):
        while len(r.blocks) * BLOCK < n_pos:
            r.blocks.append(self.free_blocks.pop())

    def _table(self, rs):
        w = max(len(r.blocks) for r in rs)
        t = torch.zeros(len(rs), w + 3, dtype=torch.int32)  # padded like the runner's block tables
        for i, r in enumerate(rs):
            t[i, : len(r.blocks)] = torch.tensor(r.blocks, dtype=torch.int32)
        return t

    # ---- steps
    def prefill_step(self, plan):
        """plan: [(request, end)] in step order"""
        rids = [r.rid for r, _ in plan]
        continuing = {r.rid for r, _ in plan if r.computed > 0 and r.rid in self.runner._req_state_slot}
        if continuing:
            slots = TTModelRunner._alloc_prefill_state_slots(self.runner, rids, continuing=continuing)
        else:
            slots = TTModelRunner._alloc_prefill_state_slots(self.runner, rids)
        self.runner.requests.update(dict.fromkeys(rids))
        for r, end in plan:
            self._blocks(r, end)
        width = max(end for _, end in plan)
        tokens = torch.zeros(len(plan), width, dtype=torch.int64)
        for i, (r, end) in enumerate(plan):
            tokens[i, :end] = torch.tensor(r.tokens[:end])
        out = self.adapter.prefill_forward(
            tokens,
            self._table([r for r, _ in plan]),
            [],
            torch.tensor([end for _, end in plan]),
            start_pos=torch.tensor([r.computed for r, _ in plan]),
            empty_slots=slots,
            prefill_final_mask=[end == len(r.tokens) for r, end in plan],
        )
        assert out.shape == (len(plan), 1, V)
        for i, (r, end) in enumerate(plan):
            final = end == len(r.tokens)
            r.computed = end
            self.stats["chunks" if not final or r.rid == self.partial else "prefill"] += 1
            if not final:
                self.partial = r.rid
                continue
            if self.partial == r.rid:
                self.partial = None
            tok = int(out[i, 0].argmax())  # the runner samples the prefill logits on the host (greedy)
            assert tok == ref_next(r.tokens), f"{r.rid}: first token {tok}, want {ref_next(r.tokens)}"
            self._emit(r, tok)
            if r.rid in self.reqs:
                self.rows.append(r.rid)
        self.layout_changed = True

    def _emit(self, r, tok):
        r.tokens.append(tok)
        r.out += 1
        self.stats["tokens"] += 1
        if r.out >= r.max_tokens or len(r.tokens) >= self.max_len:
            self.finish(r.rid)

    def decode_step(self, rids):
        rs = [self.reqs[rid] for rid in rids]
        remap = TTModelRunner._decode_state_slot_remap(self.runner, rids)
        pad = self.max_seqs
        for r in rs:
            self._blocks(r, len(r.tokens))
        steady = self.sample and not self.layout_changed and remap is None and rids == self.last_rows
        tokens = torch.zeros(pad, 1, dtype=torch.int64)
        pos = torch.full((pad,), -1, dtype=torch.int64)
        for i, r in enumerate(rs):
            tokens[i, 0] = r.tokens[-1]
            pos[i] = len(r.tokens) - 1
        if steady:
            # contract 1: tokens / start_pos one step behind on a step without reload_inputs
            tokens, pos = self.last_tokens
        table = self._table(rs)
        table = torch.cat([table, torch.zeros(pad - len(rs), table.shape[1], dtype=torch.int32)])
        sp = None
        if self.sample:
            temp = [r.sampled[0] if r.sampled else 0.0 for r in rs] + [0.0] * (pad - len(rs))
            seed = [r.sampled[1] if r.sampled else 0 for r in rs] + [0] * (pad - len(rs))
            sp = SimpleNamespace(temperature=temp, top_k=[20] * pad, top_p=[0.95] * pad, seed=seed)
        # the runner's page-table-only refresh: a steady step whose block allocation changed
        refresh = steady and not torch.equal(table, self.last_table)
        handle = self.adapter.decode_forward(
            tokens,
            pos,
            table,
            reload_inputs=not steady,
            reload_page_table=refresh,
            reload_sampling_params=self.sample and not steady,
            reset_sampling_state=self.sample and not steady,
            read_from_device=False,
            sampling_params=sp,
            slot_remap=remap,
        )
        TTModelRunner.note_decode_state_slots_settled(self.runner)
        if self.sync:
            res = self.adapter.process_decode_output_host(handle, is_tokens=self.sample)
        else:
            host, _events = self.adapter.read_decode_output(handle, async_read=True)
            res = self.adapter.process_decode_output_host(host, is_tokens=self.sample)
        self.last_rows, self.last_tokens, self.last_table, self.layout_changed = list(rids), (tokens, pos), table, False
        for i, r in enumerate(rs):
            if self.sample:
                tok = int(res[i, 0])
                want = ref_next(r.tokens, (r.sampled[1], len(r.tokens) - 1) if r.sampled else None)
            else:
                tok = int(res[i, 0].argmax())
                want = ref_next(r.tokens)
            assert tok == want, f"{r.rid} at {len(r.tokens)}: token {tok}, want {want}"
            self._emit(r, tok)
        self.stats["decode"] += 1

    def step(self, quiet=False):
        """quiet: no arrivals, aborts, preemptions or row moves (steady decode)"""
        rng = self.rng
        if not quiet and rng.random() < 0.25 and len(self.reqs) < 40:
            self.arrive()
        live = [] if quiet else [rid for rid in self.rows]
        if live and rng.random() < 0.03:
            self.finish(rng.choice(live), aborted=True)
        elif live and rng.random() < 0.03:
            self.preempt(rng.choice(live))
        elif not quiet and self.partial and rng.random() < 0.01:
            self.preempt(self.partial)
        if live and rng.random() < 0.05:
            rng.shuffle(self.rows)  # evict / re-add moves rows around
            self.layout_changed = True
        running = len(self.rows) + (1 if self.partial else 0)
        decoders = bool(self.rows)
        C = self.unit
        if self.partial:
            # exclusive chunk steps, one decode step between chunks while requests decode
            if decoders and not self.chunk_due:
                self.chunk_due = True
            else:
                r = self.reqs[self.partial]
                end = min(r.computed + C, len(r.tokens)) if decoders else len(r.tokens)
                self.chunk_due = False
                self.prefill_step([(r, end)])
                return
        elif self.waiting and running < self.max_seqs:
            plan, long_started = [], False
            for r in list(self.waiting):
                if running + len(plan) >= self.max_seqs:
                    break
                left = len(r.tokens) - r.computed
                if left > C and decoders:
                    if long_started:
                        continue
                    plan.append((r, C))
                    long_started = True
                else:
                    plan.append((r, len(r.tokens)))
            for r, _ in plan:
                self.waiting.remove(r)
            # the runner's row order is arbitrary; the adapter must take whole prompts before a first chunk
            rng.shuffle(plan)
            self.chunk_due = False
            self.prefill_step(plan)
            return
        if self.rows:
            rids = list(self.rows)
            if not quiet and len(rids) > 1 and rng.random() < 0.05:
                rids.pop(rng.randrange(len(rids)))  # the scheduler skips a running request for one step
                self.stats["skipped"] += 1
                self.layout_changed = True
            self.decode_step(rids)

    def run(self, steps, quiet_after=None):
        for i in range(steps):
            self.step(quiet=quiet_after is not None and i >= quiet_after)
        # every live resident slot is mapped from exactly one vLLM slot the plugin knows
        res = self.adapter.slots.res_of
        assert len(set(res.values())) == len(res)
        assert set(res) == set(self.runner._req_state_slot.values())
        assert self.adapter.live <= set(res.values())
        return self.stats


def _count_decodes(sim):
    """wrap the adapter's decode_forward to count steady steps (no reload) and page-table-only refreshes"""
    n = dict(steady=0, refresh=0, remaps=0)
    orig = sim.adapter.decode_forward

    def decode_forward(*args, **kw):
        n["steady"] += not kw["reload_inputs"]
        n["refresh"] += bool(kw["reload_page_table"])
        n["remaps"] += kw["slot_remap"] is not None
        return orig(*args, **kw)

    sim.adapter.decode_forward = decode_forward
    return n


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("sample", [False, True], ids=["host-sampled", "device-sampled"])
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
def test_churn(seed, sample, sync):
    sim = Sim(seed, sample=sample, sync=sync)
    n = _count_decodes(sim)
    stats = sim.run(700)
    assert stats["tokens"] > 1000 and stats["chunks"] > 5 and stats["finished"] > 10
    assert stats["preempted"] > 0 and stats["aborted"] > 0 and stats["skipped"] > 0 and n["remaps"] > 50


@pytest.mark.parametrize("seed", range(3))
def test_steady_device_decode(seed):
    """churn, then a quiet phase: hub-fed steps without reloads, with page-table-only refreshes as blocks fill"""
    sim = Sim(seed, sample=True, max_tokens=(200, 400))
    n = _count_decodes(sim)
    sim.run(600, quiet_after=150)
    assert n["steady"] > 150 and n["refresh"] > 5


def test_eight_users_share_one_launch():
    """lowest free resident slot: up to 8 users replay one launch, whatever vLLM slots they get"""
    sim = Sim(1, max_seqs=12)
    record = []
    orig = sim.model.step

    def step(tok_hosts, launches=None):
        record.append(list(launches))
        return orig(tok_hosts, launches)

    sim.model.step = step
    for n in [10] * 8:
        r = Req(f"x{n}{len(sim.reqs)}", [sim.rng.randrange(V) for _ in range(n)], 50, None)
        sim.reqs[r.rid] = r
        sim.prefill_step([(r, len(r.tokens))])
    sim.decode_step(list(sim.rows))
    assert record[-1] == [0]
    assert sorted(sim.adapter.slots.res_of.values()) == list(range(8))


def test_slot_map_remap_is_a_gather():
    m = SlotMap(16)
    assert [m.assign(v) for v in (0, 1, 2)] == [0, 1, 2]
    m.release(1)
    # vLLM slot 1 now reads slot 2 (2's request moved down a row)
    new = m.remapped([0, 2, 1] + list(range(3, 12)))
    assert new == {0: 0, 1: 2}
    m.res_of = new
    assert m.assign(2) == 1, "the lowest free resident slot"


def test_a_double_note_of_the_moves_is_refused():
    adapter, _ = make_adapter(4, 128, False)
    with raises(RuntimeError, match="do not match"):
        adapter.note_state_slots_moved({0: 1, 1: 0})


def test_a_continuation_beside_another_prompt_is_refused():
    adapter, _ = make_adapter(4, 128, False)
    toks = torch.zeros(2, 300, dtype=torch.int64)
    adapter.prefill_forward(
        toks[:1],
        torch.arange(1, 6)[None],
        [],
        torch.tensor([128]),
        start_pos=torch.tensor([0]),
        empty_slots=[0],
        prefill_final_mask=[False],
    )
    with raises(RuntimeError, match="holds one prompt"):
        adapter.prefill_forward(
            toks,
            torch.arange(1, 11).reshape(2, 5),
            [],
            torch.tensor([256, 5]),
            start_pos=torch.tensor([128, 0]),
            empty_slots=[0, 1],
            prefill_final_mask=[False, True],
        )


def test_an_unaligned_chunk_is_refused():
    adapter, _ = make_adapter(4, 128, False)
    with raises(RuntimeError, match="multiple of the PP chunk"):
        adapter.prefill_forward(
            torch.zeros(1, 300, dtype=torch.int64),
            torch.arange(1, 6)[None],
            [],
            torch.tensor([100]),
            start_pos=torch.tensor([0]),
            empty_slots=[0],
            prefill_final_mask=[False],
        )


def test_device_sampling_without_the_hub_feed_is_refused():
    adapter, _ = make_adapter(4, 128, False)
    adapter.prefill_forward(
        torch.ones(1, 3, dtype=torch.int64),
        torch.tensor([[1]]),
        [],
        torch.tensor([3]),
        start_pos=torch.tensor([0]),
        empty_slots=[0],
        prefill_final_mask=[True],
    )
    sp = SimpleNamespace(temperature=[0.0] * 4, top_k=[1] * 4, top_p=[1.0] * 4, seed=[0] * 4)
    with raises(RuntimeError, match="hub feed"):
        adapter.decode_forward(
            torch.ones(4, 1, dtype=torch.int64),
            torch.tensor([3, -1, -1, -1]),
            torch.tensor([[1]] * 4),
            sampling_params=sp,
        )


@pytest.mark.parametrize("sample", [False, True], ids=["host-sampled", "device-sampled"])
def test_sync_scheduling_processes_the_raw_handle(sample):
    """the plugin's synchronous path (finalize_decode) calls process_decode_output_host on decode_forward's own return
    value: no read_decode_output in between (review finding 2; the reviewer's probe_sync.py)"""
    from vllm_tt_plugin.async_decode import _is_host_decode_output

    adapter, _ = make_adapter(4, 128, sample)
    adapter.prefill_forward(
        torch.ones(1, 3, dtype=torch.int64),
        torch.tensor([[1]]),
        [],
        torch.tensor([3]),
        start_pos=torch.tensor([0]),
        empty_slots=[0],
        prefill_final_mask=[True],
    )
    sp = SimpleNamespace(temperature=[0.0] * 4, top_k=[1] * 4, top_p=[1.0] * 4, seed=[0] * 4) if sample else None
    h = adapter.decode_forward(
        torch.zeros(4, 1, dtype=torch.int64),
        torch.tensor([3, -1, -1, -1]),
        torch.tensor([[1]] * 4),
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=sample,
        reset_sampling_state=sample,
        read_from_device=False,
        sampling_params=sp,
    )
    assert not _is_host_decode_output(h), "the plugin must route the handle through process_decode_output_host"
    out = adapter.process_decode_output_host(h, is_tokens=sample)
    assert out.shape == ((4, 1) if sample else (4, 1, V))
