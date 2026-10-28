# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only replay of the B=8 / 1M, c8, staggered-arrival TTFT collision (branch ssinghal/dsv4p1-b8b-rowfix).

The benchmark server (B=8: 4 mesh rows x U=2 users, DSV41_PREFILL_UP=1: ONE carried prefill state per mesh row, DSV41_VLLM_MAX_PARTIAL=4) lost the carried state of
partly prefilled prompts and recomputed them from position 0 ("DSV4.1 chunk continuation at N without carried state"): 934 times in the 131072 / 262144 / 524288 c8 cells.
This file replays the plugin's slot logic (a port of ``model_runner._alloc_prefill_state_slots`` / ``_decode_state_slot_remap`` / the persistent-batch row order / the
partial-prefill cap of ``scheduler._hold_back_beyond_partial_cap``) in front of the REAL ``DeepseekV41ForCausalLM._prefill_interleaved`` + ``vllm_state`` and the REAL
``Model._slot_alloc`` (the device is a counting fake) and reproduces the two mechanisms:

  1. slot shift: the first finisher decodes and keeps its logical slot, so the plugin hands the prompts that continue the slots one higher (``held`` in
     ``_alloc_prefill_state_slots``) and the NEW prompt takes the freed batch row 0. The adapter processes the items in call order: the new prompt (row 0) is claimed first, its
     logical slot still maps to the model user of a prompt that continues LATER in the same call (``claim_balanced`` always offers ``j == logical``), so it takes that prompt's
     user; every continuation then misses its carried state and claims the user of the next one (a cascade: all of them recompute from position 0).
  2. same-row collision: two prompts that are in progress in one mesh row (decoders parked in the other rows, stale ``inprog`` of decoders in speculative serving) evict each
     other's single prefill slot (``Model._slot_alloc``), alternately recompute.

    pytest --noconftest models/demos/blackhole/deepseek_v41_flash/tests/test_prefill_row_placement.py
    python -m models.demos.blackhole.deepseek_v41_flash.tests.test_prefill_row_placement   # prints the replay timeline
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.demos.blackhole.deepseek_v41_flash.tt import vllm_state as VS

CHUNK = 512
BUDGET = 16384  # max_num_batched_tokens
CAP = 4  # DSV41_VLLM_MAX_PARTIAL
GREEDY = None


def _real_model_methods():
    """``Model._slot_alloc`` of tt/dsv41_model.py, taken from the source (importing the module needs the device stack): the REAL code, not a copy."""
    import ast
    import textwrap
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "tt" / "dsv41_model.py").read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_slot_alloc")
    ns = {}
    exec(textwrap.dedent(ast.get_source_segment(src, fn)), ns)
    return SimpleNamespace(_slot_alloc=ns["_slot_alloc"])


# ---- the fake device model ------------------------------------------------------------------------------------------------------
def make_fake_model():
    from models.demos.blackhole.deepseek_v41_flash.tests.test_vllm_interface import FakeInterleaveModel

    Model = _real_model_methods()

    class RowModel(FakeInterleaveModel):
        """``Model.prefill_interleaved`` of the real model, device work replaced by a token counter: the real ``_slot_alloc`` (waves of ``Up`` users per mesh row, continuations first),
        the resume rule (a continuation without carried state recomputes from 0)."""

        def __init__(self, B, U, Up):
            super().__init__(B, use_indexer=True, max_ctx=1 << 20)
            self.U, self.Up, self.rows = U, Up, B // U
            self.computed = 0  # tokens computed by the fake device
            self.recomputed = 0  # tokens computed again because the carried state was lost
            self.evictions = []  # (evicted user, by user)
            self.step_calls = []

        def flush_key_exports(self, users=None, why=None):
            return 0.0

        def prefill_interleaved(
            self, items, chunk, s_pad_max=None, want_logits=False, vis=None, final=None, totals=None
        ):
            resume = self.__dict__.setdefault("pf_resume", {})
            general = self.Up != self.U
            items = [(int(b), t, int(st), int(e)) for b, t, st, e in items]
            for k, (b, t, st, e) in enumerate(items):
                if st > 0 and (resume.get(b) != st or self.__dict__.get("pf_slot_of", {}).get(b) is None and general):
                    self.recomputed += st
                    items[k] = (b, t, 0, e)
            self.step_calls.append([(b, st, e) for b, _, st, e in items])
            grp = sorted(items, key=lambda it: it[2] == 0)  # continuations first
            by_row = {}
            for it in grp:
                by_row.setdefault(it[0] // self.U, []).append(it)
            waves = [
                [it for r in sorted(by_row) for it in by_row[r][k * self.Up : (k + 1) * self.Up]]
                for k in range(-(-max(len(v) for v in by_row.values()) // self.Up))
            ]
            out = {}
            for wave in waves:
                owner0 = list(self.__dict__.get("pf_slot_owner", [None] * (self.rows * self.Up)))
                Model._slot_alloc(self, [(b, None, st, e) for b, _, st, e in wave], resume)
                owner1 = self.__dict__["pf_slot_owner"]
                for s_, (o0, o1) in enumerate(zip(owner0, owner1)):
                    if o0 is not None and o1 != o0:
                        self.evictions.append((o0, o1))
                for b, t, st, e in wave:
                    self.computed += e - st
                    resume.pop(b, None)
                    if e % chunk == 0:
                        resume[b] = e
                    out[b] = (0, None)
            return out

    return RowModel


# ---- the plugin side: port of the slot logic of vllm_tt_plugin (model_runner / scheduler) --------------------------------------------
class Req:
    def __init__(self, rid, n_tokens, seed, arrival):
        g = torch.Generator().manual_seed(seed)
        self.id, self.L, self.arrival = rid, n_tokens, arrival
        self.tokens = torch.randint(1000, 100000, (n_tokens,), generator=g, dtype=torch.int64)
        self.computed = 0  # plugin's num_computed_tokens
        self.state = "waiting"  # waiting | prefill | decode | done
        self.t_first = None
        self.decode_steps = 0


class PluginSim:
    """Plugin scheduler + runner + the adapter (``gen``). Time: prefill 9.2e-5 s per computed token, decode step 0.1 s (calibrated on the 131072 c8 log, rough)."""

    def __init__(
        self,
        gen,
        m,
        n_slots=8,
        cap=CAP,
        budget=BUDGET,
        osl_steps=40,
        spec_keeps_inprog=True,
        row_cap=False,
        decode_every=2,
    ):
        self.gen, self.m, self.n_slots, self.cap, self.budget = gen, m, n_slots, cap, budget
        self.osl_steps, self.spec_keeps_inprog, self.row_cap, self.decode_every = (
            osl_steps,
            spec_keeps_inprog,
            row_cap,
            decode_every,
        )
        self.reqs, self.waiting, self.partial, self.decoders = {}, [], [], []
        self.batch = []  # persistent batch: list of req ids / None (holes)
        self.slot = {}  # request id -> device state slot (``_req_state_slot``)
        self.t = 0.0
        self.log = []  # per step: (kind, [(rid, start, end, user, slot)])
        self._prefill_run = 0
        self.future = []  # requests that have not arrived yet
        self.recomp_tokens = 0  # tokens the model had to compute again (a continuation without carried state)

    # persistent batch (vLLM InputBatch: removed rows are refilled lowest first by the new / resumed requests, the rest is condensed)
    def _set_batch(self, scheduled, new_first):
        holes = []
        for i, r in enumerate(self.batch):
            if r is not None and r not in scheduled:
                self.batch[i] = None
        for i, r in enumerate(self.batch):
            if r is None:
                holes.append(i)
        for r in new_first:  # (new requests, then requests that were not in the batch)
            if r in self.batch:
                continue
            if holes:
                self.batch[holes.pop(0)] = r
            else:
                self.batch.append(r)
        while holes:  # condense: the last request moves into the lowest hole
            h = holes.pop(0)
            while self.batch and self.batch[-1] is None:
                self.batch.pop()
            if h >= len(self.batch):
                break
            self.batch[h] = self.batch.pop()
            holes = [x for x in holes if x < len(self.batch)]
        while self.batch and self.batch[-1] is None:
            self.batch.pop()

    # port of model_runner._alloc_prefill_state_slots
    def _alloc_prefill_state_slots(self, row_req_ids):
        n = self.n_slots
        prefilling = set(row_req_ids)
        held = {
            s for r, s in self.slot.items() if r not in prefilling and r in self.reqs and self.reqs[r].state != "done"
        }
        out = []
        for row, r in enumerate(row_req_ids):
            if row not in held:
                s = row
            else:
                s = [x for x in range(n) if x not in held][0]
            held.add(s)
            self.slot[r] = s
            out.append(s)
        return out

    # port of model_runner._decode_state_slot_remap + settle
    def _decode_remap(self, row_req_ids):
        n = self.n_slots
        want = [self.slot[r] for r in row_req_ids]
        assert len(set(want)) == len(want)
        taken = set(want)
        remap = want + [s for s in range(n) if s not in taken]
        by_slot = {}
        for r, s in self.slot.items():
            by_slot.setdefault(s, []).append(r)
        moved = dict(self.slot)
        for row, s in enumerate(remap):
            for r in by_slot.get(s, ()):
                moved[r] = row
        self.slot = moved
        return remap

    def _model_capacity(self):
        fn = getattr(self.gen, "prefill_row_capacity", None)
        return None if fn is None else fn()

    def _admit(self):
        """port of scheduler._hold_back_beyond_partial_cap (the split regime: the budget cannot hold a whole prompt) + the row-aware cap: with ``row_cap`` every prompt that is admitted
        (a split one, which keeps its row, or one that fits the budget left and finishes in this step, which still prefills in a row) takes one of the ``prefill_row_capacity()`` clean rows
        """
        slots = self.cap - len(self.partial)
        rows_left = None
        if self.row_cap:
            rows_left = self._model_capacity()
        left = self.budget - sum(self.reqs[r].L - self.reqs[r].computed for r in self.partial)
        adm = []
        room = self.n_slots - len(self.partial) - len(self.decoders)  # max_num_seqs
        for rid in list(self.waiting):
            if len(adm) >= room or (rows_left is not None and rows_left <= 0):
                break
            q = self.reqs[rid]
            need = q.L - q.computed
            if need <= left:
                adm.append(rid)
                left -= need
            elif slots > 0:
                adm.append(rid)
                slots -= 1
                left = 0
            else:
                break
            if rows_left is not None:
                rows_left -= 1
        return adm

    def arrive(self):
        while self.future and self.future[0].arrival <= self.t:
            q = self.future.pop(0)
            self.reqs[q.id] = q
            self.waiting.append(q.id)

    def finish(self, rid):
        q = self.reqs[rid]
        q.state = "done"
        self.gen.release_request(self.slot[rid])
        self.decoders.remove(rid)
        self.on_finish(q)

    on_finish = staticmethod(lambda q: None)

    def step(self):
        self.arrive()
        pending = bool(self.waiting or self.partial)
        if not pending and not self.decoders:
            return False
        want_decode = bool(self.decoders) and (not pending or self._prefill_run >= self.decode_every)
        if want_decode:
            self._decode()
            self._prefill_run = 0
        else:
            self._prefill()
            self._prefill_run += 1
        return True

    def _decode(self):
        sched = list(self.decoders)
        self._set_batch(set(sched), [r for r in sched if r not in self.batch])
        rows = [r for r in self.batch if r is not None]
        remap = self._decode_remap(rows)
        self.gen.slots.apply_remap(remap)
        if not self.spec_keeps_inprog:
            for r in rows:
                self.gen.inprog.drop(self.gen.slots.phys[self.slot[r]])
        self.t += 0.1
        self.log.append(("decode", [(r, None, None, None, self.slot[r]) for r in rows]))
        for r in rows:
            q = self.reqs[r]
            q.decode_steps += 1
            if q.decode_steps >= self.osl_steps:
                self.finish(r)

    def _prefill(self):
        adm = self._admit()
        run = list(self.partial) + adm
        if not run:
            self.t += 0.1
            return
        share = self.budget // len(run) if len(run) > 1 else self.budget
        chunks = {}
        for rid in run:
            q = self.reqs[rid]
            piece = min(q.L - q.computed, share)
            if (
                q.computed + piece < q.L
            ):  # (prefill_chunk_alignment: a piece that does not end the prompt ends on a model chunk)
                piece = max(CHUNK, (q.computed + piece) // CHUNK * CHUNK - q.computed)
            chunks[rid] = piece
        for rid in adm:
            self.waiting.remove(rid)
            self.partial.append(rid)
            self.reqs[rid].state = "prefill"
        self._set_batch(
            set(run), adm + [r for r in run if r not in adm and r not in self.batch]
        )  # (new requests first, then the resumed ones: scheduler_output order)
        row_ids = [r for r in self.batch if r is not None]
        slots = self._alloc_prefill_state_slots(row_ids)
        starts = [self.reqs[r].computed for r in row_ids]
        ends = [self.reqs[r].computed + chunks[r] for r in row_ids]
        toks = torch.zeros(len(row_ids), max(ends), dtype=torch.int64)
        for i, r in enumerate(row_ids):
            toks[i, : ends[i]] = self.reqs[r].tokens[: ends[i]]
        final = [ends[i] == self.reqs[r].L for i, r in enumerate(row_ids)]
        c0, r0 = self.m.computed, self.m.recomputed
        self.gen.prefill_forward(
            toks,
            prompt_lens=ends,
            start_pos=starts,
            empty_slots=slots,
            sampling_params=GREEDY,
            prefill_final=final,
        )
        dtok = self.m.computed - c0
        eff = [x[1] for x in self.m.step_calls[-1]]
        self.recomp_tokens += sum(starts[i] for i in range(len(starts)) if starts[i] > 0 and eff[i] == 0)
        self.t += 9.2e-5 * dtok + 0.02
        users = [self.gen.slots.phys[s] for s in slots]
        self.log.append(
            (
                "prefill",
                [(r, self.m.step_calls[-1][i][1], ends[i], users[i], slots[i]) for i, r in enumerate(row_ids)],
                dtok,
                self.recomp_tokens,
            )
        )
        for i, r in enumerate(row_ids):
            q = self.reqs[r]
            q.computed = ends[i]
            if final[i]:
                q.state, q.t_first = "decode", self.t
                self.partial.remove(r)
                self.decoders.append(r)

    def run(self, max_steps=100000):
        n = 0
        while (self.future or self.waiting or self.partial or self.decoders) and n < max_steps:
            if not self.step():
                if not self.future:
                    break
                self.t = max(self.t, self.future[0].arrival)
            n += 1
        return n


def build(
    row_aware=False,
    row_cap=None,
    B=8,
    U=2,
    up=1,
    n_req=8,
    prompt=131076,
    skew=0.4,
    closed_loop_total=None,
    lengths=None,
    **kw,
):
    os.environ["DSV41_VLLM_INTERLEAVE"] = "1"
    os.environ["DSV41_VLLM_CHUNK"] = str(CHUNK)
    if row_aware:
        os.environ["DSV41_PREFILL_ROW_AWARE"] = "1"
    else:
        os.environ.pop("DSV41_PREFILL_ROW_AWARE", None)
    from models.demos.blackhole.deepseek_v41_flash.tt.generator_vllm import DeepseekV41ForCausalLM

    RowModel = make_fake_model()
    m = RowModel(B, U, up)
    gen = DeepseekV41ForCausalLM(SimpleNamespace(m=m, prefill_chunk=None, auto_chunk=lambda n: 256), B, prompt + 4096)
    sim = PluginSim(gen, m, n_slots=B, row_cap=row_aware if row_cap is None else row_cap, **kw)
    ln = (lambda i: prompt) if lengths is None else (lambda i: lengths[i % len(lengths)])
    sim.future = [Req(i, ln(i), 100 + i, i * skew) for i in range(n_req)]
    nxt = [n_req]
    if closed_loop_total:  # closed loop: a finished request is replaced by the next one (c8, n = closed_loop_total)

        def on_finish(q):
            if nxt[0] < closed_loop_total:
                sim.future.append(Req(nxt[0], ln(nxt[0]), 100 + nxt[0], sim.t))
                nxt[0] += 1

        sim.on_finish = on_finish
    return sim, gen, m


class _Count:
    def __init__(self):
        self.n = 0
        self.lines = []

    def __call__(self, msg):
        t = str(msg)
        if "without carried state" in t or "cannot resume" in t:
            self.n += 1
            self.lines.append(t.strip()[-120:])


def run_case(**kw):
    sim, gen, m = build(**kw)
    c = _Count()
    logger.remove()
    h = logger.add(c, level="INFO")
    try:
        sim.run()
    finally:
        logger.remove(h)
    ttft = {r: q.t_first - q.arrival for r, q in sim.reqs.items() if q.t_first is not None}
    return sim, m, c, ttft


def timeline(sim, n=45):
    out = []
    for i, e in enumerate(sim.log[:n]):
        if e[0] == "decode":
            out.append(f"{i:3d} decode  slots {[x[4] for x in e[1]]}")
        else:
            out.append(
                f"{i:3d} prefill chunks {[(x[1], x[2]) for x in e[1]]} users {[x[3] for x in e[1]]} slots {[x[4] for x in e[1]]} recomputed {e[3]}"
            )
    return "\n".join(out)


# ---- tests ---------------------------------------------------------------------------------------------------------------------------
def _stats(sim, m, c, ttft):
    v = sorted(ttft.values())
    return {
        "lost": c.n,
        "recomputed_frac": sim.recomp_tokens / max(m.computed, 1),
        "conflicts": getattr(sim.gen, "row_stats", {}).get("conflicts", 0),
        "ttft_mean": sum(v) / len(v),
        "ttft_max": v[-1],
    }


def test_baseline_reproduces_the_carried_state_loss():
    """the 131072 c8 pattern (8 prompts 0.4 s apart): the first finisher's decode shifts the state slots of the continuing prompts, the cascade starts, then two prompts share a row"""
    sim, m, c, ttft = run_case(row_aware=False, n_req=8, prompt=131076)
    assert c.n > 0, "the collision must reproduce with the switch off"
    assert sim.recomp_tokens / max(m.computed, 1) > 0.15
    # the cascade at the first step after the first finisher: the new prompt (batch row 0) takes the user of a continuation, every continuation recomputes from 0
    first = next(
        k for k, e in enumerate(sim.log) if e[0] == "prefill" and any(x[1] == 0 and x[2] > 64 * CHUNK for x in e[1])
    )
    lost = [x for x in sim.log[first][1] if x[1] == 0 and x[2] > 64 * CHUNK]
    assert (
        sim.log[first - 1][0] == "decode" and len(lost) == 3
    )  # right after the first finisher decoded: all 3 continuations recompute


def test_row_aware_placement_keeps_every_carried_state():
    base = _stats(*run_case(row_aware=False, n_req=8, prompt=131076))
    fix = _stats(*run_case(row_aware=True, n_req=8, prompt=131076))
    assert fix["lost"] == 0 and fix["recomputed_frac"] == 0 and fix["conflicts"] == 0
    assert fix["ttft_max"] < 0.5 * base["ttft_max"] and fix["ttft_mean"] < 0.6 * base["ttft_mean"]


@pytest.mark.parametrize("row_cap", [False, True])
def test_closed_loop_c8_n16(row_cap):
    """the benchmark pattern: closed loop c8, 16 requests; the model-side placement alone, and with the plugin cap on the clean rows (scheduler asks ``prefill_row_capacity``)"""
    base = _stats(*run_case(row_aware=False, n_req=8, closed_loop_total=16, prompt=65540))
    fix = _stats(*run_case(row_aware=True, row_cap=row_cap, n_req=8, closed_loop_total=16, prompt=65540))
    assert base["lost"] > 0
    assert fix["lost"] == 0 and fix["recomputed_frac"] == 0
    if row_cap:
        assert fix["conflicts"] == 0
    assert fix["ttft_mean"] < base["ttft_mean"]


def test_b4_is_unchanged_by_the_switch():
    """B=4 (U=1): no prefill slot sharing; the switch must not change one placement (same users, slots, chunks per step)"""
    runs = []
    for ra in (False, True):
        sim, m, c, ttft = run_case(row_aware=ra, row_cap=False, B=4, U=1, n_req=8, closed_loop_total=12, prompt=32772)
        assert c.n == 0
        runs.append([(e[0], [(x[0], x[1], x[2], x[3], x[4]) for x in e[1]]) for e in sim.log])
    assert runs[0] == runs[1]


def test_prefill_row_capacity_unit():
    sim, gen, m = build(row_aware=True)
    assert gen.prefill_row_capacity() is None  # no prefill_final flag seen yet: no limit
    gen.final_seen = True
    assert gen.prefill_row_capacity() == 4  # nothing live: every row can take a prompt
    # decoders on both users of row 3, prompts in progress in rows 0, 1, 2 (carried state intact)
    gen.slots.live.update({6, 7, 0, 2, 4})
    for p in (0, 2, 4):
        gen.inprog.update(p, 4096, torch.arange(4096))
        m.pf_resume[p] = 4096
    assert gen.prefill_row_capacity() == 0  # row 3 is full, rows 0-2 hold a prompt each
    m.pf_resume.pop(4)  # the prompt of row 2 lost its state (nothing to protect any more)
    assert gen.prefill_row_capacity() == 1
    gen.pf_done.add(2)  # the prompt of row 1 finished: decoding
    assert gen.prefill_row_capacity() == 2
    gen.row_aware = False
    assert gen.prefill_row_capacity() is None  # switch off: no answer for the plugin
    sim, gen, m = build(row_aware=True, B=4, U=1)
    gen.final_seen = True
    assert gen.prefill_row_capacity() is None  # U == Up (B=4): not applicable


def test_claim_rowaware_unit():
    st = VS.SlotTable(8, 8)
    U = 2
    # users 0 (row 0) decoding, 2 (row 1) in progress: a new prompt goes to a row without in-progress prompt and the fewest live users: row 2 or 3, not row 0 / 1
    st.live.update({0, 2})
    p, clean = st.claim_rowaware(5, {1}, U)
    assert clean and p // U in (2, 3) and p in st.live and sorted(st.phys) == list(range(8))
    # the logical slot's own user is not offered while it is live (a continuation of the call may be bound to it)
    st = VS.SlotTable(8, 8)
    st.live.update({1})
    p, _ = st.claim_rowaware(1, set(), U)
    assert p != 1
    # no clean row left: a busy row is the last resort and says so
    st = VS.SlotTable(8, 8)
    st.live.update({0, 2, 4, 6})
    p, clean = st.claim_rowaware(3, {0, 1, 2, 3}, U)
    assert not clean and p in (1, 3, 5, 7)
    # decoders spread: the row with fewer live users first
    st = VS.SlotTable(8, 8)
    st.live.update({0, 1, 2})  # rows 0 (2 live), 1 (1 live)
    p, clean = st.claim_rowaware(7, set(), U)
    assert clean and p // U in (2, 3)


if __name__ == "__main__":
    for ra in (False, True):
        sim, m, c, ttft = run_case(row_aware=ra, n_req=8, prompt=int(sys.argv[1]) if len(sys.argv) > 1 else 131076)
        print(
            f"=== row_aware={ra}: without-carried-state lines {c.n}, computed {m.computed}, recomputed {sim.recomp_tokens} ({sim.recomp_tokens / max(m.computed, 1):.0%}), evictions {len(m.evictions)}, sim end {sim.t:.0f} s"
        )
        print("TTFT (sim s):", {k: round(v, 1) for k, v in sorted(ttft.items())})
        print(timeline(sim, int(sys.argv[2]) if len(sys.argv) > 2 else 45))
