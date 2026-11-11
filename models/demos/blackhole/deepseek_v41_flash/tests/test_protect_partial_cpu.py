# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of DSV41_PROTECT_PARTIAL (T24): partly prefilled prompts keep their carried state.

Two loss mechanisms of the T3 / T9 / T12 server logs ("chunk continuation at N without carried state: recomputing from position 0"):
  A. (B=8, Up == U, no slot eviction possible) a prefix hit whose first piece is shorter than a chunk (the step budget ran out) was SKIPPED (piece lag) before the hit was restored and registered:
     its next piece found no carried state and recomputed the whole prompt (18 of 23 events in T9, 307 k tokens each).
  B. (B=16, Up=2 < U=4) a prefix hit reserved a free user on its home row although both prefill slots of the row held prompts in progress: the hit evicted one (``Model._slot_alloc``, cand[0]).

    pytest --noconftest models/demos/blackhole/deepseek_v41_flash/tests/test_protect_partial_cpu.py
"""

from types import SimpleNamespace

import torch
from loguru import logger

from models.demos.blackhole.deepseek_v41_flash.tests.test_vllm_interface import FakeInterleaveModel
from models.demos.blackhole.deepseek_v41_flash.tt import vllm_state as VS
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_waves import slot_alloc

CHUNK = 512


class HitModel(FakeInterleaveModel):
    """fake model: restores a hit (rids) at its start, counts the computed tokens, resumable only at chunk ends"""

    def __init__(self, B, U, Up):
        super().__init__(B, use_indexer=True, max_ctx=1 << 20)
        self.U, self.Up, self.rows = U, Up, B // U
        self.computed = 0
        self.steps = []

    def prefill_interleaved(
        self, items, chunk, s_pad_max=None, want_logits=False, vis=None, final=None, totals=None, rids=None
    ):
        self.steps.append([(b, st, e) for b, _, st, e in items])
        out = {}
        for b, t, st, e in items:
            hit = rids is not None and rids.get(b, "").startswith("hit")
            assert st == 0 or hit or self.pf_resume.get(b) == st, f"user {b} continues at {st} without carried state"
            if hit and st > 0:
                self.pf_resume[b] = st  # (restore)
            self.computed += e - st
            self.pf_resume.pop(b, None)
            if e % chunk == 0:
                self.pf_resume[b] = e
            out[b] = (0, None)
        return out


class Plan:
    def __init__(self, user, hit):
        self.user, self.hit_tokens = user, hit


def make_gen(monkeypatch, protect, B=8, U=2, Up=2, lag=True, resume=False, row_aware=False):
    monkeypatch.setenv("DSV41_VLLM_INTERLEAVE", "1")
    monkeypatch.setenv("DSV41_VLLM_CHUNK", str(CHUNK))
    monkeypatch.setenv("DSV41_PIECE_LAG", "1" if lag else "0")
    monkeypatch.setenv("DSV41_PROTECT_PARTIAL", "1" if protect else "0")
    monkeypatch.delenv("DSV41_PROTECT_PARTIAL_FILE", raising=False)
    monkeypatch.delenv("DSV41_RESUME_FROM_RECORD_FILE", raising=False)
    monkeypatch.setenv("DSV41_RESUME_FROM_RECORD", "1" if resume else "0")
    monkeypatch.setenv("DSV41_PREFILL_ROW_AWARE", "1" if row_aware else "0")
    from models.demos.blackhole.deepseek_v41_flash.tt.generator_vllm import DeepseekV41ForCausalLM

    m = HitModel(B, U, Up)
    gen = DeepseekV41ForCausalLM(SimpleNamespace(m=m, prefill_chunk=None, auto_chunk=lambda n: 256), B, 1 << 20)
    plans = {}
    gen.paged = SimpleNamespace(
        idx=SimpleNamespace(
            reservation=lambda rid: plans.get(rid), reserved_all=lambda: tuple(p.user for p in plans.values())
        )
    )
    return gen, m, plans


class Count:
    def __init__(self):
        self.lost = 0
        self.ext = 0

    def __call__(self, msg):
        t = str(msg)
        self.lost += "without carried state" in t
        self.ext += "shorter than a chunk" in t


def call(gen, toks, start, end, rid, slot, total, final=False):
    gen.prefill_forward(
        toks[:total].reshape(1, -1),  # (the plugin's lookahead capability: every step carries the whole sequence)
        prompt_lens=[end],
        start_pos=[start],
        empty_slots=[slot],
        sampling_params=None,
        prefill_final=[final],
        prefill_total=[total],
        request_ids=[rid],
    )


def run_hit_scenario(monkeypatch, protect, second_end):
    gen, m, plans = make_gen(monkeypatch, protect)
    c = Count()
    logger.remove()
    h = logger.add(c, level="INFO")
    try:
        L = 400000
        toks = torch.randint(1000, 90000, (L,), dtype=torch.int64)
        plans["hit1"] = Plan(user=1, hit=307200)
        call(gen, toks, 307200, 307485, "hit1", 1, L)  # the budget left 285 tokens: shorter than a chunk
        call(gen, toks, 307485, second_end, "hit1", 1, L)
    finally:
        logger.remove(h)
    return gen, m, c


def test_baseline_skipped_hit_piece_loses_the_hit(monkeypatch):
    gen, m, c = run_hit_scenario(monkeypatch, protect=False, second_end=311581)
    assert c.lost == 1
    assert m.steps[-1] == [(1, 0, 311296)]  # recomputed from position 0 (the piece cut to the chunk grid)
    assert m.computed == 311296


def test_protected_hit_piece_is_computed_one_chunk_and_resumes_ahead(monkeypatch):
    gen, m, c = run_hit_scenario(monkeypatch, protect=True, second_end=311581)
    assert c.lost == 0 and c.ext == 1
    assert m.steps[0] == [(1, 307200, 307712)]  # the hit restored, one chunk computed
    assert m.steps[1] == [(1, 307712, 311296)]  # resumes at the model's carried end, one chunk ahead of vLLM's start
    assert m.computed == 512 + (311296 - 307712)
    assert gen.protect_stats["extended_hits"] == 1 and gen.protect_stats["ahead_resumes"] == 1


def test_protected_short_second_piece_is_skipped_and_the_third_resumes(monkeypatch):
    gen, m, plans = make_gen(monkeypatch, True)
    c = Count()
    logger.remove()
    h = logger.add(c, level="INFO")
    try:
        L = 400000
        toks = torch.randint(1000, 90000, (L,), dtype=torch.int64)
        plans["hit1"] = Plan(user=1, hit=307200)
        call(gen, toks, 307200, 307485, "hit1", 1, L)
        call(gen, toks, 307485, 307600, "hit1", 1, L)  # ends before the carried 307712: nothing to compute
        assert m.steps[-1] == [(1, 307200, 307712)]  # (no new device call for it)
        call(gen, toks, 307600, 311696, "hit1", 1, L)
    finally:
        logger.remove(h)
    assert c.lost == 0
    assert m.steps[-1] == [(1, 307712, 311296)]


def test_final_hit_piece_is_never_extended(monkeypatch):
    gen, m, plans = make_gen(monkeypatch, True)
    L = 307485
    toks = torch.randint(1000, 90000, (L,), dtype=torch.int64)
    plans["hit1"] = Plan(user=1, hit=307200)
    call(gen, toks, 307200, 307485, "hit1", 1, L, final=True)
    assert m.steps[0] == [(1, 307200, 307485)]


def test_hit_with_no_room_for_a_chunk_before_the_end_is_not_extended(monkeypatch):
    gen, m, plans = make_gen(monkeypatch, True)
    L = 307600  # st + chunk (307712) >= total: the old (skip) path, the model could not stay below the end
    toks = torch.randint(1000, 90000, (L,), dtype=torch.int64)
    plans["hit1"] = Plan(user=1, hit=307200)
    call(gen, toks, 307200, 307485, "hit1", 1, L)
    assert m.steps == []  # skipped, no device call


def test_off_is_identical_without_the_flag(monkeypatch):
    gen, m, c = run_hit_scenario(monkeypatch, protect=False, second_end=311581)
    assert not any(gen.protect_stats.values())


# ---- B: slot eviction ---------------------------------------------------------------------------------------------------------------------
def test_slot_alloc_cheapest_victim():
    U, Up = 4, 2
    for cheapest, victim in ((False, 0), (True, 1)):
        owner, of, resume = (
            [0, 1] + [None] * 6,
            {0: 0, 1: 1},
            {0: 300000, 1: 4096},
        )  # user 0 = giant in slot 0, user 1 small in slot 1
        ev = []
        out = slot_alloc(owner, of, Up, U, [(2, None, 0, 4096)], resume, on_evict=ev.append, cheapest=cheapest)
        assert ev == [victim], (cheapest, ev)
        assert (0 in resume) == (victim != 0)  # the giant keeps its carried state when protected
        assert out[2] == victim


def test_slot_alloc_prefers_a_free_slot_and_logs():
    owner, of, resume = [0, None] + [None] * 6, {0: 0}, {0: 300000}
    lines = []
    slot_alloc(owner, of, 2, 4, [(2, None, 0, 4096)], resume, cheapest=True, log=lines.append)
    assert 0 in resume and not lines  # the free slot 1 is taken
    owner, of, resume = [0, 1] + [None] * 6, {0: 0, 1: 1}, {0: 300000, 1: 4096}
    slot_alloc(owner, of, 2, 4, [(2, None, 0, 4096)], resume, cheapest=True, log=lines.append)
    assert len(lines) == 1 and "carried at 4096" in lines[0]


def _inprog(gen, m, user, end):
    gen.inprog.update(user, end, torch.zeros(end, dtype=torch.int64))
    m.pf_resume[user] = end
    gen.slots.live.add(user)


def test_free_users_of_row_is_slot_aware(monkeypatch):
    for protect, expect in ((False, [2, 3]), (True, [])):
        gen, m, _ = make_gen(monkeypatch, protect, B=16, U=4, Up=2)
        _inprog(gen, m, 0, 8192)
        _inprog(gen, m, 1, 300000)
        assert gen._free_users_of_row(0) == expect
        assert gen._free_users_of_row(1) == (
            [4, 5, 6, 7] if not protect else [4, 5]
        )  # (a row with free slots: at most Up users may take them)


def test_free_users_with_one_slot_left_and_reservations(monkeypatch):
    gen, m, _ = make_gen(monkeypatch, True, B=16, U=4, Up=2)
    _inprog(gen, m, 0, 8192)
    assert gen._free_users_of_row(0) == [1]  # one slot left: ONE user may take it
    gen.paged = SimpleNamespace(idx=SimpleNamespace(reservation=lambda r: None, reserved_all=lambda: (1,)))
    assert gen._free_users_of_row(0) == [1]  # (a reserved user is still listed, the index drops it) ...
    gen.paged = SimpleNamespace(idx=SimpleNamespace(reservation=lambda r: None, reserved_all=lambda: (3,)))
    assert gen._free_users_of_row(0) == [3]  # ... the reservation took the last slot: nothing else may be offered


def test_finished_prompt_does_not_hold_a_slot(monkeypatch):
    gen, m, _ = make_gen(monkeypatch, True, B=16, U=4, Up=2)
    _inprog(gen, m, 0, 8192)
    _inprog(gen, m, 1, 9000)
    gen.pf_done.add(1)  # its last piece ended the prompt: decoding, nothing to protect (but user 1 is live: not free)
    assert gen._free_users_of_row(0) == [2]


def test_state_lost_prompt_does_not_hold_a_slot(monkeypatch):
    gen, m, _ = make_gen(monkeypatch, True, B=16, U=4, Up=2)
    _inprog(gen, m, 0, 8192)
    _inprog(gen, m, 1, 9000)
    m.pf_resume.pop(1)  # carried state gone already
    assert gen._free_users_of_row(0) == [2]


def test_live_switch_file(monkeypatch, tmp_path):
    gen, m, _ = make_gen(monkeypatch, True, B=16, U=4, Up=2)
    _inprog(gen, m, 0, 8192)
    _inprog(gen, m, 1, 9000)
    f = tmp_path / "sw"
    monkeypatch.setenv("DSV41_PROTECT_PARTIAL_FILE", str(f))
    f.write_text("0")
    assert gen._free_users_of_row(0) == [2, 3]
    f.write_text("1")
    assert gen._free_users_of_row(0) == []


def test_claim_rowaware_avoids_a_full_row():
    U = 4

    # rows of 4 users: row 0 has free users 2,3 but both prefill slots hold prompts (full), row 1 has free users 6,7 (user 5 decodes) and ONE prefill in progress (a slot left); rows 2,3 all decoding
    def table():
        st = VS.SlotTable(16, 16)
        st.live.update([0, 1, 4, 5] + list(range(8, 16)))
        return st

    p, clean = table().claim_rowaware(2, {0, 1}, U, (), full_rows={0})
    assert p // U == 1 and not clean  # (row 1 is busy but has a slot left: preferred over the full row)
    p, clean = table().claim_rowaware(2, {0, 1}, U, ())
    assert p // U == 0  # the old rule: fewest live users first, i.e. the row whose slots are full
    # claim_balanced / claim_packed take the same argument
    st = table()
    assert st.claim_balanced(2, {0: 2, 1: 1}, U, (), full_rows={0}) // U == 1
    st = table()
    assert st.claim_packed(2, {0: 2, 1: 1}, U, (), full_rows={0}) // U == 1


# ---- resume from a record ---------------------------------------------------------------------------------------------------------------
class RecIdx:
    """index stub: ``plan_hit`` answers with the deepest record <= max_hit (records every 8192 tokens) on the row of ``user``"""

    def __init__(self, plans, user_of):
        self.plans, self.user_of, self.calls = plans, user_of, []

    def reservation(self, rid):
        return self.plans.get(rid)

    def reserved_all(self):
        return tuple(p.user for p in self.plans.values())

    def plan_hit(self, rid, tokens, num_tokens, max_hit=None, **kw):
        self.calls.append((rid, int(num_tokens), max_hit))
        hit = (max_hit // 8192) * 8192
        if hit <= 0:
            return Plan(-1, 0)
        self.plans[rid] = Plan(self.user_of, hit)
        return self.plans[rid]

    def cancel(self, rid):
        self.plans.pop(rid, None)


def lose_scenario(monkeypatch, resume, row_aware):
    B, U, Up = (16, 4, 2) if row_aware else (8, 2, 2)
    gen, m, plans = make_gen(monkeypatch, True, B=B, U=U, Up=Up, resume=resume, row_aware=row_aware)
    gen.paged = SimpleNamespace(idx=RecIdx(plans, user_of=0))
    c = Count()
    logger.remove()
    h = logger.add(c, level="INFO")
    try:
        L = 100000
        toks = torch.randint(1000, 90000, (L,), dtype=torch.int64)
        call(gen, toks, 0, 20480, "hit2", 0, L)  # a cold prompt in progress (user 0)
        assert m.pf_resume[0] == 20480
        m.pf_resume.pop(0)  # its carried state is lost (slot taken / trace re-captured)
        call(gen, toks, 20480, 24576, "hit2", 0, L)
    finally:
        logger.remove(h)
    return gen, m, c


def test_baseline_lost_state_recomputes_from_zero(monkeypatch):
    for ra in (False, True):
        gen, m, c = lose_scenario(monkeypatch, resume=False, row_aware=ra)
        assert c.lost == 1 and m.steps[-1][0][1] == 0, (ra, m.steps)


def test_lost_state_resumes_from_the_deepest_record(monkeypatch):
    for ra in (False, True):
        gen, m, c = lose_scenario(monkeypatch, resume=True, row_aware=ra)
        assert c.lost == 0, ra
        assert m.steps[-1] == [(0, 16384, 24576)], (
            ra,
            m.steps,
        )  # records every 8192: 20480 -> 16384 (4096 tokens recomputed, not 20480)
        assert gen.protect_stats["record_resumes"] == 1 and gen.protect_stats["record_resume_tokens"] == 16384


def test_no_record_falls_back_to_zero(monkeypatch):
    gen, m, plans = make_gen(monkeypatch, True, resume=True)
    gen.paged = SimpleNamespace(idx=RecIdx(plans, user_of=0))
    L = 100000
    toks = torch.randint(1000, 90000, (L,), dtype=torch.int64)
    call(gen, toks, 0, 4096, "hit3", 0, L)
    m.pf_resume.pop(0)
    call(gen, toks, 4096, 8192 - 512, "hit3", 0, L)  # max_hit 4096 < first record at 8192? (records every 8192) -> none
    assert m.steps[-1][0][1] == 0 and gen.protect_stats["record_misses"] == 1
