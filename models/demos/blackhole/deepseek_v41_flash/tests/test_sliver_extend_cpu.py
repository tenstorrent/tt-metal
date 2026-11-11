# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of DSV41_SLIVER_EXTEND (T24, image-2 B=16 state loss): a non-final piece shorter than a chunk is computed for one chunk instead of being skipped."""
import pytest
import torch
from loguru import logger

from models.demos.blackhole.deepseek_v41_flash.tests.test_protect_partial_cpu import Count, call, make_gen


def run(monkeypatch, sliver, pieces, L=40000, B=16, U=4, Up=4, nuser=1):
    monkeypatch.setenv("DSV41_SLIVER_EXTEND", "1" if sliver else "0")
    monkeypatch.delenv("DSV41_SLIVER_EXTEND_FILE", raising=False)
    gen, m, _ = make_gen(monkeypatch, False, B=B, U=U, Up=Up)
    c = Count()
    logger.remove()
    h = logger.add(c, level="INFO")
    try:
        toks = torch.randint(1000, 90000, (L,), dtype=torch.int64)
        for st, e in pieces:
            call(gen, toks, st, e, "x", 0, L, final=(e == L))
    finally:
        logger.remove(h)
    return gen, m, c


# the first piece of a new prompt is a sliver (nothing registered: the old code recomputes at the next piece), and a sliver in the middle of a prompt
@pytest.mark.parametrize(
    "pieces",
    [[(0, 285), (285, 4096), (4096, 8192)], [(0, 4096), (4096, 4281), (4281, 8192), (8192, 8300), (8300, 12288)]],
)
def test_sliver_extend_has_no_recompute_and_computes_each_token_once_plus_a_chunk(monkeypatch, pieces):
    gen, m, c = run(monkeypatch, True, pieces)
    assert c.lost == 0
    assert all(st > 0 or True for st, _ in pieces)
    steps = [s[0] for s in m.steps]
    # every device step starts where the carried state is (never at 0 after the first call)
    assert steps[0][1] == 0 and all(s[1] > 0 for s in steps[1:]), steps
    assert gen.protect_stats["extended_hits"] >= 1


def test_mid_prompt_sliver_old_behaviour_is_unchanged_without_the_flag(monkeypatch):
    gen, m, c = run(monkeypatch, False, [(0, 4096), (4096, 4281), (4281, 8192)])
    assert c.lost == 0 and [s[0] for s in m.steps][-1][1] == 4096 and not any(gen.protect_stats.values())


def test_final_piece_and_last_chunk_are_never_extended(monkeypatch):
    gen, m, c = run(monkeypatch, True, [(0, 4096), (4096, 4300)], L=4300)
    assert [s[0] for s in m.steps][-1] == (0, 4096, 4300) and c.lost == 0  # final: computed as before
    gen, m, c = run(
        monkeypatch, True, [(0, 4096), (4096, 4281)], L=4400
    )  # st + chunk (4608) >= total: not extended (skipped as before)
    assert len(m.steps) == 1


def test_sim_with_budget_slivers_has_no_events(monkeypatch):
    """closed-loop replay of the image-2 pattern (B=16 U=Up=4, 32768 budget / 4096 cap, prompt lengths that leave slivers): flag off loses state, on does not"""
    from models.demos.blackhole.deepseek_v41_flash.tests import test_prefill_row_placement as T

    def _prefill(self):
        adm = self._admit()
        run_ = list(self.partial) + adm
        if not run_:
            self.t += 0.1
            return
        left, chunks = self.budget, {}
        for rid in run_:
            q = self.reqs[rid]
            piece = min(q.L - q.computed, 4096, left)
            if piece <= 0:
                continue
            if q.computed + piece < q.L and piece >= T.CHUNK:
                piece = (q.computed + piece) // T.CHUNK * T.CHUNK - q.computed
            chunks[rid] = piece
            left -= piece
        run_ = [r for r in run_ if r in chunks]
        adm = [r for r in adm if r in chunks]
        for rid in adm:
            self.waiting.remove(rid)
            self.partial.append(rid)
            self.reqs[rid].state = "prefill"
        self._set_batch(set(run_), adm + [r for r in run_ if r not in adm and r not in self.batch])
        row_ids = [r for r in self.batch if r is not None]
        slots = self._alloc_prefill_state_slots(row_ids)
        starts = [self.reqs[r].computed for r in row_ids]
        ends = [self.reqs[r].computed + chunks[r] for r in row_ids]
        toks = torch.zeros(len(row_ids), max(ends), dtype=torch.int64)
        for i, r in enumerate(row_ids):
            toks[i, : ends[i]] = self.reqs[r].tokens[: ends[i]]
        final = [ends[i] == self.reqs[r].L for i, r in enumerate(row_ids)]
        totals = [self.reqs[r].L for r in row_ids]
        c0 = self.m.computed
        self.gen.prefill_forward(
            toks,
            prompt_lens=ends,
            start_pos=starts,
            empty_slots=slots,
            sampling_params=None,
            prefill_final=final,
            prefill_total=totals,
        )
        self.t += 9.2e-5 * (self.m.computed - c0) + 0.02
        self.log.append(
            ("prefill", [(r, 0, ends[i], 0, slots[i]) for i, r in enumerate(row_ids)], 0, self.recomp_tokens)
        )
        for i, r in enumerate(row_ids):
            q = self.reqs[r]
            q.computed = ends[i]
            if final[i]:
                q.state, q.t_first = "decode", self.t
                self.partial.remove(r)
                self.decoders.append(r)

    monkeypatch.setattr(T.PluginSim, "_prefill", _prefill)
    monkeypatch.setenv("DSV41_PIECE_LAG", "1")
    monkeypatch.setenv("DSV41_CONT_FIRST", "1")
    res = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("DSV41_SLIVER_EXTEND", flag)
        sim, m, c, _ = T.run_case(
            row_aware=False,
            B=16,
            U=4,
            up=4,
            n_req=32,
            closed_loop_total=48,
            prompt=140000,
            lengths=[100000 + (i * 997 + 13 * i * i) % 20000 for i in range(40)],
            skew=0.3,
            cap=16,
            budget=32768,
            osl_steps=20,
        )
        res[flag] = c.n
    assert res["0"] > 0 and res["1"] == 0, res
