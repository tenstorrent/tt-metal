# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests of the adaptive verification length of the vLLM adapter (tt/spec_adapt.py + the ``AdaptiveVerify`` wiring of tt/generator_vllm.py) against fake SpecRunners: the gate
(DSV41_VLLM_SPEC_ADAPT), the choice of k from the confidence and the round times, the verify block width k + 1, the trimmed-offer rows, and the unchanged fixed-k path (DSV41_VLLM_SPEC_ADAPT=0).

    PYTHONPATH=<plugin>/src pytest --noconftest models/demos/blackhole/deepseek_v41_flash/tests/test_spec_adapt_cpu.py
"""

from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.deepseek_v41_flash.tests.test_vllm_interface import (  # noqa: F401 (fixture)
    GREEDY,
    FakeModel,
    FakeSpec,
    expect_error,
)
from models.demos.blackhole.deepseek_v41_flash.tt import vllm_state as VS
from models.demos.blackhole.deepseek_v41_flash.tt.generator_vllm import DeepseekV41ForCausalLM
from models.demos.blackhole.deepseek_v41_flash.tt.spec_adapt import AdaptiveVerify


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DSV41_VLLM_INTERLEAVE", "0")
    for v in ("DSV41_VLLM_SPEC_ADAPT", "DSV41_VLLM_SPEC_KS", "DSV41_SPEC_TIMES", "DSV41_SPEC_POLICY"):
        monkeypatch.delenv(v, raising=False)


# ---- the gate ----------------------------------------------------------------------------------------------------------------------------
def test_gate_default_is_on_for_k5_at_batch_4(monkeypatch):
    f = DeepseekV41ForCausalLM.adaptive_ks
    assert f(5, 4) == [3, 5]
    assert f(5, 16) == [] and f(5, 32) == []  # other batches: the fixed-k runner of today
    assert f(3, 4) == [] and f(0, 4) == []
    monkeypatch.setenv("DSV41_VLLM_SPEC_ADAPT", "0")
    assert f(5, 4) == []  # exact fixed-k path
    monkeypatch.setenv("DSV41_VLLM_SPEC_ADAPT", "1")
    assert f(5, 16) == [3, 5]
    monkeypatch.setenv("DSV41_VLLM_SPEC_KS", "2,3")
    assert f(5, 4) == [2, 3, 5]  # the plugin's draft length is always a candidate
    monkeypatch.setenv("DSV41_VLLM_SPEC_KS", "5")
    assert f(5, 4) == []


# ---- the scheduler ---------------------------------------------------------------------------------------------------------------------
class FakeRunner(FakeSpec):
    """FakeSpec with the attributes the adaptive wrapper reads: position-bucket ladder, confidence of the last round, calibrated round time."""

    def __init__(self, B, k, buckets=(512, 4096), round_ms=None):
        super().__init__(B, k)
        self.pos_buckets = list(buckets)
        self.conf = None
        self.round_ms = round_ms
        self.conf_final = torch.zeros(B, 5)
        self.prepared = []

    def bucket_for(self, mx):
        return next((b for b in self.pos_buckets if mx <= b), None)

    def prepare(self, X, base):
        self.prepared.append(tuple(X.shape))

    def seed(self, tokens, lens, first):
        super().seed(tokens, lens, first)
        self.conf_final = torch.full((self.B, 5), 0.9)

    def _round(self, X, base, force):
        a, m, d = super()._round(X, base, force)
        self.conf = torch.full((self.B, 5), 0.8)
        return a, m, d


class FakeAdaptiveSpec:
    """The part of tt/spec_model.AdaptiveSpec that AdaptiveVerify uses (the real scoring functions are borrowed from the class)."""

    def __init__(self, B, ks=(3, 5), times=None, **kw):
        self.runners = {k: FakeRunner(B, k, **kw) for k in ks}
        self.seedr = self.runners[min(ks)]
        self.ks, self.times, self.policy, self.hyst = list(ks), dict(times or {}), "adapt", 0.0
        self.cal_a, self.cal_b, self.plain_ok = 1.0, 0.0, False
        for r in self.runners.values():
            r.siblings = []

    def release(self):
        for r in self.runners.values():
            r.release()

    def _p(self, conf):
        from models.demos.blackhole.deepseek_v41_flash.tt.spec_model import AdaptiveSpec

        return AdaptiveSpec._p(self, conf)

    def expected_tokens(self, conf, active):
        from models.demos.blackhole.deepseek_v41_flash.tt.spec_model import AdaptiveSpec

        return AdaptiveSpec.expected_tokens(self, conf, active)


def test_choose_follows_confidence_and_round_times():
    av = AdaptiveVerify(FakeAdaptiveSpec(4, times={3: 65.0, 5: 108.0}))
    lo = torch.full((4, 5), 0.5)  # E3 = 1 + .5 + .25 + .125 = 1.875 (28.8 /s), E5 = 1.969 (18.2 /s)
    assert av.choose(lo) == 3
    hi = torch.full((4, 5), 0.99)  # E3 = 3.94 (60.7 /s), E5 = 5.8 (53.7 /s): k=3 is still faster
    assert av.choose(hi) == 3
    av.ad.times = {3: 65.0, 5: 70.0}  # k=5 almost as cheap: its longer chain wins on confident drafts
    assert av.choose(hi) == 5
    assert av.choose(lo) == 3
    av.ad.times = {3: 65.0}  # no profile of k=5 yet: the fixed-k behaviour
    assert av.choose(hi) == 5
    av.ad.times = {3: 65.0, 5: 108.0}
    av.ad.policy = "k5"
    assert av.choose(lo) == 5


def test_choose_hysteresis_and_ema():
    av = AdaptiveVerify(FakeAdaptiveSpec(4, times={3: 65.0, 5: 70.0}))
    av.ad.hyst = 0.5
    hi = torch.full((4, 5), 0.99)
    assert av.choose(hi) == 5  # first choice: best
    assert av.choose(torch.full((4, 5), 0.5)) == 5  # 3 is better but not by 50 %: stay
    av.note_round(5, 170.0)
    assert av.ad.times[5] == pytest.approx(0.8 * 70.0 + 0.2 * 170.0)
    assert av.k_hist == {5: 1}


class _HostAdaptiveSpec(FakeAdaptiveSpec):
    """FakeAdaptiveSpec with the expected-token rule written out (no import of tt.spec_model: runs without ttnn): E[k] = 1 + sum_{j<=k} mean_rows prod_{i<=j} p_i."""

    def expected_tokens(self, conf, active):
        S = torch.cumprod(conf[active], dim=1).mean(0)
        return {k: 1.0 + float(S[:k].sum()) for k in self.ks}


def _drive(av, conf, rounds, ms):
    """``rounds`` rounds of the plugin protocol (choose, then note_round of the chosen k with ms[k]); the chosen k's."""
    out = []
    for _ in range(rounds):
        k = av.choose(conf)
        av.note_round(k, ms[k])
        out.append(k)
    return out


def test_stale_estimate_is_probed_and_replaced(monkeypatch):
    # k=5 was measured once at 120 ms (e.g. during a prefill burst), its real round is 60 ms (k=3: 57 ms): E5 / E3 = 1.2 makes k=5 the better one
    monkeypatch.setenv("DSV41_SPEC_PROBE_EVERY", "0")
    old = AdaptiveVerify(_HostAdaptiveSpec(4, times={3: 57.0, 5: 120.0}))
    conf = torch.full((4, 5), 0.85)
    assert (
        old.ad.expected_tokens(conf, torch.ones(4, dtype=torch.bool))[5]
        / old.ad.expected_tokens(conf, torch.ones(4, dtype=torch.bool))[3]
        > 60.0 / 57.0
    )
    assert set(_drive(old, conf, 2000, {3: 57.0, 5: 60.0})) == {3}  # no probes: k=5 is never retried (the lock-in)
    monkeypatch.setenv("DSV41_SPEC_PROBE_EVERY", "128")
    av = AdaptiveVerify(_HostAdaptiveSpec(4, times={3: 57.0, 5: 120.0}))
    ks = _drive(av, conf, 400, {3: 57.0, 5: 60.0})
    assert ks.index(5) <= 129  # the first probe comes after at most probe_every rounds
    i = ks.index(5)
    assert (
        ks[i + 1 : i + 100].count(5) >= 98
    )  # and k=5 then runs as the better k (the stale k=3 gets its own probe round later)
    assert av.ad.times[5] == pytest.approx(60.0)  # replaced, not blended with the stale 120
    assert av.probes >= 1


def test_probes_are_bounded_where_the_other_k_is_really_slower(monkeypatch):
    monkeypatch.setenv("DSV41_SPEC_PROBE_EVERY", "128")
    av = AdaptiveVerify(_HostAdaptiveSpec(4, times={3: 57.0, 5: 63.0}))
    conf = torch.full((4, 5), 0.4)  # low acceptance: k=3 is the better one
    ks = _drive(av, conf, 12800, {3: 57.0, 5: 120.0})
    assert 5 in ks[:200]  # one probe measures k=5
    assert ks.count(5) <= 12800 // 128 + 1  # at most one probe round in probe_every
    assert ks.count(5) >= 12800 // 129 - 1 and av.ad.times[5] == pytest.approx(120.0)
    assert ks[-300:].count(3) >= 297


def test_candidates_skip_a_runner_whose_ladder_ends_below_the_positions():
    ad = FakeAdaptiveSpec(4, times={3: 65.0, 5: 108.0})
    ad.runners[3].pos_buckets = [512]  # k=3 ladder is short: beyond 512 it replays its whole-slab traces
    av = AdaptiveVerify(ad)
    hi = torch.full((4, 5), 0.5)
    assert av.choose(hi, max_pos=300) == 3
    assert av.choose(hi, max_pos=2000) == 5  # only k=5 has a bucket there
    ad.runners[5].pos_buckets = [512]
    assert av.choose(hi, max_pos=2000) == 3  # no runner has one: the profile decides


def test_runner_for_width():
    av = AdaptiveVerify(FakeAdaptiveSpec(4))
    assert av.runner_for(0)[0] == 3 and av.runner_for(3)[0] == 3 and av.runner_for(5)[0] == 5
    assert av.runner_for(4)[0] == 5  # not a candidate: the next larger
    assert av.k == 5 and av.n == 6 and sorted(av.runners) == [3, 5]


# ---- the adapter wiring ----------------------------------------------------------------------------------------------------------------------
def make_adaptive(times=None, B=4):
    Bp = VS.padded_batch(B)
    m = FakeModel(Bp)
    ad = FakeAdaptiveSpec(Bp, times=times or {3: 65.0, 5: 108.0})
    spec = AdaptiveVerify(ad)
    gen = SimpleNamespace(m=m, spec=spec, prefill_chunk=None, auto_chunk=lambda max_len: 256)
    return DeepseekV41ForCausalLM(gen, B, 4096), m, spec


def _prefill_two(gen):
    toks = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.int32)
    return gen.prefill_forward(toks, prompt_lens=[3, 2], empty_slots=[0, 1], sampling_params=GREEDY)


def _propose(gen, first, lens=(3, 2), n=5):
    B = gen.B
    committed = torch.zeros(B, 1 + n, dtype=torch.int32)
    cpos = torch.full((B, 1 + n), -1)
    for i, (f, L) in enumerate(zip(first.tolist(), lens)):
        committed[i, 0], cpos[i, 0] = f, L
    return gen.propose_draft_tokens(n, committed, cpos, torch.ones(B, dtype=torch.int32))


def test_prefill_seeds_through_the_seed_runner_and_stashes_conf():
    gen, m, spec = make_adaptive()
    first = _prefill_two(gen)
    assert gen.spec_adapt is spec
    r3, r5 = spec.runners[3], spec.runners[5]
    assert r3.released == 1 and r5.released == 1  # every runner is released with the prefill
    assert (
        r3.rounds[-1][0] == "seed" and not r5.rounds
    )  # the seed runner (smallest k) seeds, the other is captured by its siblings' pass
    assert gen.spec_drafts.shape == (gen.B, 5)
    p0 = gen.slots.physical(0)
    assert gen.spec_conf[p0].tolist() == pytest.approx([0.9] * 5)
    assert gen.spec_drafts[p0].tolist() == [(int(first[0]) + j + 1) % 7 for j in range(5)]


def test_propose_offers_the_chosen_k_to_every_offered_row():
    gen, m, spec = make_adaptive({3: 65.0, 5: 108.0})
    first = _prefill_two(gen)  # conf 0.9: E3 = 3.44 / 65, E5 = 4.1 / 108 -> k = 3
    out = _propose(gen, first)
    assert out.num_valid.tolist() == [3, 3, 0, 0]
    assert out.draft_token_ids.shape == (gen.B, 5)
    assert out.draft_token_ids[0].tolist() == [
        (int(first[0]) + j + 1) % 7 for j in range(5)
    ]  # all 5 are handed out, nv says how many count
    assert gen.choose_hist == {3: 1}
    spec.ad.times = {3: 65.0, 5: 66.0}  # k=5 as cheap: chosen
    gen.spec_conf[:] = 0.99
    assert _propose(gen, first).num_valid.tolist() == [5, 5, 0, 0]
    assert gen.choose_hist == {3: 1, 5: 1}


def _verify(gen, first, k, nvs, lens=(3, 2), width=6):
    """one verify call like the plugin's: block [B, 1 + 5] (drafts right where offered, filler -1 beyond ``nv``)."""
    B = gen.B
    X = torch.full((B, width), -1, dtype=torch.int32)
    pos = torch.full((B, width), -1)
    nv = torch.zeros(B, dtype=torch.int32)
    for i, (f, L, v) in enumerate(zip(first.tolist(), lens, nvs)):
        X[i, 0], pos[i, 0] = f, L
        for j in range(v):
            X[i, 1 + j] = (f + 1 + j) % 7
        nv[i] = v
    return gen.decode_forward(
        X, pos, num_valid_drafts=nv, accepted_counts=torch.ones(B, dtype=torch.int32), spec_mode="argmax_ids"
    )


def test_verify_runs_the_runner_of_k_and_answers_width_k_plus_1():
    gen, m, spec = make_adaptive()
    first = _prefill_two(gen)
    r3, r5 = spec.runners[3], spec.runners[5]
    f0 = int(first[0])
    vo = _verify(gen, first, 3, [3, 3])  # both rows offered 3 drafts
    assert (
        r3.rounds[-1][0] == "round"
        and r3.rounds[-1][1].shape == (gen.B, 4)
        and not any(x[0] == "round" for x in r5.rounds)
    )
    assert vo.argmax_ids.shape == (gen.B, 4)
    assert vo.argmax_ids[0].tolist() == [(f0 + 1 + j) % 7 for j in range(4)]
    p0 = gen.slots.physical(0)
    assert int(gen.book.n[p0]) == 3 + 4  # token + 3 accepted drafts
    assert gen.spec_conf[p0].tolist() == pytest.approx([0.8] * 5)  # the round's confidence replaces the seeding's
    vo = _verify(
        gen, first, 5, [5, 5], lens=(7, 3 + 0)
    )  # a k=5 round on the other runner (row 1 has not advanced: positions are the test's business)
    assert r5.rounds[-1][1].shape == (gen.B, 6) and vo.argmax_ids.shape == (gen.B, 6)
    assert gen.spec.k_hist == {3: 1, 5: 1}


def test_verify_without_any_draft_uses_the_smallest_runner_and_forces_zero():
    gen, m, spec = make_adaptive()
    first = _prefill_two(gen)
    vo = _verify(gen, first, 3, [0, 0])
    r = spec.runners[3].rounds[-1]
    assert r[0] == "round" and r[1].shape[1] == 4 and r[3][0] == 0 and r[3][1] == 0
    assert vo.argmax_ids.shape == (gen.B, 4)


def test_verify_rows_without_drafts_are_forced_to_zero_beside_a_row_with_drafts():
    gen, m, spec = make_adaptive()
    first = _prefill_two(gen)
    _verify(gen, first, 3, [3, 0])
    r = spec.runners[3].rounds[-1]
    assert r[3][0] == -1 and r[3][1] == 0


def test_trimmed_offer_is_accepted_naturally_not_refused():
    """a row whose offer the scheduler cut to 0 < nv < k (context room): its filler columns never match, the device accepts at most nv."""
    gen, m, spec = make_adaptive()
    first = _prefill_two(gen)
    vo = _verify(gen, first, 3, [3, 2])
    r = spec.runners[3].rounds[-1]
    assert r[1].shape[1] == 4 and r[3][1] == -1  # natural accept (not forced)
    p1 = gen.slots.physical(1)
    assert (
        int(gen.book.n[p1]) == 2 + 3
    )  # token + the 2 offered drafts accepted; the filler (token 0) never matches the chain
    assert vo.argmax_ids.shape == (gen.B, 4)


def test_offer_beyond_the_block_is_refused(expect_error):
    gen, m, spec = make_adaptive()
    _prefill_two(gen)
    B = gen.B
    pos = torch.full((B, 6), -1)
    pos[0] = torch.arange(3, 9)
    with expect_error(ValueError, "valid drafts"):
        gen.decode_forward(
            torch.zeros(B, 6, dtype=torch.int32),
            pos,
            num_valid_drafts=torch.tensor([9] + [0] * (B - 1), dtype=torch.int32),
            accepted_counts=torch.ones(B, dtype=torch.int32),
            spec_mode="argmax_ids",
        )


def test_wrong_block_width_is_refused(expect_error):
    gen, m, spec = make_adaptive()
    _prefill_two(gen)
    with expect_error(ValueError, "verify block width"):
        _verify(gen, torch.tensor([1, 1]), 3, [0, 0], width=4)


# ---- the fixed-k path stays what it was -------------------------------------------------------------------------------------------------------
def test_fixed_k_path_is_unchanged_without_the_wrapper(expect_error):
    from models.demos.blackhole.deepseek_v41_flash.tests.test_vllm_interface import make_spec

    gen, m, spec = make_spec(8, 3)
    assert gen.spec_adapt is None
    first = gen.prefill_forward(
        torch.tensor([[1, 2, 3, 0]], dtype=torch.int32), prompt_lens=[3], empty_slots=[0], sampling_params=GREEDY
    )
    B = gen.B
    committed = torch.zeros(B, 4, dtype=torch.int32)
    committed[0, 0] = first[0]
    cpos = torch.full((B, 4), -1)
    cpos[0, 0] = 3
    out = gen.propose_draft_tokens(3, committed, cpos, torch.ones(B, dtype=torch.int32))
    assert out.num_valid[0] == 3 and not hasattr(gen, "choose_hist") or gen.choose_hist == {}
    pos = torch.full((B, 4), -1)
    pos[0] = torch.arange(3, 7)
    with expect_error(ValueError, "valid drafts"):  # partial counts are still refused
        gen.decode_forward(
            torch.zeros(B, 4, dtype=torch.int32),
            pos,
            num_valid_drafts=torch.tensor([2] + [0] * (B - 1), dtype=torch.int32),
            accepted_counts=torch.ones(B, dtype=torch.int32),
            spec_mode="argmax_ids",
        )
