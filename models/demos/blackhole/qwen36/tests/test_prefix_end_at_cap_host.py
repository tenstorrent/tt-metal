# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) for QWEN36_PREFIX_END_AT_CAP: the spec end-of-prompt checkpoint at the resume cap of the next
request that extends the prompt (prefix_cache.spec_followup_cap) instead of floor((L-3)/128)*128.

* the planner math for every L in a wide range: the new position is the follow-up's spec_resume_cap for any longer
  follow-up, equals the old position except for L mod 128 in [3, 63] (one block lower there), stays <= L-3;
* spec_save_points: flag off (end_block None) is today's result; on, the end point moves and the point count does not;
* plan_spec_prefix over a real GdnPrefixStateCache (device copies faked): turn 1 saves its points, turn 2 (prompt =
  turn 1 + new tokens, vLLM start_pos = floor(L/64)*64) resumes exactly at its cap for every L mod 128 with the flag on,
  and misses it exactly for L mod 128 in [3, 63] with the flag off (the production fallback);
* the flag reaches spec_save_points only when set (flag off = today's call, same kwargs), with the paged KV block size.

Run (worktree first on PYTHONPATH, ttnn from the built checkout):
  PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools $TT_METAL_HOME/python_env/bin/python -m pytest \
      models/demos/blackhole/qwen36/tests/test_prefix_end_at_cap_host.py -q
"""
import types

import pytest
import torch

from models.demos.blackhole.qwen36.tests.test_spec_prefix_fast_host import _cache, fake_ttnn
from models.demos.blackhole.qwen36.tt import prefix_cache as pc
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.prefix_cache import (
    SPEC_VLLM_BLOCK,
    spec_followup_cap,
    spec_learn_positions,
    spec_resume_cap,
    spec_save_points,
)

__all__ = ["fake_ttnn"]  # fixture re-export

ENV = "QWEN36_PREFIX_END_AT_CAP"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        ENV,
        "QWEN36_PREFIX_SNAPSHOT_MIN_LEN",
        "QWEN36_PREFIX_SNAPSHOT_STRIDE",
        "QWEN36_PREFIX_CACHE_STRIDE_LOW_PRIO",
        "QWEN36_PREFIX_CACHE_VALUE_EVICT",
    ):
        monkeypatch.delenv(name, raising=False)


def _old_end(L):
    return ((L - 3) // 128) * 128


# --------------------------------------------------------------------------- #
# planner math
# --------------------------------------------------------------------------- #
def test_followup_cap_is_the_resume_cap_of_every_longer_follow_up():
    for L in range(1, 3 * 8192 + 257):
        e = spec_followup_cap(L)
        sp = (L // SPEC_VLLM_BLOCK) * SPEC_VLLM_BLOCK  # what vLLM reports cached for a request extending the prompt
        assert e % 128 == 0 and 0 <= e
        assert e == 0 or e <= L - 3, L
        for extra in (1, 2, 3, 64, 127, 1000):
            assert e == spec_resume_cap(sp, L + extra), (L, extra)


def test_followup_cap_vs_todays_end_position():
    for L in range(8192, 8192 + 128 * 16):
        r = L % 128
        want = _old_end(L) - 128 if 3 <= r <= 63 else _old_end(L)
        assert spec_followup_cap(L) == want, L
    # the residues that move: 61 of 128 (the profile's "~48% of exact-prefix follow-ups")
    moved = [r for r in range(128) if spec_followup_cap(128 * 80 + r) != _old_end(128 * 80 + r)]
    assert moved == list(range(3, 64))


@pytest.mark.parametrize(
    "L,want",
    [
        (8492, 8320),  # the profile's harness case: old checkpoint 8448, follow-up cap 8320
        (8448, 8320),  # r = 0: unchanged (old = 8320 too)
        (8450, 8320),  # r = 2: unchanged
        (8451, 8320),  # r = 3: old 8448
        (8511, 8320),  # r = 63: old 8448
        (8512, 8448),  # r = 64: unchanged
        (8575, 8448),  # r = 127: unchanged
        (128, 0),
        (64, 0),
        (0, 0),
    ],
)
def test_followup_cap_examples(L, want):
    assert spec_followup_cap(L) == want


def test_followup_cap_other_block_sizes():
    for L in range(8192, 8192 + 1024):
        for block in (32, 64, 128):
            sp = (L // block) * block
            assert spec_followup_cap(L, block) == spec_resume_cap(sp, L + 500)


def test_regenerate_of_the_same_prompt():
    """vLLM caps a full hit at L-1 tokens: start_pos = floor((L-1)/64)*64. The new position is <= that cap for every
    L mod 128 except 64 (where it is one block above, as the old one is); the old one misses for 3..64."""
    for L in range(8192, 8192 + 128 * 4):
        cap = spec_resume_cap(((L - 1) // 64) * 64, L)
        r = L % 128
        assert (spec_followup_cap(L) <= cap) == (r != 64), L
        assert (_old_end(L) <= cap) == (r < 3 or r > 64), L


# --------------------------------------------------------------------------- #
# spec_save_points
# --------------------------------------------------------------------------- #
_GRID = [
    (L, s, learn, end_point)
    for L in (100, 2000, 5000, 8191, 8192, 8195, 8250, 8300, 8320, 8451, 8492, 8511, 8512, 10000, 20000, 40000, 65600)
    for s in (0, 128, 4096, 8192)
    if s < L
    for learn in (set(), {256}, spec_learn_positions(s + 64, L), spec_learn_positions((L // 64) * 64, L))
    for end_point in (True, False)
]


def test_save_points_without_end_block_are_todays():
    for L, s, learn, ep in _GRID:
        a = spec_save_points(L, s, learn, end_point=ep)
        b = spec_save_points(L, s, learn, end_point=ep, end_block=None)
        assert list(a) == list(b) and a.low == b.low and a.end == b.end


def test_save_points_end_block_moves_only_the_end_point():
    for L, s, learn, ep in _GRID:
        old = spec_save_points(L, s, learn, end_point=ep)
        new = spec_save_points(L, s, learn, end_point=ep, end_block=64)
        assert new.low == old.low - new.end  # strides untouched, except one the moved end point now sits on
        # never an extra split: the moved end point replaces the old one, or is dropped when the resume point (or a
        # learn point) already sits at / above it (then that snapshot serves the next turn)
        assert len(old) - 1 <= len(new) <= len(old), (L, s, learn, ep)
        assert set(old) - set(new) <= old.end and set(new) - set(old) <= new.end
        if not ep or L < 8192 or (L % 128) not in range(3, 64):
            assert list(new) == list(old) and new.end == old.end, (L, s, learn, ep)
        for p in new:
            assert s < p <= L - 3 and p % 128 == 0


def test_end_point_long_prompt_from_zero():
    for L in list(range(8192, 8192 + 512)) + list(range(16384, 16384 + 256)):
        pts = spec_save_points(L, 0, set(), end_block=64)
        e = spec_followup_cap(L)
        strides = set(range(8192, L - 2, 8192))
        if _old_end(L) != e:
            strides.discard(_old_end(L))  # the stride that stood in for today's end point goes with it
        assert pts.end == frozenset({e}) and e in pts
        assert list(pts) == sorted({e} | strides), L
        assert pts.low == frozenset(strides - {e})
        assert len(pts) <= len(spec_save_points(L, 0, set()))


def test_end_point_on_a_stride_point_moves_without_an_extra_split():
    """Today's end point 128q can coincide with a stride point (L in [8195, 8255]: 8192 is both). The moved end point
    128(q-1) replaces it; the stride is not kept as an extra split (one segment more and a 128-token replay)."""
    for L in range(8195, 8256):
        old = spec_save_points(L, 0, set())
        new = spec_save_points(L, 0, set(), end_block=64)
        assert list(old) == [8192] and old.end == frozenset({8192}) and old.low == frozenset()
        assert list(new) == [8064] and new.end == frozenset({8064}) and new.low == frozenset()
    # the moved end point lands ON a stride point: one split fewer than today (stride + end were two)
    old = spec_save_points(8492, 128, set())
    new = spec_save_points(8492, 128, set(), end_block=64)
    assert list(old) == [8320, 8448] and old.low == frozenset({8320}) and old.end == frozenset({8448})
    assert list(new) == [8320] and new.low == frozenset() and new.end == frozenset({8320})
    # same with a later stride, resumed at s = 128
    L = 128 + 2 * 8192 + 10
    old = spec_save_points(L, 128, set())
    new = spec_save_points(L, 128, set(), end_block=64)
    assert list(old) == [128 + 8192, 128 + 16384] and old.low == frozenset({128 + 8192})
    assert list(new) == [128 + 8192, 128 + 16384 - 128] and new.low == frozenset({128 + 8192})


def test_resumed_at_the_new_end_point_saves_no_end_point():
    """A request resuming exactly at spec_followup_cap(L) (the snapshot it restored) gets no end split; today's rule
    would split once more at the old position one block higher."""
    for r in range(3, 64):
        L = 128 * 80 + r
        e = spec_followup_cap(L)
        assert spec_save_points(L, e, set(), end_block=64) == []
        assert spec_save_points(L, e, set()) == [e + 128]


def test_learn_point_at_the_followup_cap_is_the_end_point():
    """A request whose start_pos is 128q (its prompt ends 3..63 tokens later): the learn points are 128q and 128(q-1);
    today 128q is marked as the end checkpoint, with the flag 128(q-1), the next turn's cap."""
    q = 70
    for r in range(3, 64):
        L = 128 * q + r
        learn = spec_learn_positions(128 * q, L)
        assert learn == {128 * q, 128 * (q - 1)}
        assert spec_save_points(L, 0, learn).end == frozenset({128 * q})
        assert spec_save_points(L, 0, learn, end_block=64).end == frozenset({128 * (q - 1)})


def test_no_end_point_when_resumed_at_or_past_it():
    L = 128 * 70 + 10
    e = spec_followup_cap(L)
    assert spec_save_points(L, e, set(), end_block=64).end == frozenset()
    assert spec_save_points(L, e + 128, set(), end_block=64).end == frozenset()


# --------------------------------------------------------------------------- #
# plan_spec_prefix over a real GdnPrefixStateCache: turn 1 saves, turn 2 resumes
# --------------------------------------------------------------------------- #
def _model(cache, kv=None):
    return types.SimpleNamespace(
        _prefix_cache=cache,
        args=types.SimpleNamespace(max_batch_size=8),
        _chunked_chunk_size=2048,
        _paged_kv_caches=kv,
    )


def _toks(n, seed=0):
    return torch.tensor([(seed * 7919 + 3 * i) % 997 + 1 for i in range(n)], dtype=torch.int32)


def _run(cache, model, toks, start):
    """plan + the cache saves of _prefill_fast / _prefill_for_spec_resumable (the state after tokens[:p] per point)."""
    plan = Qwen36Model.plan_spec_prefix(model, toks, len(toks), start)
    if plan is None:
        return 0, []
    s, src, points = plan
    for p in points:
        cache.save(toks, p, low_prio=p in points.low, protect=p in points.end)
    return s, list(points)


@pytest.mark.parametrize("flag", ["0", "1"])
def test_next_turn_resumes_at_its_cap(fake_ttnn, monkeypatch, flag):
    monkeypatch.setenv(ENV, flag)
    for r in range(128):
        L = 128 * 75 + r  # >= 8192: the end checkpoint is taken
        cache = _cache(64)
        m = _model(cache)
        turn2 = _toks(L + 700, seed=r)
        assert _run(cache, m, turn2[:L], 0)[0] == 0  # turn 1: from 0, saves its points
        start = (L // 64) * 64  # vLLM: turn 1's full blocks are cached
        cap = spec_resume_cap(start, L + 700)
        s, _ = _run(cache, m, turn2, start)
        if flag == "1":
            assert s == cap, (r, s, cap)
        else:
            # today: the miss set is exactly L mod 128 in [3, 63]; the resume drops to the 8192 stride
            assert (s == cap) == (r not in range(3, 64)), (r, s, cap)
            if r in range(3, 64):
                assert s == 8192


def test_flag_off_calls_spec_save_points_with_todays_kwargs(monkeypatch):
    seen = []
    real = pc.spec_save_points

    def rec(*a, **k):
        seen.append(k)
        return real(*a, **k)

    monkeypatch.setattr(pc, "spec_save_points", rec)

    class _C:
        def lookup(self, tokens, max_pos):
            return 0, None

    toks = list(range(9000))
    Qwen36Model.plan_spec_prefix(_model(_C()), toks, len(toks), 4096)
    assert "end_block" not in seen[-1]
    assert set(seen[-1]) == {"chunk", "min_len", "stride", "end_point"}
    monkeypatch.setenv(ENV, "1")
    Qwen36Model.plan_spec_prefix(_model(_C()), toks, len(toks), 4096)
    assert seen[-1]["end_block"] == SPEC_VLLM_BLOCK  # no paged KV bound: the serving block size
    kv = [(torch.zeros(10, 1, 32, 8), torch.zeros(10, 1, 32, 8))]  # [blocks, heads, block_size, head_dim]
    Qwen36Model.plan_spec_prefix(_model(_C(), kv=kv), toks, len(toks), 4096)
    assert seen[-1]["end_block"] == 32  # the bound paged KV's block size


def test_flag_on_plan_points(monkeypatch):
    monkeypatch.setenv(ENV, "1")

    class _C:
        def lookup(self, tokens, max_pos):
            return 0, None

    L = 8492
    _, _, pts = Qwen36Model.plan_spec_prefix(_model(_C()), list(range(L)), L, 0)
    assert list(pts) == [8192, 8320] and pts.end == frozenset({8320}) and pts.low == frozenset({8192})
    monkeypatch.setenv(ENV, "0")
    _, _, pts = Qwen36Model.plan_spec_prefix(_model(_C()), list(range(L)), L, 0)
    assert list(pts) == [8192, 8448] and pts.end == frozenset({8448})
