# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) for the prefix-cache restore + chunk_start replay planning of the traced fast spec prefill
(MTPSpecEngine._prefill_fast / _fast_segment with a plan from Qwen36Model.plan_spec_prefix). A real
GdnPrefixStateCache (device copies faked) drives hit / partial hit / full hit / miss / eviction; the model is the fake of
test_spec_fastprefill_host.py, which records every replay.

Run (worktree first on PYTHONPATH, ttnn from the built checkout):
  PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools $TT_METAL_HOME/python_env/bin/python -m pytest \
      models/demos/blackhole/qwen36/tests/test_spec_prefix_fast_host.py -q
"""
import types
from collections import OrderedDict

import pytest
import torch

from models.demos.blackhole.qwen36.tests.test_spec_fastprefill_host import (
    BS,
    SCRATCH,
    _engine,
    _fake_model,
    _row,
    fake_ttnn,
)
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.prefix_cache import GdnPrefixStateCache
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine, mtp_prefill_inputs

__all__ = ["fake_ttnn"]  # fixture re-export


def _cache(num_slots=4):
    """A GdnPrefixStateCache without the device allocation: slot i holds ('rec', i) / ('carry', i) strings."""
    c = GdnPrefixStateCache.__new__(GdnPrefixStateCache)
    c.model, c.name, c.num_slots, c.debug = None, "test", num_slots, False
    c._dns = [types.SimpleNamespace(rec_state="bound_rec", conv_carry="bound_carry")]
    c._slots = [[(("rec", i), ("carry", i))] for i in range(num_slots)]
    c._free = list(range(num_slots))
    c._lru, c._by_pos, c._positions = OrderedDict(), {}, []
    c.stats = dict(lookups=0, hits=0, tokens_skipped=0, saves=0, evictions=0)
    return c


def _plan(cache, tokens, start, B=8):
    m = types.SimpleNamespace(
        _prefix_cache=cache, args=types.SimpleNamespace(max_batch_size=B), _chunked_chunk_size=2048
    )
    return Qwen36Model.plan_spec_prefix(m, torch.tensor(tokens), len(tokens), start)


def _prompt(L, seed=0):
    return [(seed * 7919 + 3 * i) % 997 + 1 for i in range(L)]


def _model(calls, cache, trace_lengths):
    m = _fake_model(calls, trace_lengths=trace_lengths)
    m._prefix_cache = cache
    # record start / reset_gdn of every chunk replay
    m._prefill_traced_chunked_tp = (
        lambda *a, **k: calls.append(
            ("chunks", a[3], a[5], k.get("start", 0), k.get("reset_gdn", True), k.get("run_tail"))
        )
        or "REPL"
    )
    return m


def _run(Tp, cache, trace_lengths, start, tokens=None, B=8):
    tokens = tokens or _prompt(Tp)
    calls = []
    plan = _plan(cache, tokens, start, B)
    e = _engine(B, model=_model(calls, cache, trace_lengths))
    prompt = torch.tensor([tokens], dtype=torch.int32)
    captured = [False]
    mtp = [tokens]
    on_chunk_calls = []
    out = MTPSpecEngine._prefill_fast(
        e,
        3,
        tokens,
        prompt,
        _row(),
        Tp,
        lambda h, cs, L: on_chunk_calls.append((cs, L)),
        captured,
        plan=plan,
        mtp_prompt=mtp,
    )
    return out, calls, plan, captured[0], mtp, tokens, on_chunk_calls


def _seed(cache, tokens, positions):
    for p in positions:
        cache.save(torch.tensor(tokens), p)


def test_partial_hit_short_tail_restores_then_replays_at_chunk_start(fake_ttnn):
    Tp = 1600
    tokens = _prompt(Tp)
    cache = _cache()
    _seed(cache, tokens, [1024])
    # vLLM hit of 1088 cached tokens: cap = floor(1087/128)*128 = 1024 -> the snapshot at 1024 is used.
    out, calls, plan, captured, _, _, oc = _run(Tp, cache, trace_lengths=(Tp - 1024,), start=1088, tokens=tokens)
    assert plan[0] == 1024
    kinds = [c[0] for c in calls]
    assert kinds == ["bind", "rope", "replay", "unbind", "slot_write"], kinds
    # restore: the snapshot's rec_state / conv_carry are copied INTO the bound scratch before the replay
    restores = [(src, dst) for src, dst in fake_ttnn["copy"] if dst in ("bound_rec", "bound_carry")]
    assert [d for _, d in restores] == ["bound_rec", "bound_carry"] and restores[0][0][0] == "rec"
    _, toks, pt, L, strict, cs, carry, spec_inputs = calls[2]
    assert (L, cs, carry, strict) == (Tp - 1024, 1024, True, False)
    assert torch.equal(toks[:, :L], torch.tensor([tokens[1024:]], dtype=torch.int32))  # the segment's tokens
    ref = mtp_prefill_inputs(tokens, 1024, 2048, _row(), BS, SCRATCH, 64)
    assert all(torch.equal(a, b) for a, b in zip(spec_inputs, ref))
    assert captured is True and out.shape[0] > 0


def test_hit_with_chunks_and_tail_uses_start_and_no_reset(fake_ttnn):
    Tp = 5000
    tokens = _prompt(Tp)
    cache = _cache()
    _seed(cache, tokens, [2048])
    # vLLM hit 2200: cap = floor(2199/128)*128 = 2176 -> snapshot 2048 used; learn point 2176 splits the prefill:
    # [2048, 2176) (128 tokens, short trace) then [2176, 5000) = 1 full chunk at 2176 + tail 776 at 4224.
    out, calls, plan, *_ = _run(Tp, cache, trace_lengths=(128, 776), start=2200, tokens=tokens)
    assert plan[0] == 2048 and plan[2] == [2176]
    seq = [(c[0],) + ((c[3], c[5], c[6]) if c[0] == "replay" else c[1:]) for c in calls if c[0] in ("replay", "chunks")]
    assert seq == [
        ("replay", 128, 2048, True),
        ("chunks", 1, 776, 2176, False, False),
        ("replay", 776, 4224, True),
    ], seq
    assert cache.stats["saves"] == 2  # the seeded 2048 + the new 2176


def test_full_hit_leaves_min_tail_of_three(fake_ttnn):
    Tp = 1027  # a hit on 1026 cached tokens: cap = 1024 (>= 3 tokens remain)
    tokens = _prompt(Tp)
    cache = _cache()
    _seed(cache, tokens, [1024])
    out, calls, plan, *_ = _run(Tp, cache, trace_lengths=(3,), start=1026, tokens=tokens)
    assert plan[0] == 1024
    rep = [c for c in calls if c[0] == "replay"][-1]
    assert rep[3] == 3 and rep[5] == 1024 and rep[6] is True
    # whole prompt cached by vLLM (start = Tp - 1 is the max): still at least 3 tokens recomputed, never 0
    assert _plan(cache, tokens, Tp - 1)[0] <= Tp - 3


def test_miss_runs_from_zero_without_restore_and_learns(fake_ttnn):
    Tp = 1600
    tokens = _prompt(Tp)
    cache = _cache()
    # cold request that vLLM reports as hit 1088 (a different request wrote the blocks, no GDN snapshot yet):
    # s = 0, learn points 1024 (cap) and 1024 (floor): segments [0,1024) + [1024,1600); no restore.
    out, calls, plan, captured, mtp, _, _ = _run(Tp, cache, trace_lengths=(1024, 576), start=1088, tokens=tokens)
    assert plan[0] == 0 and plan[1] is None and plan[2] == [1024]
    reps = [c for c in calls if c[0] == "replay"]
    assert [(r[3], r[5], r[6]) for r in reps] == [(1024, 0, False), (576, 1024, True)]
    # first segment: the drafter fuses only prompt[:1025] (slot 1023 <- token 1024), no rows past the segment
    toks0 = reps[0][7][0]
    assert torch.equal(toks0[0, :1023], torch.tensor(tokens[1:1024], dtype=torch.int32))
    assert int(toks0[0, 1023]) == tokens[1024] and not toks0[0, 1024:].any()
    # snapshot taken between segments, nothing restored
    assert cache.stats["saves"] == 1 and cache._positions == [1024]
    assert not any(dst in ("bound_rec", "bound_carry") for _, dst in fake_ttnn["copy"])  # s == 0: nothing restored
    assert mtp[0] == tokens  # restored to the full prompt after the call
    # second request, same prefix: the snapshot hits
    plan2 = _plan(cache, tokens, 1088)
    assert plan2[0] == 1024 and plan2[1][0] is cache


def test_masked_tail_when_no_trace_for_a_resumed_segment(fake_ttnn):
    Tp = 1600
    tokens = _prompt(Tp)
    cache = _cache()
    _seed(cache, tokens, [1024])
    out, calls, plan, captured, *_ = _run(Tp, cache, trace_lengths=(), start=1088, tokens=tokens)
    kinds = [c[0] for c in calls]
    # no short trace: the eager masked bucket serves the segment at chunk_start 1024 (state carried, never declined)
    assert ("masked", Tp - 1024, 1024) in calls and out is not None


def test_cold_short_prompt_without_trace_still_declines_before_side_effects(fake_ttnn):
    Tp = 1600
    cache = _cache()
    calls = []
    plan = _plan(cache, _prompt(Tp), 1088)  # s = 0, points [1024]
    e = _engine(8, model=_model(calls, cache, trace_lengths=()))
    tokens = _prompt(Tp)
    res = MTPSpecEngine._prefill_fast(
        e,
        3,
        tokens,
        torch.tensor([tokens], dtype=torch.int32),
        _row(),
        Tp,
        lambda *a: None,
        [False],
        plan=plan,
        mtp_prompt=[tokens],
    )
    assert res is None and cache.stats["saves"] == 0
    assert [c[0] for c in calls if c[0] in ("slot_write", "masked")] == []


def test_eviction_falls_back_to_earlier_snapshot_then_zero(fake_ttnn):
    tokens = _prompt(4000)
    cache = _cache(num_slots=2)
    _seed(cache, tokens, [128, 256])
    assert _plan(cache, tokens, 300)[0] == 256 and cache.stats["evictions"] == 0
    _seed(cache, tokens, [384])  # evicts the least recently used: 128 (256 was just looked up)
    assert cache.stats["evictions"] == 1 and cache._positions == [256, 384]
    assert _plan(cache, tokens, 200)[0] == 0  # cap 128: that snapshot is gone -> full recompute (correct, slower)
    assert _plan(cache, tokens, 300)[0] == 256
    other = tokens[:200] + [5] * 3800  # same first 200 tokens, divergent after: the 256 snapshot digest must not match
    assert _plan(cache, other, 300)[0] == 0


@pytest.mark.parametrize("start", [127, 128, 129, 255, 256])
def test_hit_on_block_boundaries_never_resumes_at_start(start, fake_ttnn):
    tokens = _prompt(2000)
    cache = _cache()
    _seed(cache, tokens, [128, 256])
    s = (_plan(cache, tokens, start) or (0,))[0]
    assert s == {127: 0, 128: 0, 129: 128, 255: 128, 256: 128}[start]
    assert s <= max(start - 1, 0)  # the drafter slot s-1 must fuse a token inside the cached span


def test_mid_prompt_segment_ending_on_a_chunk_boundary_skips_logits(fake_ttnn):
    Tp = 4096 + 600
    tokens = _prompt(Tp)
    cache = _cache()
    # cold request (start_pos 0): the last full-chunk boundary 4096 is a save point -> segment [0, 4096) = 2 chunks, tail 0
    out, calls, plan, captured, *_ = _run(Tp, cache, trace_lengths=(600,), start=0, tokens=tokens)
    assert plan[0] == 0 and plan[2] == [4096]
    assert not any(c[0] == "last_logits" for c in calls[: [c[0] for c in calls].index("replay")])  # mid segment: none
    assert cache.stats["saves"] == 1 and cache._positions == [4096]
    reps = [c for c in calls if c[0] == "replay"]
    assert [(r[3], r[5], r[6]) for r in reps] == [(600, 4096, True)]
    assert [c[1:] for c in calls if c[0] == "chunks"] == [(2, 0, 0, True, False)]
