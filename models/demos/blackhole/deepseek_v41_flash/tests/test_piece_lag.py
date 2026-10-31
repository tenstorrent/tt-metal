# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the piece lag (DSV41_PIECE_LAG): a prefill piece that does not end its prompt is cut back to a chunk boundary / before an image span; vLLM's next piece (reported start = the
uncut end) resumes at the model's carried end (``PrefillTracker.find(lag=True)`` + ``DeepseekV41ForCausalLM._lag_start``) instead of recomputing from position 0."""
from types import SimpleNamespace

import torch

from models.demos.blackhole.deepseek_v41_flash.tt import vllm_state as VS
from models.demos.blackhole.deepseek_v41_flash.tt.generator_vllm import DeepseekV41ForCausalLM


def _gen(piece_lag, resume):
    return SimpleNamespace(piece_lag=piece_lag, inprog=VS.PrefillTracker(), m=SimpleNamespace(pf_resume=resume))


def _lag_start(g, p, st):
    return DeepseekV41ForCausalLM._lag_start(g, p, st)


def test_exact_continuation_unchanged():
    g = _gen(True, {3: 1024})
    toks = torch.arange(5000)
    g.inprog.update(3, 1024, toks)
    assert g.inprog.find(1024, toks[:1024], lag=True) == 3
    assert _lag_start(g, 3, 1024) == 1024


def test_cut_back_piece_resumes_at_the_carried_end():
    g = _gen(True, {3: 32256})
    full = torch.arange(33000)
    g.inprog.update(3, 32256, full, reported=32593)
    assert g.inprog.find(32593, full[:32593], lag=False) is None  # legacy: no match -> recompute from 0
    p = g.inprog.find(32593, full[:32593], lag=True)
    assert p == 3
    assert _lag_start(g, p, 32593) == 32256


def test_lag_off_is_the_legacy_behaviour():
    g = _gen(False, {3: 32256})
    full = torch.arange(33000)
    g.inprog.update(3, 32256, full, reported=32593)
    assert g.inprog.find(32593, full[:32593], lag=g.piece_lag) is None
    assert _lag_start(g, 3, 32593) is None


def test_lost_state_or_other_prompt_is_not_resumed():
    full = torch.arange(33000)
    g = _gen(True, {3: 512})  # the model's carried position moved (another prompt took the slot)
    g.inprog.update(3, 32256, full, reported=32593)
    assert _lag_start(g, 3, 32593) is None
    other = full.clone()
    other[100] = -5
    g2 = _gen(True, {3: 32256})
    g2.inprog.update(3, 32256, full, reported=32593)
    assert g2.inprog.find(32593, other[:32593], lag=True) is None  # a different prompt prefix


def test_skipped_piece_bumps_the_reported_end():
    full = torch.arange(40000)
    g = _gen(True, {3: 32256})
    g.inprog.update(3, 32256, full, reported=32593)
    g.inprog.bump(3, 36688)  # nothing computed for the next piece either (cut before the image)
    assert g.inprog.find(36688, full[:36688], lag=True) == 3
    assert g.inprog.find(32593, full[:32593], lag=True) is None
    assert _lag_start(g, 3, 36688) == 32256


def test_two_users_with_the_same_prefix_get_distinct_slots():
    full = torch.arange(40000)
    g = _gen(True, {3: 32256, 5: 32256})
    g.inprog.update(3, 32256, full, reported=32593)
    g.inprog.update(5, 32256, full, reported=32593)
    a = g.inprog.find(32593, full[:32593], lag=True)
    b = g.inprog.find(32593, full[:32593], exclude=[a], lag=True)
    assert {a, b} == {3, 5}


def test_drop_forgets_reported():
    g = _gen(True, {})
    g.inprog.update(3, 512, torch.arange(1000), reported=700)
    g.inprog.drop(3)
    assert 3 not in g.inprog.users and 3 not in g.inprog.reported


# ---- piece ends vs image spans (VisionBook.cut_end) -------------------------------------------------------------------------------------------------------------------------------
import random

from models.demos.blackhole.deepseek_v41_flash.tt import vision_state as VSN

IMG = 7


def _prompt(spans_at, L):
    t = torch.ones(L, dtype=torch.long)
    for s, n in spans_at:
        t[s : s + n] = IMG
    return t


def test_cut_end_never_lands_inside_an_earlier_span():
    # the failure of the first device run: image 32203..32397, next image at ~32600 straddles the 32768 budget; aligned-down 32256 is inside the first image
    t = _prompt([(32203, 194), (32600, 194)], 40000)
    c = VSN.cut_end(t, [194, 194], IMG, 32768, align=512)
    assert c == 31744 and not (32203 < c < 32397)
    # the lag cut to the chunk boundary 32256 (a non-final piece ending at 32300) must also move below the span
    assert VSN.cut_end(t, [194, 194], IMG, 32256, align=512) == 31744


def test_cut_end_unchanged_when_no_span_is_cut():
    t = _prompt([(100, 50), (900, 50)], 3000)
    assert VSN.cut_end(t, [50, 50], IMG, 2048, align=512) == 2048
    assert VSN.cut_end(t, [50, 50], IMG, 900, align=512) == 900  # ends exactly at a span start: fine


def _plan(rng, t, span_lens, C):
    """vLLM-like pieces (random budget, sometimes rolled back before an image, sometimes ending inside one) + the adapter's lag cut -> list of (start, end) the model computes"""
    L = t.numel()
    rep, carried, computed = 0, 0, []
    while rep < L:
        e = min(L, rep + rng.choice([97, 600, 1500, 4095, 4096, 9000, 32768]))
        if e < L and rng.random() < 0.5:  # scheduler rolls back before an image that does not fit
            starts = (t[:e] == IMG).nonzero().reshape(-1)
            if starts.numel():
                pass
        final = e == L
        end = e
        if not final:
            a = e // C * C
            end = VSN.cut_end(t, span_lens, IMG, a, align=C) if a > carried else a
        if end > carried:
            assert end == L or end % C == 0, (carried, end)
            computed.append((carried, end))
            carried = end
        rep = e
    return computed


def test_random_layouts_piece_ends_outside_spans_and_full_coverage():
    rng = random.Random(1234)
    C = 512
    for _ in range(300):
        pos, spans = rng.randrange(0, 700), []
        for _i in range(rng.randrange(1, 40)):
            n = rng.choice([66, 130, 194, 258, 322, 700])
            spans.append((pos, n))
            pos += n + rng.choice([0, 1, 5, 80, 400, 1500])
        L = pos + rng.randrange(1, 800)
        t = _prompt(spans, L)
        comp = _plan(rng, t, [n for _, n in spans], C)
        assert comp[0][0] == 0 and comp[-1][1] == L
        for (a, b), (c, d) in zip(comp, comp[1:]):
            assert b == c  # contiguous: no position skipped or recomputed
        for _, end in comp[:-1]:
            assert end % C == 0
            assert not any(s < end < s + n for s, n in spans), (end, spans)
