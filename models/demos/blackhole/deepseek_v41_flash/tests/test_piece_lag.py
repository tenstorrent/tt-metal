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
