# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Gemma4 MTP decode block loop.

One MTP iteration commits an accepted prefix + bonus, which is VARIABLE (1..N)
because acceptance is content-dependent. The step's row must still be exactly N
wide, so iterations run back-to-back until the row is full.

Why this matters: a short row has to be padded and every pad value is wrong.
``-1`` puts an invalid token id on the wire; EOS terminates the request at the
first pad, because upstream trims committed tokens at the first stop token. That
regression capped solo generation at about one block (osl=128 returned 6 tokens)
while the batched baseline path, which never pads, ran to full length -- so a
throughput sweep did not catch it.

Host-only: the loop is driven through a stub session, no device.
"""

import torch

from models.demos.gemma4.tt.generator_vllm import Gemma4MTPForCausalLM as MTP


class _StubSession:
    """Returns a scripted (committed, accepted_m) per ``serving_step`` call."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

    def serving_step(self, cur_token, cur_pos):
        self.calls.append((cur_token, cur_pos))
        committed = self.scripted.pop(0)
        return committed, len(committed) - 1  # m = accepted, committed = m+1


class _Harness:
    """Minimal stand-in exposing only what ``decode_forward`` touches on the
    solo path with a live session."""

    _SPEC_N = 6
    _spec_budget_end = 10_000
    _spec_horizon = 2048  # only read by the horizon-exhausted log line

    def __init__(self, scripted, start_pos=100):
        self._spec = _StubSession(scripted)
        self._spec_cur = (11, start_pos)
        self._spec_pending = None  # no bootstrap on this path
        self._spec_pending_owner = None
        self._eos = 1

    def _eos_fill_id(self):
        return self._eos

    # real implementations, so the tests exercise shipped logic
    _spec_pt_identity = MTP._spec_pt_identity
    _spec_pending_is_mine = MTP._spec_pending_is_mine
    run = MTP.decode_forward


def _row(h, tokens=None):
    """Drive just the block loop by calling decode_forward with a B=1 input."""
    toks = torch.tensor([[11]], dtype=torch.int32) if tokens is None else tokens
    return h.run(tokens=toks, start_pos=torch.tensor([[100]], dtype=torch.int32))


def test_short_iterations_are_concatenated_to_a_full_row():
    """Three 2-token iterations must produce ONE 6-wide row, not a padded 2."""
    h = _Harness([[21, 22], [23, 24], [25, 26]])
    out = _row(h)
    assert out.shape == (1, 6)
    assert out[0].tolist() == [21, 22, 23, 24, 25, 26]
    assert len(h._spec.calls) == 3  # looped rather than padding after one


def test_position_advances_by_tokens_emitted():
    """The session must resume from what was EMITTED. Iterations are driven from
    (token, pos), so a wrong position re-drafts the wrong prefix."""
    h = _Harness([[21, 22], [23, 24], [25, 26]], start_pos=100)
    _row(h)
    tok, pos = h._spec_cur
    assert pos == 100 + 6
    assert tok == 26
    # each iteration was driven from the previous one's last token / next pos
    assert h._spec.calls == [(11, 100), (22, 102), (24, 104)]


def test_overshoot_truncates_and_resumes_from_emitted():
    """A final iteration may overshoot N. The extra tokens are dropped and the
    position reflects the row, so the next step re-drafts and overwrites them."""
    h = _Harness([[21, 22, 23, 24], [25, 26, 27, 28]], start_pos=200)
    out = _row(h)
    assert out[0].tolist() == [21, 22, 23, 24, 25, 26]
    tok, pos = h._spec_cur
    assert pos == 200 + 6 and tok == 26


def test_full_iteration_needs_no_loop():
    """High acceptance fills the row in one call; do not spin further."""
    h = _Harness([[21, 22, 23, 24, 25, 26]])
    out = _row(h)
    assert out[0].tolist() == [21, 22, 23, 24, 25, 26]
    assert len(h._spec.calls) == 1


def test_eos_stops_the_loop_and_fills_the_tail():
    """A genuine stop is the ONLY case that may leave a short row; the tail is
    EOS so upstream trims there."""
    h = _Harness([[21, 1], [99, 99]])  # eos id is 1
    out = _row(h)
    assert out[0].tolist()[:2] == [21, 1]
    assert out[0].tolist()[2:] == [1, 1, 1, 1]  # EOS-filled tail
    assert len(h._spec.calls) == 1  # stopped, did not consume the next script


def test_horizon_exhaustion_emits_eos_without_stepping():
    """Past the horizon the packed verify would attend past its capture, so the
    request ends cleanly instead of stepping again."""
    h = _Harness([[21, 22]])
    h._spec_budget_end = 100  # cur_pos starts at 100
    out = _row(h)
    assert out[0].tolist() == [1, 1, 1, 1, 1, 1]
    assert h._spec.calls == []
