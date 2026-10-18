# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MTP serving verify-width ladder and its per-step selection.

``serving_setup`` captured ONE fused trace whose packed-verify ``S_k`` covered
``anchor_pos + max_new_tokens``, and a replay may only attend positions the mask
it was captured with covers. That made the horizon a hard GENERATION BUDGET: at
the default ``GEMMA4_MTP_SERVE_HORIZON`` the class ended a request with a full
end-of-sequence row at 2048 output tokens, delivered to the caller as an
ordinary stop. The ladder replaces the budget with a migration.

Host-only: the ladder is config-time arithmetic and selection is arithmetic over
the captured set, so the decoder is built with ``__new__`` and no device.
"""

import pytest

# The gemma4 vLLM generator imports vllm at module scope (through
# tt_transformers.generator_vllm), so COLLECTING this file fails outright on a
# runner without vLLM installed -- which is the tt-metal unit-test job. Skip
# before the import rather than inside the tests: the failure is at import.
pytest.importorskip("vllm")
from models.demos.gemma4.tt.generator_vllm import mtp_pv_width_ladder
from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder


def _session(widths, ladder=None, draft_len=5):
    """A decoder carrying a captured width set and nothing else."""
    spec = SpeculativeDecoder.__new__(SpeculativeDecoder)
    spec.draft_len = draft_len
    spec._srv_widths = {int(w): {"pv_S_k": int(w)} for w in widths}
    spec._srv_ladder = sorted(int(w) for w in (ladder if ladder is not None else widths))
    return spec


def test_ladder_doubles_and_covers_the_whole_context():
    ladder = mtp_pv_width_ladder(262144, draft_len=5)
    assert ladder == sorted(ladder)
    assert ladder[0] == 1024
    assert ladder[-1] >= 262144 + 7
    # doubling -> logarithmic rung count
    assert len(ladder) <= 12
    for a, b in zip(ladder, ladder[1:]):
        assert b >= 2 * a or b == ladder[-1]


def test_every_position_in_the_context_has_a_rung():
    ladder = mtp_pv_width_ladder(262144, draft_len=5)
    spec = _session(ladder)
    for pos in (0, 1, 1000, 2048, 4096, 131072, 262143):
        assert spec.srv_width_for(pos) is not None, pos


def test_selection_is_the_narrowest_covering_width():
    """Width is per-iteration BANDWIDTH here -- the masks are rebuilt on the
    host and uploaded at [1, 1, P_v, S_k] every replay -- so a request must not
    sit on a wider trace than its position needs."""
    spec = _session([1024, 2048, 4096], draft_len=5)
    assert spec.srv_width_for(0) == 1024
    assert spec.srv_width_for(1024 - 7) == 1024
    assert spec.srv_width_for(1024 - 6) == 2048
    assert spec.srv_width_for(2048 - 6) == 4096


def test_position_past_the_captured_set_is_none_so_the_caller_can_capture():
    """None is the signal to take the next rung off the ladder, which is what
    replaces ending the request at the horizon."""
    spec = _session([1024], ladder=[1024, 2048, 4096], draft_len=5)
    assert spec.srv_width_for(2000) is None
    need = 2000 + spec.draft_len + 2  # 2007
    assert next(r for r in spec._srv_ladder if r >= need) == 2048
    # and a position past that rung takes the one after it
    assert next(r for r in spec._srv_ladder if r >= 3000 + spec.draft_len + 2) == 4096


def test_no_captured_widths_means_no_width_set():
    """A session set up the old way (single capture) must not take the width
    path: srv_width_for reports None and serving_step leaves it alone."""
    spec = SpeculativeDecoder.__new__(SpeculativeDecoder)
    spec.draft_len = 5
    spec._srv_widths = None
    assert spec.srv_width_for(0) is None


@pytest.mark.parametrize("draft_len", [3, 5, 7])
def test_tail_follows_the_draft_length(draft_len):
    """A verify block is P_v = K + 1 rows at pos..pos+K, so the width has to
    cover K + 2 past the position, not just the position."""
    ladder = mtp_pv_width_ladder(8192, draft_len=draft_len)
    spec = _session(ladder, draft_len=draft_len)
    top = max(ladder)
    assert spec.srv_width_for(top - draft_len - 2) == top
    assert spec.srv_width_for(top - draft_len - 1) is None


def test_cap_keeps_the_largest_rungs():
    ladder = mtp_pv_width_ladder(262144, draft_len=5, max_rungs=4)
    assert len(ladder) == 4
    assert ladder[-1] == max(mtp_pv_width_ladder(262144, draft_len=5))
