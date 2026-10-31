# SPDX-License-Identifier: Apache-2.0
"""Device-free contracts for the DFlash scheduler look-ahead wrapper."""

from __future__ import annotations

import laguna_vllm_ext.dflash_lookahead as dflash_lookahead
from laguna_vllm_ext.dflash_lookahead import DFLASH_LOOKAHEAD_TOKENS, PATCH_MARKER, _patch_scheduler


def _scheduler_class():
    class Scheduler:
        def __init__(self, lookahead=0):
            self.num_lookahead_tokens = lookahead

    return Scheduler


def test_lookahead_raised_to_verify_rows_only_with_dflash(monkeypatch):
    monkeypatch.setitem(dflash_lookahead._APPLIED, "tokens", 0)
    Scheduler = _scheduler_class()
    assert _patch_scheduler(Scheduler)
    assert not _patch_scheduler(Scheduler)
    assert Scheduler.__dict__[PATCH_MARKER]

    monkeypatch.delenv("TT_LAGUNA_DFLASH", raising=False)
    assert Scheduler().num_lookahead_tokens == 0
    assert dflash_lookahead.applied_lookahead_tokens() == 0

    monkeypatch.setenv("TT_LAGUNA_DFLASH", "1")
    assert Scheduler().num_lookahead_tokens == DFLASH_LOOKAHEAD_TOKENS == 16
    assert dflash_lookahead.applied_lookahead_tokens() == 16
    # A larger look-ahead someone else configured is kept.
    assert Scheduler(lookahead=20).num_lookahead_tokens == 20
    assert dflash_lookahead.applied_lookahead_tokens() == 20
