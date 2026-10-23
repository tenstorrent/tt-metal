# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the enumerable packed-verify bucket ladder.

Prerequisite for moving the dFlash fused-trace capture from per-session to
warmup (review item on tenstorrent/tt-metal#56048): the bucket set has to be
enumerable at config time.

``pv_bucket`` keys on prompt length, so the exact set is ~258 buckets at 256K.
The ladder is serveable instead because a bucket captured LARGER than a request
needs is numerically exact for it -- ``_pv_setup`` masks every column past the
live top to NEG. These tests pin that rounding-UP property against the real
formula, not against the ladder's own arithmetic.
"""

import pytest

# The gemma4 vLLM generator imports vllm at module scope (through
# tt_transformers.generator_vllm), so COLLECTING this file fails outright on a
# runner without vLLM installed -- which is the tt-metal unit-test job. Skip
# before the import rather than inside the tests: the failure is at import.
pytest.importorskip("vllm")
from models.demos.gemma4.tt.generator_vllm import dflash_bucket_for, dflash_pv_bucket_ladder

_H, _V = 2048, 5
_MAX_CTX = 262144


def _exact_bucket(start, horizon=_H, verify=_V):
    """Mirror of DFlashFusedDecoder.pv_bucket -- the value capture would use."""
    return ((start + horizon + (verify + 1) + 64 + 1023) // 1024) * 1024


@pytest.fixture(autouse=True)
def _pin_env(monkeypatch):
    monkeypatch.setenv("GEMMA4_DFLASH_SERVE_HORIZON", str(_H))
    monkeypatch.setenv("GEMMA4_DFLASH_VERIFY", str(_V))


def test_ladder_is_enumerable_and_small():
    """Logarithmic in context: a 256K server must not need ~258 captures."""
    ladder = dflash_pv_bucket_ladder(_MAX_CTX)
    assert len(ladder) <= 12
    assert ladder == sorted(set(ladder))
    assert all(r % 1024 == 0 for r in ladder)


@pytest.mark.parametrize("start", [0, 1, 127, 128, 1000, 4096, 8192, 16384, 32768, 65536, 131072, 200000, _MAX_CTX])
def test_every_prompt_length_rounds_up_to_a_rung(start):
    """The whole point: no prompt inside max_context may be uncovered, and the
    chosen rung must be >= the bucket capture would have used (never smaller,
    or the verify would attend past what it captured)."""
    ladder = dflash_pv_bucket_ladder(_MAX_CTX)
    rung = dflash_bucket_for(start, ladder)
    assert rung is not None
    assert rung >= _exact_bucket(start)


def test_rounding_waste_is_bounded():
    """Rounding up costs verify width, so the ladder may not be arbitrarily
    coarse. Doubling rungs bound it below 2x the exact bucket."""
    ladder = dflash_pv_bucket_ladder(_MAX_CTX)
    worst = max(dflash_bucket_for(s, ladder) / _exact_bucket(s) for s in range(0, 65536, 97))
    assert worst < 2.0


def test_beyond_ladder_is_none_not_a_wrong_rung():
    """A request past the ladder must signal fallback, NOT silently get the
    largest rung -- that would attend past the captured width."""
    ladder = dflash_pv_bucket_ladder(_MAX_CTX)
    assert dflash_bucket_for(_MAX_CTX * 2, ladder) is None


def test_cap_keeps_the_largest_rungs():
    """Capping trades wasted width on SHORT prompts for a bounded capture count.
    Dropping the largest rungs instead would leave long prompts uncovered."""
    full = dflash_pv_bucket_ladder(_MAX_CTX)
    capped = dflash_pv_bucket_ladder(_MAX_CTX, max_rungs=3)
    assert len(capped) == 3
    assert capped == full[-3:]
    # Long prompts stay covered; that is the invariant the cap must not break.
    assert dflash_bucket_for(_MAX_CTX, capped) is not None


def test_horizon_and_verify_widen_the_smallest_rung():
    """The bucket carries the generation horizon, so a bigger horizon must not
    silently leave a short prompt under-covered."""
    small = dflash_pv_bucket_ladder(_MAX_CTX, horizon=512, verify=_V)
    large = dflash_pv_bucket_ladder(_MAX_CTX, horizon=8192, verify=_V)
    assert large[0] > small[0]
    assert dflash_bucket_for(0, large) >= _exact_bucket(0, horizon=8192)
