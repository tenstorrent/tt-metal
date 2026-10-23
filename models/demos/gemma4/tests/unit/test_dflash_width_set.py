# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dFlash packed-verify WIDTH SET selection and migration.

The verify widths are derived from ``max_model_len`` at config time and all of
them are captured in warmup, so serving never captures and no captured shape
depends on a request existing (vllm-tt-plugin#110 section 8). Per step the
narrowest covering width is selected, and a request that outgrows its width
moves to the next one -- which is what stops a captured width from acting as a
generation budget (the review finding on tt-metal#56048: at the default horizon
``_spec_budget_end`` was a server-wide 2048-token output cap, delivered to the
caller as an ordinary stop).

Host-only: selection is pure arithmetic over the captured set, so the decoder is
built with ``__new__`` and no device.
"""

import pytest

# The gemma4 vLLM generator imports vllm at module scope (through
# tt_transformers.generator_vllm), so COLLECTING this file fails outright on a
# runner without vLLM installed -- which is the tt-metal unit-test job. Skip
# before the import rather than inside the tests: the failure is at import.
pytest.importorskip("vllm")
from models.demos.gemma4.tt.dflash_drafter import DFlashFusedDecoder
from models.demos.gemma4.tt.generator_vllm import dflash_pv_bucket_ladder


def _decoder(widths, captured=None, P_v=6):
    """A decoder carrying a captured width set and nothing else."""
    dec = DFlashFusedDecoder.__new__(DFlashFusedDecoder)
    dec.P_v = P_v
    dec.use_packed = True
    captured = set(widths if captured is None else captured)
    dec._pv_widths = {int(w): {"pv_sk": int(w), "trace": (object() if w in captured else None)} for w in widths}
    dec.pv_sk = min(int(w) for w in widths)
    return dec


def test_narrowest_covering_width_is_selected():
    dec = _decoder([2048, 4096, 8192])
    assert dec.width_for(0) == 2048
    assert dec.width_for(2048 - 6 - 64) == 2048
    assert dec.width_for(2048 - 6 - 64 + 1) == 4096
    assert dec.width_for(4096 - 6 - 64 + 1) == 8192


def test_uncaptured_widths_are_never_selected():
    """A width whose trace was not captured cannot serve a step: replaying
    another width's trace would read the wrong pv_iota length."""
    dec = _decoder([2048, 4096, 8192], captured=[2048, 8192])
    assert dec.width_for(2048) == 8192


def test_past_the_widest_width_is_none_not_a_wrong_width():
    dec = _decoder([2048, 4096])
    assert dec.width_for(4096) is None


def test_ladder_covers_every_position_up_to_max_model_len():
    """The migration guarantee: the largest rung covers max_model_len, so a
    request never runs out of widths before vLLM's own length limit."""
    max_model_len = 262144
    ladder = dflash_pv_bucket_ladder(max_model_len, horizon=2048, verify=5)
    dec = _decoder(ladder)
    for pos in (0, 1, 2047, 4096, 131072, max_model_len - 1, max_model_len):
        assert dec.width_for(pos) is not None, pos


@pytest.mark.parametrize("pos", [0, 1000, 5000, 100000])
def test_selection_is_monotonic_in_position(pos):
    """Width never shrinks as a request generates: a migration only ever moves
    to a wider trace, so the mask and page table only ever grow."""
    ladder = dflash_pv_bucket_ladder(262144, horizon=2048, verify=5)
    dec = _decoder(ladder)
    w = dec.width_for(pos)
    assert dec.width_for(pos + 1) >= w
    assert dec.width_for(pos + 1000) >= w


# ── the width set is bounded by the KV pool, not by max_model_len ────────────


def test_top_width_is_the_block_budget_not_the_rounded_context():
    """Reproduces the warmup kill: the ladder's top rung rounds UP past
    max_model_len (265216 for a 262144 server = 4144 blocks), and
    paged_update_cache requires max_num_blocks_per_seq < max_num_blocks, so
    warmup died with 'max_num_blocks_per_seq=4144, max_num_blocks=4128'."""
    from models.demos.gemma4.tt.generator_vllm import dflash_width_set

    widths = dflash_width_set(262144, num_blocks=4096, horizon=2048, verify=5)
    assert max(widths) == 262144
    assert max(widths) // 64 == 4096
    assert all(w // 64 <= 4096 for w in widths)


def test_width_set_drops_rungs_past_the_budget_and_keeps_the_budget():
    from models.demos.gemma4.tt.generator_vllm import dflash_width_set

    widths = dflash_width_set(262144, num_blocks=1024, horizon=2048, verify=5)
    assert max(widths) == 65536  # 1024 blocks x 64
    assert all(w <= 65536 for w in widths)
    assert widths == sorted(widths)


def test_width_set_without_a_block_budget_is_the_plain_ladder():
    from models.demos.gemma4.tt.generator_vllm import dflash_pv_bucket_ladder, dflash_width_set

    assert dflash_width_set(262144, num_blocks=0, horizon=2048, verify=5) == dflash_pv_bucket_ladder(
        262144, horizon=2048, verify=5
    )


def test_width_set_covers_every_position_the_pool_can_hold():
    from models.demos.gemma4.tt.generator_vllm import dflash_width_set

    widths = dflash_width_set(262144, num_blocks=4096, horizon=2048, verify=5)
    dec = _decoder(widths)
    # Everything up to the last verify block's worth of the context is covered;
    # past that vLLM's own max_model_len stop arrives first.
    for pos in (0, 4096, 131072, 262144 - 6 - 64):
        assert dec.width_for(pos) is not None, pos
