# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A prefill chunk may only continue the prefix its own slot already holds.

Attention KV is paged, so a chunk starting at an arbitrary position can attend to pages another
request filled. GDN state is not paged: `recurrent` and `conv` are a sequential summary of the
tokens that slot has seen, carried implicitly from chunk to chunk. A chunk whose start does not
equal the slot's held prefix length would compute 48 of the 64 layers from unrelated state and
return a plausible wrong answer with no error.

Nothing enforced that before these tests. `start == 0` reset the slot and anything else was
accepted on the assumption that it was a later chunk of the same request in the same slot.
"""

from collections import Counter

import pytest

from models.demos.qwen38_27b_t3k.tt.generator import validate_prefix_continuity


def test_a_fresh_slot_accepts_a_chunk_that_starts_at_zero():
    validate_prefix_continuity([0], [2], [0, 0, 0, 0])


def test_a_started_over_slot_accepts_zero_whatever_it_held():
    # Zero begins a new request; the caller zeroes the recurrent state for exactly these rows.
    validate_prefix_continuity([0], [1], [0, 4096, 0, 0])


def test_a_slot_accepts_the_chunk_that_continues_it():
    validate_prefix_continuity([64], [1], [0, 64, 0, 0])


def test_several_rows_each_continue_their_own_slot():
    validate_prefix_continuity([0, 4096, 64], [3, 0, 1], [4096, 64, 0, 0])


@pytest.mark.parametrize("start", [1, 32, 64, 4096])
def test_a_never_advanced_slot_refuses_a_nonzero_start(expect_error, start):
    # The prefix-cache case: vLLM reports N tokens already computed, but the GDN state for them
    # exists nowhere, and the prefix tokens are not resent so it cannot be recomputed either.
    with expect_error(ValueError, "recurrent state"):
        validate_prefix_continuity([start], [0], [0, 0, 0, 0])


def test_a_chunk_that_skips_ahead_of_the_held_prefix_is_refused(expect_error):
    with expect_error(ValueError, "recurrent state"):
        validate_prefix_continuity([128], [1], [0, 64, 0, 0])


def test_a_chunk_that_restarts_behind_the_held_prefix_is_refused(expect_error):
    with expect_error(ValueError, "recurrent state"):
        validate_prefix_continuity([32], [1], [0, 64, 0, 0])


def test_the_offending_slot_and_both_lengths_are_named(expect_error):
    # The message has to carry enough to tell a prefix-cache hit from a chunking bug.
    with expect_error(ValueError, "slot 2"):
        validate_prefix_continuity([0, 512], [1, 2], [0, 0, 128, 0])


def test_one_bad_row_rejects_the_whole_batch(expect_error):
    # Rows are validated before any device effect, so a mixed batch must not half-apply.
    with expect_error(ValueError, "recurrent state"):
        validate_prefix_continuity([0, 99], [0, 1], [0, 64, 0, 0])


def test_reset_clears_the_held_lengths_it_just_zeroed_the_state_for():
    """reset_cache zeroes conv and recurrent, so every held length now describes nothing.

    Leaving one behind lets a continuation chunk pass validate_prefix_continuity against a slot
    whose state is zero, and makes save_slot_state store zeros under a nonzero length.
    """
    from types import MethodType, SimpleNamespace

    from models.demos.qwen38_27b_t3k.tests.unit.test_serving_prefill_trace import load_methods

    gen = SimpleNamespace(
        model=SimpleNamespace(_resident_decode_bucket=None, reset_cache=lambda cache: None),
        cache=SimpleNamespace(batch_size=4),
        positions=None,
        rope_indices=None,
        tokens="tokens",
        sampler=SimpleNamespace(seeds_tt_tensor="seeds"),
        seed=0,
        counters=Counter(),
        remaining_steps=7,
        reset_active_slots=False,
        _copy=lambda source, target, name: None,
        _reset_history=lambda: None,
        _slot_prefix_len=[0, 64, 4096, 0],
    )
    methods = load_methods("generator.py", "Qwen38Generator", ["reset"], SimpleNamespace())
    MethodType(methods["reset"], gen)()
    assert gen._slot_prefix_len == [0, 0, 0, 0]
