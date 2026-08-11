# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched prefill lays its device rows out by physical slot, not by prefill position.

``empty_slots[i]`` is the device slot that owns request ``i``'s per-slot state, and
every slot-indexed buffer the batched path builds (``prefill_ids``,
``padded_last_token_idx``, the padded page table) is bounded by ``padded_batch``. vLLM
hands out the slot a request already owns, so a batch of N requests can land on slots
above N and the device batch has to span them. Pure index arithmetic, no device.
"""

from models.tt_transformers.tt.generator import batched_prefill_padded_batch


def test_dense_slots_keep_todays_batch_shape():
    """The common case is unchanged, so existing shapes and traces are reused."""
    assert batched_prefill_padded_batch(7, list(range(7)), 32) == 8
    assert batched_prefill_padded_batch(2, [0, 1], 32) == 2
    assert batched_prefill_padded_batch(32, list(range(32)), 32) == 32


def test_batch_spans_the_highest_slot_in_use():
    """A live off-batch request holding a low slot pushes a prefill onto a high one."""
    # THE BUG: seven requests whose slots reach 7. The count-based rule returned 8,
    # which is fine here, but a request on slot 20 got a 1-row batch.
    assert batched_prefill_padded_batch(7, [0, 1, 2, 3, 4, 5, 7], 32) == 8
    assert batched_prefill_padded_batch(1, [20], 32) == 32
    assert batched_prefill_padded_batch(3, [3, 4, 5], 32) == 8


def test_no_slots_means_the_request_count_is_the_span():
    """Callers that omit the slots get ``range(N)``, so N bounds the rows."""
    assert batched_prefill_padded_batch(4, None, 32) == 4
    assert batched_prefill_padded_batch(4, [], 32) == 4


def test_a_slot_past_capacity_reports_the_fallback_value():
    """Over capacity the caller disables batched prefill on the returned value."""
    assert batched_prefill_padded_batch(2, [40], 32) == 32
