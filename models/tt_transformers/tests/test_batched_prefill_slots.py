# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched prefill lays its device rows out by physical slot, not by prefill position.

``empty_slots[i]`` is the device slot that owns request ``i``'s per-slot state, and
every slot-indexed buffer the batched path builds (``prefill_ids``,
``padded_last_token_idx``, the padded page table) is bounded by ``padded_batch``. vLLM
hands out the slot a request already owns, so a batch of N requests can land on slots
above N and the device batch has to span them. The arrays handed back to the caller
stay in prefill order, so the readback reads by slot and writes by position. Pure host
index bookkeeping, no device execution.
"""

import torch

from models.common.sampling import SamplingParams, slice_sampling_params
from models.common.sampling.tt_log_probs import LogProbsResult
from models.tt_transformers.tt.generator import batched_prefill_padded_batch, gather_batched_prefill_samples


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


class _SlotTaggedLogProbs(LogProbsResult):
    """Stands in for a device top-k result: reports which slot it was read from."""

    def __init__(self):
        super().__init__(topk_logprobs=None, topk_indices=None, topk_logprobs_host=None, topk_indices_host=None)

    def extract_user(self, user_batch_idx: int):
        return f"slot{int(user_batch_idx)}"


def test_samples_come_back_in_prefill_order_not_slot_order():
    """Read the device row by slot, write the caller's row by position.

    Row i of the device batch holds slot i's sample, so a request on slot 5 has to
    end up at output row 0 if it prefilled first.
    """
    slots = [5, 0, 3]
    # Device rows: index == slot, so slot 5 sampled token 105, slot 0 token 100, ...
    tokens_host = torch.tensor([100, 101, 102, 103, 104, 105, 106, 107])
    plain_log_probs_host = torch.tensor([-0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7])
    output_tokens = torch.zeros(len(slots), 1, dtype=torch.int64)
    output_log_probs = [None] * len(slots)

    gather_batched_prefill_samples(slots, tokens_host, None, plain_log_probs_host, output_tokens, output_log_probs)

    assert [int(t) for t in output_tokens.reshape(-1)] == [105, 100, 103]
    assert [round(float(lp), 1) for lp in output_log_probs] == [-0.5, -0.0, -0.3]


def test_a_slot_at_the_request_count_does_not_overflow_the_output():
    """THE CRASH: three requests reaching slot 7 wrote past a 3-row output."""
    slots = [0, 1, 7]
    tokens_host = torch.tensor([200, 201, 202, 203, 204, 205, 206, 207])
    output_tokens = torch.zeros(len(slots), 1, dtype=torch.int64)
    output_log_probs = [None] * len(slots)

    gather_batched_prefill_samples(slots, tokens_host, None, None, output_tokens, output_log_probs)

    assert [int(t) for t in output_tokens.reshape(-1)] == [200, 201, 207]
    assert output_log_probs == [None, None, None]


def test_topk_logprobs_are_extracted_from_the_slot_row():
    slots = [4, 1]
    tokens_host = torch.arange(8)
    output_tokens = torch.zeros(len(slots), 1, dtype=torch.int64)
    output_log_probs = [None] * len(slots)

    gather_batched_prefill_samples(slots, tokens_host, _SlotTaggedLogProbs(), None, output_tokens, output_log_probs)

    assert output_log_probs == ["slot4", "slot1"]


def test_slice_sampling_params_gives_each_chunk_its_own_requests():
    """A chunked prefill must not hand every chunk the first N requests' params."""
    params = SamplingParams(
        temperature=[0.1, 0.2, 0.3, 0.4], top_k=[1, 2, 3, 4], top_p=[0.5, 0.6, 0.7, 0.8], seed=[11, 12, 13, 14]
    )

    second = slice_sampling_params(params, 2, 4)

    assert second.temperature == [0.3, 0.4]
    assert second.top_k == [3, 4]
    assert second.top_p == [0.7, 0.8]
    assert second.seed == [13, 14]
    assert params.temperature == [0.1, 0.2, 0.3, 0.4]
    assert slice_sampling_params(None, 0, 2) is None


def test_a_span_no_bucket_covers_reports_the_span():
    """The caller's ``> max_batch_size`` guard has to fire and pick sequential prefill.

    Reporting ``max_batch_size`` instead would leave batched prefill enabled and
    scatter into a row the buffers do not have.
    """
    assert batched_prefill_padded_batch(2, [40], 32) == 41
    assert batched_prefill_padded_batch(2, [40], 32) > 32
    # A wider model still covers the slot, so batching stays on as it did before.
    assert batched_prefill_padded_batch(2, [40], 64) == 64
