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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from models.common.sampling import SamplingParams, slice_sampling_params
from models.common.sampling.tt_log_probs import LogProbsResult
from models.tt_transformers.tt.generator import (
    Generator,
    batched_prefill_padded_batch,
    gather_batched_prefill_samples,
    prepare_prefill_page_tables_per_layer,
)


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


@pytest.mark.parametrize("rows,batch", [([0, 1, 2], 4), ([7, 0, 3], 8)])
def test_layer_pages_follow_prefill_rows_and_mask_unowned_tail(rows, batch):
    # The fourth column is another request's stale scheduler mapping. Different
    # layer groups use different page sizes, even when sharing the host table.
    table = torch.tensor([[11, 12, 13, 99], [21, 22, 98, 97], [31, 32, 33, 96]], dtype=torch.int32)
    saved = table.clone()
    result = prepare_prefill_page_tables_per_layer(
        [table, table, table, None], [0, 1, 2], rows, batch, [129, 65, 128], [64, 64, 128, 64]
    )
    assert result[0] is result[1]
    assert result[2] is not result[0]
    assert result[3] is None
    assert result[0][rows].tolist() == [[11, 12, 13], [21, 22, -1], [31, 32, -1]]
    assert result[2][rows].tolist() == [[11, 12], [21, -1], [31, -1]]
    padding = [i for i in range(batch) if i not in rows]
    assert (result[0][padding] == -1).all()
    torch.testing.assert_close(table, saved)


def test_sequential_layer_pages_select_request_not_physical_slot():
    table = torch.tensor([[11, 12, 99], [21, 98, 97]], dtype=torch.int32)
    result = prepare_prefill_page_tables_per_layer([table], [1], [0], 1, [80, 32], [64])
    assert result[0].tolist() == [[21]]


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("sample", [False, True], ids=["host_logits", "device_sampling_contract"])
@pytest.mark.parametrize("existing_decode_state", [False, True])
def test_prefill_routes_tokens_pages_and_output_by_the_same_rows(monkeypatch, compact, sample, existing_decode_state):
    from models.tt_transformers.tt import generator as generator_module

    generator = object.__new__(Generator)
    generator.data_parallel = 1
    if existing_decode_state:
        generator._slots_prefilled_since_decode = {7}
    generator.model_args = [
        SimpleNamespace(
            max_batch_size=32,
            max_prefill_chunk_size=8192,
            vocab_size=1,
            batched_prefill_compact_rows=compact,
            can_enable_trace=lambda *args: False,
        )
    ]
    model = SimpleNamespace(
        process_logits_after_prefill_trace=lambda hidden, last: hidden[:, :, last : last + 1],
        process_output_prefill=lambda logits, **kwargs: logits.reshape(1, 1),
        mesh_device=object(),
    )
    if sample:
        sampler = SimpleNamespace(
            tt_sampling=SimpleNamespace(
                max_batch_size=32,
                log_probs_calculator=SimpleNamespace(enable_log_probs=False),
                force_argmax_sampling=True,
            ),
            _penalties_active=False,
            apply_prefill_state=Mock(),
            sample=lambda logits, **kwargs: (logits.long() + 100, None),
        )

        def extract(hidden, last, padded_batch, seq_len, target_batch, slot_map=None):
            assert slot_map == ([31, 4, 17] if compact else None)
            output = torch.zeros(target_batch, 1, dtype=torch.long)
            for row, slot in enumerate(slot_map if slot_map is not None else range(padded_batch)):
                output[slot] = hidden[row, 0, last[row], 0]
            return output

        model.sampling = sampler
        model._supports_on_device_sampling = True
        model.extract_last_tokens_batched_prefill = extract
        model._apply_norm_and_lm_head = lambda hidden: hidden
        monkeypatch.setattr(generator_module.ttnn, "synchronize_device", lambda *args: None)
        monkeypatch.setattr(generator_module.ttnn, "get_device_tensors", lambda tensor: [tensor])
        monkeypatch.setattr(generator_module.ttnn, "to_torch", lambda tensor: tensor)
    generator.model = [model]
    generator._will_row_shard_prefill = lambda *args: False
    observed = []

    def forward(tokens, **kwargs):
        observed.append((tokens.clone(), kwargs))
        return tokens.float().unsqueeze(-1)

    generator.prefill_forward_single_user_text = forward
    monkeypatch.setattr(generator_module.ttnn, "reshape", torch.reshape)
    monkeypatch.setattr(generator_module.ttnn, "to_layout", lambda tensor, *args, **kwargs: tensor)
    tokens = torch.tensor([[11] * 65, [22] * 65, [33] * 65])
    pages = torch.tensor([[10, 11, 99], [20, 21, 98], [30, 31, 97]], dtype=torch.int32)
    cache = [[(SimpleNamespace(shape=(100, 1, 64, 128)), None)]]
    output = generator._prefill_forward_text_impl(
        tokens,
        page_table=pages,
        kv_cache=cache,
        prompt_lens=[65, 65, 65],
        empty_slots=[31, 4, 17],
        enable_trace=False,
        warmup_prefill=False,
        sampling_params=SamplingParams(temperature=[0.1, 0.2, 0.3], top_k=[1, 2, 3], top_p=[0.5, 0.6, 0.7])
        if sample
        else None,
        page_tables_per_layer=[pages],
    )
    if sample:
        assert output[0].flatten().tolist() == [111, 122, 133]
        state = sampler.apply_prefill_state.call_args.kwargs
        assert state["prompt_tokens"][[31, 4, 17], 64].tolist() == [11, 22, 33]
        # The sampler contract stores inverse temperatures.
        assert [state["sampling_params"].temperature[i] for i in [31, 4, 17]] == pytest.approx([10, 5, 10 / 3])
    else:
        assert output.flatten().tolist() == [11, 22, 33]
    packed, args = observed[0]
    rows = [0, 1, 2] if compact else [31, 4, 17]
    assert packed.shape == (4 if compact else 32, 128)
    assert packed[rows, 64].tolist() == [11, 22, 33]
    assert args["user_id"] == rows
    assert args["page_tables_per_layer"][0][rows].tolist() == [[10, 11], [20, 21], [30, 31]]
    # The version-1 decode contract takes explicit reload commands. Prefill
    # must neither infer a later reload nor overwrite existing decode state.
    if existing_decode_state:
        assert generator._slots_prefilled_since_decode == {7}
    else:
        assert not hasattr(generator, "_slots_prefilled_since_decode")


def test_single_user_forward_passes_explicit_layer_tables():
    generator = object.__new__(Generator)
    generator.data_parallel = 1
    generator.model_args = [SimpleNamespace(max_prefill_chunk_size=8192)]
    model = SimpleNamespace(
        prepare_inputs_prefill=Mock(return_value=(object(), object(), object(), object())),
        ttnn_prefill_forward=Mock(return_value=object()),
    )
    generator.model = [model]
    tables = [torch.tensor([[42]], dtype=torch.int32)]
    generator.prefill_forward_single_user_text(
        torch.zeros(1, 128), None, 0, 64, model_id=0, page_tables_per_layer=tables
    )
    assert model.ttnn_prefill_forward.call_args.kwargs["page_tables_per_layer"] is tables
