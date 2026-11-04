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
from models.ttt_compat.tt.generator import (
    Generator,
    batched_prefill_padded_batch,
    gather_batched_prefill_samples,
    prepare_prefill_page_tables_per_layer,
    validate_compact_prefill_model,
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
        [table, table, table, None], [0, 1, 2], rows, batch, [129, 65, 128], [64, 64, 128, 64], 1024
    )
    assert result[0] is result[1]
    assert result[2] is not result[0]
    assert result[3] is None
    assert result[0][rows, :3].tolist() == [[11, 12, 13], [21, 22, -1], [31, 32, -1]]
    assert result[2][rows, :2].tolist() == [[11, 12], [21, -1], [31, -1]]
    padding = [i for i in range(batch) if i not in rows]
    assert (result[0][padding] == -1).all()
    torch.testing.assert_close(table, saved)


def test_sequential_layer_pages_select_request_not_physical_slot():
    table = torch.tensor([[11, 12, 99], [21, 98, 97]], dtype=torch.int32)
    result = prepare_prefill_page_tables_per_layer([table], [1], [0], 1, [80, 32], [64], 128)
    assert result[0].tolist() == [[21, -1]]


@pytest.mark.parametrize("prompt_len,padded_len", [(40, 128), (65, 128), (200, 1024)])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("sample", [False, True], ids=["host_logits", "device_sampling_contract"])
@pytest.mark.parametrize("existing_decode_state", [False, True])
def test_prefill_routes_tokens_pages_and_output_by_the_same_rows(
    monkeypatch, compact, sample, existing_decode_state, prompt_len, padded_len
):
    from models.ttt_compat.tt import generator as generator_module

    generator = object.__new__(Generator)
    generator.data_parallel = 1
    if existing_decode_state:
        generator._slots_prefilled_since_decode = {7}
    generator.model_args = [
        SimpleNamespace(
            max_batch_size=32,
            max_prefill_chunk_size=32768,
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
    tokens = torch.tensor([[11] * prompt_len, [22] * prompt_len, [33] * prompt_len])
    owned_blocks = (prompt_len + 63) // 64
    owned_pages = [[base + j for j in range(owned_blocks)] for base in (10, 20, 30)]
    pages = torch.tensor([row + [99] for row in owned_pages], dtype=torch.int32)
    cache = [[(SimpleNamespace(shape=(100, 1, 64, 128)), None)]]
    output = generator._prefill_forward_text_impl(
        tokens,
        page_table=pages,
        kv_cache=cache,
        prompt_lens=[prompt_len] * 3,
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
        assert state["prompt_tokens"][[31, 4, 17], prompt_len - 1].tolist() == [11, 22, 33]
        # The sampler contract stores inverse temperatures.
        assert [state["sampling_params"].temperature[i] for i in [31, 4, 17]] == pytest.approx([10, 5, 10 / 3])
    else:
        assert output.flatten().tolist() == [11, 22, 33]
    packed, args = observed[0]
    rows = [0, 1, 2] if compact else [31, 4, 17]
    assert packed.shape == (4 if compact else 32, padded_len)
    assert packed[rows, prompt_len - 1].tolist() == [11, 22, 33]
    assert args["user_id"] == rows
    prepared = args["page_tables_per_layer"][0]
    assert prepared.shape[1] * 64 >= packed.shape[1]
    assert prepared[rows, :owned_blocks].tolist() == owned_pages
    assert (prepared[:, owned_blocks:] == -1).all()
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


@pytest.mark.parametrize("prompt_len,padded_len", [(40, 128), (200, 1024)])
@pytest.mark.parametrize("block_size", [64, 128])
def test_layer_page_width_covers_kernel_padding_without_exposing_stale_pages(prompt_len, padded_len, block_size):
    owned = (prompt_len + block_size - 1) // block_size
    table = torch.tensor([list(range(10, 10 + owned)) + [999]], dtype=torch.int32)
    result = prepare_prefill_page_tables_per_layer([table], [0], [3], 4, [prompt_len], [block_size], padded_len)[0]
    assert result.shape == (4, (padded_len + block_size - 1) // block_size)
    assert result[3, :owned].tolist() == list(range(10, 10 + owned))
    assert (result[3, owned:] == -1).all()
    assert (result[:3] == -1).all()
    assert 999 not in result


def test_compact_opt_in_rejects_row_sharded_cache_at_startup():
    with pytest.raises(ValueError, match="row-sharded"):  # allow-pytest.raises: host-only startup validation
        Generator([SimpleNamespace(users_row_sharded=True)], [SimpleNamespace(batched_prefill_compact_rows=True)], None)


@pytest.mark.parametrize("extract", [None, lambda hidden, **kwargs: hidden])
def test_compact_device_sampling_requires_explicit_slot_map_at_startup(extract):
    model = SimpleNamespace(_supports_on_device_sampling=True, extract_last_tokens_batched_prefill=extract)
    with pytest.raises(ValueError, match="explicit slot_map"):  # allow-pytest.raises: host-only startup validation
        Generator([model], [SimpleNamespace(batched_prefill_compact_rows=True)], None)


def test_compact_slot_map_capability_and_default_path():
    model = SimpleNamespace(
        _supports_on_device_sampling=True, extract_last_tokens_batched_prefill=lambda hidden, slot_map=None: hidden
    )
    validate_compact_prefill_model(model, SimpleNamespace(batched_prefill_compact_rows=True))
    validate_compact_prefill_model(SimpleNamespace(users_row_sharded=True), SimpleNamespace())


def test_layer_page_width_is_stable_inside_a_padded_trace_bucket():
    table = torch.arange(16, dtype=torch.int32).reshape(1, 16)
    prepared = [
        prepare_prefill_page_tables_per_layer([table], [0], [0], 1, [prompt], [64], 1024)[0]
        for prompt in (129, 200, 1023)
    ]
    assert [tuple(p.shape) for p in prepared] == [(1, 16)] * 3
    assert (prepared[0][0, 3:] == -1).all()
    assert (prepared[1][0, 4:] == -1).all()
    assert prepared[2].tolist() == table.tolist()


def test_explicit_layer_tables_cannot_silently_enter_trace_replay():
    generator = object.__new__(Generator)
    with pytest.raises(ValueError, match="require untraced prefill"):  # allow-pytest.raises: host-only contract refusal
        generator._easy_trace_prefill(torch.zeros(1, 128), page_tables_per_layer=[torch.tensor([[42]])])


def test_resumed_layer_tables_retain_absolute_cached_page_positions():
    table = torch.tensor([[10, 11, 12, 13, 99]], dtype=torch.int32)
    # 128 cached tokens plus a padded128-token suffix cover four absolute pages.
    result = prepare_prefill_page_tables_per_layer([table], [0], [0], 1, [200], [64], 128 + 128)[0]
    assert result.tolist() == [[10, 11, 12, 13]]
