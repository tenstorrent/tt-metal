# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

import models.common.llm_runtime.prefill.runtime as prefill_module
import models.common.llm_runtime.prefill.sampling_helpers as sampling_helpers
import models.common.llm_runtime.tensor_resources as tensor_resources_module
from models.common.llm_runtime.config import PageTableLayout
from models.common.llm_runtime.output_reader import OutputReader
from models.common.llm_runtime.prefill.config import PrefillRuntimeConfig
from models.common.llm_runtime.prefill.plan import _plan_prefill_requests
from models.common.llm_runtime.prefill.runtime import (
    InvocationResult,
    PrefillDeviceInputs,
    PrefillPersistentInputs,
    PrefillPositionInputs,
    PrefillRuntime,
    _process_output_tokens,
)
from models.common.sampling import SamplingParams


class FakeReader(OutputReader):
    def __init__(self, mesh_device):
        self.mesh_device = mesh_device

    def read(self, value, *, blocking):
        assert blocking
        return value

    def read_synchronized(self, value):
        return value


class FakeModel:
    vocab_size = 8

    def __init__(self, mesh_device, *, sampling_batch_size=32, allow_force_argmax=False):
        self.config = SimpleNamespace(dim=32, mesh_device=mesh_device)
        self.sampling = SimpleNamespace(
            config=SimpleNamespace(
                max_batch_size=sampling_batch_size,
                allow_force_argmax=allow_force_argmax,
            )
        )
        self.chunk_starts = []

    def embed_prefill(self, tokens):
        return tokens

    def prefill_forward(
        self,
        x_embed,
        rot_mats,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        batch_size=1,
        chunk_start_idx_tensor=None,
        last_token_slice=None,
        last_token_index=None,
    ):
        self.chunk_starts.append(chunk_start_idx)
        return SimpleNamespace(shape=(1, 1, 1, self.vocab_size))


def _runtime(
    *,
    trace_lengths=(128, 1024),
    allow_force_argmax=False,
    sampling_batch_size=32,
    device_sampling_enabled=True,
):
    mesh_device = SimpleNamespace(shape=(1, 1))
    config = PrefillRuntimeConfig.resolve(
        model=FakeModel(
            mesh_device,
            sampling_batch_size=sampling_batch_size,
            allow_force_argmax=allow_force_argmax,
        ),
        output_reader=FakeReader(mesh_device),
        page_table_layout=PageTableLayout(
            block_size=32,
            raw_capacity_width=256,
            prefill_width=264,
            decode_width=256,
        ),
        max_batch_size=32,
        max_prefill_chunk_size=2048,
        device_sampling_enabled=device_sampling_enabled,
        can_enable_trace=lambda length, cached: cached == 0 and length in trace_lengths,
    )
    return PrefillRuntime(config)


def _inputs(*, prompt_length, cached_tokens=0, token_width=None, page_width=256, rows=1):
    token_width = prompt_length if token_width is None else token_width
    tokens = torch.arange(rows * token_width, dtype=torch.long).reshape(rows, token_width)
    page_table = torch.arange(rows * page_width, dtype=torch.int32).reshape(rows, page_width)
    prompt_lens = torch.full((rows,), prompt_length, dtype=torch.long)
    start_pos = torch.full((rows,), cached_tokens, dtype=torch.long)
    return tokens, page_table, prompt_lens, start_pos


def _plan(*, prompt_length, cached_tokens=0, token_width=None, page_width=256, slots=(0,), maximum=2048):
    tokens, page_table, prompt_lens, start_pos = _inputs(
        prompt_length=prompt_length,
        cached_tokens=cached_tokens,
        token_width=token_width,
        page_width=page_width,
        rows=len(slots),
    )
    return _plan_prefill_requests(
        tokens=tokens,
        page_table=page_table,
        prompt_lens=prompt_lens,
        empty_slots=slots,
        start_pos=start_pos,
        block_size=32,
        max_batch_size=32,
        max_prefill_chunk_size=maximum,
        max_actual_page_table_width=256,
        canonical_page_table_width=264,
    )


def test_trace_applicability_classification_does_not_allocate_a_prefill_plan(monkeypatch):
    runtime = _runtime()
    tokens, _, prompt_lens, start_pos = _inputs(prompt_length=80, rows=2)

    def fail_if_planned(
        *,
        tokens,
        page_table,
        prompt_lens,
        empty_slots,
        start_pos,
        block_size,
        max_batch_size,
        max_prefill_chunk_size,
        max_actual_page_table_width=None,
        canonical_page_table_width=None,
    ):
        raise AssertionError("planned")

    monkeypatch.setattr(prefill_module, "_plan_prefill_requests", fail_if_planned)

    assert runtime.can_trace(tokens=tokens, prompt_lens=prompt_lens, start_pos=start_pos)
    assert not runtime.can_trace(
        tokens=tokens,
        prompt_lens=prompt_lens,
        start_pos=torch.tensor([0, 32]),
    )
    assert not runtime.can_trace(
        tokens=torch.zeros((1, 2049), dtype=torch.long),
        prompt_lens=torch.tensor([2049]),
    )


def test_sampling_values_keep_logical_prefill_user_contract_for_partial_batch():
    values = sampling_helpers._formatted_sampling_values(
        SamplingParams(temperature=1.0, top_k=32, top_p=0.08),
        1,
    )

    assert tuple(len(field) for field in values[:3]) == (1, 1, 1)
    assert values[0][0] == 32
    assert values[1][0] == pytest.approx(0.08)
    assert values[2][0] == 1.0


def test_sampling_values_accept_vector_tensor_fields_for_full_batch():
    values = sampling_helpers._formatted_sampling_values(
        SamplingParams(
            temperature=torch.zeros(32),
            top_k=torch.ones(32, dtype=torch.int32),
            top_p=torch.ones(32),
        ),
        32,
    )

    assert tuple(len(field) for field in values[:3]) == (32, 32, 32)
    assert values[0] == (1,) * 32
    assert values[1] == (0.0,) * 32
    assert values[2] == (1.0,) * 32


def test_single_greedy_prefill_uses_argmax_without_changing_batched_sampling():
    runtime = _runtime(allow_force_argmax=True)
    greedy = SamplingParams(temperature=0.0, top_k=32, top_p=0.08)
    single_inputs = _inputs(prompt_length=80)

    single = runtime.prepare(
        tokens=single_inputs[0],
        page_table=single_inputs[1],
        prompt_lens=single_inputs[2],
        sampling_params=greedy,
    )
    batched_inputs = _inputs(prompt_length=80, rows=2)
    batched = runtime.prepare(
        tokens=batched_inputs[0],
        page_table=batched_inputs[1],
        prompt_lens=batched_inputs[2],
        empty_slots=(0, 1),
        sampling_params=greedy,
    )

    assert single[0].sampling_path == "argmax"
    assert batched[0].sampling_path == "topk"
    assert single[0].program_signatures[0].sampling_path == "argmax"
    assert single[0].program_signatures[0].last_token_tile_start == 64


def test_trace_finish_reuses_trace_owned_sample_output(monkeypatch):
    runtime = _runtime()
    request = _plan(prompt_length=80)[0]
    prepared = SimpleNamespace(
        request=request,
        sampling_params=SamplingParams(temperature=1.0, top_k=32, top_p=0.08),
        sampling_path="topk",
    )
    persistent = PrefillPersistentInputs(
        device_inputs=object(),
        position_inputs=PrefillPositionInputs("start", "end", "row"),
        kpt=("k", "p", "temperature"),
        sampled_output="persistent-output",
    )
    seen = []

    def finish_regular_prefill(
        prepared,
        hidden,
        kpt,
        position_inputs,
        *,
        sampled_output=None,
        owned=None,
    ):
        seen.append(((prepared, hidden, kpt, position_inputs), {"sampled_output": sampled_output, "owned": owned}))
        return "tokens"

    monkeypatch.setattr(runtime, "_finish_regular_prefill", finish_regular_prefill)

    result = runtime.finish_trace(prepared, "hidden", persistent)

    assert result.value == "tokens"
    assert result.owned == ()
    assert seen[0][1]["sampled_output"] == "persistent-output"


def test_trace_refresh_skips_unchanged_position_and_sampling_inputs(monkeypatch):
    runtime = _runtime()
    request = _plan(prompt_length=80)[0]
    sampling = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    prepared = SimpleNamespace(request=request, sampling_params=sampling, sampling_path="topk")
    k, p, temperature, _ = sampling_helpers._formatted_sampling_values(
        sampling,
        runtime._sampling_output_rows(prepared),
    )
    persistent = PrefillPersistentInputs(
        device_inputs=PrefillDeviceInputs("tokens", "cos", "sin", "page", None, "positions", None),
        position_inputs=PrefillPositionInputs("start", "end", "row"),
        kpt=("k", "p", "temperature"),
        position_signature=[79],
        kpt_signature=[(k, p, temperature)],
    )
    monkeypatch.setattr(prefill_module.ttnn, "ReplicateTensorToMesh", lambda mesh: "mapper")
    # ttnn.from_torch is an overloaded backend API; this test only distinguishes source dtype.
    monkeypatch.setattr(
        prefill_module.ttnn,
        "from_torch",
        lambda value, **kwargs: "host-page" if value.dtype == torch.int32 else "host-tokens",
    )
    copied = []
    monkeypatch.setattr(
        prefill_module.ttnn,
        "copy_host_to_device_tensor",
        lambda host, device: copied.append((host, device)),
    )

    def fail_position_refresh(relative_last, sequence_length):
        pytest.fail("position refreshed")

    def fail_sampling_refresh(device_kpt, sampling_params, batch_size, force_topk):
        pytest.fail("sampling refreshed")

    monkeypatch.setattr(runtime, "_prepare_position_inputs_host", fail_position_refresh)
    monkeypatch.setattr(runtime, "_refresh_kpt", fail_sampling_refresh)

    runtime.refresh_trace(prepared, persistent)

    assert copied == [("host-tokens", "tokens"), ("host-page", "page")]


def test_eager_sampled_prefill_uses_preallocated_output(monkeypatch):
    runtime = _runtime()
    request = _plan(prompt_length=80)[0]
    prepared = SimpleNamespace(
        request=request,
        sampling_params=SamplingParams(temperature=1.0, top_k=32, top_p=0.08),
        sampling_path="topk",
    )
    events = []
    monkeypatch.setattr(
        runtime,
        "_stage_prefill_step",
        lambda prepared, chunk, final_relative_last: events.append("stage") or ("device-inputs", "positions"),
    )
    monkeypatch.setattr(
        runtime,
        "_make_device_kpt",
        lambda sampling_params, batch_size, force_topk: events.append("kpt") or "kpt",
    )
    monkeypatch.setattr(
        runtime,
        "_execute_prefill_step",
        lambda prepared, chunk, device_inputs, position_inputs: events.append("execute") or "hidden",
    )
    monkeypatch.setattr(
        runtime,
        "_make_sampling_output",
        lambda batch_size: events.append("sample-output") or "sample-output",
    )
    seen = []

    def finish_regular_prefill(
        prepared,
        hidden,
        kpt,
        position_inputs,
        *,
        sampled_output=None,
        owned=None,
    ):
        events.append("finish")
        seen.append(
            (
                (prepared, hidden, kpt, position_inputs),
                {"sampled_output": sampled_output, "owned": owned},
            )
        )
        return sampled_output

    monkeypatch.setattr(runtime, "_finish_regular_prefill", finish_regular_prefill)

    result = runtime._run_prefill_sequence(prepared)

    assert result.value == "sample-output"
    assert result.owned == ("device-inputs", "positions", "kpt", "hidden", "sample-output")
    assert seen[0][1]["sampled_output"] == "sample-output"
    assert seen[0][1]["owned"] is not None
    assert events == ["stage", "kpt", "execute", "sample-output", "finish"]


def test_sampling_output_is_allocated_before_capture_with_device_shape(monkeypatch):
    runtime = _runtime()
    seen = []
    # ttnn.from_torch is overloaded; allocation options are the behavior under test.
    monkeypatch.setattr(
        prefill_module.ttnn,
        "from_torch",
        lambda tensor, **kwargs: seen.append((tensor, kwargs)) or "device-output",
    )
    monkeypatch.setattr(prefill_module.ttnn, "ReplicateTensorToMesh", lambda mesh: "replicate")

    assert runtime._make_sampling_output(1) == "device-output"
    assert tuple(seen[0][0].shape) == (1, 1, 1, 1)
    assert seen[0][1]["device"] is runtime.config.mesh_device
    assert seen[0][1]["mesh_mapper"] == "replicate"


@pytest.mark.parametrize("shape", [(1, 1, 1, 32), (1, 1, 32, 1)])
def test_sampled_token_normalization_converts_only_first_replica(monkeypatch, shape):
    class DistributedTensor:
        pass

    first = torch.arange(32, dtype=torch.int32).reshape(shape)
    second = torch.full(shape, 99, dtype=torch.int32)
    converted = []
    monkeypatch.setattr(prefill_module.ttnn, "Tensor", DistributedTensor)
    monkeypatch.setattr(prefill_module.ttnn, "get_device_tensors", lambda value: [first, second])
    monkeypatch.setattr(
        prefill_module.ttnn,
        "to_torch",
        lambda value: converted.append(value) or value,
    )

    tokens = _process_output_tokens(DistributedTensor(), 32, (1, 2))

    assert tokens.tolist() == list(range(32))
    assert converted == [first]


@pytest.mark.parametrize("uncached_length", [1, 31, 32, 33, 127, 128, 129, 2048, 2049])
@pytest.mark.parametrize("cached_tokens", [0, 32, 64])
def test_planning_preserves_uncached_slice_and_absolute_chunk_positions(uncached_length, cached_tokens):
    prompt_length = cached_tokens + uncached_length
    request = _plan(prompt_length=prompt_length, cached_tokens=cached_tokens)[0]

    assert request.cached_tokens == (cached_tokens,)
    assert request.prompt_lengths == (prompt_length,)
    assert request.last_token_indices == (prompt_length - 1,)
    assert torch.equal(
        request.tokens[0, :uncached_length],
        torch.arange(cached_tokens, prompt_length, dtype=torch.long),
    )
    assert request.chunks[0].token_slice.start == 0
    assert request.chunks[0].chunk_start_idx == cached_tokens
    assert request.chunks[-1].contains_last_token
    assert request.chunks[-1].token_slice.start <= uncached_length - 1 < request.chunks[-1].token_slice.stop


def test_four_regular_cached_and_multi_chunk_cases_share_one_plan_shape():
    regular = _plan(prompt_length=128)[0]
    cached_one = _plan(prompt_length=160, cached_tokens=32)[0]
    uncached_multi = _plan(prompt_length=4096)[0]
    cached_multi = _plan(prompt_length=4129, cached_tokens=96)[0]

    assert not regular.uses_chunked_prefill and len(regular.chunks) == 1
    assert cached_one.uses_chunked_prefill and len(cached_one.chunks) == 1
    assert uncached_multi.uses_chunked_prefill and len(uncached_multi.chunks) == 2
    assert cached_multi.uses_chunked_prefill and len(cached_multi.chunks) == 2
    assert all(
        chunk.chunk_page_table is not None
        for request in (cached_one, uncached_multi, cached_multi)
        for chunk in request.chunks
    )


def test_chunk_mapping_uses_absolute_blocks_pads_sentinels_and_stops_after_last_token():
    request = _plan(prompt_length=4129, cached_tokens=96)[0]
    first, second = request.chunks

    assert (first.chunk_start_idx, second.chunk_start_idx) == (96, 2144)
    assert torch.equal(first.chunk_page_table[0], request.page_table[0, 3:67])
    assert torch.equal(second.chunk_page_table[0, :63], request.page_table[0, 67:130])
    assert second.chunk_page_table[0, 63].item() == -1

    early_stop = _plan(prompt_length=4097)[0]
    assert early_stop.padded_sequence_length == 8192
    assert len(early_stop.chunks) == 3
    assert early_stop.chunks[-1].token_slice == slice(4096, 6144)
    assert early_stop.chunks[-1].contains_last_token


def test_full_and_truncated_scheduler_tables_produce_equivalent_semantic_plans():
    prompt_length = 160
    cached_tokens = 32
    actual_width = 5
    full = _plan(prompt_length=prompt_length, cached_tokens=cached_tokens, page_width=256)[0]
    truncated = _plan(prompt_length=prompt_length, cached_tokens=cached_tokens, page_width=actual_width)[0]

    assert torch.equal(full.tokens, truncated.tokens)
    assert torch.equal(full.page_table, truncated.page_table)
    assert full.source_rows == truncated.source_rows
    assert full.slots == truncated.slots
    assert len(full.chunks) == len(truncated.chunks)
    assert torch.equal(full.chunks[0].chunk_page_table, truncated.chunks[0].chunk_page_table)


def test_q128_batching_and_noncontiguous_slot_fallback_preserve_source_order():
    batched = _plan(prompt_length=80, slots=(0, 1, 2))[0]
    assert batched.kind == "batched"
    assert batched.source_rows == (0, 1, 2)
    assert batched.slots == (0, 1, 2)
    assert batched.padded_batch_size == 4

    fallback = _plan(prompt_length=80, slots=(7, 3, 11))
    assert [request.kind for request in fallback] == ["single", "single", "single"]
    assert [request.source_rows for request in fallback] == [(0,), (1,), (2,)]
    assert [request.slots for request in fallback] == [(7,), (3,), (11,)]


def test_q128_batching_accepts_different_exact_prompt_lengths():
    tokens = torch.arange(3 * 128, dtype=torch.long).reshape(3, 128)
    page_table = torch.arange(3 * 256, dtype=torch.int32).reshape(3, 256)

    requests = _plan_prefill_requests(
        tokens=tokens,
        page_table=page_table,
        prompt_lens=torch.tensor([87, 115, 125]),
        empty_slots=[0, 1, 2],
        start_pos=torch.zeros(3, dtype=torch.long),
        block_size=32,
        max_batch_size=32,
        max_prefill_chunk_size=2048,
        max_actual_page_table_width=256,
        canonical_page_table_width=264,
    )

    assert len(requests) == 1
    assert requests[0].kind == "batched"
    assert requests[0].prompt_lengths == (87, 115, 125)
    assert requests[0].padded_sequence_length == 128
    assert requests[0].padded_batch_size == 4


def test_mixed_lengths_fall_back_per_row_without_reordering():
    tokens = torch.arange(3 * 160, dtype=torch.long).reshape(3, 160)
    page_table = torch.arange(3 * 256, dtype=torch.int32).reshape(3, 256)
    requests = _plan_prefill_requests(
        tokens=tokens,
        page_table=page_table,
        prompt_lens=torch.tensor([33, 128, 129]),
        empty_slots=[4, 1, 9],
        start_pos=torch.zeros(3, dtype=torch.long),
        block_size=32,
        max_batch_size=32,
        max_prefill_chunk_size=2048,
        max_actual_page_table_width=256,
        canonical_page_table_width=264,
    )
    assert [request.source_rows for request in requests] == [(0,), (1,), (2,)]
    assert [request.slots for request in requests] == [(4,), (1,), (9,)]
    assert [request.padded_sequence_length for request in requests] == [128, 128, 1024]


def test_program_signatures_are_material_and_trace_classification_is_separate_from_planning():
    runtime = _runtime()

    def prepare(prompt_length, cached_tokens=0, sampling_params=None):
        tokens, page_table, prompt_lens, start_pos = _inputs(
            prompt_length=prompt_length,
            cached_tokens=cached_tokens,
        )
        return runtime.prepare(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            sampling_params=sampling_params,
        )[0]

    logits = prepare(80)
    topk = prepare(80, sampling_params=SamplingParams(temperature=1.0, top_k=32, top_p=0.08))
    cached = prepare(112, cached_tokens=32)
    multi = prepare(4096)

    assert logits.program_signatures != topk.program_signatures
    assert dict(logits.program_signatures[0].key_material())["sampling_path"] == "logits"
    assert logits.trace_signature is not None
    assert cached.trace_signature is None
    assert multi.trace_signature is None
    assert logits.request.uses_chunked_prefill is False
    assert cached.request.uses_chunked_prefill is True


def test_q128_single_topk_tile_is_program_material_but_not_trace_material():
    runtime = _runtime()
    sampling = SamplingParams(temperature=1.0, top_k=32, top_p=0.08)

    def prepare(prompt_length, sampling_params):
        tokens, page_table, prompt_lens, start_pos = _inputs(prompt_length=prompt_length)
        return runtime.prepare(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            sampling_params=sampling_params,
        )[0]

    first_tile = prepare(32, sampling)
    third_tile = prepare(96, sampling)
    first_program = first_tile.program_signatures[0]
    third_program = third_tile.program_signatures[0]

    assert first_program.last_token_tile_start == 0
    assert third_program.last_token_tile_start == 64
    assert first_program != third_program
    assert first_tile.trace_signature == third_tile.trace_signature
    assert prepare(32, None).program_signatures[0].last_token_tile_start is None


def test_static_q128_single_topk_uses_tile_output_and_exact_host_row(monkeypatch):
    runtime = _runtime()
    tokens, page_table, prompt_lens, start_pos = _inputs(prompt_length=80)
    sampling = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    prepared = runtime.prepare(
        tokens=tokens,
        page_table=page_table,
        prompt_lens=prompt_lens,
        empty_slots=[0],
        start_pos=start_pos,
        sampling_params=sampling,
    )[0]
    seen = []

    def post_process_prefill_output(
        hidden_states,
        last_token_idx,
        last_token_slice=None,
        last_token_index=None,
    ):
        seen.append(((hidden_states, last_token_idx), {}))
        return "logits"

    runtime.config.model.post_process_prefill_output = post_process_prefill_output
    monkeypatch.setattr(prefill_module, "_pad_prefill_logits", lambda logits, sampler: logits)
    monkeypatch.setattr(runtime, "_sample_device", lambda logits, kpt, sampled_output=None: sampled_output)

    assert runtime._sampling_output_rows(prepared) == 32
    assert (
        runtime._finish_regular_prefill(
            prepared,
            "hidden",
            "kpt",
            PrefillPositionInputs("dynamic-start", "dynamic-end", "dynamic-row"),
            sampled_output="sampled",
        )
        == "sampled"
    )
    assert seen == [(("hidden", 79), {})]

    host_tokens = torch.zeros(1, 1, 32, 1, dtype=torch.int64)
    host_tokens[0, 0, 15, 0] = 123
    host_log_probs = torch.arange(32, dtype=torch.float32).reshape(1, 1, 1, 32)
    released = []
    monkeypatch.setattr(runtime, "_release_or_retain_transient", lambda value: released.append(value) or [])
    output, log_probs = runtime.assemble(
        [(prepared, InvocationResult((host_tokens, host_log_probs), "owned"))],
        batch_size=1,
        sampling_params=sampling,
    )

    assert output.tolist() == [123]
    assert log_probs.item() == 15.0
    assert released == ["owned"]


def test_static_q128_output_sizing_does_not_change_chunked_or_non_q128_paths():
    sampling = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)

    def prepare(runtime, prompt_length, cached_tokens=0):
        tokens, page_table, prompt_lens, start_pos = _inputs(
            prompt_length=prompt_length,
            cached_tokens=cached_tokens,
        )
        return runtime.prepare(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            empty_slots=[0],
            start_pos=start_pos,
            sampling_params=sampling,
        )[0]

    runtime = _runtime(sampling_batch_size=32)
    static = prepare(runtime, 80)
    assert runtime._sampling_output_rows(static) == 32

    runtime = _runtime(sampling_batch_size=16)
    static = prepare(runtime, 80)
    assert runtime._sampling_output_rows(static) == 16

    runtime = _runtime(sampling_batch_size=1)
    cached = prepare(runtime, 160, cached_tokens=32)
    non_q128 = prepare(runtime, 129)
    assert runtime._sampling_output_rows(cached) == 1
    assert runtime._sampling_output_rows(non_q128) == 1


@pytest.mark.parametrize(
    ("allow_force_argmax", "sampling_params", "expected_path"),
    [
        (True, SamplingParams(temperature=0.0, top_k=32, top_p=0.08), "argmax"),
        (False, SamplingParams(temperature=0.0, top_k=32, top_p=0.08), "topk"),
        (True, SamplingParams(temperature=1.0, top_k=32, top_p=0.08), "topk"),
    ],
)
def test_prepare_selects_single_prefill_sampling_path(allow_force_argmax, sampling_params, expected_path):
    runtime = _runtime(allow_force_argmax=allow_force_argmax)
    tokens, page_table, prompt_lens, start_pos = _inputs(prompt_length=80)

    prepared = runtime.prepare(
        tokens=tokens,
        page_table=page_table,
        prompt_lens=prompt_lens,
        empty_slots=[0],
        start_pos=start_pos,
        sampling_params=sampling_params,
    )[0]

    assert prepared.sampling_path == expected_path
    assert prepared.program_signatures[0].sampling_path == expected_path


def test_prepare_classifies_once_and_invoke_uses_only_sequence_executor(monkeypatch):
    runtime = _runtime()
    tokens, page_table, prompt_lens, start_pos = _inputs(prompt_length=160, cached_tokens=32)
    prepared = runtime.prepare(
        tokens=tokens,
        page_table=page_table,
        prompt_lens=prompt_lens,
        empty_slots=[0],
        start_pos=start_pos,
    )[0]
    seen = []
    expected = InvocationResult("value", "owned")

    def run_sequence(received):
        seen.append(received)
        return expected

    monkeypatch.setattr(runtime, "_run_prefill_sequence", run_sequence)
    assert runtime.invoke(prepared) is expected
    assert seen == [prepared]
    assert not hasattr(runtime, "_run_regular_prefill")
    assert not hasattr(runtime, "_run_chunked_prefill")


def test_cached_one_chunk_uses_chunk_model_contract(monkeypatch):
    runtime = _runtime()
    request = _plan(prompt_length=160, cached_tokens=32)[0]
    prepared = SimpleNamespace(request=request, sampling_params=None, sampling_path="logits")
    chunk = request.chunks[0]
    device_inputs = PrefillDeviceInputs("tokens", "cos", "sin", "page", "chunk-page", "pos", "chunk-start")
    position_inputs = PrefillPositionInputs("slice-start", "slice-end", "row")
    seen = []

    def prefill_forward(
        x_embed,
        rot_mats,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        batch_size=1,
        chunk_start_idx_tensor=None,
        last_token_slice=None,
        last_token_index=None,
    ):
        seen.append(
            (
                (x_embed, rot_mats),
                {
                    "user_id": user_id,
                    "page_table": page_table,
                    "chunk_page_table": chunk_page_table,
                    "chunk_start_idx": chunk_start_idx,
                    "get_last_token": get_last_token,
                    "chunk_start_idx_tensor": chunk_start_idx_tensor,
                    "last_token_slice": last_token_slice,
                    "last_token_index": last_token_index,
                },
            )
        )
        return "output"

    def fail_regular_model_body(request, device_inputs):
        pytest.fail("regular model body used")

    runtime.config.model.prefill_forward = prefill_forward
    monkeypatch.setattr(runtime, "_run_hidden_body", fail_regular_model_body)

    assert runtime._execute_prefill_step(prepared, chunk, device_inputs, position_inputs) == "output"
    assert seen == [
        (
            ("tokens", ["cos", "sin"]),
            {
                "user_id": 0,
                "page_table": "page",
                "chunk_page_table": "chunk-page",
                "chunk_start_idx": 32,
                "get_last_token": -1,
                "chunk_start_idx_tensor": "chunk-start",
                "last_token_slice": ("slice-start", "slice-end"),
                "last_token_index": None,
            },
        )
    ]


def test_regular_batched_step_and_finalization_preserve_exact_model_contract(monkeypatch):
    runtime = _runtime()
    request = _plan(prompt_length=80, slots=(0, 1, 2))[0]
    prepared = SimpleNamespace(request=request, sampling_params=None, sampling_path="logits")
    chunk = request.chunks[0]
    device_inputs = PrefillDeviceInputs("tokens", "cos", "sin", "page", None, "pos", None)
    position_inputs = PrefillPositionInputs("slice-start", "slice-end", "row")
    calls = []
    runtime.config.model.embed_prefill = lambda tokens: calls.append(("embed", tokens)) or "embedded"

    def prefill_forward(
        x_embed,
        rot_mats,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        batch_size=1,
        chunk_start_idx_tensor=None,
        last_token_slice=None,
        last_token_index=None,
    ):
        calls.append(
            (
                "forward",
                (x_embed, rot_mats),
                {
                    "user_id": user_id,
                    "page_table": page_table,
                    "chunk_page_table": chunk_page_table,
                    "get_last_token": get_last_token,
                    "batch_size": batch_size,
                    "chunk_start_idx_tensor": chunk_start_idx_tensor,
                },
            )
        )
        return "hidden"

    def post_process_batched_prefill_output(
        hidden_states,
        last_token_idx_list,
        padded_batch,
        prefill_seq_len,
        last_token_slice=None,
        last_token_index=None,
    ):
        calls.append(
            (
                "postprocess",
                (hidden_states, last_token_idx_list, padded_batch, prefill_seq_len),
                {
                    "last_token_slice": last_token_slice,
                    "last_token_index": last_token_index,
                },
            )
        )
        return "logits"

    runtime.config.model.prefill_forward = prefill_forward
    runtime.config.model.post_process_batched_prefill_output = post_process_batched_prefill_output
    # ttnn.untilize is overloaded; this test intentionally records backend options.
    monkeypatch.setattr(
        prefill_module.ttnn,
        "untilize",
        lambda value, **kwargs: calls.append(("untilize", value, kwargs)) or "output",
    )

    hidden = runtime._execute_prefill_step(prepared, chunk, device_inputs, position_inputs)
    output = runtime._finish_prefill_sequence(
        prepared,
        hidden,
        None,
        position_inputs,
        sampled_output=None,
        owned=[],
    )

    assert output == "output"
    assert calls == [
        ("embed", "tokens"),
        (
            "forward",
            ("embedded", ["cos", "sin"]),
            {
                "user_id": [0, 1, 2, 3],
                "page_table": "page",
                "chunk_page_table": None,
                "get_last_token": -1,
                "batch_size": 4,
                "chunk_start_idx_tensor": None,
            },
        ),
        (
            "postprocess",
            ("hidden", [79, 79, 79, 0], 4, 128),
            {
                "last_token_slice": ("slice-start", "slice-end"),
                "last_token_index": "row",
            },
        ),
        ("untilize", "logits", {"use_multicore": True}),
    ]


@pytest.mark.parametrize(
    ("sampling_path", "sampling_params", "expected_tail"),
    [
        ("logits", None, ["postprocess", "untilize"]),
        (
            "argmax",
            SamplingParams(temperature=0.0, top_k=1, top_p=1.0),
            ["postprocess", "pad", "sample"],
        ),
    ],
)
def test_regular_logits_and_argmax_preserve_operation_order(
    monkeypatch,
    sampling_path,
    sampling_params,
    expected_tail,
):
    runtime = _runtime()
    request = _plan(prompt_length=129)[0]
    prepared = SimpleNamespace(
        request=request,
        sampling_params=sampling_params,
        sampling_path=sampling_path,
    )
    events = []
    monkeypatch.setattr(
        runtime,
        "_stage_prefill_step",
        lambda prepared, chunk, final_relative_last: events.append("stage")
        or ("device", PrefillPositionInputs("start", "end", "row")),
    )
    monkeypatch.setattr(
        runtime,
        "_make_device_kpt",
        lambda sampling_params, batch_size, force_topk: events.append("kpt") or None,
    )
    monkeypatch.setattr(
        runtime,
        "_execute_prefill_step",
        lambda prepared, chunk, device_inputs, position_inputs: events.append("execute") or "hidden",
    )

    def post_process_prefill_output(
        hidden_states,
        last_token_idx,
        last_token_slice=None,
        last_token_index=None,
    ):
        events.append("postprocess")
        return "logits"

    runtime.config.model.post_process_prefill_output = post_process_prefill_output
    monkeypatch.setattr(
        prefill_module,
        "_pad_prefill_logits",
        lambda logits, sampler: events.append("pad") or "padded",
    )
    monkeypatch.setattr(
        runtime,
        "_sample_device",
        lambda logits, kpt, sampled_output=None: events.append("sample") or "sampled",
    )
    # ttnn.untilize is overloaded; this test only checks operation order.
    monkeypatch.setattr(
        prefill_module.ttnn,
        "untilize",
        lambda *args, **kwargs: events.append("untilize") or "untilized",
    )
    monkeypatch.setattr(runtime, "_make_sampling_output", lambda batch_size: pytest.fail("output preallocated"))

    runtime._run_prefill_sequence(prepared)

    assert events == ["stage", "kpt", "execute", *expected_tail]


def test_cached_one_chunk_stages_planned_chunk_metadata(monkeypatch):
    runtime = _runtime()
    request = _plan(prompt_length=160, cached_tokens=32)[0]
    prepared = SimpleNamespace(request=request)
    chunk = request.chunks[0]
    seen = []

    def prepare_inputs_host(
        tokens,
        page_table,
        *,
        start_pos=0,
        chunk_page_table=None,
        chunk_start_idx=None,
        last_token_idx=None,
    ):
        seen.append(
            (
                tokens,
                page_table,
                {
                    "start_pos": start_pos,
                    "chunk_page_table": chunk_page_table,
                    "chunk_start_idx": chunk_start_idx,
                    "last_token_idx": last_token_idx,
                },
            )
        )
        return "host"

    monkeypatch.setattr(runtime, "_prepare_inputs_host", prepare_inputs_host)
    monkeypatch.setattr(runtime, "_stage_device_inputs", lambda host_inputs: "device")
    monkeypatch.setattr(
        runtime,
        "_prepare_position_inputs_host",
        lambda relative_last, sequence_length: PrefillPositionInputs(relative_last, sequence_length, "row"),
    )
    monkeypatch.setattr(
        prefill_module,
        "_copy_host_to_device",
        lambda host_tensors, device_tensors=None, mesh_device=None: host_tensors,
    )

    assert runtime._stage_prefill_step(prepared, chunk, 127) == (
        "device",
        PrefillPositionInputs(127, 128, "row"),
    )
    assert torch.equal(seen[0][0], request.tokens[:, chunk.token_slice])
    assert seen[0][1] is request.page_table
    assert seen[0][2] == {
        "start_pos": 32,
        "chunk_page_table": chunk.chunk_page_table,
        "chunk_start_idx": 32,
        "last_token_idx": 159,
    }


def test_chunk_sequence_allocates_kpt_before_steps_and_reuses_final_position(monkeypatch):
    runtime = _runtime()
    request = _plan(prompt_length=4097)[0]
    prepared = SimpleNamespace(request=request, sampling_params=None, sampling_path="logits")
    events = []
    relative_positions = []
    step_outputs = iter(object() for _ in request.chunks)

    monkeypatch.setattr(
        runtime,
        "_make_device_kpt",
        lambda sampling_params, batch_size, force_topk: events.append("kpt") or None,
    )

    def stage(_prepared, chunk, relative_last):
        events.append(("stage", chunk.chunk_start_idx))
        relative_positions.append(relative_last)
        return object(), object()

    monkeypatch.setattr(runtime, "_stage_prefill_step", stage)
    monkeypatch.setattr(
        runtime,
        "_execute_prefill_step",
        lambda prepared, chunk, device_inputs, position_inputs: events.append(("execute", chunk.chunk_start_idx))
        or next(step_outputs),
    )
    monkeypatch.setattr(runtime, "_release_or_retain_transient", lambda value: events.append("release") or [])
    monkeypatch.setattr(
        runtime,
        "_finish_prefill_sequence",
        lambda prepared, final_step_output, kpt, position_inputs, *, sampled_output, owned: events.append("finish")
        or final_step_output,
    )

    runtime._run_prefill_sequence(prepared)

    assert relative_positions == [0] * len(request.chunks)
    assert events[0] == "kpt"
    assert events[1:] == [
        ("stage", 0),
        ("execute", 0),
        "release",
        ("stage", 2048),
        ("execute", 2048),
        "release",
        ("stage", 4096),
        ("execute", 4096),
        "finish",
    ]


@pytest.mark.parametrize(
    ("prompt_length", "cached_tokens", "sampling_path", "expect_preallocated"),
    [
        (80, 0, "topk", True),
        (80, 0, "argmax", False),
        (160, 32, "topk", False),
    ],
)
def test_sequence_preserves_sampling_output_preallocation_matrix(
    monkeypatch,
    prompt_length,
    cached_tokens,
    sampling_path,
    expect_preallocated,
):
    runtime = _runtime()
    request = _plan(prompt_length=prompt_length, cached_tokens=cached_tokens)[0]
    prepared = SimpleNamespace(
        request=request,
        sampling_params=SamplingParams(temperature=0.0, top_k=1, top_p=1.0),
        sampling_path=sampling_path,
    )
    allocated = []
    monkeypatch.setattr(runtime, "_make_device_kpt", lambda sampling_params, batch_size, force_topk: None)
    monkeypatch.setattr(
        runtime,
        "_stage_prefill_step",
        lambda prepared, chunk, final_relative_last: (object(), object()),
    )
    monkeypatch.setattr(
        runtime,
        "_execute_prefill_step",
        lambda prepared, chunk, device_inputs, position_inputs: object(),
    )
    monkeypatch.setattr(
        runtime,
        "_make_sampling_output",
        lambda rows: allocated.append(rows) or object(),
    )
    monkeypatch.setattr(
        runtime,
        "_finish_prefill_sequence",
        lambda prepared, final_step_output, kpt, position_inputs, *, sampled_output, owned: final_step_output,
    )

    runtime._run_prefill_sequence(prepared)

    assert bool(allocated) is expect_preallocated


def test_prefill_sequence_consumes_planned_chunks_and_releases_intermediate(monkeypatch):
    runtime = _runtime()
    request = _plan(prompt_length=4097)[0]
    prepared = SimpleNamespace(request=request, sampling_params=None, sampling_path="logits")
    device_inputs = [object() for _ in request.chunks]
    position_inputs = [object() for _ in request.chunks]
    step_outputs = [object() for _ in request.chunks]
    final_output = object()
    released = []
    staged = iter(zip(device_inputs, position_inputs))
    executed = iter(step_outputs)
    monkeypatch.setattr(runtime, "_make_device_kpt", lambda sampling_params, batch_size, force_topk: None)
    monkeypatch.setattr(
        runtime,
        "_stage_prefill_step",
        lambda prepared, chunk, final_relative_last: next(staged),
    )
    monkeypatch.setattr(
        runtime,
        "_execute_prefill_step",
        lambda prepared, chunk, device_inputs, position_inputs: next(executed),
    )
    # ttnn.untilize is overloaded; this test only verifies its input and output.
    monkeypatch.setattr(
        prefill_module.ttnn,
        "untilize",
        lambda value, **kwargs: final_output if value is step_outputs[-1] else pytest.fail("wrong final output"),
    )
    monkeypatch.setattr(runtime, "_release_or_retain_transient", lambda value: released.append(value) or [])

    result = runtime._run_prefill_sequence(prepared)

    assert released == step_outputs[:-1]
    assert result.value is final_output
    assert result.owned == (
        device_inputs[0],
        position_inputs[0],
        device_inputs[1],
        position_inputs[1],
        device_inputs[2],
        position_inputs[2],
        step_outputs[-1],
        final_output,
    )


def test_sequence_retains_raw_padded_and_sampled_outputs(monkeypatch):
    runtime = _runtime()
    regular_request = _plan(prompt_length=80)[0]
    regular = SimpleNamespace(
        request=regular_request,
        sampling_params=SamplingParams(temperature=1.0, top_k=32, top_p=0.08),
        sampling_path="topk",
    )
    raw_regular, padded_regular, sampled_regular = object(), object(), object()
    regular_owned = []

    def post_process_prefill_output(
        hidden_states,
        last_token_idx,
        last_token_slice=None,
        last_token_index=None,
    ):
        return raw_regular

    runtime.config.model.post_process_prefill_output = post_process_prefill_output
    monkeypatch.setattr(prefill_module, "_pad_prefill_logits", lambda logits, sampler: padded_regular)
    monkeypatch.setattr(
        runtime,
        "_sample_device",
        lambda logits, kpt, sampled_output=None: sampled_regular,
    )

    assert (
        runtime._finish_regular_prefill(
            regular,
            "hidden",
            "kpt",
            PrefillPositionInputs("start", "end", "row"),
            owned=regular_owned,
        )
        is sampled_regular
    )
    assert regular_owned == [raw_regular, padded_regular, sampled_regular]

    chunked_request = _plan(prompt_length=160, cached_tokens=32)[0]
    chunked = SimpleNamespace(
        request=chunked_request,
        sampling_params=regular.sampling_params,
        sampling_path="topk",
    )
    raw_chunked, padded_chunked, sampled_chunked = object(), object(), object()
    chunked_owned = [raw_chunked]
    monkeypatch.setattr(prefill_module, "_pad_prefill_logits", lambda logits, sampler: padded_chunked)
    monkeypatch.setattr(
        runtime,
        "_sample_device",
        lambda logits, kpt, sampled_output=None: sampled_chunked,
    )

    assert (
        runtime._finish_prefill_sequence(
            chunked,
            raw_chunked,
            "kpt",
            PrefillPositionInputs("start", "end", "row"),
            sampled_output=None,
            owned=chunked_owned,
        )
        is sampled_chunked
    )
    assert chunked_owned == [raw_chunked, padded_chunked, sampled_chunked]

    alias_owned = []
    runtime.config.model.post_process_prefill_output = post_process_prefill_output
    monkeypatch.setattr(prefill_module, "_pad_prefill_logits", lambda logits, sampler: raw_regular)
    monkeypatch.setattr(
        runtime,
        "_sample_device",
        lambda logits, kpt, sampled_output=None: raw_regular,
    )
    assert (
        runtime._finish_regular_prefill(
            regular,
            "hidden",
            "kpt",
            PrefillPositionInputs("start", "end", "row"),
            owned=alias_owned,
        )
        is raw_regular
    )
    assert alias_owned == [raw_regular]


def test_sampling_output_failure_releases_all_sequence_resources(monkeypatch, expect_error):
    runtime = _runtime()
    request = _plan(prompt_length=80)[0]
    prepared = SimpleNamespace(
        request=request,
        sampling_params=SamplingParams(temperature=1.0, top_k=32, top_p=0.08),
        sampling_path="topk",
    )
    device_inputs, position_inputs, kpt, hidden = object(), object(), object(), object()
    released = []
    monkeypatch.setattr(
        runtime,
        "_stage_prefill_step",
        lambda prepared, chunk, final_relative_last: (device_inputs, position_inputs),
    )
    monkeypatch.setattr(
        runtime,
        "_make_device_kpt",
        lambda sampling_params, batch_size, force_topk: kpt,
    )
    monkeypatch.setattr(
        runtime,
        "_execute_prefill_step",
        lambda prepared, chunk, device_inputs, position_inputs: hidden,
    )
    monkeypatch.setattr(
        runtime,
        "_make_sampling_output",
        lambda batch_size: (_ for _ in ()).throw(RuntimeError("allocation failed")),
    )
    monkeypatch.setattr(runtime, "_release_or_retain_transient", lambda values: released.append(values) or [])

    with expect_error(RuntimeError, "allocation failed"):
        runtime._run_prefill_sequence(prepared)

    assert released == [(device_inputs, position_inputs, kpt, hidden)]


def test_step_execution_failure_releases_staged_sequence_resources(monkeypatch, expect_error):
    runtime = _runtime()
    request = _plan(prompt_length=80)[0]
    prepared = SimpleNamespace(request=request, sampling_params=None, sampling_path="logits")
    device_inputs, position_inputs = object(), object()
    released = []
    monkeypatch.setattr(
        runtime,
        "_stage_prefill_step",
        lambda prepared, chunk, final_relative_last: (device_inputs, position_inputs),
    )
    monkeypatch.setattr(runtime, "_make_device_kpt", lambda sampling_params, batch_size, force_topk: None)

    def fail_step_execution(prepared, chunk, device_inputs, position_inputs):
        raise RuntimeError("execution failed")

    monkeypatch.setattr(runtime, "_execute_prefill_step", fail_step_execution)
    monkeypatch.setattr(runtime, "_release_or_retain_transient", lambda values: released.append(values) or [])

    with expect_error(RuntimeError, "execution failed"):
        runtime._run_prefill_sequence(prepared)

    assert released == [(device_inputs, position_inputs)]


def test_staging_failure_after_prior_chunk_releases_prior_sequence_resources(monkeypatch, expect_error):
    runtime = _runtime()
    request = _plan(prompt_length=4097)[0]
    prepared = SimpleNamespace(request=request, sampling_params=None, sampling_path="logits")
    device_inputs, position_inputs, intermediate = object(), object(), object()
    stage_calls = 0
    released = []
    monkeypatch.setattr(runtime, "_make_device_kpt", lambda sampling_params, batch_size, force_topk: None)

    def stage(prepared, chunk, final_relative_last):
        nonlocal stage_calls
        stage_calls += 1
        if stage_calls == 2:
            raise RuntimeError("second staging failed")
        return device_inputs, position_inputs

    monkeypatch.setattr(runtime, "_stage_prefill_step", stage)
    monkeypatch.setattr(
        runtime,
        "_execute_prefill_step",
        lambda prepared, chunk, device_inputs, position_inputs: intermediate,
    )
    monkeypatch.setattr(runtime, "_release_or_retain_transient", lambda values: released.append(values) or [])

    with expect_error(RuntimeError, "second staging failed"):
        runtime._run_prefill_sequence(prepared)

    assert released == [intermediate, (device_inputs, position_inputs)]


@pytest.mark.parametrize("failure_point", ["postprocess", "pad", "sample", "untilize"])
def test_finalization_failure_releases_every_resource_acquired_before_failure(monkeypatch, failure_point, expect_error):
    runtime = _runtime()
    request = _plan(prompt_length=129)[0]
    sampled = failure_point in ("pad", "sample")
    prepared = SimpleNamespace(
        request=request,
        sampling_params=(SamplingParams(temperature=0.0, top_k=1, top_p=1.0) if sampled else None),
        sampling_path="argmax" if sampled else "logits",
    )
    device_inputs, hidden, raw, padded = (object() for _ in range(4))
    position_inputs = PrefillPositionInputs("start", "end", "row")
    released = []
    monkeypatch.setattr(
        runtime,
        "_stage_prefill_step",
        lambda prepared, chunk, final_relative_last: (device_inputs, position_inputs),
    )
    monkeypatch.setattr(runtime, "_make_device_kpt", lambda sampling_params, batch_size, force_topk: None)
    monkeypatch.setattr(
        runtime,
        "_execute_prefill_step",
        lambda prepared, chunk, device_inputs, position_inputs: hidden,
    )

    def postprocess(
        hidden_states,
        last_token_idx,
        last_token_slice=None,
        last_token_index=None,
    ):
        if failure_point == "postprocess":
            raise RuntimeError("postprocess failed")
        return raw

    def pad(logits, sampler):
        if failure_point == "pad":
            raise RuntimeError("pad failed")
        return padded

    def sample(logits, kpt, sampled_output=None):
        if failure_point == "sample":
            raise RuntimeError("sample failed")
        return object()

    # ttnn.untilize is overloaded; the failure hook must accept its backend options.
    def untilize(*args, **kwargs):
        if failure_point == "untilize":
            raise RuntimeError("untilize failed")
        return object()

    runtime.config.model.post_process_prefill_output = postprocess
    monkeypatch.setattr(prefill_module, "_pad_prefill_logits", pad)
    monkeypatch.setattr(runtime, "_sample_device", sample)
    monkeypatch.setattr(prefill_module.ttnn, "untilize", untilize)
    monkeypatch.setattr(runtime, "_release_or_retain_transient", lambda values: released.append(values) or [])

    with expect_error(RuntimeError, f"{failure_point} failed"):
        runtime._run_prefill_sequence(prepared)

    expected = [device_inputs, position_inputs, hidden]
    if failure_point != "postprocess":
        expected.append(raw)
    if failure_point == "sample":
        expected.append(padded)
    assert released == [tuple(expected)]


def test_unified_path_preserves_primary_error_and_retries_cleanup_orphan(monkeypatch, expect_error):
    runtime = _runtime()
    request = _plan(prompt_length=80)[0]
    prepared = SimpleNamespace(request=request, sampling_params=None, sampling_path="logits")
    primary = RuntimeError("execution failed")
    cleanup_failure = RuntimeError("device busy")
    calls = []
    monkeypatch.setattr(
        runtime,
        "_stage_prefill_step",
        lambda prepared, chunk, final_relative_last: ("device", "position"),
    )
    monkeypatch.setattr(runtime, "_make_device_kpt", lambda sampling_params, batch_size, force_topk: None)

    def fail_step_execution(prepared, chunk, device_inputs, position_inputs):
        raise primary

    monkeypatch.setattr(runtime, "_execute_prefill_step", fail_step_execution)

    def release(values, completed):
        calls.append(values)
        return [cleanup_failure] if len(calls) == 1 else []

    monkeypatch.setattr(prefill_module, "best_effort_deallocate_owned_tensors", release)
    monkeypatch.setattr(tensor_resources_module, "best_effort_deallocate_owned_tensors", release)

    with expect_error(RuntimeError, "execution failed") as caught:
        runtime._run_prefill_sequence(prepared)

    assert caught.value is primary
    assert primary.cleanup_failures == (cleanup_failure,)
    assert runtime.transient_orphan_count == 1
    runtime.cleanup()
    assert calls == [("device", "position"), ("device", "position")]
    assert runtime.transient_orphan_count == 0


def test_intermediate_release_failure_is_not_reowned_by_sequence(monkeypatch, expect_error):
    runtime = _runtime()
    request = _plan(prompt_length=4097)[0]
    prepared = SimpleNamespace(request=request, sampling_params=None, sampling_path="logits")
    device_inputs, position_inputs, intermediate = object(), object(), object()
    released = []
    monkeypatch.setattr(runtime, "_make_device_kpt", lambda sampling_params, batch_size, force_topk: None)
    monkeypatch.setattr(
        runtime,
        "_stage_prefill_step",
        lambda prepared, chunk, final_relative_last: (device_inputs, position_inputs),
    )
    monkeypatch.setattr(
        runtime,
        "_execute_prefill_step",
        lambda prepared, chunk, device_inputs, position_inputs: intermediate,
    )

    def release(values):
        released.append(values)
        return [RuntimeError("release failed")] if values is intermediate else []

    monkeypatch.setattr(runtime, "_release_or_retain_transient", release)

    with expect_error(RuntimeError, "release failed"):
        runtime._run_prefill_sequence(prepared)

    assert released == [intermediate, (device_inputs, position_inputs)]


def test_trace_capture_uses_hidden_body_without_eager_sequence(monkeypatch):
    runtime = _runtime()
    tokens, page_table, prompt_lens, start_pos = _inputs(prompt_length=80)
    prepared = runtime.prepare(
        tokens=tokens,
        page_table=page_table,
        prompt_lens=prompt_lens,
        empty_slots=[0],
        start_pos=start_pos,
    )[0]
    persistent = PrefillPersistentInputs(
        device_inputs="device-inputs",
        position_inputs=PrefillPositionInputs("start", "end", "row"),
        kpt=None,
    )
    seen = []
    monkeypatch.setattr(
        runtime,
        "_run_hidden_body",
        lambda request, device_inputs: seen.append((request, device_inputs)) or "hidden",
    )
    monkeypatch.setattr(runtime, "_run_prefill_sequence", lambda prepared: pytest.fail("eager sequence used"))

    assert runtime.capture_plan(prepared).capture(persistent) == "hidden"
    assert seen == [(prepared.request, "device-inputs")]


def test_assemble_restores_source_rows_and_releases_each_owned_result(monkeypatch):
    runtime = _runtime()
    requests = _plan(prompt_length=80, slots=(7, 3))
    prepared = [SimpleNamespace(request=request, sampling_params=None) for request in requests]
    first = torch.zeros(1, 1, 32, runtime.config.model.vocab_size)
    second = torch.zeros_like(first)
    first[0, 0, 15, :] = 1
    second[0, 0, 15, :] = 2
    released = []
    monkeypatch.setattr(runtime, "_release_or_retain_transient", lambda value: released.append(value) or [])

    output = runtime.assemble(
        [(prepared[0], InvocationResult(first, "owned-0")), (prepared[1], InvocationResult(second, "owned-1"))],
        batch_size=2,
    )

    assert output.shape == (2, 1, runtime.config.model.vocab_size)
    assert torch.equal(output[0, 0], torch.ones(runtime.config.model.vocab_size))
    assert torch.equal(output[1, 0], torch.full((runtime.config.model.vocab_size,), 2.0))
    assert released == ["owned-0", "owned-1"]


def test_transient_cleanup_retries_failed_release(monkeypatch):
    runtime = _runtime()
    calls = []

    def release(value, completed):
        calls.append(value)
        return [RuntimeError("busy")] if len(calls) == 1 else []

    monkeypatch.setattr(prefill_module, "best_effort_deallocate_owned_tensors", release)
    monkeypatch.setattr(tensor_resources_module, "best_effort_deallocate_owned_tensors", release)
    failures = runtime._release_or_retain_transient("tensor")
    assert failures and runtime.transient_orphan_count == 1

    runtime.cleanup()
    assert calls == ["tensor", "tensor"]
    assert runtime.transient_orphan_count == 0


def test_zero_uncached_tokens_preserve_current_empty_plan_and_output_contract():
    runtime = _runtime()
    tokens, page_table, prompt_lens, start_pos = _inputs(prompt_length=32, cached_tokens=32)
    assert (
        runtime.prepare(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            empty_slots=[0],
            start_pos=start_pos,
        )
        == ()
    )

    logits = runtime.assemble([], batch_size=1)
    assert logits.shape == (1, 1, runtime.config.model.vocab_size)
    sampled, log_probs = runtime.assemble(
        [],
        batch_size=1,
        sampling_params=SamplingParams(temperature=0.0, top_k=1, top_p=1.0),
    )
    assert sampled.dtype == torch.int64
    assert sampled.tolist() == [0]
    assert log_probs is None


def test_prefill_runtime_config_resolves_frozen_static_capabilities(expect_error):
    runtime = _runtime(allow_force_argmax=True, sampling_batch_size=16)
    config = runtime.config

    assert config.cluster_shape == (1, 1)
    assert config.allow_force_argmax
    assert config.sampling_batch_size == 16
    assert not config.static_q128_topk_supported
    with expect_error(FrozenInstanceError, "cannot assign to field"):
        config.max_batch_size = 8


def test_prefill_runtime_config_rejects_mesh_and_sampler_mismatches(expect_error):
    mesh_device = SimpleNamespace(shape=(1, 1))
    other_mesh = SimpleNamespace(shape=(1, 1))
    model = FakeModel(mesh_device)
    reader = FakeReader(mesh_device)
    layout = PageTableLayout(32, 256, 264, 256)
    arguments = dict(
        model=model,
        output_reader=reader,
        page_table_layout=layout,
        max_batch_size=32,
        max_prefill_chunk_size=2048,
        device_sampling_enabled=True,
        can_enable_trace=lambda length, cached: True,
    )

    with expect_error(ValueError, "model and prefill runtime"):
        PrefillRuntimeConfig.resolve(**(arguments | {"output_reader": FakeReader(other_mesh)}))
    model.sampling = None
    with expect_error(TypeError, "model.sampling.config"):
        PrefillRuntimeConfig.resolve(**arguments)


def test_page_table_layout_replacement_is_immutable_and_bounded(expect_error):
    runtime = _runtime()
    original = runtime.config
    smaller = PageTableLayout(32, 128, 136, 128)

    runtime.configure_page_table_layout(smaller)

    assert runtime.config is not original
    assert runtime.config.page_table_layout is smaller
    assert original.page_table_layout.raw_capacity_width == 256
    with expect_error(ValueError, "block_size"):
        runtime.configure_page_table_layout(PageTableLayout(16, 128, 136, 128))
    with expect_error(ValueError, "capacity ceiling"):
        runtime.configure_page_table_layout(PageTableLayout(32, 512, 520, 512))
    with expect_error(ValueError, "canonical geometry"):
        runtime.configure_page_table_layout(PageTableLayout(32, 128, 520, 128))


def test_disabled_device_sampling_rejects_sampling_at_runtime_boundary(expect_error):
    runtime = _runtime(device_sampling_enabled=False)
    tokens, page_table, prompt_lens, start_pos = _inputs(prompt_length=80)

    with expect_error(ValueError, "device sampling is disabled"):
        runtime.prepare(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            empty_slots=[0],
            start_pos=start_pos,
            sampling_params=SamplingParams(temperature=0.0, top_k=1, top_p=1.0),
        )


def test_prefill_runtime_request_signatures_are_exact():
    empty = inspect.Parameter.empty
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    keyword_only = inspect.Parameter.KEYWORD_ONLY
    expected = {
        PrefillRuntime.can_trace: (
            (
                ("self", positional, empty, empty),
                ("tokens", keyword_only, empty, "torch.Tensor"),
                ("prompt_lens", keyword_only, None, "torch.Tensor | None"),
                ("start_pos", keyword_only, None, "torch.Tensor | None"),
            ),
            "bool",
        ),
        PrefillRuntime.prepare: (
            (
                ("self", positional, empty, empty),
                ("tokens", keyword_only, empty, "torch.Tensor"),
                ("page_table", keyword_only, empty, "torch.Tensor"),
                ("prompt_lens", keyword_only, None, "torch.Tensor | None"),
                ("start_pos", keyword_only, None, "torch.Tensor | None"),
                ("empty_slots", keyword_only, None, "Sequence[int] | None"),
                ("sampling_params", keyword_only, None, "SamplingParams | None"),
            ),
            "tuple[PreparedPrefill, ...]",
        ),
    }

    for method, (expected_parameters, expected_return) in expected.items():
        signature = inspect.signature(method)
        parameters = tuple(
            (parameter.name, parameter.kind, parameter.default, parameter.annotation)
            for parameter in signature.parameters.values()
        )
        assert parameters == expected_parameters
        assert signature.return_annotation == expected_return


def test_prefill_runtime_is_plain_orchestration_with_one_config_surface():
    source = inspect.getsource(prefill_module)
    assert hasattr(prefill_module, "PrefillRuntimeConfig")
    assert not hasattr(PrefillRuntime, "from_config")
    assert "LightweightModule" not in source
    assert PrefillRuntime.__bases__ == (object,)
    assert tuple(inspect.signature(PrefillRuntime).parameters) == ("config",)


def test_prefill_package_has_no_compatibility_barrel():
    prefill_package = importlib.import_module("models.common.llm_runtime.prefill")
    assert not hasattr(prefill_package, "PrefillRuntime")
    assert not hasattr(prefill_package, "PrefillRequest")
    assert not hasattr(prefill_package, "__all__")
