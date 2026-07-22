# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

import models.common.llm_runtime.prefill as prefill_module
import models.common.llm_runtime.tensor_resources as tensor_resources_module
from models.common.llm_runtime.prefill import (
    InvocationResult,
    PrefillDeviceInputs,
    PrefillPersistentInputs,
    PrefillPositionInputs,
    PrefillRuntime,
    _plan_prefill_requests,
    _process_output_tokens,
)
from models.common.sampling import SamplingParams


class FakeReader:
    def read(self, value, *, blocking):
        assert blocking
        return value

    def read_synchronized(self, value):
        return value


class FakeModel:
    vocab_size = 8

    def __init__(self):
        self.config = SimpleNamespace(dim=32)
        self.sampling = SimpleNamespace(config=SimpleNamespace(max_batch_size=32, allow_force_argmax=False))
        self.chunk_starts = []

    def embed_prefill(self, tokens):
        return tokens

    def prefill_forward(self, hidden, rot_mats, **kwargs):
        self.chunk_starts.append(kwargs["chunk_start_idx"])
        return SimpleNamespace(shape=(1, 1, 1, self.vocab_size))


def _runtime(*, trace_lengths=(128, 1024)):
    layout = SimpleNamespace(block_size=32, raw_capacity_width=256, prefill_width=264, decode_width=256)
    return PrefillRuntime(
        model=FakeModel(),
        mesh_device="mesh",
        output_reader=FakeReader(),
        page_table_layout=layout,
        max_batch_size=32,
        max_prefill_chunk_size=2048,
        cluster_shape=(1, 1),
        device_sampling_enabled=True,
        can_enable_trace=lambda length, cached: cached == 0 and length in trace_lengths,
    )


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


def test_sampling_values_keep_logical_prefill_user_contract_for_partial_batch():
    values = prefill_module._formatted_sampling_values(
        SamplingParams(temperature=1.0, top_k=32, top_p=0.08),
        1,
    )

    assert tuple(len(field) for field in values[:3]) == (1, 1, 1)
    assert values[0][0] == 32
    assert values[1][0] == pytest.approx(0.08)
    assert values[2][0] == 1.0


def test_sampling_values_accept_vector_tensor_fields_for_full_batch():
    values = prefill_module._formatted_sampling_values(
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
    runtime = _runtime()
    runtime.model.sampling.config.allow_force_argmax = True
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
    monkeypatch.setattr(
        runtime,
        "_finish_regular_prefill",
        lambda *args, **kwargs: seen.append((args, kwargs)) or "tokens",
    )

    result = runtime.finish_trace(prepared, "hidden", persistent)

    assert result.value == "tokens"
    assert result.owned == ()
    assert seen[0][1]["sampled_output"] == "persistent-output"


def test_trace_refresh_skips_unchanged_position_and_sampling_inputs(monkeypatch):
    runtime = _runtime()
    request = _plan(prompt_length=80)[0]
    sampling = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    prepared = SimpleNamespace(request=request, sampling_params=sampling, sampling_path="topk")
    sampling_batch = runtime._sampling_batch_size(request)
    persistent = PrefillPersistentInputs(
        device_inputs=PrefillDeviceInputs("tokens", "cos", "sin", "page", None, "positions", None),
        position_inputs=PrefillPositionInputs("start", "end", "row"),
        kpt=("k", "p", "temperature"),
        position_signature=[79],
        kpt_signature=[runtime._kpt_signature(sampling, sampling_batch)],
    )
    monkeypatch.setattr(prefill_module.ttnn, "ReplicateTensorToMesh", lambda mesh: "mapper")
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
    monkeypatch.setattr(runtime, "_prepare_position_inputs_host", lambda *args: pytest.fail("position refreshed"))
    monkeypatch.setattr(runtime, "_refresh_kpt", lambda *args, **kwargs: pytest.fail("sampling refreshed"))

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
    monkeypatch.setattr(runtime, "_prepare_inputs_host", lambda *args, **kwargs: "host")
    monkeypatch.setattr(
        runtime,
        "_stage_inputs_and_kpt",
        lambda *args, **kwargs: ("device-inputs", "positions", "kpt"),
    )
    monkeypatch.setattr(runtime, "_run_hidden_body", lambda *args: "hidden")
    monkeypatch.setattr(runtime, "_make_sampling_output", lambda batch_size: "sample-output")
    seen = []
    monkeypatch.setattr(
        runtime,
        "_finish_regular_prefill",
        lambda *args, **kwargs: seen.append((args, kwargs)) or kwargs["sampled_output"],
    )

    result = runtime._run_regular_prefill(prepared)

    assert result.value == "sample-output"
    assert result.owned == ("sample-output", "hidden", "device-inputs", "positions", "kpt")
    assert seen[0][1]["sampled_output"] == "sample-output"


def test_sampling_output_is_allocated_before_capture_with_device_shape(monkeypatch):
    runtime = _runtime()
    seen = []
    monkeypatch.setattr(
        prefill_module.ttnn,
        "from_torch",
        lambda tensor, **kwargs: seen.append((tensor, kwargs)) or "device-output",
    )
    monkeypatch.setattr(prefill_module.ttnn, "ReplicateTensorToMesh", lambda mesh: "replicate")

    assert runtime._make_sampling_output(1) == "device-output"
    assert tuple(seen[0][0].shape) == (1, 1, 1, 1)
    assert seen[0][1]["device"] == "mesh"
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
    regular = _plan(prompt_length=80)[0]
    cached = _plan(prompt_length=112, cached_tokens=32)[0]
    multi = _plan(prompt_length=4096)[0]

    logits = runtime.program_signatures(regular, "logits")
    topk = runtime.program_signatures(regular, "topk")
    assert logits != topk
    assert dict(logits[0].key_material())["sampling_path"] == "logits"
    assert runtime.trace_signature(regular) is not None
    assert runtime.trace_signature(cached) is None
    assert runtime.trace_signature(multi) is None
    assert regular.uses_chunked_prefill is False
    assert cached.uses_chunked_prefill is True


def test_q128_single_topk_tile_is_program_material_but_not_trace_material():
    runtime = _runtime()
    first_tile = _plan(prompt_length=32)[0]
    third_tile = _plan(prompt_length=96)[0]

    first_program = runtime.program_signatures(first_tile, "topk")[0]
    third_program = runtime.program_signatures(third_tile, "topk")[0]

    assert first_program.last_token_tile_start == 0
    assert third_program.last_token_tile_start == 64
    assert first_program != third_program
    assert runtime.trace_signature(first_tile) == runtime.trace_signature(third_tile)
    assert runtime.program_signatures(first_tile, "logits")[0].last_token_tile_start is None


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
    runtime.model.post_process_prefill_output = lambda *args, **kwargs: seen.append((args, kwargs)) or "logits"
    monkeypatch.setattr(prefill_module, "_pad_prefill_logits", lambda logits, sampler: logits)
    monkeypatch.setattr(runtime, "_sample_device", lambda logits, kpt, output: output)

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
    runtime = _runtime()
    sampling = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)

    def prepare(prompt_length, cached_tokens=0):
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

    static = prepare(80)
    runtime.model.sampling.config.max_batch_size = 32
    assert runtime._sampling_output_rows(static) == 32
    assert runtime._sampling_parameter_batch_size(static) == 32

    runtime.model.sampling.config.max_batch_size = 16
    assert runtime._sampling_output_rows(static) == 16
    assert runtime._sampling_parameter_batch_size(static) == 16

    runtime.model.sampling.config.max_batch_size = 1
    cached = prepare(160, cached_tokens=32)
    non_q128 = prepare(129)
    assert runtime._sampling_output_rows(cached) == 1
    assert runtime._sampling_parameter_batch_size(cached) == 1
    assert runtime._sampling_output_rows(non_q128) == 1
    assert runtime._sampling_parameter_batch_size(non_q128) == 1


@pytest.mark.parametrize(
    ("allow_force_argmax", "sampling_params", "expected_path"),
    [
        (True, SamplingParams(temperature=0.0, top_k=32, top_p=0.08), "argmax"),
        (False, SamplingParams(temperature=0.0, top_k=32, top_p=0.08), "topk"),
        (True, SamplingParams(temperature=1.0, top_k=32, top_p=0.08), "topk"),
    ],
)
def test_prepare_selects_single_prefill_sampling_path(allow_force_argmax, sampling_params, expected_path):
    runtime = _runtime()
    runtime.model.sampling.config.allow_force_argmax = allow_force_argmax
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


def test_prepare_classifies_once_and_invoke_dispatches_the_same_request(monkeypatch):
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

    def run_chunked(received):
        seen.append(received)
        return expected

    monkeypatch.setattr(runtime, "_run_chunked_prefill", run_chunked)
    assert runtime.invoke(prepared) is expected
    assert seen == [prepared]


def test_private_chunked_path_consumes_planned_chunks_and_releases_intermediate(monkeypatch):
    runtime = _runtime()
    request = _plan(prompt_length=4097)[0]
    prepared = SimpleNamespace(request=request, sampling_params=None, sampling_path="logits")
    released = []

    monkeypatch.setattr(runtime, "_prepare_inputs_host", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runtime,
        "_stage_device_inputs",
        lambda host: PrefillDeviceInputs("tokens", "cos", "sin", "page", "chunk-page", "pos", "chunk-start"),
    )
    monkeypatch.setattr(runtime, "_prepare_position_inputs_host", lambda *args: PrefillPositionInputs(1, 2, 3))
    monkeypatch.setattr(prefill_module, "_copy_host_to_device", lambda values, **kwargs: list(values))
    monkeypatch.setattr(prefill_module.ttnn, "untilize", lambda value, **kwargs: value)
    monkeypatch.setattr(runtime, "_release_or_retain_transient", lambda value: released.append(value) or [])

    result = runtime._run_chunked_prefill(prepared)

    assert runtime.model.chunk_starts == [chunk.chunk_start_idx for chunk in request.chunks]
    assert len(released) == len(request.chunks) - 1
    assert result.value is not None
    assert result.owned[0] is result.value


def test_assemble_restores_source_rows_and_releases_each_owned_result(monkeypatch):
    runtime = _runtime()
    requests = _plan(prompt_length=80, slots=(7, 3))
    prepared = [SimpleNamespace(request=request, sampling_params=None) for request in requests]
    first = torch.zeros(1, 1, 32, runtime.model.vocab_size)
    second = torch.zeros_like(first)
    first[0, 0, 15, :] = 1
    second[0, 0, 15, :] = 2
    released = []
    monkeypatch.setattr(runtime, "_release_or_retain_transient", lambda value: released.append(value) or [])

    output = runtime.assemble(
        [(prepared[0], InvocationResult(first, "owned-0")), (prepared[1], InvocationResult(second, "owned-1"))],
        batch_size=2,
    )

    assert output.shape == (2, 1, runtime.model.vocab_size)
    assert torch.equal(output[0, 0], torch.ones(runtime.model.vocab_size))
    assert torch.equal(output[1, 0], torch.full((runtime.model.vocab_size,), 2.0))
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
    assert logits.shape == (1, 1, runtime.model.vocab_size)
    sampled, log_probs = runtime.assemble(
        [],
        batch_size=1,
        sampling_params=SamplingParams(temperature=0.0, top_k=1, top_p=1.0),
    )
    assert sampled.dtype == torch.int64
    assert sampled.tolist() == [0]
    assert log_probs is None


def test_prefill_runtime_is_plain_orchestration_without_duplicate_config_surface():
    source = inspect.getsource(prefill_module)
    assert not hasattr(prefill_module, "PrefillRuntimeConfig")
    assert not hasattr(PrefillRuntime, "from_config")
    assert "LightweightModule" not in source
    assert PrefillRuntime.__bases__ == (object,)
