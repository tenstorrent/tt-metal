# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import inspect
from types import SimpleNamespace

import pytest
import torch

import models.common.llm_runtime.executor as executor_module
import ttnn
from models.common.llm_runtime.config import LLMExecutorConfig, PagedKVCacheConfig, TraceConfig, WarmupConfig
from models.common.llm_runtime.executor import (
    LLMExecutor,
    _copy_host_to_device,
    _deallocate_owned_ttnn,
    _ExternalDecodeOutput,
    _plan_prefill_requests,
    _PrefillRequest,
    _slice_sampling_params,
)
from models.common.llm_runtime.graph_compiler import PersistentInputs
from models.common.sampling import SamplingParams


class FakeMesh:
    shape = (1, 1)

    def get_num_devices(self):
        return 1

    def num_program_cache_entries(self):
        return 0


class FakeSampling:
    def __init__(self, *, allow_force_argmax=True, max_batch_size=4):
        self.config = SimpleNamespace(
            allow_force_argmax=allow_force_argmax,
            max_batch_size=max_batch_size,
            is_resolved=lambda: True,
        )
        self.calls = []
        self.loaded = 0
        self.released = 0

    def load_device_buffers(self):
        self.loaded += 1

    def release(self):
        self.released += 1

    def decode_forward(self, logits, **kwargs):
        self.calls.append((logits, kwargs))
        return logits, None


class FakeModel:
    def __init__(self, sampling=None, *, max_batch_size=4):
        mesh = FakeMesh()
        attention = SimpleNamespace(
            n_kv_heads=1,
            head_dim=8,
            kv_cache_dtype=ttnn.bfloat8_b,
            paged_attention_config=SimpleNamespace(block_size=32, max_num_blocks=128),
        )
        self.config = SimpleNamespace(
            mesh_device=mesh,
            num_devices=1,
            n_layers=1,
            dim=32,
            max_seq_len=4096,
            max_batch_size=max_batch_size,
            block_configs=[SimpleNamespace(attention_config=attention)],
        )
        self.mesh_device = mesh
        self.num_devices = 1
        self.vocab_size = 16
        self.sampling = sampling
        self.bound_cache = None
        self.prefill_position_indices = []

    def iter_executor_named_modules(self):
        return iter(())

    def set_kv_cache(self, cache):
        self.bound_cache = cache

    def prepare_prefill_rot_mats(self, position_indices):
        self.prefill_position_indices.append(position_indices)
        return "cos", "sin"


class FakeRuntimeConfig:
    model_cache_path = "cache"
    max_prefill_chunk_size = 2048
    trace_prefill_supported_seq_lens = (128, 1024)

    @staticmethod
    def can_enable_trace(sequence_length, cached_tokens=0):
        return cached_tokens == 0 and sequence_length in (128, 1024)


def make_config(*, sampling=False, trace="none", warmup=None):
    return LLMExecutorConfig(
        trace=TraceConfig(trace),
        warmup=warmup or WarmupConfig(),
        paged_kv_cache=PagedKVCacheConfig(
            block_size=32,
            max_num_blocks=128,
            dtype=ttnn.bfloat8_b,
        ),
        device_sampling_enabled=sampling,
    )


def make_executor(monkeypatch, *, sampling=False, trace="none", warmup=None):
    monkeypatch.setattr(executor_module, "Sampling1D", FakeSampling)
    sampler = FakeSampling() if sampling else None
    model = FakeModel(sampler)
    return LLMExecutor(model, FakeRuntimeConfig(), make_config(sampling=sampling, trace=trace, warmup=warmup))


def test_constructor_requires_model_validation_hook_and_configured_sampling(monkeypatch, expect_error):
    monkeypatch.setattr(executor_module, "Sampling1D", FakeSampling)
    model = FakeModel()
    model.iter_executor_named_modules = None
    with expect_error(TypeError, "iter_executor_named_modules"):
        LLMExecutor(model, FakeRuntimeConfig(), make_config())

    with expect_error(TypeError, "Sampling1D"):
        LLMExecutor(FakeModel(), FakeRuntimeConfig(), make_config(sampling=True))

    executor = LLMExecutor(FakeModel(FakeSampling()), FakeRuntimeConfig(), make_config(sampling=True))
    assert executor.model_args is executor.runtime_config
    assert executor.cache_path == "cache"
    assert executor.requires_prefill_trace_warmup


def test_sampling_disabled_rejects_request_before_device_execution(monkeypatch, expect_error):
    executor = make_executor(monkeypatch, sampling=False)
    with expect_error(ValueError, "device sampling is disabled"):
        executor._validate_sampling_request(SamplingParams(temperature=0.0, top_k=1, top_p=1.0))


def test_greedy_recipe_canonicalizes_to_argmax_and_true_topk_is_distinct(monkeypatch):
    executor = make_executor(monkeypatch, sampling=True)
    greedy_recipe = SamplingParams(temperature=0.0, top_k=32, top_p=0.08)
    true_topk = SamplingParams(temperature=1.0, top_k=32, top_p=0.08)

    assert executor._decode_sampling_path(greedy_recipe, 4) == "argmax"
    assert executor._decode_sampling_path(true_topk, 4) == "topk"

    executor.model.sampling.config.allow_force_argmax = False
    assert executor._decode_sampling_path(greedy_recipe, 4) == "topk"


def test_compile_decode_warms_the_trace_device_feedback_path(monkeypatch):
    executor = make_executor(monkeypatch, sampling=True)
    executor.model.increment_positions = lambda *args: None
    sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    calls = []

    def run_decode(*args, **kwargs):
        calls.append(kwargs)
        return "output", "owned"

    executor._run_decode_eager = run_decode

    result = executor._compile_decode_invocation(
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.zeros((4, 1), dtype=torch.int32),
        sampling_params,
        "argmax",
    )

    assert calls == [{"device_feedback": True}]
    assert result == {"output": "output", "owned": "owned"}


def test_public_compile_and_forward_decode_delegate_and_preserve_logits_tuple(monkeypatch):
    executor = make_executor(monkeypatch, trace="all")
    executor._validate_bound_cache = lambda cache: None
    executor._validation_context = lambda mode: contextlib.nullcontext()
    tokens = torch.arange(4, dtype=torch.long)
    start_pos = torch.arange(4, dtype=torch.long)
    page_table = torch.arange(8, dtype=torch.int32).view(4, 2)
    compile_calls = []

    class Compiler:
        def compile(self, key, invoke, *, output_spec):
            compile_calls.append((key, invoke("bound-context"), output_spec))

    executor._graph_compiler = Compiler()
    executor._compile_decode_invocation = lambda *args: {
        "output": (args, "bound-context"),
        "owned": None,
    }

    assert executor.compile_decode(tokens=tokens, start_pos=start_pos, page_table=page_table) is None
    key, compiled, output_spec = compile_calls[0]
    assert (key.mode, key.batch_size, key.page_table_width, key.sampling_path) == ("decode", 4, 128, "logits")
    compile_args, compile_context = compiled["output"]
    assert compile_args[0] is tokens
    assert compile_args[1] is start_pos
    normalized_page_table = compile_args[2]
    assert tuple(normalized_page_table.shape) == (4, 128)
    assert torch.equal(normalized_page_table[:, :1], page_table[:, :1])
    assert torch.count_nonzero(normalized_page_table[:, 1:]) == 0
    assert compile_args[3:] == (None, "logits")
    assert compile_context == "bound-context"
    assert callable(output_spec)

    execute_calls = []
    executor._execute_decode_request = lambda *args, **kwargs: execute_calls.append((args, kwargs)) or ("raw", None)
    log_probs = torch.tensor([0.25])
    host_logits = torch.arange(4 * 16, dtype=torch.float32).view(1, 1, 4, 16)
    reads = []
    executor._output_reader = SimpleNamespace(
        read=lambda value, *, blocking: reads.append((value, blocking)) or (host_logits, log_probs)
    )

    output, returned_log_probs = executor.decode_forward(
        tokens,
        start_pos,
        page_table,
        reset_batch=True,
    )

    execute_args, execute_kwargs = execute_calls[0]
    assert execute_args[0] is tokens
    assert execute_args[1] is start_pos
    assert torch.equal(execute_args[2], normalized_page_table)
    assert execute_args[2] is not page_table
    assert execute_args[3:5] == (None, "logits")
    assert execute_kwargs == {"enable_trace": True, "reset_batch": True}
    assert reads == [("raw", True)]
    assert output.shape == (4, 1, 16)
    assert torch.equal(output, host_logits.view(4, 1, 16))
    assert returned_log_probs is log_probs
    assert executor.mode.value == "decode"
    assert torch.equal(executor._previous_decode_page_table, normalized_page_table)
    assert executor._previous_decode_page_table is not execute_args[2]


def test_public_compile_and_forward_enable_feedback_page_table_lookahead(monkeypatch):
    executor = make_executor(monkeypatch, sampling=True, trace="all")
    executor.model.increment_positions = lambda *args: None
    executor._validate_bound_cache = lambda cache: None
    executor._validation_context = lambda mode: contextlib.nullcontext()
    executor._graph_compiler = SimpleNamespace(trace_active=False, compile=lambda *args, **kwargs: None)
    executor._make_decode_trace_plan = lambda *args: None
    executor._execute_decode_request = lambda *args, **kwargs: ("raw", None)
    sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    tokens = torch.zeros(4, dtype=torch.long)
    start_pos = torch.tensor([63, -1, -1, -1], dtype=torch.long)
    page_table = torch.zeros((4, 3), dtype=torch.int32)
    page_table[0] = torch.tensor([7, 8, 9], dtype=torch.int32)
    normalize = executor._normalize_decode_page_table
    feedback_flags = []

    def normalize_spy(*args, **kwargs):
        feedback_flags.append(kwargs.get("allow_one_step_feedback_lag"))
        return normalize(*args, **kwargs)

    executor._normalize_decode_page_table = normalize_spy

    executor.compile_decode(
        tokens=tokens,
        start_pos=start_pos,
        page_table=page_table,
        sampling_params=sampling_params,
    )
    assert (
        executor.decode_forward(
            tokens,
            start_pos,
            page_table,
            read_from_device=False,
            sampling_params=sampling_params,
        )
        == "raw"
    )

    assert feedback_flags == [True, True]


@pytest.mark.parametrize(
    ("can_sample_on_device", "canonical_sampling_path"),
    [(False, "logits"), (True, "topk")],
)
def test_prefill_warmup_registers_one_canonical_trace_plan_per_shape(
    monkeypatch,
    can_sample_on_device,
    canonical_sampling_path,
):
    executor = make_executor(
        monkeypatch,
        sampling=True,
        trace="all",
        warmup=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
    )
    executor._validate_bound_cache = lambda cache: None
    executor._validation_context = lambda mode: contextlib.nullcontext()
    executor._warm_eager_invocation = lambda invoke: None
    compiled = []

    class Compiler:
        trace_active = False

        def compile(self, key, invoke, *, output_spec, trace_eligible):
            compiled.append((key, trace_eligible))
            return SimpleNamespace(key=key, trace_eligible=trace_eligible)

    executor._graph_compiler = Compiler()
    executor.warmup_model_prefill(
        kv_cache="cache",
        enable_trace=True,
        can_sample_on_device=can_sample_on_device,
    )

    expected_registrations = 2 if can_sample_on_device else 1
    assert len(compiled) == expected_registrations
    assert sum(eligible for _, eligible in compiled) == 1
    assert {key.sampling_path for key, eligible in compiled if eligible} == {canonical_sampling_path}
    assert len(executor._trace_plans) == 1
    assert {key.sampling_path for key in executor._trace_plans} == {canonical_sampling_path}


def test_sampling_disabled_prefill_uses_logits_as_canonical_trace(monkeypatch):
    executor = make_executor(monkeypatch, sampling=False, trace="all")
    executor._validate_bound_cache = lambda cache: None
    executor._validation_context = lambda mode: contextlib.nullcontext()
    compiled = []

    class Compiler:
        trace_active = False

        def compile(self, key, invoke, *, output_spec, trace_eligible):
            compiled.append((key, trace_eligible))
            return SimpleNamespace(key=key, trace_eligible=trace_eligible)

    executor._graph_compiler = Compiler()
    executor.compile_prefill(
        tokens=torch.zeros((1, 128), dtype=torch.long),
        page_table=torch.zeros((1, 4), dtype=torch.int32),
        prompt_lens=torch.tensor([128]),
        empty_slots=[0],
    )

    assert [(key.sampling_path, eligible) for key, eligible in compiled] == [("logits", True)]
    assert [key.sampling_path for key in executor._trace_plans] == ["logits"]


def test_prefill_planning_preserves_mixed_length_fallback_source_order_and_prefix_metadata():
    tokens = torch.arange(2 * 128, dtype=torch.long).view(2, 128)
    page_table = torch.arange(10, dtype=torch.int32).view(2, 5)
    requests = _plan_prefill_requests(
        tokens=tokens,
        page_table=page_table,
        prompt_lens=torch.tensor([64, 100]),
        empty_slots=[2, 1],
        start_pos=torch.tensor([0, 32]),
        block_size=32,
        max_batch_size=4,
        max_prefill_chunk_size=2048,
        can_enable_trace=FakeRuntimeConfig.can_enable_trace,
    )

    assert [request.kind for request in requests] == ["single", "single"]
    assert [request.source_rows for request in requests] == [(0,), (1,)]
    assert [request.slots for request in requests] == [(2,), (1,)]
    assert requests[0].trace_eligible
    assert not requests[1].trace_eligible
    assert requests[1].cached_tokens == (32,)
    assert requests[1].chunk_page_table_width == 4


def test_cached_prefill_normalizes_full_and_truncated_page_tables_and_requires_block_alignment(
    monkeypatch, expect_error
):
    executor = make_executor(monkeypatch)

    def plan(prefix, *, full_width):
        prompt_length = prefix + 31
        actual_width = (prompt_length + 31) // 32
        page_table = torch.full((1, 128 if full_width else actual_width), 999, dtype=torch.int32)
        page_table[0, :actual_width] = torch.arange(actual_width, dtype=torch.int32)
        return executor._plan_prefill(
            torch.zeros((1, prompt_length), dtype=torch.long),
            page_table,
            torch.tensor([prompt_length]),
            [0],
            torch.tensor([prefix]),
        )[0]

    requests = []
    for prefix in (96, 160):
        truncated = plan(prefix, full_width=False)
        full = plan(prefix, full_width=True)
        actual_width = (prefix + 31 + 31) // 32
        assert tuple(truncated.page_table.shape) == (1, 160)
        assert torch.equal(truncated.page_table, full.page_table)
        assert torch.count_nonzero(truncated.page_table[:, actual_width:]) == 0
        assert truncated.cached_tokens == (prefix,)
        assert truncated.chunk_page_table_width == 4
        assert not truncated.trace_eligible
        requests.append(truncated)

    assert requests[0].graph_key("topk") == requests[1].graph_key("topk")

    uncached = executor._plan_prefill(
        torch.zeros((1, 31), dtype=torch.long),
        torch.tensor([[7]], dtype=torch.int32),
        torch.tensor([31]),
        [0],
        torch.tensor([0]),
    )[0]
    assert uncached.page_table[0, 0] == 7
    assert torch.all(uncached.page_table[0, 1:] == -1)

    batched = executor._plan_prefill(
        torch.zeros((3, 31), dtype=torch.long),
        torch.tensor([[7], [8], [9]], dtype=torch.int32),
        torch.full((3,), 31, dtype=torch.long),
        [0, 1, 2],
        torch.zeros(3, dtype=torch.long),
    )[0]
    assert batched.kind == "batched"
    assert torch.equal(batched.page_table[:3, 0], torch.tensor([7, 8, 9], dtype=torch.int32))
    assert torch.all(batched.page_table[:3, 1:] == -1)
    assert torch.all(batched.page_table[3] == -1)

    with expect_error(ValueError, "block aligned"):
        executor._plan_prefill(
            torch.zeros((1, 128), dtype=torch.long),
            torch.zeros((1, 4), dtype=torch.int32),
            torch.tensor([128]),
            [0],
            torch.tensor([97]),
        )


def test_decode_page_table_normalization_ignores_unused_full_width_tail(monkeypatch):
    executor = make_executor(monkeypatch)
    start_pos = torch.tensor([0, 31, 32, 95], dtype=torch.long)
    truncated = torch.arange(12, dtype=torch.int32).view(4, 3)
    full = torch.full((4, 128), 999, dtype=torch.int32)
    full[:, :3] = truncated

    normalized_truncated = executor._normalize_decode_page_table(truncated, start_pos)
    normalized_full = executor._normalize_decode_page_table(full, start_pos)

    assert tuple(normalized_truncated.shape) == (4, 128)
    assert torch.equal(normalized_truncated, normalized_full)
    assert torch.equal(normalized_truncated[0, :1], truncated[0, :1])
    assert torch.count_nonzero(normalized_truncated[0, 1:]) == 0
    assert torch.equal(normalized_truncated[2, :2], truncated[2, :2])
    assert torch.equal(normalized_truncated[3, :3], truncated[3, :3])


def test_decode_page_table_normalization_preserves_one_feedback_lookahead_block(monkeypatch, expect_error):
    executor = make_executor(monkeypatch)
    start_pos = torch.tensor([63, 62, -1], dtype=torch.long)
    page_table = torch.full((3, 128), 999, dtype=torch.int32)
    page_table[:, :3] = torch.tensor([[7, 8, 9], [10, 11, 12], [13, 14, 15]])

    without_lookahead = executor._normalize_decode_page_table(page_table, start_pos)
    with_lookahead = executor._normalize_decode_page_table(
        page_table,
        start_pos,
        allow_one_step_feedback_lag=True,
    )

    assert torch.equal(without_lookahead[0, :2], page_table[0, :2])
    assert torch.count_nonzero(without_lookahead[0, 2:]) == 0
    assert torch.equal(with_lookahead[0, :3], page_table[0, :3])
    assert torch.count_nonzero(with_lookahead[0, 3:]) == 0
    assert torch.equal(with_lookahead[1, :2], page_table[1, :2])
    assert torch.count_nonzero(with_lookahead[1, 2:]) == 0
    assert torch.count_nonzero(with_lookahead[2]) == 0

    previous_page_table = page_table.clone()
    previous_page_table[0, 2] = 0
    normalized_previous = executor._normalize_decode_page_table(
        previous_page_table,
        start_pos,
        allow_one_step_feedback_lag=True,
    )
    assert not torch.equal(normalized_previous, with_lookahead)

    truncated_boundary = executor._normalize_decode_page_table(
        torch.tensor([[7]], dtype=torch.int32),
        torch.tensor([31]),
        allow_one_step_feedback_lag=True,
    )
    assert truncated_boundary[0, 0] == 7
    assert torch.count_nonzero(truncated_boundary[0, 1:]) == 0

    position_zero = executor._normalize_decode_page_table(
        torch.tensor([[7, 8]], dtype=torch.int32),
        torch.tensor([0]),
        allow_one_step_feedback_lag=True,
    )
    assert position_zero[0, 0] == 7
    assert torch.count_nonzero(position_zero[0, 1:]) == 0

    capacity_table = torch.arange(128, dtype=torch.int32).view(1, 128)
    at_capacity = executor._normalize_decode_page_table(
        capacity_table,
        torch.tensor([executor.model.config.max_seq_len - 1]),
        allow_one_step_feedback_lag=True,
    )
    assert torch.equal(at_capacity, capacity_table)
    with expect_error(ValueError, "exceeds the configured paged-KV capacity"):
        executor._normalize_decode_page_table(
            capacity_table,
            torch.tensor([executor.model.config.max_seq_len]),
            allow_one_step_feedback_lag=True,
        )


def test_chunked_prefill_stops_after_chunk_containing_actual_last_token(monkeypatch):
    executor = make_executor(monkeypatch)
    request = _PrefillRequest(
        kind="single",
        source_rows=(0,),
        slots=(0,),
        tokens=torch.zeros((1, 8192), dtype=torch.long),
        page_table=torch.arange(256, dtype=torch.int32).reshape(1, 256),
        prompt_lengths=(4097,),
        cached_tokens=(0,),
        last_token_indices=(4096,),
        padded_sequence_length=8192,
        padded_batch_size=1,
        chunk_page_table_width=64,
        trace_eligible=False,
    )
    chunk_starts = []
    chunk_page_tables = []

    executor._make_device_kpt = lambda *args, **kwargs: None

    def prepare_inputs(*args, **kwargs):
        chunk_page_tables.append(kwargs["chunk_page_table"].clone())
        return (
            "tokens",
            "position-indices",
            "page-table",
            "chunk-page-table",
            "chunk-start",
        )

    executor._prepare_prefill_inputs_host = prepare_inputs
    executor._prepare_prefill_position_inputs_host = lambda *args: ("block-start", "block-end", "row")
    monkeypatch.setattr(
        executor_module,
        "_copy_host_to_device",
        lambda host_inputs, **kwargs: host_inputs,
    )
    executor.model.embed_prefill = lambda value: value

    def prefill_forward(*args, chunk_start_idx, **kwargs):
        chunk_starts.append(chunk_start_idx)
        return f"chunk-{chunk_start_idx}"

    executor.model.prefill_forward = prefill_forward
    monkeypatch.setattr(
        executor_module.ttnn,
        "untilize",
        lambda output, **kwargs: f"untilized-{output}",
    )

    output, _ = executor._run_untraceable_prefill(request, sampling_params=None)

    assert chunk_starts == [0, 2048, 4096]
    assert all(tuple(table.shape) == (1, 64) for table in chunk_page_tables)
    assert torch.equal(chunk_page_tables[0], torch.arange(64, dtype=torch.int32).reshape(1, 64))
    assert torch.equal(chunk_page_tables[1], torch.arange(64, 128, dtype=torch.int32).reshape(1, 64))
    assert chunk_page_tables[2][0, 0] == 128
    assert torch.all(chunk_page_tables[2][0, 1:] == -1)
    assert output == "untilized-chunk-4096"


def test_equal_length_128_prefill_uses_supported_padded_batch():
    requests = _plan_prefill_requests(
        tokens=torch.zeros(3, 128, dtype=torch.long),
        page_table=torch.zeros(3, 4, dtype=torch.int32),
        prompt_lens=torch.tensor([128, 128, 128]),
        empty_slots=[0, 1, 2],
        start_pos=None,
        block_size=32,
        max_batch_size=4,
        max_prefill_chunk_size=2048,
        can_enable_trace=FakeRuntimeConfig.can_enable_trace,
    )

    assert len(requests) == 1
    assert requests[0].kind == "batched"
    assert requests[0].padded_batch_size == 4
    assert requests[0].source_rows == (0, 1, 2)


def test_prefill_output_rows_are_assembled_in_source_order(monkeypatch):
    executor = make_executor(monkeypatch)
    executor._validate_bound_cache = lambda cache: None

    def execute(request, sampling_params, sampling_path, *, enable_trace):
        assert sampling_path == "logits"
        assert not enable_trace
        output = torch.zeros(1, 1, 32, executor.model.vocab_size)
        relative_last = request.last_token_indices[0] - request.cached_tokens[0]
        output[0, 0, relative_last % 32] = request.source_rows[0] + 10
        return output, None

    executor._execute_prefill_request = execute
    output = executor.prefill_forward(
        torch.zeros(2, 128, dtype=torch.long),
        torch.zeros(2, 5, dtype=torch.int32),
        prompt_lens=torch.tensor([64, 100]),
        empty_slots=[2, 1],
        start_pos=torch.tensor([0, 32]),
    )

    assert torch.equal(output[0, 0], torch.full((16,), 10.0))
    assert torch.equal(output[1, 0], torch.full((16,), 11.0))


def test_prefill_sampling_params_are_sliced_for_each_source_row():
    params = SamplingParams(
        temperature=[0.4, 0.8, 1.2],
        top_k=[4, 8, 12],
        top_p=[0.1, 0.2, 0.3],
        enable_log_probs=[False, True, False],
    )

    sliced = _slice_sampling_params(params, (2,))

    assert sliced.temperature == [1.2]
    assert sliced.top_k == [12]
    assert sliced.top_p == [0.3]
    assert sliced.enable_log_probs == [False]

    broadcast = _slice_sampling_params(
        SamplingParams(
            temperature=[0.7],
            top_k=(8,),
            top_p=torch.tensor([0.4]),
            enable_log_probs=False,
        ),
        (2, 0),
    )
    assert broadcast.temperature == [0.7, 0.7]
    assert broadcast.top_k == (8, 8)
    assert torch.equal(broadcast.top_p, torch.tensor([0.4, 0.4]))


def test_same_prefill_graph_key_uses_current_request_last_token_after_replay(monkeypatch):
    executor = make_executor(monkeypatch)
    first = _plan_prefill_requests(
        tokens=torch.zeros(1, 128, dtype=torch.long),
        page_table=torch.zeros(1, 4, dtype=torch.int32),
        prompt_lens=torch.tensor([64]),
        empty_slots=[0],
        start_pos=None,
        block_size=32,
        max_batch_size=4,
        max_prefill_chunk_size=2048,
        can_enable_trace=FakeRuntimeConfig.can_enable_trace,
    )[0]
    second = _plan_prefill_requests(
        tokens=torch.zeros(1, 128, dtype=torch.long),
        page_table=torch.zeros(1, 4, dtype=torch.int32),
        prompt_lens=torch.tensor([100]),
        empty_slots=[0],
        start_pos=None,
        block_size=32,
        max_batch_size=4,
        max_prefill_chunk_size=2048,
        can_enable_trace=FakeRuntimeConfig.can_enable_trace,
    )[0]
    assert first.graph_key("logits") == second.graph_key("logits")

    position_inputs = ("block-start", "block-end", "row")
    graph = SimpleNamespace(
        trace=SimpleNamespace(persistent_inputs=PersistentInputs({"kpt": None, "position_inputs": position_inputs}))
    )
    executor._graph_compiler = SimpleNamespace(
        trace_active=False,
        assert_executable=lambda key: graph,
        replay=lambda *args, **kwargs: "captured-hidden",
    )
    selected_rows = []

    def finish(request, hidden, sampling_params, kpt, current_position_inputs):
        assert hidden == "captured-hidden"
        assert current_position_inputs is position_inputs
        selected_rows.append(request.last_token_indices[0] - request.cached_tokens[0])
        return torch.zeros(1)

    executor._finish_traceable_prefill = finish
    executor._execute_prefill_request(first, None, "logits", enable_trace=True)
    executor._execute_prefill_request(second, None, "logits", enable_trace=True)

    assert selected_rows == [63, 99]


def test_sampling_prefill_paths_reuse_canonical_trace_and_require_requested_graph(monkeypatch, expect_error):
    executor = make_executor(monkeypatch, sampling=True, trace="all")
    executor._prefill_trace_sampling_path = "topk"
    request = _plan_prefill_requests(
        tokens=torch.zeros(1, 128, dtype=torch.long),
        page_table=torch.zeros(1, 4, dtype=torch.int32),
        prompt_lens=torch.tensor([128]),
        empty_slots=[0],
        start_pos=None,
        block_size=32,
        max_batch_size=4,
        max_prefill_chunk_size=2048,
        can_enable_trace=FakeRuntimeConfig.can_enable_trace,
    )[0]
    logits_key = request.graph_key("logits")
    topk_key = request.graph_key("topk")
    canonical_kpt = object()
    canonical_positions = ("block-start", "block-end", "row")
    canonical_trace = SimpleNamespace(
        persistent_inputs=PersistentInputs({"kpt": canonical_kpt, "position_inputs": canonical_positions})
    )
    graphs = {
        logits_key: SimpleNamespace(trace=None),
        topk_key: SimpleNamespace(trace=canonical_trace),
    }
    guarded = []
    replayed = []

    class Compiler:
        trace_active = True

        def assert_executable(self, key):
            guarded.append(key)
            if key not in graphs:
                raise RuntimeError(f"Graph {key!r} was not compiled after trace activation")
            return graphs[key]

        def replay(self, key, refresh, **kwargs):
            replayed.append(key)
            return "captured-hidden"

    executor._graph_compiler = Compiler()
    finished = []
    executor._finish_traceable_prefill = lambda request, hidden, params, kpt, positions: finished.append(
        (hidden, params, kpt, positions)
    )
    sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)

    executor._execute_prefill_request(request, None, "logits", enable_trace=True)
    executor._execute_prefill_request(request, sampling_params, "topk", enable_trace=True)

    assert guarded == [logits_key, topk_key, topk_key]
    assert replayed == [topk_key, topk_key]
    assert [(hidden, kpt, positions) for hidden, _, kpt, positions in finished] == [
        ("captured-hidden", canonical_kpt, canonical_positions),
        ("captured-hidden", canonical_kpt, canonical_positions),
    ]
    assert finished[0][1] is None
    assert finished[1][1] is sampling_params

    executor._prefill_trace_sampling_path = "logits"
    del graphs[topk_key]
    with expect_error(RuntimeError, "not compiled after trace activation"):
        executor._execute_prefill_request(request, sampling_params, "topk", enable_trace=True)


def test_trace_ineligible_cached_prefill_stays_eager_after_trace_activation(monkeypatch):
    executor = make_executor(monkeypatch, sampling=True, trace="all")

    def cached_request(prefix, new_tokens, *, full_width=False):
        prompt_length = prefix + new_tokens
        actual_width = (prompt_length + 31) // 32
        page_width = 128 if full_width else actual_width
        return executor._plan_prefill(
            torch.zeros((1, prompt_length), dtype=torch.long),
            torch.zeros((1, page_width), dtype=torch.int32),
            torch.tensor([prompt_length]),
            [0],
            torch.tensor([prefix]),
        )[0]

    first = cached_request(96, 31)
    same_program = cached_request(160, 31, full_width=True)
    unseen = cached_request(96, 129)
    known_key = first.graph_key("logits")
    unseen_key = unseen.graph_key("logits")
    assert known_key == same_program.graph_key("logits")
    assert known_key != unseen_key
    assert not first.trace_eligible
    assert not same_program.trace_eligible
    graph = SimpleNamespace(trace=None)
    guarded = []

    class Compiler:
        trace_active = True

        def assert_executable(self, key):
            guarded.append(key)
            raise AssertionError("trace-ineligible eager prefill accessed graph compiler")

    executor._graph_compiler = Compiler()
    eager_results = [(object(), (object(),)), (object(), (object(),)), (object(), (object(),))]
    eager_calls = []

    def run_eager(current, params):
        eager_calls.append(current.cached_tokens[0])
        return eager_results[len(eager_calls) - 1]

    executor._run_prefill_eager = run_eager

    assert executor._execute_prefill_request(first, None, "logits", enable_trace=True) is eager_results[0]
    assert executor._execute_prefill_request(same_program, None, "logits", enable_trace=True) is eager_results[1]
    assert executor._execute_prefill_request(unseen, None, "logits", enable_trace=True) is eager_results[2]

    assert eager_calls == [96, 160, 96]
    assert guarded == []


def test_prefill_trace_capture_contains_hidden_body_not_request_dependent_postprocess(monkeypatch):
    executor = make_executor(monkeypatch)
    request = _PrefillRequest(
        kind="single",
        source_rows=(0,),
        slots=(0,),
        tokens=torch.zeros(1, 128, dtype=torch.long),
        page_table=torch.zeros(1, 4, dtype=torch.int32),
        prompt_lengths=(64,),
        cached_tokens=(0,),
        last_token_indices=(63,),
        padded_sequence_length=128,
        padded_batch_size=1,
        chunk_page_table_width=None,
        trace_eligible=True,
    )
    executor._stage_prefill_inputs_and_kpt = lambda *args, **kwargs: (
        ("device-input",),
        ("block-start", "block-end", "row"),
        None,
    )
    executor._prepare_prefill_inputs_host = lambda *args, **kwargs: ("host-input",)
    executor._run_prefill_hidden_body = lambda current, inputs: ("hidden", current.last_token_indices)
    executor._finish_traceable_prefill = lambda *args: (_ for _ in ()).throw(
        AssertionError("postprocess entered capture")
    )

    plan = executor._make_prefill_trace_plan(request, None)
    persistent = PersistentInputs(plan.prepare_inputs())

    assert plan.capture(persistent) == ("hidden", (63,))


def test_prefill_refresh_copies_only_dynamic_inputs(monkeypatch):
    executor = make_executor(monkeypatch, sampling=True)
    request = SimpleNamespace(
        tokens=None,
        page_table=None,
        last_token_indices=(63,),
        cached_tokens=(0,),
        padded_batch_size=1,
        padded_sequence_length=128,
    )
    host = ("tokens", "position-indices", "page", None, None)
    device = ("d_tokens", "d_cos", "d_sin", "d_page", None, "d_position-indices", None)
    position_host = ("block-start", "block-end", "row")
    position_device = ("d_block-start", "d_block-end", "d_row")
    executor._prepare_prefill_inputs_host = lambda *args, **kwargs: host
    executor._prepare_prefill_position_inputs_host = lambda *args: position_host
    copied = []
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda source, target: copied.append((source, target)))
    executor._refresh_kpt = lambda device_kpt, params, *args: copied.append(("kpt", params))
    artifact = SimpleNamespace(
        persistent_inputs=PersistentInputs(
            {"device_inputs": device, "position_inputs": position_device, "kpt": "canonical"}
        )
    )

    executor._refresh_prefill_trace(artifact, request, None, SimpleNamespace())

    expected = [("tokens", "d_tokens"), ("page", "d_page"), *zip(position_host, position_device)]
    assert copied == expected

    copied.clear()
    sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    executor._refresh_prefill_trace(artifact, request, sampling_params, SimpleNamespace())

    assert copied == [*expected, ("kpt", sampling_params)]


def test_decode_page_table_change_copies_only_page_table(monkeypatch):
    executor = make_executor(monkeypatch, sampling=True)
    host = ("tokens", "positions", "rope", "page")
    device = ("d_tokens", "d_positions", "d_rope", "d_page")
    executor._prepare_decode_inputs_host = lambda *args: host
    monkeypatch.setattr(
        executor_module,
        "_copy_host_to_device",
        lambda *args: (_ for _ in ()).throw(AssertionError("stale decode inputs were restaged")),
    )
    copied = []
    monkeypatch.setattr(
        ttnn,
        "copy_host_to_device_tensor",
        lambda source, target: copied.append((source, target)),
    )
    executor._refresh_kpt = lambda device_kpt, params, *args: copied.append(("kpt", device_kpt))
    artifact = SimpleNamespace(persistent_inputs=PersistentInputs({"device_inputs": device, "kpt": "canonical"}))

    executor._refresh_decode_trace(
        artifact,
        torch.zeros(1, 1, dtype=torch.long),
        torch.tensor([64]),
        torch.zeros(1, 3, dtype=torch.int32),
        None,
        "device",
        SimpleNamespace(full=False, page_table=True),
    )

    assert copied == [("page", "d_page"), ("kpt", "canonical")]

    copied.clear()
    executor._prepare_decode_inputs_host = lambda *args: (_ for _ in ()).throw(
        AssertionError("unused host inputs were prepared")
    )
    executor._refresh_decode_trace(
        artifact,
        torch.zeros(1, 1, dtype=torch.long),
        torch.tensor([65]),
        torch.zeros(1, 3, dtype=torch.int32),
        None,
        "device",
        SimpleNamespace(full=False, page_table=False),
    )

    assert copied == [("kpt", "canonical")]


class AsyncValue:
    def __init__(self, host):
        self.host = host
        self.calls = []

    def cpu(self, *, blocking):
        self.calls.append(blocking)
        return self.host


def test_async_read_processes_exact_value_retires_pending_and_preserves_log_probs(monkeypatch):
    executor = make_executor(monkeypatch)
    events = []
    synchronized = []
    monkeypatch.setattr(ttnn, "record_event", lambda mesh, queue: events.append(object()) or events[-1])
    monkeypatch.setattr(ttnn, "event_synchronize", synchronized.append)

    token_host = torch.arange(4, dtype=torch.int32).view(1, 1, 4, 1)
    log_probs = torch.tensor([0.25])
    raw = (AsyncValue(token_host), AsyncValue(log_probs))
    record = _ExternalDecodeOutput(raw_value=raw, owned_values=None)
    executor._external_by_raw_id[id(raw)] = record

    host, read_events = executor.read_decode_output(raw, async_read=True)
    assert executor.output_reader.pending_count == 1
    assert read_events == events
    ttnn.event_synchronize(read_events[0])

    tokens, returned_log_probs = executor.process_decode_output_host(host, is_tokens=True)

    assert torch.equal(tokens, torch.arange(4, dtype=torch.int32))
    assert returned_log_probs is log_probs
    assert executor.output_reader.pending_count == 0
    assert executor._external_by_raw_id == {}
    assert executor._external_by_host_id == {}
    assert synchronized == [events[0], events[0]]


class OwnedTensor:
    pass


def test_copy_host_to_device_cleans_partial_allocation_and_preserves_primary(monkeypatch, expect_error):
    first = OwnedTensor()
    calls = 0
    deallocated = []
    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)

    def to_device(value, *, device):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second allocation failed")
        return first

    monkeypatch.setattr(ttnn, "to_device", to_device)

    with expect_error(RuntimeError, "second allocation failed"):
        _copy_host_to_device((object(), object()), mesh_device="mesh")

    assert deallocated == [first]


@pytest.mark.parametrize("trace_prepare", [False, True])
def test_kpt_allocation_failure_cleans_staged_inputs_for_eager_and_trace(monkeypatch, trace_prepare, expect_error):
    executor = make_executor(monkeypatch)
    allocated = []
    deallocated = []
    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)
    monkeypatch.setattr(
        ttnn,
        "to_device",
        lambda value, *, device: allocated.append(OwnedTensor()) or allocated[-1],
    )
    executor._make_device_kpt = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("kpt allocation failed"))

    if trace_prepare:
        request = _PrefillRequest(
            kind="single",
            source_rows=(0,),
            slots=(0,),
            tokens=torch.zeros(1, 128, dtype=torch.long),
            page_table=torch.zeros(1, 4, dtype=torch.int32),
            prompt_lengths=(128,),
            cached_tokens=(0,),
            last_token_indices=(127,),
            padded_sequence_length=128,
            padded_batch_size=1,
            chunk_page_table_width=None,
            trace_eligible=True,
        )
        executor._prepare_prefill_inputs_host = lambda *args, **kwargs: (
            object(),
            object(),
            object(),
            object(),
            object(),
        )
        executor._prepare_prefill_position_inputs_host = lambda *args: (object(), object(), object())
        operation = executor._make_prefill_trace_plan(request, None).prepare_inputs
    else:
        executor._prepare_decode_inputs_host = lambda *args: (object(), object())
        operation = lambda: executor._run_decode_eager(
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4, 1, dtype=torch.int32),
            None,
            "logits",
            device_feedback=False,
        )

    with expect_error(RuntimeError, "kpt allocation failed"):
        operation()

    assert len(deallocated) == len(allocated)
    assert {id(value) for value in deallocated} == {id(value) for value in allocated}


def test_eager_owned_bundle_includes_raw_output(monkeypatch):
    executor = make_executor(monkeypatch)
    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    deallocated = []
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)
    request = SimpleNamespace(
        trace_eligible=True,
        tokens=torch.zeros(1, 128),
        page_table=torch.zeros(1, 4),
        cached_tokens=(0,),
        last_token_indices=(127,),
        padded_batch_size=1,
        padded_sequence_length=128,
    )
    staged = OwnedTensor()
    position = OwnedTensor()
    hidden = OwnedTensor()
    output = OwnedTensor()
    executor._prepare_prefill_inputs_host = lambda *args, **kwargs: (object(),)
    executor._stage_prefill_inputs_and_kpt = lambda *args, **kwargs: ((staged,), (position,), None)
    executor._run_prefill_hidden_body = lambda *args: hidden
    executor._finish_traceable_prefill = lambda *args: output

    raw, owned = executor._run_prefill_eager(request, None)
    _deallocate_owned_ttnn(owned)

    assert raw is output
    assert deallocated.count(output) == 1
    assert deallocated.count(hidden) == 1
    assert deallocated.count(staged) == 1
    assert deallocated.count(position) == 1


def test_external_decode_deallocation_failure_is_best_effort_and_retryable(monkeypatch, expect_error):
    executor = make_executor(monkeypatch)
    first = OwnedTensor()
    second = OwnedTensor()
    attempts = []
    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)

    def fail_second_once(value):
        attempts.append(value)
        if value is second and attempts.count(second) == 1:
            raise RuntimeError("deallocate failed")

    monkeypatch.setattr(ttnn, "deallocate", fail_second_once)
    raw = (first, second)
    record = _ExternalDecodeOutput(raw_value=raw, owned_values=None)
    executor._external_by_raw_id[id(raw)] = record

    with expect_error(RuntimeError, "deallocate failed"):
        executor._release_external_record(record)

    assert not record.released
    assert attempts == [first, second]

    executor._release_external_record(record)

    assert record.released
    assert attempts.count(first) == 1
    assert attempts.count(second) == 2
    assert executor._external_by_raw_id == {}


def test_eager_failure_retains_transient_cleanup_for_retry_and_blocks_execution(monkeypatch, expect_error):
    executor = make_executor(monkeypatch)
    first = OwnedTensor()
    retry = OwnedTensor()
    primary = RuntimeError("decode compute failed")
    operation_cleanup_error = RuntimeError("operation cleanup failed")
    cleanup_retry_error = RuntimeError("cleanup retry failed")
    attempts = []

    def deallocate(value):
        attempts.append(value)
        retry_attempt = attempts.count(retry)
        if value is retry and retry_attempt == 1:
            raise operation_cleanup_error
        if value is retry and retry_attempt == 2:
            raise cleanup_retry_error

    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", deallocate)
    executor._prepare_decode_inputs_host = lambda *args: (object(),)
    executor._stage_device_inputs_and_kpt = lambda *args, **kwargs: ((first, retry), None)
    executor._run_decode_body = lambda *args, **kwargs: (_ for _ in ()).throw(primary)

    with expect_error(RuntimeError, "decode compute failed") as caught:
        executor._run_decode_eager(
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            torch.zeros((4, 1), dtype=torch.int32),
            None,
            "logits",
            device_feedback=False,
        )

    assert caught.value is primary
    assert caught.value.cleanup_failures == (operation_cleanup_error,)
    assert attempts == [first, retry]
    with expect_error(RuntimeError, "unreleased transient"):
        executor._ensure_active()

    events = []
    executor._drain_external_decode_outputs = lambda: []
    executor._output_reader = SimpleNamespace(drain=lambda: events.append("reader"))
    executor._graph_compiler = SimpleNamespace(cleanup=lambda: events.append("graph"))
    executor._kv_manager = SimpleNamespace(release=lambda: events.append("kv"))

    with expect_error(RuntimeError, "cleanup retry failed") as cleanup_caught:
        executor.cleanup()

    assert cleanup_caught.value is cleanup_retry_error
    assert attempts.count(first) == 1
    assert attempts.count(retry) == 2
    assert events == ["reader"]

    executor.cleanup()

    assert attempts.count(first) == 1
    assert attempts.count(retry) == 3
    assert executor._transient_orphans == []
    assert events == ["reader", "reader", "graph", "kv"]


def test_warmup_sampling_hint_false_covers_only_logits(monkeypatch):
    warmup = WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1, 2, 4))
    executor = make_executor(monkeypatch, sampling=True, trace="all", warmup=warmup)
    executor._validate_bound_cache = lambda cache: None
    executor._ensure_sampling_buffers = lambda: None
    calls = []
    executor.compile_prefill = lambda **kwargs: calls.append(kwargs)

    executor.warmup_model_prefill(
        kv_cache="cache",
        enable_trace=False,
        can_sample_on_device=False,
    )

    assert len(calls) == 4
    assert all(call["sampling_params"] is None for call in calls)
    assert all(call["enable_trace"] is False for call in calls)
    assert int(calls[-1]["tokens"].shape[1]) == 160
    assert torch.equal(calls[-1]["start_pos"], torch.tensor([32]))
    assert executor.already_warmed_up_prefill


def test_default_warmup_uses_model_lengths_lane_batches_and_all_configured_decode_paths(monkeypatch):
    executor = make_executor(
        monkeypatch,
        sampling=True,
        warmup=WarmupConfig(include_decode_top_k=True),
    )
    executor._validate_bound_cache = lambda cache: None
    executor._ensure_sampling_buffers = lambda: None
    prefill_calls = []
    decode_calls = []
    executor.compile_prefill = lambda **kwargs: prefill_calls.append(kwargs)
    executor.compile_decode = lambda **kwargs: decode_calls.append(kwargs)

    executor.warmup_model_prefill(
        kv_cache="cache",
        enable_trace=False,
        can_sample_on_device=True,
    )
    executor.warmup_model_decode(
        kv_cache="cache",
        enable_trace=False,
        max_batch_size=4,
        num_blocks=7,
        can_sample_on_device=True,
    )

    assert [
        (int(call["tokens"].shape[1]), int(call["tokens"].shape[0]), call["sampling_params"] is not None)
        for call in prefill_calls
    ] == [
        (128, 1, False),
        (128, 1, True),
        (128, 2, False),
        (128, 2, True),
        (128, 4, False),
        (128, 4, True),
        (160, 1, False),
        (160, 1, True),
        (1024, 1, False),
        (1024, 1, True),
        (1056, 1, False),
        (1056, 1, True),
    ]
    assert [int(call["page_table"].shape[1]) for call in prefill_calls] == [4] * 6 + [5] * 2 + [32] * 2 + [33] * 2
    assert [
        None if call.get("start_pos") is None else tuple(int(value) for value in call["start_pos"])
        for call in prefill_calls
    ] == [None] * 6 + [(32,), (32,), None, None, (32,), (32,)]
    assert all(call["kv_cache"] == "cache" for call in prefill_calls)
    assert [executor._decode_sampling_path(call["sampling_params"], 4) for call in decode_calls] == [
        "logits",
        "argmax",
        "topk",
    ]
    assert all(tuple(call["tokens"].shape) == (4,) for call in decode_calls)
    assert all(tuple(call["start_pos"].shape) == (4,) for call in decode_calls)
    assert all(tuple(call["page_table"].shape) == (4, 7) for call in decode_calls)
    assert all(call["kv_cache"] == "cache" for call in decode_calls)
    assert executor.already_warmed_up_prefill


def test_warmup_sampling_true_conflicts_with_disabled_static_policy(monkeypatch, expect_error):
    executor = make_executor(monkeypatch, sampling=False)
    with expect_error(ValueError, "static sampling policy"):
        executor.warmup_model_prefill(
            kv_cache=None,
            enable_trace=False,
            can_sample_on_device=True,
        )


def test_failed_prefill_warmup_remains_retryable(monkeypatch, expect_error):
    warmup = WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,))
    executor = make_executor(monkeypatch, warmup=warmup)
    executor._validate_bound_cache = lambda cache: None
    executor.compile_prefill = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("compile failed"))

    with expect_error(RuntimeError, "compile failed"):
        executor.warmup_model_prefill(None, False, False)
    assert not executor.already_warmed_up_prefill

    calls = []
    executor.compile_prefill = lambda **kwargs: calls.append(kwargs)
    executor.warmup_model_prefill(None, False, False)
    assert len(calls) == 2
    assert executor.already_warmed_up_prefill


def test_cleanup_orders_reader_graph_sampling_and_kv_and_is_idempotent(monkeypatch, expect_error):
    executor = make_executor(monkeypatch, sampling=True)
    events = []
    model_weight_events = []
    model_weight = SimpleNamespace(
        deallocate=lambda: model_weight_events.append("deallocate"),
        release=lambda: model_weight_events.append("release"),
    )
    executor.model.weight = model_weight
    monkeypatch.setattr(ttnn, "deallocate", lambda value: model_weight_events.append(("ttnn", value)))
    executor._drain_external_decode_outputs = lambda: events.append("external") or []
    executor._output_reader = SimpleNamespace(drain=lambda: events.append("reader"))
    executor._graph_compiler = SimpleNamespace(cleanup=lambda: events.append("graph"))
    executor.model.sampling.release = lambda: events.append("sampling")
    executor._kv_manager = SimpleNamespace(release=lambda: events.append("kv"))
    executor._trace_plans = {}

    executor.cleanup()
    executor.cleanup()

    assert events == ["external", "reader", "graph", "sampling", "kv"]
    assert executor.model.weight is model_weight
    assert model_weight_events == []
    with expect_error(RuntimeError, "terminal"):
        executor._ensure_active()


def test_executor_has_no_legacy_executor_dependency():
    source = inspect.getsource(executor_module)
    assert "models.common.models.executor" not in source
    assert "EagerLLMExecutor" not in source
    assert "TracedLLMExecutor" not in source
