# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import models.common.llm_runtime.executor as executor_module
import ttnn
from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from models.common.llm_runtime.vllm_adapter import VLLMAdapter
from models.common.sampling import SamplingParams
from models.common.tests.llm_runtime.test_executor import make_executor


class _ForbiddenGraphCompiler:
    """Fail if an eager execution crosses the trace-registry boundary."""

    def __getattribute__(self, name):
        raise AssertionError(f"eager execution accessed graph compiler method {name}")


def _fail_graph_key(*args, **kwargs):
    raise AssertionError("eager execution constructed an LLMGraphCompiler.GraphKey")


def _adapter(trace_mode: str) -> VLLMAdapter:
    return VLLMAdapter(
        trace_config=TraceConfig(trace_mode),
        paged_kv_cache_config=PagedKVCacheConfig(
            block_size=32,
            max_num_blocks=128,
            dtype=ttnn.bfloat8_b,
        ),
        expected_num_layers=1,
        expected_kv_heads_per_device=1,
        expected_head_dim=8,
        model_kv_cache_dtype=ttnn.bfloat8_b,
    )


def _normalize(adapter: VLLMAdapter, operation: str, enable_trace: bool):
    if operation == "prefill":
        return adapter.normalize_prefill(
            (torch.zeros((1, 128)), torch.zeros((1, 4))),
            {"enable_trace": enable_trace},
        )
    return adapter.normalize_decode(
        (torch.zeros(4), torch.zeros(4), torch.zeros((4, 1))),
        {"enable_trace": enable_trace},
    )


@pytest.mark.parametrize("trace_mode", ["none", "decode_only", "all"])
@pytest.mark.parametrize("operation", ["prefill", "decode"])
@pytest.mark.parametrize("enable_trace", [False, True])
def test_adapter_retains_dynamic_trace_choice_within_static_ceiling(
    trace_mode,
    operation,
    enable_trace,
    expect_error,
):
    adapter = _adapter(trace_mode)
    configured = getattr(adapter.trace_config, f"{operation}_enabled")

    if enable_trace and not configured:
        with expect_error(ValueError, "enable_trace"):
            _normalize(adapter, operation, enable_trace)
        return

    normalized = _normalize(adapter, operation, enable_trace)
    assert normalized["enable_trace"] is enable_trace


@pytest.mark.parametrize(
    ("trace_mode", "enable_trace"),
    [("all", False), ("none", None)],
)
def test_eager_prefill_never_constructs_or_looks_up_graph_keys(
    monkeypatch,
    trace_mode,
    enable_trace,
):
    executor = make_executor(monkeypatch, trace=trace_mode)
    request = SimpleNamespace(
        kind="single",
        source_rows=(0,),
        slots=(0,),
        cached_tokens=(0,),
        last_token_indices=(0,),
        padded_batch_size=1,
        trace_eligible=True,
        graph_key=_fail_graph_key,
    )
    executor._validate_bound_cache = lambda cache: None
    executor._plan_prefill = lambda *args: [request]
    executor._graph_compiler = _ForbiddenGraphCompiler()
    executor._prefill_trace_key = _fail_graph_key
    executor._run_prefill_eager = lambda *args: ("eager-prefill", None)
    executor._output_reader = SimpleNamespace(
        read=lambda value, *, blocking: torch.zeros(1, 1, 32, executor.model.vocab_size)
    )

    output = executor.prefill_forward(
        torch.zeros((1, 128), dtype=torch.long),
        torch.zeros((1, 4), dtype=torch.int32),
        enable_trace=enable_trace,
    )

    assert tuple(output.shape) == (1, 1, executor.model.vocab_size)


@pytest.mark.parametrize(
    ("trace_mode", "enable_trace"),
    [("all", False), ("none", None)],
)
def test_eager_decode_remains_available_after_trace_activation_for_unseen_shape(
    monkeypatch,
    trace_mode,
    enable_trace,
):
    executor = make_executor(monkeypatch, trace=trace_mode)
    executor._validate_bound_cache = lambda cache: None
    executor._graph_compiler = _ForbiddenGraphCompiler()
    executor._decode_graph_key = _fail_graph_key
    eager_calls = []
    executor._run_decode_eager = lambda *args, **kwargs: eager_calls.append((args, kwargs)) or (
        "eager-decode",
        None,
    )

    output = executor.decode_forward(
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.zeros((4, 7), dtype=torch.int32),
        enable_trace=enable_trace,
        read_from_device=False,
    )

    assert output == "eager-decode"
    assert len(eager_calls) == 1


def test_trace_ineligible_prefill_falls_back_to_eager_without_graph_key(monkeypatch):
    executor = make_executor(monkeypatch, trace="all")
    request = SimpleNamespace(
        kind="single",
        source_rows=(0,),
        slots=(0,),
        cached_tokens=(0,),
        last_token_indices=(0,),
        padded_batch_size=1,
        trace_eligible=False,
        graph_key=_fail_graph_key,
    )
    executor._validate_bound_cache = lambda cache: None
    executor._plan_prefill = lambda *args: [request]
    executor._graph_compiler = _ForbiddenGraphCompiler()
    executor._prefill_trace_key = _fail_graph_key
    executor._run_prefill_eager = lambda *args: ("eager-prefill", None)
    executor._output_reader = SimpleNamespace(
        read=lambda value, *, blocking: torch.zeros(1, 1, 32, executor.model.vocab_size)
    )

    output = executor.prefill_forward(
        torch.zeros((1, 100), dtype=torch.long),
        torch.zeros((1, 4), dtype=torch.int32),
        enable_trace=True,
    )

    assert tuple(output.shape) == (1, 1, executor.model.vocab_size)


def test_traced_decode_requires_a_captured_graph(monkeypatch, expect_error):
    executor = make_executor(monkeypatch, trace="all")
    executor._validate_bound_cache = lambda cache: None
    executor._run_decode_eager = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("trace-eligible decode unexpectedly ran eagerly")
    )

    class _MissingTraceCompiler:
        trace_active = True

        @staticmethod
        def get(key):
            return None

        @staticmethod
        def assert_executable(key):
            raise RuntimeError(f"required trace {key!r} was not captured")

        @staticmethod
        def replay(*args, **kwargs):
            raise RuntimeError("required trace was not captured")

    executor._graph_compiler = _MissingTraceCompiler()

    with expect_error(RuntimeError, "trace.*not captured"):
        executor.decode_forward(
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            torch.zeros((4, 1), dtype=torch.int32),
            enable_trace=True,
            read_from_device=False,
        )


def test_trace_request_cannot_exceed_static_ceiling(monkeypatch, expect_error):
    executor = make_executor(monkeypatch, trace="none")
    executor._validate_bound_cache = lambda cache: None
    executor._decode_graph_key = _fail_graph_key

    with expect_error(ValueError, "static decode trace policy"):
        executor.decode_forward(
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            torch.zeros((4, 1), dtype=torch.int32),
            enable_trace=True,
            read_from_device=False,
        )


def test_sampling_params_select_device_sampling_while_host_fallback_keeps_logits(monkeypatch):
    executor = make_executor(monkeypatch, sampling=True, trace="all")
    executor._sampling_buffers_loaded = True
    executor._validate_bound_cache = lambda cache: None
    executor._graph_compiler = _ForbiddenGraphCompiler()
    executor._decode_graph_key = _fail_graph_key
    eager_calls = []
    executor._run_decode_eager = lambda *args, **kwargs: eager_calls.append((args, kwargs)) or (
        object(),
        None,
    )
    inputs = (
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.zeros((4, 1), dtype=torch.int32),
    )
    sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)

    executor.decode_forward(*inputs, enable_trace=False, read_from_device=False)
    executor.decode_forward(
        *inputs,
        enable_trace=False,
        read_from_device=False,
        sampling_params=sampling_params,
    )

    assert [call[0][3] for call in eager_calls] == [None, sampling_params]
    assert [call[0][4] for call in eager_calls] == ["logits", "argmax"]


@pytest.mark.parametrize("operation", ["prefill", "decode"])
@pytest.mark.parametrize("can_sample_on_device", [False, True])
def test_eager_warmup_bypasses_graph_registry_and_controls_sampling_coverage(
    monkeypatch,
    operation,
    can_sample_on_device,
):
    executor = make_executor(
        monkeypatch,
        sampling=True,
        trace="all",
        warmup=WarmupConfig(
            prefill_seq_lens=(128,),
            prefill_batch_sizes=(1,),
            include_decode_top_k=True,
        ),
    )
    executor._validate_bound_cache = lambda cache: None
    executor._validation_context = lambda mode: pytest.MonkeyPatch.context()
    executor._graph_compiler = _ForbiddenGraphCompiler()
    monkeypatch.setattr(executor_module, "GraphKey", _fail_graph_key)
    sampling_buffer_loads = []

    def ensure_sampling_buffers():
        sampling_buffer_loads.append(True)
        executor._sampling_buffers_loaded = True

    executor._ensure_sampling_buffers = ensure_sampling_buffers
    executor._warm_eager_invocation = lambda invoke: invoke()
    eager_calls = []
    executor._run_prefill_eager = lambda request, params: eager_calls.append(params) or (object(), None)
    executor._run_decode_eager = lambda tokens, positions, page_table, params, path, **kwargs: (
        eager_calls.append(params) or (object(), None)
    )

    if operation == "prefill":
        executor.warmup_model_prefill(
            kv_cache="cache",
            enable_trace=False,
            can_sample_on_device=can_sample_on_device,
        )
        expected_logits = 2
        expected_sampling = 2
    else:
        executor.warmup_model_decode(
            kv_cache="cache",
            enable_trace=False,
            max_batch_size=4,
            num_blocks=7,
            can_sample_on_device=can_sample_on_device,
        )
        expected_logits = 1
        expected_sampling = 2

    assert sum(params is None for params in eager_calls) == expected_logits
    assert sum(params is not None for params in eager_calls) == (expected_sampling if can_sample_on_device else 0)
    assert bool(sampling_buffer_loads) is can_sample_on_device


@pytest.mark.parametrize("can_sample_on_device", [False, True])
def test_traced_prefill_warmup_registers_requested_sampling_variants(
    monkeypatch,
    can_sample_on_device,
):
    executor = make_executor(
        monkeypatch,
        sampling=True,
        trace="all",
        warmup=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
    )
    executor._validate_bound_cache = lambda cache: None
    sampling_buffer_loads = []
    executor._ensure_sampling_buffers = lambda: sampling_buffer_loads.append(True)
    compile_calls = []
    executor.compile_prefill = lambda **kwargs: compile_calls.append(kwargs)
    capture_calls = []
    executor._maybe_capture_traces = lambda *args, **kwargs: capture_calls.append((args, kwargs))

    executor.warmup_model_prefill(
        kv_cache="cache",
        enable_trace=True,
        can_sample_on_device=can_sample_on_device,
    )

    sampling_variants = [call for call in compile_calls if call["sampling_params"] is not None]
    logits_variants = [call for call in compile_calls if call["sampling_params"] is None]
    assert len(logits_variants) == 2
    assert len(sampling_variants) == (2 if can_sample_on_device else 0)
    assert all(call["enable_trace"] is True for call in compile_calls)
    assert bool(sampling_buffer_loads) is can_sample_on_device
    assert len(capture_calls) == 1


@pytest.mark.parametrize("can_sample_on_device", [False, True])
def test_traced_decode_warmup_registers_requested_sampling_variants(
    monkeypatch,
    can_sample_on_device,
):
    executor = make_executor(
        monkeypatch,
        sampling=True,
        trace="all",
        warmup=WarmupConfig(include_decode_top_k=True),
    )
    executor._validate_bound_cache = lambda cache: None
    sampling_buffer_loads = []
    executor._ensure_sampling_buffers = lambda: sampling_buffer_loads.append(True)
    compile_calls = []
    executor.compile_decode = lambda **kwargs: compile_calls.append(kwargs)
    capture_calls = []
    executor._maybe_capture_traces = lambda *args, **kwargs: capture_calls.append((args, kwargs))

    executor.warmup_model_decode(
        kv_cache="cache",
        enable_trace=True,
        max_batch_size=4,
        num_blocks=7,
        can_sample_on_device=can_sample_on_device,
    )

    sampling_variants = [call for call in compile_calls if call["sampling_params"] is not None]
    logits_variants = [call for call in compile_calls if call["sampling_params"] is None]
    assert len(logits_variants) == 1
    assert len(sampling_variants) == (2 if can_sample_on_device else 0)
    assert all(call["enable_trace"] is True for call in compile_calls)
    assert bool(sampling_buffer_loads) is can_sample_on_device
    assert len(capture_calls) == 1


def test_static_all_dynamic_decode_only_warmup_captures_only_decode_plans(monkeypatch):
    executor = make_executor(monkeypatch, sampling=False, trace="all")
    executor._validate_bound_cache = lambda cache: None
    executor._validation_context = lambda mode: pytest.MonkeyPatch.context()
    executor._warm_eager_invocation = lambda invoke: None
    captured = []

    class _RecordingCompiler:
        trace_active = False

        def __init__(self):
            self.graphs = {}

        def compile(self, key, invoke, **kwargs):
            self.graphs[key] = SimpleNamespace(
                key=key,
                trace_eligible=kwargs.get("trace_eligible", True),
            )

        def capture_all(self, plans):
            captured.append(tuple(plans))
            self.trace_active = True

    executor._graph_compiler = _RecordingCompiler()

    executor.warmup_model_prefill(
        kv_cache="cache",
        enable_trace=False,
        can_sample_on_device=False,
    )
    executor.warmup_model_decode(
        kv_cache="cache",
        enable_trace=True,
        max_batch_size=4,
        num_blocks=7,
        can_sample_on_device=False,
    )

    assert len(captured) == 1
    assert {key.mode for key in captured[0]} == {"decode"}


def test_disabled_static_sampling_rejects_warmup_and_runtime_expansion(monkeypatch, expect_error):
    executor = make_executor(monkeypatch, sampling=False, trace="all")
    executor._validate_bound_cache = lambda cache: None
    sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)

    with expect_error(ValueError, "static sampling policy"):
        executor.warmup_model_decode(
            kv_cache="cache",
            enable_trace=False,
            max_batch_size=4,
            num_blocks=1,
            can_sample_on_device=True,
        )

    with expect_error(ValueError, "device sampling is disabled"):
        executor.decode_forward(
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            torch.zeros((4, 1), dtype=torch.int32),
            enable_trace=False,
            sampling_params=sampling_params,
        )
