# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import contextlib
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.llm_runtime.config import LLMExecutorConfig, PagedKVCacheConfig, TraceConfig, WarmupConfig
from models.common.llm_runtime.executor import LLMExecutor
from models.common.llm_runtime.lane_group import LaneGroupExecutor
from models.common.models import generator as legacy_generator
from models.common.models.llama3_8b import executor as llama_executor
from models.common.models.llama3_8b import generator as llama_generator
from models.common.tests.demos.llama3_8b import demo as llama_demo


def _model(mesh_device, *, max_batch_size=2, n_layers=2):
    attention_configs = [
        SimpleNamespace(
            n_kv_heads=8,
            head_dim=128,
            kv_cache_dtype=ttnn.bfloat8_b,
        )
        for _ in range(n_layers)
    ]
    return SimpleNamespace(
        config=SimpleNamespace(
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            n_layers=n_layers,
            num_devices=1,
        ),
        layers=[
            SimpleNamespace(attention=SimpleNamespace(config=attention_config))
            for attention_config in attention_configs
        ],
    )


def _llm(mesh_device, *, max_batch_size=2, n_layers=2):
    return SimpleNamespace(
        model=_model(
            mesh_device,
            max_batch_size=max_batch_size,
            n_layers=n_layers,
        ),
        runtime_config=SimpleNamespace(model_cache_path=f"cache-{mesh_device}"),
    )


class _FakeLaneExecutor:
    requires_prefill_trace_warmup = True

    def __init__(self, llm, config):
        self.model = llm.model
        self.model_args = llm.runtime_config
        self.mesh_device = llm.model.config.mesh_device
        self.cache_path = llm.runtime_config.model_cache_path
        self.config = config
        self.paged_kv_cache_config = config.paged_kv_cache
        self.already_warmed_up_prefill = False
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1


def test_build_llama3_executor_uses_the_prebuilt_product(monkeypatch):
    llm = _llm("mesh")
    config = object()
    sentinel = object()
    calls = []

    def fake_runtime_executor(model, runtime_config, executor_config):
        calls.append((model, runtime_config, executor_config))
        return sentinel

    monkeypatch.setattr(llama_executor, "LLMExecutor", fake_runtime_executor)

    assert llama_executor.build_llama3_executor(llm, config) is sentinel
    assert calls == [(llm.model, llm.runtime_config, config)]
    assert not hasattr(llama_executor, "from_pretrained")
    assert legacy_generator.EagerLlamaExecutor is llama_executor.EagerLlamaExecutor
    assert legacy_generator.TracedLlamaExecutor is llama_executor.TracedLlamaExecutor


def test_initialize_vllm_model_threads_construction_policy_to_the_builder(monkeypatch):
    captured = []
    sentinel = object()
    global_mesh = object()

    def fake_builder(config):
        captured.append(config)
        return sentinel

    monkeypatch.setattr(llama_generator, "build_llama3_generator", fake_builder)

    result = llama_generator.Llama3Generator.initialize_vllm_model(
        SimpleNamespace(_name_or_path="meta-llama/Llama-3.1-8B-Instruct"),
        global_mesh,
        8,
        4096,
        n_layers=3,
        tt_data_parallel=2,
        optimizations="accuracy",
        trace_mode="decode_only",
        device_sampling_enabled=True,
    )

    assert result is sentinel
    assert len(captured) == 1
    config = captured[0]
    assert config.hf_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert config.mesh_device is global_mesh
    assert config.max_batch_size == 8
    assert config.max_seq_len == 4096
    assert config.n_layers == 3
    assert config.tt_data_parallel == 2
    assert config.optimizations == "accuracy"
    assert config.trace_mode == "decode_only"
    assert config.device_sampling_enabled is True


def test_initialize_vllm_model_uses_vllm_capability_defaults(monkeypatch):
    captured = []
    monkeypatch.setattr(
        llama_generator,
        "build_llama3_generator",
        lambda config: captured.append(config) or object(),
    )

    llama_generator.Llama3Generator.initialize_vllm_model(
        SimpleNamespace(_name_or_path="meta-llama/Llama-3.1-8B-Instruct"),
        object(),
        8,
        4096,
    )

    assert captured[0].trace_mode == "all"
    assert captured[0].device_sampling_enabled is True


def test_build_llama3_generator_constructs_distinct_lane_products_and_configs(monkeypatch):
    from_pretrained_calls = []
    executor_calls = []
    global_mesh = object()

    monkeypatch.setattr(
        llama_generator,
        "_create_submeshes",
        lambda mesh_device, data_parallel: ["lane-mesh-0", "lane-mesh-1"],
    )

    def fake_from_pretrained(**kwargs):
        from_pretrained_calls.append(kwargs)
        return _llm(
            kwargs["mesh_device"],
            max_batch_size=kwargs["max_batch_size"],
            n_layers=kwargs["n_layers"],
        )

    def fake_build_executor(llm, config):
        executor_calls.append((llm, config))
        return _FakeLaneExecutor(llm, config)

    monkeypatch.setattr(llama_generator, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(llama_generator, "build_llama3_executor", fake_build_executor)

    generator = llama_generator.build_llama3_generator(
        llama_generator.Llama3GeneratorConfig(
            hf_model="meta-llama/Llama-3.1-8B-Instruct",
            mesh_device=global_mesh,
            max_batch_size=4,
            max_seq_len=128,
            n_layers=2,
            tt_data_parallel=2,
            trace_mode="decode_only",
            device_sampling_enabled=True,
        )
    )
    try:
        assert [call["mesh_device"] for call in from_pretrained_calls] == [
            "lane-mesh-0",
            "lane-mesh-1",
        ]
        assert [call["max_batch_size"] for call in from_pretrained_calls] == [2, 2]
        assert len({id(llm) for llm, _ in executor_calls}) == 2

        executor_configs = [config for _, config in executor_calls]
        assert executor_configs[0] is not executor_configs[1]
        assert all(config.trace.mode == "decode_only" for config in executor_configs)
        assert all(config.device_sampling_enabled for config in executor_configs)
        assert all(config.paged_kv_cache.max_num_blocks == 6 for config in executor_configs)
        assert all(config.paged_kv_cache.num_blocks is None for config in executor_configs)
        assert generator.model == [lane.model for lane in generator.target.lanes]
        assert generator.mesh_device is global_mesh
        assert generator.target.mesh_device is global_mesh
    finally:
        lanes = list(generator.target.lanes)
        generator.cleanup()

    assert [lane.cleanup_calls for lane in lanes] == [1, 1]


@pytest.mark.parametrize(
    ("max_batch_size", "data_parallel", "message"),
    [
        (4, 0, "positive integer"),
        (3, 2, "must be divisible"),
    ],
)
def test_generator_config_rejects_invalid_dp_before_model_construction(
    monkeypatch,
    max_batch_size,
    data_parallel,
    message,
    expect_error,
):
    model_construction_calls = []
    monkeypatch.setattr(
        llama_generator,
        "from_pretrained",
        lambda **kwargs: model_construction_calls.append(kwargs),
    )

    with expect_error(ValueError, message):
        llama_generator.Llama3GeneratorConfig(
            hf_model="model",
            mesh_device="mesh",
            max_batch_size=max_batch_size,
            max_seq_len=128,
            tt_data_parallel=data_parallel,
        )

    assert model_construction_calls == []


def test_partial_lane_construction_failure_cleans_constructed_executors(monkeypatch, expect_error):
    built_lanes = []
    call_count = 0

    monkeypatch.setattr(
        llama_generator,
        "_create_submeshes",
        lambda mesh_device, data_parallel: ["lane-mesh-0", "lane-mesh-1"],
    )
    monkeypatch.setattr(
        llama_generator,
        "from_pretrained",
        lambda **kwargs: _llm(
            kwargs["mesh_device"],
            max_batch_size=kwargs["max_batch_size"],
            n_layers=kwargs["n_layers"],
        ),
    )

    def fail_second_executor(llm, config):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("second lane failed")
        lane = _FakeLaneExecutor(llm, config)
        built_lanes.append(lane)
        return lane

    monkeypatch.setattr(
        llama_generator,
        "build_llama3_executor",
        fail_second_executor,
    )

    with expect_error(RuntimeError, "second lane failed"):
        llama_generator.build_llama3_generator(
            llama_generator.Llama3GeneratorConfig(
                hf_model="model",
                mesh_device="mesh",
                max_batch_size=4,
                max_seq_len=128,
                n_layers=2,
                tt_data_parallel=2,
            )
        )

    assert [lane.cleanup_calls for lane in built_lanes] == [1]


class _RecordingAdapter:
    def __init__(self):
        self.calls = []
        self.resolved = object()

    def normalize_prefill(self, args, kwargs):
        self.calls.append(("normalize_prefill", args, kwargs))
        return {"normalized": "prefill"}

    def normalize_decode(self, args, kwargs):
        self.calls.append(("normalize_decode", args, kwargs))
        return {"normalized": "decode"}

    def resolve_legacy_kv_cache_config(self, shape, dtype, num_layers):
        self.calls.append(("resolve_kv", shape, dtype, num_layers))
        return self.resolved


class _RecordingTarget:
    def __init__(self):
        self.model = object()
        self.model_args = object()
        self.mesh_device = object()
        self.cache_path = "cache"
        self.already_warmed_up_prefill = False
        self.calls = []

    def _record(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        return method

    def configure_paged_kv_cache(self, *args, **kwargs):
        return self._record("configure_paged_kv_cache", *args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return self._record("allocate_kv_cache", *args, **kwargs)

    def compile_prefill(self, *args, **kwargs):
        return self._record("compile_prefill", *args, **kwargs)

    def compile_decode(self, *args, **kwargs):
        return self._record("compile_decode", *args, **kwargs)

    def prefill_forward(self, *args, **kwargs):
        return self._record("prefill_forward", *args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return self._record("decode_forward", *args, **kwargs)

    def read_decode_output(self, *args, **kwargs):
        return self._record("read_decode_output", *args, **kwargs)

    def process_decode_output_host(self, *args, **kwargs):
        return self._record("process_decode_output_host", *args, **kwargs)

    def warmup_model_prefill(self, *args, **kwargs):
        return self._record("warmup_model_prefill", *args, **kwargs)

    def warmup_model_decode(self, *args, **kwargs):
        return self._record("warmup_model_decode", *args, **kwargs)

    def cleanup(self, *args, **kwargs):
        return self._record("cleanup", *args, **kwargs)


class _OtherRecordingTarget(_RecordingTarget):
    pass


@pytest.mark.parametrize("target_type", [_RecordingTarget, _OtherRecordingTarget])
def test_generator_delegates_the_shared_surface_without_target_type_assumptions(target_type):
    target = target_type()
    adapter = _RecordingAdapter()
    generator = llama_generator.Llama3Generator(target, adapter)

    assert generator.model is target.model
    assert generator.model_args is target.model_args
    assert generator.mesh_device is target.mesh_device
    assert generator.cache_path == "cache"
    generator.already_warmed_up_prefill = True
    assert target.already_warmed_up_prefill is True

    assert generator.compile_prefill("tokens", "page-table", ignored=True) == "compile_prefill"
    assert generator.compile_decode("tokens", "positions", "page-table") == "compile_decode"
    assert generator.prefill_forward("tokens", "page-table") == "prefill_forward"
    assert generator.decode_forward("tokens", "positions", "page-table") == "decode_forward"
    assert generator.read_decode_output("raw", async_read=True) == "read_decode_output"
    assert generator.process_decode_output_host("host", is_tokens=True) == "process_decode_output_host"
    assert generator.warmup_model_prefill(kv_cache="cache") == "warmup_model_prefill"
    assert generator.warmup_model_decode(kv_cache="cache") == "warmup_model_decode"
    assert generator.cleanup() == "cleanup"

    assert [call[0] for call in adapter.calls] == [
        "normalize_prefill",
        "normalize_decode",
        "normalize_prefill",
        "normalize_decode",
    ]
    normalized_calls = [
        call
        for call in target.calls
        if call[0] in {"compile_prefill", "compile_decode", "prefill_forward", "decode_forward"}
    ]
    assert [call[2]["normalized"] for call in normalized_calls] == [
        "prefill",
        "decode",
        "prefill",
        "decode",
    ]


class _ContractAdapter:
    def __init__(self):
        self.calls = []

    def normalize_prefill(self, args, kwargs):
        self.calls.append("prefill")
        return _bind_contract_call(args, kwargs, ("tokens", "page_table"))

    def normalize_decode(self, args, kwargs):
        self.calls.append("decode")
        return _bind_contract_call(args, kwargs, ("tokens", "start_pos", "page_table"))


class _ContractKVManager:
    def __init__(self, config):
        self.config = config
        self.bound_context = object()
        self.bound_cache = object()
        self.released = False

    def validate_borrowed_handle(self, handle):
        assert handle is self.bound_cache

    def release(self):
        self.released = True


class _ContractGraphCompiler:
    def __init__(self):
        self.keys = []
        self.cleaned_up = False

    def compile(self, key, invocation, **kwargs):
        self.keys.append(key)
        invocation(object())

    def cleanup(self):
        self.cleaned_up = True


class _ContractOutputReader:
    def __init__(self):
        self.drained = False

    def read(self, value, *, blocking):
        assert blocking is True
        return value

    def drain(self):
        self.drained = True


def _bind_contract_call(args, kwargs, names):
    bound = dict(kwargs)
    for name, value in zip(names, args):
        assert name not in bound
        bound[name] = value
    return bound


def _make_contract_executor(max_batch_size, mesh_device):
    executor = object.__new__(LLMExecutor)
    executor.model = SimpleNamespace(
        config=SimpleNamespace(max_batch_size=max_batch_size, max_seq_len=128),
        vocab_size=4,
        num_devices=1,
    )
    executor.runtime_config = SimpleNamespace(model_cache_path="host-cache")
    executor.model_args = executor.runtime_config
    executor.config = LLMExecutorConfig(
        trace=TraceConfig(mode="none"),
        warmup=WarmupConfig(),
        paged_kv_cache=PagedKVCacheConfig(
            block_size=32,
            max_num_blocks=4,
            num_blocks=2,
            dtype=ttnn.bfloat8_b,
        ),
        device_sampling_enabled=False,
    )
    executor.mesh_device = mesh_device
    executor.cache_path = "host-cache"
    executor.mode = None
    executor.already_warmed_up_prefill = False
    executor.device_decode_feedback_enabled = False
    executor._kv_manager = _ContractKVManager(executor.config.paged_kv_cache)
    executor._graph_compiler = _ContractGraphCompiler()
    executor._output_reader = _ContractOutputReader()
    executor._trace_plans = {}
    executor._previous_decode_page_table = None
    executor._external_by_raw_id = {}
    executor._external_by_host_id = {}
    executor._transient_orphans = []
    executor._terminal = False
    executor._cleaned_up = False
    executor._sampling_buffers_loaded = False

    def plan_prefill(tokens, page_table, prompt_lens, empty_slots, start_pos):
        batch_size = int(tokens.shape[0])
        request = SimpleNamespace(
            kind="batched",
            source_rows=tuple(range(batch_size)),
            slots=tuple(range(batch_size)),
            tokens=tokens,
            page_table=page_table,
            cached_tokens=(0,) * batch_size,
            last_token_indices=(0,) * batch_size,
            padded_batch_size=batch_size,
            trace_eligible=False,
        )
        request.graph_key = lambda sampling_path: ("prefill", batch_size, sampling_path)
        return (request,)

    def prefill_output(request, sampling_params, sampling_path, *, enable_trace):
        assert enable_trace is False
        logits = torch.zeros(1, 1, len(request.source_rows), executor.model.vocab_size)
        for row, token in enumerate(request.tokens[:, 0]):
            logits[0, 0, row] = token + 10
        return logits, None

    def decode_output(tokens, *args, **kwargs):
        logits = torch.zeros(1, 1, len(tokens), executor.model.vocab_size)
        for row, token in enumerate(tokens):
            logits[0, 0, row] = token + 20
        return (logits, None), None

    executor._validation_context = lambda mode: contextlib.nullcontext()
    executor._plan_prefill = plan_prefill
    executor._compile_prefill_invocation = lambda request, sampling_params: None
    executor._compile_decode_invocation = lambda *args: None
    executor._make_decode_trace_plan = lambda *args: object()
    executor._execute_prefill_request = prefill_output
    executor._execute_decode_request = decode_output
    return executor


@pytest.mark.parametrize(
    ("target_kind", "target_type"),
    [("direct", LLMExecutor), ("lane_group", LaneGroupExecutor)],
)
def test_generator_runs_the_same_contract_against_actual_execution_targets(monkeypatch, target_kind, target_type):
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh_device: None)
    mesh_device = SimpleNamespace(shape=(1, 1))
    if target_kind == "direct":
        lanes = [_make_contract_executor(2, mesh_device)]
        target = lanes[0]
    else:
        lanes = [_make_contract_executor(1, mesh_device) for _ in range(2)]
        target = LaneGroupExecutor(lanes, mesh_device=mesh_device)

    assert type(target) is target_type
    adapter = _ContractAdapter()
    generator = llama_generator.Llama3Generator(target, adapter)
    tokens = torch.tensor([[1], [2]], dtype=torch.long)
    decode_tokens = torch.tensor([3, 4], dtype=torch.long)
    start_pos = torch.tensor([0, 0], dtype=torch.long)
    page_table = torch.zeros((2, 1), dtype=torch.int32)

    try:
        assert generator.mesh_device is mesh_device
        assert generator.cache_path == "host-cache"
        generator.already_warmed_up_prefill = True
        assert all(lane.already_warmed_up_prefill for lane in lanes)

        assert generator.compile_prefill(tokens, page_table) is None
        assert generator.compile_decode(decode_tokens, start_pos, page_table) is None
        assert all(lane._graph_compiler.keys == [] for lane in lanes)
        prefill = generator.prefill_forward(tokens, page_table)
        decode, log_probs = generator.decode_forward(decode_tokens, start_pos, page_table)

        torch.testing.assert_close(prefill[:, 0, 0], torch.tensor([11.0, 12.0]))
        torch.testing.assert_close(decode[:, 0, 0], torch.tensor([23.0, 24.0]))
        assert log_probs is None
        assert adapter.calls == ["prefill", "decode", "prefill", "decode"]
    finally:
        generator.cleanup()

    assert all(lane.terminal for lane in lanes)
    assert all(lane._graph_compiler.cleaned_up for lane in lanes)
    assert all(lane._output_reader.drained for lane in lanes)
    assert all(lane._kv_manager.released for lane in lanes)


def test_generator_resolves_vllm_kv_policy_before_target_allocation():
    target = _RecordingTarget()
    adapter = _RecordingAdapter()
    generator = llama_generator.Llama3Generator(target, adapter)

    assert generator.allocate_kv_cache((64, 8, 32, 128), torch.bfloat16, 32) == "allocate_kv_cache"

    assert adapter.calls == [
        ("resolve_kv", (64, 8, 32, 128), torch.bfloat16, 32),
    ]
    assert target.calls == [
        ("configure_paged_kv_cache", (adapter.resolved,), {}),
        ("allocate_kv_cache", (), {}),
    ]


def test_direct_demo_builds_a_pre_resolved_immutable_executor_config(monkeypatch, expect_error):
    attention_config = SimpleNamespace(
        paged_attention_config=SimpleNamespace(block_size=32, max_num_blocks=64),
        kv_cache_dtype=ttnn.bfloat8_b,
    )
    llm = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(block_configs=[SimpleNamespace(attention_config=attention_config)])
        ),
        runtime_config=object(),
    )
    captured = {}
    sentinel = object()

    def fake_builder(model_product, config):
        captured["llm"] = model_product
        captured["config"] = config
        return sentinel

    monkeypatch.setattr(llama_demo, "build_llama3_executor", fake_builder)

    assert (
        llama_demo._build_demo_executor(
            llm,
            trace_mode="all",
            device_sampling_enabled=True,
            include_decode_top_k=True,
        )
        is sentinel
    )

    config = captured["config"]
    assert captured["llm"] is llm
    assert config.trace.mode == "all"
    assert config.device_sampling_enabled is True
    assert config.warmup.include_decode_top_k is True
    assert config.paged_kv_cache.num_blocks == 64
    assert config.paged_kv_cache.max_num_blocks == 64
    with expect_error(FrozenInstanceError, ""):
        config.device_sampling_enabled = False


def test_demo_warmup_compiles_prefill_then_decode_with_static_policy():
    calls = []
    executor = SimpleNamespace(
        config=SimpleNamespace(
            trace=TraceConfig(mode="all"),
            device_sampling_enabled=True,
        ),
        model=SimpleNamespace(config=SimpleNamespace(max_batch_size=4)),
        warmup_model_prefill=lambda **kwargs: calls.append(("prefill", kwargs)),
        warmup_model_decode=lambda **kwargs: calls.append(("decode", kwargs)),
    )
    kv_cache = object()
    page_table = torch.zeros((4, 7), dtype=torch.int32)

    llama_demo._warmup_demo_executor(
        executor,
        kv_cache=kv_cache,
        page_table=page_table,
    )

    assert calls == [
        (
            "prefill",
            {
                "kv_cache": kv_cache,
                "enable_trace": False,
                "can_sample_on_device": True,
            },
        ),
        (
            "decode",
            {
                "kv_cache": kv_cache,
                "enable_trace": False,
                "max_batch_size": 4,
                "num_blocks": 7,
                "can_sample_on_device": True,
            },
        ),
        (
            "prefill",
            {
                "kv_cache": kv_cache,
                "enable_trace": True,
                "can_sample_on_device": True,
            },
        ),
        (
            "decode",
            {
                "kv_cache": kv_cache,
                "enable_trace": True,
                "max_batch_size": 4,
                "num_blocks": 7,
                "can_sample_on_device": True,
            },
        ),
    ]


def test_perf_demo_warms_up_and_cleans_up_before_optional_accuracy_executor(
    monkeypatch,
):
    events = []
    build_kwargs = []
    kv_cache = object()
    executor = SimpleNamespace(
        paged_kv_cache_config=SimpleNamespace(num_blocks=4),
        allocate_kv_cache=lambda: events.append("allocate") or kv_cache,
        cleanup=lambda: events.append("cleanup"),
    )
    llm = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(max_batch_size=1),
            supports_on_device_sampling=True,
        ),
        tokenizer=object(),
    )

    monkeypatch.setenv("SAMPLING_MODE", "on_device_topk")
    monkeypatch.setenv("LLAMA3_8B_TTTV2_SKIP_PERF_TEACHER_FORCING", "0")
    monkeypatch.setattr(
        llama_demo,
        "_build_demo_executor",
        lambda *args, **kwargs: build_kwargs.append(kwargs) or executor,
    )
    monkeypatch.setattr(
        llama_demo,
        "_warmup_demo_executor",
        lambda *args, **kwargs: events.append("warmup"),
    )
    monkeypatch.setattr(
        llama_demo,
        "load_input_prompts",
        lambda *args, **kwargs: ["prompt"],
    )
    monkeypatch.setattr(
        llama_demo,
        "preprocess_llama3_8b_chat_prompts",
        lambda *args, **kwargs: (
            torch.zeros((1, 128), dtype=torch.long),
            torch.tensor([128]),
        ),
    )
    monkeypatch.setattr(
        llama_demo,
        "run_perf_benchmark",
        lambda *args, **kwargs: events.append("benchmark")
        or SimpleNamespace(
            ttft_ms=1.0,
            tok_s_u=1.0,
            tok_s=1.0,
            decode_latency_mean_ms=1.0,
            generated_token_ids=[[1]],
        ),
    )
    monkeypatch.setattr(llama_demo, "log_generated_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        llama_demo,
        "_measure_teacher_forcing_accuracy",
        lambda *args, **kwargs: events.append("teacher_forcing") or (99.0, 100.0),
    )

    llama_demo._run_perf_benchmark(
        llm,
        mesh_device=object(),
        expected={},
        batch_size=1,
        case_name="test",
    )

    assert build_kwargs == [
        {
            "trace_mode": "all",
            "device_sampling_enabled": True,
            "include_decode_top_k": True,
        }
    ]
    assert events == ["allocate", "warmup", "benchmark", "cleanup", "teacher_forcing"]
