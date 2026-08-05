# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Sequence
from unittest.mock import create_autospec

import pytest
import torch

import ttnn
from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from models.common.llm_runtime.execution import EagerExecutor, TracedExecutor
from models.common.llm_runtime.lane_group import LaneGroupExecutor
from models.common.models.llama3_8b import executor as llama_executor
from models.common.models.llama3_8b import generator as llama_generator
from models.common.models.llama3_8b import model as llama_model
from models.common.tests.demos.llama3_8b import demo as llama_demo


class _Mesh:
    shape = (1, 1)

    @staticmethod
    def get_num_devices():
        return 1


def _model(*, max_batch_size=4, max_seq_len=4096):
    paged = SimpleNamespace(block_size=32, max_num_blocks=132)
    attention = SimpleNamespace(
        n_kv_heads=8,
        head_dim=128,
        kv_cache_dtype=ttnn.bfloat8_b,
        paged_attention_config=paged,
        use_vllm_paged_kv_cache=True,
        kv_cache=None,
    )
    attention_module = SimpleNamespace(config=attention, kv_cache=None)
    model = SimpleNamespace(
        config=SimpleNamespace(
            mesh_device=_Mesh(),
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=1,
            num_devices=1,
            block_configs=(SimpleNamespace(attention_config=attention),),
        ),
        layers=(SimpleNamespace(attention=attention_module),),
        iter_executor_named_modules=lambda: (),
        vocab_size=128,
        num_devices=1,
    )

    def configure_paged_attention(*, block_size, max_num_blocks):
        assert attention.kv_cache is None
        assert attention_module.kv_cache is None
        attention.paged_attention_config = SimpleNamespace(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )

    model.configure_paged_attention = configure_paged_attention
    return model


def _runtime_config():
    return SimpleNamespace(
        model_cache_path="cache",
        max_prefill_chunk_size=2048,
        trace_prefill_supported_seq_lens=(128, 1024),
        can_enable_trace=lambda sequence_length, num_cached_tokens=0: (
            num_cached_tokens == 0 and sequence_length in (128, 1024)
        ),
    )


def _config(mode="none", *, num_blocks=None):
    return llama_executor.Llama3ExecutorConfig(
        trace=TraceConfig(mode),
        warmup=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
        paged_kv_cache=PagedKVCacheConfig(
            block_size=32,
            max_num_blocks=132,
            dtype=ttnn.bfloat8_b,
            num_blocks=num_blocks,
        ),
        device_sampling_enabled=False,
    )


@pytest.mark.parametrize("mode", ["none", "decode_only", "all"])
def test_model_owned_executor_constructs_exact_composition(mode):
    executor = llama_executor.Llama3Executor(_model(), _runtime_config(), _config(mode))

    assert executor.eager_executor.program_compiler is executor.program_compiler
    assert executor.eager_executor.prefill is executor.prefill_runtime
    assert executor.eager_executor.decode is executor.decode_runtime
    assert executor.warmup.eager is executor.eager_executor
    assert executor.warmup.trace_compiler is executor.trace_compiler
    if mode == "none":
        assert executor.warmup.execution is executor.eager_executor
        assert executor.eager_execution is executor.eager_executor
        assert executor.traced_prefill_execution is None
        assert executor.traced_decode_execution is None
        assert executor.trace_compiler is None
        assert executor.traced_executor is None
    else:
        assert executor.warmup.execution is executor.traced_executor
        expected_prefill = executor.traced_executor if mode == "all" else None
        assert executor.eager_execution is executor.eager_executor
        assert executor.traced_prefill_execution is expected_prefill
        assert executor.traced_decode_execution is executor.traced_executor
        assert executor.traced_executor.eager_executor is executor.eager_executor
        assert executor.traced_executor.trace_compiler is executor.trace_compiler
        assert executor.trace_compiler.program_compiler is executor.program_compiler


@pytest.mark.parametrize(
    ("method_name", "positional_names", "keyword_only_names"),
    [
        (
            "compile_prefill",
            ("self",),
            (
                "tokens",
                "page_table",
                "prompt_lens",
                "start_pos",
                "empty_slots",
                "kv_cache",
                "sampling_params",
                "execution",
            ),
        ),
        (
            "compile_decode",
            ("self",),
            (
                "tokens",
                "start_pos",
                "page_table",
                "kv_cache",
                "sampling_params",
                "reset_batch",
                "execution",
            ),
        ),
        (
            "prefill_forward",
            ("self", "tokens", "page_table"),
            (
                "prompt_lens",
                "start_pos",
                "empty_slots",
                "kv_cache",
                "sampling_params",
                "execution",
            ),
        ),
        (
            "decode_forward",
            ("self", "tokens", "start_pos", "page_table"),
            ("kv_cache", "sampling_params", "reset_batch", "read_from_device", "execution"),
        ),
        ("read_decode_output", ("self", "tt_out"), ("async_read",)),
        ("process_decode_output_host", ("self", "tt_out"), ("is_tokens",)),
        (
            "can_trace_prefill",
            ("self",),
            ("tokens", "prompt_lens", "start_pos", "empty_slots"),
        ),
        (
            "warmup_model_prefill",
            ("self",),
            ("kv_cache", "can_sample_on_device", "enable_trace"),
        ),
        (
            "warmup_model_decode",
            ("self",),
            ("kv_cache", "max_batch_size", "num_blocks", "can_sample_on_device", "enable_trace"),
        ),
    ],
)
def test_model_owned_executor_has_exact_call_contract(method_name, positional_names, keyword_only_names):
    signature = inspect.signature(getattr(llama_executor.Llama3Executor, method_name))
    parameters = signature.parameters
    required_names = {
        "compile_prefill": {"tokens", "page_table"},
        "compile_decode": {"tokens", "start_pos", "page_table"},
        "prefill_forward": {"tokens", "page_table"},
        "decode_forward": {"tokens", "start_pos", "page_table"},
        "read_decode_output": {"tt_out"},
        "process_decode_output_host": {"tt_out"},
        "can_trace_prefill": {"tokens"},
        "warmup_model_prefill": {"kv_cache", "can_sample_on_device", "enable_trace"},
        "warmup_model_decode": {
            "kv_cache",
            "max_batch_size",
            "num_blocks",
            "can_sample_on_device",
            "enable_trace",
        },
    }[method_name]
    special_defaults = {
        "reset_batch": False,
        "read_from_device": True,
        "async_read": False,
        "is_tokens": False,
    }

    assert tuple(parameters) == positional_names + keyword_only_names
    assert all(parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for name in positional_names)
    assert all(parameters[name].kind is inspect.Parameter.KEYWORD_ONLY for name in keyword_only_names)
    for name, parameter in tuple(parameters.items())[1:]:
        expected_default = inspect.Parameter.empty if name in required_names else special_defaults.get(name)
        assert parameter.default == expected_default
        assert parameter.annotation is not inspect.Parameter.empty
    assert signature.return_annotation is not inspect.Signature.empty
    if "execution" in parameters:
        assert parameters["execution"].annotation == "EagerExecutor | TracedExecutor | None"


def test_model_owned_executor_validates_cache_then_omits_it_from_execution():
    execution = create_autospec(EagerExecutor, instance=True)
    execution.prefill_forward.return_value = "prefill"
    execution.decode_forward.return_value = "decode"
    executor = object.__new__(llama_executor.Llama3Executor)
    executor._prefill_execution = execution
    executor._decode_execution = execution
    executor._ensure_active = lambda: None
    validated_caches = []
    sampling_values = []
    executor._validate_bound_cache = validated_caches.append
    executor._ensure_sampling_for = sampling_values.append

    tokens = torch.zeros((1, 4), dtype=torch.long)
    start_pos = torch.zeros((1,), dtype=torch.long)
    page_table = torch.zeros((1, 1), dtype=torch.int32)
    prompt_lens = torch.full((1,), 4, dtype=torch.long)
    empty_slots = [0]
    kv_cache = object()
    sampling_params = object()

    executor.compile_prefill(
        tokens=tokens,
        page_table=page_table,
        prompt_lens=prompt_lens,
        start_pos=start_pos,
        empty_slots=empty_slots,
        kv_cache=kv_cache,
        sampling_params=sampling_params,
    )
    executor.compile_decode(
        tokens=tokens,
        start_pos=start_pos,
        page_table=page_table,
        kv_cache=kv_cache,
        sampling_params=sampling_params,
        reset_batch=True,
    )
    assert (
        executor.prefill_forward(
            tokens,
            page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )
        == "prefill"
    )
    assert (
        executor.decode_forward(
            tokens,
            start_pos,
            page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            reset_batch=True,
            read_from_device=False,
        )
        == "decode"
    )

    assert validated_caches == [kv_cache] * 4
    assert sampling_values == [sampling_params] * 4
    for target, expected_names in (
        (
            execution.compile_prefill,
            ("tokens", "page_table", "prompt_lens", "start_pos", "empty_slots", "sampling_params"),
        ),
        (
            execution.compile_decode,
            ("tokens", "start_pos", "page_table", "sampling_params", "reset_batch"),
        ),
        (
            execution.prefill_forward,
            ("tokens", "page_table", "prompt_lens", "start_pos", "empty_slots", "sampling_params"),
        ),
        (
            execution.decode_forward,
            ("tokens", "start_pos", "page_table", "sampling_params", "reset_batch", "read_from_device"),
        ),
    ):
        assert target.call_count == 1
        assert tuple(target.call_args.kwargs) == expected_names


def test_model_owned_executor_trace_output_and_warmup_forwarding_is_named():
    calls = []

    class _PrefillRuntime:
        def can_trace(
            self,
            *,
            tokens: torch.Tensor,  # ↓ Core request
            prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
            start_pos: torch.Tensor | None = None,
        ) -> bool:
            calls.append(("can_trace", tokens, prompt_lens, start_pos))
            return True

    class _DecodeRuntime:
        def read_decode_output(self, tt_out: Any, *, async_read: bool = False) -> Any:
            calls.append(("read_decode_output", tt_out, async_read))
            return "read"

        def process_decode_output_host(self, tt_out: Any, *, is_tokens: bool = False) -> tuple[Any, Any]:
            calls.append(("process_decode_output_host", tt_out, is_tokens))
            return "processed", "event"

    class _Warmup:
        def warmup_prefill(
            self,
            *,
            kv_cache: Any,  # ↓ Borrowed resources
            can_sample_on_device: bool,  # ↓ Execution policy
            enable_trace: bool,
        ) -> None:
            calls.append(("warmup_prefill", kv_cache, can_sample_on_device, enable_trace))

        def warmup_decode(
            self,
            *,
            kv_cache: Any,  # ↓ Borrowed resources
            max_batch_size: int,  # ↓ Coverage dimensions
            num_blocks: int,
            can_sample_on_device: bool,  # ↓ Execution policy
            enable_trace: bool,
        ) -> None:
            calls.append(("warmup_decode", kv_cache, max_batch_size, num_blocks, can_sample_on_device, enable_trace))

    executor = object.__new__(llama_executor.Llama3Executor)
    executor.traced_executor = object()
    executor.config = SimpleNamespace(trace=SimpleNamespace(prefill_enabled=True))
    executor.prefill_runtime = _PrefillRuntime()
    executor.decode_runtime = _DecodeRuntime()
    executor.warmup = _Warmup()
    executor._ensure_active = lambda: None

    tokens = torch.zeros((1, 4), dtype=torch.long)
    prompt_lens = torch.full((1,), 4, dtype=torch.long)
    start_pos = torch.zeros((1,), dtype=torch.long)
    kv_cache = object()

    assert executor.can_trace_prefill(
        tokens=tokens,
        prompt_lens=prompt_lens,
        start_pos=start_pos,
        empty_slots=[0],
    )
    assert executor.read_decode_output("device-output", async_read=True) == "read"
    assert executor.process_decode_output_host("host-output", is_tokens=True) == ("processed", "event")
    executor.warmup_model_prefill(
        kv_cache=kv_cache,
        can_sample_on_device=True,
        enable_trace=False,
    )
    executor.warmup_model_decode(
        kv_cache=kv_cache,
        max_batch_size=4,
        num_blocks=8,
        can_sample_on_device=True,
        enable_trace=False,
    )

    assert calls == [
        ("can_trace", tokens, prompt_lens, start_pos),
        ("read_decode_output", "device-output", True),
        ("process_decode_output_host", "host-output", True),
        ("warmup_prefill", kv_cache, True, False),
        ("warmup_decode", kv_cache, 4, 8, True, False),
    ]


def test_vllm_capacity_resolution_reconfigures_existing_runtime_owners_before_allocation(monkeypatch):
    executor = llama_executor.Llama3Executor(_model(), _runtime_config(), _config())
    owner_ids = tuple(
        id(owner)
        for owner in (
            executor.prefill_runtime,
            executor.decode_runtime,
            executor.warmup,
            executor.program_compiler,
        )
    )
    assert executor.page_table_layout.raw_capacity_width == 128

    executor.configure_paged_kv_cache(
        PagedKVCacheConfig(
            block_size=16,
            max_num_blocks=200,
            dtype=ttnn.bfloat8_b,
            num_blocks=200,
        )
    )

    assert (
        tuple(
            id(owner)
            for owner in (
                executor.prefill_runtime,
                executor.decode_runtime,
                executor.warmup,
                executor.program_compiler,
            )
        )
        == owner_ids
    )
    assert executor.config.paged_kv_cache is executor.kv_cache_manager.config
    assert executor.kv_cache_manager.config.block_size == 16
    assert executor.kv_cache_manager.config.max_num_blocks == executor.kv_cache_manager.config.num_blocks == 200
    assert executor.model.layers[0].attention.config.paged_attention_config.block_size == 16
    assert executor.model.layers[0].attention.config.paged_attention_config.max_num_blocks == 200
    assert executor.page_table_layout.block_size == 16
    assert executor.page_table_layout.raw_capacity_width == 200
    assert executor.prefill_runtime.config.page_table_layout is executor.page_table_layout
    assert executor.decode_runtime.config.page_table_layout is executor.page_table_layout
    assert executor.warmup.config.page_table_layout is executor.page_table_layout
    assert (
        executor.prefill_runtime.config.max_page_table_capacity_width
        == executor.decode_runtime.config.max_page_table_capacity_width
        == executor.warmup.config.max_page_table_capacity_width
        == executor.page_table_layout.raw_capacity_width
    )
    assert (
        executor.prefill_runtime.config.max_prefill_page_table_width
        == executor.warmup.config.max_prefill_page_table_width
        == executor.page_table_layout.prefill_width
    )
    assert (
        executor.prefill_runtime.config.max_decode_page_table_width
        == executor.decode_runtime.config.max_decode_page_table_width
        == executor.warmup.config.max_decode_page_table_width
        == executor.page_table_layout.decode_width
    )

    def fake_allocate():
        assert executor._runtime_configuration_sealed
        assert executor.warmup._configuration_sealed
        return ["allocated"]

    monkeypatch.setattr(executor.kv_cache_manager, "allocate", fake_allocate)
    assert executor.allocate_kv_cache() == ["allocated"]


def test_model_reconfigures_construction_and_live_attention_without_allocating():
    construction_paged = llama_model.Llama31_8BPagedAttentionConfig(block_size=32, max_num_blocks=132)
    live_paged = llama_model.Llama31_8BPagedAttentionConfig(block_size=32, max_num_blocks=132)
    construction_attention = SimpleNamespace(
        use_vllm_paged_kv_cache=True,
        paged_attention_config=construction_paged,
    )
    live_attention = SimpleNamespace(
        use_vllm_paged_kv_cache=True,
        paged_attention_config=live_paged,
        kv_cache=None,
    )
    model = object.__new__(llama_model.Llama3Transformer1D)
    model.config = SimpleNamespace(
        block_configs=(SimpleNamespace(attention_config=construction_attention),),
    )
    model.layers = (SimpleNamespace(attention=SimpleNamespace(config=live_attention, kv_cache=None)),)

    model.configure_paged_attention(block_size=16, max_num_blocks=200)

    assert construction_attention.paged_attention_config.block_size == 16
    assert construction_attention.paged_attention_config.max_num_blocks == 200
    assert live_attention.paged_attention_config.block_size == 16
    assert live_attention.paged_attention_config.max_num_blocks == 200
    assert live_attention.kv_cache is None
    assert model.layers[0].attention.kv_cache is None


def test_late_vllm_capacity_resolution_fails_before_mutating_kv_configuration(expect_error):
    executor = llama_executor.Llama3Executor(_model(), _runtime_config(), _config())
    executor._seal_runtime_configuration()
    unresolved = executor.kv_cache_manager.config

    with expect_error(RuntimeError, "runtime configuration is sealed"):
        executor.configure_paged_kv_cache(
            PagedKVCacheConfig(
                block_size=32,
                max_num_blocks=132,
                dtype=ttnn.bfloat8_b,
                num_blocks=64,
            )
        )

    assert executor.kv_cache_manager.config is unresolved
    assert not executor.kv_cache_manager.config.is_resolved()


def test_direct_demo_resolves_physical_capacity_to_configured_maximum(monkeypatch):
    attention = SimpleNamespace(
        paged_attention_config=SimpleNamespace(block_size=32, max_num_blocks=128),
        kv_cache_dtype=ttnn.bfloat8_b,
    )
    llm = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(block_configs=(SimpleNamespace(attention_config=attention),)))
    )
    captured = []
    monkeypatch.setattr(
        llama_demo, "build_llama3_executor", lambda product, config: captured.append(config) or object()
    )

    llama_demo._build_demo_executor(llm, trace_mode="all", device_sampling_enabled=False)

    assert captured[0].paged_kv_cache.num_blocks == captured[0].paged_kv_cache.max_num_blocks == 128


def test_model_owned_cleanup_is_ordered_best_effort_retryable_and_idempotent(expect_error):
    calls = []
    failures = {"reader", "trace"}

    class _Owner:
        def __init__(self, name):
            self.name = name

        def cleanup(self):
            calls.append(self.name)
            if self.name in failures:
                raise RuntimeError(self.name)

        drain = cleanup
        drain_external_outputs = cleanup
        cleanup_transients = cleanup
        release = cleanup

    executor = object.__new__(llama_executor.Llama3Executor)
    executor._terminal = False
    executor._cleaned_up = False
    executor.decode_runtime = _Owner("decode-external")
    executor.output_reader = _Owner("reader")
    executor.prefill_runtime = _Owner("prefill")
    executor.trace_compiler = _Owner("trace")
    executor.program_compiler = _Owner("program")
    executor.config = SimpleNamespace(device_sampling_enabled=True)
    executor.model = SimpleNamespace(sampling=_Owner("sampling"))
    executor.kv_cache_manager = _Owner("kv")

    with expect_error(RuntimeError, "reader") as raised:
        executor.cleanup()

    expected_order = [
        "decode-external",
        "reader",
        "prefill",
        "decode-external",
        "trace",
        "program",
        "sampling",
        "kv",
    ]
    assert calls == expected_order
    assert tuple(error.args[0] for error in raised.value.cleanup_failures) == ("trace",)
    assert executor.terminal
    assert not executor._cleaned_up

    failures.clear()
    executor.cleanup()
    assert calls == expected_order * 2
    assert executor._cleaned_up

    executor.cleanup()
    assert calls == expected_order * 2


def test_build_llama3_executor_uses_prebuilt_product(monkeypatch):
    sentinel = object()
    calls = []
    llm = SimpleNamespace(model=object(), runtime_config=object())
    monkeypatch.setattr(
        llama_executor,
        "Llama3Executor",
        lambda model, runtime_config, config: calls.append((model, runtime_config, config)) or sentinel,
    )

    config = object()
    assert llama_executor.build_llama3_executor(llm, config) is sentinel
    assert calls == [(llm.model, llm.runtime_config, config)]


def test_configured_path_has_no_legacy_or_common_aggregate_surface():
    source = inspect.getsource(llama_executor)
    assert not hasattr(llama_executor, "EagerLlamaExecutor")
    assert not hasattr(llama_executor, "TracedLlamaExecutor")
    assert "models.common.models.executor" not in source
    assert "llm_runtime.executor" not in source
    assert "class LLMExecutor" not in source
    assert EagerExecutor not in TracedExecutor.__mro__


def test_generator_explicitly_accepts_static_trace_mode():
    assert llama_generator.Llama3Generator.model_capabilities["accepts_trace_mode"] is True


def test_initialize_vllm_model_threads_policy(monkeypatch):
    captured = []
    sentinel = object()
    monkeypatch.setattr(
        llama_generator,
        "build_llama3_generator",
        lambda config: captured.append(config) or sentinel,
    )

    result = llama_generator.Llama3Generator.initialize_vllm_model(
        SimpleNamespace(_name_or_path="meta-llama/Llama-3.1-8B-Instruct"),
        object(),
        8,
        4096,
        n_layers=3,
        tt_data_parallel=2,
        optimizations="accuracy",
        trace_mode="decode_only",
        device_sampling_enabled=True,
    )

    assert result is sentinel
    assert captured[0].tt_data_parallel == 2
    assert captured[0].trace_mode == "decode_only"
    assert captured[0].device_sampling_enabled is True


class _FakeLane:
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


def test_generator_constructs_model_owned_lane_configs(monkeypatch):
    executor_calls = []
    monkeypatch.setattr(llama_generator, "_create_submeshes", lambda mesh, dp: [_Mesh(), _Mesh()])

    def fake_from_pretrained(
        mesh_device,
        *,
        hf_model: str | None = None,
        instruct: bool | None = None,
        max_batch_size: int,
        max_seq_len: int,
        optimizations="performance",
        n_layers: int | None = None,
        dtype=ttnn.bfloat8_b,
        paged_attention_config=None,
    ):
        return SimpleNamespace(model=_model(max_batch_size=max_batch_size), runtime_config=_runtime_config())

    def fake_build_executor(llm, config):
        executor_calls.append((llm, config))
        return _FakeLane(llm, config)

    monkeypatch.setattr(llama_generator, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(llama_generator, "build_llama3_executor", fake_build_executor)
    monkeypatch.setattr(llama_generator, "_model_kv_metadata", lambda model: ((ttnn.bfloat8_b,), 1, 8, 128))

    generator = llama_generator.build_llama3_generator(
        llama_generator.Llama3GeneratorConfig(
            hf_model="meta-llama/Llama-3.1-8B-Instruct",
            mesh_device=object(),
            max_batch_size=4,
            max_seq_len=4096,
            n_layers=1,
            tt_data_parallel=2,
            trace_mode="all",
            device_sampling_enabled=True,
        )
    )

    assert isinstance(generator.target, LaneGroupExecutor)
    assert len(executor_calls) == 2
    assert all(isinstance(config, llama_executor.Llama3ExecutorConfig) for _, config in executor_calls)
    assert all(not config.warmup.include_decode_top_k for _, config in executor_calls)
    assert isinstance(generator._adapter.config, llama_generator.VLLMAdapterConfig)
    assert vars(generator._adapter) == {"config": generator._adapter.config}
    assert generator._adapter.config.trace.mode == "all"
    assert generator._adapter.config.expected_num_layers == 1
    assert generator._adapter.config.expected_kv_heads_per_device == 8
    assert generator._adapter.config.expected_head_dim == 128


@pytest.mark.parametrize(
    ("sampling_mode", "sampling_params", "num_devices", "expected"),
    [
        ("on_device_topk", object(), 1, False),
        ("on_device_topk", object(), 2, False),
        ("on_device_topk", object(), 8, True),
        ("on_device", object(), 8, False),
        ("on_device_topk", None, 8, False),
    ],
)
def test_direct_demo_forces_decode_top_k_only_on_t3k(sampling_mode, sampling_params, num_devices, expected):
    assert llama_demo._force_decode_top_k(sampling_mode, sampling_params, num_devices) is expected


class _RecordingTarget:
    model = SimpleNamespace(config=SimpleNamespace(max_batch_size=4))
    model_args = object()
    mesh_device = object()
    cache_path = "cache"
    already_warmed_up_prefill = False
    eager_execution = object()
    traced_prefill_execution = object()
    traced_decode_execution = object()

    def __init__(self, *, traceable_prefill=True):
        self.calls = []
        self.traceable_prefill = traceable_prefill

    def _record(self, name, arguments):
        arguments = {key: value for key, value in arguments.items() if key != "self"}
        self.calls.append((name, (), arguments))
        return name

    def can_trace_prefill(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
    ) -> bool:
        self._record("can_trace_prefill", locals())
        return self.traceable_prefill

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        execution: EagerExecutor | TracedExecutor | None = None,  # ↓ Internal dispatch
    ) -> str:
        return self._record("prefill_forward", locals())

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        reset_batch: bool = False,  # ↓ State transition
        read_from_device: bool = True,  # ↓ Output policy
        execution: EagerExecutor | TracedExecutor | None = None,  # ↓ Internal dispatch
    ) -> str:
        return self._record("decode_forward", locals())

    def cleanup(self) -> str:
        self.calls.append(("cleanup", (), {}))
        return "cleanup"


def _recording_generator(target):
    target.model = _model()
    target.config = _config("all")
    return llama_generator.Llama3Generator(target, adapter=llama_generator._build_vllm_adapter(target))


def test_generator_delegates_without_concrete_type_checks():
    target = _RecordingTarget()
    generator = _recording_generator(target)
    tokens = torch.tensor([[1]], dtype=torch.long)
    start_pos = torch.tensor([0], dtype=torch.long)
    page_table = torch.tensor([[0]], dtype=torch.int32)

    assert generator.prefill_forward(tokens, page_table, enable_trace=True) == "prefill_forward"
    assert generator.decode_forward(tokens, start_pos, page_table, enable_trace=False) == "decode_forward"
    assert generator.cleanup() == "cleanup"
    assert [name for name, _, _ in target.calls] == [
        "can_trace_prefill",
        "prefill_forward",
        "decode_forward",
        "cleanup",
    ]
    assert target.calls[1][2]["execution"] is target.traced_prefill_execution
    assert target.calls[2][2]["execution"] is target.eager_execution


def test_generator_selects_eager_before_trace_ineligible_prefill_enters_execution():
    target = _RecordingTarget(traceable_prefill=False)
    generator = _recording_generator(target)
    tokens = torch.tensor([[1]], dtype=torch.long)
    page_table = torch.tensor([[0]], dtype=torch.int32)

    assert generator.prefill_forward(tokens, page_table, enable_trace=True) == "prefill_forward"
    assert [name for name, _, _ in target.calls] == ["can_trace_prefill", "prefill_forward"]
    assert target.calls[1][2]["execution"] is target.eager_execution


def test_generator_rejects_unavailable_traced_execution(expect_error):
    target = _RecordingTarget()
    target.traced_decode_execution = None
    generator = _recording_generator(target)
    tokens = torch.tensor([1], dtype=torch.long)
    start_pos = torch.tensor([0], dtype=torch.long)
    page_table = torch.tensor([[0]], dtype=torch.int32)

    with expect_error(RuntimeError, "unavailable traced decode execution"):
        generator.decode_forward(tokens, start_pos, page_table, enable_trace=True)

    assert target.calls == []


def test_demo_uses_model_owned_config_and_order_independent_warmup(monkeypatch):
    attention = SimpleNamespace(
        paged_attention_config=SimpleNamespace(block_size=32, max_num_blocks=128),
        kv_cache_dtype=ttnn.bfloat8_b,
    )
    llm = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(block_configs=(SimpleNamespace(attention_config=attention),)))
    )
    captured = []
    monkeypatch.setattr(
        llama_demo, "build_llama3_executor", lambda product, config: captured.append(config) or object()
    )

    llama_demo._build_demo_executor(llm, trace_mode="all", device_sampling_enabled=False)
    assert isinstance(captured[0], llama_executor.Llama3ExecutorConfig)

    calls = []

    def fake_warmup_model_prefill(
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        can_sample_on_device: bool,  # ↓ Execution policy
        enable_trace: bool,
    ) -> None:
        calls.append(("prefill", kv_cache, can_sample_on_device, enable_trace))

    def fake_warmup_model_decode(
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        max_batch_size: int,  # ↓ Coverage dimensions
        num_blocks: int,
        can_sample_on_device: bool,  # ↓ Execution policy
        enable_trace: bool,
    ) -> None:
        calls.append(("decode", kv_cache, max_batch_size, num_blocks, can_sample_on_device, enable_trace))

    executor = SimpleNamespace(
        config=SimpleNamespace(
            trace=TraceConfig("all"),
            device_sampling_enabled=False,
        ),
        model=SimpleNamespace(config=SimpleNamespace(max_batch_size=4)),
        warmup_model_prefill=fake_warmup_model_prefill,
        warmup_model_decode=fake_warmup_model_decode,
    )
    kv_cache = object()
    llama_demo._warmup_demo_executor(executor, kv_cache=kv_cache, page_table=SimpleNamespace(shape=(4, 8)))
    assert calls == [
        ("prefill", kv_cache, False, False),
        ("decode", kv_cache, 4, 8, False, False),
        ("prefill", kv_cache, False, True),
        ("decode", kv_cache, 4, 8, False, True),
    ]
