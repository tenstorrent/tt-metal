# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import ttnn
from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from models.common.llm_runtime.execution import EagerExecutor, TracedExecutor
from models.common.llm_runtime.lane_group import LaneGroupExecutor
from models.common.models.llama3_8b import executor as llama_executor
from models.common.models.llama3_8b import generator as llama_generator
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
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            mesh_device=_Mesh(),
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=1,
            num_devices=1,
            block_configs=(SimpleNamespace(attention_config=attention),),
        ),
        layers=(SimpleNamespace(attention=SimpleNamespace(config=attention)),),
        iter_executor_named_modules=lambda: (),
        vocab_size=128,
        num_devices=1,
    )
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


def _config(mode="none"):
    return llama_executor.Llama3ExecutorConfig(
        trace=TraceConfig(mode),
        warmup=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
        paged_kv_cache=PagedKVCacheConfig(
            block_size=32,
            max_num_blocks=132,
            dtype=ttnn.bfloat8_b,
        ),
        device_sampling_enabled=False,
    )


@pytest.mark.parametrize("mode", ["none", "decode_only", "all"])
def test_model_owned_executor_constructs_exact_composition(mode):
    executor = llama_executor.Llama3Executor(_model(), _runtime_config(), _config(mode))

    assert executor.eager_executor.program_compiler is executor.program_compiler
    assert executor.eager_executor.prefill is executor.prefill_runtime
    assert executor.eager_executor.decode is executor.decode_runtime
    if mode == "none":
        assert executor.execution is executor.eager_executor
        assert executor.trace_compiler is None
        assert executor.traced_executor is None
    else:
        assert executor.execution is executor.traced_executor
        assert executor.traced_executor.eager is executor.eager_executor
        assert executor.traced_executor.trace_compiler is executor.trace_compiler
        assert executor.trace_compiler.program_compiler is executor.program_compiler


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

    def fake_from_pretrained(**kwargs):
        return SimpleNamespace(model=_model(max_batch_size=kwargs["max_batch_size"]), runtime_config=_runtime_config())

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

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs)) or name


def test_generator_delegates_without_concrete_type_checks():
    target = _RecordingTarget()
    adapter = SimpleNamespace(
        normalize_prefill=lambda args, kwargs: {"tokens": args[0], **kwargs},
        normalize_decode=lambda args, kwargs: {
            "tokens": args[0],
            "start_pos": args[1],
            "page_table": args[2],
            **kwargs,
        },
    )
    generator = llama_generator.Llama3Generator(target, adapter=adapter)

    assert generator.prefill_forward("tokens", page_table="pages") == "prefill_forward"
    assert generator.decode_forward("tokens", "positions", "pages") == "decode_forward"
    assert generator.cleanup() == "cleanup"
    assert [name for name, _, _ in target.calls] == ["prefill_forward", "decode_forward", "cleanup"]


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
    executor = SimpleNamespace(
        config=SimpleNamespace(
            trace=TraceConfig("all"),
            device_sampling_enabled=False,
        ),
        model=SimpleNamespace(config=SimpleNamespace(max_batch_size=4)),
        warmup_model_prefill=lambda **kwargs: calls.append(("prefill", kwargs["enable_trace"])),
        warmup_model_decode=lambda **kwargs: calls.append(("decode", kwargs["enable_trace"])),
    )
    llama_demo._warmup_demo_executor(executor, kv_cache=object(), page_table=SimpleNamespace(shape=(4, 8)))
    assert calls == [("prefill", False), ("decode", False), ("prefill", True), ("decode", True)]
