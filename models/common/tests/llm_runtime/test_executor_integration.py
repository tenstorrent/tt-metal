# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Generic host-only executor/generator integration for migrated siblings."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import pytest
import torch

import ttnn
from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from models.common.llm_runtime.execution import EagerExecutor
from models.common.llm_runtime.lane_group import LaneGroupExecutor
from models.common.models.llama32_1b import executor as llama32_executor
from models.common.models.llama32_1b import generator as llama32_generator

EXECUTOR_BINDINGS = {
    "llama32_1b": SimpleNamespace(
        executor_module=llama32_executor,
        executor_class=llama32_executor.Llama32_1BExecutor,
        executor_config_class=llama32_executor.Llama32_1BExecutorConfig,
        generator_module=llama32_generator,
        generator_class=llama32_generator.Llama32_1BGenerator,
        generator_config_class=llama32_generator.Llama32_1BGeneratorConfig,
        build_generator_name="build_llama32_1b_generator",
        build_executor_name="build_llama32_1b_executor",
        make_model=lambda **kwargs: _make_llama32_model(**kwargs),
        make_runtime_config=lambda: _make_llama32_runtime_config(),
        make_executor_config=lambda mode="none": _make_llama32_executor_config(mode),
        make_recording_target=lambda **kwargs: _RecordingTarget(_make_llama32_model(), **kwargs),
        make_product=lambda mesh_device, max_batch_size: _make_llama32_product(mesh_device, max_batch_size),
        make_lane=lambda llm, config: _FakeLane(llm, config),
    ),
}


@pytest.fixture(params=EXECUTOR_BINDINGS.items(), ids=lambda item: item[0])
def binding(request):
    return request.param[1]


class _Mesh:
    shape = (1, 1)

    @staticmethod
    def get_num_devices():
        return 1


def _make_llama32_model(max_batch_size=4):
    paged = SimpleNamespace(block_size=32, max_num_blocks=132)
    attention = SimpleNamespace(
        n_kv_heads=8,
        head_dim=64,
        kv_cache_dtype=ttnn.bfloat8_b,
        paged_attention_config=paged,
        use_vllm_paged_kv_cache=True,
        kv_cache=None,
    )
    live = SimpleNamespace(config=attention, kv_cache=None)
    model = SimpleNamespace(
        config=SimpleNamespace(
            mesh_device=_Mesh(),
            max_batch_size=max_batch_size,
            max_seq_len=4096,
            n_layers=1,
            num_devices=1,
            block_configs=(SimpleNamespace(attention_config=attention),),
        ),
        layers=(SimpleNamespace(attention=live),),
        iter_executor_named_modules=lambda: (),
        vocab_size=128256,
        num_devices=1,
    )

    def configure_paged_attention(*, block_size, max_num_blocks):
        assert attention.kv_cache is None
        assert live.kv_cache is None
        attention.paged_attention_config = SimpleNamespace(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )

    model.configure_paged_attention = configure_paged_attention
    return model


def _make_llama32_runtime_config():
    return SimpleNamespace(
        model_cache_path="cache",
        max_prefill_chunk_size=2048,
        trace_prefill_supported_seq_lens=(128,),
        can_enable_trace=lambda length, num_cached_tokens=0: length == 128,
        supports_batched_prefill=True,
        disable_batched_prefill=False,
        max_prefill_batch_size=32,
        batched_prefill_batched_extract=True,
    )


def _make_llama32_executor_config(mode="none"):
    return llama32_executor.Llama32_1BExecutorConfig(
        trace=TraceConfig(mode),
        warmup=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
        paged_kv_cache=PagedKVCacheConfig(block_size=32, max_num_blocks=132, dtype=ttnn.bfloat8_b),
        device_sampling_enabled=False,
    )


def _make_llama32_product(mesh_device, max_batch_size):
    model = _make_llama32_model(max_batch_size=max_batch_size)
    model.config.mesh_device = mesh_device
    return SimpleNamespace(model=model, runtime_config=_make_llama32_runtime_config())


@pytest.mark.parametrize("mode", ["none", "decode_only", "all"])
def test_model_owned_executor_has_exact_composition_and_owner_counts(binding, mode, monkeypatch):
    owner_names = (
        "PagedKVCacheManager",
        "OutputReader",
        "PrefillRuntime",
        "DecodeRuntime",
        "ProgramCompiler",
        "EagerExecutor",
        "TraceCompiler",
        "TracedExecutor",
        "WarmupCoordinator",
    )
    owner_factories = {}
    for name in owner_names:
        factory = MagicMock(wraps=getattr(binding.executor_module, name))
        monkeypatch.setattr(binding.executor_module, name, factory)
        owner_factories[name] = factory

    executor = binding.executor_class(
        binding.make_model(),
        binding.make_runtime_config(),
        binding.make_executor_config(mode),
    )
    expected_counts = {name: 1 for name in owner_names}
    if mode == "none":
        expected_counts["TraceCompiler"] = 0
        expected_counts["TracedExecutor"] = 0
    assert {name: factory.call_count for name, factory in owner_factories.items()} == expected_counts
    assert executor.eager_executor.program_compiler is executor.program_compiler
    assert executor.eager_executor.prefill is executor.prefill_runtime
    assert executor.eager_executor.decode is executor.decode_runtime
    assert executor.warmup.eager is executor.eager_executor
    assert executor.warmup.trace_compiler is executor.trace_compiler
    assert executor.eager_execution is executor.eager_executor
    if mode == "none":
        assert executor.warmup.execution is executor.eager_executor
        assert executor.trace_compiler is None
        assert executor.traced_executor is None
        assert executor.traced_prefill_execution is None
        assert executor.traced_decode_execution is None
    else:
        assert executor.warmup.execution is executor.traced_executor
        assert executor.traced_executor.eager_executor is executor.eager_executor
        assert executor.traced_executor.trace_compiler is executor.trace_compiler
        assert executor.trace_compiler.program_compiler is executor.program_compiler
        assert (executor.traced_prefill_execution is not None) is (mode == "all")
        assert executor.traced_decode_execution is executor.traced_executor


@pytest.mark.parametrize(
    "method,positional,keyword_only",
    [
        (
            "compile_prefill",
            ["self"],
            [
                "tokens",
                "page_table",
                "prompt_lens",
                "start_pos",
                "empty_slots",
                "kv_cache",
                "sampling_params",
                "execution",
            ],
        ),
        (
            "compile_decode",
            ["self"],
            ["tokens", "start_pos", "page_table", "kv_cache", "sampling_params", "reset_batch", "execution"],
        ),
        (
            "prefill_forward",
            ["self", "tokens", "page_table"],
            ["prompt_lens", "start_pos", "empty_slots", "kv_cache", "sampling_params", "execution"],
        ),
        (
            "decode_forward",
            ["self", "tokens", "start_pos", "page_table"],
            ["kv_cache", "sampling_params", "reset_batch", "read_from_device", "execution"],
        ),
        ("read_decode_output", ["self", "tt_out"], ["async_read"]),
        ("process_decode_output_host", ["self", "tt_out"], ["is_tokens"]),
        ("can_trace_prefill", ["self"], ["tokens", "prompt_lens", "start_pos", "empty_slots"]),
        ("warmup_model_prefill", ["self"], ["kv_cache", "can_sample_on_device", "enable_trace"]),
        (
            "warmup_model_decode",
            ["self"],
            ["kv_cache", "max_batch_size", "num_blocks", "can_sample_on_device", "enable_trace"],
        ),
    ],
)
def test_executor_call_contract(binding, method, positional, keyword_only):
    signature = inspect.signature(getattr(binding.executor_class, method))
    parameters = signature.parameters
    required = {
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
    }[method]
    non_none_defaults = {"reset_batch": False, "read_from_device": True, "async_read": False, "is_tokens": False}

    assert list(parameters) == positional + keyword_only
    assert all(parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for name in positional)
    assert all(parameters[name].kind is inspect.Parameter.KEYWORD_ONLY for name in keyword_only)
    for name, parameter in tuple(parameters.items())[1:]:
        expected_default = inspect.Parameter.empty if name in required else non_none_defaults.get(name)
        assert parameter.default == expected_default
        assert parameter.annotation is not inspect.Parameter.empty
    assert signature.return_annotation is not inspect.Signature.empty


class _RecordingTarget:
    model_args = object()
    mesh_device = object()
    cache_path = "cache"
    already_warmed_up_prefill = False
    eager_execution = object()
    traced_prefill_execution = object()
    traced_decode_execution = object()

    def __init__(self, model, traceable=True):
        self.model = model
        self.traceable = traceable
        self.calls = []

    def can_trace_prefill(self, **kwargs):
        self.calls.append(("can_trace_prefill", kwargs))
        return self.traceable

    def prefill_forward(self, **kwargs):
        self.calls.append(("prefill_forward", kwargs))
        return kwargs["execution"]

    def cleanup(self):
        self.calls.append(("cleanup", {}))


def test_generator_falls_back_to_eager_before_trace_ineligible_prefill(binding):
    target = binding.make_recording_target(traceable=False)
    target.config = binding.make_executor_config("all")
    generator = binding.generator_class(target, binding.generator_module._build_vllm_adapter(target))
    tokens = __import__("torch").tensor([[1]])
    page_table = __import__("torch").tensor([[0]], dtype=__import__("torch").int32)
    assert generator.prefill_forward(tokens, page_table, enable_trace=True) is target.eager_execution
    assert [name for name, _ in target.calls] == ["can_trace_prefill", "prefill_forward"]


def test_executor_validates_borrowed_cache_then_omits_it_from_execution(binding):
    execution = create_autospec(EagerExecutor, instance=True)
    events = []
    execution.compile_prefill.side_effect = lambda **kwargs: events.append("dispatch_compile_prefill")
    execution.compile_decode.side_effect = lambda **kwargs: events.append("dispatch_compile_decode")
    execution.prefill_forward.side_effect = lambda **kwargs: events.append("dispatch_prefill") or "prefill"
    execution.decode_forward.side_effect = lambda **kwargs: events.append("dispatch_decode") or "decode"
    executor = object.__new__(binding.executor_class)
    executor._prefill_execution = execution
    executor._decode_execution = execution
    executor._ensure_active = lambda: None
    executor._validate_bound_cache = lambda cache: events.append(("validate_cache", cache))
    executor._ensure_sampling_for = lambda params: events.append(("validate_sampling", params))

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

    expected_validation = [("validate_cache", kv_cache), ("validate_sampling", sampling_params)]
    assert events == [
        *expected_validation,
        "dispatch_compile_prefill",
        *expected_validation,
        "dispatch_compile_decode",
        *expected_validation,
        "dispatch_prefill",
        *expected_validation,
        "dispatch_decode",
    ]
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
        assert "kv_cache" not in target.call_args.kwargs


def test_late_capacity_reconfigures_existing_owners_before_allocation(binding, monkeypatch):
    executor = binding.executor_class(
        binding.make_model(), binding.make_runtime_config(), binding.make_executor_config()
    )
    owner_ids = tuple(
        id(owner)
        for owner in (executor.prefill_runtime, executor.decode_runtime, executor.warmup, executor.program_compiler)
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
            for owner in (executor.prefill_runtime, executor.decode_runtime, executor.warmup, executor.program_compiler)
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

    def fake_allocate():
        assert executor._runtime_configuration_sealed
        assert executor.warmup._configuration_sealed
        return ["allocated"]

    monkeypatch.setattr(executor.kv_cache_manager, "allocate", fake_allocate)
    assert executor.allocate_kv_cache() == ["allocated"]


def test_late_capacity_failure_is_atomic(binding, expect_error):
    executor = binding.executor_class(
        binding.make_model(), binding.make_runtime_config(), binding.make_executor_config()
    )
    executor._seal_runtime_configuration()
    unresolved = executor.kv_cache_manager.config
    original_layout = executor.page_table_layout
    original_model_paged = executor.model.layers[0].attention.config.paged_attention_config

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
    assert not unresolved.is_resolved()
    assert executor.page_table_layout is original_layout
    assert executor.model.layers[0].attention.config.paged_attention_config is original_model_paged


def test_generator_rejects_unavailable_traced_execution(binding, expect_error):
    target = binding.make_recording_target()
    target.traced_decode_execution = None
    target.config = binding.make_executor_config("none")
    generator = binding.generator_class(target, binding.generator_module._build_vllm_adapter(target))

    with expect_error(RuntimeError, "unavailable traced decode execution"):
        generator._select_execution("decode", True)


def test_initialize_vllm_model_threads_policy(binding, monkeypatch):
    captured = []
    sentinel = object()
    mesh_device = object()
    monkeypatch.setattr(
        binding.generator_module,
        binding.build_generator_name,
        lambda config: captured.append(config) or sentinel,
    )

    result = binding.generator_class.initialize_vllm_model(
        SimpleNamespace(_name_or_path="meta-llama/Llama-3.2-1B-Instruct"),
        mesh_device,
        8,
        4096,
        n_layers=3,
        tt_data_parallel=2,
        optimizations="accuracy",
        trace_mode="decode_only",
        device_sampling_enabled=True,
    )

    assert result is sentinel
    config = captured[0]
    assert isinstance(config, binding.generator_config_class)
    assert config.hf_model == "meta-llama/Llama-3.2-1B-Instruct"
    assert config.mesh_device is mesh_device
    assert config.max_batch_size == 8
    assert config.max_seq_len == 4096
    assert config.n_layers == 3
    assert config.tt_data_parallel == 2
    assert config.optimizations == "accuracy"
    assert config.trace_mode == "decode_only"
    assert config.device_sampling_enabled is True


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
        self.eager_execution = object()
        self.traced_prefill_execution = object()
        self.traced_decode_execution = object()
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1


def test_generator_constructs_data_parallel_lane_group(binding, monkeypatch):
    executor_calls = []
    built_lanes = []
    pretrained_calls = []
    parent_mesh = object()
    submeshes = [_Mesh(), _Mesh()]
    create_submeshes = MagicMock(return_value=submeshes)
    monkeypatch.setattr(binding.generator_module, "_create_submeshes", create_submeshes)

    def fake_from_pretrained(mesh_device, **kwargs):
        pretrained_calls.append((mesh_device, kwargs))
        return binding.make_product(mesh_device, kwargs["max_batch_size"])

    def fake_build_executor(llm, config):
        executor_calls.append((llm, config))
        lane = binding.make_lane(llm, config)
        built_lanes.append(lane)
        return lane

    monkeypatch.setattr(binding.generator_module, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(binding.generator_module, binding.build_executor_name, fake_build_executor)
    monkeypatch.setattr(
        binding.generator_module,
        "_model_kv_metadata",
        lambda model: ((ttnn.bfloat8_b,), 1, 8, 64),
    )

    generator = getattr(binding.generator_module, binding.build_generator_name)(
        binding.generator_config_class(
            hf_model="meta-llama/Llama-3.2-1B-Instruct",
            mesh_device=parent_mesh,
            max_batch_size=4,
            max_seq_len=4096,
            n_layers=1,
            tt_data_parallel=2,
            trace_mode="all",
            device_sampling_enabled=True,
        )
    )

    try:
        create_submeshes.assert_called_once_with(parent_mesh, 2)
        assert [mesh for mesh, _ in pretrained_calls] == submeshes
        assert all(call[1]["max_batch_size"] == 2 for call in pretrained_calls)
        assert all(call[1]["max_seq_len"] == 4096 for call in pretrained_calls)
        assert all(call[1]["n_layers"] == 1 for call in pretrained_calls)
        assert isinstance(generator.target, LaneGroupExecutor)
        assert generator.target.mesh_device is parent_mesh
        assert generator.target.tt_data_parallel == 2
        assert len(executor_calls) == 2
        assert executor_calls[0][0] is not executor_calls[1][0]
        assert generator.target.lanes == built_lanes
        assert [lane.model for lane in generator.target.lanes] == [llm.model for llm, _ in executor_calls]
        assert [lane.mesh_device for lane in generator.target.lanes] == submeshes
        assert len({id(lane) for lane in generator.target.lanes}) == 2
        assert all(isinstance(config, binding.executor_config_class) for _, config in executor_calls)
        assert all(llm.model.config.max_batch_size == 2 for llm, _ in executor_calls)
        assert generator._adapter.config.trace.mode == "all"
        assert generator._adapter.config.expected_num_layers == 1
        assert generator._adapter.config.expected_kv_heads_per_device == 8
        assert generator._adapter.config.expected_head_dim == 64
    finally:
        generator.cleanup()


def test_executor_cleanup_is_ordered_retryable_and_idempotent(binding, expect_error):
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

    executor = object.__new__(binding.executor_class)
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
