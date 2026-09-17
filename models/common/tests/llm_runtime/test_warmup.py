# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
import torch

from models.common.llm_runtime.config import PageTableLayout, TraceConfig, WarmupConfig
from models.common.llm_runtime.decode import DecodeRuntimeConfig
from models.common.llm_runtime.output_reader import OutputReader
from models.common.llm_runtime.prefill.config import PrefillRuntimeConfig
from models.common.llm_runtime.program_compiler import CompiledProgram, OutputSpec, ProgramKey
from models.common.llm_runtime.warmup import (
    CoverageAlias,
    WarmupCoordinator,
    WarmupCoordinatorConfig,
    _resolve_coverage_manifest,
)


class RecordingExecution:
    def __init__(self, events=None):
        self.prefill_calls = []
        self.decode_calls = []
        self.events = events if events is not None else []
        self.fail_decode_call = None
        self.prefill_replays = []

    def compile_prefill(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        sampling_params: Any = None,  # ↓ Sampling
    ) -> None:
        self.events.append("compile_prefill")
        self.prefill_calls.append(
            {
                "tokens": tokens,
                "page_table": page_table,
                "prompt_lens": prompt_lens,
                "start_pos": start_pos,
                "empty_slots": empty_slots,
                "sampling_params": sampling_params,
            }
        )

    def compile_decode(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        sampling_params: Any = None,  # ↓ Sampling
        reset_batch: bool = False,  # ↓ State transition
    ) -> None:
        call = len(self.decode_calls) + 1
        self.events.append("compile_decode")
        if call == self.fail_decode_call:
            self.fail_decode_call = None
            raise RuntimeError("decode compile failed")
        self.decode_calls.append(
            {
                "tokens": tokens,
                "start_pos": start_pos,
                "page_table": page_table,
                "sampling_params": sampling_params,
                "reset_batch": reset_batch,
            }
        )

    def prefill_forward(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        sampling_params: Any = None,  # ↓ Sampling
    ) -> None:
        self.events.append("prefill_replay")
        self.prefill_replays.append(
            {
                "tokens": tokens,
                "page_table": page_table,
                "prompt_lens": prompt_lens,
                "start_pos": start_pos,
                "empty_slots": empty_slots,
                "sampling_params": sampling_params,
            }
        )


class RecordingTraceCompiler:
    def __init__(self, events=None):
        self.calls = 0
        self.events = events if events is not None else []

    def capture_all(self):
        self.events.append("capture")
        self.calls += 1


class Mesh:
    shape = (1, 1)


def make_runtime_configs(
    *,
    sampling=True,
    lane_capacity=4,
    allow_force_argmax=True,
    page_table_layout=None,
    sampling_config=None,
    model=None,
):
    mesh = Mesh()
    sampling_config = sampling_config or SimpleNamespace(
        allow_force_argmax=allow_force_argmax,
        max_top_k=32,
    )
    sampling_config.max_batch_size = lane_capacity
    model = model or SimpleNamespace(
        config=SimpleNamespace(max_batch_size=lane_capacity, mesh_device=mesh, num_devices=1),
        sampling=SimpleNamespace(
            config=sampling_config,
            decode_forward=lambda logits, *, k=None, p=None, temp=None, seeds=None, tt_out_tok=None, enable_log_probs=False: None,
        ),
        vocab_size=128,
    )
    mesh = model.config.mesh_device
    layout = page_table_layout or PageTableLayout(
        block_size=32,
        raw_capacity_width=128,
        prefill_width=192,
        decode_width=128,
    )
    output_reader = OutputReader(mesh)
    return (
        PrefillRuntimeConfig.resolve(
            model=model,
            output_reader=output_reader,
            page_table_layout=layout,
            max_batch_size=lane_capacity,
            max_prefill_chunk_size=128,
            device_sampling_enabled=sampling,
            can_enable_trace=lambda _sequence_length, _batch_size: True,
        ),
        DecodeRuntimeConfig.resolve(
            model=model,
            output_reader=output_reader,
            lane_capacity=lane_capacity,
            page_table_layout=layout,
            device_sampling_enabled=sampling,
        ),
    )


def make_coordinator(
    *,
    trace_mode="all",
    sampling=True,
    warmup_config=None,
    sequence_lengths=(128, 1024),
    lane_capacity=4,
    execution=None,
    trace_compiler=None,
    events=None,
    allow_force_argmax=True,
    page_table_layout=None,
    sampling_config=None,
):
    events = events if events is not None else []
    execution = execution or RecordingExecution(events)
    if trace_compiler is None and trace_mode != "none":
        trace_compiler = RecordingTraceCompiler(events)
    sampling_calls = []
    bound_calls = []

    def ensure_sampling():
        events.append("sampling")
        sampling_calls.append(True)

    def validate_bound(value):
        bound_calls.append(value)

    layout = page_table_layout or PageTableLayout(
        block_size=32,
        raw_capacity_width=128,
        prefill_width=192,
        decode_width=128,
    )
    prefill_config, decode_config = make_runtime_configs(
        sampling=sampling,
        lane_capacity=lane_capacity,
        allow_force_argmax=allow_force_argmax,
        page_table_layout=layout,
        sampling_config=sampling_config,
    )
    execution.prefill = SimpleNamespace(config=prefill_config)
    execution.decode = SimpleNamespace(config=decode_config)
    execution.eager_executor = execution
    execution.trace_compiler = trace_compiler

    coordinator = WarmupCoordinator(
        config=WarmupCoordinatorConfig.resolve(
            warmup=warmup_config or WarmupConfig(),
            trace=TraceConfig(trace_mode),
            prefill=prefill_config,
            decode=decode_config,
            prefill_sequence_lengths=sequence_lengths,
        ),
        execution=execution,
        ensure_sampling_buffers=ensure_sampling,
        validate_bound_cache=validate_bound,
    )
    return coordinator, execution, trace_compiler, sampling_calls, bound_calls, events


@pytest.mark.parametrize(
    ("method_name", "parameter_names"),
    [
        ("warmup_prefill", ("self", "kv_cache", "can_sample_on_device", "enable_trace")),
        (
            "warmup_decode",
            ("self", "kv_cache", "max_batch_size", "num_blocks", "can_sample_on_device", "enable_trace"),
        ),
    ],
)
def test_warmup_signatures_match_registered_plugin_contract(method_name, parameter_names):
    parameters = inspect.signature(getattr(WarmupCoordinator, method_name)).parameters

    assert tuple(parameters) == parameter_names
    assert parameters["self"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in parameter_names[1:]:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty


def test_registered_plugin_warmup_calls_validate_cache_without_forwarding_it():
    cache = object()
    coordinator, execution, _, _, bound_calls, _ = make_coordinator(
        trace_mode="none",
        sampling=False,
        warmup_config=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
        sequence_lengths=(128,),
        lane_capacity=1,
    )
    prefill_kwargs = {
        "kv_cache": cache,
        "can_sample_on_device": False,
    }
    decode_kwargs = {
        "kv_cache": cache,
        "max_batch_size": 1,
        "num_blocks": 8,
        "can_sample_on_device": False,
    }

    coordinator.warmup_prefill(enable_trace=False, **prefill_kwargs)
    coordinator.warmup_decode(enable_trace=False, **decode_kwargs)

    assert coordinator.coverage_manifest is None
    assert bound_calls == [cache, cache]
    assert all(
        tuple(call) == ("tokens", "page_table", "prompt_lens", "start_pos", "empty_slots", "sampling_params")
        for call in execution.prefill_calls
    )
    assert all(
        tuple(call) == ("tokens", "start_pos", "page_table", "sampling_params", "reset_batch")
        for call in execution.decode_calls
    )


@pytest.mark.parametrize(
    ("method_name", "plugin_kwargs", "unexpected_name"),
    [
        (
            "warmup_prefill",
            {"kv_cache": "cache", "can_sample_on_device": False, "enable_trace": False},
            "greedy_only",
        ),
        (
            "warmup_decode",
            {
                "kv_cache": "cache",
                "max_batch_size": 1,
                "num_blocks": 8,
                "can_sample_on_device": False,
                "enable_trace": False,
            },
            "read_from_device",
        ),
    ],
)
def test_warmup_contract_rejects_unregistered_plugin_keywords(
    method_name,
    plugin_kwargs,
    unexpected_name,
    expect_error,
):
    coordinator, *_ = make_coordinator(trace_mode="none", sampling=False, lane_capacity=1)
    plugin_kwargs[unexpected_name] = False

    with expect_error(TypeError, unexpected_name):
        getattr(coordinator, method_name)(**plugin_kwargs)


def test_configured_prefill_lengths_override_model_supported_defaults():
    coordinator, execution, *_ = make_coordinator(
        warmup_config=WarmupConfig(prefill_seq_lens=(1024,), prefill_batch_sizes=(1,)),
        sequence_lengths=(128,),
        sampling=False,
    )

    coordinator.warmup_prefill(kv_cache="cache", enable_trace=False, can_sample_on_device=False)

    regular_lengths = [int(call["tokens"].shape[-1]) for call in execution.prefill_calls if call["start_pos"] is None]
    assert regular_lengths == [1024]


@pytest.mark.parametrize(
    ("sequence_lengths", "message"),
    [
        ((), "non-empty tuple"),
        ((True,), "positive integers"),
        ((128, 128), "unique"),
    ],
)
def test_model_supported_prefill_lengths_are_validated_once(sequence_lengths, message, expect_error):
    with expect_error(ValueError, message):
        make_coordinator(sequence_lengths=sequence_lengths)


def test_sampler_argmax_capability_is_resolved_once():
    class SamplingConfig:
        reads = 0
        max_top_k = 32

        @property
        def allow_force_argmax(self):
            self.reads += 1
            return True

    sampling_config = SamplingConfig()
    coordinator, *_ = make_coordinator(sampling_config=sampling_config)
    resolved_reads = sampling_config.reads

    coordinator.warmup_prefill(kv_cache="cache", enable_trace=False, can_sample_on_device=True)
    coordinator.warmup_decode(
        kv_cache="cache",
        enable_trace=False,
        max_batch_size=4,
        num_blocks=8,
        can_sample_on_device=True,
    )

    assert sampling_config.reads == resolved_reads


def test_page_table_layout_can_be_reconfigured_only_before_use(expect_error):
    coordinator, execution, *_ = make_coordinator(
        warmup_config=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
        sampling=False,
    )
    final_layout = PageTableLayout(
        block_size=32,
        raw_capacity_width=4,
        prefill_width=64,
        decode_width=8,
    )

    coordinator.configure_page_table_layout(final_layout)
    assert coordinator.config.page_table_layout is final_layout
    coordinator.warmup_prefill(kv_cache="cache", enable_trace=False, can_sample_on_device=False)

    assert all(call["start_pos"] is None for call in execution.prefill_calls)
    with expect_error(RuntimeError, "configuration is sealed"):
        coordinator.configure_page_table_layout(final_layout)


def test_explicit_configuration_seal_precedes_physical_kv_allocation(expect_error):
    coordinator, *_ = make_coordinator()
    coordinator.seal_configuration()

    with expect_error(RuntimeError, "configuration is sealed"):
        coordinator.configure_page_table_layout(
            PageTableLayout(
                block_size=32,
                raw_capacity_width=64,
                prefill_width=128,
                decode_width=64,
            )
        )


def test_page_table_layout_reconfiguration_requires_immutable_layout(expect_error):
    coordinator, *_ = make_coordinator()

    with expect_error(TypeError, "PageTableLayout"):
        coordinator.configure_page_table_layout(SimpleNamespace(block_size=32))


def test_resolved_config_is_frozen_and_owns_both_coverage_plans(expect_error):
    coordinator, *_ = make_coordinator(
        warmup_config=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
        lane_capacity=2,
    )

    assert coordinator.config.eager_plan.decode == (coordinator.config.eager_plan.decode[0],)
    assert [case.sampling_path for case in coordinator.config.sampled_plan.decode] == ["logits", "argmax"]
    with expect_error(AttributeError, "cannot assign"):
        coordinator.config.lane_batch_size = 4


def test_direct_config_construction_rejects_inconsistent_derived_plan(expect_error):
    coordinator, *_ = make_coordinator()

    with expect_error(ValueError, "plans must match"):
        replace(coordinator.config, eager_plan=coordinator.config.sampled_plan)


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("model", "share one model"),
        ("layout", "share one page-table layout"),
        ("lane", "share one lane capacity"),
        ("sampling", "share device-sampling policy"),
        ("argmax", "share force-argmax capability"),
        ("raw_ceiling", "share one page-table layout ceiling"),
        ("decode_ceiling", "share one page-table layout ceiling"),
    ],
)
def test_resolution_rejects_inconsistent_runtime_configs(mismatch, message, expect_error):
    prefill, decode = make_runtime_configs()
    if mismatch == "model":
        _, decode = make_runtime_configs()
    elif mismatch == "layout":
        decode = decode.with_page_table_layout(PageTableLayout(32, 64, 128, 64))
    elif mismatch == "lane":
        decode = DecodeRuntimeConfig.resolve(
            model=prefill.model,
            output_reader=prefill.output_reader,
            lane_capacity=2,
            page_table_layout=prefill.page_table_layout,
            device_sampling_enabled=True,
        )
    elif mismatch == "sampling":
        decode = DecodeRuntimeConfig.resolve(
            model=prefill.model,
            output_reader=prefill.output_reader,
            lane_capacity=prefill.max_batch_size,
            page_table_layout=prefill.page_table_layout,
            device_sampling_enabled=False,
        )
    elif mismatch == "argmax":
        prefill.model.sampling.config.allow_force_argmax = False
        decode = DecodeRuntimeConfig.resolve(
            model=prefill.model,
            output_reader=prefill.output_reader,
            lane_capacity=prefill.max_batch_size,
            page_table_layout=prefill.page_table_layout,
            device_sampling_enabled=True,
        )
    elif mismatch == "raw_ceiling":
        ceiling = prefill.page_table_layout_ceiling
        prefill = replace(
            prefill,
            page_table_layout_ceiling=replace(
                ceiling,
                raw_capacity_width=ceiling.raw_capacity_width + 1,
                decode_width=ceiling.decode_width + 8,
            ),
        )
    else:
        ceiling = prefill.page_table_layout_ceiling
        prefill = replace(
            prefill,
            page_table_layout_ceiling=replace(ceiling, decode_width=ceiling.decode_width + 8),
        )

    with expect_error(ValueError, message):
        WarmupCoordinatorConfig.resolve(
            warmup=WarmupConfig(),
            trace=TraceConfig("all"),
            prefill=prefill,
            decode=decode,
            prefill_sequence_lengths=(128,),
        )


def test_constructor_rejects_execution_disagreement_with_resolved_config(expect_error):
    coordinator, execution, *_ = make_coordinator()
    config = coordinator.config
    execution.prefill.config = replace(
        execution.prefill.config,
        page_table_layout=PageTableLayout(32, 64, 128, 64),
    )

    with expect_error(ValueError, "warmup config page-table layout"):
        WarmupCoordinator(
            config=config,
            execution=execution,
            ensure_sampling_buffers=lambda: None,
            validate_bound_cache=lambda _: None,
        )


def test_runtime_does_not_copy_static_config_fields():
    coordinator, *_ = make_coordinator()

    assert {
        "page_table_layout",
        "prefill_sequence_lengths",
        "lane_batch_size",
        "device_sampling_enabled",
        "allow_force_argmax",
        "prime_q128_tile_ends",
        "prefill_trace_enabled",
        "decode_trace_enabled",
        "eager_plan",
        "sampled_plan",
    }.isdisjoint(vars(coordinator))


def test_layout_replacement_is_immutable_bounded_and_rebuilds_coverage(expect_error):
    coordinator, *_ = make_coordinator(
        warmup_config=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
        page_table_layout=PageTableLayout(32, 128, 192, 128),
    )
    original = coordinator.config
    replacement = PageTableLayout(32, 4, 64, 8)

    coordinator.configure_page_table_layout(replacement)

    assert coordinator.config is not original
    assert coordinator.config.page_table_layout_ceiling is original.page_table_layout
    assert original.page_table_layout.raw_capacity_width == 128
    assert not any(case.cached_tokens for case in coordinator.config.eager_plan.prefill)
    with expect_error(ValueError, "cannot change block_size"):
        original.with_page_table_layout(PageTableLayout(16, 4, 64, 8))
    with expect_error(ValueError, "capacity ceiling"):
        original.with_page_table_layout(PageTableLayout(32, 129, 192, 136))
    with expect_error(ValueError, "canonical geometry"):
        original.with_page_table_layout(PageTableLayout(32, 128, 200, 128))


def test_q128_batches_are_capped_by_lane_and_non128_is_batch_one():
    config = WarmupConfig(prefill_batch_sizes=(1, 2, 4, 8, 16, 32))
    coordinator, execution, *_ = make_coordinator(warmup_config=config, lane_capacity=8)

    coordinator.warmup_prefill(kv_cache="cache", enable_trace=False, can_sample_on_device=False)

    regular_q128 = [
        int(call["tokens"].shape[0])
        for call in execution.prefill_calls
        if int(call["tokens"].shape[-1]) == 128 and call["start_pos"] is None
    ]
    regular_q1024 = [
        int(call["tokens"].shape[0])
        for call in execution.prefill_calls
        if int(call["tokens"].shape[-1]) == 1024 and call["start_pos"] is None
    ]
    assert regular_q128 == [1, 2, 4, 8]
    assert regular_q1024 == [1]


def test_sampling_paths_include_forced_prefill_topk_and_opt_in_true_topk_decode():
    config = WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,), include_decode_top_k=True)
    coordinator, execution, *_ = make_coordinator(warmup_config=config, sequence_lengths=(128,), lane_capacity=2)

    coordinator.warmup_prefill(kv_cache="cache", enable_trace=False, can_sample_on_device=True)
    coordinator.warmup_decode(
        kv_cache="cache",
        enable_trace=False,
        max_batch_size=2,
        num_blocks=8,
        can_sample_on_device=True,
    )

    assert execution.prefill_calls[0]["sampling_params"] is None
    assert execution.prefill_calls[1]["sampling_params"].top_k.tolist() == [32]
    assert execution.decode_calls[0]["sampling_params"] is None
    assert execution.decode_calls[1]["sampling_params"].top_k.tolist() == [1, 1]
    assert execution.decode_calls[2]["sampling_params"].top_k.tolist() == [32, 32]
    # Preserve the established true-top-k recipe, not merely a top-k label.
    assert execution.decode_calls[2]["sampling_params"].top_p.tolist() == pytest.approx([0.08, 0.08])


def test_q128_single_topk_primes_all_tile_ends():
    config = WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,))
    coordinator, execution, *_ = make_coordinator(
        warmup_config=config,
        sequence_lengths=(128,),
        lane_capacity=32,
    )
    coordinator.warmup_prefill(kv_cache="cache", enable_trace=False, can_sample_on_device=True)

    topk_calls = [
        call
        for call in execution.prefill_calls
        if call["sampling_params"] is not None
        and float(call["sampling_params"].temperature[0]) == 1.0
        and call["start_pos"] is None
    ]
    assert [int(call["prompt_lens"][0]) for call in topk_calls] == [32, 64, 96, 128]
    assert [int(call["tokens"].shape[-1]) for call in topk_calls] == [32, 64, 96, 128]
    argmax_calls = [
        call
        for call in execution.prefill_calls
        if call["sampling_params"] is not None
        and float(call["sampling_params"].temperature[0]) == 0.0
        and call["start_pos"] is None
    ]
    assert [int(call["prompt_lens"][0]) for call in argmax_calls] == [32, 64, 96, 128]


def test_decode_warmup_uses_topk_as_the_platform_greedy_path_when_argmax_is_disabled():
    coordinator, execution, *_ = make_coordinator(
        sequence_lengths=(128,),
        lane_capacity=2,
        allow_force_argmax=False,
    )

    coordinator.warmup_decode(
        kv_cache="cache",
        enable_trace=False,
        max_batch_size=2,
        num_blocks=8,
        can_sample_on_device=True,
    )

    assert execution.decode_calls[0]["sampling_params"] is None
    assert execution.decode_calls[1]["sampling_params"].top_k.tolist() == [32, 32]


def test_eager_and_trace_coverage_are_separately_idempotent():
    config = WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,))
    coordinator, execution, trace_compiler, *_ = make_coordinator(
        warmup_config=config, sequence_lengths=(128,), lane_capacity=1
    )

    coordinator.warmup_prefill(kv_cache="cache", enable_trace=False, can_sample_on_device=False)
    eager_calls = len(execution.prefill_calls)
    coordinator.warmup_prefill(kv_cache="cache", enable_trace=False, can_sample_on_device=False)
    assert len(execution.prefill_calls) == eager_calls

    coordinator.warmup_prefill(kv_cache="cache", enable_trace=True, can_sample_on_device=True)
    trace_calls = len(execution.prefill_calls)
    coordinator.warmup_prefill(kv_cache="cache", enable_trace=True, can_sample_on_device=True)
    assert len(execution.prefill_calls) == trace_calls
    assert trace_compiler.calls == 0


def test_trace_warmup_routes_cached_prefill_through_traced_execution_target():
    coordinator, eager, *_ = make_coordinator(
        warmup_config=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
        sequence_lengths=(128,),
        lane_capacity=1,
        sampling=False,
    )
    traced = RecordingExecution()
    coordinator.execution = traced

    coordinator.warmup_prefill(kv_cache="cache", enable_trace=True, can_sample_on_device=False)

    assert not eager.prefill_calls
    assert any(call["start_pos"] is None for call in traced.prefill_calls)
    assert any(call["start_pos"] is not None for call in traced.prefill_calls)


def test_coverage_manifest_uses_compiler_registries_and_deduplicates_trace_identities():
    eager_program = CompiledProgram(ProgramKey("0" * 64), "eager", OutputSpec((1,), torch.float32))
    first_traced = CompiledProgram(ProgramKey("1" * 64), "traced-a", OutputSpec((1,), torch.float32))
    second_traced = CompiledProgram(ProgramKey("2" * 64), "traced-b", OutputSpec((1,), torch.float32))
    shared_trace_key = ProgramKey("a" * 64)
    program_compiler = SimpleNamespace(compiled_programs=(eager_program, first_traced, second_traced))
    eager = SimpleNamespace(program_compiler=program_compiler)
    trace_compiler = SimpleNamespace(
        trace_key_for_program=lambda key: None if key == eager_program.key else shared_trace_key,
        get=lambda key: SimpleNamespace(signature="shared-trace") if key == shared_trace_key else None,
    )

    manifest = _resolve_coverage_manifest(eager, trace_compiler)

    assert manifest.eager_program_signatures == ("eager",)
    assert manifest.traced_source_program_signatures == ("traced-a", "traced-b")
    assert manifest.trace_signatures == ("shared-trace",)
    assert manifest.aliases == (
        CoverageAlias("traced-a", "shared-trace"),
        CoverageAlias("traced-b", "shared-trace"),
    )


def test_coverage_manifest_rejects_any_missing_required_trace_alias(expect_error):
    first_traced = CompiledProgram(ProgramKey("1" * 64), "traced-a", OutputSpec((1,), torch.float32))
    second_traced = CompiledProgram(ProgramKey("2" * 64), "traced-b", OutputSpec((1,), torch.float32))
    trace_key = ProgramKey("a" * 64)
    eager = SimpleNamespace(program_compiler=SimpleNamespace(compiled_programs=(first_traced, second_traced)))
    trace_compiler = SimpleNamespace(
        trace_key_for_program=lambda key: trace_key if key == first_traced.key else None,
        get=lambda key: SimpleNamespace(signature="trace-a") if key == trace_key else None,
    )

    with expect_error(RuntimeError, "required trace alias"):
        _resolve_coverage_manifest(
            eager,
            trace_compiler,
            required_program_keys={first_traced.key, second_traced.key},
            required_trace_program_keys={first_traced.key, second_traced.key},
        )


def test_activation_validates_every_program_returned_by_trace_warmup(expect_error):
    coordinator, execution, trace_compiler, *_ = make_coordinator(
        warmup_config=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
        sequence_lengths=(128,),
        lane_capacity=1,
        sampling=False,
    )
    programs = tuple(
        CompiledProgram(ProgramKey(str(index) * 64), f"program-{index}", OutputSpec((1,), torch.float32))
        for index in range(1, 4)
    )
    execution.program_compiler = SimpleNamespace(compiled_programs=programs)
    prefill_programs = iter(programs[:2])
    execution.compile_prefill = lambda **_kwargs: (next(prefill_programs),)
    execution.compile_decode = lambda **_kwargs: programs[2]
    trace_keys = {
        programs[0].key: ProgramKey("a" * 64),
        programs[2].key: ProgramKey("c" * 64),
    }
    trace_compiler.trace_key_for_program = trace_keys.get
    trace_compiler.get = lambda key: SimpleNamespace(signature=f"trace-{key.digest[0]}")

    coordinator.warmup_prefill(kv_cache="cache", enable_trace=True, can_sample_on_device=False)
    with expect_error(RuntimeError, programs[1].key.digest):
        coordinator.warmup_decode(
            kv_cache="cache",
            enable_trace=True,
            max_batch_size=1,
            num_blocks=8,
            can_sample_on_device=False,
        )

    assert trace_compiler.calls == 0


@pytest.mark.parametrize("order", [("prefill", "decode"), ("decode", "prefill")])
def test_prefill_decode_order_is_independent_and_capture_waits_for_both(order):
    config = WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,))
    coordinator, execution, trace_compiler, *_ = make_coordinator(
        warmup_config=config, sequence_lengths=(128,), lane_capacity=1
    )

    def run(operation):
        if operation == "prefill":
            coordinator.warmup_prefill(kv_cache="cache", enable_trace=True, can_sample_on_device=True)
        else:
            coordinator.warmup_decode(
                kv_cache="cache",
                enable_trace=True,
                max_batch_size=1,
                num_blocks=8,
                can_sample_on_device=True,
            )

    run(order[0])
    assert trace_compiler.calls == 0
    run(order[1])
    assert trace_compiler.calls == 1
    run(order[0])
    run(order[1])
    assert trace_compiler.calls == 1


@pytest.mark.parametrize("order", [("prefill", "decode"), ("decode", "prefill")])
def test_capture_uses_phase_specific_sampling_decisions(order):
    config = WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,))
    coordinator, execution, trace_compiler, *_ = make_coordinator(
        trace_mode="all",
        warmup_config=config,
        sequence_lengths=(128,),
        lane_capacity=1,
    )

    def run(operation):
        if operation == "prefill":
            coordinator.warmup_prefill(kv_cache="cache", enable_trace=True, can_sample_on_device=False)
        else:
            coordinator.warmup_decode(
                kv_cache="cache",
                enable_trace=True,
                max_batch_size=1,
                num_blocks=8,
                can_sample_on_device=True,
            )

    coordinator.warmup_prefill(kv_cache="cache", enable_trace=False, can_sample_on_device=False)
    coordinator.warmup_decode(
        kv_cache="cache",
        enable_trace=False,
        max_batch_size=1,
        num_blocks=8,
        can_sample_on_device=True,
    )
    with coordinator.defer_capture():
        run(order[0])
        assert trace_compiler.calls == 0
        run(order[1])
        assert coordinator.capture_pending
        coordinator.activate_pending_capture()

    assert trace_compiler.calls == 1
    assert coordinator.already_warmed_up_prefill
    assert all(call["sampling_params"] is None for call in execution.prefill_calls)
    assert any(call["sampling_params"] is not None for call in execution.decode_calls)
    assert not execution.prefill_replays


def test_capture_deferral_stages_complete_registration_until_explicit_activation():
    coordinator, _, trace_compiler, *_ = make_coordinator(
        warmup_config=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
        sequence_lengths=(128,),
        lane_capacity=1,
        sampling=False,
    )

    with coordinator.defer_capture():
        coordinator.warmup_prefill(kv_cache="cache", enable_trace=True, can_sample_on_device=False)
        coordinator.warmup_decode(
            kv_cache="cache",
            enable_trace=True,
            max_batch_size=1,
            num_blocks=8,
            can_sample_on_device=False,
        )
        assert coordinator.capture_pending
        assert not coordinator.trace_activated
        assert trace_compiler.calls == 0
        coordinator.activate_pending_capture()
        assert coordinator.trace_activated
        assert trace_compiler.calls == 1

    assert not coordinator.capture_pending


def test_capture_deferral_exception_discards_pending_activation(expect_error):
    coordinator, _, trace_compiler, *_ = make_coordinator(
        warmup_config=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
        sequence_lengths=(128,),
        lane_capacity=1,
        sampling=False,
    )

    with expect_error(RuntimeError, "staging failed"):
        with coordinator.defer_capture():
            coordinator.warmup_prefill(kv_cache="cache", enable_trace=True, can_sample_on_device=False)
            coordinator.warmup_decode(
                kv_cache="cache",
                enable_trace=True,
                max_batch_size=1,
                num_blocks=8,
                can_sample_on_device=False,
            )
            assert coordinator.capture_pending
            raise RuntimeError("staging failed")

    assert not coordinator.capture_pending
    assert not coordinator.trace_activated
    assert trace_compiler.calls == 0


def test_static_all_can_capture_decode_only_runtime_trace():
    config = WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,))
    coordinator, execution, trace_compiler, *_ = make_coordinator(
        trace_mode="all",
        warmup_config=config,
        sequence_lengths=(128,),
        lane_capacity=1,
    )

    coordinator.warmup_prefill(kv_cache="cache", enable_trace=False, can_sample_on_device=True)
    assert coordinator.already_warmed_up_prefill
    assert trace_compiler.calls == 0

    coordinator.warmup_decode(
        kv_cache="cache",
        enable_trace=True,
        max_batch_size=1,
        num_blocks=8,
        can_sample_on_device=True,
    )

    assert trace_compiler.calls == 1
    assert not execution.prefill_replays


def test_two_phase_static_all_waits_for_phase_two_decode_before_capture():
    config = WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,))
    coordinator, _, trace_compiler, *_ = make_coordinator(
        trace_mode="all",
        warmup_config=config,
        sequence_lengths=(128,),
        lane_capacity=1,
    )

    coordinator.warmup_prefill(kv_cache="cache", enable_trace=False, can_sample_on_device=True)
    coordinator.warmup_decode(
        kv_cache="cache",
        enable_trace=False,
        max_batch_size=1,
        num_blocks=8,
        can_sample_on_device=True,
    )
    coordinator.warmup_prefill(kv_cache="cache", enable_trace=True, can_sample_on_device=True)
    assert trace_compiler.calls == 0

    coordinator.warmup_decode(
        kv_cache="cache",
        enable_trace=True,
        max_batch_size=1,
        num_blocks=8,
        can_sample_on_device=True,
    )

    assert trace_compiler.calls == 1


def test_sampling_buffers_are_materialized_before_first_compile_and_capture():
    events = []
    config = WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,))
    coordinator, _, _, _, _, events = make_coordinator(
        warmup_config=config,
        sequence_lengths=(128,),
        lane_capacity=1,
        events=events,
    )

    coordinator.warmup_decode(
        kv_cache="cache",
        enable_trace=True,
        max_batch_size=1,
        num_blocks=8,
        can_sample_on_device=True,
    )
    coordinator.warmup_prefill(kv_cache="cache", enable_trace=True, can_sample_on_device=True)

    assert events.index("sampling") < events.index("compile_decode")
    assert events.index("sampling") < events.index("compile_prefill")
    assert max(index for index, event in enumerate(events) if event.startswith("compile_")) < events.index("capture")


def test_failed_case_is_not_marked_complete_and_retry_skips_completed_case(expect_error):
    execution = RecordingExecution()
    execution.fail_decode_call = 2
    config = WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,))
    coordinator, execution, *_ = make_coordinator(
        trace_mode="none",
        sampling=True,
        warmup_config=config,
        sequence_lengths=(128,),
        lane_capacity=1,
        execution=execution,
    )

    with expect_error(RuntimeError, "decode compile failed"):
        coordinator.warmup_decode(
            kv_cache="cache",
            enable_trace=False,
            max_batch_size=1,
            num_blocks=8,
            can_sample_on_device=True,
        )
    assert len(execution.decode_calls) == 1

    coordinator.warmup_decode(
        kv_cache="cache",
        enable_trace=False,
        max_batch_size=1,
        num_blocks=8,
        can_sample_on_device=True,
    )
    assert len(execution.decode_calls) == 2


def test_dynamic_hints_cannot_expand_static_trace_or_sampling_ceilings(expect_error):
    coordinator, *_ = make_coordinator(trace_mode="decode_only", sampling=False)

    with expect_error(ValueError, "prefill trace warmup exceeds"):
        coordinator.warmup_prefill(kv_cache="cache", enable_trace=True, can_sample_on_device=False)
    with expect_error(ValueError, "statically disabled"):
        coordinator.warmup_decode(
            kv_cache="cache",
            enable_trace=False,
            max_batch_size=4,
            num_blocks=8,
            can_sample_on_device=True,
        )

    coordinator.warmup_prefill(kv_cache="cache", enable_trace=False, can_sample_on_device=False)
