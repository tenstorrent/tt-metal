# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.common.llm_runtime.config import PageTableLayout, TraceConfig, WarmupConfig
from models.common.llm_runtime.warmup import WarmupCase, WarmupCoordinator


class RecordingExecution:
    def __init__(self, events=None):
        self.prefill = []
        self.decode = []
        self.events = events if events is not None else []
        self.fail_decode_call = None
        self.prefill_replays = []

    def compile_prefill(self, **kwargs):
        self.events.append("compile_prefill")
        self.prefill.append(kwargs)

    def compile_decode(self, **kwargs):
        call = len(self.decode) + 1
        self.events.append("compile_decode")
        if call == self.fail_decode_call:
            self.fail_decode_call = None
            raise RuntimeError("decode compile failed")
        self.decode.append(kwargs)

    def prefill_forward(self, **kwargs):
        self.events.append("prefill_replay")
        self.prefill_replays.append(kwargs)


class RecordingTraceCompiler:
    def __init__(self, events=None):
        self.calls = 0
        self.events = events if events is not None else []

    def capture_all(self):
        self.events.append("capture")
        self.calls += 1


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

    coordinator = WarmupCoordinator(
        config=warmup_config or WarmupConfig(),
        trace_config=TraceConfig(trace_mode),
        execution=execution,
        eager=object(),
        trace_compiler=trace_compiler,
        model=SimpleNamespace(
            config=SimpleNamespace(max_batch_size=lane_capacity),
            sampling=SimpleNamespace(config=SimpleNamespace(allow_force_argmax=allow_force_argmax)),
        ),
        page_table_layout=PageTableLayout(
            block_size=32,
            raw_capacity_width=128,
            prefill_width=192,
            decode_width=128,
        ),
        prefill_sequence_lengths=sequence_lengths,
        device_sampling_enabled=sampling,
        ensure_sampling_buffers=ensure_sampling,
        validate_bound_cache=validate_bound,
    )
    return coordinator, execution, trace_compiler, sampling_calls, bound_calls, events


def case_tuple(case):
    return (
        case.operation,
        case.batch_size,
        case.sequence_length,
        case.sampling_path,
        case.cached_tokens,
    )


def test_default_case_snapshot_matches_existing_coverage():
    coordinator, *_ = make_coordinator()

    plan = coordinator.plan(can_sample_on_device=True)

    assert tuple(map(case_tuple, plan.prefill)) == (
        ("prefill", 1, 128, "logits", 0),
        ("prefill", 1, 128, "topk", 0),
        ("prefill", 1, 128, "argmax", 0),
        ("prefill", 2, 128, "logits", 0),
        ("prefill", 2, 128, "topk", 0),
        ("prefill", 4, 128, "logits", 0),
        ("prefill", 4, 128, "topk", 0),
        ("prefill", 1, 128, "logits", 32),
        ("prefill", 1, 128, "topk", 32),
        ("prefill", 1, 128, "argmax", 32),
        ("prefill", 1, 1024, "logits", 0),
        ("prefill", 1, 1024, "topk", 0),
        ("prefill", 1, 1024, "argmax", 0),
        ("prefill", 1, 1024, "logits", 32),
        ("prefill", 1, 1024, "topk", 32),
        ("prefill", 1, 1024, "argmax", 32),
    )
    assert tuple(map(case_tuple, plan.decode)) == (
        ("decode", 4, None, "logits", 0),
        ("decode", 4, None, "argmax", 0),
    )


def test_q128_batches_are_capped_by_lane_and_non128_is_batch_one():
    config = WarmupConfig(prefill_batch_sizes=(1, 2, 4, 8, 16, 32))
    coordinator, *_ = make_coordinator(warmup_config=config, lane_capacity=8)

    plan = coordinator.plan(can_sample_on_device=False)

    regular_q128 = [case.batch_size for case in plan.prefill if case.sequence_length == 128 and not case.cached_tokens]
    regular_q1024 = [
        case.batch_size for case in plan.prefill if case.sequence_length == 1024 and not case.cached_tokens
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

    assert [case.sampling_path for case in coordinator.plan(can_sample_on_device=True).decode] == [
        "logits",
        "argmax",
        "topk",
    ]
    assert execution.prefill[0]["sampling_params"] is None
    assert execution.prefill[1]["sampling_params"].top_k.tolist() == [32]
    assert execution.decode[0]["sampling_params"] is None
    assert execution.decode[1]["sampling_params"].top_k.tolist() == [1, 1]
    assert execution.decode[2]["sampling_params"].top_k.tolist() == [32, 32]
    # Preserve the established true-top-k recipe, not merely a top-k label.
    assert execution.decode[2]["sampling_params"].top_p.tolist() == pytest.approx([0.08, 0.08])


def test_q128_single_topk_primes_all_tile_ends_without_expanding_public_plan():
    config = WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,))
    coordinator, execution, *_ = make_coordinator(
        warmup_config=config,
        sequence_lengths=(128,),
        lane_capacity=32,
    )
    plan_before = coordinator.plan(can_sample_on_device=True)

    coordinator.warmup_prefill(kv_cache="cache", enable_trace=False, can_sample_on_device=True)

    assert coordinator.plan(can_sample_on_device=True) == plan_before
    topk_calls = [
        call
        for call in execution.prefill
        if call["sampling_params"] is not None
        and float(call["sampling_params"].temperature[0]) == 1.0
        and call["start_pos"] is None
    ]
    assert [int(call["prompt_lens"][0]) for call in topk_calls] == [32, 64, 96, 128]
    assert [int(call["tokens"].shape[-1]) for call in topk_calls] == [32, 64, 96, 128]
    argmax_calls = [
        call
        for call in execution.prefill
        if call["sampling_params"] is not None
        and float(call["sampling_params"].temperature[0]) == 0.0
        and call["start_pos"] is None
    ]
    assert [int(call["prompt_lens"][0]) for call in argmax_calls] == [32, 64, 96, 128]
    assert coordinator.coverage.eager == frozenset(plan_before.prefill)


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

    assert [case.sampling_path for case in coordinator.plan(can_sample_on_device=True).decode] == [
        "logits",
        "topk",
    ]
    assert execution.decode[0]["sampling_params"] is None
    assert execution.decode[1]["sampling_params"].top_k.tolist() == [32, 32]


def test_eager_and_trace_coverage_are_separately_idempotent():
    config = WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,))
    coordinator, execution, trace_compiler, *_ = make_coordinator(
        warmup_config=config, sequence_lengths=(128,), lane_capacity=1
    )

    coordinator.warmup_prefill(kv_cache="cache", enable_trace=False, can_sample_on_device=False)
    eager_calls = len(execution.prefill)
    coordinator.warmup_prefill(kv_cache="cache", enable_trace=False, can_sample_on_device=False)
    assert len(execution.prefill) == eager_calls

    coordinator.warmup_prefill(kv_cache="cache", enable_trace=True, can_sample_on_device=True)
    trace_calls = len(execution.prefill)
    coordinator.warmup_prefill(kv_cache="cache", enable_trace=True, can_sample_on_device=True)
    assert len(execution.prefill) == trace_calls
    assert coordinator.coverage.eager
    assert coordinator.coverage.trace_registered
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
    assert coordinator.coverage.traces_captured
    run(order[0])
    run(order[1])
    assert trace_compiler.calls == 1


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
    assert coordinator.coverage.traces_captured
    assert coordinator.coverage.trace_registered == frozenset(coordinator.plan(can_sample_on_device=True).decode)
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
    assert coordinator.coverage.traces_captured


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
    assert coordinator.coverage.eager == frozenset({WarmupCase("decode", 1, None, "logits")})

    coordinator.warmup_decode(
        kv_cache="cache",
        enable_trace=False,
        max_batch_size=1,
        num_blocks=8,
        can_sample_on_device=True,
    )
    assert coordinator.coverage.eager.issuperset(
        {
            WarmupCase("decode", 1, None, "logits"),
            WarmupCase("decode", 1, None, "argmax"),
        }
    )
    assert len(execution.decode) == 2


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
