# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA recurrence summaries."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import kda_performance_model_test_utils as perf_model
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    BF16_ALLOWED,
    CHUNK_SIZE,
    PROTOCOL_NAMES,
    assert_outputs_accurate,
    assert_runtime_contract,
    assert_summary_reconstructs_state,
    device_protocol,
    group_summary_height_sharded,
    host_protocol,
    run_summary,
    summary_oracle,
    to_device,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    assert_equal,
    collect_accuracy_and_determinism_results,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device({"l1_small_size": 24576, "trace_region_size": 2_000_000}),
]


@dataclass(frozen=True)
class _PerformanceCase:
    case_id: str
    batch_heads: int
    num_chunks: int
    dim: int
    expected_duration_ns: int


_PERF_REGRESSION_MARGIN = 0.05
_REGRESSION_CASE = _PerformanceCase(
    "bh8-n4-d32",
    batch_heads=8,
    num_chunks=4,
    dim=32,
    expected_duration_ns=20_336,
)
_PRODUCTION_CASE = _PerformanceCase(
    "pr7-leaf-bh96-n20-d128",
    batch_heads=96,
    num_chunks=20,
    dim=128,
    expected_duration_ns=299_691,
)
_PRODUCTION_BF16 = frozenset({"kd", "q_decay", "final_decay"})


def _summarize_chunk_recurrence_ops(
    inputs: Sequence[torch.Tensor | ttnn.Tensor],
    outputs: Sequence[torch.Tensor | ttnn.Tensor],
) -> tuple[perf_model.FpuOps, perf_model.SfpuOps]:
    if len(inputs) != 7 or len(outputs) != 2:
        raise ValueError("chunk-recurrence summary requires seven inputs and two outputs")
    tensors = (*inputs, *outputs)
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in tensors):
        raise ValueError("chunk-recurrence summary tensor shapes must be positive")
    if any(len(tensor.shape) != 4 for tensor in inputs):
        raise ValueError("chunk-recurrence summary tensor shapes are inconsistent")

    batch_heads, num_chunks, chunk_size, value_dim = inputs[0].shape
    key_dim = inputs[1].shape[-1]
    expected_input_shapes = (
        (batch_heads, num_chunks, CHUNK_SIZE, value_dim),
        (batch_heads, num_chunks, CHUNK_SIZE, key_dim),
        (batch_heads, num_chunks, CHUNK_SIZE, key_dim),
        (batch_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
        (batch_heads, num_chunks, key_dim, CHUNK_SIZE),
        (batch_heads, num_chunks, key_dim, 1),
        (batch_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
    )
    if (
        chunk_size != CHUNK_SIZE
        or key_dim != value_dim
        or any(tensor.shape != expected for tensor, expected in zip(inputs, expected_input_shapes, strict=True))
        or outputs[0].shape != (batch_heads, key_dim, key_dim)
        or outputs[1].shape != (batch_heads, key_dim, value_dim)
    ):
        raise ValueError("chunk-recurrence summary tensor shapes are inconsistent")

    instances = batch_heads * num_chunks
    return (
        perf_model.FpuOps(
            matrix_flops=instances * (8 * CHUNK_SIZE * key_dim * value_dim + 4 * CHUNK_SIZE**2 * value_dim),
            multiply_ops=instances * 2 * key_dim * value_dim,
            add_ops=instances * (2 * CHUNK_SIZE * value_dim + 2 * key_dim * value_dim)
            + batch_heads * key_dim * value_dim,
        ),
        perf_model.SfpuOps(),
    )


def _summarize_chunk_recurrence_performance(
    inputs: Sequence[ttnn.Tensor],
    outputs: Sequence[ttnn.Tensor],
    *,
    measured_ns: float,
    math_fidelity: ttnn.MathFidelity,
) -> perf_model.KdaPerformance:
    fpu, sfpu = _summarize_chunk_recurrence_ops(inputs, outputs)
    return perf_model.performance(
        fpu=fpu,
        sfpu=sfpu,
        inputs=inputs,
        outputs=outputs,
        measured_ns=measured_ns,
        math_fidelity=math_fidelity,
    )


def test_summarize_chunk_recurrence_work_golden() -> None:
    inputs = (
        torch.empty((1, 1, 32, 2)),
        torch.empty((1, 1, 32, 2)),
        torch.empty((1, 1, 32, 2)),
        torch.empty((1, 1, 32, 32)),
        torch.empty((1, 1, 2, 32)),
        torch.empty((1, 1, 2, 1)),
        torch.empty((1, 1, 32, 32)),
    )
    fpu, sfpu = _summarize_chunk_recurrence_ops(
        inputs,
        (torch.empty((1, 2, 2)), torch.empty((1, 2, 2))),
    )

    assert fpu == perf_model.FpuOps(matrix_flops=9216, multiply_ops=8, add_ops=140)
    assert sfpu == perf_model.SfpuOps()


@pytest.mark.parametrize(
    ("batch_heads", "num_chunks", "dim", "bf16_names"),
    [
        pytest.param(2, 1, 32, frozenset(), id="single-chunk-fp32"),
        pytest.param(4, 3, 64, BF16_ALLOWED, id="three-chunk-all-allowed-bf16"),
        pytest.param(8, 4, 32, frozenset({"v_beta", "kd", "final_decay"}), id="grouped-four-chunk"),
    ],
)
def test_summarize_chunk_recurrence_contract_trace_and_semantics(
    device: ttnn.Device,
    batch_heads: int,
    num_chunks: int,
    dim: int,
    bf16_names: frozenset[str],
) -> None:
    host_inputs = host_protocol(batch_heads, num_chunks, dim, dim, bf16_names=bf16_names, seed=811)
    expected = summary_oracle(host_inputs)
    inputs = device_protocol(host_inputs, device)

    first = assert_runtime_contract(
        device,
        inputs,
        lambda: run_summary(inputs),
        expected,
        names=("affine_a", "affine_b"),
        dtypes=(ttnn.float32, ttnn.float32),
        shapes=((batch_heads, dim, dim), (batch_heads, dim, dim)),
    )
    assert_summary_reconstructs_state(host_inputs, ttnn.to_torch(first[0]), ttnn.to_torch(first[1]))


@pytest.mark.parametrize(
    "indicator_value,wrap_chunk,emit_tail",
    [
        (0, 0, False),
        (1, 0, False),
        (0, 2, False),
        (1, 2, False),
        (None, 2, False),
        (None, 2, True),
        (1, 0, True),
        (1, 4, True),
    ],
)
def test_summarize_chunk_recurrence_rejects_unsupported_wrap_modes(
    device: ttnn.Device,
    indicator_value: int | None,
    wrap_chunk: int,
    emit_tail: bool,
    expect_error,
) -> None:
    inputs = device_protocol(host_protocol(2, 4, 32, 32, seed=821), device)
    indicator = None if indicator_value is None else to_device(torch.tensor([[[float(indicator_value)]]]), device)
    with expect_error(RuntimeError, "wrap|tail summaries"):
        run_summary(inputs, wrap_indicator=indicator, wrap_chunk=wrap_chunk, emit_tail_summaries=emit_tail)


@pytest.mark.parametrize("keyword", ["chunk_start", "chunk_count"])
def test_summarize_chunk_recurrence_has_no_range_controls(device: ttnn.Device, keyword: str, expect_error) -> None:
    inputs = device_protocol(host_protocol(2, 4, 32, 32), device)
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.experimental.kda.summarize_chunk_recurrence(*inputs, **{keyword: 1})


def _segmented_summary_oracle(
    host_inputs: tuple[torch.Tensor, ...], groups_per_head: int, chunks_per_group: int, wrap_chunk: int
) -> tuple[torch.Tensor, ...]:
    folded_heads = host_inputs[0].shape[0]
    dim = host_inputs[1].shape[-1]
    expected_parts: list[list[torch.Tensor]] = [[], [], [], []]
    identity_a = torch.eye(dim, dtype=torch.float32).unsqueeze(0)
    identity_b = torch.zeros((1, dim, dim), dtype=torch.float32)
    for folded_head in range(folded_heads):
        group = folded_head % groups_per_head
        group_start = group * chunks_per_group
        group_end = group_start + chunks_per_group
        head_count = max(min(wrap_chunk, group_end) - group_start, 0)
        tail_start = min(max(wrap_chunk - group_start, 0), chunks_per_group)
        for segment_start, segment_end, destination in (
            (0, head_count, 0),
            (tail_start, chunks_per_group, 2),
        ):
            if segment_start == segment_end:
                affine_a, affine_b = identity_a, identity_b
            else:
                segment = tuple(
                    tensor[folded_head : folded_head + 1, segment_start:segment_end] for tensor in host_inputs
                )
                affine_a, affine_b = summary_oracle(segment)
            expected_parts[destination].append(affine_a)
            expected_parts[destination + 1].append(affine_b)
    return tuple(torch.cat(parts, dim=0) for parts in expected_parts)


@pytest.mark.parametrize(
    ("groups_per_head", "chunks_per_group", "wrap_chunk"),
    [
        pytest.param(1, 4, 1, id="g1-first"),
        pytest.param(1, 4, 3, id="g1-last"),
        pytest.param(2, 4, 4, id="g2-boundary"),
        pytest.param(2, 4, 5, id="g2-straddle"),
        pytest.param(4, 4, 7, id="g4-before-boundary"),
        pytest.param(4, 4, 8, id="g4-boundary"),
        pytest.param(4, 4, 9, id="g4-after-boundary"),
        pytest.param(4, 4, 15, id="g4-final"),
    ],
)
def test_summarize_chunk_recurrence_emits_grouped_head_and_tail_segments(
    device: ttnn.Device,
    groups_per_head: int,
    chunks_per_group: int,
    wrap_chunk: int,
) -> None:
    batch_heads = 2
    dim = 32
    folded_heads = batch_heads * groups_per_head
    host_inputs = host_protocol(folded_heads, chunks_per_group, dim, dim, seed=831 + wrap_chunk)
    inputs = device_protocol(host_inputs, device)
    indicator = to_device(torch.ones(1, 1, 1), device)

    expected = _segmented_summary_oracle(host_inputs, groups_per_head, chunks_per_group, wrap_chunk)

    actual = run_summary(
        inputs,
        wrap_indicator=indicator,
        wrap_chunk=wrap_chunk,
        groups_per_head=groups_per_head,
        emit_tail_summaries=True,
    )
    assert_outputs_accurate(
        expected,
        actual,
        names=("head_a", "head_b", "tail_a", "tail_b"),
        context=f"G={groups_per_head} g={chunks_per_group} wrap={wrap_chunk}",
        pcc_threshold=0.999,
    )
    head_a, head_b, tail_a, tail_b = (ttnn.to_torch(t).float() for t in actual)
    full_a, full_b = summary_oracle(host_inputs)
    assert_accurate(full_a, tail_a @ head_a, name="tail-after-head A", pcc_threshold=0.999)
    assert_accurate(full_b, tail_a @ head_b + tail_b, name="tail-after-head B", pcc_threshold=0.999)


def test_summarize_chunk_recurrence_segmented_cache_trace_and_ordinary_equivalence(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    batch_heads, groups_per_head, chunks_per_group, dim, wrap_chunk = 2, 4, 4, 32, 9

    def make(seed: int, boundary: bool) -> tuple[tuple[torch.Tensor, ...], tuple[ttnn.Tensor, ...], ttnn.Tensor]:
        host = host_protocol(batch_heads * groups_per_head, chunks_per_group, dim, dim, seed=seed)
        inputs = device_protocol(host, device)
        indicator = to_device(torch.tensor([[[float(boundary)]]]), device)
        return host, inputs, indicator

    host_a, inputs_a, indicator_a = make(1931, True)
    host_b, inputs_b, indicator_b = make(1932, True)
    outputs_a = run_summary(
        inputs_a,
        wrap_indicator=indicator_a,
        wrap_chunk=wrap_chunk,
        groups_per_head=groups_per_head,
        emit_tail_summaries=True,
    )
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    outputs_b = run_summary(
        inputs_b,
        wrap_indicator=indicator_b,
        wrap_chunk=wrap_chunk,
        groups_per_head=groups_per_head,
        emit_tail_summaries=True,
    )
    ttnn.synchronize_device(device)
    assert device.num_program_cache_entries() == entries
    assert all(
        a.buffer_address() != b.buffer_address()
        for a, b in zip((*inputs_a, indicator_a), (*inputs_b, indicator_b), strict=True)
    )
    assert_outputs_accurate(
        _segmented_summary_oracle(host_a, groups_per_head, chunks_per_group, wrap_chunk),
        outputs_a,
        names=("head_a", "head_b", "tail_a", "tail_b"),
        context="segmented summary cache miss",
    )
    assert_outputs_accurate(
        _segmented_summary_oracle(host_b, groups_per_head, chunks_per_group, wrap_chunk),
        outputs_b,
        names=("head_a", "head_b", "tail_a", "tail_b"),
        context="segmented summary cache hit",
    )

    trace_id = None
    capturing = False
    traced = None
    try:
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        capturing = True
        traced = run_summary(
            inputs_b,
            wrap_indicator=indicator_b,
            wrap_chunk=wrap_chunk,
            groups_per_head=groups_per_head,
            emit_tail_summaries=True,
        )
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        capturing = False
        for replay in range(3):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            for name, cached, replayed in zip(("head_a", "head_b", "tail_a", "tail_b"), outputs_b, traced, strict=True):
                assert_bit_identical(
                    ttnn.to_torch(cached), ttnn.to_torch(replayed), name=f"segmented {name} trace replay"
                )
    finally:
        try:
            if capturing:
                ttnn.end_trace_capture(device, trace_id, cq_id=0)
        finally:
            if trace_id is not None:
                ttnn.release_trace(device, trace_id)
            if traced is not None:
                for tensor in traced:
                    ttnn.deallocate(tensor)

    host_o, inputs_o, ordinary_indicator = make(1933, False)
    baseline = run_summary(inputs_o)
    ordinary = run_summary(
        inputs_o,
        wrap_indicator=ordinary_indicator,
        wrap_chunk=wrap_chunk,
        groups_per_head=groups_per_head,
        emit_tail_summaries=True,
    )
    for name, baseline_output, ordinary_output in zip(("affine_a", "affine_b"), baseline, ordinary[:2], strict=True):
        assert_bit_identical(
            ttnn.to_torch(baseline_output), ttnn.to_torch(ordinary_output), name=f"ordinary segmented {name}"
        )


def _regression_protocol(
    device: ttnn.Device,
    *,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[ttnn.Tensor, ...]]:
    case = _REGRESSION_CASE
    host_inputs = host_protocol(case.batch_heads, case.num_chunks, case.dim, case.dim, seed=seed)
    return host_inputs, device_protocol(host_inputs, device)


def _production_protocol(
    device: ttnn.Device,
    *,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[ttnn.Tensor, ...]]:
    case = _PRODUCTION_CASE
    host_inputs = host_protocol(
        case.batch_heads,
        case.num_chunks,
        case.dim,
        case.dim,
        bf16_names=_PRODUCTION_BF16,
        seed=seed,
    )
    return host_inputs, device_protocol(host_inputs, device)


def _production_compute_config(device: ttnn.Device) -> ttnn.DeviceComputeKernelConfig:
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _log_summary_subtraction_conditioning(
    case_id: str,
    expected: tuple[torch.Tensor, torch.Tensor],
    actual: list[ttnn.Tensor],
) -> None:
    expected_a, expected_b = expected
    actual_a, actual_b = (ttnn.to_torch(output).float() for output in actual)
    raw_identity_state = expected_a + expected_b
    a_scale = expected_a.abs().max().item()
    subtraction_scale = max(raw_identity_state.abs().max().item(), expected_b.abs().max().item())
    a_max_abs_error = (actual_a - expected_a).abs().max().item()
    b_max_abs_error = (actual_b - expected_b).abs().max().item()
    logger.info(
        f"summary conditioning {case_id}: "
        f"A=[{expected_a.min().item():.6e},{expected_a.max().item():.6e}] max_abs={a_scale:.6e}; "
        f"B=[{expected_b.min().item():.6e},{expected_b.max().item():.6e}] "
        f"max_abs={expected_b.abs().max().item():.6e}; "
        f"raw_identity_state_max_abs={raw_identity_state.abs().max().item():.6e}; "
        f"subtraction_amplification={subtraction_scale / a_scale:.6e}; "
        f"A_max_abs_error={a_max_abs_error:.6e} ({a_max_abs_error / a_scale:.6e} of A scale); "
        f"B_max_abs_error={b_max_abs_error:.6e}"
    )
    assert_outputs_accurate(
        expected,
        actual,
        names=("affine_a", "affine_b"),
        context=f"summary subtraction conditioning {case_id}",
    )


@pytest.mark.parametrize("case_id", ["regression", "production"])
def test_summarize_chunk_recurrence_subtraction_conditioning(device: ttnn.Device, case_id: str) -> None:
    if case_id == "production":
        host_inputs, inputs = _production_protocol(device, seed=117)
        outputs = run_summary(inputs, compute_kernel_config=_production_compute_config(device))
    else:
        host_inputs, inputs = _regression_protocol(device, seed=117)
        outputs = run_summary(inputs)
    _log_summary_subtraction_conditioning(case_id, summary_oracle(host_inputs), outputs)


def test_summarize_chunk_recurrence_is_device_deterministic(device: ttnn.Device) -> None:
    host_inputs, inputs = _regression_protocol(device, seed=1441)
    reference, outputs, mismatch_marker = collect_accuracy_and_determinism_results(device, lambda: run_summary(inputs))
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name="summary outputs device-side exact-value determinism marker",
    )
    for name, golden, output in zip(("affine_a", "affine_b"), summary_oracle(host_inputs), outputs, strict=True):
        assert_accurate(golden, output, name=f"deterministic summary reference {name}", pcc_threshold=0.999)
    assert_summary_reconstructs_state(host_inputs, outputs[0], outputs[1])
    for output in reference:
        ttnn.deallocate(output)


def test_summarize_chunk_recurrence_cache_hit_rebinds_fresh_tensors(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    host_a, inputs_a = _regression_protocol(device, seed=1911)
    host_b, inputs_b = _regression_protocol(device, seed=1912)
    outputs_a = run_summary(inputs_a)
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    outputs_b = run_summary(inputs_b)
    ttnn.synchronize_device(device)

    assert device.num_program_cache_entries() == entries
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(inputs_a, inputs_b, strict=True))
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(outputs_a, outputs_b, strict=True))
    assert_outputs_accurate(
        summary_oracle(host_a),
        outputs_a,
        names=("affine_a", "affine_b"),
        context="summary cache miss tensors",
    )
    assert_outputs_accurate(
        summary_oracle(host_b),
        outputs_b,
        names=("affine_a", "affine_b"),
        context="summary cache hit fresh tensors",
    )


def test_summarize_chunk_recurrence_default_compute_config_matches_explicit_defaults(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    _, inputs = _regression_protocol(device, seed=817)
    implicit = run_summary(inputs)
    entries = device.num_program_cache_entries()
    explicit_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
        throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
    )
    explicit = run_summary(inputs, compute_kernel_config=explicit_config)
    assert device.num_program_cache_entries() == entries
    for name, implicit_tt, explicit_tt in zip(("affine_a", "affine_b"), implicit, explicit, strict=True):
        assert_bit_identical(ttnn.to_torch(implicit_tt), ttnn.to_torch(explicit_tt), name=f"{name} explicit defaults")


def test_summarize_chunk_recurrence_approximate_math_uses_distinct_accurate_program(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    host_inputs, inputs = _regression_protocol(device, seed=818)
    exact = run_summary(inputs)
    entries = device.num_program_cache_entries()
    approximate_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    approximate = run_summary(inputs, compute_kernel_config=approximate_config)
    assert device.num_program_cache_entries() == entries + 1
    expected = summary_oracle(host_inputs)
    assert_outputs_accurate(expected, exact, names=("affine_a", "affine_b"), context="exact summary math")
    assert_outputs_accurate(
        expected,
        approximate,
        names=("affine_a", "affine_b"),
        context="approximate summary math",
    )


def test_summarize_chunk_recurrence_rejects_unsupported_compute_config(
    device: ttnn.Device, expect_error: Callable
) -> None:
    _, inputs = _regression_protocol(device, seed=819)
    unsupported_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        packer_l1_acc=True,
    )
    with expect_error(RuntimeError, "packer_l1_acc=true is unsupported"):
        run_summary(inputs, compute_kernel_config=unsupported_config)


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_summarize_chunk_recurrence_regression_performance(device: ttnn.Device) -> None:
    case = _REGRESSION_CASE
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for recurrence-summary performance checks")
    _, inputs = _regression_protocol(device, seed=117)

    def run() -> list[ttnn.Tensor]:
        return run_summary(inputs)

    outputs, perf_record = profile_realtime_program(device, run)
    duration_ns = perf_record["duration_ns"]
    assert tuple(outputs[0].shape) == (case.batch_heads, case.dim, case.dim)
    performance = _summarize_chunk_recurrence_performance(
        inputs,
        outputs,
        measured_ns=duration_ns,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )
    logger.info(
        f"recurrence summary regression {case.case_id}: measured_ns={duration_ns:.0f}, "
        f"runtime_id={perf_record['runtime_id']}, work={performance.work}, "
        f"ideal_fpu_ns={performance.ideal_fpu_ns:.2f}, ideal_dram_ns={performance.ideal_dram_ns:.2f}, "
        f"ideal_ns={performance.ideal_ns:.2f}, "
        f"fpu_utilization_pct={performance.fpu_utilization_pct:.2f}, "
        f"dram_utilization_pct={performance.dram_utilization_pct:.2f}, "
        f"utilization_pct={performance.utilization_pct:.2f}"
    )
    upper = case.expected_duration_ns * (1 + _PERF_REGRESSION_MARGIN)
    assert duration_ns <= upper, (
        f"{case.case_id} duration {duration_ns:.0f} ns exceeds {upper:.0f} ns "
        f"(reference {case.expected_duration_ns} ns, upper margin {_PERF_REGRESSION_MARGIN * 100:.0f}%)"
    )


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_summarize_chunk_recurrence_production_performance(device: ttnn.Device) -> None:
    case = _PRODUCTION_CASE
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for recurrence-summary performance checks")
    _, inputs = _production_protocol(device, seed=117)
    output_memory = group_summary_height_sharded(device, case.batch_heads, case.dim)
    compute_config = _production_compute_config(device)

    def run() -> list[ttnn.Tensor]:
        return run_summary(inputs, memory_config=output_memory, compute_kernel_config=compute_config)

    outputs, perf_record = profile_realtime_program(device, run)
    duration_ns = perf_record["duration_ns"]
    assert tuple(outputs[0].shape) == (case.batch_heads, case.dim, case.dim)
    assert outputs[0].memory_config() == output_memory
    performance = _summarize_chunk_recurrence_performance(
        inputs,
        outputs,
        measured_ns=duration_ns,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )
    logger.info(
        f"recurrence summary production {case.case_id}: measured_ns={duration_ns:.0f}, "
        f"runtime_id={perf_record['runtime_id']}, work={performance.work}, "
        f"ideal_fpu_ns={performance.ideal_fpu_ns:.2f}, ideal_dram_ns={performance.ideal_dram_ns:.2f}, "
        f"ideal_ns={performance.ideal_ns:.2f}, "
        f"fpu_utilization_pct={performance.fpu_utilization_pct:.2f}, "
        f"dram_utilization_pct={performance.dram_utilization_pct:.2f}, "
        f"utilization_pct={performance.utilization_pct:.2f}"
    )
    upper = case.expected_duration_ns * (1 + _PERF_REGRESSION_MARGIN)
    assert duration_ns <= upper, (
        f"{case.case_id} duration {duration_ns:.0f} ns exceeds {upper:.0f} ns "
        f"(reference {case.expected_duration_ns} ns, upper margin {_PERF_REGRESSION_MARGIN * 100:.0f}%)"
    )


def test_summarize_chunk_recurrence_height_sharded_l1_output(device: ttnn.Device) -> None:
    batch_heads, num_chunks, dim = 4, 2, 32
    host_inputs = host_protocol(batch_heads, num_chunks, dim, dim, seed=812)
    expected = summary_oracle(host_inputs)
    inputs = device_protocol(host_inputs, device)
    output_memory = group_summary_height_sharded(device, batch_heads, dim)

    first = assert_runtime_contract(
        device,
        inputs,
        lambda: run_summary(inputs, memory_config=output_memory),
        expected,
        names=("affine_a", "affine_b"),
        dtypes=(ttnn.float32, ttnn.float32),
        shapes=((batch_heads, dim, dim), (batch_heads, dim, dim)),
        expected_memory_config=output_memory,
    )
    assert_summary_reconstructs_state(host_inputs, ttnn.to_torch(first[0]), ttnn.to_torch(first[1]))


@pytest.mark.parametrize("host_index", range(7))
def test_summarize_chunk_recurrence_rejects_host_protocol_inputs(
    device: ttnn.Device, expect_error: Callable, host_index: int
) -> None:
    host_inputs = host_protocol(2, 2, 32, 32)
    inputs = list(device_protocol(host_inputs, device))
    host = host_inputs[host_index]
    dtype = ttnn.bfloat16 if host.dtype == torch.bfloat16 else ttnn.float32
    inputs[host_index] = ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, f"{PROTOCOL_NAMES[host_index]} must be an allocated device tensor"):
        run_summary(tuple(inputs))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("key_value_mismatch", "K must equal V"),
        ("q_decay_dtype", "q_decay must be FLOAT32 or BFLOAT16"),
        ("intra_dtype", "intra must be FLOAT32"),
    ],
)
def test_summarize_chunk_recurrence_rejects_invalid_inputs(
    device: ttnn.Device, expect_error: Callable, case: str, message: str
) -> None:
    host_inputs = list(host_protocol(2, 2, 32, 32))
    inputs = list(device_protocol(host_inputs, device))
    memory_config = None
    if case == "key_value_mismatch":
        host_inputs = list(host_protocol(2, 2, 32, 64))
        inputs = list(device_protocol(host_inputs, device))
    elif case == "q_decay_dtype":
        inputs[2] = to_device(host_inputs[2], device, dtype=ttnn.bfloat8_b)
    elif case == "intra_dtype":
        inputs[3] = to_device(host_inputs[3], device, dtype=ttnn.bfloat16)
    with expect_error(RuntimeError, message):
        run_summary(tuple(inputs), memory_config=memory_config)


@pytest.mark.parametrize(
    "removed_keyword",
    ["chunk_size", "initial_state", "state_only", "identity_tile", "summary_pair", "output_bf16", "raw_seed"],
)
def test_summarize_chunk_recurrence_does_not_expose_prototype_modes(
    device: ttnn.Device, expect_error: Callable, removed_keyword: str
) -> None:
    inputs = device_protocol(host_protocol(2, 2, 32, 32), device)
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.experimental.kda.summarize_chunk_recurrence(*inputs, **{removed_keyword: True})


def test_segmented_k128_sharded_summary_rebinds_indicator(device: ttnn.Device, isolated_program_cache: None) -> None:
    from tests.ttnn.nightly.unit_tests.operations.experimental.kda.test_affine_exclusive_scan import (
        _height_sharded_memory_config,
    )

    groups, chunks, dim = 4, 4, 128
    memory = _height_sharded_memory_config(device, 2 * groups, dim, dim)
    retained = []
    entries = None
    for seed, boundary in ((971, True), (972, False), (973, True)):
        host = host_protocol(2 * groups, chunks, dim, dim, bf16_names=BF16_ALLOWED, seed=seed)
        inputs = device_protocol(host, device)
        indicator = to_device(torch.tensor([[[float(boundary)]]]), device)
        before = tuple(ttnn.to_torch(t).clone() for t in (*inputs, indicator))
        outputs = run_summary(
            inputs,
            groups_per_head=groups,
            wrap_indicator=indicator,
            wrap_chunk=9,
            emit_tail_summaries=True,
            memory_config=memory,
        )
        expected = (
            _segmented_summary_oracle(host, groups, chunks, 9)
            if boundary
            else (
                *summary_oracle(host),
                torch.eye(dim).expand(2 * groups, dim, dim),
                torch.zeros(2 * groups, dim, dim),
            )
        )
        assert len(outputs) == 4
        assert_outputs_accurate(
            expected,
            outputs,
            names=("head_a", "head_b", "tail_a", "tail_b"),
            context=f"K128 sharded segmented boundary={boundary}",
        )
        addresses = {t.buffer_address() for t in (*inputs, indicator)}
        for output in outputs:
            assert tuple(output.shape) == (2 * groups, dim, dim)
            assert output.dtype == ttnn.float32 and output.layout == ttnn.TILE_LAYOUT
            assert output.memory_config() == memory
            assert output.buffer_address() not in addresses
            addresses.add(output.buffer_address())
        for old, tensor in zip(before, (*inputs, indicator), strict=True):
            assert_bit_identical(old, ttnn.to_torch(tensor), name="summary input immutability")
        if entries is None:
            entries = device.num_program_cache_entries()
        else:
            assert device.num_program_cache_entries() == entries
            assert indicator.buffer_address() != retained[-1][len(inputs)].buffer_address()
        retained.append((*inputs, indicator))
        for output in outputs:
            ttnn.deallocate(output)
    for invocation in retained:
        for tensor in invocation:
            ttnn.deallocate(tensor)


@pytest.mark.parametrize("groups", [1, 4])
def test_device_chronology_summaries_single_capture(device: ttnn.Device, groups: int) -> None:
    """Validate only live head/tail slots while one trace changes split and boundary rank."""
    chunks_per_group = 4
    rows = groups * chunks_per_group * 32
    host = host_protocol(2 * groups, chunks_per_group, 32, 32, seed=981)
    inputs = device_protocol(host, device)

    def scalar(value: int) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.tensor([value], dtype=torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    start, rank = scalar(0), scalar(0)

    def run():
        controls = ttnn.experimental.kda.chronological_topology(start, rank, 2, rows, 2, 32, 32)
        return ttnn.experimental.kda.summarize_chunk_recurrence(
            *inputs, groups_per_head=groups, emit_tail_summaries=True, chronology=controls
        )

    for _ in range(2):
        for tensor in run():
            ttnn.deallocate(tensor)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    outputs = run()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for offset in range(0, 2 * rows + 32, 32):
            source = scalar(offset)
            ttnn.copy(source, start)
            ttnn.deallocate(source)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            split = offset % rows != 0 and (offset // rows) % 2 == 0
            wrap = (rows - offset % rows) // 32 if split else rows // 32
            expected = _segmented_summary_oracle(host, groups, chunks_per_group, wrap)
            actual = [ttnn.to_torch(t).float() for t in outputs]
            assert all(t.dtype == ttnn.bfloat16 for t in outputs)
            for folded_head in range(2 * groups):
                group = folded_head % groups
                for part in range(4):
                    active = (
                        group * chunks_per_group < wrap if part < 2 else split and (group + 1) * chunks_per_group > wrap
                    )
                    if active:
                        assert_accurate(
                            expected[part][folded_head].bfloat16().float(),
                            actual[part][folded_head],
                            name=f"summary G={groups} start={offset} group={group} part={part}",
                            pcc_threshold=0.999,
                        )
    finally:
        ttnn.release_trace(device, trace)
