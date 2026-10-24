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
from tt_py_test_utils_common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
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
    initial_state,
    recurrent_oracle,
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
    pytest.mark.use_module_device({"l1_small_size": 24576}),
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
    expected_duration_ns=14_567,
)
# Kimi-K3 on Galaxy SP8xTP4: one device's 640 tokens of 24 heads.
_PRODUCTION_CASE = _PerformanceCase(
    "bh24-n20-d128",
    batch_heads=24,
    num_chunks=20,
    dim=128,
    expected_duration_ns=123_100,
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
    zero_actual_start,
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
        lambda: run_summary(inputs, actual_start=zero_actual_start),
        expected,
        names=("affine_a", "affine_b"),
        dtypes=(ttnn.bfloat16, ttnn.bfloat16),
        shapes=((batch_heads, dim, dim), (batch_heads, dim, dim)),
        nrmse_threshold=0.004,
    )
    assert_summary_reconstructs_state(
        host_inputs, ttnn.to_torch(first[0]), ttnn.to_torch(first[1]), nrmse_threshold=0.004
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
        nrmse_threshold=0.02,
    )


@pytest.mark.parametrize("case_id", ["regression", "production"])
def test_summarize_chunk_recurrence_subtraction_conditioning(
    zero_actual_start, device: ttnn.Device, case_id: str
) -> None:
    if case_id == "production":
        host_inputs, inputs = _production_protocol(device, seed=117)
        outputs = run_summary(
            inputs, compute_kernel_config=_production_compute_config(device), actual_start=zero_actual_start
        )
    else:
        host_inputs, inputs = _regression_protocol(device, seed=117)
        outputs = run_summary(inputs, actual_start=zero_actual_start)
    _log_summary_subtraction_conditioning(case_id, summary_oracle(host_inputs), outputs)


def test_summarize_chunk_recurrence_is_device_deterministic(zero_actual_start, device: ttnn.Device) -> None:
    host_inputs, inputs = _regression_protocol(device, seed=1441)
    reference, outputs, mismatch_marker = collect_accuracy_and_determinism_results(
        device, lambda: run_summary(inputs, actual_start=zero_actual_start)
    )
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name="summary outputs device-side exact-value determinism marker",
    )
    for name, golden, output in zip(("affine_a", "affine_b"), summary_oracle(host_inputs), outputs, strict=True):
        assert_accurate(
            golden.float(),
            output.float(),
            name=f"deterministic summary reference {name}",
            pcc_threshold=0.999,
            nrmse_threshold=0.005,
        )
    assert_summary_reconstructs_state(host_inputs, outputs[0], outputs[1], nrmse_threshold=0.005)
    for output in reference:
        ttnn.deallocate(output)


def test_summarize_chunk_recurrence_cache_hit_rebinds_fresh_tensors(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    host_a, inputs_a = _regression_protocol(device, seed=1911)
    host_b, inputs_b = _regression_protocol(device, seed=1912)
    outputs_a = run_summary(inputs_a, actual_start=zero_actual_start)
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    outputs_b = run_summary(inputs_b, actual_start=zero_actual_start)
    ttnn.synchronize_device(device)

    assert device.num_program_cache_entries() == entries
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(inputs_a, inputs_b, strict=True))
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(outputs_a, outputs_b, strict=True))
    assert_outputs_accurate(
        summary_oracle(host_a),
        outputs_a,
        names=("affine_a", "affine_b"),
        context="summary cache miss tensors",
        nrmse_threshold=0.005,
    )
    assert_outputs_accurate(
        summary_oracle(host_b),
        outputs_b,
        names=("affine_a", "affine_b"),
        context="summary cache hit fresh tensors",
        nrmse_threshold=0.005,
    )


def test_summarize_chunk_recurrence_default_compute_config_matches_explicit_defaults(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    _, inputs = _regression_protocol(device, seed=817)
    implicit = run_summary(inputs, actual_start=zero_actual_start)
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
    explicit = run_summary(inputs, compute_kernel_config=explicit_config, actual_start=zero_actual_start)
    assert device.num_program_cache_entries() == entries
    for name, implicit_tt, explicit_tt in zip(("affine_a", "affine_b"), implicit, explicit, strict=True):
        assert_bit_identical(ttnn.to_torch(implicit_tt), ttnn.to_torch(explicit_tt), name=f"{name} explicit defaults")


def test_summarize_chunk_recurrence_approximate_math_uses_distinct_accurate_program(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    host_inputs, inputs = _regression_protocol(device, seed=818)
    exact = run_summary(inputs, actual_start=zero_actual_start)
    entries = device.num_program_cache_entries()
    approximate_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    approximate = run_summary(inputs, compute_kernel_config=approximate_config, actual_start=zero_actual_start)
    assert device.num_program_cache_entries() == entries + 1
    expected = summary_oracle(host_inputs)
    assert_outputs_accurate(
        expected, exact, names=("affine_a", "affine_b"), context="exact summary math", nrmse_threshold=0.005
    )
    assert_outputs_accurate(
        expected,
        approximate,
        names=("affine_a", "affine_b"),
        context="approximate summary math",
        nrmse_threshold=0.005,
    )


def test_summarize_chunk_recurrence_rejects_unsupported_compute_config(
    zero_actual_start, device: ttnn.Device, expect_error: Callable
) -> None:
    _, inputs = _regression_protocol(device, seed=819)
    unsupported_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        packer_l1_acc=True,
    )
    with expect_error(RuntimeError, "packer_l1_acc=true is unsupported"):
        run_summary(inputs, compute_kernel_config=unsupported_config, actual_start=zero_actual_start)


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_summarize_chunk_recurrence_regression_performance(zero_actual_start, device: ttnn.Device) -> None:
    case = _REGRESSION_CASE
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for recurrence-summary performance checks")
    _, inputs = _regression_protocol(device, seed=117)

    def run() -> list[ttnn.Tensor]:
        return run_summary(inputs, actual_start=zero_actual_start)

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
def test_summarize_chunk_recurrence_production_performance(zero_actual_start, device: ttnn.Device) -> None:
    case = _PRODUCTION_CASE
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for recurrence-summary performance checks")
    _, inputs = _production_protocol(device, seed=117)
    output_memory = group_summary_height_sharded(device, case.batch_heads, case.dim)
    compute_config = _production_compute_config(device)

    def run() -> list[ttnn.Tensor]:
        return run_summary(
            inputs, memory_config=output_memory, compute_kernel_config=compute_config, actual_start=zero_actual_start
        )

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


def test_summarize_chunk_recurrence_height_sharded_l1_output(zero_actual_start, device: ttnn.Device) -> None:
    batch_heads, num_chunks, dim = 4, 2, 32
    host_inputs = host_protocol(batch_heads, num_chunks, dim, dim, seed=812)
    expected = summary_oracle(host_inputs)
    inputs = device_protocol(host_inputs, device)
    output_memory = group_summary_height_sharded(device, batch_heads, dim)

    first = assert_runtime_contract(
        device,
        inputs,
        lambda: run_summary(inputs, memory_config=output_memory, actual_start=zero_actual_start),
        expected,
        names=("affine_a", "affine_b"),
        dtypes=(ttnn.bfloat16, ttnn.bfloat16),
        shapes=((batch_heads, dim, dim), (batch_heads, dim, dim)),
        expected_memory_config=output_memory,
        nrmse_threshold=0.004,
    )
    assert_summary_reconstructs_state(
        host_inputs, ttnn.to_torch(first[0]), ttnn.to_torch(first[1]), nrmse_threshold=0.003
    )


def test_summarize_chunk_recurrence_activates_columns_on_idle_cores(zero_actual_start, device: ttnn.Device) -> None:
    """The column activation run beside the summary on the cores it leaves idle writes ttnn.sigmoid's FP32-accurate
    sigmoid of a BF16 column window into a window of an FP32 tensor, bit for bit, leaves the tensor's other columns
    alone, and leaves the summary unchanged."""
    batch_heads, num_chunks, dim = 4, 2, 32
    host_inputs = host_protocol(batch_heads, num_chunks, dim, dim, seed=814)
    inputs = device_protocol(host_inputs, device)
    rows, input_column, output_column, columns = 96, 3, 2, 4
    generator = torch.Generator().manual_seed(815)
    source_host = torch.randn(1, rows, 9 * 32, generator=generator, dtype=torch.bfloat16)
    source = ttnn.from_torch(
        source_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    sentinel = torch.full((1, rows, 7 * 32), -3.0)
    target = ttnn.from_torch(
        sentinel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    with ttnn.manage_config("throw_exception_on_fallback", True):
        plain = ttnn.experimental.kda.summarize_chunk_recurrence(*inputs, actual_start=zero_actual_start)
        activated = ttnn.experimental.kda.summarize_chunk_recurrence(
            *inputs,
            actual_start=zero_actual_start,
            column_activation_input=source,
            column_activation_input_column=input_column,
            column_activation_output=target,
            column_activation_output_column=output_column,
            column_activation_columns=columns,
            column_activation=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID, 4.0, 0.0)],
        )
    # The head transform (the tail is unspecified on an unsplit rank).
    for name, expected, actual in zip(("affine_a", "affine_b"), plain[:2], activated[:2], strict=True):
        assert_bit_identical(ttnn.to_torch(expected), ttnn.to_torch(actual), name=f"{name} beside the activation")

    window = slice(input_column * 32, (input_column + columns) * 32)
    expected = ttnn.to_torch(ttnn.sigmoid(ttnn.typecast(source, ttnn.float32)))[..., window]
    got = ttnn.to_torch(target)
    written = slice(output_column * 32, (output_column + columns) * 32)
    assert_bit_identical(expected, got[..., written], name="activated columns")
    untouched = torch.cat((got[..., : written.start], got[..., written.stop :]), dim=-1)
    assert torch.equal(untouched, torch.full_like(untouched, -3.0)), "the activation wrote outside its columns"


@pytest.mark.parametrize(
    ("batch_heads", "num_chunks", "dim", "bf16_names"),
    [
        pytest.param(2, 3, 32, frozenset(), id="three-chunk-fp32"),
        pytest.param(4, 4, 128, _PRODUCTION_BF16, id="four-chunk-production-bf16"),
    ],
)
def test_summarize_chunk_recurrence_emitted_outputs_reconstruct_scan(
    zero_actual_start,
    device: ttnn.Device,
    batch_heads: int,
    num_chunks: int,
    dim: int,
    bf16_names: frozenset[str],
) -> None:
    """Unsplit from the sequence start, every chunk enters at the entry state E: the emitted F(0) and F(I) must give
    the recurrent scan's outputs as F(0) + (F(I) - F(0)) @ E, and the emitted final transform its final state."""
    host_inputs = host_protocol(batch_heads, num_chunks, dim, dim, bf16_names=bf16_names, seed=813)
    entry = initial_state(batch_heads, dim, dim, seed=3142)
    expected_output, expected_final = recurrent_oracle(host_inputs, entry)
    expected_a, expected_b = summary_oracle(host_inputs)
    inputs = device_protocol(host_inputs, device)

    with ttnn.manage_config("throw_exception_on_fallback", True):
        packed, base, sensitivity, final_a, final_b = ttnn.experimental.kda.summarize_chunk_recurrence(
            *inputs, actual_start=zero_actual_start, packed_head=True, emit_outputs=True
        )
    assert packed.dtype == ttnn.bfloat16
    assert tuple(packed.shape) == (1, batch_heads, dim, 2 * dim)
    for tensor, shape in (
        (base, (batch_heads, num_chunks, CHUNK_SIZE, dim)),
        (sensitivity, (batch_heads, num_chunks, CHUNK_SIZE, dim)),
        (final_a, (batch_heads, dim, dim)),
        (final_b, (batch_heads, dim, dim)),
    ):
        assert tensor.dtype == ttnn.float32
        assert tuple(tensor.shape) == shape
    zero, identity, a, b = (ttnn.to_torch(tensor) for tensor in (base, sensitivity, final_a, final_b))

    assert_accurate(
        expected_output.float(),
        zero + torch.matmul(identity - zero, entry.unsqueeze(1)),
        name="chunk outputs from the emitted terms",
        nrmse_threshold=0.004,
    )
    assert_accurate(
        expected_final,
        torch.matmul(a, entry) + b,
        name="final state from the emitted transform",
        nrmse_threshold=0.004,
    )
    assert_outputs_accurate(
        (expected_a, expected_b),
        (final_a, final_b),
        names=("final_a", "final_b"),
        context="emitted transform",
        nrmse_threshold=0.004,
    )
    # The packed head is the same transform rounded to BFLOAT16.
    packed_host = ttnn.to_torch(packed).float()[0]
    assert_accurate(a, packed_host[:, :, :dim], name="packed head A", nrmse_threshold=0.004)
    assert_accurate(b, packed_host[:, :, dim:], name="packed head B", nrmse_threshold=0.004)


@pytest.mark.parametrize("host_index", range(7))
def test_summarize_chunk_recurrence_rejects_host_protocol_inputs(
    zero_actual_start, device: ttnn.Device, expect_error: Callable, host_index: int
) -> None:
    host_inputs = host_protocol(2, 2, 32, 32)
    inputs = list(device_protocol(host_inputs, device))
    host = host_inputs[host_index]
    dtype = ttnn.bfloat16 if host.dtype == torch.bfloat16 else ttnn.float32
    inputs[host_index] = ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, f"{PROTOCOL_NAMES[host_index]} must be an allocated device tensor"):
        run_summary(tuple(inputs), actual_start=zero_actual_start)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("key_value_mismatch", "K must equal V"),
        ("q_decay_dtype", "q_decay must be FLOAT32 or BFLOAT16"),
        ("intra_dtype", "intra must be FLOAT32"),
        ("value_tiles_exceed_dest", "value tiles exceed the 4 destination-register tiles"),
    ],
)
def test_summarize_chunk_recurrence_rejects_invalid_inputs(
    zero_actual_start, device: ttnn.Device, expect_error: Callable, case: str, message: str
) -> None:
    host_inputs = list(host_protocol(2, 2, 32, 32))
    inputs = list(device_protocol(host_inputs, device))
    memory_config = None
    compute_kernel_config = None
    if case == "key_value_mismatch":
        host_inputs = list(host_protocol(2, 2, 32, 64))
        inputs = list(device_protocol(host_inputs, device))
    elif case == "q_decay_dtype":
        inputs[2] = to_device(host_inputs[2], device, dtype=ttnn.bfloat8_b)
    elif case == "intra_dtype":
        inputs[3] = to_device(host_inputs[3], device, dtype=ttnn.bfloat16)
    elif case == "value_tiles_exceed_dest":
        # One head per core leaves no room to split the 8 value tiles, and FP32 half-sync dest holds 4.
        grid = device.compute_with_storage_grid_size()
        inputs = list(device_protocol(host_protocol(grid.x * grid.y, 1, 256, 256), device))
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(), fp32_dest_acc_en=True, dst_full_sync_en=False, packer_l1_acc=False
        )
    with expect_error(RuntimeError, message):
        run_summary(
            tuple(inputs),
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
            actual_start=zero_actual_start,
        )


@pytest.mark.parametrize(
    "removed_keyword",
    ["chunk_size", "initial_state", "state_only", "identity_tile", "summary_pair", "output_bf16", "raw_seed"],
)
def test_summarize_chunk_recurrence_does_not_expose_prototype_modes(
    zero_actual_start, device: ttnn.Device, expect_error: Callable, removed_keyword: str
) -> None:
    inputs = device_protocol(host_protocol(2, 2, 32, 32), device)
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.experimental.kda.summarize_chunk_recurrence(
            *inputs, actual_start=zero_actual_start, **{removed_keyword: True}
        )
