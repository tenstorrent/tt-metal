# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA chunk-recurrence preparation."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import kda_performance_model_test_utils as perf_model
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    assert_equal,
    collect_accuracy_and_determinism_results,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device,
]

CHUNK_SIZE = 32
OUTPUT_NAMES = ("v_beta", "kd", "q_decay", "intra", "k_dec_t", "final_decay", "t_inv")


@dataclass(frozen=True)
class _TestCase:
    case_id: str
    num_heads: int
    num_chunks: int
    key_dim: int
    value_dim: int


_PERFORMANCE_MARGIN = 0.05
_PRODUCTION_OUTPUT_BF16_MASK = 0x26
_PRODUCTION_EXPECTED_DURATION_NS = 816_534
_T_INV_MAX_ABS = 0.01
_NUMERICAL_STRESS_T_INV_MAX_ABS = 0.05
_UNIT_TEST_CASE = _TestCase("unit-h2-n4-k32-v64", 2, 4, 32, 64)
_PRODUCTION_CASE = _TestCase("sp2-tp4-h24-n80-k128-v128", 24, 80, 128, 128)


def _prepare_chunk_recurrence_ops(
    inputs: Sequence[torch.Tensor | ttnn.Tensor],
    outputs: Sequence[torch.Tensor | ttnn.Tensor],
) -> tuple[perf_model.FpuOps, perf_model.SfpuOps]:
    if len(inputs) != 5 or len(outputs) != 7:
        raise ValueError("chunk-recurrence preparation requires five inputs and seven outputs")
    tensors = (*inputs, *outputs)
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in tensors):
        raise ValueError("chunk-recurrence preparation tensor shapes must be positive")

    q, k, v, g, beta = inputs
    if len(q.shape) != 3 or len(v.shape) != 3 or len(beta.shape) != 4:
        raise ValueError("chunk-recurrence preparation tensor shapes are inconsistent")
    num_heads, num_chunks, chunk_size, trailing = beta.shape
    if chunk_size != CHUNK_SIZE or trailing != 1 or q.shape[-1] % num_heads or v.shape[-1] % num_heads:
        raise ValueError("chunk-recurrence preparation tensor shapes are inconsistent")
    key_dim = q.shape[-1] // num_heads
    value_dim = v.shape[-1] // num_heads
    if (
        q.shape != (1, num_chunks * CHUNK_SIZE, num_heads * key_dim)
        or k.shape != q.shape
        or g.shape != q.shape
        or v.shape != (1, num_chunks * CHUNK_SIZE, num_heads * value_dim)
    ):
        raise ValueError("chunk-recurrence preparation tensor shapes are inconsistent")
    expected_output_shapes = (
        (num_heads, num_chunks, CHUNK_SIZE, value_dim),
        (num_heads, num_chunks, CHUNK_SIZE, key_dim),
        (num_heads, num_chunks, CHUNK_SIZE, key_dim),
        (num_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
        (num_heads, num_chunks, key_dim, CHUNK_SIZE),
        (num_heads, num_chunks, key_dim, 1),
        (num_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
    )
    if any(output.shape != expected for output, expected in zip(outputs, expected_output_shapes, strict=True)):
        raise ValueError("chunk-recurrence preparation tensor shapes are inconsistent")

    instances = num_heads * num_chunks
    inverse_flops = CHUNK_SIZE * (CHUNK_SIZE - 1) * (CHUNK_SIZE + 1) // 3
    return (
        perf_model.FpuOps(
            matrix_flops=instances * (4 * CHUNK_SIZE**2 * key_dim + inverse_flops),
            multiply_ops=instances * (10 * CHUNK_SIZE * key_dim + CHUNK_SIZE * value_dim),
            add_ops=instances * (2 * CHUNK_SIZE + (CHUNK_SIZE - 1) * key_dim + CHUNK_SIZE * key_dim + CHUNK_SIZE**2),
            reduction_ops=instances * 2 * CHUNK_SIZE * (key_dim - 1),
        ),
        perf_model.SfpuOps(
            exp_ops=instances * (3 * CHUNK_SIZE * key_dim + key_dim),
            rsqrt_ops=instances * 2 * CHUNK_SIZE,
        ),
    )


def _prepare_chunk_recurrence_performance(
    inputs: Sequence[ttnn.Tensor],
    outputs: Sequence[ttnn.Tensor],
    *,
    measured_ns: float,
    math_fidelity: ttnn.MathFidelity,
) -> perf_model.KdaPerformance:
    fpu, sfpu = _prepare_chunk_recurrence_ops(inputs, outputs)
    return perf_model.performance(
        fpu=fpu,
        sfpu=sfpu,
        inputs=inputs,
        outputs=outputs,
        measured_ns=measured_ns,
        math_fidelity=math_fidelity,
    )


def test_prepare_chunk_recurrence_work_golden() -> None:
    inputs = (
        torch.empty((1, 32, 2)),
        torch.empty((1, 32, 2)),
        torch.empty((1, 32, 1)),
        torch.empty((1, 32, 2)),
        torch.empty((1, 1, 32, 1)),
    )
    outputs = (
        torch.empty((1, 1, 32, 1)),
        torch.empty((1, 1, 32, 2)),
        torch.empty((1, 1, 32, 2)),
        torch.empty((1, 1, 32, 32)),
        torch.empty((1, 1, 2, 32)),
        torch.empty((1, 1, 2, 1)),
        torch.empty((1, 1, 32, 32)),
    )

    fpu, sfpu = _prepare_chunk_recurrence_ops(inputs, outputs)
    assert fpu == perf_model.FpuOps(
        matrix_flops=19104,
        multiply_ops=672,
        add_ops=1214,
        reduction_ops=64,
    )
    assert sfpu == perf_model.SfpuOps(exp_ops=194, rsqrt_ops=64)


def _host_inputs(
    num_heads: int,
    num_chunks: int,
    key_dim: int,
    value_dim: int,
    *,
    seed: int = 1731,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    sequence = num_chunks * CHUNK_SIZE
    q = (0.3 * torch.randn(1, sequence, num_heads * key_dim, generator=generator)).to(torch.bfloat16).float()
    k = (0.3 * torch.randn(1, sequence, num_heads * key_dim, generator=generator)).to(torch.bfloat16).float()
    v = (0.2 * torch.randn(1, sequence, num_heads * value_dim, generator=generator)).to(torch.bfloat16).float()
    g = (-0.001 - 0.05 * torch.rand(1, sequence, num_heads * key_dim, generator=generator)).to(torch.bfloat16).float()
    beta = torch.sigmoid(torch.randn(num_heads, num_chunks, CHUNK_SIZE, 1, generator=generator)).float()
    return q, k, v, g, beta


def _reshape_flat(tensor: torch.Tensor, num_heads: int, num_chunks: int, dim: int) -> torch.Tensor:
    return (
        tensor.float()
        .reshape(num_chunks * CHUNK_SIZE, num_heads, dim)
        .permute(1, 0, 2)
        .reshape(num_heads, num_chunks, CHUNK_SIZE, dim)
    )


def _oracle(
    inputs: tuple[torch.Tensor, ...],
    num_heads: int,
    output_bf16_mask: int,
) -> tuple[torch.Tensor, ...]:
    q, k, v, g, beta, *_ = inputs
    num_chunks = beta.shape[1]
    key_dim = q.shape[-1] // num_heads
    value_dim = v.shape[-1] // num_heads
    q = _reshape_flat(q, num_heads, num_chunks, key_dim)
    k = _reshape_flat(k, num_heads, num_chunks, key_dim)
    v = _reshape_flat(v, num_heads, num_chunks, value_dim)
    g = _reshape_flat(g, num_heads, num_chunks, key_dim)
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6) * (key_dim**-0.5)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    cumulative_g = torch.cumsum(g, dim=2)
    decay = torch.exp(cumulative_g)
    final_g = cumulative_g[:, :, -1]

    v_beta = beta * v
    kd = beta * k * decay
    q_decay = q * decay
    k_dec_t = (k * torch.exp(final_g.unsqueeze(2) - cumulative_g)).transpose(-1, -2)
    final_decay = torch.exp(final_g).unsqueeze(-1)
    k_fp64 = k.double()
    cumulative_g_fp64 = cumulative_g.double()
    anchor_g = cumulative_g_fp64[:, :, -1:].mul(0.5)
    akk = torch.matmul(
        beta.double() * k_fp64 * torch.exp(cumulative_g_fp64 - anchor_g),
        (k_fp64 * torch.exp(anchor_g - cumulative_g_fp64)).transpose(-1, -2),
    )
    # The unused upper triangle can exceed FP32 range for real gates near -5.
    # Mask in FP64 before converting the causal reference to the output dtype.
    intra = (
        torch.matmul(
            q.double() * torch.exp(cumulative_g_fp64 - anchor_g),
            (k_fp64 * torch.exp(anchor_g - cumulative_g_fp64)).transpose(-1, -2),
        )
        .tril()
        .float()
    )
    identity = torch.eye(CHUNK_SIZE, dtype=torch.float64).reshape(1, 1, CHUNK_SIZE, CHUNK_SIZE)
    t_inv = torch.linalg.inv(identity + torch.tril(akk, diagonal=-1)).float()
    outputs = (v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv)
    return tuple(
        output.to(torch.bfloat16) if output_bf16_mask & (1 << index) else output.float()
        for index, output in enumerate(outputs)
    )


def _to_device(
    tensor: torch.Tensor,
    device: ttnn.Device,
    dtype: ttnn.DataType,
    *,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _device_inputs(inputs: tuple[torch.Tensor, ...], device: ttnn.Device) -> tuple[ttnn.Tensor, ...]:
    return tuple(
        _to_device(tensor, device, ttnn.bfloat16 if index < 4 else ttnn.float32) for index, tensor in enumerate(inputs)
    )


def _run(
    inputs: tuple[ttnn.Tensor, ...],
    num_heads: int,
    *,
    output_bf16_mask: int = 0,
    memory_config: ttnn.MemoryConfig | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
) -> list[ttnn.Tensor]:
    with ttnn.manage_config("throw_exception_on_fallback", True):
        return ttnn.experimental.kda.prepare_chunk_recurrence(
            *inputs,
            num_heads,
            output_bf16_mask=output_bf16_mask,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )


def _case_host_inputs(case: _TestCase, *, seed: int) -> tuple[torch.Tensor, ...]:
    return _host_inputs(case.num_heads, case.num_chunks, case.key_dim, case.value_dim, seed=seed)


def _t_inv_numerical_stress_inputs() -> tuple[torch.Tensor, ...]:
    inputs = list(_host_inputs(2, 2, 128, 128, seed=54813))
    inputs[1] = torch.full_like(inputs[1], 0.25)
    inputs[3] = torch.full_like(inputs[3], -0.01).to(torch.bfloat16).float()
    inputs[4] = torch.full_like(inputs[4], 0.5)
    return tuple(inputs)


def _production_compute_config(device: ttnn.Device) -> ttnn.DeviceComputeKernelConfig:
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _assert_outputs_accurate(
    expected: tuple[torch.Tensor, ...],
    actual: list[ttnn.Tensor],
    *,
    context: str,
) -> None:
    for name, expected_output, actual_tt in zip(OUTPUT_NAMES, expected, actual, strict=True):
        assert_accurate(expected_output, ttnn.to_torch(actual_tt), name=f"{context} {name}", pcc_threshold=0.999)


def _assert_t_inv_strict_lower_accurate(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    context: str,
    max_abs_threshold: float,
) -> None:
    lower_rows, lower_columns = torch.tril_indices(CHUNK_SIZE, CHUNK_SIZE, offset=-1)
    expected_strict_lower = expected[..., lower_rows, lower_columns]
    actual_strict_lower = actual[..., lower_rows, lower_columns]

    assert_accurate(
        expected_strict_lower,
        actual_strict_lower,
        name=f"{context} t_inv strictly-lower",
        pcc_threshold=0.999,
    )
    max_abs = float((expected_strict_lower - actual_strict_lower).abs().max())
    assert (
        max_abs <= max_abs_threshold
    ), f"{context} t_inv strictly-lower max abs error {max_abs:.6f} exceeds {max_abs_threshold:.6f}"


@pytest.mark.parametrize(
    "case",
    [_UNIT_TEST_CASE, _PRODUCTION_CASE],
    ids=lambda case: case.case_id,
)
def test_prepare_chunk_recurrence_contract_accuracy_and_determinism(
    device: ttnn.Device,
    case: _TestCase,
) -> None:
    output_bf16_mask = _PRODUCTION_OUTPUT_BF16_MASK
    compute_kernel_config = _production_compute_config(device)
    host_inputs = _case_host_inputs(case, seed=52797)
    expected = _oracle(host_inputs, case.num_heads, output_bf16_mask)
    inputs = _device_inputs(host_inputs, device)

    def run() -> tuple[ttnn.Tensor, ...]:
        return tuple(
            _run(
                inputs,
                case.num_heads,
                output_bf16_mask=output_bf16_mask,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=compute_kernel_config,
            )
        )

    reference_outputs, actual, mismatch_marker = collect_accuracy_and_determinism_results(device, run)
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name=f"{case.case_id} outputs device-side exact-value determinism marker",
    )
    assert len(reference_outputs) == 7
    expected_shapes = (
        (case.num_heads, case.num_chunks, CHUNK_SIZE, case.value_dim),
        (case.num_heads, case.num_chunks, CHUNK_SIZE, case.key_dim),
        (case.num_heads, case.num_chunks, CHUNK_SIZE, case.key_dim),
        (case.num_heads, case.num_chunks, CHUNK_SIZE, CHUNK_SIZE),
        (case.num_heads, case.num_chunks, case.key_dim, CHUNK_SIZE),
        (case.num_heads, case.num_chunks, case.key_dim, 1),
        (case.num_heads, case.num_chunks, CHUNK_SIZE, CHUNK_SIZE),
    )
    input_addresses = {tensor.buffer_address() for tensor in inputs}
    output_addresses = set()
    for index, (output, shape) in enumerate(zip(reference_outputs, expected_shapes, strict=True)):
        expected_dtype = ttnn.bfloat16 if output_bf16_mask & (1 << index) else ttnn.float32
        assert output.dtype == expected_dtype
        assert output.layout == ttnn.TILE_LAYOUT
        assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert tuple(output.shape) == shape
        assert output.buffer_address() not in input_addresses
        output_addresses.add(output.buffer_address())
    assert len(output_addresses) == 7

    for name, expected_output, actual_output in zip(OUTPUT_NAMES, expected, actual, strict=True):
        assert_accurate(
            expected_output,
            actual_output,
            name=f"{case.case_id} {name} invocation 0",
            pcc_threshold=0.999,
        )
    _assert_t_inv_strict_lower_accurate(
        expected[-1],
        actual[-1],
        context=case.case_id,
        max_abs_threshold=_T_INV_MAX_ABS,
    )
    for output in reference_outputs:
        ttnn.deallocate(output)


def test_prepare_chunk_recurrence_t_inv_is_stable_for_correlated_keys(device: ttnn.Device) -> None:
    host_inputs = _t_inv_numerical_stress_inputs()
    num_heads = host_inputs[-1].shape[0]
    expected = _oracle(host_inputs, num_heads, 0)
    device_inputs = _device_inputs(host_inputs, device)

    reference, outputs, mismatch_marker = collect_accuracy_and_determinism_results(
        device,
        lambda: tuple(_run(device_inputs, num_heads)),
    )
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name="numerical-stress outputs device-side exact-value determinism marker",
    )
    _assert_t_inv_strict_lower_accurate(
        expected[-1],
        outputs[-1],
        context="correlated keys",
        max_abs_threshold=_NUMERICAL_STRESS_T_INV_MAX_ABS,
    )
    for output in reference:
        ttnn.deallocate(output)


def test_prepare_chunk_recurrence_cache_hit_rebinds_fresh_tensors(device: ttnn.Device) -> None:
    case = _UNIT_TEST_CASE
    host_a = _case_host_inputs(case, seed=1911)
    host_b = _case_host_inputs(case, seed=1912)
    inputs_a = _device_inputs(host_a, device)
    inputs_b = _device_inputs(host_b, device)

    outputs_a = _run(inputs_a, case.num_heads)
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    outputs_b = _run(inputs_b, case.num_heads)
    ttnn.synchronize_device(device)

    assert device.num_program_cache_entries() == entries
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(inputs_a, inputs_b, strict=True))
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(outputs_a, outputs_b, strict=True))
    _assert_outputs_accurate(_oracle(host_a, case.num_heads, 0), outputs_a, context="cache miss tensors")
    _assert_outputs_accurate(_oracle(host_b, case.num_heads, 0), outputs_b, context="cache hit fresh tensors")
    assert not torch.equal(ttnn.to_torch(outputs_a[0]), ttnn.to_torch(outputs_b[0]))


def test_prepare_chunk_recurrence_default_compute_config_matches_explicit_defaults(device: ttnn.Device) -> None:
    case = _UNIT_TEST_CASE
    inputs = _device_inputs(_case_host_inputs(case, seed=817), device)
    implicit = _run(inputs, case.num_heads)
    entries = device.num_program_cache_entries()
    explicit_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
        throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
    )
    explicit = _run(inputs, case.num_heads, compute_kernel_config=explicit_config)
    assert device.num_program_cache_entries() == entries
    for name, implicit_tt, explicit_tt in zip(OUTPUT_NAMES, implicit, explicit, strict=True):
        assert_bit_identical(
            ttnn.to_torch(implicit_tt), ttnn.to_torch(explicit_tt), name=f"{name} implicit vs explicit defaults"
        )


def test_prepare_chunk_recurrence_precise_math_uses_distinct_accurate_program(device: ttnn.Device) -> None:
    case = _UNIT_TEST_CASE
    host_inputs = _case_host_inputs(case, seed=818)
    inputs = _device_inputs(host_inputs, device)
    approximate = _run(inputs, case.num_heads)
    entries = device.num_program_cache_entries()
    precise_config = _production_compute_config(device)
    precise = _run(inputs, case.num_heads, compute_kernel_config=precise_config)
    assert device.num_program_cache_entries() == entries + 1
    expected = _oracle(host_inputs, case.num_heads, 0)
    _assert_outputs_accurate(expected, approximate, context="default approximate math")
    _assert_outputs_accurate(expected, precise, context="explicit precise math")


def test_prepare_chunk_recurrence_rejects_unsupported_compute_config(
    device: ttnn.Device, expect_error: Callable
) -> None:
    case = _UNIT_TEST_CASE
    inputs = _device_inputs(_case_host_inputs(case, seed=819), device)
    unsupported_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        packer_l1_acc=True,
    )
    with expect_error(RuntimeError, "packer_l1_acc=true is unsupported"):
        _run(inputs, case.num_heads, compute_kernel_config=unsupported_config)


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_prepare_chunk_recurrence_production_performance(device: ttnn.Device) -> None:
    case = _PRODUCTION_CASE
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for chunk-recurrence preparation performance checks")

    host_inputs = _case_host_inputs(case, seed=117)
    inputs = _device_inputs(host_inputs, device)

    def run() -> list[ttnn.Tensor]:
        return _run(
            inputs,
            case.num_heads,
            output_bf16_mask=_PRODUCTION_OUTPUT_BF16_MASK,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=_production_compute_config(device),
        )

    samples = int(os.getenv("KDA_PREP_PERF_SAMPLES", "1"))
    assert samples > 0
    if samples > 1:
        warmup = run()
        ttnn.synchronize_device(device)
        del warmup
    records = []
    for _ in range(samples):
        outputs, perf_record = profile_realtime_program(device, run)
        records.append(perf_record)
    print("KDA_PREP_TIMINGS_NS=" + json.dumps([record["duration_ns"] for record in records]))
    perf_record = sorted(records, key=lambda record: record["duration_ns"])[samples // 2]
    duration_ns = perf_record["duration_ns"]
    assert len(outputs) == 7
    assert tuple(outputs[0].shape) == (case.num_heads, case.num_chunks, CHUNK_SIZE, case.value_dim)
    performance = _prepare_chunk_recurrence_performance(
        inputs,
        outputs,
        measured_ns=duration_ns,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )
    logger.info(
        f"chunk-recurrence preparation {case.case_id}: measured_ns={duration_ns:.0f}, "
        f"runtime_id={perf_record['runtime_id']}, work={performance.work}, "
        f"ideal_fpu_ns={performance.ideal_fpu_ns:.2f}, ideal_dram_ns={performance.ideal_dram_ns:.2f}, "
        f"ideal_ns={performance.ideal_ns:.2f}, "
        f"fpu_utilization_pct={performance.fpu_utilization_pct:.2f}, "
        f"dram_utilization_pct={performance.dram_utilization_pct:.2f}, "
        f"utilization_pct={performance.utilization_pct:.2f}"
    )
    upper = _PRODUCTION_EXPECTED_DURATION_NS * (1 + _PERFORMANCE_MARGIN)
    assert duration_ns <= upper, (
        f"{case.case_id} duration {duration_ns:.0f} ns exceeds {upper:.0f} ns "
        f"(reference {_PRODUCTION_EXPECTED_DURATION_NS} ns, margin {_PERFORMANCE_MARGIN * 100:.0f}%)"
    )


@pytest.mark.parametrize("host_index", range(5))
def test_prepare_chunk_recurrence_rejects_host_inputs(
    device: ttnn.Device,
    expect_error: Callable,
    host_index: int,
) -> None:
    host_inputs = _host_inputs(2, 2, 32, 32)
    inputs = list(_device_inputs(host_inputs, device))
    dtype = ttnn.bfloat16 if host_index < 4 else ttnn.float32
    inputs[host_index] = ttnn.from_torch(host_inputs[host_index], dtype=dtype, layout=ttnn.TILE_LAYOUT)
    with expect_error(
        RuntimeError,
        f"{('q', 'k', 'v', 'g', 'beta')[host_index]} must be an allocated device tensor",
    ):
        _run(tuple(inputs), 2)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("g_dtype", "g must be BFLOAT16, got DataType::FLOAT32"),
        ("layout", "k must use TILE layout"),
        ("rank", "rank 3 production-flat"),
        ("leading", "leading dimension 1"),
        ("qkg_shape", "q, k, and g must have matching shapes"),
        ("sequence", "matching sequence lengths"),
        ("sequence_chunk", "sequence length must be positive and divisible by 32"),
        ("head_divisibility", "flat widths must be divisible by num_heads"),
        ("key_alignment", "K and V must be positive and tile aligned"),
        ("value_alignment", "K and V must be positive and tile aligned"),
        ("beta", "beta shape must be"),
        ("sharded", "q must use interleaved memory"),
    ],
)
def test_prepare_chunk_recurrence_rejects_invalid_inputs(
    device: ttnn.Device,
    expect_error: Callable,
    case: str,
    message: str,
) -> None:
    host_inputs = list(_host_inputs(2, 2, 32, 32))
    inputs = list(_device_inputs(tuple(host_inputs), device))
    num_heads = 2
    if case == "g_dtype":
        inputs[3] = _to_device(host_inputs[3], device, ttnn.float32)
    elif case == "layout":
        inputs[1] = _to_device(host_inputs[1], device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "rank":
        inputs[1] = _to_device(host_inputs[1].reshape(1, 2, 32, 64), device, ttnn.bfloat16)
    elif case == "leading":
        inputs[0] = _to_device(host_inputs[0].expand(2, -1, -1).clone(), device, ttnn.bfloat16)
    elif case == "qkg_shape":
        inputs[3] = _to_device(host_inputs[3][:, :, :32], device, ttnn.bfloat16)
    elif case == "sequence":
        inputs[2] = _to_device(host_inputs[2][:, :32], device, ttnn.bfloat16)
    elif case == "sequence_chunk":
        for index in range(4):
            inputs[index] = _to_device(host_inputs[index][:, :48], device, ttnn.bfloat16)
    elif case == "head_divisibility":
        num_heads = 3
    elif case == "key_alignment":
        num_heads = 2
        for index in (0, 1, 3):
            tensor = torch.randn(1, 64, 96).to(torch.bfloat16).float()
            inputs[index] = _to_device(tensor, device, ttnn.bfloat16)
    elif case == "value_alignment":
        inputs[2] = _to_device(torch.randn(1, 64, 96).to(torch.bfloat16).float(), device, ttnn.bfloat16)
    elif case == "beta":
        inputs[4] = _to_device(host_inputs[4][:, :1], device, ttnn.float32)
    elif case == "sharded":
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [64, 64],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        inputs[0] = _to_device(host_inputs[0], device, ttnn.bfloat16, memory_config=sharded)
    with expect_error(RuntimeError, message):
        _run(tuple(inputs), num_heads)


def test_prepare_chunk_recurrence_rejects_invalid_options(device: ttnn.Device, expect_error: Callable) -> None:
    host_inputs = _host_inputs(2, 2, 32, 32)
    inputs = _device_inputs(host_inputs, device)
    with expect_error(RuntimeError, "num_heads must be positive"):
        _run(inputs, 0)
    for mask in (0x08, 0x40):
        with expect_error(RuntimeError, "unsupported KDA prep BF16 mask"):
            _run(inputs, 2, output_bf16_mask=mask)

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [64, 64],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    with expect_error(RuntimeError, "output memory layout must be INTERLEAVED, got HEIGHT_SHARDED"):
        _run(inputs, 2, memory_config=sharded)


@pytest.mark.skipif(
    not os.getenv("KDA_REAL_TRACE_ROOT"), reason="set KDA_REAL_TRACE_ROOT for captured KDA input replay"
)
@pytest.mark.parametrize(
    "layer_idx", [None, 5, 13, 20], ids=["captured-layer0", "derived-layer5", "derived-layer13", "derived-layer20"]
)
def test_prepare_chunk_recurrence_captured_inputs(device: ttnn.Device, layer_idx: int | None) -> None:
    """Replay captured recurrence inputs; never substitute synthetic rows or weights."""
    from models.demos.deepseek_v3_d_p.tests.kda.trace_utils import load_trace_rows

    root = Path(os.environ["KDA_REAL_TRACE_ROOT"])
    start = int(os.getenv("KDA_REAL_TRACE_START", "0"))
    sequence = int(os.getenv("KDA_REAL_TRACE_SEQUENCE", "1024"))
    assert start >= 0 and sequence > 0 and sequence % CHUNK_SIZE == 0
    num_heads, key_dim = 96, 128
    if layer_idx is not None:
        from models.demos.deepseek_v3_d_p.tests.kda.trace_utils import (
            decoder_probe_recurrence_inputs,
            load_decoder_stream_probe,
        )

        assert start == 0, "derived probes start with zero convolution history at token zero"
        hidden, weights, config = load_decoder_stream_probe(
            Path(os.environ["KIMI_K3_CKPT"]), Path(os.environ["KDA_REAL_DECODER_TRACE_ROOT"]), layer_idx, sequence
        )
        inputs = decoder_probe_recurrence_inputs(hidden, weights, config)
        del hidden, weights
    else:
        host = []
        for name in ("q", "k", "v", "gate", "beta"):
            key = f"kda_{name}_layer_0"
            value = load_trace_rows(root / "kda" / f"{key}.safetensors", start, sequence)
            width = num_heads if name == "beta" else num_heads * key_dim
            assert tuple(value.shape) == (sequence, width), (key, value.shape, sequence, width)
            if name == "beta":
                value = value.T.reshape(num_heads, sequence // CHUNK_SIZE, CHUNK_SIZE, 1).float().contiguous()
            else:
                value = value.to(torch.bfloat16).float().unsqueeze(0)
            host.append(value)
        inputs = tuple(host)
    output_mask = int(os.getenv("KDA_REAL_OUTPUT_BF16_MASK", str(_PRODUCTION_OUTPUT_BF16_MASK)), 0)
    expected = _oracle(inputs, num_heads, output_mask)
    device_inputs = _device_inputs(inputs, device)
    outputs = _run(
        device_inputs, num_heads, output_bf16_mask=output_mask, compute_kernel_config=_production_compute_config(device)
    )
    actual = [ttnn.to_torch(output).float() for output in outputs]
    metrics = {}
    for name, want, got in zip(OUTPUT_NAMES, expected, actual, strict=True):
        error = (want.float() - got).abs()
        metrics[name] = {
            "nonfinite": int((~torch.isfinite(got)).sum()),
            "max_abs_error": float(error.max()),
            "worst_index": list(torch.unravel_index(error.flatten().argmax(), error.shape)),
        }
        metrics[name]["worst_index"] = [int(index) for index in metrics[name]["worst_index"]]
    print(
        "KDA_CAPTURED_INPUT_METRICS="
        + json.dumps(
            {
                "layer_idx": layer_idx,
                "start": start,
                "sequence": sequence,
                "output_bf16_mask": output_mask,
                "outputs": metrics,
            }
        )
    )
    artifact = os.getenv("KDA_REAL_TRACE_ARTIFACT")
    if artifact:
        torch.save({"start": start, "inputs": inputs, "expected": expected, "actual": actual}, artifact)
    for name, want, got in zip(OUTPUT_NAMES, expected, actual, strict=True):
        assert torch.isfinite(got).all(), f"{name} contains nonfinite values"
        assert_accurate(want.float(), got, name=f"captured {name}", pcc_threshold=0.999)
    # Real traces exercise the existing numerical-stress contract. The first
    # subdiagonal also includes preparation error upstream of the inverse;
    # keep every maximum error in the report rather than relying on PCC alone.
    _assert_t_inv_strict_lower_accurate(
        expected[-1], actual[-1], context="captured inputs", max_abs_threshold=_NUMERICAL_STRESS_T_INV_MAX_ABS
    )


def _real_chunk_inputs() -> tuple[torch.Tensor, ...]:
    from safetensors.torch import load_file

    tensors = load_file(str(Path(__file__).with_name("fixtures") / "layer13_head50_chunk6.safetensors"))
    return tuple(tensors[name].float() for name in ("q", "k", "v", "g", "beta"))


def test_prepare_chunk_recurrence_real_chunk_inverse(device: ttnn.Device) -> None:
    """A single real chunk exposes cancellation missed by random-key tests."""
    inputs = _real_chunk_inputs()
    expected = _oracle(inputs, 1, 0)
    outputs = _run(_device_inputs(inputs, device), 1, compute_kernel_config=_production_compute_config(device))
    actual = ttnn.to_torch(outputs[-1]).float()
    _assert_t_inv_strict_lower_accurate(
        expected[-1], actual, context="real layer13/head50/chunk6", max_abs_threshold=0.01
    )


@pytest.mark.parametrize("output_bf16_mask", [0x00, 0x20, 0x26], ids=["all-fp32", "decay-bf16", "production"])
def test_prepare_chunk_recurrence_scratch_wrap(device: ttnn.Device, output_bf16_mask: int) -> None:
    """Repeat the small fixture far enough to cross the old mixed-width scratch rings on every core."""
    grid = device.compute_with_storage_grid_size()
    chunks = grid.x * grid.y * 32
    chunk = _real_chunk_inputs()
    inputs = tuple(x.repeat(1, chunks, 1) for x in chunk[:4]) + (chunk[4].repeat(1, chunks, 1, 1),)
    expected = torch.exp(chunk[3].sum(dim=1)).reshape(1, 1, 128, 1).repeat(1, chunks, 1, 1)
    if output_bf16_mask & (1 << 5):
        expected = expected.to(torch.bfloat16).float()
    outputs = _run(
        _device_inputs(inputs, device),
        1,
        output_bf16_mask=output_bf16_mask,
        compute_kernel_config=_production_compute_config(device),
    )
    actual = ttnn.to_torch(outputs[5]).float()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.004)
