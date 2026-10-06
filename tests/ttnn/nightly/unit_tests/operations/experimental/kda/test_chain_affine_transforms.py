# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA affine-transform chaining."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
    pytest.mark.use_module_device({"l1_small_size": 24576}),
]


@dataclass(frozen=True)
class _Case:
    case_id: str
    batch_heads: int
    key_dim: int
    value_dim: int


# A single device is a one-rank sequence-parallel mesh, so every chain here has one step; the mesh tests in
# test_sp_contract.py cover chronological order across ranks.
_STEPS = 1
_PRODUCTION_PERF_MARGIN = 0.05
_SMALL_CASE = _Case("bh2-k32-v64", batch_heads=2, key_dim=32, value_dim=64)
_PRODUCTION_PERF_CASE = _Case("p1-bh24-k128-v128", batch_heads=24, key_dim=128, value_dim=128)
# Measured on a Blackhole Galaxy chip, where reduce_affine_transforms' production case reads 2.6% above its reference.
_PRODUCTION_PERF_EXPECTED_DURATION_NS = 33_400


def _chain_affine_transforms_ops(
    transforms: torch.Tensor | ttnn.Tensor,
    outputs: tuple[torch.Tensor | ttnn.Tensor, ...],
) -> tuple[perf_model.FpuOps, perf_model.SfpuOps]:
    if len(outputs) != 2:
        raise ValueError("affine-transform chaining requires two outputs")
    if len(transforms.shape) != 4 or any(len(output.shape) != 3 for output in outputs):
        raise ValueError("affine-transform chaining tensor shapes are inconsistent")
    steps, batch_heads, key_dim, row_dim = transforms.shape
    value_dim = row_dim - key_dim
    if value_dim <= 0 or any(tuple(output.shape) != (batch_heads, key_dim, value_dim) for output in outputs):
        raise ValueError("affine-transform chaining tensor shapes are inconsistent")

    head_steps = steps * batch_heads
    return (
        perf_model.FpuOps(
            matrix_flops=head_steps * 2 * key_dim**2 * value_dim,
            add_ops=head_steps * key_dim * value_dim,
        ),
        perf_model.SfpuOps(),
    )


def _chain_affine_transforms_performance(
    transforms: ttnn.Tensor,
    initial_state: ttnn.Tensor,
    outputs: tuple[ttnn.Tensor, ...],
    *,
    measured_ns: float,
    math_fidelity: ttnn.MathFidelity,
) -> perf_model.KdaPerformance:
    fpu, sfpu = _chain_affine_transforms_ops(transforms, outputs)
    return perf_model.performance(
        fpu=fpu,
        sfpu=sfpu,
        inputs=(transforms, initial_state),
        outputs=outputs,
        measured_ns=measured_ns,
        math_fidelity=math_fidelity,
    )


def test_chain_affine_transforms_work_golden() -> None:
    fpu, sfpu = _chain_affine_transforms_ops(
        torch.empty((2, 1, 2, 3)),
        (torch.empty((1, 2, 1)), torch.empty((1, 2, 1))),
    )

    assert fpu == perf_model.FpuOps(matrix_flops=16, add_ops=4)
    assert sfpu == perf_model.SfpuOps()


def _host_inputs(
    batch_heads: int, key_dim: int, value_dim: int, *, seed: int = 914
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return BF16-representable A and B, which cross the mesh as BF16, and an FP32 initial state."""
    generator = torch.Generator().manual_seed(seed)
    eye = torch.eye(key_dim).reshape(1, 1, key_dim, key_dim)
    a = (0.94 * eye).expand(_STEPS, batch_heads, -1, -1).clone()
    a += 0.0125 * torch.randn(a.shape, generator=generator)
    b = 0.025 * torch.randn(_STEPS, batch_heads, key_dim, value_dim, generator=generator)
    initial = 0.1 * torch.randn(batch_heads, key_dim, value_dim, generator=generator)
    return a.bfloat16().float(), b.bfloat16().float(), initial


def _oracle(a: torch.Tensor, b: torch.Tensor, initial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    carry = initial
    for step in range(a.shape[0]):
        carry = a[step] @ carry + b[step]
    return initial, carry


def _to_device(
    tensor: torch.Tensor,
    device: ttnn.Device,
    dtype: ttnn.DataType = ttnn.float32,
    *,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _device_inputs(
    a: torch.Tensor, b: torch.Tensor, initial: torch.Tensor, device: ttnn.Device
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    return _to_device(torch.cat([a, b], dim=-1), device, ttnn.bfloat16), _to_device(initial, device)


def _run(
    transforms: ttnn.Tensor,
    initial_state: ttnn.Tensor,
    *,
    actual_start: ttnn.Tensor,
    local_rows: int = 32,
    memory_config: ttnn.MemoryConfig | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    with ttnn.manage_config("throw_exception_on_fallback", True):
        return ttnn.experimental.kda.chain_affine_transforms(
            transforms,
            initial_state,
            actual_start=actual_start,
            local_rows=local_rows,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )


def _composed_ttnn_baseline(
    a: torch.Tensor,
    b: torch.Tensor,
    initial: torch.Tensor,
    device: ttnn.Device,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig,
) -> ttnn.Tensor:
    """Express the same chain using only ordinary TTNN typecast, matmul and add."""
    carry = _to_device(initial.unsqueeze(0), device)
    for step in range(a.shape[0]):
        transform_a = ttnn.typecast(_to_device(a[step : step + 1], device, ttnn.bfloat16), ttnn.float32)
        transform_b = ttnn.typecast(_to_device(b[step : step + 1], device, ttnn.bfloat16), ttnn.float32)
        product = ttnn.matmul(transform_a, carry, dtype=ttnn.float32, compute_kernel_config=compute_kernel_config)
        carry = ttnn.add(product, transform_b)
    return carry


def _production_compute_config(device: ttnn.Device) -> ttnn.DeviceComputeKernelConfig:
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )


def test_chain_affine_transforms_contract_and_trace(zero_actual_start, device: ttnn.Device) -> None:
    case = _SMALL_CASE
    host = _host_inputs(case.batch_heads, case.key_dim, case.value_dim)
    expected = _oracle(*host)
    transforms_tt, initial_tt = _device_inputs(*host, device)
    snapshots = (ttnn.to_torch(transforms_tt).clone(), ttnn.to_torch(initial_tt).clone())

    first = _run(transforms_tt, initial_tt, actual_start=zero_actual_start)
    for output in first:
        assert output.dtype == ttnn.float32
        assert output.layout == ttnn.TILE_LAYOUT
        assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert tuple(ttnn.to_torch(output).shape) == (case.batch_heads, case.key_dim, case.value_dim)
        assert output.buffer_address() not in (transforms_tt.buffer_address(), initial_tt.buffer_address())
    assert first[0].buffer_address() != first[1].buffer_address()

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = _run(transforms_tt, initial_tt, actual_start=zero_actual_start)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for _ in range(2):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    for name, golden, actual_tt, traced_tt in zip(("entry", "final"), expected, first, traced, strict=True):
        actual = ttnn.to_torch(actual_tt)
        assert_accurate(golden, actual, name=f"chained {name}", pcc_threshold=0.999)
        assert_bit_identical(actual, ttnn.to_torch(traced_tt), name=f"{name} trace replay")

    assert_bit_identical(snapshots[0], ttnn.to_torch(transforms_tt), name="transforms immutability")
    assert_bit_identical(snapshots[1], ttnn.to_torch(initial_tt), name="initial_state immutability")
    ttnn.release_trace(device, trace_id)


@pytest.mark.parametrize(
    ("batch_heads", "key_dim", "value_dim"),
    [
        pytest.param(1, 32, 32, id="bh1-k32-v32"),
        pytest.param(3, 32, 64, id="bh3-k32-v64"),
        pytest.param(2, 64, 32, id="bh2-k64-v32"),
        pytest.param(2, 160, 32, id="bh2-k160-v32"),
        pytest.param(2, 32, 160, id="bh2-k32-v160"),
        pytest.param(24, 128, 128, id="bh24-k128-v128"),
    ],
)
def test_chain_affine_transforms_shape_accuracy(
    zero_actual_start, device: ttnn.Device, batch_heads: int, key_dim: int, value_dim: int
) -> None:
    host = _host_inputs(batch_heads, key_dim, value_dim)
    expected = _oracle(*host)

    outputs = _run(*_device_inputs(*host, device), actual_start=zero_actual_start)

    for name, golden, output in zip(("entry", "final"), expected, outputs, strict=True):
        assert_accurate(golden, ttnn.to_torch(output), name=f"shape-sweep chained {name}", pcc_threshold=0.999)


def test_chain_affine_transforms_is_accurate_and_deterministic(zero_actual_start, device: ttnn.Device) -> None:
    case = _SMALL_CASE
    host = _host_inputs(case.batch_heads, case.key_dim, case.value_dim, seed=1441)
    transforms_tt, initial_tt = _device_inputs(*host, device)
    expected = _oracle(*host)

    def run() -> tuple[ttnn.Tensor, ...]:
        return _run(transforms_tt, initial_tt, actual_start=zero_actual_start)

    reference_outputs, outputs, mismatch_marker = collect_accuracy_and_determinism_results(device, run)
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name="chained outputs device-side exact-value determinism marker",
    )
    for name, golden, output in zip(("entry", "final"), expected, outputs, strict=True):
        assert_accurate(golden, output, name=f"chained {name}", pcc_threshold=0.999)
    for output in reference_outputs:
        ttnn.deallocate(output)


def test_chain_affine_transforms_cache_hit_rebinds_fresh_tensors(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _SMALL_CASE
    host_a = _host_inputs(case.batch_heads, case.key_dim, case.value_dim, seed=1911)
    host_b = _host_inputs(case.batch_heads, case.key_dim, case.value_dim, seed=1912)
    device_a = _device_inputs(*host_a, device)
    device_b = _device_inputs(*host_b, device)

    output_a = _run(*device_a, actual_start=zero_actual_start)
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    output_b = _run(*device_b, actual_start=zero_actual_start)
    ttnn.synchronize_device(device)

    assert device.num_program_cache_entries() == entries
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(device_a, device_b, strict=True))
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(output_a, output_b, strict=True))
    for name, golden_a, golden_b, actual_a_tt, actual_b_tt in zip(
        ("entry", "final"), _oracle(*host_a), _oracle(*host_b), output_a, output_b, strict=True
    ):
        actual_a = ttnn.to_torch(actual_a_tt)
        actual_b = ttnn.to_torch(actual_b_tt)
        assert_accurate(golden_a, actual_a, name=f"{name} cache miss tensors", pcc_threshold=0.999)
        assert_accurate(golden_b, actual_b, name=f"{name} cache hit fresh tensors", pcc_threshold=0.999)
        assert not torch.equal(actual_a, actual_b)


def test_chain_affine_transforms_default_compute_config_matches_explicit_defaults(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _SMALL_CASE
    host = _host_inputs(case.batch_heads, case.key_dim, case.value_dim, seed=817)
    transforms_tt, initial_tt = _device_inputs(*host, device)
    implicit = _run(transforms_tt, initial_tt, actual_start=zero_actual_start)
    entries = device.num_program_cache_entries()
    explicit_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
        throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
    )
    explicit = _run(transforms_tt, initial_tt, compute_kernel_config=explicit_config, actual_start=zero_actual_start)
    assert device.num_program_cache_entries() == entries
    for name, implicit_output, explicit_output in zip(("entry", "final"), implicit, explicit, strict=True):
        assert_bit_identical(
            ttnn.to_torch(implicit_output),
            ttnn.to_torch(explicit_output),
            name=f"{name} implicit vs explicit compute defaults",
        )


def test_chain_affine_transforms_matches_composed_ttnn_baseline(zero_actual_start, device: ttnn.Device) -> None:
    host = _host_inputs(3, 64, 96, seed=117)
    compute_config = _production_compute_config(device)
    fused = _run(*_device_inputs(*host, device), compute_kernel_config=compute_config, actual_start=zero_actual_start)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        composed = _composed_ttnn_baseline(*host, device, compute_config)
    ttnn.synchronize_device(device)
    # Same arithmetic as the composed chain: an FP32-accumulated matmul over the whole key dimension and an FP32 add.
    assert_bit_identical(ttnn.to_torch(composed).squeeze(0), ttnn.to_torch(fused[1]), name="fused vs composed TTNN")


def test_chain_affine_transforms_rejects_unsupported_compute_config(
    zero_actual_start, device: ttnn.Device, expect_error: Callable
) -> None:
    case = _SMALL_CASE
    transforms_tt, initial_tt = _device_inputs(*_host_inputs(case.batch_heads, case.key_dim, case.value_dim), device)
    unsupported_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    with expect_error(RuntimeError, "packer_l1_acc=true is unsupported"):
        _run(transforms_tt, initial_tt, compute_kernel_config=unsupported_config, actual_start=zero_actual_start)


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_chain_affine_transforms_production_performance(zero_actual_start, device: ttnn.Device) -> None:
    case = _PRODUCTION_PERF_CASE
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for affine-transform chaining performance checks")

    host = _host_inputs(case.batch_heads, case.key_dim, case.value_dim, seed=117)
    expected = _oracle(*host)
    transforms_tt, initial_tt = _device_inputs(*host, device)
    production_compute_config = _production_compute_config(device)

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return _run(
            transforms_tt,
            initial_tt,
            compute_kernel_config=production_compute_config,
            actual_start=zero_actual_start,
        )

    outputs, perf_record = profile_realtime_program(device, run)
    duration_ns = perf_record["duration_ns"]
    assert all(tuple(output.shape) == (case.batch_heads, case.key_dim, case.value_dim) for output in outputs)
    assert all(output.dtype == ttnn.float32 for output in outputs)
    for name, golden, output in zip(("entry", "final"), expected, outputs, strict=True):
        assert_accurate(golden, ttnn.to_torch(output), name=f"production chained {name}", pcc_threshold=0.999)
    performance = _chain_affine_transforms_performance(
        transforms_tt,
        initial_tt,
        outputs,
        measured_ns=duration_ns,
        math_fidelity=ttnn.MathFidelity.HiFi2,
    )
    logger.info(
        f"affine-transform chaining {case.case_id}: measured_ns={duration_ns:.0f}, "
        f"runtime_id={perf_record['runtime_id']}, work={performance.work}, "
        f"ideal_fpu_ns={performance.ideal_fpu_ns:.2f}, ideal_dram_ns={performance.ideal_dram_ns:.2f}, "
        f"ideal_ns={performance.ideal_ns:.2f}, "
        f"fpu_utilization_pct={performance.fpu_utilization_pct:.2f}, "
        f"dram_utilization_pct={performance.dram_utilization_pct:.2f}, "
        f"utilization_pct={performance.utilization_pct:.2f}"
    )
    lower = _PRODUCTION_PERF_EXPECTED_DURATION_NS * (1 - _PRODUCTION_PERF_MARGIN)
    upper = _PRODUCTION_PERF_EXPECTED_DURATION_NS * (1 + _PRODUCTION_PERF_MARGIN)
    assert lower <= duration_ns <= upper, (
        f"{case.case_id} duration {duration_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}] ns "
        f"(reference {_PRODUCTION_PERF_EXPECTED_DURATION_NS} ns, margin +/- {_PRODUCTION_PERF_MARGIN * 100:.0f}%)"
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("host_transforms", "transforms must be an allocated device tensor"),
        ("host_initial_state", "initial_state must be an allocated device tensor"),
        ("layout", "transforms must use TILE layout"),
        ("transforms_dtype", "transforms must be BFLOAT16"),
        ("initial_state_dtype", "initial_state must be FLOAT32"),
        ("transforms_rank", "transforms must be rank 4"),
        ("initial_state_rank", "initial_state must be rank 3"),
        ("shape", "must match initial_state"),
        ("unaligned", "K and V must be positive and tile aligned"),
        ("steps", "transforms must hold one transition per sequence-parallel rank"),
        ("local_rows", "local_rows must be positive and 32-aligned"),
        ("bf16_dest", "fp32_dest_acc_en must be enabled"),
        ("l1", "need"),
    ],
)
def test_chain_affine_transforms_rejects_invalid_inputs(
    zero_actual_start,
    device: ttnn.Device,
    expect_error: Callable,
    case: str,
    message: str,
) -> None:
    a, b, initial = _host_inputs(1, 32, 32)
    packed = torch.cat([a, b], dim=-1)
    transforms_tt, initial_tt = _device_inputs(a, b, initial, device)
    local_rows = 32
    compute_kernel_config = None

    if case == "host_transforms":
        transforms_tt = ttnn.from_torch(packed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    elif case == "host_initial_state":
        initial_tt = ttnn.from_torch(initial, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    elif case == "layout":
        transforms_tt = _to_device(packed, device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "transforms_dtype":
        transforms_tt = _to_device(packed, device, ttnn.float32)
    elif case == "initial_state_dtype":
        initial_tt = _to_device(initial, device, ttnn.bfloat16)
    elif case == "transforms_rank":
        transforms_tt = _to_device(packed[0], device, ttnn.bfloat16)
    elif case == "initial_state_rank":
        initial_tt = _to_device(initial.unsqueeze(0), device)
    elif case == "shape":
        transforms_tt = _to_device(packed[..., :32], device, ttnn.bfloat16)
    elif case == "unaligned":
        transforms_tt = _to_device(packed[..., :63], device, ttnn.bfloat16)
        initial_tt = _to_device(initial[..., :31], device)
    elif case == "steps":
        transforms_tt = _to_device(torch.cat([packed, packed]), device, ttnn.bfloat16)
    elif case == "local_rows":
        local_rows = 48
    elif case == "bf16_dest":
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(), fp32_dest_acc_en=False, packer_l1_acc=False
        )
    elif case == "l1":
        transforms_tt = _to_device(torch.zeros(1, 1, 512, 1024), device, ttnn.bfloat16)
        initial_tt = _to_device(torch.zeros(1, 512, 512), device)
    with expect_error(RuntimeError, message):
        _run(
            transforms_tt,
            initial_tt,
            actual_start=zero_actual_start,
            local_rows=local_rows,
            compute_kernel_config=compute_kernel_config,
        )


def test_chain_affine_transforms_rejects_excess_batch_heads(
    zero_actual_start, device: ttnn.Device, expect_error: Callable
) -> None:
    grid = device.compute_with_storage_grid_size()
    worker_limit = grid.x * grid.y
    transforms_tt, initial_tt = _device_inputs(*_host_inputs(worker_limit + 1, 32, 32), device)

    with expect_error(RuntimeError, f"supports at most {worker_limit} batch-heads on this device"):
        _run(transforms_tt, initial_tt, actual_start=zero_actual_start)


def test_chain_affine_transforms_rejects_invalid_configuration(
    zero_actual_start, device: ttnn.Device, expect_error: Callable
) -> None:
    case = _SMALL_CASE
    transforms_tt, initial_tt = _device_inputs(*_host_inputs(case.batch_heads, case.key_dim, case.value_dim), device)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [case.batch_heads * case.key_dim, case.value_dim],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    with expect_error(RuntimeError, "output memory layout must be INTERLEAVED, got HEIGHT_SHARDED"):
        _run(transforms_tt, initial_tt, memory_config=sharded, actual_start=zero_actual_start)
