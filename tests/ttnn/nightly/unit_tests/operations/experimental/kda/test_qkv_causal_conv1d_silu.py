# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA QKV causal Conv1D plus SiLU."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_HISTORY_EXCHANGE_ROWS, KDA_HISTORY_EXCHANGE_SEMAPHORES
from tt_py_test_utils_common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import kda_performance_model_test_utils as perf_model
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    collect_accuracy_and_determinism_results,
    assert_equal,
)

from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    qkv_to_device,
    qkv_device_inputs,
    qkv_reference,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device({"l1_small_size": 24576}),
]

_SEQUENCE = 64
_DEFAULT_WIDTHS = (512, 512, 512)


@dataclass(frozen=True)
class _BenchmarkCase:
    case_id: str
    widths: tuple[int, int, int]
    channel_chunk_size: int
    expected_duration_ns: int


_PRODUCTION_PERF_MARGIN = 0.05

_PRODUCTION_CASES = (
    _BenchmarkCase("single-block", widths=(512, 512, 512), channel_chunk_size=1536, expected_duration_ns=63_900),
    _BenchmarkCase("multiple-blocks", widths=(1024, 1024, 1024), channel_chunk_size=768, expected_duration_ns=39_500),
    _BenchmarkCase("asymmetric-split", widths=(512, 256, 128), channel_chunk_size=896, expected_duration_ns=43_300),
)


def _qkv_causal_conv1d_silu_ops(
    input_tensor: torch.Tensor | ttnn.Tensor,
    history: torch.Tensor | ttnn.Tensor,
    taps: tuple[torch.Tensor | ttnn.Tensor, ...],
    outputs: tuple[torch.Tensor | ttnn.Tensor, ...],
) -> tuple[perf_model.FpuOps, perf_model.SfpuOps]:
    if len(taps) != 4 or len(outputs) != 3:
        raise ValueError("QKV causal Conv1D plus SiLU requires four taps and three outputs")
    tensors = (input_tensor, history, *taps, *outputs)
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in tensors):
        raise ValueError("QKV causal Conv1D plus SiLU tensor shapes must be positive")
    if (
        len(input_tensor.shape) != 3
        or len(history.shape) != 3
        or any(len(tap.shape) == 0 for tap in taps)
        or any(len(output.shape) != 3 for output in outputs)
    ):
        raise ValueError("QKV causal Conv1D plus SiLU tensor shapes are inconsistent")

    batch, sequence, width = input_tensor.shape
    if (
        history.shape != (batch, 3, width)
        or any(tap.shape[-1] != width or math.prod(tap.shape) != width for tap in taps)
        or any(output.shape != (batch, sequence, output.shape[-1]) for output in outputs)
        or sum(output.shape[-1] for output in outputs) != width
    ):
        raise ValueError("QKV causal Conv1D plus SiLU tensor shapes are inconsistent")

    elements = batch * sequence * width
    return (
        perf_model.FpuOps(multiply_ops=4 * elements, add_ops=3 * elements),
        perf_model.SfpuOps(silu_ops=elements),
    )


def _qkv_causal_conv1d_silu_performance(
    input_tensor: ttnn.Tensor,
    history: ttnn.Tensor,
    taps: tuple[ttnn.Tensor, ...],
    outputs: tuple[ttnn.Tensor, ...],
    *,
    measured_ns: float,
    math_fidelity: ttnn.MathFidelity,
) -> perf_model.KdaPerformance:
    fpu, sfpu = _qkv_causal_conv1d_silu_ops(input_tensor, history, taps, outputs)
    return perf_model.performance(
        fpu=fpu,
        sfpu=sfpu,
        inputs=(input_tensor, history, *taps),
        outputs=outputs,
        measured_ns=measured_ns,
        math_fidelity=math_fidelity,
    )


def test_qkv_causal_conv1d_silu_work_golden() -> None:
    fpu, sfpu = _qkv_causal_conv1d_silu_ops(
        torch.empty((1, 2, 4)),
        torch.empty((1, 3, 4)),
        tuple(torch.empty((1, 1, 4)) for _ in range(4)),
        (torch.empty((1, 2, 1)), torch.empty((1, 2, 2)), torch.empty((1, 2, 1))),
    )

    assert fpu == perf_model.FpuOps(multiply_ops=32, add_ops=24)
    assert sfpu == perf_model.SfpuOps(silu_ops=8)


def _run(
    input_tt: ttnn.Tensor,
    history_tt: ttnn.Tensor,
    taps_tt: tuple[ttnn.Tensor, ...],
    *,
    actual_start: ttnn.Tensor,
    channel_chunk_size: int,
    widths: tuple[int, int, int] = _DEFAULT_WIDTHS,
    memory_config: ttnn.MemoryConfig | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    return ttnn.experimental.kda.qkv_causal_conv1d_silu(
        input_tt,
        history_tt,
        *taps_tt,
        *widths,
        actual_start=actual_start,
        predecessor_carry=history_tt,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=channel_chunk_size),
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )


@pytest.mark.parametrize(
    ("widths", "channel_chunk_size", "full_contract"),
    [
        pytest.param((512, 512, 512), 1536, False, id="single-block"),
        pytest.param((1024, 1024, 1024), 768, False, id="multiple-blocks"),
        pytest.param((512, 256, 128), 896, True, id="asymmetric-split-full-contract"),
    ],
)
def test_qkv_causal_conv1d_silu_contract(
    zero_actual_start,
    device: ttnn.Device,
    widths: tuple[int, int, int],
    channel_chunk_size: int,
    full_contract: bool,
) -> None:
    """Cover every geometry numerically and the invariant output/trace contract once."""
    host, device_inputs = qkv_device_inputs(device, widths=widths)
    inputs, history, taps = host
    input_tt, history_tt, taps_tt = device_inputs
    expected = qkv_reference(inputs, history, taps, widths)
    input_tensors = (input_tt, history_tt, *taps_tt)
    snapshots = tuple(ttnn.to_torch(tensor).clone() for tensor in input_tensors) if full_contract else ()

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        with ttnn.manage_config("throw_exception_on_fallback", True):
            return _run(
                input_tt,
                history_tt,
                taps_tt,
                widths=widths,
                channel_chunk_size=channel_chunk_size,
                actual_start=zero_actual_start,
            )

    outputs = run()
    for output, width in zip(outputs, widths, strict=True):
        assert tuple(ttnn.to_torch(output).shape) == (1, _SEQUENCE, width)
        if full_contract:
            assert output.dtype == ttnn.bfloat16
            assert output.layout == ttnn.TILE_LAYOUT
            assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            assert all(output.buffer_address() != tensor.buffer_address() for tensor in input_tensors)

    for name, golden, output in zip(("q", "k", "v"), expected, outputs, strict=True):
        actual = ttnn.to_torch(output)
        assert_accurate(golden, actual, name=name, pcc_threshold=0.999, nrmse_threshold=0.008)

    if full_contract:
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        traced_outputs = run()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        for _ in range(2):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)

        for name, output, traced in zip(("q", "k", "v"), outputs, traced_outputs, strict=True):
            assert_bit_identical(ttnn.to_torch(output), ttnn.to_torch(traced), name=f"{name} trace replay")
        for name, before, tensor in zip(
            ("input", "history", "tap0", "tap1", "tap2", "tap3"), snapshots, input_tensors, strict=True
        ):
            assert_bit_identical(before, ttnn.to_torch(tensor), name=f"{name} immutability")
        ttnn.release_trace(device, trace_id)


@pytest.mark.parametrize(
    ("sequence", "channel_chunk_size"),
    [
        # 4 channel blocks x 65 row tiles = 260 work items: cores own several items, and on 110-140 core grids
        # some own items on both sides of a channel-block boundary.
        pytest.param(2080, 768, id="items-cross-blocks"),
        # 6- and 3-tile channel blocks: compute groups two tiles and one tile per destination acquire.
        pytest.param(64, 192, id="two-tile-groups"),
        pytest.param(64, 96, id="one-tile-groups"),
    ],
)
def test_qkv_causal_conv1d_silu_work_distribution(
    zero_actual_start, device: ttnn.Device, sequence: int, channel_chunk_size: int
) -> None:
    """Weight reuse across a core's work items and every destination grouping match the reference."""
    widths = (1024, 1024, 1024)
    host, device_inputs = qkv_device_inputs(device, sequence=sequence, widths=widths)
    inputs, history, taps = host
    input_tt, history_tt, taps_tt = device_inputs
    outputs = _run(
        input_tt,
        history_tt,
        taps_tt,
        widths=widths,
        channel_chunk_size=channel_chunk_size,
        actual_start=zero_actual_start,
    )
    for name, golden, output in zip(
        ("q", "k", "v"), qkv_reference(inputs, history, taps, widths), outputs, strict=True
    ):
        assert_accurate(golden, ttnn.to_torch(output), name=name, pcc_threshold=0.999, nrmse_threshold=0.008)


@pytest.mark.parametrize("case", _PRODUCTION_CASES, ids=lambda case: case.case_id)
def test_qkv_causal_conv1d_silu_is_device_deterministic(
    zero_actual_start, device: ttnn.Device, case: _BenchmarkCase
) -> None:
    """Compare repeated large outputs on device; cache behavior is tested separately."""
    host, (input_tt, history_tt, taps_tt) = qkv_device_inputs(device, widths=case.widths)
    expected = qkv_reference(*host, case.widths)

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        with ttnn.manage_config("throw_exception_on_fallback", True):
            return _run(
                input_tt,
                history_tt,
                taps_tt,
                widths=case.widths,
                channel_chunk_size=case.channel_chunk_size,
                actual_start=zero_actual_start,
            )

    output_tensors, outputs, mismatch_marker = collect_accuracy_and_determinism_results(device, run)
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name=f"{case.case_id} device-side exact-value determinism marker",
    )
    for name, golden, output in zip(("q", "k", "v"), expected, outputs, strict=True):
        assert_accurate(golden, output, name=f"{case.case_id} {name}", pcc_threshold=0.999, nrmse_threshold=0.008)
    for output in output_tensors:
        ttnn.deallocate(output)


def test_qkv_causal_conv1d_silu_cache_hit_rebinds_fresh_tensors(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    widths = (128, 128, 128)
    host_a, device_inputs_a = qkv_device_inputs(device, widths=widths, sequence=32, seed=1911)
    host_b, device_inputs_b = qkv_device_inputs(device, widths=widths, sequence=32, seed=1912)

    output_a = _run(*device_inputs_a, widths=widths, channel_chunk_size=384, actual_start=zero_actual_start)
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    output_b = _run(*device_inputs_b, widths=widths, channel_chunk_size=384, actual_start=zero_actual_start)
    ttnn.synchronize_device(device)

    flat_inputs_a = (device_inputs_a[0], device_inputs_a[1], *device_inputs_a[2])
    flat_inputs_b = (device_inputs_b[0], device_inputs_b[1], *device_inputs_b[2])
    assert device.num_program_cache_entries() == entries
    assert all(
        tensor_a.buffer_address() != tensor_b.buffer_address()
        for tensor_a, tensor_b in zip(flat_inputs_a, flat_inputs_b, strict=True)
    )
    assert all(
        tensor_a.buffer_address() != tensor_b.buffer_address()
        for tensor_a, tensor_b in zip(output_a, output_b, strict=True)
    )

    expected_a = qkv_reference(*host_a, widths)
    expected_b = qkv_reference(*host_b, widths)
    for name, golden_a, golden_b, actual_a_tt, actual_b_tt in zip(
        ("q", "k", "v"), expected_a, expected_b, output_a, output_b, strict=True
    ):
        actual_a = ttnn.to_torch(actual_a_tt)
        actual_b = ttnn.to_torch(actual_b_tt)
        assert_accurate(
            golden_a, actual_a, name=f"{name} cache miss tensors", pcc_threshold=0.999, nrmse_threshold=0.008
        )
        assert_accurate(
            golden_b, actual_b, name=f"{name} cache hit fresh tensors", pcc_threshold=0.999, nrmse_threshold=0.008
        )
        assert not torch.equal(actual_a, actual_b)


def test_qkv_causal_conv1d_silu_default_compute_config_matches_explicit_defaults(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    _, (input_tt, history_tt, taps_tt) = qkv_device_inputs(device, sequence=32, widths=(128, 128, 128), seed=817)
    implicit = _run(
        input_tt, history_tt, taps_tt, widths=(128, 128, 128), channel_chunk_size=384, actual_start=zero_actual_start
    )
    entries = device.num_program_cache_entries()
    explicit_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
        throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
    )
    explicit = _run(
        input_tt,
        history_tt,
        taps_tt,
        widths=(128, 128, 128),
        channel_chunk_size=384,
        compute_kernel_config=explicit_config,
        actual_start=zero_actual_start,
    )
    assert device.num_program_cache_entries() == entries
    for name, implicit_output, explicit_output in zip(("q", "k", "v"), implicit, explicit, strict=True):
        assert_bit_identical(
            ttnn.to_torch(implicit_output),
            ttnn.to_torch(explicit_output),
            name=f"{name} implicit vs explicit production compute defaults",
        )


@pytest.mark.parametrize(
    ("widths", "channel_chunk_size"),
    [
        # Two channel tiles per block: one destination group, eight diagonal tiles in two FP32 groups.
        pytest.param((128, 128, 128), 64, id="kimi-k3-block"),
        # Twelve channel tiles per block: three FP32 groups of four, against two BFLOAT16 groups of six.
        pytest.param((128, 128, 128), 384, id="wide-block"),
    ],
)
def test_qkv_causal_conv1d_silu_fp32_accumulation(
    zero_actual_start, device: ttnn.Device, widths: tuple[int, int, int], channel_chunk_size: int
) -> None:
    """FP32 destination accumulation halves the tiles per destination acquire and still matches the reference."""
    (inputs, history, taps), (input_tt, history_tt, taps_tt) = qkv_device_inputs(
        device, sequence=64, widths=widths, seed=819
    )
    fp32_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    outputs = _run(
        input_tt,
        history_tt,
        taps_tt,
        widths=widths,
        channel_chunk_size=channel_chunk_size,
        compute_kernel_config=fp32_config,
        actual_start=zero_actual_start,
    )
    for name, golden, output in zip(("q", "k", "v"), qkv_reference(inputs, history, taps, widths), outputs):
        assert_accurate(golden, ttnn.to_torch(output), name=f"fp32 {name}", pcc_threshold=0.999, nrmse_threshold=0.008)


def test_qkv_causal_conv1d_silu_rejects_approximate_math(
    zero_actual_start, device: ttnn.Device, expect_error: Callable
) -> None:
    _, (input_tt, history_tt, taps_tt) = qkv_device_inputs(device, sequence=32, widths=(128, 128, 128), seed=818)
    approximate_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    with expect_error(RuntimeError, "math_approx_mode=true is unsupported"):
        _run(
            input_tt,
            history_tt,
            taps_tt,
            widths=(128, 128, 128),
            channel_chunk_size=384,
            compute_kernel_config=approximate_config,
            actual_start=zero_actual_start,
        )


def test_qkv_causal_conv1d_silu_rejects_unsupported_compute_config(
    zero_actual_start, device: ttnn.Device, expect_error: Callable
) -> None:
    _, (input_tt, history_tt, taps_tt) = qkv_device_inputs(device, sequence=32, widths=(128, 128, 128))
    unsupported_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        packer_l1_acc=True,
    )
    with expect_error(RuntimeError, "packer_l1_acc=true is unsupported"):
        _run(
            input_tt,
            history_tt,
            taps_tt,
            widths=(128, 128, 128),
            channel_chunk_size=384,
            compute_kernel_config=unsupported_config,
            actual_start=zero_actual_start,
        )


@pytest.mark.requires_host_iommu
@pytest.mark.parametrize("case", _PRODUCTION_CASES, ids=lambda case: case.case_id)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_qkv_causal_conv1d_silu_production_performance(
    zero_actual_start, device: ttnn.Device, case: _BenchmarkCase
) -> None:
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for QKV causal Conv1D plus SiLU performance checks")

    _, (input_tt, history_tt, taps_tt) = qkv_device_inputs(device, widths=case.widths)

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        return _run(
            input_tt,
            history_tt,
            taps_tt,
            widths=case.widths,
            channel_chunk_size=case.channel_chunk_size,
            actual_start=zero_actual_start,
        )

    outputs, programs = profile_realtime_program_merged(device, run)
    (runtime_id, perf_record), *others = [
        (runtime_id, program)
        for runtime_id, program in programs.items()
        if any("reader_qkv_causal_conv1d_silu.cpp" in source for source in program["kernel_sources"])
    ]
    assert not others, "expected one convolution program in the profiled region"
    duration_ns = perf_record["duration_ns"]
    assert tuple(tuple(output.shape) for output in outputs) == tuple((1, _SEQUENCE, width) for width in case.widths)
    performance = _qkv_causal_conv1d_silu_performance(
        input_tt,
        history_tt,
        taps_tt,
        outputs,
        measured_ns=duration_ns,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )
    logger.info(
        f"QKV causal Conv1D plus SiLU {case.case_id}: measured_ns={duration_ns:.0f}, "
        f"runtime_id={runtime_id}, work={performance.work}, "
        f"ideal_fpu_ns={performance.ideal_fpu_ns:.2f}, ideal_dram_ns={performance.ideal_dram_ns:.2f}, "
        f"ideal_ns={performance.ideal_ns:.2f}, "
        f"fpu_utilization_pct={performance.fpu_utilization_pct:.2f}, "
        f"dram_utilization_pct={performance.dram_utilization_pct:.2f}, "
        f"utilization_pct={performance.utilization_pct:.2f}"
    )
    lower = case.expected_duration_ns * (1 - _PRODUCTION_PERF_MARGIN)
    upper = case.expected_duration_ns * (1 + _PRODUCTION_PERF_MARGIN)
    assert lower <= duration_ns <= upper, (
        f"{case.case_id} duration {duration_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}] ns "
        f"(reference {case.expected_duration_ns} ns, margin +/- {_PRODUCTION_PERF_MARGIN * 100:.0f}%)"
    )


# Kimi-K3's per-device convolution on Galaxy SP8xTP4: 640 rows of Q, K and V (24 heads of 128) read in place from the
# TILE input projection in L1, whose decay-rank, output-gate and beta columns follow them, with 64-channel blocks, L1
# outputs and the history exchange, which a single rank runs without the fabric.
_K3_ROWS = 640
_K3_WIDTHS = (3072, 3072, 3072)
_K3_TRAILING_COLUMNS = 128 + 3072 + 32
_K3_CHANNEL_CHUNK_SIZE = 64
_K3_EXPECTED_DURATION_NS = 61_130


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_qkv_causal_conv1d_silu_kimi_k3_performance(zero_actual_start, device: ttnn.Device) -> None:
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for QKV causal Conv1D plus SiLU performance checks")

    (inputs, history, taps), (_, history_tt, taps_tt) = qkv_device_inputs(device, sequence=_K3_ROWS, widths=_K3_WIDTHS)
    projected = torch.cat((inputs, torch.randn(1, _K3_ROWS, _K3_TRAILING_COLUMNS, dtype=inputs.dtype)), dim=-1)
    projected_tt = qkv_to_device(projected, device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    exchange_core = ttnn._ttnn.operations.experimental.kda.qkv_causal_conv1d_silu_exchange_core(
        device.compute_with_storage_grid_size()
    )
    exchange_cores = ttnn.CoreRangeSet({ttnn.CoreRange(exchange_core, exchange_core)})
    exchange_buffer = ttnn.zeros(
        (1, KDA_HISTORY_EXCHANGE_ROWS, sum(_K3_WIDTHS)),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    exchange_semaphores = [
        ttnn.create_global_semaphore(device, exchange_cores, 0, ttnn.BufferType.L1_SMALL)
        for _ in range(KDA_HISTORY_EXCHANGE_SEMAPHORES)
    ]

    def run() -> tuple[ttnn.Tensor, ...]:
        return ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry(
            projected_tt,
            history_tt,
            *taps_tt,
            *_K3_WIDTHS,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=_K3_CHANNEL_CHUNK_SIZE),
            actual_start=zero_actual_start,
            exchange_buffer=exchange_buffer,
            exchange_semaphores=exchange_semaphores,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    # Warm the program cache so the profiled call measures the cached program.
    for output in run():
        ttnn.deallocate(output)
    outputs, programs = profile_realtime_program_merged(device, run)
    # The profiler's window may also hold the warm-up call; the measured call is the last convolution program.
    *_, (runtime_id, perf_record) = [
        (runtime_id, program)
        for runtime_id, program in programs.items()
        if any("reader_qkv_causal_conv1d_silu.cpp" in source for source in program["kernel_sources"])
    ]
    duration_ns = perf_record["duration_ns"]
    q, k, v, carry = outputs
    for name, golden, output in zip(("q", "k", "v"), qkv_reference(inputs, history, taps, _K3_WIDTHS), (q, k, v)):
        assert output.memory_config().buffer_type == ttnn.BufferType.L1
        assert_accurate(
            golden, ttnn.to_torch(output), name=f"Kimi-K3 {name}", pcc_threshold=0.999, nrmse_threshold=0.008
        )
    assert_bit_identical(inputs[:, -3:, :], ttnn.to_torch(carry), name="Kimi-K3 replacement carry")
    logger.info(f"QKV causal Conv1D plus SiLU kimi-k3: measured_ns={duration_ns:.0f}, runtime_id={runtime_id}")
    lower = _K3_EXPECTED_DURATION_NS * (1 - _PRODUCTION_PERF_MARGIN)
    upper = _K3_EXPECTED_DURATION_NS * (1 + _PRODUCTION_PERF_MARGIN)
    assert lower <= duration_ns <= upper, (
        f"kimi-k3 duration {duration_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}] ns "
        f"(reference {_K3_EXPECTED_DURATION_NS} ns, margin +/- {_PRODUCTION_PERF_MARGIN * 100:.0f}%)"
    )


def test_qkv_causal_conv1d_silu_program_key_includes_split_widths(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    _, (input_tt, history_tt, taps_tt) = qkv_device_inputs(device, widths=(128, 128, 128), sequence=32, seed=772)
    _run(input_tt, history_tt, taps_tt, widths=(128, 128, 128), channel_chunk_size=384, actual_start=zero_actual_start)
    entries = device.num_program_cache_entries()
    _run(input_tt, history_tt, taps_tt, widths=(64, 128, 192), channel_chunk_size=384, actual_start=zero_actual_start)
    assert device.num_program_cache_entries() == entries + 1
    _run(input_tt, history_tt, taps_tt, widths=(64, 128, 192), channel_chunk_size=384, actual_start=zero_actual_start)
    assert device.num_program_cache_entries() == entries + 1


def test_qkv_causal_conv1d_silu_program_key_includes_channel_chunk_size(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    widths = (128, 128, 128)
    _, (input_tt, history_tt, taps_tt) = qkv_device_inputs(device, widths=widths, sequence=32, seed=773)
    _run(input_tt, history_tt, taps_tt, widths=widths, channel_chunk_size=384, actual_start=zero_actual_start)
    entries = device.num_program_cache_entries()
    _run(input_tt, history_tt, taps_tt, widths=widths, channel_chunk_size=192, actual_start=zero_actual_start)
    assert device.num_program_cache_entries() == entries + 1
    _run(input_tt, history_tt, taps_tt, widths=widths, channel_chunk_size=192, actual_start=zero_actual_start)
    assert device.num_program_cache_entries() == entries + 1


@pytest.mark.parametrize(
    ("channel_chunk_size", "message"),
    [
        (0, "channel_chunk_size must be positive"),
        (16, "channel_chunk_size must be tile aligned"),
        (416, r"channel_chunk_size must not exceed Q\+K\+V width"),
        (160, r"channel_chunk_size must divide Q\+K\+V width exactly"),
    ],
)
def test_qkv_causal_conv1d_silu_rejects_invalid_channel_chunk_size(
    zero_actual_start, device: ttnn.Device, expect_error: Callable, channel_chunk_size: int, message: str
) -> None:
    widths = (128, 128, 128)
    _, (input_tt, history_tt, taps_tt) = qkv_device_inputs(device, widths=widths, sequence=32)
    with expect_error(RuntimeError, message):
        _run(
            input_tt,
            history_tt,
            taps_tt,
            widths=widths,
            channel_chunk_size=channel_chunk_size,
            actual_start=zero_actual_start,
        )


def test_qkv_causal_conv1d_silu_rejects_channel_chunk_size_over_l1(
    zero_actual_start, device: ttnn.Device, expect_error: Callable
) -> None:
    # One 96-tile channel block needs about 2.4 MB of DFBs and window per core.
    widths = (1024, 1024, 1024)
    _, (input_tt, history_tt, taps_tt) = qkv_device_inputs(device, widths=widths, sequence=32)
    with expect_error(RuntimeError, "channel_chunk_size = 3072 needs .* bytes of L1 per core"):
        _run(
            input_tt,
            history_tt,
            taps_tt,
            widths=widths,
            channel_chunk_size=3072,
            actual_start=zero_actual_start,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("host_input", "allocated device tensor"),
        ("batch", r"input must be \[1,T,W\]"),
        ("history_shape", r"history must be \[1,3,Q\+K\+V\]"),
        ("tap_last_dimension", r"tap2 last dimension must equal Q\+K\+V"),
        ("tap_volume", "tap2 logical volume must equal"),
        ("history_layout", "history must use ROW_MAJOR layout, got Layout::TILE"),
        ("tap_layout", "tap1 must use TILE layout, got Layout::ROW_MAJOR"),
        ("input_dtype", "input must be BFLOAT16"),
        ("history_dtype", "history must be BFLOAT16"),
        ("tap_dtype", "tap3 must be BFLOAT16"),
        ("sequence_alignment", "sequence must be positive and tile aligned"),
        ("sharded_history", "history must use interleaved memory"),
    ],
)
def test_qkv_causal_conv1d_silu_rejects_invalid_tensors(
    zero_actual_start, device: ttnn.Device, expect_error: Callable, case: str, message: str
) -> None:
    batch = 2 if case == "batch" else 1
    history_rows = 2 if case == "history_shape" else 3
    sequence = 33 if case == "sequence_alignment" else 32
    host, device_inputs = qkv_device_inputs(
        device,
        widths=(128, 128, 128),
        batch=batch,
        history_rows=history_rows,
        sequence=sequence,
        seed=991,
    )
    inputs, history, taps = host
    input_tt, history_tt, taps_tt = device_inputs
    taps_list = list(taps_tt)

    if case == "host_input":
        input_tt = ttnn.from_torch(inputs, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "tap_last_dimension":
        taps_list[2] = qkv_to_device(taps[2].reshape(-1, 1), device, layout=ttnn.TILE_LAYOUT)
    elif case == "tap_volume":
        taps_list[2] = qkv_to_device(torch.cat((taps[2], taps[2]), dim=0), device, layout=ttnn.TILE_LAYOUT)
    elif case == "history_layout":
        history_tt = qkv_to_device(history, device, layout=ttnn.TILE_LAYOUT)
    elif case == "tap_layout":
        taps_list[1] = qkv_to_device(taps[1], device, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "input_dtype":
        input_tt = qkv_to_device(inputs.float(), device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "history_dtype":
        history_tt = qkv_to_device(history.float(), device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "tap_dtype":
        taps_list[3] = qkv_to_device(taps[3].float(), device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    elif case == "sharded_history":
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [history.shape[0] * history.shape[1], history.shape[2]],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        history_tt = qkv_to_device(history, device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=sharded_config)

    with expect_error(RuntimeError, message):
        _run(
            input_tt,
            history_tt,
            tuple(taps_list),
            widths=(128, 128, 128),
            channel_chunk_size=384,
            actual_start=zero_actual_start,
        )


@pytest.mark.parametrize(
    ("widths", "message"),
    [
        ((0, 128, 256), "Q/K/V widths must be positive"),
        ((100, 128, 156), "Q/K/V widths must be tile aligned"),
        ((128, 128, 256), r"input must be \[1,T,W\]"),
    ],
)
def test_qkv_causal_conv1d_silu_rejects_invalid_widths(
    zero_actual_start, device: ttnn.Device, expect_error: Callable, widths: tuple[int, int, int], message: str
) -> None:
    _, (input_tt, history_tt, taps_tt) = qkv_device_inputs(device, widths=(128, 128, 128), sequence=32)
    with expect_error(RuntimeError, message):
        _run(
            input_tt, history_tt, taps_tt, widths=widths, channel_chunk_size=sum(widths), actual_start=zero_actual_start
        )


def test_qkv_causal_conv1d_silu_rejects_sharded_output(
    zero_actual_start, device: ttnn.Device, expect_error: Callable
) -> None:
    _, (input_tt, history_tt, taps_tt) = qkv_device_inputs(device, widths=(128, 128, 128), sequence=32)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [32, 128],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    with expect_error(RuntimeError, "output memory layout must be INTERLEAVED, got HEIGHT_SHARDED"):
        _run(
            input_tt,
            history_tt,
            taps_tt,
            widths=(128, 128, 128),
            channel_chunk_size=384,
            memory_config=sharded_config,
            actual_start=zero_actual_start,
        )


@pytest.mark.parametrize(
    ("widths", "channel_chunk_size", "column_offset", "trailing_columns", "sequence"),
    [
        pytest.param((512, 512, 512), 64, 0, 0, 64, id="exact-width"),
        pytest.param((256, 128, 128), 128, 32, 96, 96, id="column-window"),
        pytest.param((128, 128, 128), 384, 64, 32, 32, id="one-tile-row"),
    ],
)
def test_qkv_causal_conv1d_silu_tiled_input(
    zero_actual_start,
    device: ttnn.Device,
    widths: tuple[int, int, int],
    channel_chunk_size: int,
    column_offset: int,
    trailing_columns: int,
    sequence: int,
) -> None:
    """A TILE input read in place from a wider tensor matches the reference; the exchanging variant, on a single rank,
    also returns the replacement carry."""
    host, device_inputs = qkv_device_inputs(device, widths=widths, sequence=sequence)
    inputs, history, taps = host
    _, history_tt, taps_tt = device_inputs
    rows = inputs.shape[1]
    channels = sum(widths)
    torch.manual_seed(17)
    wide = torch.cat(
        (
            torch.randn(1, rows, column_offset, dtype=inputs.dtype),
            inputs,
            torch.randn(1, rows, trailing_columns, dtype=inputs.dtype),
        ),
        dim=-1,
    )
    wide_tt = qkv_to_device(wide, device, layout=ttnn.TILE_LAYOUT)
    program_config = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=channel_chunk_size)
    expected = qkv_reference(inputs, history, taps, widths)

    q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
        wide_tt,
        history_tt,
        *taps_tt,
        *widths,
        program_config=program_config,
        actual_start=zero_actual_start,
        predecessor_carry=history_tt,
        input_column_offset=column_offset,
    )
    for name, golden, output in zip(("q", "k", "v"), expected, (q, k, v), strict=True):
        assert_accurate(golden, ttnn.to_torch(output), name=f"tiled {name}", pcc_threshold=0.999, nrmse_threshold=0.008)

    exchange_core = ttnn._ttnn.operations.experimental.kda.qkv_causal_conv1d_silu_exchange_core(
        device.compute_with_storage_grid_size()
    )
    exchange_cores = ttnn.CoreRangeSet({ttnn.CoreRange(exchange_core, exchange_core)})
    exchange_buffer = ttnn.zeros(
        (1, KDA_HISTORY_EXCHANGE_ROWS, channels), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    q, k, v, carry = ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry(
        wide_tt,
        history_tt,
        *taps_tt,
        *widths,
        program_config=program_config,
        actual_start=zero_actual_start,
        exchange_buffer=exchange_buffer,
        exchange_semaphores=[
            ttnn.create_global_semaphore(device, exchange_cores, 0) for _ in range(KDA_HISTORY_EXCHANGE_SEMAPHORES)
        ],
        input_column_offset=column_offset,
    )
    for name, golden, output in zip(("q", "k", "v"), expected, (q, k, v), strict=True):
        assert_accurate(
            golden, ttnn.to_torch(output), name=f"with-carry {name}", pcc_threshold=0.999, nrmse_threshold=0.008
        )
    # One rank: the carry is the last three tokens.
    last_rows = inputs[:, -3:, :]
    assert_bit_identical(last_rows, ttnn.to_torch(carry), name="replacement carry")
