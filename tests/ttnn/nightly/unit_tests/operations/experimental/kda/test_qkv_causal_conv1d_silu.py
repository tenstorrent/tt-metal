# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA QKV causal Conv1D plus SiLU."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import kda_performance_model_test_utils as perf_model
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    collect_accuracy_and_determinism_results,
    assert_equal,
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

# Recalibrated 2026-08-19 on Blackhole P150b device 0, firmware 19.5.0.0. The
# previous references (92_606, 50_123, 55_324) were calibrated before the
# hoisted unpack/init optimization, which made every shape 3.7-4.6% faster with
# bit-identical outputs. Against a symmetric band that left each case only
# 335-672 ns above its lower bound, so a further improvement would have failed
# the gate for being too fast. Seven real-time-profiler samples of the current
# implementation produced 88358-88443 ns, 47952-48257 ns, and 53230-53373 ns;
# the inline references are their medians. The 5% symmetric margin now leaves
# 2.2-4.4 us on both sides, against an observed spread of 85-305 ns.
_PRODUCTION_CASES = (
    _BenchmarkCase("single-block", widths=(512, 512, 512), channel_chunk_size=1536, expected_duration_ns=88_383),
    _BenchmarkCase("multiple-blocks", widths=(1024, 1024, 1024), channel_chunk_size=768, expected_duration_ns=48_090),
    _BenchmarkCase("asymmetric-split", widths=(512, 256, 128), channel_chunk_size=896, expected_duration_ns=53_270),
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


def _host_inputs(
    *,
    sequence: int = _SEQUENCE,
    widths: tuple[int, int, int] = _DEFAULT_WIDTHS,
    batch: int = 1,
    history_rows: int = 3,
    seed: int = 223,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    generator = torch.Generator().manual_seed(seed)
    channels = sum(widths)
    inputs = torch.randn(batch, sequence, channels, generator=generator, dtype=torch.bfloat16)
    history = torch.randn(batch, history_rows, channels, generator=generator, dtype=torch.bfloat16)
    taps = tuple(torch.randn(1, 1, channels, generator=generator, dtype=torch.bfloat16) for _ in range(4))
    return inputs, history, taps


def _to_device(
    tensor: torch.Tensor,
    device: ttnn.Device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _device_inputs(
    device: ttnn.Device,
    *,
    sequence: int = _SEQUENCE,
    widths: tuple[int, int, int] = _DEFAULT_WIDTHS,
    batch: int = 1,
    history_rows: int = 3,
    seed: int = 223,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]],
    tuple[ttnn.Tensor, ttnn.Tensor, tuple[ttnn.Tensor, ...]],
]:
    host = _host_inputs(
        sequence=sequence,
        widths=widths,
        batch=batch,
        history_rows=history_rows,
        seed=seed,
    )
    inputs, history, taps = host
    return host, (
        _to_device(inputs, device, layout=ttnn.ROW_MAJOR_LAYOUT),
        _to_device(history, device, layout=ttnn.ROW_MAJOR_LAYOUT),
        tuple(_to_device(tap, device, layout=ttnn.TILE_LAYOUT) for tap in taps),
    )


def _reference(
    inputs: torch.Tensor,
    history: torch.Tensor,
    taps: tuple[torch.Tensor, ...],
    widths: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    window = torch.cat((history, inputs), dim=1)
    convolved = sum(window[:, tap : tap + inputs.shape[1]] * taps[tap] for tap in range(4))
    return F.silu(convolved).split(widths, dim=-1)


def _run(
    input_tt: ttnn.Tensor,
    history_tt: ttnn.Tensor,
    taps_tt: tuple[ttnn.Tensor, ...],
    *,
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
    device: ttnn.Device,
    widths: tuple[int, int, int],
    channel_chunk_size: int,
    full_contract: bool,
) -> None:
    """Cover every geometry numerically and the invariant output/trace contract once."""
    host, device_inputs = _device_inputs(device, widths=widths)
    inputs, history, taps = host
    input_tt, history_tt, taps_tt = device_inputs
    expected = _reference(inputs, history, taps, widths)
    input_tensors = (input_tt, history_tt, *taps_tt)
    snapshots = tuple(ttnn.to_torch(tensor).clone() for tensor in input_tensors) if full_contract else ()

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        with ttnn.manage_config("throw_exception_on_fallback", True):
            return _run(input_tt, history_tt, taps_tt, widths=widths, channel_chunk_size=channel_chunk_size)

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
        assert_accurate(golden, actual, name=name, pcc_threshold=0.999)

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


@pytest.mark.parametrize("case", _PRODUCTION_CASES, ids=lambda case: case.case_id)
def test_qkv_causal_conv1d_silu_is_device_deterministic(device: ttnn.Device, case: _BenchmarkCase) -> None:
    """Compare repeated large outputs on device; cache behavior is tested separately."""
    host, (input_tt, history_tt, taps_tt) = _device_inputs(device, widths=case.widths)
    expected = _reference(*host, case.widths)

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        with ttnn.manage_config("throw_exception_on_fallback", True):
            return _run(
                input_tt,
                history_tt,
                taps_tt,
                widths=case.widths,
                channel_chunk_size=case.channel_chunk_size,
            )

    output_tensors, outputs, mismatch_marker = collect_accuracy_and_determinism_results(device, run)
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name=f"{case.case_id} device-side exact-value determinism marker",
    )
    for name, golden, output in zip(("q", "k", "v"), expected, outputs, strict=True):
        assert_accurate(golden, output, name=f"{case.case_id} {name}", pcc_threshold=0.999)
    for output in output_tensors:
        ttnn.deallocate(output)


def test_qkv_causal_conv1d_silu_cache_hit_rebinds_fresh_tensors(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    widths = (128, 128, 128)
    host_a, device_inputs_a = _device_inputs(device, widths=widths, sequence=32, seed=1911)
    host_b, device_inputs_b = _device_inputs(device, widths=widths, sequence=32, seed=1912)

    output_a = _run(*device_inputs_a, widths=widths, channel_chunk_size=384)
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    output_b = _run(*device_inputs_b, widths=widths, channel_chunk_size=384)
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

    expected_a = _reference(*host_a, widths)
    expected_b = _reference(*host_b, widths)
    for name, golden_a, golden_b, actual_a_tt, actual_b_tt in zip(
        ("q", "k", "v"), expected_a, expected_b, output_a, output_b, strict=True
    ):
        actual_a = ttnn.to_torch(actual_a_tt)
        actual_b = ttnn.to_torch(actual_b_tt)
        assert_accurate(golden_a, actual_a, name=f"{name} cache miss tensors", pcc_threshold=0.999)
        assert_accurate(golden_b, actual_b, name=f"{name} cache hit fresh tensors", pcc_threshold=0.999)
        assert not torch.equal(actual_a, actual_b)


def test_qkv_causal_conv1d_silu_default_compute_config_matches_explicit_defaults(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    _, (input_tt, history_tt, taps_tt) = _device_inputs(device, sequence=32, widths=(128, 128, 128), seed=817)
    implicit = _run(input_tt, history_tt, taps_tt, widths=(128, 128, 128), channel_chunk_size=384)
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
    )
    assert device.num_program_cache_entries() == entries
    for name, implicit_output, explicit_output in zip(("q", "k", "v"), implicit, explicit, strict=True):
        assert_bit_identical(
            ttnn.to_torch(implicit_output),
            ttnn.to_torch(explicit_output),
            name=f"{name} implicit vs explicit production compute defaults",
        )


def test_qkv_causal_conv1d_silu_rejects_approximate_math(device: ttnn.Device, expect_error: Callable) -> None:
    _, (input_tt, history_tt, taps_tt) = _device_inputs(device, sequence=32, widths=(128, 128, 128), seed=818)
    approximate_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    with expect_error(
        RuntimeError, "math_approx_mode=true is unsupported because silu_tile always uses precise sigmoid"
    ):
        _run(
            input_tt,
            history_tt,
            taps_tt,
            widths=(128, 128, 128),
            channel_chunk_size=384,
            compute_kernel_config=approximate_config,
        )


def test_qkv_causal_conv1d_silu_rejects_unsupported_compute_config(device: ttnn.Device, expect_error: Callable) -> None:
    _, (input_tt, history_tt, taps_tt) = _device_inputs(device, sequence=32, widths=(128, 128, 128))
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
        )


@pytest.mark.requires_host_iommu
@pytest.mark.parametrize("case", _PRODUCTION_CASES, ids=lambda case: case.case_id)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_qkv_causal_conv1d_silu_production_performance(device: ttnn.Device, case: _BenchmarkCase) -> None:
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for QKV causal Conv1D plus SiLU performance checks")

    _, (input_tt, history_tt, taps_tt) = _device_inputs(device, widths=case.widths)

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        return _run(
            input_tt,
            history_tt,
            taps_tt,
            widths=case.widths,
            channel_chunk_size=case.channel_chunk_size,
        )

    outputs, perf_record = profile_realtime_program(device, run)
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
        f"runtime_id={perf_record['runtime_id']}, work={performance.work}, "
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


def test_qkv_causal_conv1d_silu_program_key_includes_split_widths(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    _, (input_tt, history_tt, taps_tt) = _device_inputs(device, widths=(128, 128, 128), sequence=32, seed=772)
    _run(input_tt, history_tt, taps_tt, widths=(128, 128, 128), channel_chunk_size=384)
    entries = device.num_program_cache_entries()
    _run(input_tt, history_tt, taps_tt, widths=(64, 128, 192), channel_chunk_size=384)
    assert device.num_program_cache_entries() == entries + 1
    _run(input_tt, history_tt, taps_tt, widths=(64, 128, 192), channel_chunk_size=384)
    assert device.num_program_cache_entries() == entries + 1


def test_qkv_causal_conv1d_silu_program_key_includes_channel_chunk_size(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    widths = (128, 128, 128)
    _, (input_tt, history_tt, taps_tt) = _device_inputs(device, widths=widths, sequence=32, seed=773)
    _run(input_tt, history_tt, taps_tt, widths=widths, channel_chunk_size=384)
    entries = device.num_program_cache_entries()
    _run(input_tt, history_tt, taps_tt, widths=widths, channel_chunk_size=192)
    assert device.num_program_cache_entries() == entries + 1
    _run(input_tt, history_tt, taps_tt, widths=widths, channel_chunk_size=192)
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
    device: ttnn.Device, expect_error: Callable, channel_chunk_size: int, message: str
) -> None:
    widths = (128, 128, 128)
    _, (input_tt, history_tt, taps_tt) = _device_inputs(device, widths=widths, sequence=32)
    with expect_error(RuntimeError, message):
        _run(
            input_tt,
            history_tt,
            taps_tt,
            widths=widths,
            channel_chunk_size=channel_chunk_size,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("host_input", "allocated device tensor"),
        ("batch", r"input must be \[1,T,Q\+K\+V\]"),
        ("history_shape", r"history must be \[1,3,Q\+K\+V\]"),
        ("tap_last_dimension", r"tap2 last dimension must equal Q\+K\+V"),
        ("tap_volume", "tap2 logical volume must equal"),
        ("input_layout", "input must use ROW_MAJOR layout, got Layout::TILE"),
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
    device: ttnn.Device, expect_error: Callable, case: str, message: str
) -> None:
    batch = 2 if case == "batch" else 1
    history_rows = 2 if case == "history_shape" else 3
    sequence = 33 if case == "sequence_alignment" else 32
    host, device_inputs = _device_inputs(
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
        taps_list[2] = _to_device(taps[2].reshape(-1, 1), device, layout=ttnn.TILE_LAYOUT)
    elif case == "tap_volume":
        taps_list[2] = _to_device(torch.cat((taps[2], taps[2]), dim=0), device, layout=ttnn.TILE_LAYOUT)
    elif case == "input_layout":
        input_tt = _to_device(inputs, device, layout=ttnn.TILE_LAYOUT)
    elif case == "history_layout":
        history_tt = _to_device(history, device, layout=ttnn.TILE_LAYOUT)
    elif case == "tap_layout":
        taps_list[1] = _to_device(taps[1], device, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "input_dtype":
        input_tt = _to_device(inputs.float(), device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "history_dtype":
        history_tt = _to_device(history.float(), device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "tap_dtype":
        taps_list[3] = _to_device(taps[3].float(), device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    elif case == "sharded_history":
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [history.shape[0] * history.shape[1], history.shape[2]],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        history_tt = _to_device(history, device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=sharded_config)

    with expect_error(RuntimeError, message):
        _run(input_tt, history_tt, tuple(taps_list), widths=(128, 128, 128), channel_chunk_size=384)


@pytest.mark.parametrize(
    ("widths", "message"),
    [
        ((0, 128, 256), "Q/K/V widths must be positive"),
        ((100, 128, 156), "Q/K/V widths must be tile aligned"),
        ((128, 128, 64), r"input must be \[1,T,Q\+K\+V\]"),
    ],
)
def test_qkv_causal_conv1d_silu_rejects_invalid_widths(
    device: ttnn.Device, expect_error: Callable, widths: tuple[int, int, int], message: str
) -> None:
    _, (input_tt, history_tt, taps_tt) = _device_inputs(device, widths=(128, 128, 128), sequence=32)
    with expect_error(RuntimeError, message):
        _run(input_tt, history_tt, taps_tt, widths=widths, channel_chunk_size=sum(widths))


def test_qkv_causal_conv1d_silu_rejects_sharded_output(device: ttnn.Device, expect_error: Callable) -> None:
    _, (input_tt, history_tt, taps_tt) = _device_inputs(device, widths=(128, 128, 128), sequence=32)
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
        )
