# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA sigmoid-gated RMSNorm."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.reference.kda.ops import sigmoid_gated_rms_norm_reference
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_bit_identical

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True),
]

_BATCH = 1
_SEQUENCE = 64
_NUM_HEADS = 12
_VALUE_DIM = 128
_EPSILON = 1e-5


@dataclass(frozen=True)
class _ProductionCase:
    case_id: str
    num_heads: int
    sequence: int
    expected_duration_ns: int


_PRODUCTION_BATCH = 1
_PRODUCTION_VALUE_DIM = 128
_PRODUCTION_INPUT_DTYPE = ttnn.bfloat16
_PRODUCTION_OUTPUT_DTYPE = ttnn.bfloat16
_PRODUCTION_PERF_MARGIN = 0.05

# Calibrated 2026-08-13 on bh_loudbox host bh-lb-42, P150b firmware 19.5.0.0,
# operation commit 012a3aef206. Seven samples per case produced ranges
# 186342-188082 ns, 186847-187676 ns, and 186821-187901 ns respectively; the
# inline references are their medians. The 5% symmetric margin covers the <1%
# observed spread plus board/thermal variance while still guarding regressions.
_PRODUCTION_CASES = (
    _ProductionCase("sp1-tp8-local", num_heads=12, sequence=5120, expected_duration_ns=187_149),
    _ProductionCase("sp2-tp4-local", num_heads=24, sequence=2560, expected_duration_ns=187_277),
    _ProductionCase("sp4-tp2-local", num_heads=48, sequence=1280, expected_duration_ns=186_913),
)


def _torch_dtype(dtype: ttnn.DataType) -> torch.dtype:
    return torch.float32 if dtype == ttnn.float32 else torch.bfloat16


def _host_inputs(
    *,
    batch: int = _BATCH,
    sequence: int = _SEQUENCE,
    num_heads: int = _NUM_HEADS,
    value_dim: int = _VALUE_DIM,
    input_dtype: ttnn.DataType = ttnn.float32,
    seed: int = 319,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(
        batch * num_heads,
        sequence,
        value_dim,
        generator=generator,
        dtype=_torch_dtype(input_dtype),
    )
    gate = torch.randn(batch, sequence, num_heads * value_dim, generator=generator, dtype=torch.bfloat16)
    weight = torch.randn(value_dim, generator=generator, dtype=torch.bfloat16)
    return inputs, gate, weight


def _to_device(
    tensor: torch.Tensor,
    device: ttnn.Device,
    *,
    dtype: ttnn.DataType,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _device_inputs(
    device: ttnn.Device,
    *,
    batch: int = _BATCH,
    sequence: int = _SEQUENCE,
    num_heads: int = _NUM_HEADS,
    value_dim: int = _VALUE_DIM,
    input_dtype: ttnn.DataType = ttnn.float32,
    seed: int = 319,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]]:
    host = _host_inputs(
        batch=batch,
        sequence=sequence,
        num_heads=num_heads,
        value_dim=value_dim,
        input_dtype=input_dtype,
        seed=seed,
    )
    inputs, gate, weight = host
    device_tensors = (
        _to_device(inputs, device, dtype=input_dtype),
        _to_device(gate, device, dtype=ttnn.bfloat16),
        _to_device(weight, device, dtype=ttnn.bfloat16),
    )
    return host, device_tensors


def _run(
    input_tt: ttnn.Tensor,
    gate_tt: ttnn.Tensor,
    weight_tt: ttnn.Tensor,
    *,
    num_heads: int = _NUM_HEADS,
    epsilon: float = _EPSILON,
    memory_config: ttnn.MemoryConfig | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
    output_dtype: ttnn.DataType = ttnn.float32,
) -> ttnn.Tensor:
    return ttnn.experimental.kda.sigmoid_gated_rms_norm(
        input_tt,
        gate_tt,
        weight_tt,
        num_heads,
        epsilon=epsilon,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
        output_dtype=output_dtype,
    )


def _production_compute_kernel_config(device: ttnn.Device) -> ttnn.DeviceComputeKernelConfig:
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _reference(
    host: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    batch: int,
    sequence: int,
    num_heads: int,
    value_dim: int,
    output_dtype: ttnn.DataType,
) -> torch.Tensor:
    inputs, gate, weight = host
    return (
        sigmoid_gated_rms_norm_reference(
            inputs.reshape(batch, num_heads, sequence, value_dim).permute(0, 2, 1, 3),
            gate.reshape(batch, sequence, num_heads, value_dim),
            weight,
            eps=_EPSILON,
        )
        .reshape(batch, sequence, num_heads * value_dim)
        .to(_torch_dtype(output_dtype))
    )


def _collect_eager_results(
    device: ttnn.Device,
    run: Callable[[], ttnn.Tensor],
    *,
    count: int = 3,
) -> tuple[list[ttnn.Tensor], list[torch.Tensor]]:
    device_outputs = []
    host_outputs = []
    cache_entries = None
    for _ in range(count):
        output_tt = run()
        ttnn.synchronize_device(device)
        output = ttnn.to_torch(output_tt).clone()
        device_outputs.append(output_tt)
        host_outputs.append(output)
        if cache_entries is None:
            cache_entries = device.num_program_cache_entries()
        else:
            assert device.num_program_cache_entries() == cache_entries
    return device_outputs, host_outputs


def _assert_output_contract(
    output_tt: ttnn.Tensor,
    output: torch.Tensor,
    device_inputs: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor],
    *,
    batch: int,
    sequence: int,
    num_heads: int,
    value_dim: int,
    output_dtype: ttnn.DataType,
) -> None:
    assert output_tt.dtype == output_dtype
    assert output_tt.layout == ttnn.TILE_LAYOUT
    assert output_tt.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert tuple(output.shape) == (batch, sequence, num_heads * value_dim)
    assert all(output_tt.buffer_address() != tensor.buffer_address() for tensor in device_inputs)


def _assert_accurate_and_deterministic(expected: torch.Tensor, actual: list[torch.Tensor], *, name: str) -> None:
    for invocation, output in enumerate(actual):
        assert_accurate(expected, output, name=f"{name} invocation {invocation}", pcc_threshold=0.999)
    for invocation, output in enumerate(actual[1:], start=1):
        assert_bit_identical(actual[0], output, name=f"{name} invocation 0 vs {invocation}")


def _assert_inputs_unchanged(
    before: tuple[torch.Tensor, ...],
    device_inputs: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor],
) -> None:
    for name, snapshot, tensor in zip(("input", "gate", "weight"), before, device_inputs, strict=True):
        assert_bit_identical(snapshot, ttnn.to_torch(tensor), name=f"{name} immutability")


@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32-input", "bf16-input"])
@pytest.mark.parametrize("output_dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32-output", "bf16-output"])
def test_sigmoid_gated_rms_norm_is_accurate_and_deterministic(
    device: ttnn.Device, input_dtype: ttnn.DataType, output_dtype: ttnn.DataType
) -> None:
    host, device_inputs = _device_inputs(device, input_dtype=input_dtype)
    input_tt, gate_tt, weight_tt = device_inputs
    expected = _reference(
        host,
        batch=_BATCH,
        sequence=_SEQUENCE,
        num_heads=_NUM_HEADS,
        value_dim=_VALUE_DIM,
        output_dtype=output_dtype,
    )
    input_snapshots = tuple(ttnn.to_torch(tensor).clone() for tensor in device_inputs)

    def run() -> ttnn.Tensor:
        with ttnn.manage_config("throw_exception_on_fallback", True):
            return _run(input_tt, gate_tt, weight_tt, output_dtype=output_dtype)

    device_outputs, host_outputs = _collect_eager_results(device, run)
    _assert_output_contract(
        device_outputs[0],
        host_outputs[0],
        device_inputs,
        batch=_BATCH,
        sequence=_SEQUENCE,
        num_heads=_NUM_HEADS,
        value_dim=_VALUE_DIM,
        output_dtype=output_dtype,
    )
    _assert_accurate_and_deterministic(expected, host_outputs, name=f"{input_dtype} to {output_dtype}")
    _assert_inputs_unchanged(input_snapshots, device_inputs)


@pytest.mark.parametrize("case", _PRODUCTION_CASES, ids=lambda case: case.case_id)
def test_sigmoid_gated_rms_norm_production_is_accurate_and_deterministic(
    device: ttnn.Device, case: _ProductionCase
) -> None:
    host, device_inputs = _device_inputs(
        device,
        batch=_PRODUCTION_BATCH,
        sequence=case.sequence,
        num_heads=case.num_heads,
        value_dim=_PRODUCTION_VALUE_DIM,
        input_dtype=_PRODUCTION_INPUT_DTYPE,
    )
    input_tt, gate_tt, weight_tt = device_inputs
    expected = _reference(
        host,
        batch=_PRODUCTION_BATCH,
        sequence=case.sequence,
        num_heads=case.num_heads,
        value_dim=_PRODUCTION_VALUE_DIM,
        output_dtype=_PRODUCTION_OUTPUT_DTYPE,
    )
    input_snapshots = tuple(ttnn.to_torch(tensor).clone() for tensor in device_inputs)
    compute_kernel_config = _production_compute_kernel_config(device)

    def run() -> ttnn.Tensor:
        with ttnn.manage_config("throw_exception_on_fallback", True):
            return _run(
                input_tt,
                gate_tt,
                weight_tt,
                num_heads=case.num_heads,
                compute_kernel_config=compute_kernel_config,
                output_dtype=_PRODUCTION_OUTPUT_DTYPE,
            )

    device_outputs, host_outputs = _collect_eager_results(device, run)
    _assert_output_contract(
        device_outputs[0],
        host_outputs[0],
        device_inputs,
        batch=_PRODUCTION_BATCH,
        sequence=case.sequence,
        num_heads=case.num_heads,
        value_dim=_PRODUCTION_VALUE_DIM,
        output_dtype=_PRODUCTION_OUTPUT_DTYPE,
    )
    _assert_accurate_and_deterministic(expected, host_outputs, name=case.case_id)
    _assert_inputs_unchanged(input_snapshots, device_inputs)


@pytest.mark.parametrize("case", _PRODUCTION_CASES, ids=lambda case: case.case_id)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_sigmoid_gated_rms_norm_production_performance(device: ttnn.Device, case: _ProductionCase) -> None:
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for sigmoid-gated RMSNorm performance checks")

    _, (input_tt, gate_tt, weight_tt) = _device_inputs(
        device,
        batch=_PRODUCTION_BATCH,
        sequence=case.sequence,
        num_heads=case.num_heads,
        value_dim=_PRODUCTION_VALUE_DIM,
        input_dtype=_PRODUCTION_INPUT_DTYPE,
    )
    compute_kernel_config = _production_compute_kernel_config(device)

    def run() -> ttnn.Tensor:
        return _run(
            input_tt,
            gate_tt,
            weight_tt,
            num_heads=case.num_heads,
            compute_kernel_config=compute_kernel_config,
            output_dtype=_PRODUCTION_OUTPUT_DTYPE,
        )

    output_tt, perf_record = profile_realtime_program(device, run)
    duration_ns = perf_record["duration_ns"]
    lower = case.expected_duration_ns * (1 - _PRODUCTION_PERF_MARGIN)
    upper = case.expected_duration_ns * (1 + _PRODUCTION_PERF_MARGIN)
    assert output_tt.dtype == _PRODUCTION_OUTPUT_DTYPE
    assert tuple(output_tt.shape) == (
        _PRODUCTION_BATCH,
        case.sequence,
        case.num_heads * _PRODUCTION_VALUE_DIM,
    )
    logger.info(
        f"sigmoid-gated RMSNorm {case.case_id}: duration={duration_ns:.0f} ns, "
        f"reference={case.expected_duration_ns} ns, band=[{lower:.0f}, {upper:.0f}] ns, "
        f"profiler_runtime_id={perf_record['runtime_id']}"
    )
    assert lower <= duration_ns <= upper, (
        f"{case.case_id} duration {duration_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}] ns "
        f"(reference {case.expected_duration_ns} ns, margin +/- {_PRODUCTION_PERF_MARGIN * 100:.0f}%)"
    )


def test_sigmoid_gated_rms_norm_program_key_includes_epsilon(device: ttnn.Device) -> None:
    _, (input_tt, gate_tt, weight_tt) = _device_inputs(
        device, batch=1, sequence=32, num_heads=2, value_dim=64, seed=1321
    )
    _run(input_tt, gate_tt, weight_tt, num_heads=2, epsilon=1e-5)
    entries = device.num_program_cache_entries()
    _run(input_tt, gate_tt, weight_tt, num_heads=2, epsilon=2e-5)
    assert device.num_program_cache_entries() == entries + 1
    _run(input_tt, gate_tt, weight_tt, num_heads=2, epsilon=2e-5)
    assert device.num_program_cache_entries() == entries + 1


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("host_input", "allocated device tensor"),
        ("input_dtype", "input has unsupported dtype"),
        ("gate_dtype", "gate has unsupported dtype"),
        ("weight_dtype", "weight has unsupported dtype"),
        ("input_layout", "input must use TILE layout"),
        ("gate_shape", "gate must have shape"),
        ("weight_shape", r"weight must be \[V\]"),
        ("sequence_alignment", "sequence must be positive and tile aligned"),
        ("value_alignment", "value_dim must be positive and tile aligned"),
        ("sharded_input", "input must use interleaved memory"),
    ],
)
def test_sigmoid_gated_rms_norm_rejects_invalid_tensors(
    device: ttnn.Device, expect_error: Callable, case: str, message: str
) -> None:
    input_dtype = ttnn.float32
    sequence = 33 if case == "sequence_alignment" else 32
    value_dim = 100 if case == "value_alignment" else 128
    host = _host_inputs(sequence=sequence, value_dim=value_dim, input_dtype=input_dtype, seed=9321)
    inputs, gate, weight = host
    input_tt = _to_device(inputs, device, dtype=input_dtype)
    gate_tt = _to_device(gate, device, dtype=ttnn.bfloat16)
    weight_tt = _to_device(weight, device, dtype=ttnn.bfloat16)

    if case == "host_input":
        input_tt = ttnn.from_torch(inputs, dtype=input_dtype, layout=ttnn.TILE_LAYOUT)
    elif case == "input_dtype":
        input_tt = _to_device(inputs, device, dtype=ttnn.bfloat8_b)
    elif case == "gate_dtype":
        gate_tt = _to_device(gate.float(), device, dtype=ttnn.float32)
    elif case == "weight_dtype":
        weight_tt = _to_device(weight.float(), device, dtype=ttnn.float32)
    elif case == "input_layout":
        input_tt = _to_device(inputs, device, dtype=input_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "gate_shape":
        bad_gate = gate[..., :-32]
        gate_tt = _to_device(bad_gate, device, dtype=ttnn.bfloat16)
    elif case == "weight_shape":
        weight_tt = _to_device(weight.reshape(2, -1), device, dtype=ttnn.bfloat16)
    elif case == "sharded_input":
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [inputs.shape[0] * inputs.shape[1], inputs.shape[2]],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        input_tt = _to_device(inputs, device, dtype=input_dtype, memory_config=sharded_config)

    with expect_error(RuntimeError, message):
        _run(input_tt, gate_tt, weight_tt)


@pytest.mark.parametrize(
    ("num_heads", "epsilon", "output_dtype", "message"),
    [
        (0, 1e-5, ttnn.float32, "num_heads must be positive"),
        (5, 1e-5, ttnn.float32, "leading dimension must be divisible"),
        (12, 0.0, ttnn.float32, "epsilon must be finite and positive"),
        (12, float("nan"), ttnn.float32, "epsilon must be finite and positive"),
        (12, 1e-5, ttnn.uint32, "output_dtype must be FLOAT32 or BFLOAT16"),
    ],
)
def test_sigmoid_gated_rms_norm_rejects_invalid_options(
    device: ttnn.Device,
    expect_error: Callable,
    num_heads: int,
    epsilon: float,
    output_dtype: ttnn.DataType,
    message: str,
) -> None:
    _, (input_tt, gate_tt, weight_tt) = _device_inputs(device)
    with expect_error(RuntimeError, message):
        _run(input_tt, gate_tt, weight_tt, num_heads=num_heads, epsilon=epsilon, output_dtype=output_dtype)


def test_sigmoid_gated_rms_norm_rejects_sharded_output(device: ttnn.Device, expect_error: Callable) -> None:
    _, (input_tt, gate_tt, weight_tt) = _device_inputs(device)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [_SEQUENCE, _NUM_HEADS * _VALUE_DIM],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    with expect_error(RuntimeError, "output memory configuration must be interleaved"):
        _run(input_tt, gate_tt, weight_tt, memory_config=sharded_config)
