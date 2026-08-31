# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from itertools import count
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import pytest

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import kda_performance_model_test_utils as perf_model
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.kda_realtime_profiler_test_utils import (
    profile_realtime_program,
)

_addresses = count(0x1000, 0x1000)


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        element_size: int = 2,
        buffer_type: ttnn.BufferType = ttnn.BufferType.L1,
        address: int | None = None,
        allocated: bool = True,
        storage_type: ttnn.StorageType = ttnn.StorageType.DEVICE,
    ) -> None:
        self.shape = shape
        self._element_size = element_size
        self._buffer_type = buffer_type
        self._address = next(_addresses) if address is None else address
        self._allocated = allocated
        self._storage_type = storage_type

    def storage_type(self) -> ttnn.StorageType:
        return self._storage_type

    def is_allocated(self) -> bool:
        return self._allocated

    def volume(self) -> int:
        return math.prod(self.shape)

    def element_size(self) -> int:
        return self._element_size

    def buffer_address(self) -> int:
        return self._address

    def memory_config(self) -> Any:
        return SimpleNamespace(buffer_type=self._buffer_type)


def _hardware() -> dict[str, Any]:
    return {
        "measured_ns": 1000.0,
        "core_count": 130,
        "frequency_ghz": 1.35,
        "math_fidelity": ttnn.MathFidelity.HiFi2,
    }


def _protocol(batch_heads: int, num_chunks: int, key_dim: int, value_dim: int) -> tuple[_FakeTensor, ...]:
    base = (batch_heads, num_chunks)
    return (
        _FakeTensor((*base, 32, value_dim)),
        _FakeTensor((*base, 32, key_dim)),
        _FakeTensor((*base, 32, key_dim)),
        _FakeTensor((*base, 32, 32)),
        _FakeTensor((*base, key_dim, 32)),
        _FakeTensor((*base, key_dim, 1)),
        _FakeTensor((*base, 32, 32)),
    )


def test_affine_exclusive_scan_returns_complete_performance() -> None:
    batch_heads, groups, key_dim, value_dim = 3, 4, 5, 7
    leading = batch_heads * groups
    a = _FakeTensor((leading, key_dim, key_dim), buffer_type=ttnn.BufferType.DRAM)
    b = _FakeTensor((leading, key_dim, value_dim), buffer_type=ttnn.BufferType.DRAM)
    state = _FakeTensor((batch_heads, key_dim, value_dim), element_size=4, buffer_type=ttnn.BufferType.DRAM)
    output = _FakeTensor((leading, key_dim, value_dim), element_size=4, buffer_type=ttnn.BufferType.DRAM)

    result = perf_model.affine_exclusive_scan_performance(a, b, state, output, **_hardware())

    transitions = batch_heads * (groups - 1)
    assert result.work == perf_model.KdaWork(
        fpu_matrix_flops=transitions * 2 * key_dim**2 * value_dim,
        fpu_add_ops=transitions * key_dim * value_dim,
        dram_bytes=sum(tensor.volume() * tensor.element_size() for tensor in (a, b, state, output)),
    )
    assert result.ideal_dram_ns == pytest.approx(result.work.dram_bytes / 512)
    assert result.ideal_ns == max(result.ideal_fpu_ns, result.ideal_dram_ns)
    assert result.fpu_utilization_pct == pytest.approx(100 * result.ideal_fpu_ns / 1000)
    assert result.dram_utilization_pct == pytest.approx(100 * result.ideal_dram_ns / 1000)
    assert result.utilization_pct == pytest.approx(100 * result.ideal_ns / 1000)


def test_all_operation_specific_functions_derive_mathematical_work() -> None:
    rms = perf_model.sigmoid_gated_rms_norm_performance(
        _FakeTensor((6, 5, 7)),
        _FakeTensor((2, 5, 21)),
        _FakeTensor((7,)),
        _FakeTensor((2, 5, 21)),
        **_hardware(),
    ).work
    assert rms == perf_model.KdaWork(
        fpu_multiply_ops=4 * 2 * 3 * 5 * 7,
        fpu_add_ops=2 * 3 * 5,
        fpu_reduction_ops=2 * 3 * 5 * (7 - 1),
        sfpu_rsqrt_ops=2 * 3 * 5,
        sfpu_sigmoid_ops=2 * 3 * 5 * 7,
    )

    qkv = perf_model.qkv_causal_conv1d_silu_performance(
        _FakeTensor((2, 5, 31)),
        _FakeTensor((2, 3, 31)),
        tuple(_FakeTensor((1, 1, 31)) for _ in range(4)),
        (_FakeTensor((2, 5, 7)), _FakeTensor((2, 5, 11)), _FakeTensor((2, 5, 13))),
        **_hardware(),
    ).work
    elements = 2 * 5 * 31
    assert qkv == perf_model.KdaWork(
        fpu_multiply_ops=4 * elements,
        fpu_add_ops=3 * elements,
        sfpu_silu_ops=elements,
    )

    a = _FakeTensor((12, 5, 5))
    b = _FakeTensor((12, 5, 7))
    reduced = perf_model.reduce_affine_transforms_performance(
        a, b, (_FakeTensor((3, 5, 5)), _FakeTensor((3, 5, 7))), **_hardware()
    ).work
    compositions = 3 * (4 - 1)
    assert reduced == perf_model.KdaWork(
        fpu_matrix_flops=compositions * (2 * 5**3 + 2 * 5**2 * 7),
        fpu_add_ops=compositions * 5 * 7,
    )

    heads, chunks, key_dim, value_dim = 2, 3, 5, 7
    prepare_inputs = (
        _FakeTensor((1, chunks * 32, heads * key_dim)),
        _FakeTensor((1, chunks * 32, heads * key_dim)),
        _FakeTensor((1, chunks * 32, heads * value_dim)),
        _FakeTensor((1, chunks * 32, heads * key_dim)),
        _FakeTensor((heads, chunks, 32, 1)),
    )
    prepare_outputs = (
        _FakeTensor((heads, chunks, 32, value_dim)),
        _FakeTensor((heads, chunks, 32, key_dim)),
        _FakeTensor((heads, chunks, 32, key_dim)),
        _FakeTensor((heads, chunks, 32, 32)),
        _FakeTensor((heads, chunks, key_dim, 32)),
        _FakeTensor((heads, chunks, key_dim, 1)),
        _FakeTensor((heads, chunks, 32, 32)),
    )
    prepared = perf_model.prepare_chunk_recurrence_performance(prepare_inputs, prepare_outputs, **_hardware()).work
    instances = heads * chunks
    assert prepared.fpu_matrix_flops == instances * (4 * 32**2 * key_dim + 32 * 31 * 33 // 3)
    assert prepared.fpu_reduction_ops == instances * 2 * 32 * (key_dim - 1)
    assert prepared.sfpu_exp_ops == instances * (3 * 32 * key_dim + key_dim)
    assert prepared.sfpu_rsqrt_ops == instances * 2 * 32

    protocol = _protocol(heads, chunks, key_dim, value_dim)
    recurrent = perf_model.recurrent_chunk_scan_performance(
        protocol,
        _FakeTensor((heads, key_dim, value_dim)),
        (_FakeTensor((heads, chunks, 32, value_dim)), _FakeTensor((heads, key_dim, value_dim))),
        **_hardware(),
    ).work
    assert recurrent == perf_model.KdaWork(
        fpu_matrix_flops=instances * (6 * 32 * key_dim * value_dim + 4 * 32**2 * value_dim),
        fpu_multiply_ops=instances * key_dim * value_dim,
        fpu_add_ops=instances * (2 * 32 * value_dim + key_dim * value_dim),
    )

    summary_protocol = _protocol(heads, chunks, key_dim, key_dim)
    summarized = perf_model.summarize_chunk_recurrence_performance(
        summary_protocol,
        (_FakeTensor((heads, key_dim, key_dim)), _FakeTensor((heads, key_dim, key_dim))),
        **_hardware(),
    ).work
    assert summarized == perf_model.KdaWork(
        fpu_matrix_flops=instances * (8 * 32 * key_dim**2 + 4 * 32**2 * key_dim),
        fpu_multiply_ops=instances * 2 * key_dim**2,
        fpu_add_ops=instances * (2 * 32 * key_dim + 2 * key_dim**2) + heads * key_dim**2,
    )


def test_dram_traffic_counts_each_read_and_write() -> None:
    address = 0xCAFE
    a = _FakeTensor((2, 4, 4), buffer_type=ttnn.BufferType.DRAM, address=address)
    b = _FakeTensor((2, 4, 4), buffer_type=ttnn.BufferType.DRAM)
    state = _FakeTensor((1, 4, 4), buffer_type=ttnn.BufferType.L1)
    output = _FakeTensor((2, 4, 4), buffer_type=ttnn.BufferType.DRAM, address=address)
    work = perf_model.affine_exclusive_scan_performance(a, b, state, output, **_hardware()).work
    assert work.dram_bytes == sum(tensor.volume() * tensor.element_size() for tensor in (a, b, output))


def test_input_aliases_raise(expect_error: Callable) -> None:
    address = 0xCAFE
    with expect_error(ValueError, "aliased inputs"):
        perf_model.affine_exclusive_scan_performance(
            _FakeTensor((2, 4, 4), address=address),
            _FakeTensor((2, 4, 4), address=address),
            _FakeTensor((1, 4, 4)),
            _FakeTensor((2, 4, 4)),
            **_hardware(),
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"measured_ns": 0.0}, "measured_ns"),
        ({"frequency_ghz": float("nan")}, "frequency_ghz"),
        ({"core_count": 0}, "core_count"),
        ({"math_fidelity": ttnn.MathFidelity.Invalid}, "math fidelity"),
    ],
)
def test_invalid_hardware_or_measurement_raises(override: dict[str, Any], message: str, expect_error: Callable) -> None:
    arguments = _hardware() | override
    with expect_error(ValueError, message):
        perf_model.affine_exclusive_scan_performance(
            _FakeTensor((2, 4, 4)),
            _FakeTensor((2, 4, 4)),
            _FakeTensor((1, 4, 4)),
            _FakeTensor((2, 4, 4)),
            **arguments,
        )


def test_invalid_tensor_or_shape_raises(expect_error: Callable) -> None:
    with expect_error(ValueError, "allocated device tensors"):
        perf_model.affine_exclusive_scan_performance(
            _FakeTensor((2, 4, 4), allocated=False),
            _FakeTensor((2, 4, 4)),
            _FakeTensor((1, 4, 4)),
            _FakeTensor((2, 4, 4)),
            **_hardware(),
        )
    with expect_error(ValueError, "shapes are inconsistent"):
        perf_model.affine_exclusive_scan_performance(
            _FakeTensor((3, 4, 4)),
            _FakeTensor((3, 4, 4)),
            _FakeTensor((2, 4, 4)),
            _FakeTensor((3, 4, 4)),
            **_hardware(),
        )
    with expect_error(ValueError, "shapes must be positive"):
        perf_model.affine_exclusive_scan_performance(
            _FakeTensor((2, 0, 4)),
            _FakeTensor((2, 0, 4)),
            _FakeTensor((1, 0, 4)),
            _FakeTensor((2, 0, 4)),
            **_hardware(),
        )


def test_realtime_profile_record_exposes_frequency_ghz(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = None

    def register(collector: Callable) -> int:
        nonlocal callback
        callback = collector
        return 7

    monkeypatch.setattr(ttnn.device, "RegisterProgramRealtimeProfilerCallback", register)
    monkeypatch.setattr(ttnn.device, "UnregisterProgramRealtimeProfilerCallback", lambda _handle: None)
    monkeypatch.setattr(ttnn, "synchronize_device", lambda _device: None)

    record = SimpleNamespace(
        runtime_id=19,
        chip_id=0,
        start_timestamp=100,
        end_timestamp=1450,
        frequency=1.35,
        kernel_sources=("reader.cpp",),
    )

    def run() -> str:
        assert callback is not None
        callback(SimpleNamespace(dropped=0, records=(record,)))
        return "result"

    result, profile = profile_realtime_program(object(), run)
    assert result == "result"
    assert profile["frequency_ghz"] == pytest.approx(1.35)
    assert profile["duration_ns"] == pytest.approx(1000.0)
