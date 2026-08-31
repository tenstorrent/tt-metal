# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from itertools import count
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch

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
    ) -> None:
        self.shape = shape
        self._element_size = element_size
        self._buffer_type = buffer_type
        self._address = next(_addresses) if address is None else address

    def storage_type(self) -> ttnn.StorageType:
        return ttnn.StorageType.DEVICE

    def is_allocated(self) -> bool:
        return True

    def volume(self) -> int:
        return math.prod(self.shape)

    def element_size(self) -> int:
        return self._element_size

    def buffer_address(self) -> int:
        return self._address

    def memory_config(self) -> Any:
        return SimpleNamespace(buffer_type=self._buffer_type)


def _measurement() -> dict[str, Any]:
    return {
        "measured_ns": 1.0,
        "core_count": 1,
        "frequency_ghz": 1.0,
        "math_fidelity": ttnn.MathFidelity.LoFi,
    }


def _protocol(key_dim: int, value_dim: int) -> tuple[_FakeTensor, ...]:
    return (
        _FakeTensor((1, 1, 32, value_dim)),
        _FakeTensor((1, 1, 32, key_dim)),
        _FakeTensor((1, 1, 32, key_dim)),
        _FakeTensor((1, 1, 32, 32)),
        _FakeTensor((1, 1, key_dim, 32)),
        _FakeTensor((1, 1, key_dim, 1)),
        _FakeTensor((1, 1, 32, 32)),
    )


def test_tensor_volume_includes_tile_padding() -> None:
    logical_shape = (3, 17, 17)
    tensor = ttnn.Tensor(torch.zeros(logical_shape), ttnn.bfloat16).pad_to_tile(0.0)

    assert tensor.volume() == 3 * 32 * 32
    assert tensor.volume() != math.prod(logical_shape)


def test_sigmoid_gated_rms_norm_work_golden() -> None:
    work = perf_model.sigmoid_gated_rms_norm_performance(
        _FakeTensor((2, 1, 2)),
        _FakeTensor((1, 1, 4)),
        _FakeTensor((2,)),
        _FakeTensor((1, 1, 4)),
        **_measurement(),
    ).work

    assert work == perf_model.KdaWork(
        fpu_multiply_ops=16,
        fpu_add_ops=2,
        fpu_reduction_ops=2,
        sfpu_rsqrt_ops=2,
        sfpu_sigmoid_ops=4,
    )


def test_qkv_causal_conv1d_silu_work_golden() -> None:
    work = perf_model.qkv_causal_conv1d_silu_performance(
        _FakeTensor((1, 2, 4)),
        _FakeTensor((1, 3, 4)),
        tuple(_FakeTensor((1, 1, 4)) for _ in range(4)),
        (_FakeTensor((1, 2, 1)), _FakeTensor((1, 2, 2)), _FakeTensor((1, 2, 1))),
        **_measurement(),
    ).work

    assert work == perf_model.KdaWork(fpu_multiply_ops=32, fpu_add_ops=24, sfpu_silu_ops=8)


def test_reduce_affine_transforms_work_golden() -> None:
    work = perf_model.reduce_affine_transforms_performance(
        _FakeTensor((2, 2, 2)),
        _FakeTensor((2, 2, 1)),
        (_FakeTensor((1, 2, 2)), _FakeTensor((1, 2, 1))),
        **_measurement(),
    ).work

    assert work == perf_model.KdaWork(fpu_matrix_flops=24, fpu_add_ops=2)


def test_affine_exclusive_scan_performance_golden() -> None:
    output_address = 0xCAFE
    result = perf_model.affine_exclusive_scan_performance(
        _FakeTensor((2, 2, 2), buffer_type=ttnn.BufferType.DRAM),
        _FakeTensor((2, 2, 1), buffer_type=ttnn.BufferType.DRAM, address=output_address),
        _FakeTensor((1, 2, 1), buffer_type=ttnn.BufferType.DRAM),
        _FakeTensor((2, 2, 1), buffer_type=ttnn.BufferType.DRAM, address=output_address),
        **_measurement(),
    )

    assert result.work == perf_model.KdaWork(fpu_matrix_flops=8, fpu_add_ops=2, dram_bytes=36)
    assert result.ideal_fpu_ns == 0.017578125
    assert result.ideal_dram_ns == 0.0703125
    assert result.ideal_ns == 0.0703125
    assert result.fpu_utilization_pct == 1.7578125
    assert result.dram_utilization_pct == 7.03125
    assert result.utilization_pct == 7.03125


def test_prepare_chunk_recurrence_work_golden() -> None:
    inputs = (
        _FakeTensor((1, 32, 2)),
        _FakeTensor((1, 32, 2)),
        _FakeTensor((1, 32, 1)),
        _FakeTensor((1, 32, 2)),
        _FakeTensor((1, 1, 32, 1)),
    )
    outputs = (
        _FakeTensor((1, 1, 32, 1)),
        _FakeTensor((1, 1, 32, 2)),
        _FakeTensor((1, 1, 32, 2)),
        _FakeTensor((1, 1, 32, 32)),
        _FakeTensor((1, 1, 2, 32)),
        _FakeTensor((1, 1, 2, 1)),
        _FakeTensor((1, 1, 32, 32)),
    )

    work = perf_model.prepare_chunk_recurrence_performance(inputs, outputs, **_measurement()).work

    assert work == perf_model.KdaWork(
        fpu_matrix_flops=19104,
        fpu_multiply_ops=672,
        fpu_add_ops=1214,
        fpu_reduction_ops=64,
        sfpu_exp_ops=194,
        sfpu_rsqrt_ops=64,
    )


def test_recurrent_chunk_scan_work_golden() -> None:
    work = perf_model.recurrent_chunk_scan_performance(
        _protocol(2, 1),
        _FakeTensor((1, 2, 1)),
        (_FakeTensor((1, 1, 32, 1)), _FakeTensor((1, 2, 1))),
        **_measurement(),
    ).work

    assert work == perf_model.KdaWork(fpu_matrix_flops=4480, fpu_multiply_ops=2, fpu_add_ops=66)


def test_summarize_chunk_recurrence_work_golden() -> None:
    work = perf_model.summarize_chunk_recurrence_performance(
        _protocol(2, 2),
        (_FakeTensor((1, 2, 2)), _FakeTensor((1, 2, 2))),
        **_measurement(),
    ).work

    assert work == perf_model.KdaWork(fpu_matrix_flops=9216, fpu_multiply_ops=8, fpu_add_ops=140)


def test_invalid_public_inputs_raise(expect_error: Callable) -> None:
    with expect_error(ValueError, "aliased inputs"):
        perf_model.affine_exclusive_scan_performance(
            _FakeTensor((2, 2, 2), address=0xBAD),
            _FakeTensor((2, 2, 1), address=0xBAD),
            _FakeTensor((1, 2, 1)),
            _FakeTensor((2, 2, 1)),
            **_measurement(),
        )

    with expect_error(ValueError, "shapes must be positive"):
        perf_model.affine_exclusive_scan_performance(
            _FakeTensor((2, 0, 2)),
            _FakeTensor((2, 0, 1)),
            _FakeTensor((1, 0, 1)),
            _FakeTensor((2, 0, 1)),
            **_measurement(),
        )

    with expect_error(ValueError, "shapes are inconsistent"):
        perf_model.affine_exclusive_scan_performance(
            _FakeTensor((3, 2, 2)),
            _FakeTensor((3, 2, 1)),
            _FakeTensor((2, 2, 1)),
            _FakeTensor((3, 2, 1)),
            **_measurement(),
        )

    with expect_error(ValueError, "measured_ns"):
        perf_model.affine_exclusive_scan_performance(
            _FakeTensor((2, 2, 2)),
            _FakeTensor((2, 2, 1)),
            _FakeTensor((1, 2, 1)),
            _FakeTensor((2, 2, 1)),
            **(_measurement() | {"measured_ns": 0.0}),
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
