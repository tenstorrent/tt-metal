# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from itertools import count
from types import SimpleNamespace
from typing import Any, Callable

import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import kda_performance_model_test_utils as perf_model

_addresses = count(0x1000, 0x1000)


class _FakeDevice:
    def arch(self) -> ttnn.device.Arch:
        return ttnn.device.Arch.BLACKHOLE

    def compute_with_storage_grid_size(self) -> Any:
        return SimpleNamespace(x=1, y=1)


_DEVICE = _FakeDevice()


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        element_size: int = 2,
        buffer_type: ttnn.BufferType = ttnn.BufferType.L1,
        address: int | None = None,
        device: _FakeDevice = _DEVICE,
    ) -> None:
        self.shape = shape
        self._element_size = element_size
        self._buffer_type = buffer_type
        self._address = next(_addresses) if address is None else address
        self._device = device

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

    def device(self) -> _FakeDevice:
        return self._device


def _measurement() -> dict[str, Any]:
    return {"measured_ns": 1.0, "math_fidelity": ttnn.MathFidelity.LoFi}


def test_tensor_volume_includes_tile_padding() -> None:
    logical_shape = (3, 17, 17)
    tensor = ttnn.Tensor(torch.zeros(logical_shape), ttnn.bfloat16).pad_to_tile(0.0)

    assert tensor.volume() == 3 * 32 * 32
    assert tensor.volume() != math.prod(logical_shape)


def test_shared_roofline_golden() -> None:
    aliased_input_address = 0xCAFE
    aliased_input_a = _FakeTensor((2, 2, 2), buffer_type=ttnn.BufferType.DRAM, address=aliased_input_address)
    aliased_input_b = _FakeTensor((2, 2, 1), buffer_type=ttnn.BufferType.DRAM, address=aliased_input_address)
    assert aliased_input_a.buffer_address() == aliased_input_b.buffer_address()

    fpu = perf_model.FpuOps(matrix_flops=8, add_ops=2)
    sfpu = perf_model.SfpuOps(exp_ops=3)
    result = perf_model.performance(
        fpu=fpu,
        sfpu=sfpu,
        inputs=(
            aliased_input_a,
            aliased_input_b,
            _FakeTensor((1, 2, 1), buffer_type=ttnn.BufferType.DRAM),
        ),
        outputs=(_FakeTensor((2, 2, 1), buffer_type=ttnn.BufferType.DRAM),),
        **_measurement(),
    )

    assert result.work == perf_model.KdaWork(fpu=fpu, sfpu=sfpu, dram_bytes=36)
    assert result.ideal_fpu_ns == 0.013020833333333332
    assert result.ideal_dram_ns == 0.0703125
    assert result.ideal_ns == 0.0703125
    assert result.fpu_utilization_pct == 1.3020833333333333
    assert result.dram_utilization_pct == 7.03125
    assert result.utilization_pct == 7.03125


def test_invalid_public_inputs_raise(expect_error: Callable) -> None:
    with expect_error(ValueError, "operation counts"):
        perf_model.performance(
            fpu=perf_model.FpuOps(add_ops=-1),
            sfpu=perf_model.SfpuOps(),
            inputs=(_FakeTensor((1,)),),
            outputs=(_FakeTensor((1,)),),
            **_measurement(),
        )

    with expect_error(ValueError, "measured_ns"):
        perf_model.performance(
            fpu=perf_model.FpuOps(),
            sfpu=perf_model.SfpuOps(),
            inputs=(_FakeTensor((1,)),),
            outputs=(_FakeTensor((1,)),),
            **(_measurement() | {"measured_ns": 0.0}),
        )
