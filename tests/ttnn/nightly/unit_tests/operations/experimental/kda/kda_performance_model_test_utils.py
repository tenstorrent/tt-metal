# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared hardware conversion for KDA theoretical performance models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import ttnn

MATRIX_FLOPS_PER_CORE_CYCLE = 4096
# Nominal clock used by the repository's Blackhole realtime utilization models.
_BLACKHOLE_CLOCK_GHZ = 1.35
# Blackhole ceiling used by the canonical operation model: ttnn/core/operation.cpp.
DRAM_BYTES_PER_NS = 512


@dataclass(frozen=True, slots=True)
class FpuOps:
    matrix_flops: int = 0
    multiply_ops: int = 0
    add_ops: int = 0
    reduction_ops: int = 0


@dataclass(frozen=True, slots=True)
class SfpuOps:
    exp_ops: int = 0
    rsqrt_ops: int = 0
    sigmoid_ops: int = 0
    silu_ops: int = 0


@dataclass(frozen=True, slots=True)
class KdaWork:
    fpu: FpuOps
    sfpu: SfpuOps
    dram_bytes: int


@dataclass(frozen=True, slots=True)
class KdaPerformance:
    work: KdaWork
    ideal_fpu_ns: float
    ideal_dram_ns: float
    ideal_ns: float
    fpu_utilization_pct: float
    dram_utilization_pct: float
    utilization_pct: float


def _math_fidelity_factor(math_fidelity: ttnn.MathFidelity) -> int:
    import ttnn

    try:
        return {
            ttnn.MathFidelity.LoFi: 1,
            ttnn.MathFidelity.HiFi2: 2,
            ttnn.MathFidelity.HiFi3: 3,
            ttnn.MathFidelity.HiFi4: 4,
        }[math_fidelity]
    except KeyError as error:
        raise ValueError(f"unsupported math fidelity: {math_fidelity}") from error


def _dram_bytes_and_core_count(inputs: Sequence[ttnn.Tensor], outputs: Sequence[ttnn.Tensor]) -> tuple[int, int]:
    import ttnn

    if not inputs:
        raise ValueError("KDA performance model requires at least one input tensor")

    tensors = (*inputs, *outputs)
    if any(tensor.storage_type() != ttnn.StorageType.DEVICE or not tensor.is_allocated() for tensor in tensors):
        raise ValueError("KDA performance model requires allocated device tensors")

    device = inputs[0].device()
    if device is None or device.arch() != ttnn.device.Arch.BLACKHOLE:
        raise ValueError("KDA performance model supports Blackhole device tensors only")
    if any(tensor.device() != device for tensor in tensors):
        raise ValueError("KDA performance model requires all tensors on the same device")

    input_addresses = tuple(int(tensor.buffer_address()) for tensor in inputs)
    if len(input_addresses) != len(set(input_addresses)):
        raise ValueError("KDA performance model does not support aliased inputs")

    dram_bytes = sum(
        int(tensor.volume()) * int(tensor.element_size())
        for tensor in tensors
        if tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    )
    grid = device.compute_with_storage_grid_size()
    core_count = int(grid.x) * int(grid.y)
    if core_count <= 0:
        raise ValueError("KDA performance model requires at least one compute core")
    return dram_bytes, core_count


def performance(
    *,
    fpu: FpuOps,
    sfpu: SfpuOps,
    inputs: Sequence[ttnn.Tensor],
    outputs: Sequence[ttnn.Tensor],
    measured_ns: float,
    math_fidelity: ttnn.MathFidelity,
) -> KdaPerformance:
    if not isinstance(fpu, FpuOps) or not isinstance(sfpu, SfpuOps):
        raise TypeError("fpu and sfpu must be FpuOps and SfpuOps")
    counts = (
        fpu.matrix_flops,
        fpu.multiply_ops,
        fpu.add_ops,
        fpu.reduction_ops,
        sfpu.exp_ops,
        sfpu.rsqrt_ops,
        sfpu.sigmoid_ops,
        sfpu.silu_ops,
    )
    if any(not isinstance(count, int) or count < 0 for count in counts):
        raise ValueError("FPU and SFPU operation counts must be nonnegative integers")
    if not math.isfinite(measured_ns) or measured_ns <= 0:
        raise ValueError("measured_ns must be finite and positive")

    dram_bytes, core_count = _dram_bytes_and_core_count(inputs, outputs)
    fidelity_factor = _math_fidelity_factor(math_fidelity)
    cycle_numerator = (
        fpu.matrix_flops * fidelity_factor
        + 32 * fpu.multiply_ops * fidelity_factor
        + 32 * fpu.add_ops
        + 16 * fpu.reduction_ops * fidelity_factor
    )
    ideal_fpu_ns = cycle_numerator / (MATRIX_FLOPS_PER_CORE_CYCLE * core_count * _BLACKHOLE_CLOCK_GHZ)
    ideal_dram_ns = dram_bytes / DRAM_BYTES_PER_NS
    ideal_ns = max(ideal_fpu_ns, ideal_dram_ns)
    return KdaPerformance(
        work=KdaWork(fpu=fpu, sfpu=sfpu, dram_bytes=dram_bytes),
        ideal_fpu_ns=ideal_fpu_ns,
        ideal_dram_ns=ideal_dram_ns,
        ideal_ns=ideal_ns,
        fpu_utilization_pct=100 * ideal_fpu_ns / measured_ns,
        dram_utilization_pct=100 * ideal_dram_ns / measured_ns,
        utilization_pct=100 * ideal_ns / measured_ns,
    )
