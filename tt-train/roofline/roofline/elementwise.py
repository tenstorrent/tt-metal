# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance model for elementwise operations."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..hardware import HardwareSpec, DataType, MathFidelity
from .roofline import (
    sfpu_flops_per_core_per_cycle,
    fpu_eltwise_flops_per_core_per_cycle,
)

if TYPE_CHECKING:
    from .roofline import RooflineEstimate


def elementwise_roofline(
    hw: HardwareSpec,
    num_elements: int,
    num_inputs: int = 1,
    sfpu_ops_per_element: float = 1.0,
    fpu_ops_per_element: float = 0.0,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "Elementwise",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for elementwise operations (add, mul, gelu, etc).

    These operations are typically memory-bound as they have low
    arithmetic intensity (few FLOPs per byte transferred).

    Args:
        hw: Hardware specification
        num_elements: Total number of elements to process
        num_inputs: Number of input tensors (1 for unary, 2 for binary)
        sfpu_ops_per_element: SFPU operations per element (1 for add/mul, ~10 for gelu)
        fpu_ops_per_element: FPU operations per element
        dtype: Data type for operands
        fidelity: Math fidelity level
        num_cores: Number of cores to use (default: all)
        operation: Name for the operation
        phase: "forward" or "backward"

    Returns:
        RooflineEstimate with performance metrics
    """
    # Local import to avoid circular dependency
    from .roofline import RooflineEstimate

    if num_cores is None:
        num_cores = hw.tensix_cores_per_chip

    bytes_per_elem = dtype.value

    # Memory: read inputs + write output
    total_bytes = int((num_inputs + 1) * num_elements * bytes_per_elem)

    # FLOPs
    total_flops = int(num_elements)

    # TODO: verify idea with mixing sfpu and fpu
    ops_per_cycle = 0.0
    if sfpu_ops_per_element > 0.0:
        ops_per_cycle += sfpu_flops_per_core_per_cycle(fidelity) / sfpu_ops_per_element
    if fpu_ops_per_element > 0.0:
        ops_per_cycle += (
            fpu_eltwise_flops_per_core_per_cycle(fidelity) / fpu_ops_per_element
        )

    # No compute case (Avoid division by zero)
    if sfpu_ops_per_element <= 0.0 and fpu_ops_per_element <= 0.0:
        ideal_compute_ns = 0.0
    else:
        ideal_compute_ns = total_flops / (num_cores * ops_per_cycle) / hw.clock_ghz

    # Memory time
    ideal_memory_ns = total_bytes / hw.dram_bw_gb_s

    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=total_flops,
        total_bytes=total_bytes,
        ideal_compute_ns=ideal_compute_ns,
        ideal_memory_ns=ideal_memory_ns,
        hw=hw,
    )
