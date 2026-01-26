# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance model for dropout operations."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..hardware import HardwareSpec, DataType, MathFidelity

if TYPE_CHECKING:
    from .roofline import RooflineEstimate


def dropout_roofline(
    hw: HardwareSpec,
    num_elements: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "Dropout",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for dropout operation.

    Forward:
        - Generate random mask
        - Multiply input by mask
        - Scale by 1/(1-p) for inverted dropout

    Backward:
        - Reuse mask from forward
        - Multiply grad_output by mask
        - Scale by 1/(1-p)

    Args:
        hw: Hardware specification
        num_elements: Total number of elements
        dtype: Data type for operands
        fidelity: Math fidelity level
        num_cores: Number of cores to use (default: all)
        operation: Name for the operation
        phase: "forward" or "backward"

    Returns:
        RooflineEstimate with performance metrics
    """
    # Local import to avoid circular dependency
    from .roofline import RooflineEstimate, sfpu_flops_per_core_per_cycle

    if num_cores is None:
        num_cores = hw.tensix_cores_per_chip

    bytes_per_elem = dtype.value

    if phase == "forward":
        # Read input, generate mask (internal), write output
        # Mask generation is compute-bound but typically fast
        # For simplicity, model as read input + write output
        total_bytes = int(2 * num_elements * bytes_per_elem)
        total_flops = 3 * num_elements  # mask comparison + multiply + scale
    else:
        # Read grad_output and mask, write grad_input
        # Mask could be stored as 1-bit but typically stored as dtype
        total_bytes = int(3 * num_elements * bytes_per_elem)
        total_flops = 2 * num_elements  # multiply + scale

    # Compute time
    ops_per_cycle = sfpu_flops_per_core_per_cycle(fidelity)
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
