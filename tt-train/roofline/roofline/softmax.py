# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance model for softmax operations."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..hardware import HardwareSpec, DataType, MathFidelity

if TYPE_CHECKING:
    from .roofline import RooflineEstimate


# TODO: We currently have composite softmax, so this is incorrect
def softmax_roofline(
    hw: HardwareSpec,
    num_rows: int,
    row_size: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "Softmax",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for softmax operation.

    Softmax = exp(x - max(x)) / sum(exp(x - max(x)))

    Forward involves:
    1. Find max across row (reduction)
    2. Subtract max and compute exp (elementwise)
    3. Sum exp values (reduction)
    4. Divide by sum (elementwise)

    Backward involves:
    - grad_input = softmax * (grad_output - sum(grad_output * softmax))

    Args:
        hw: Hardware specification
        num_rows: Number of rows (batch * heads * seq for attention)
        row_size: Size of each row (sequence length for attention)
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
    num_elements = num_rows * row_size

    # TODO: Those numbers seem quite arbitrary.
    if phase == "forward":
        # Forward: read input, write output
        # Multiple passes but can be fused
        total_bytes = int(2 * num_elements * bytes_per_elem)

        # FLOPs:
        # - Max reduction: row_size per row
        # - Subtract + exp: 2 per element (~exp is expensive, count as 5)
        # - Sum reduction: row_size per row
        # - Division: 1 per element
        # Total: ~8 ops per element
        total_flops = 8 * num_elements
    else:
        # Backward:
        # - Read grad_output, softmax_output
        # - Compute grad_output * softmax, sum
        # - Compute grad_input
        total_bytes = int(4 * num_elements * bytes_per_elem)
        total_flops = 6 * num_elements

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
