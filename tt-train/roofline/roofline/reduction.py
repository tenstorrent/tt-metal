# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance model for reduction operations."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..hardware import HardwareSpec, DataType, MathFidelity

if TYPE_CHECKING:
    from .roofline import RooflineEstimate


def reduction_roofline(
    hw: HardwareSpec,
    input_elements: int,
    reduction_dim_size: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "Reduction",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for reduction operations (sum, mean, max).

    Matrix engine does 16x16 reduce in a single cycle = 512 ops/cycle at LoFi.

    Args:
        hw: Hardware specification
        input_elements: Total number of input elements
        reduction_dim_size: Size of the dimension being reduced
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

    output_elements = input_elements // reduction_dim_size
    bytes_per_elem = dtype.value

    total_bytes = int((input_elements + output_elements) * bytes_per_elem)
    total_flops = input_elements  # ~1 op per input element

    ops_per_cycle = sfpu_flops_per_core_per_cycle(fidelity)
    ideal_compute_ns = total_flops / (num_cores * ops_per_cycle) / hw.clock_ghz

    # Ideal memory time
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
