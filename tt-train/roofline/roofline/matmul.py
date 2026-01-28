# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance model for matrix multiplication."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import math

from ..hardware import HardwareSpec, DataType, MathFidelity

if TYPE_CHECKING:
    from .roofline import RooflineEstimate, fpu_mm_flops_per_core_per_cycle


def matmul_roofline(
    hw: HardwareSpec,
    M: int,
    K: int,
    N: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "MatMul",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for matrix multiplication.

    Computes A[M, K] @ B[K, N] -> C[M, N]

    Based on tt-metal's create_op_performance_model_for_matmul:
    - 4096 multiply-adds per cycle per core
    - Scaled by math fidelity multiplier

    Args:
        hw: Hardware specification
        M: Number of rows in A
        K: Inner dimension (columns of A, rows of B)
        N: Number of columns in B
        dtype: Data type for operands
        fidelity: Math fidelity level
        num_cores: Number of cores to use (default: all)
        operation: Name for the operation
        phase: "forward" or "backward"

    Returns:
        RooflineEstimate with performance metrics
    """
    # Local import to avoid circular dependency
    from .roofline import RooflineEstimate, fpu_mm_flops_per_core_per_cycle

    if num_cores is None:
        num_cores = hw.tensix_cores_per_chip

    # FLOPs calculation: 2 * M * K * N (multiply + add per element)
    total_flops = 2 * M * K * N

    # Bytes calculation (all tensors from/to DRAM worst case)
    bytes_per_elem = dtype.value
    total_bytes = int((M * K + K * N + M * N) * bytes_per_elem)

    # Ideal compute time
    tensix_mul_adds_per_cycle = fpu_mm_flops_per_core_per_cycle(fidelity)
    ideal_compute_cycles = math.ceil(
        (total_flops / (num_cores * tensix_mul_adds_per_cycle))
    )
    ideal_compute_ns = ideal_compute_cycles / hw.clock_ghz  # cycles / (cycles/ns)

    # Ideal memory time
    ideal_memory_ns = total_bytes / hw.dram_bw_gb_s  # bytes / (bytes/ns)

    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=total_flops,
        total_bytes=total_bytes,
        ideal_compute_ns=ideal_compute_ns,
        ideal_memory_ns=ideal_memory_ns,
        hw=hw,
    )
