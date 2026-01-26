# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance model for layer normalization."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..hardware import HardwareSpec, DataType, MathFidelity

if TYPE_CHECKING:
    from .roofline import RooflineEstimate


def layernorm_roofline(
    hw: HardwareSpec,
    num_tokens: int,
    embedding_dim: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "LayerNorm",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for layer normalization.

    LayerNorm involves two passes over the data:
    1. Compute mean and variance (reduction)
    2. Normalize and apply scale/bias (elementwise)

    Forward:
        - Read input: num_tokens * embedding_dim
        - Compute mean: reduction over embedding_dim
        - Compute variance: reduction over embedding_dim
        - Normalize + scale/shift: elementwise
        - Write output: num_tokens * embedding_dim

    Backward:
        - More complex: involves gradient of mean/variance
        - Approximately 3x the forward cost

    Args:
        hw: Hardware specification
        num_tokens: Number of tokens (batch * seq)
        embedding_dim: Embedding dimension (normalization dimension)
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
    num_elements = num_tokens * embedding_dim

    # TODO: Investigate coefficients
    if phase == "forward":
        # Forward pass:
        # Read input once, write output once
        # Also read gamma/beta (small, amortized over tokens)
        total_bytes = int(
            5 * num_elements * bytes_per_elem + 4 * embedding_dim * bytes_per_elem
        )

        # FLOPs: (N = num_elements)
        # - Mean: sum(x) / H - ~N FLOPS
        # - Variance: (x - mean)^2 -- N substractions, N squarings, N additions - ~3N FLOPS
        # - Normalize: num_elements (x - mean) / sqrt(var + eps) * gamma + beta ~4N FLOPS
        # Total: ~8N FLOPS
        total_flops = 8 * num_elements
    else:
        total_bytes = int(
            4 * num_elements * bytes_per_elem + 4 * embedding_dim * bytes_per_elem
        )
        total_flops = 9 * num_elements

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
