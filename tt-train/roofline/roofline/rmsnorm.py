# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance model for RMS normalization."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..hardware import HardwareSpec, DataType, MathFidelity

if TYPE_CHECKING:
    from .roofline import RooflineEstimate


def rmsnorm_roofline(
    hw: HardwareSpec,
    num_tokens: int,
    embedding_dim: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "RMSNorm",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for RMS normalization.

    RMSNorm is simpler than LayerNorm - no mean subtraction, no bias:
        y = x / sqrt(mean(x^2) + eps) * weight

    Forward:
        1. Compute sum of squares (reduction over embedding_dim)
        2. Compute RMS = sqrt(mean_sq + eps)
        3. Normalize: x / RMS * weight

    Backward:
        - grad_input involves recomputing RMS and scaling
        - grad_weight = sum(grad_output * normalized_input)

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

    if phase == "forward":
        # Forward pass:
        # Read input once, write output once
        # Also read weight (small, amortized over tokens)
        # No bias in RMSNorm
        total_bytes = int(
            2 * num_elements * bytes_per_elem  # input + output
            + embedding_dim * bytes_per_elem  # weight
        )

        # FLOPs: (N = num_elements, H = embedding_dim, T = num_tokens)
        # - Square: x^2 -> N ops
        # - Sum reduction: sum(x^2) -> N ops
        # - Add eps: + eps -> T ops
        # - Mean: / H -> T ops (1 per token)
        # - Sqrt: sqrt(mean) -> T ops
        # - Normalize: x / rms -> N ops
        # - Scale: * weight -> N ops
        # Total: ~4N + 3T ≈ 4N FLOPs (T << N typically)
        total_flops = 4 * num_elements
    else:
        # Backward pass:
        # Read grad_output, input (or cached normalized), weight
        # Write grad_input, grad_weight
        total_bytes = int(
            3 * num_elements * bytes_per_elem  # grad_output + input + grad_input
            + 2 * embedding_dim * bytes_per_elem  # weight + grad_weight
        )

        total_flops = 6 * num_elements

    # Compute time (SFPU for elementwise ops)
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
