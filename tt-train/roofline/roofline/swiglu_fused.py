# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance models for fused SwiGLU operations.

This module provides roofline estimates for two fused SwiGLU implementations:
1. swiglu_fused_row_mcast: Reads input once, weights multiple times (8x for Wormhole, 11x for Blackhole)
2. swiglu_fused_mcast: Reads everything only once (optimal)

SwiGLU computation: output = silu(x @ w1) * (x @ w2) @ w3
Where:
- x: [batch, seq_len, embedding_size]
- w1, w2: [embedding_size, hidden_size] (gate and up projections)
- w3: [hidden_size, embedding_size] (down projection)
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import math

from ..hardware import HardwareSpec, DataType, MathFidelity

if TYPE_CHECKING:
    from .roofline import RooflineEstimate


def _get_weight_read_multiplier(hw: HardwareSpec) -> int:
    """Get the weight read multiplier based on hardware.

    For row multicast implementations:
    - Wormhole (n150/n300): 8x weight reads
    - Blackhole (p100/p150): 11x weight reads

    Args:
        hw: Hardware specification

    Returns:
        Weight read multiplier
    """
    if "Wormhole" in hw.name:
        return 8
    elif "Blackhole" in hw.name:
        return 11
    else:
        # Default to Wormhole behavior
        return 8


def swiglu_fused_row_mcast_roofline(
    hw: HardwareSpec,
    batch_seq: int,
    embedding_size: int,
    hidden_size: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "SwiGLU.fused_row_mcast",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for fused SwiGLU with row multicast.

    In this implementation:
    - Input is read once
    - Weight matrices (w1, w2, w3) are read multiple times (8x for Wormhole, 11x for Blackhole)
    - SiLU and element-wise mul are done on the fly

    Args:
        hw: Hardware specification
        batch_seq: Batch size * sequence length (M dimension)
        embedding_size: Input/output embedding dimension
        hidden_size: Intermediate hidden dimension
        dtype: Data type for operands
        fidelity: Math fidelity level
        num_cores: Number of cores to use (default: all)
        operation: Name for the operation
        phase: "forward" or "backward"

    Returns:
        RooflineEstimate with performance metrics
    """
    from .roofline import (
        RooflineEstimate,
        fpu_mm_flops_per_core_per_cycle,
        sfpu_flops_per_core_per_cycle,
    )

    if num_cores is None:
        num_cores = hw.tensix_cores_per_chip

    bytes_per_elem = dtype.value
    weight_read_mult = _get_weight_read_multiplier(hw)

    # Memory traffic calculation
    # Input read once: batch_seq * embedding_size
    input_bytes = batch_seq * embedding_size * bytes_per_elem

    # Weights read multiple times:
    # w1, w2: embedding_size * hidden_size each
    # w3: hidden_size * embedding_size
    weight_bytes_per_read = (
        2 * embedding_size * hidden_size + hidden_size * embedding_size  # w1 + w2  # w3
    ) * bytes_per_elem
    weights_total_bytes = weight_read_mult * weight_bytes_per_read

    # Output write once: batch_seq * embedding_size
    output_bytes = batch_seq * embedding_size * bytes_per_elem

    total_bytes = int(input_bytes + weights_total_bytes + output_bytes)

    # FLOPs calculation
    # x @ w1: 2 * batch_seq * embedding_size * hidden_size
    # x @ w2: 2 * batch_seq * embedding_size * hidden_size
    # (result) @ w3: 2 * batch_seq * hidden_size * embedding_size
    # Total matmul FLOPs: 6 * batch_seq * embedding_size * hidden_size
    matmul_flops = 6 * batch_seq * embedding_size * hidden_size

    # SiLU and mul are done on the fly (on SFPU)
    # silu: ~8 ops per element (approximation)
    # mul: 1 op per element
    eltwise_flops = 9 * batch_seq * hidden_size

    total_flops = matmul_flops + eltwise_flops

    # Compute time - matmuls on FPU, elementwise on SFPU
    fpu_ops_per_cycle = fpu_mm_flops_per_core_per_cycle(fidelity)
    sfpu_ops_per_cycle = sfpu_flops_per_core_per_cycle(fidelity)

    matmul_compute_cycles = math.ceil(matmul_flops / (num_cores * fpu_ops_per_cycle))
    eltwise_compute_cycles = math.ceil(eltwise_flops / (num_cores * sfpu_ops_per_cycle))

    # Total compute time (matmul dominates, eltwise is pipelined but we add it conservatively)
    ideal_compute_cycles = matmul_compute_cycles + eltwise_compute_cycles
    ideal_compute_ns = ideal_compute_cycles / hw.clock_ghz

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


def swiglu_fused_mcast_roofline(
    hw: HardwareSpec,
    batch_seq: int,
    embedding_size: int,
    hidden_size: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "SwiGLU.fused_mcast",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for fully fused SwiGLU with optimal multicast.

    In this implementation everything is read only once:
    - Input is read once
    - All weight matrices (w1, w2, w3) are read once
    - SiLU and element-wise mul are done on the fly

    This represents the optimal memory bandwidth for fused SwiGLU.

    Args:
        hw: Hardware specification
        batch_seq: Batch size * sequence length (M dimension)
        embedding_size: Input/output embedding dimension
        hidden_size: Intermediate hidden dimension
        dtype: Data type for operands
        fidelity: Math fidelity level
        num_cores: Number of cores to use (default: all)
        operation: Name for the operation
        phase: "forward" or "backward"

    Returns:
        RooflineEstimate with performance metrics
    """
    from .roofline import (
        RooflineEstimate,
        fpu_mm_flops_per_core_per_cycle,
        sfpu_flops_per_core_per_cycle,
    )

    if num_cores is None:
        num_cores = hw.tensix_cores_per_chip

    bytes_per_elem = dtype.value

    # Memory traffic calculation - everything read once
    # Input: batch_seq * embedding_size
    input_bytes = batch_seq * embedding_size * bytes_per_elem

    # Weights read once:
    # w1, w2: embedding_size * hidden_size each
    # w3: hidden_size * embedding_size
    weights_bytes = (
        2 * embedding_size * hidden_size + hidden_size * embedding_size  # w1 + w2  # w3
    ) * bytes_per_elem

    # Output write once: batch_seq * embedding_size
    output_bytes = batch_seq * embedding_size * bytes_per_elem

    total_bytes = int(input_bytes + weights_bytes + output_bytes)

    # FLOPs calculation (same as row_mcast - compute doesn't change)
    # x @ w1: 2 * batch_seq * embedding_size * hidden_size
    # x @ w2: 2 * batch_seq * embedding_size * hidden_size
    # (result) @ w3: 2 * batch_seq * hidden_size * embedding_size
    # Total matmul FLOPs: 6 * batch_seq * embedding_size * hidden_size
    matmul_flops = 6 * batch_seq * embedding_size * hidden_size

    # SiLU and mul are done on the fly (on SFPU)
    # silu: ~8 ops per element (approximation)
    # mul: 1 op per element
    eltwise_flops = 9 * batch_seq * hidden_size

    total_flops = matmul_flops + eltwise_flops

    # Compute time - matmuls on FPU, elementwise on SFPU
    fpu_ops_per_cycle = fpu_mm_flops_per_core_per_cycle(fidelity)
    sfpu_ops_per_cycle = sfpu_flops_per_core_per_cycle(fidelity)

    matmul_compute_cycles = math.ceil(matmul_flops / (num_cores * fpu_ops_per_cycle))
    eltwise_compute_cycles = math.ceil(eltwise_flops / (num_cores * sfpu_ops_per_cycle))

    # Total compute time (matmul dominates, eltwise is pipelined but we add it conservatively)
    ideal_compute_cycles = matmul_compute_cycles + eltwise_compute_cycles
    ideal_compute_ns = ideal_compute_cycles / hw.clock_ghz

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
