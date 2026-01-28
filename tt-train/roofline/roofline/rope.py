# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance model for Rotary Position Embedding (RoPE)."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..hardware import HardwareSpec, DataType, MathFidelity

if TYPE_CHECKING:
    from .roofline import RooflineEstimate


def rope_roofline(
    hw: HardwareSpec,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "RoPE",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for Rotary Position Embedding (fused kernel).

    RoPE applies rotation based on position to Q and K tensors in attention.
    The implementation uses a fused kernel with transformation matrix approach:

        rotated_input = input @ trans_mat       # FPU matmul: permute [x₀,x₁,x₂,x₃] → [x₁,-x₀,x₃,-x₂]
        cos_part = input * cos_cache            # SFPU elementwise mul
        sin_part = rotated_input * sin_cache    # SFPU elementwise mul
        output = cos_part + sin_part            # SFPU elementwise add

    Since this is a fused kernel, intermediates stay in L1 and we only count
    input/output DRAM transfers. Compute includes both FPU (trans_mat) and SFPU ops.

    Args:
        hw: Hardware specification
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension per head
        dtype: Data type for operands
        fidelity: Math fidelity level
        num_cores: Number of cores to use (default: all)
        operation: Name for the operation
        phase: "forward" or "backward"

    Returns:
        RooflineEstimate for the fused operation
    """
    from .roofline import (
        RooflineEstimate,
        sfpu_flops_per_core_per_cycle,
        fpu_mm_flops_per_core_per_cycle,
    )

    if num_cores is None:
        num_cores = hw.tensix_cores_per_chip

    bytes_per_elem = dtype.value
    num_elements = batch_size * num_heads * seq_len * head_dim
    cache_elements = seq_len * head_dim

    # === Memory (fused kernel - intermediates stay in L1) ===
    # Inputs from DRAM:
    #   - input tensor: num_elements
    #   - cos_cache: cache_elements (broadcasted over batch*heads)
    #   - sin_cache: cache_elements (broadcasted over batch*heads)
    #   - trans_mat: 32x32 (tile size, broadcasted)
    # Output to DRAM:
    #   - output tensor: num_elements
    total_bytes = int(
        num_elements * bytes_per_elem  # input
        + 2 * cache_elements * bytes_per_elem  # sin + cos cache
        + 32 * 32 * bytes_per_elem  # trans_mat (tile size)
        + num_elements * bytes_per_elem  # output
    )

    # === Compute ===
    # FPU: rotated_input = input @ trans_mat
    #   M = B*H*S, K = D, N = D -> 2*M*K*N FLOPs
    M = batch_size * num_heads * seq_len
    K = head_dim
    N = head_dim
    fpu_flops = 2 * M * K * N

    # SFPU: 3 elementwise ops (cos_mul, sin_mul, add) -> 3 * num_elements
    sfpu_flops = 3 * num_elements

    total_flops = fpu_flops + sfpu_flops

    # Compute time: FPU and SFPU operations are sequential per tile in fused kernel
    fpu_ops_per_cycle = fpu_mm_flops_per_core_per_cycle(fidelity)
    sfpu_ops_per_cycle = sfpu_flops_per_core_per_cycle(fidelity)

    fpu_compute_ns = fpu_flops / (num_cores * fpu_ops_per_cycle) / hw.clock_ghz
    sfpu_compute_ns = sfpu_flops / (num_cores * sfpu_ops_per_cycle) / hw.clock_ghz

    # In fused kernel, FPU and SFPU ops are sequential per tile
    ideal_compute_ns = fpu_compute_ns + sfpu_compute_ns

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
