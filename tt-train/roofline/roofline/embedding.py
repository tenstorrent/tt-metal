# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance model for embedding operations."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..hardware import HardwareSpec, DataType, MathFidelity

if TYPE_CHECKING:
    from .roofline import RooflineEstimate


def embedding_roofline(
    hw: HardwareSpec,
    batch_seq: int,
    embedding_dim: int,
    vocab_size: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "Embedding",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for embedding lookup operation.

    Forward: Pure memory-bound gather operation.
    Backward: Scatter-add gradients to weight table.

    Args:
        hw: Hardware specification
        batch_seq: Batch size * sequence length (number of tokens)
        embedding_dim: Dimension of embeddings
        vocab_size: Size of vocabulary
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
        # Forward: Read embeddings for each token index
        # Read batch_seq embeddings, each of size embedding_dim
        # In worst case, may read from scattered locations in the embedding table
        total_bytes = int(batch_seq * embedding_dim * bytes_per_elem)
        total_flops = 0  # Pure gather, no compute
    else:
        # Backward: Scatter-add gradients to embedding table
        # Read grad_output, scatter-add to grad_weight
        # Grad output: batch_seq * embedding_dim
        # Grad weight (sparse update): batch_seq * embedding_dim (in practice may be more due to atomics)
        total_bytes = int(2 * batch_seq * embedding_dim * bytes_per_elem)
        total_flops = batch_seq * embedding_dim  # Addition for accumulation

    # Compute time (minimal for forward, some for backward accumulation)
    ops_per_cycle = sfpu_flops_per_core_per_cycle(fidelity)
    ideal_compute_ns = (
        total_flops / (num_cores * ops_per_cycle) / hw.clock_ghz
        if total_flops > 0
        else 0
    )

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
