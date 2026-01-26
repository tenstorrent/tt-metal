# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance model for cross-entropy loss."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..hardware import HardwareSpec, DataType, MathFidelity

if TYPE_CHECKING:
    from .roofline import RooflineEstimate


def cross_entropy_roofline(
    hw: HardwareSpec,
    batch_seq: int,
    vocab_size: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "CrossEntropyLoss",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for cross-entropy loss.

    CrossEntropyLoss = -log(softmax(logits)[target])

    Forward:
        1. Softmax over vocab dimension (expensive for large vocab)
        2. Gather: select probability for target class
        3. Log + negate
        4. Reduction: mean/sum over batch

    Backward:
        - grad_logits = softmax(logits) - one_hot(target)
        - Then scale by upstream gradient

    Args:
        hw: Hardware specification
        batch_seq: Batch size * sequence length (number of predictions)
        vocab_size: Vocabulary size (number of classes)
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
    num_logits = batch_seq * vocab_size

    # TODO: Those numbers seem quite arbitrary.
    if phase == "forward":
        # Read logits, compute softmax, gather, log, reduce
        # Softmax dominates for large vocab
        total_bytes = int(num_logits * bytes_per_elem + batch_seq * bytes_per_elem)

        # FLOPs:
        # - Softmax: ~8 ops per element (max, exp, sum, div)
        # - Log: 1 op per selected element
        # - Reduction: batch_seq ops
        total_flops = 8 * num_logits + 2 * batch_seq
    else:
        # Backward: compute softmax gradient
        # grad = softmax - one_hot(target)
        # Need to recompute or cache softmax
        total_bytes = int(2 * num_logits * bytes_per_elem)

        # FLOPs: softmax recompute + subtraction
        total_flops = 9 * num_logits

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
