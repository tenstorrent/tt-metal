# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Gradient utility functions for roofline modeling.

This module provides mock implementations of gradient utilities
for roofline estimation.
"""

from __future__ import annotations
from typing import Dict, TYPE_CHECKING
import math

from ..mock_tensor import MockTensor
from ..roofline import elementwise_roofline, reduction_roofline, RooflineEstimate

if TYPE_CHECKING:
    from ..roofline import RooflineContext


def mock_clip_grad_norm(
    ctx: "RooflineContext",
    parameters: Dict[str, MockTensor],
    max_norm: float = 1.0,
) -> None:
    """Estimate cost of gradient norm clipping.

    This function estimates the roofline cost of:
    1. Computing the total gradient norm (sum of squared norms per param)
    2. Conditionally scaling all gradients

    Args:
        ctx: Roofline context for estimates
        parameters: Dictionary of parameter names to MockTensors
        max_norm: Maximum gradient norm
    """
    # Calculate total gradient elements
    total_elements = sum(p.logical_volume() for p in parameters.values())

    # Use first parameter's dtype (assume all same)
    dtype = next(iter(parameters.values())).dtype

    # Phase 1: Compute gradient norm
    # For each parameter: compute sum of squares
    # Then sum across all parameters and sqrt

    # Sum of squares per parameter (elementwise square + reduction)
    estimate = elementwise_roofline(
        ctx.hw,
        total_elements,
        num_inputs=1,  # Read gradient
        sfpu_ops_per_element=1.0,  # Square
        fpu_ops_per_element=0.0,
        dtype=dtype,
        operation="GradNorm.square",
        phase="grad_clip",
    )
    ctx.add_perf_result(estimate)

    # Reduction to compute sum
    estimate = reduction_roofline(
        ctx.hw,
        total_elements,
        total_elements,  # Reduce all to single value
        dtype=dtype,
        operation="GradNorm.reduce",
        phase="grad_clip",
    )
    ctx.add_perf_result(estimate)

    # Phase 2: Scale gradients (conditional, but we model worst case)
    # Read gradients, multiply by scale, write back
    estimate = elementwise_roofline(
        ctx.hw,
        total_elements,
        num_inputs=1,  # Read gradient
        sfpu_ops_per_element=0.0,
        fpu_ops_per_element=1.0,  # Multiply by scale
        dtype=dtype,
        operation="GradNorm.scale",
        phase="grad_clip",
    )
    ctx.add_perf_result(estimate)


def mock_zero_grad(
    ctx: "RooflineContext",
    parameters: Dict[str, MockTensor],
) -> None:
    """Estimate cost of zeroing gradients.

    Args:
        ctx: Roofline context for estimates
        parameters: Dictionary of parameter names to MockTensors
    """
    total_elements = sum(p.logical_volume() for p in parameters.values())
    dtype = next(iter(parameters.values())).dtype

    # Just write zeros to all gradients
    estimate = elementwise_roofline(
        ctx.hw,
        total_elements,
        num_inputs=0,  # No reads needed for zero
        sfpu_ops_per_element=0.0,
        fpu_ops_per_element=0.0,  # No compute
        dtype=dtype,
        operation="ZeroGrad",
        phase="grad_zero",
    )

    # Adjust to only count writes
    estimate = RooflineEstimate(
        operation=estimate.operation,
        phase=estimate.phase,
        total_flops=0,
        total_bytes=int(total_elements * dtype.value),
        ideal_compute_ns=0,
        ideal_memory_ns=total_elements * dtype.value / ctx.hw.dram_bw_gb_s,
        hw=ctx.hw,
    )
    ctx.add_perf_result(estimate)
