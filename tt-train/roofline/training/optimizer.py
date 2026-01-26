# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Optimizer roofline estimation for training.

This module provides MockAdamW for roofline estimation of
optimizer weight update operations.
"""

from __future__ import annotations
from typing import Dict, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..roofline import RooflineEstimate, elementwise_roofline

if TYPE_CHECKING:
    from ..roofline import RooflineContext
    from ..modules import MockModule


class MockAdamW:
    """Mock AdamW optimizer for roofline estimation.

    Estimates the cost of optimizer weight updates:
    - Read: weight, grad, m (momentum), v (variance) - 4 tensors per param
    - Write: weight, m, v - 3 tensors per param
    - Compute: ~10 FLOPs per element (sqrt, div, mul, add, etc.)

    Example:
        >>> model = MockNanoGPT(config)
        >>> optimizer = MockAdamW(model.parameters())
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> # ... run forward/backward ...
        >>> optimizer.step(ctx)
    """

    def __init__(
        self,
        parameters: Dict[str, MockTensor],
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """Initialize AdamW optimizer.

        Args:
            parameters: Dictionary of parameter names to MockTensors
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term for numerical stability
            weight_decay: Weight decay coefficient
        """
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Calculate total parameter count
        self.total_params = sum(p.logical_volume() for p in parameters.values())

    def step(self, ctx: "RooflineContext") -> None:
        """Estimate optimizer step cost.

        This adds roofline estimates for the weight update operation
        to the context.

        Args:
            ctx: Roofline context for estimates
        """
        for name, param in self.parameters.items():
            num_elements = param.logical_volume()

            # AdamW update per parameter:
            # Read: weight, grad, m, v (4 tensors)
            # Write: weight, m, v (3 tensors)
            # Total: 7 tensor reads/writes
            # FLOPs: ~10 per element (m update, v update, bias correction, update)

            # TODO: check if adamw uses fpu. Check fused kernel
            estimate = elementwise_roofline(
                ctx.hw,
                num_elements,
                num_inputs=4,  # weight, grad, m, v
                sfpu_ops_per_element=10.0,  # sqrt, div, mul, add operations
                fpu_ops_per_element=0.0,
                dtype=param.dtype,
                operation=f"AdamW.{name}",
                phase="optimizer",
            )

            # Adjust bytes for writing 3 tensors (weight, m, v)
            # The elementwise_roofline assumes 1 output, we have 3
            estimate = RooflineEstimate(
                operation=estimate.operation,
                phase=estimate.phase,
                total_flops=estimate.total_flops,
                total_bytes=int(7 * num_elements * param.dtype.value),
                ideal_compute_ns=estimate.ideal_compute_ns,
                ideal_memory_ns=7
                * num_elements
                * param.dtype.value
                / ctx.hw.dram_bw_gb_s,
                hw=ctx.hw,
            )

            ctx.add_perf_result(estimate)

    def estimate_memory(self) -> int:
        """Estimate optimizer state memory.

        AdamW stores m (momentum) and v (variance) for each parameter.

        Returns:
            Total optimizer state memory in bytes
        """
        total_bytes = 0
        for param in self.parameters.values():
            # m and v states, same size as parameter
            total_bytes += 2 * param.bytes()
        return total_bytes

    def __repr__(self) -> str:
        return (
            f"MockAdamW(params={len(self.parameters)}, "
            f"total_elements={self.total_params:,}, "
            f"lr={self.lr}, weight_decay={self.weight_decay})"
        )
