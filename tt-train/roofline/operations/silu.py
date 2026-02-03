# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SiLU (Swish) activation operation for roofline modeling.

This module provides MockSiLUOp for roofline estimation of
SiLU activation operations.
"""

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..roofline import elementwise_roofline
from .operation import (
    RooflineFunctionContext,
    RooflineFunction,
    create_grad_tensor,
    create_activation_tensor,
)

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockSiLUOp(RooflineFunction):
    """Roofline estimation for SiLU (Swish) activation.

    SiLU(x) = x * sigmoid(x)

    Forward: output = silu(input)
    Backward: grad_input = grad_output * silu_bw(input)
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        input: MockTensor,
    ) -> MockTensor:
        """Forward pass: output = silu(input).

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            input: Input tensor

        Returns:
            Output tensor with same shape as input
        """
        ctx.save_for_backward(input)

        num_elements = input.logical_volume()

        # SiLU = x * sigmoid(x)
        # sigmoid is ~4 SFPU ops (~5 for exp, add, recip)
        # then 1 mul = ~5 total ops per element
        estimate = elementwise_roofline(
            roofline_ctx.hw,
            num_elements,
            num_inputs=1,
            sfpu_ops_per_element=8.0,
            fpu_ops_per_element=0.0,
            dtype=input.dtype,
            operation="SiLU.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        return create_activation_tensor(input.shape, input.dtype, input.layout)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor]:
        """Backward pass: grad_input = grad_output * silu_bw(input).

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor from upstream

        Returns:
            (grad_input,) tuple
        """
        (input,) = ctx.saved_tensors

        num_elements = input.logical_volume()

        # SiLU backward: silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        # = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        # ~8 ops per element: sigmoid (4), mul, sub, mul, add, mul with grad
        estimate = elementwise_roofline(
            roofline_ctx.hw,
            num_elements,
            num_inputs=2,  # grad_output and input
            sfpu_ops_per_element=8.0,
            fpu_ops_per_element=0.0,
            dtype=input.dtype,
            operation="SiLU.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        grad_input = create_grad_tensor(
            input.shape, input.dtype, input.layout, name="grad_input"
        )

        return (grad_input,)
