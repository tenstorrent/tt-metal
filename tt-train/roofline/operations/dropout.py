# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Dropout operation for roofline modeling.

This module provides MockDropoutOp for roofline estimation of
dropout operations.
"""

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..roofline import dropout_roofline
from .operation import RooflineFunctionContext, RooflineFunction

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockDropoutOp(RooflineFunction):
    """Roofline estimation for dropout.

    Forward: output = input * mask * scale (where scale = 1/(1-p))
    Backward: grad_input = grad_output * mask * scale
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        input: MockTensor,
        dropout_prob: float = 0.1,
    ) -> MockTensor:
        """Forward pass: apply dropout.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            input: Input tensor
            dropout_prob: Dropout probability

        Returns:
            Output tensor with same shape as input
        """
        ctx.save_for_backward(input, dropout_prob)

        num_elements = input.logical_volume()

        estimate = dropout_roofline(
            roofline_ctx.hw,
            num_elements,
            dtype=input.dtype,
            operation="Dropout.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        return MockTensor(input.shape, input.dtype, input.layout, requires_grad=True)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor]:
        """Backward pass: apply same mask to gradient.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor from upstream

        Returns:
            (grad_input,) tuple
        """
        input, dropout_prob = ctx.saved_tensors

        num_elements = input.logical_volume()

        estimate = dropout_roofline(
            roofline_ctx.hw,
            num_elements,
            dtype=input.dtype,
            operation="Dropout.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        grad_input = MockTensor(
            input.shape, input.dtype, input.layout, requires_grad=False
        )

        return (grad_input,)
