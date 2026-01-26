# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Linear operation for roofline modeling.

This module provides MockLinearOp for roofline estimation of
linear/fully-connected operations.
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..roofline import matmul_roofline, reduction_roofline
from .operation import RooflineFunctionContext, RooflineFunction

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockLinearOp(RooflineFunction):
    """Roofline estimation for linear/fully-connected operation.

    Forward: output = input @ weight.T + bias
    Backward:
        - grad_input = grad_output @ weight
        - grad_weight = grad_output.T @ input
        - grad_bias = sum(grad_output)
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        input: MockTensor,
        weight: MockTensor,
        bias: Optional[MockTensor] = None,
    ) -> MockTensor:
        """Forward pass: output = input @ weight.T + bias.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            input: Input tensor [*, in_features]
            weight: Weight tensor [1, 1, out_features, in_features]
            bias: Optional bias tensor [1, 1, 1, out_features]

        Returns:
            Output tensor [*, out_features]
        """
        ctx.save_for_backward(input, weight, bias)

        # Get dimensions
        in_features = input.shape[-1]
        out_features = weight.shape[2]
        M = input.logical_volume() // in_features  # batch * seq
        K = in_features
        N = out_features

        # Add forward matmul estimate
        estimate = matmul_roofline(
            roofline_ctx.hw,
            M,
            K,
            N,
            dtype=input.dtype,
            operation="Linear.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        # Compute output shape
        output_shape = input.shape[:-1] + (out_features,)
        return MockTensor(output_shape, input.dtype, input.layout, requires_grad=True)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, MockTensor, Optional[MockTensor]]:
        """Backward pass: compute gradients for input, weight, bias.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor [*, out_features]

        Returns:
            (grad_input, grad_weight, grad_bias) tuple
        """
        input, weight, bias = ctx.saved_tensors

        in_features = input.shape[-1]
        out_features = weight.shape[2]
        M = input.logical_volume() // in_features

        # grad_input = grad_output @ weight
        # Shape: [M, out_features] @ [out_features, in_features] -> [M, in_features]
        estimate = matmul_roofline(
            roofline_ctx.hw,
            M,
            out_features,
            in_features,
            dtype=input.dtype,
            operation="Linear.backward.grad_input",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)
        grad_input = MockTensor(
            input.shape, input.dtype, input.layout, requires_grad=False
        )

        # grad_weight = grad_output.T @ input
        # Shape: [out_features, M] @ [M, in_features] -> [out_features, in_features]
        estimate = matmul_roofline(
            roofline_ctx.hw,
            out_features,
            M,
            in_features,
            dtype=weight.dtype,
            operation="Linear.backward.grad_weight",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)
        grad_weight = MockTensor(
            weight.shape, weight.dtype, weight.layout, requires_grad=False
        )

        # grad_bias = sum(grad_output, dim=0)
        grad_bias = None
        if bias is not None:
            estimate = reduction_roofline(
                roofline_ctx.hw,
                M * out_features,
                M,
                dtype=bias.dtype,
                operation="Linear.backward.grad_bias",
                phase="backward",
            )
            roofline_ctx.add_perf_result(estimate)
            grad_bias = MockTensor(
                bias.shape, bias.dtype, bias.layout, requires_grad=False
            )

        return grad_input, grad_weight, grad_bias
