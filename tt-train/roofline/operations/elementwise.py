# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Elementwise operations for roofline modeling.

This module provides MockAddOp, MockMulOp, and MockGELUOp for roofline
estimation of elementwise operations.
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..roofline import elementwise_roofline
from .operation import (
    RooflineFunctionContext,
    RooflineFunction,
    create_grad_tensor,
    create_activation_tensor,
)

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockAddOp(RooflineFunction):
    """Roofline estimation for elementwise addition.

    Forward: output = a + b
    Backward:
        - grad_a = grad_output
        - grad_b = grad_output
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        a: MockTensor,
        b: MockTensor,
    ) -> MockTensor:
        """Forward pass: output = a + b.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            a: First input tensor
            b: Second input tensor

        Returns:
            Output tensor with same shape as inputs
        """
        ctx.save_for_backward(a, b)

        num_elements = a.logical_volume()

        estimate = elementwise_roofline(
            roofline_ctx.hw,
            num_elements,
            num_inputs=2,
            sfpu_ops_per_element=0.0,
            fpu_ops_per_element=1.0,
            dtype=a.dtype,
            operation="Add.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        return create_activation_tensor(a.shape, a.dtype, a.layout)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, MockTensor]:
        """Backward pass: grad_a = grad_b = grad_output.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor from upstream

        Returns:
            (grad_a, grad_b) tuple
        """
        a, b = ctx.saved_tensors

        num_elements = a.logical_volume()

        # grad_a and grad_b are just copies of grad_output
        # This is essentially 2 memory copy operations
        estimate = elementwise_roofline(
            roofline_ctx.hw,
            num_elements * 2,  # Two outputs
            num_inputs=1,
            sfpu_ops_per_element=0.0,
            fpu_ops_per_element=0.0,
            dtype=a.dtype,
            operation="Add.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        grad_a = create_grad_tensor(a.shape, a.dtype, a.layout, name="grad_a")
        grad_b = create_grad_tensor(b.shape, b.dtype, b.layout, name="grad_b")

        return grad_a, grad_b


class MockMulOp(RooflineFunction):
    """Roofline estimation for elementwise multiplication.

    Forward: output = a * b
    Backward:
        - grad_a = grad_output * b
        - grad_b = grad_output * a
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        a: MockTensor,
        b: MockTensor,
    ) -> MockTensor:
        """Forward pass: output = a * b.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            a: First input tensor
            b: Second input tensor

        Returns:
            Output tensor with same shape as inputs
        """
        ctx.save_for_backward(a, b)

        num_elements = a.logical_volume()

        estimate = elementwise_roofline(
            roofline_ctx.hw,
            num_elements,
            num_inputs=2,
            sfpu_ops_per_element=0.0,
            fpu_ops_per_element=1.0,
            dtype=a.dtype,
            operation="Mul.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        return create_activation_tensor(a.shape, a.dtype, a.layout)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, MockTensor]:
        """Backward pass: grad_a = grad_output * b, grad_b = grad_output * a.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor from upstream

        Returns:
            (grad_a, grad_b) tuple
        """
        a, b = ctx.saved_tensors

        num_elements = a.logical_volume()

        # grad_a = grad_output * b: read grad_output and b, write grad_a
        # TODO: I think add and mul are using fpu right now. Verify this
        estimate = elementwise_roofline(
            roofline_ctx.hw,
            num_elements,
            num_inputs=2,
            sfpu_ops_per_element=0.0,
            fpu_ops_per_element=1.0,
            dtype=a.dtype,
            operation="Mul.backward.grad_a",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        # grad_b = grad_output * a: read grad_output and a, write grad_b
        estimate = elementwise_roofline(
            roofline_ctx.hw,
            num_elements,
            num_inputs=2,
            sfpu_ops_per_element=0.0,
            fpu_ops_per_element=1.0,
            dtype=b.dtype,
            operation="Mul.backward.grad_b",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        grad_a = create_grad_tensor(a.shape, a.dtype, a.layout, name="grad_a")
        grad_b = create_grad_tensor(b.shape, b.dtype, b.layout, name="grad_b")

        return grad_a, grad_b


class MockGELUOp(RooflineFunction):
    """Roofline estimation for GELU activation.

    GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Forward: output = gelu(input)
    Backward: grad_input = grad_output * gelu'(input)
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        input: MockTensor,
    ) -> MockTensor:
        """Forward pass: output = gelu(input).

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            input: Input tensor

        Returns:
            Output tensor with same shape as input
        """
        ctx.save_for_backward(input)

        num_elements = input.logical_volume()

        # GELU is compute-intensive: exp/tanh approximation ~10 FLOPs per element
        estimate = elementwise_roofline(
            roofline_ctx.hw,
            num_elements,
            num_inputs=1,
            sfpu_ops_per_element=10.0,  # TODO: Find better estimate for this
            fpu_ops_per_element=0.0,
            dtype=input.dtype,
            operation="GELU.forward",
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
        """Backward pass: grad_input = grad_output * gelu'(input).

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor from upstream

        Returns:
            (grad_input,) tuple
        """
        (input,) = ctx.saved_tensors

        num_elements = input.logical_volume()

        # GELU backward also involves computing derivative ~15 FLOPs per element
        estimate = elementwise_roofline(
            roofline_ctx.hw,
            num_elements,
            num_inputs=2,  # grad_output and input
            sfpu_ops_per_element=15.0,
            fpu_ops_per_element=0.0,
            dtype=input.dtype,
            operation="GELU.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        grad_input = create_grad_tensor(
            input.shape, input.dtype, input.layout, name="grad_input"
        )

        return (grad_input,)
