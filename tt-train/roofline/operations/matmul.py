# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Matrix multiplication operation for roofline modeling.

This module provides MockMatMulOp for roofline estimation of
matrix multiplication operations.
"""

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..roofline import matmul_roofline
from .operation import RooflineFunctionContext, RooflineFunction

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockMatMulOp(RooflineFunction):
    """Roofline estimation for matrix multiplication.

    Forward: output = A @ B
    Backward:
        - grad_A = grad_output @ B.T
        - grad_B = A.T @ grad_output
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        A: MockTensor,
        B: MockTensor,
    ) -> MockTensor:
        """Forward pass: output = A @ B.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            A: First input tensor [..., M, K]
            B: Second input tensor [..., K, N]

        Returns:
            Output tensor [..., M, N]
        """
        ctx.save_for_backward(A, B)

        M = A.shape[-2]
        K = A.shape[-1]
        N = B.shape[-1]

        # Batch size (product of all dims except last 2)
        batch = A.logical_volume() // (M * K)

        estimate = matmul_roofline(
            roofline_ctx.hw,
            batch * M,
            K,
            N,
            dtype=A.dtype,
            operation="MatMul.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        output_shape = A.shape[:-1] + (N,)
        return MockTensor(output_shape, A.dtype, A.layout, requires_grad=True)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, MockTensor]:
        """Backward pass for matmul.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor [..., M, N]

        Returns:
            (grad_A, grad_B) tuple
        """
        A, B = ctx.saved_tensors

        M = A.shape[-2]
        K = A.shape[-1]
        N = B.shape[-1]
        batch = A.logical_volume() // (M * K)

        # grad_A = grad_output @ B.T
        estimate = matmul_roofline(
            roofline_ctx.hw,
            batch * M,
            N,
            K,
            dtype=A.dtype,
            operation="MatMul.backward.grad_A",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)
        grad_A = MockTensor(A.shape, A.dtype, A.layout, requires_grad=False)

        # grad_B = A.T @ grad_output
        estimate = matmul_roofline(
            roofline_ctx.hw,
            batch * K,
            M,
            N,
            dtype=B.dtype,
            operation="MatMul.backward.grad_B",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)
        grad_B = MockTensor(B.shape, B.dtype, B.layout, requires_grad=False)

        return grad_A, grad_B
