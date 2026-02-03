# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Layer normalization operation for roofline modeling.

This module provides MockLayerNormOp for roofline estimation of
layer normalization operations.
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..roofline import layernorm_roofline
from .operation import (
    RooflineFunctionContext,
    RooflineFunction,
    create_grad_tensor,
    create_activation_tensor,
)

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockLayerNormOp(RooflineFunction):
    """Roofline estimation for layer normalization.

    Forward: output = (input - mean) / sqrt(var + eps) * gamma + beta
    Backward: Complex - involves gradients for mean, variance, gamma, beta
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        input: MockTensor,
        gamma: MockTensor,
        beta: Optional[MockTensor] = None,
    ) -> MockTensor:
        """Forward pass: apply layer normalization.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            input: Input tensor [..., embedding_dim]
            gamma: Scale parameter [1, 1, 1, embedding_dim]
            beta: Optional bias parameter [1, 1, 1, embedding_dim]

        Returns:
            Normalized output tensor with same shape as input
        """
        ctx.save_for_backward(input, gamma, beta)

        # Get dimensions
        embedding_dim = input.shape[-1]
        num_tokens = input.logical_volume() // embedding_dim

        estimate = layernorm_roofline(
            roofline_ctx.hw,
            num_tokens,
            embedding_dim,
            dtype=input.dtype,
            operation="LayerNorm.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        return create_activation_tensor(input.shape, input.dtype, input.layout)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, MockTensor, Optional[MockTensor]]:
        """Backward pass: compute gradients for input, gamma, beta.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor from upstream

        Returns:
            (grad_input, grad_gamma, grad_beta) tuple
        """
        input, gamma, beta = ctx.saved_tensors

        embedding_dim = input.shape[-1]
        num_tokens = input.logical_volume() // embedding_dim

        estimate = layernorm_roofline(
            roofline_ctx.hw,
            num_tokens,
            embedding_dim,
            dtype=input.dtype,
            operation="LayerNorm.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        grad_input = create_grad_tensor(
            input.shape, input.dtype, input.layout, name="grad_input"
        )
        grad_gamma = create_grad_tensor(
            gamma.shape, gamma.dtype, gamma.layout, name="grad_gamma"
        )
        grad_beta = None
        if beta is not None:
            grad_beta = create_grad_tensor(
                beta.shape, beta.dtype, beta.layout, name="grad_beta"
            )

        return grad_input, grad_gamma, grad_beta
