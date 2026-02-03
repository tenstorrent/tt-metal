# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm operation for roofline modeling.

This module provides MockRMSNormOp for roofline estimation of
RMS normalization operations.
"""

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..roofline import rmsnorm_roofline
from .operation import (
    RooflineFunctionContext,
    RooflineFunction,
    create_grad_tensor,
    create_activation_tensor,
)

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockRMSNormOp(RooflineFunction):
    """Roofline estimation for RMS normalization.

    RMSNorm(x) = x / rms(x) * gamma
    where rms(x) = sqrt(mean(x^2) + eps)

    Unlike LayerNorm, RMSNorm does not subtract mean or add beta.
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        input: MockTensor,
        gamma: MockTensor,
        epsilon: float = 1e-6,
    ) -> MockTensor:
        """Forward pass: output = rmsnorm(input) * gamma.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            input: Input tensor [..., embedding_dim]
            gamma: Scale parameter [1, 1, 1, embedding_dim]
            epsilon: Small constant for numerical stability

        Returns:
            Normalized output tensor
        """
        ctx.save_for_backward(input, gamma, epsilon)

        # Calculate dimensions
        # Input shape: [batch, 1, seq_len, embedding_dim]
        embedding_dim = input.shape[-1]
        num_tokens = input.logical_volume() // embedding_dim

        estimate = rmsnorm_roofline(
            roofline_ctx.hw,
            num_tokens,
            embedding_dim,
            dtype=input.dtype,
            operation="RMSNorm.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        return create_activation_tensor(input.shape, input.dtype, input.layout)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, MockTensor, None]:
        """Backward pass: compute gradients for input and gamma.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor from upstream

        Returns:
            (grad_input, grad_gamma, None) tuple
        """
        input, gamma, epsilon = ctx.saved_tensors

        embedding_dim = input.shape[-1]
        num_tokens = input.logical_volume() // embedding_dim

        estimate = rmsnorm_roofline(
            roofline_ctx.hw,
            num_tokens,
            embedding_dim,
            dtype=input.dtype,
            operation="RMSNorm.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        grad_input = create_grad_tensor(
            input.shape, input.dtype, input.layout, name="grad_input"
        )
        grad_gamma = create_grad_tensor(
            gamma.shape, gamma.dtype, gamma.layout, name="grad_gamma"
        )

        return grad_input, grad_gamma, None
