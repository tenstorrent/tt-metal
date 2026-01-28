# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Rotary Position Embedding (RoPE) operation for roofline modeling.

This module provides MockRoPEOp for roofline estimation of
rotary position embedding operations.
"""

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..roofline import rope_roofline
from .operation import RooflineFunctionContext, RooflineFunction

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockRoPEOp(RooflineFunction):
    """Roofline estimation for Rotary Position Embedding.

    RoPE applies rotation based on position to attention Q and K tensors.
    The rotation uses precomputed sin/cos caches.

    Forward: output = x * cos + rotate_half(x) * sin
    Backward: grad_input = grad_output * cos + rotate_half(grad_output) * (-sin)
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        input: MockTensor,
        head_dim: int,
        token_position: int = 0,
    ) -> MockTensor:
        """Forward pass: apply rotary position embedding.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            input: Input tensor [B, num_heads, S, head_dim]
            head_dim: Dimension per head
            token_position: Starting token position (for caching)

        Returns:
            Output tensor with rotary embedding applied
        """
        ctx.save_for_backward(input, head_dim, token_position)

        # Parse dimensions [B, num_heads, S, head_dim]
        batch_size = input.shape[0]
        num_heads = input.shape[1]
        seq_len = input.shape[2]
        d = input.shape[3]

        estimate = rope_roofline(
            roofline_ctx.hw,
            batch_size,
            num_heads,
            seq_len,
            d,
            dtype=input.dtype,
            operation="RoPE.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        return MockTensor(input.shape, input.dtype, input.layout, requires_grad=True)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, None, None]:
        """Backward pass: compute gradient for input.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor from upstream

        Returns:
            (grad_input, None, None) tuple
        """
        input, head_dim, token_position = ctx.saved_tensors

        # Parse dimensions
        batch_size = input.shape[0]
        num_heads = input.shape[1]
        seq_len = input.shape[2]
        d = input.shape[3]

        estimate = rope_roofline(
            roofline_ctx.hw,
            batch_size,
            num_heads,
            seq_len,
            d,
            dtype=input.dtype,
            operation="RoPE.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        grad_input = MockTensor(
            input.shape, input.dtype, input.layout, requires_grad=False
        )

        return grad_input, None, None
