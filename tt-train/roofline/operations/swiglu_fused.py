# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fused SwiGLU operation for roofline modeling.

This module provides MockSwiGLUFusedOp for roofline estimation of
fused SwiGLU operations with different multicast strategies.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..roofline import swiglu_fused_row_mcast_roofline, swiglu_fused_mcast_roofline
from .operation import (
    RooflineFunctionContext,
    RooflineFunction,
    create_grad_tensor,
    create_activation_tensor,
)

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class SwiGLUFusedImpl(Enum):
    """Fused SwiGLU implementation variants."""

    ROW_MCAST = "swiglu_fused_row_mcast"
    MCAST = "swiglu_fused_mcast"


class MockSwiGLUFusedOp(RooflineFunction):
    """Roofline estimation for fused SwiGLU operation.

    Fused computation: output = silu(x @ w1) * (x @ w2) @ w3

    Supports two implementation variants:
    - ROW_MCAST: Reads input once, weights multiple times
    - MCAST: Reads everything only once (optimal)

    Forward: output = silu(x @ w1) * (x @ w2) @ w3
    Backward:
        - grad_input = backward through all three matmuls
        - grad_w1, grad_w2, grad_w3 = weight gradients
    """

    # Class variable to control which implementation to use
    impl: SwiGLUFusedImpl = SwiGLUFusedImpl.ROW_MCAST

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        input: MockTensor,
        w1: MockTensor,
        w2: MockTensor,
        w3: MockTensor,
        impl: SwiGLUFusedImpl = SwiGLUFusedImpl.ROW_MCAST,
    ) -> MockTensor:
        """Forward pass: fused SwiGLU computation.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            input: Input tensor [batch, 1, seq_len, embedding_size]
            w1: Gate weight tensor [1, 1, hidden_size, embedding_size]
            w2: Up weight tensor [1, 1, hidden_size, embedding_size]
            w3: Down weight tensor [1, 1, embedding_size, hidden_size]
            impl: Which fused implementation to model

        Returns:
            Output tensor [batch, 1, seq_len, embedding_size]
        """
        ctx.save_for_backward(input, w1, w2, w3, impl)

        # Get dimensions
        embedding_size = input.shape[-1]
        hidden_size = w1.shape[2]  # w1: [1, 1, hidden_size, embedding_size]
        batch_seq = input.logical_volume() // embedding_size

        # Select roofline model based on implementation
        if impl == SwiGLUFusedImpl.ROW_MCAST:
            estimate = swiglu_fused_row_mcast_roofline(
                roofline_ctx.hw,
                batch_seq,
                embedding_size,
                hidden_size,
                dtype=input.dtype,
                operation="SwiGLU.fused_row_mcast.forward",
                phase="forward",
            )
        else:  # MCAST
            estimate = swiglu_fused_mcast_roofline(
                roofline_ctx.hw,
                batch_seq,
                embedding_size,
                hidden_size,
                dtype=input.dtype,
                operation="SwiGLU.fused_mcast.forward",
                phase="forward",
            )

        roofline_ctx.add_perf_result(estimate)

        # Output has same shape as input
        output_shape = input.shape
        return create_activation_tensor(output_shape, input.dtype, input.layout)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, MockTensor, MockTensor, MockTensor]:
        """Backward pass: compute gradients for input and weights.

        For fused SwiGLU backward, we need to compute:
        - grad_input: gradient w.r.t. input
        - grad_w1, grad_w2, grad_w3: gradients w.r.t. weights

        The backward is roughly 2-3x the forward FLOPs (similar to 3 linear backward passes).

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor [batch, 1, seq_len, embedding_size]

        Returns:
            (grad_input, grad_w1, grad_w2, grad_w3) tuple
        """
        input, w1, w2, w3, impl = ctx.saved_tensors

        embedding_size = input.shape[-1]
        hidden_size = w1.shape[2]
        batch_seq = input.logical_volume() // embedding_size

        # TODO: Need to make a proper backward model
        # Backward pass has approximately 2x the forward compute
        # (dL/dx, dL/dw1, dL/dw2, dL/dw3)
        # Using same memory model but with backward operation name

        if impl == SwiGLUFusedImpl.ROW_MCAST:
            estimate = swiglu_fused_row_mcast_roofline(
                roofline_ctx.hw,
                batch_seq,
                embedding_size,
                hidden_size,
                dtype=input.dtype,
                operation="SwiGLU.fused_row_mcast.backward",
                phase="backward",
            )
        else:  # MCAST
            estimate = swiglu_fused_mcast_roofline(
                roofline_ctx.hw,
                batch_seq,
                embedding_size,
                hidden_size,
                dtype=input.dtype,
                operation="SwiGLU.fused_mcast.backward",
                phase="backward",
            )

        # Scale up for backward (approximately 2x forward)
        # This accounts for: grad_input, grad_w1, grad_w2, grad_w3 computations
        estimate_bwd = type(estimate)(
            operation=estimate.operation,
            phase=estimate.phase,
            total_flops=int(estimate.total_flops * 2),
            total_bytes=int(estimate.total_bytes * 2),
            ideal_compute_ns=estimate.ideal_compute_ns * 2,
            ideal_memory_ns=estimate.ideal_memory_ns * 2,
            hw=estimate.hw,
        )
        roofline_ctx.add_perf_result(estimate_bwd)

        # Create gradient tensors
        grad_input = create_grad_tensor(
            input.shape, input.dtype, input.layout, name="grad_input"
        )
        grad_w1 = create_grad_tensor(w1.shape, w1.dtype, w1.layout, name="grad_w1")
        grad_w2 = create_grad_tensor(w2.shape, w2.dtype, w2.layout, name="grad_w2")
        grad_w3 = create_grad_tensor(w3.shape, w3.dtype, w3.layout, name="grad_w3")

        return grad_input, grad_w1, grad_w2, grad_w3
