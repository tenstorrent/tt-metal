# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-entropy loss operation for roofline modeling.

This module provides MockCrossEntropyLossOp for roofline estimation of
cross-entropy loss computation.
"""

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..roofline import cross_entropy_roofline
from .operation import (
    RooflineFunctionContext,
    RooflineFunction,
    create_grad_tensor,
    create_activation_tensor,
)

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockCrossEntropyLossOp(RooflineFunction):
    """Roofline estimation for cross-entropy loss.

    CrossEntropyLoss = -log(softmax(logits)[target])

    Forward: softmax + log + gather + reduce
    Backward: grad_logits = softmax - one_hot(target)
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        logits: MockTensor,
        targets: MockTensor,
    ) -> MockTensor:
        """Forward pass: compute cross-entropy loss.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            logits: Logits tensor [B, 1, S, vocab_size]
            targets: Target indices tensor [B, 1, 1, S]

        Returns:
            Loss tensor (scalar or per-token depending on reduction)
        """
        ctx.save_for_backward(logits, targets)

        # Get dimensions
        batch_seq = targets.logical_volume()
        vocab_size = logits.shape[-1]

        estimate = cross_entropy_roofline(
            roofline_ctx.hw,
            batch_seq,
            vocab_size,
            dtype=logits.dtype,
            operation="CrossEntropyLoss.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        # Output is scalar loss (or per-sample losses)
        output_shape = (1,)  # Scalar after reduction
        return create_activation_tensor(output_shape, logits.dtype, logits.layout)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, None]:
        """Backward pass: compute gradient for logits.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient from upstream (scalar)

        Returns:
            (grad_logits, None) tuple - None for targets (not differentiable)
        """
        logits, targets = ctx.saved_tensors

        batch_seq = targets.logical_volume()
        vocab_size = logits.shape[-1]

        estimate = cross_entropy_roofline(
            roofline_ctx.hw,
            batch_seq,
            vocab_size,
            dtype=logits.dtype,
            operation="CrossEntropyLoss.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        # Gradient has same shape as logits
        grad_logits = create_grad_tensor(
            logits.shape, logits.dtype, logits.layout, name="grad_logits"
        )

        return grad_logits, None
