# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Embedding operation for roofline modeling.

This module provides MockEmbeddingOp for roofline estimation of
embedding lookup operations.
"""

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..roofline import embedding_roofline
from .operation import RooflineFunctionContext, RooflineFunction

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockEmbeddingOp(RooflineFunction):
    """Roofline estimation for embedding lookup.

    Forward: output = weight[indices]
    Backward: grad_weight = scatter_add(grad_output, indices)
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        indices: MockTensor,
        weight: MockTensor,
    ) -> MockTensor:
        """Forward pass: gather embeddings for indices.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            indices: Token indices tensor [batch, 1, 1, seq_len]
            weight: Embedding weight tensor [1, 1, vocab_size, embedding_dim]

        Returns:
            Output tensor [batch, 1, seq_len, embedding_dim]
        """
        ctx.save_for_backward(indices, weight)

        # Get dimensions
        batch_seq = indices.logical_volume()
        vocab_size = weight.shape[2]
        embedding_dim = weight.shape[3]

        estimate = embedding_roofline(
            roofline_ctx.hw,
            batch_seq,
            embedding_dim,
            vocab_size,
            dtype=weight.dtype,
            operation="Embedding.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        # Output shape: indices shape with last dim replaced by embedding_dim
        # [batch, 1, 1, seq_len] -> [batch, 1, seq_len, embedding_dim]
        output_shape = indices.shape[:-2] + (indices.shape[-1], embedding_dim)
        return MockTensor(output_shape, weight.dtype, weight.layout, requires_grad=True)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[None, MockTensor]:
        """Backward pass: scatter-add gradients to weight.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor [batch, 1, seq_len, embedding_dim]

        Returns:
            (None, grad_weight) tuple - None for indices (not differentiable)
        """
        indices, weight = ctx.saved_tensors

        batch_seq = indices.logical_volume()
        vocab_size = weight.shape[2]
        embedding_dim = weight.shape[3]

        estimate = embedding_roofline(
            roofline_ctx.hw,
            batch_seq,
            embedding_dim,
            vocab_size,
            dtype=weight.dtype,
            operation="Embedding.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        # Gradient has same shape as weight
        grad_weight = MockTensor(
            weight.shape, weight.dtype, weight.layout, requires_grad=False
        )

        return None, grad_weight
