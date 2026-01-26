# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Attention operations for roofline modeling.

This module provides MockHeadsCreationOp, MockHeadsFusionOp, and
MockScaledDotProductAttentionOp for roofline estimation of attention operations.
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..roofline import (
    attention_roofline,
    heads_creation_roofline,
    heads_fusion_roofline,
)
from .operation import RooflineFunctionContext, RooflineFunction

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockHeadsCreationOp(RooflineFunction):
    """Roofline estimation for heads creation (split QKV).

    Splits combined QKV tensor into separate Q, K, V tensors with head dimension.
    Input: [B, 1, S, 3*H*d] -> Output: 3x [B, H, S, d]
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        qkv: MockTensor,
        num_heads: int,
    ) -> Tuple[MockTensor, MockTensor, MockTensor]:
        """Forward pass: split QKV into separate tensors.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            qkv: Combined QKV tensor [B, 1, S, 3*H*d]
            num_heads: Number of attention heads

        Returns:
            Tuple of (query, key, value) tensors, each [B, H, S, d]
        """
        ctx.save_for_backward(qkv, num_heads)

        # Parse dimensions from qkv shape [B, 1, S, 3*H*d]
        batch_size = qkv.shape[0]
        seq_len = qkv.shape[2]
        total_dim = qkv.shape[3]
        head_dim = total_dim // (3 * num_heads)

        estimate = heads_creation_roofline(
            roofline_ctx.hw,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            num_tensors=3,
            dtype=qkv.dtype,
            operation="HeadsCreation.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        # Output shapes: [B, H, S, d]
        output_shape = (batch_size, num_heads, seq_len, head_dim)
        query = MockTensor(output_shape, qkv.dtype, qkv.layout, requires_grad=True)
        key = MockTensor(output_shape, qkv.dtype, qkv.layout, requires_grad=True)
        value = MockTensor(output_shape, qkv.dtype, qkv.layout, requires_grad=True)

        return query, key, value

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_query: MockTensor,
        grad_key: MockTensor,
        grad_value: MockTensor,
    ) -> Tuple[MockTensor]:
        """Backward pass: concatenate gradients back.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_query: Gradient for query
            grad_key: Gradient for key
            grad_value: Gradient for value

        Returns:
            (grad_qkv,) tuple
        """
        qkv, num_heads = ctx.saved_tensors

        batch_size = qkv.shape[0]
        seq_len = qkv.shape[2]
        total_dim = qkv.shape[3]
        head_dim = total_dim // (3 * num_heads)

        estimate = heads_creation_roofline(
            roofline_ctx.hw,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            num_tensors=3,
            dtype=qkv.dtype,
            operation="HeadsCreation.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        grad_qkv = MockTensor(qkv.shape, qkv.dtype, qkv.layout, requires_grad=False)

        return (grad_qkv,)


class MockHeadsFusionOp(RooflineFunction):
    """Roofline estimation for heads fusion (merge heads back).

    Merges attention output heads back into single tensor.
    Input: [B, H, S, d] -> Output: [B, 1, S, H*d]
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        attention_out: MockTensor,
    ) -> MockTensor:
        """Forward pass: merge heads back.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            attention_out: Attention output [B, H, S, d]

        Returns:
            Merged tensor [B, 1, S, H*d]
        """
        ctx.save_for_backward(attention_out)

        # Parse dimensions [B, H, S, d]
        batch_size = attention_out.shape[0]
        num_heads = attention_out.shape[1]
        seq_len = attention_out.shape[2]
        head_dim = attention_out.shape[3]

        estimate = heads_fusion_roofline(
            roofline_ctx.hw,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            dtype=attention_out.dtype,
            operation="HeadsFusion.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        # Output shape: [B, 1, S, H*d]
        output_shape = (batch_size, 1, seq_len, num_heads * head_dim)
        return MockTensor(
            output_shape, attention_out.dtype, attention_out.layout, requires_grad=True
        )

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor]:
        """Backward pass: split gradient back into heads.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor [B, 1, S, H*d]

        Returns:
            (grad_attention_out,) tuple with shape [B, H, S, d]
        """
        (attention_out,) = ctx.saved_tensors

        batch_size = attention_out.shape[0]
        num_heads = attention_out.shape[1]
        seq_len = attention_out.shape[2]
        head_dim = attention_out.shape[3]

        estimate = heads_fusion_roofline(
            roofline_ctx.hw,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            dtype=attention_out.dtype,
            operation="HeadsFusion.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        grad_attention_out = MockTensor(
            attention_out.shape,
            attention_out.dtype,
            attention_out.layout,
            requires_grad=False,
        )

        return (grad_attention_out,)


class MockScaledDotProductAttentionOp(RooflineFunction):
    """Roofline estimation for scaled dot-product attention.

    SDPA: softmax(Q @ K^T / sqrt(d_k)) @ V

    This operation combines:
    1. Q @ K^T matmul
    2. Scale by 1/sqrt(d_k)
    3. Softmax
    4. Attn @ V matmul
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        query: MockTensor,
        key: MockTensor,
        value: MockTensor,
        mask: Optional[MockTensor] = None,
    ) -> MockTensor:
        """Forward pass: compute scaled dot-product attention.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            query: Query tensor [B, H, S, d]
            key: Key tensor [B, H, S, d]
            value: Value tensor [B, H, S, d]
            mask: Optional attention mask

        Returns:
            Attention output tensor [B, H, S, d]
        """
        ctx.save_for_backward(query, key, value, mask)

        # Parse dimensions [B, H, S, d]
        batch_size = query.shape[0]
        num_heads = query.shape[1]
        seq_len = query.shape[2]
        head_dim = query.shape[3]

        estimates = attention_roofline(
            roofline_ctx.hw,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            dtype=query.dtype,
            operation="SDPA",
            phase="forward",
        )
        for estimate in estimates:
            roofline_ctx.add_perf_result(estimate)

        # Output shape same as input: [B, H, S, d]
        return MockTensor(query.shape, query.dtype, query.layout, requires_grad=True)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, MockTensor, MockTensor, None]:
        """Backward pass: compute gradients for Q, K, V.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor [B, H, S, d]

        Returns:
            (grad_query, grad_key, grad_value, None) tuple
        """
        query, key, value, mask = ctx.saved_tensors

        batch_size = query.shape[0]
        num_heads = query.shape[1]
        seq_len = query.shape[2]
        head_dim = query.shape[3]

        estimates = attention_roofline(
            roofline_ctx.hw,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            dtype=query.dtype,
            operation="SDPA",
            phase="backward",
        )
        for estimate in estimates:
            roofline_ctx.add_perf_result(estimate)

        grad_query = MockTensor(
            query.shape, query.dtype, query.layout, requires_grad=False
        )
        grad_key = MockTensor(key.shape, key.dtype, key.layout, requires_grad=False)
        grad_value = MockTensor(
            value.shape, value.dtype, value.layout, requires_grad=False
        )

        return grad_query, grad_key, grad_value, None
