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
    fused_attention_roofline,
    heads_creation_roofline,
    heads_fusion_roofline,
    grouped_heads_creation_roofline,
)
from .operation import (
    RooflineFunctionContext,
    RooflineFunction,
    create_grad_tensor,
    create_activation_tensor,
)

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
        query = create_activation_tensor(output_shape, qkv.dtype, qkv.layout)
        key = create_activation_tensor(output_shape, qkv.dtype, qkv.layout)
        value = create_activation_tensor(output_shape, qkv.dtype, qkv.layout)

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

        grad_qkv = create_grad_tensor(qkv.shape, qkv.dtype, qkv.layout, name="grad_qkv")

        return (grad_qkv,)


class MockGroupedHeadsCreationOp(RooflineFunction):
    """Roofline estimation for grouped heads creation (split Q and KV for GQA).

    For Grouped Query Attention:
    - Q: [B, 1, S, E] -> [B, num_heads, S, head_dim]
    - KV: [B, 1, S, 2*num_groups*head_dim] -> K [B, num_groups, S, head_dim], V [B, num_groups, S, head_dim]
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        q: MockTensor,
        kv: MockTensor,
        num_heads: int,
        num_groups: int,
    ) -> Tuple[MockTensor, MockTensor, MockTensor]:
        """Forward pass: split Q and KV into separate head tensors.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            q: Query tensor [B, 1, S, E] where E = num_heads * head_dim
            kv: Combined KV tensor [B, 1, S, 2*num_groups*head_dim]
            num_heads: Number of attention heads for Q
            num_groups: Number of KV groups

        Returns:
            Tuple of (query, key, value) tensors
            - query: [B, num_heads, S, head_dim]
            - key: [B, num_groups, S, head_dim]
            - value: [B, num_groups, S, head_dim]
        """
        ctx.save_for_backward(q, kv, num_heads, num_groups)

        # Parse dimensions from q shape [B, 1, S, E]
        batch_size = q.shape[0]
        seq_len = q.shape[2]
        embedding_dim = q.shape[3]
        head_dim = embedding_dim // num_heads

        estimate = grouped_heads_creation_roofline(
            roofline_ctx.hw,
            batch_size,
            seq_len,
            num_heads,
            num_groups,
            head_dim,
            dtype=q.dtype,
            operation="GroupedHeadsCreation.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        # Output shapes
        q_shape = (batch_size, num_heads, seq_len, head_dim)
        kv_shape = (batch_size, num_groups, seq_len, head_dim)

        query = create_activation_tensor(q_shape, q.dtype, q.layout)
        key = create_activation_tensor(kv_shape, kv.dtype, kv.layout)
        value = create_activation_tensor(kv_shape, kv.dtype, kv.layout)

        return query, key, value

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_query: MockTensor,
        grad_key: MockTensor,
        grad_value: MockTensor,
    ) -> Tuple[MockTensor, MockTensor, None, None]:
        """Backward pass: concatenate gradients back.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_query: Gradient for query
            grad_key: Gradient for key
            grad_value: Gradient for value

        Returns:
            (grad_q, grad_kv, None, None) tuple
        """
        q, kv, num_heads, num_groups = ctx.saved_tensors

        batch_size = q.shape[0]
        seq_len = q.shape[2]
        embedding_dim = q.shape[3]
        head_dim = embedding_dim // num_heads

        estimate = grouped_heads_creation_roofline(
            roofline_ctx.hw,
            batch_size,
            seq_len,
            num_heads,
            num_groups,
            head_dim,
            dtype=q.dtype,
            operation="GroupedHeadsCreation.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        grad_q = create_grad_tensor(q.shape, q.dtype, q.layout, name="grad_q")
        grad_kv = create_grad_tensor(kv.shape, kv.dtype, kv.layout, name="grad_kv")

        return grad_q, grad_kv, None, None


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
        return create_activation_tensor(
            output_shape, attention_out.dtype, attention_out.layout
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

        grad_attention_out = create_grad_tensor(
            attention_out.shape,
            attention_out.dtype,
            attention_out.layout,
            name="grad_attention_out",
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
            key: Key tensor [B, H, S, d] or [B, G, S, d] for GQA
            value: Value tensor [B, H, S, d] or [B, G, S, d] for GQA
            mask: Optional attention mask

        Returns:
            Attention output tensor [B, H, S, d]
        """
        # Parse dimensions [B, H, S, d]
        batch_size = query.shape[0]
        num_heads = query.shape[1]
        seq_len = query.shape[2]
        head_dim = query.shape[3]

        # Create attention_weights tensor (B, H, S, S) - this is the softmax output
        # that must be saved for backward pass (see scaled_dot_product_attention.cpp)
        # This is a significant memory allocation, especially for long sequences!
        attention_weights_shape = (batch_size, num_heads, seq_len, seq_len)
        attention_weights = create_activation_tensor(
            attention_weights_shape,
            query.dtype,
            query.layout,
            name="attention_weights",
        )

        # Save tensors for backward - must include attention_weights
        ctx.save_for_backward(query, key, value, mask, attention_weights)

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
        return create_activation_tensor(query.shape, query.dtype, query.layout)

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
            (None for mask which doesn't need gradients)

        Note:
            attention_weights is saved for backward but is not an input,
            so we don't return a gradient for it.
        """
        query, key, value, mask, attention_weights = ctx.saved_tensors

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

        grad_query = create_grad_tensor(
            query.shape, query.dtype, query.layout, name="grad_query"
        )
        grad_key = create_grad_tensor(key.shape, key.dtype, key.layout, name="grad_key")
        grad_value = create_grad_tensor(
            value.shape, value.dtype, value.layout, name="grad_value"
        )

        # No gradient for mask (it's optional and doesn't require grad)
        return grad_query, grad_key, grad_value, None


class MockScaledDotProductAttentionFusedOp(RooflineFunction):
    """Roofline estimation for fused scaled dot-product attention (Flash Attention style).

    Fused SDPA: softmax(Q @ K^T / sqrt(d_k)) @ V computed in a single kernel.

    Key benefits over unfused SDPA:
    1. Memory: O(S) instead of O(S²) - no need to materialize full attention matrix
    2. Speed: Single fused kernel with better memory access patterns
    3. Backward: Uses recomputation instead of storing attention weights

    This models hardware-optimized attention like Flash Attention, where the
    attention matrix is computed and consumed in tiles without full materialization.
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
        """Forward pass: compute fused scaled dot-product attention.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            query: Query tensor [B, H, S, d]
            key: Key tensor [B, H, S, d] or [B, G, S, d] for GQA
            value: Value tensor [B, H, S, d] or [B, G, S, d] for GQA
            mask: Optional attention mask (not used in fused kernel)

        Returns:
            Attention output tensor [B, H, S, d]
        """
        # Parse dimensions [B, H, S, d]
        batch_size = query.shape[0]
        num_heads = query.shape[1]
        seq_len = query.shape[2]
        head_dim = query.shape[3]

        #   - max values per attention row: [B, H, S]
        #   - sum(exp-max) per row: [B, H, S]
        max_values = create_activation_tensor(
            (batch_size, num_heads, seq_len),
            query.dtype,
            query.layout,
            name="attention_max_values",
        )
        sum_values = create_activation_tensor(
            (batch_size, num_heads, seq_len),
            query.dtype,
            query.layout,
            name="attention_sum_values",
        )

        # Save for backward: inputs + intermediate values
        # Backward will recompute attention on-the-fly using these saved values
        ctx.save_for_backward(query, key, value, mask, max_values, sum_values)

        estimate = fused_attention_roofline(
            roofline_ctx.hw,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            dtype=query.dtype,
            operation="FusedSDPA",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        # Output shape same as input: [B, H, S, d]
        return create_activation_tensor(query.shape, query.dtype, query.layout)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, MockTensor, MockTensor, None, None, None]:
        """Backward pass: compute gradients for Q, K, V using recomputation.

        Fused backward recomputes attention weights on-the-fly instead of
        reading them from memory. This trades compute for memory bandwidth.
        Uses saved max and sum values to efficiently recompute softmax.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor [B, H, S, d]

        Returns:
            (grad_query, grad_key, grad_value, None, None, None) tuple
            (None for mask, max_values, and sum_values which don't need gradients)
        """
        query, key, value, mask, max_values, sum_values = ctx.saved_tensors

        batch_size = query.shape[0]
        num_heads = query.shape[1]
        seq_len = query.shape[2]
        head_dim = query.shape[3]

        estimate = fused_attention_roofline(
            roofline_ctx.hw,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            dtype=query.dtype,
            operation="FusedSDPA",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        grad_query = create_grad_tensor(
            query.shape, query.dtype, query.layout, name="grad_query"
        )
        grad_key = create_grad_tensor(key.shape, key.dtype, key.layout, name="grad_key")
        grad_value = create_grad_tensor(
            value.shape, value.dtype, value.layout, name="grad_value"
        )

        # No gradients for mask, max_values, or sum_values (intermediate values)
        return grad_query, grad_key, grad_value, None, None, None
