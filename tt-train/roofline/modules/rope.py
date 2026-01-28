# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Rotary Position Embedding module for roofline modeling.

This module provides MockRotaryEmbedding for roofline estimation of
rotary position embedding operations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import MockRoPEOp
from .module import MockModule

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockRotaryEmbedding(MockModule):
    """Rotary Position Embedding module for roofline estimation.

    Mirrors ttml rotary embedding for roofline modeling.
    RoPE applies rotation based on position to Q and K tensors in attention.

    Note: This module does not have trainable parameters. The sin/cos caches
    are precomputed based on position and not learned.

    Example:
        >>> rope = MockRotaryEmbedding(head_dim=64, max_seq_len=2048)
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> q = MockTensor((1, 32, 1024, 64))
        >>> q_rotated = rope(ctx, q)
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        dtype: DataType = DataType.BFLOAT16,
    ):
        """Initialize rotary position embedding.

        Args:
            head_dim: Dimension per attention head
            max_seq_len: Maximum sequence length for position caching
            theta: Base for frequency computation (default: 10000.0)
            dtype: Data type for caches
        """
        super().__init__()

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.dtype = dtype

        # Note: No trainable parameters - sin/cos caches are computed
        # at initialization time in actual implementation

    def forward(
        self,
        ctx: "RooflineContext",
        x: MockTensor,
        token_position: int = 0,
    ) -> MockTensor:
        """Forward pass: apply rotary position embedding.

        Args:
            ctx: Roofline context for estimates
            x: Input tensor [B, num_heads, S, head_dim]
            token_position: Starting token position (for KV caching)

        Returns:
            Tensor with rotary embedding applied
        """
        return MockRoPEOp.apply(ctx, x, self.head_dim, token_position)

    def __repr__(self) -> str:
        return (
            f"MockRotaryEmbedding(head_dim={self.head_dim}, "
            f"max_seq_len={self.max_seq_len}, theta={self.theta})"
        )
