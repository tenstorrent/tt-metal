# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Dropout module for roofline modeling.

This module provides MockDropout for roofline estimation of
dropout operations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..operations import MockDropoutOp
from .module import MockModule

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockDropout(MockModule):
    """Dropout module for roofline estimation.

    Example:
        >>> dropout = MockDropout(0.1)
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> x = MockTensor((1, 1, 1024, 768))
        >>> output = dropout(ctx, x)
    """

    def __init__(self, dropout_prob: float = 0.1):
        """Initialize dropout module.

        Args:
            dropout_prob: Dropout probability
        """
        super().__init__()
        self.dropout_prob = dropout_prob

    def forward(self, ctx: "RooflineContext", x: MockTensor) -> MockTensor:
        """Forward pass: apply dropout.

        Args:
            ctx: Roofline context for estimates
            x: Input tensor

        Returns:
            Output tensor with dropout applied
        """
        if self.dropout_prob > 0:
            return MockDropoutOp.apply(ctx, x, self.dropout_prob)
        return x

    def __repr__(self) -> str:
        return f"MockDropout(p={self.dropout_prob})"
