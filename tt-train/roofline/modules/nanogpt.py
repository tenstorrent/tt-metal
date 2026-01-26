# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""NanoGPT model module for roofline modeling.

This module provides MockNanoGPT and MockNanoGPTConfig for roofline
estimation of the full NanoGPT model.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import MockLayerNormOp
from .module import MockModule, MockParameter, MockModuleList
from .embedding import MockEmbedding, MockTrainablePositionalEmbedding
from .linear import MockLinearLayer
from .gpt_block import MockGPTBlock

if TYPE_CHECKING:
    from ..roofline import RooflineContext


@dataclass
class MockNanoGPTConfig:
    """Configuration for MockNanoGPT model.

    Mirrors ttml.models.nanogpt.NanoGPTConfig for roofline modeling.
    """

    vocab_size: int = 50304  # GPT-2 vocab size, padded to be divisible by 64
    block_size: int = 1024  # Maximum sequence length
    n_embd: int = 768  # Embedding dimension
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12  # Number of attention heads
    dropout: float = 0.2  # Dropout probability
    bias: bool = True  # Use bias in linear layers and layer norm


class MockNanoGPT(MockModule):
    """NanoGPT model module for roofline estimation.

    Mirrors ttml.models.nanogpt.NanoGPT for roofline modeling.

    This module implements a full GPT model with:
    1. Token embedding
    2. Positional embedding (with dropout)
    3. N transformer blocks
    4. Final layer norm
    5. Output projection to vocabulary

    Example:
        >>> config = MockNanoGPTConfig(n_layer=12, n_embd=768, n_head=12)
        >>> model = MockNanoGPT(config)
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> indices = MockTensor((1, 1, 1, 1024), dtype=DataType.INT32)
        >>> logits = model(ctx, indices)
        >>> logits.backward(ctx)
        >>> print(ctx.summary(model))
    """

    def __init__(
        self,
        config: MockNanoGPTConfig,
        dtype: DataType = DataType.BFLOAT16,
    ):
        """Initialize NanoGPT model.

        Args:
            config: Model configuration
            dtype: Data type for parameters
        """
        super().__init__()

        self.config = config
        self.dtype = dtype

        # Round vocab size up to be divisible by 32 (for efficient tiling)
        vocab_size_divisible_by_32 = (config.vocab_size + 31) // 32 * 32

        # Token embedding
        self.tok_emb = MockEmbedding(
            vocab_size_divisible_by_32, config.n_embd, dtype=dtype
        )

        # Positional embedding with dropout
        self.pos_emb = MockTrainablePositionalEmbedding(
            config.block_size, config.n_embd, dropout=config.dropout, dtype=dtype
        )

        # Transformer blocks
        self.blocks = MockModuleList(
            [
                MockGPTBlock(
                    config.n_embd,
                    config.n_head,
                    dropout=config.dropout,
                    bias=config.bias,
                    dtype=dtype,
                )
                for _ in range(config.n_layer)
            ]
        )

        # Final layer norm - using direct parameters
        self.ln_f_gamma = MockParameter(
            MockTensor((1, 1, 1, config.n_embd), dtype=dtype)
        )
        if config.bias:
            self.ln_f_beta = MockParameter(
                MockTensor((1, 1, 1, config.n_embd), dtype=dtype)
            )
        else:
            self.ln_f_beta = None

        # Output projection to vocabulary (no bias)
        self.fc = MockLinearLayer(
            config.n_embd, config.vocab_size, has_bias=False, dtype=dtype
        )

    def forward(
        self,
        ctx: "RooflineContext",
        indices: MockTensor,
        mask: Optional[MockTensor] = None,
    ) -> MockTensor:
        """Forward pass: compute logits from token indices.

        Args:
            ctx: Roofline context for estimates
            indices: Token indices [batch, 1, 1, seq_len]
            mask: Optional attention mask [1, 1, seq_len, seq_len]

        Returns:
            Logits tensor [batch, 1, seq_len, vocab_size]
        """
        # Token embedding
        tok_emb_out = self.tok_emb(ctx, indices)

        # Add positional embeddings
        out = self.pos_emb(ctx, tok_emb_out)

        # Transformer blocks
        for block in self.blocks:
            out = block(ctx, out, mask)

        # Final layer norm
        beta_tensor = self.ln_f_beta.tensor if self.ln_f_beta is not None else None
        out = MockLayerNormOp.apply(ctx, out, self.ln_f_gamma.tensor, beta_tensor)

        # Output projection
        logits = self.fc(ctx, out)

        return logits

    def __repr__(self) -> str:
        return (
            f"MockNanoGPT(\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  block_size={self.config.block_size},\n"
            f"  n_embd={self.config.n_embd},\n"
            f"  n_layer={self.config.n_layer},\n"
            f"  n_head={self.config.n_head},\n"
            f"  dropout={self.config.dropout}\n"
            f")"
        )


def create_mock_nanogpt(config: MockNanoGPTConfig) -> MockNanoGPT:
    """Factory function to create a MockNanoGPT model.

    Args:
        config: Configuration for the model

    Returns:
        A MockNanoGPT model instance
    """
    return MockNanoGPT(config)
