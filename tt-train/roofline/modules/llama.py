# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Llama model module for roofline modeling.

This module provides MockLlama and MockLlamaConfig for roofline
estimation of the full Llama model.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations.rmsnorm import MockRMSNormOp
from .module import MockModule, MockParameter, MockModuleList
from .embedding import MockEmbedding
from .linear import MockLinearLayer
from .rmsnorm import MockRMSNormLayer
from .llama_block import MockLlamaBlock
from .grouped_query_attention import RoPEParams

if TYPE_CHECKING:
    from ..roofline import RooflineContext


@dataclass
class MockLlamaConfig:
    """Configuration for MockLlama model.

    Mirrors ttml.models.llama.LlamaConfig for roofline modeling.
    """

    vocab_size: int = 32000  # Default Llama vocab size
    max_sequence_length: int = 2048  # Maximum sequence length
    embedding_dim: int = 4096  # Embedding dimension
    intermediate_dim: Optional[int] = None  # MLP hidden dim (computed if None)
    num_heads: int = 32  # Number of attention heads
    num_groups: int = 8  # Number of KV groups (for GQA)
    dropout_prob: float = 0.0  # Dropout probability
    num_blocks: int = 32  # Number of transformer blocks
    theta: float = 10000.0  # RoPE theta parameter
    weight_tying: bool = False  # Whether to tie embedding and output weights


class MockLlama(MockModule):
    """Llama model module for roofline estimation.

    Mirrors ttml.models.llama.Llama for roofline modeling.

    This module implements a full Llama model with:
    1. Token embedding (possibly weight-tied with output)
    2. N transformer blocks with GQA and SwiGLU MLP
    3. Final RMS normalization
    4. Output projection to vocabulary

    Example:
        >>> config = MockLlamaConfig(num_blocks=16, embedding_dim=2048)
        >>> model = MockLlama(config)
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> indices = MockTensor((1, 1, 1, 1024), dtype=DataType.BFLOAT16)
        >>> logits = model(ctx, indices)
        >>> logits.backward(ctx)
        >>> print(ctx.summary(model))
    """

    def __init__(
        self,
        config: MockLlamaConfig,
        dtype: DataType = DataType.BFLOAT16,
    ):
        """Initialize Llama model.

        Args:
            config: Model configuration
            dtype: Data type for parameters
        """
        super().__init__()

        self.config = config
        self.dtype = dtype

        # Round vocab size up to be divisible by 32 (for efficient tiling)
        vocab_size_padded = (config.vocab_size + 31) // 32 * 32

        # Build RoPE params
        head_dim = config.embedding_dim // config.num_heads
        rope_params = RoPEParams(
            head_dim=head_dim,
            max_seq_len=config.max_sequence_length,
            theta=config.theta,
        )

        # Handle weight tying
        if config.weight_tying:
            # Output projection first (will be shared with embedding)
            self.fc = MockLinearLayer(
                config.embedding_dim, vocab_size_padded, has_bias=False, dtype=dtype
            )
            # Token embedding shares weight with fc
            # Note: In actual implementation, this uses tied weights
            # For roofline, we create separate parameters but account for shared memory
            self.tok_emb = MockEmbedding(
                vocab_size_padded, config.embedding_dim, dtype=dtype
            )
        else:
            # Token embedding
            self.tok_emb = MockEmbedding(
                vocab_size_padded, config.embedding_dim, dtype=dtype
            )
            # Output projection (no bias)
            self.fc = MockLinearLayer(
                config.embedding_dim, vocab_size_padded, has_bias=False, dtype=dtype
            )

        # Transformer blocks
        self.blocks = MockModuleList(
            [
                MockLlamaBlock(
                    embedding_size=config.embedding_dim,
                    num_heads=config.num_heads,
                    num_groups=config.num_groups,
                    rope_params=rope_params,
                    dropout=config.dropout_prob,
                    intermediate_dim=config.intermediate_dim,
                    dtype=dtype,
                )
                for _ in range(config.num_blocks)
            ]
        )

        # Final RMS normalization
        self.ln_fc = MockRMSNormLayer(config.embedding_dim, dtype=dtype)

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
        out = self.tok_emb(ctx, indices)

        # Transformer blocks
        for block in self.blocks:
            out = block(ctx, out, mask)

        # Final normalization
        out = self.ln_fc(ctx, out)

        # Output projection
        logits = self.fc(ctx, out)

        return logits

    def __repr__(self) -> str:
        return (
            f"MockLlama(\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  max_sequence_length={self.config.max_sequence_length},\n"
            f"  embedding_dim={self.config.embedding_dim},\n"
            f"  intermediate_dim={self.config.intermediate_dim},\n"
            f"  num_heads={self.config.num_heads},\n"
            f"  num_groups={self.config.num_groups},\n"
            f"  num_blocks={self.config.num_blocks},\n"
            f"  dropout={self.config.dropout_prob},\n"
            f"  weight_tying={self.config.weight_tying}\n"
            f")"
        )


def create_mock_llama(config: MockLlamaConfig) -> MockLlama:
    """Factory function to create a MockLlama model.

    Args:
        config: Configuration for the model

    Returns:
        A MockLlama model instance
    """
    return MockLlama(config)
