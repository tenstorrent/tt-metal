# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""NanoGPT model package.

This package provides a Python implementation of NanoGPT (a small GPT model)
using ttml's C++ operations for computation.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import ml_dtypes

import ttml
from ttml.modules import AbstractModuleBase, Parameter, RunMode

from .embedding import Embedding
from .gpt_block import GPTBlock


@dataclass
class NanoGPTConfig:
    """Configuration for NanoGPT model."""

    vocab_size: int = 50304  # GPT-2 vocab size
    block_size: int = 1024  # Maximum sequence length
    n_embd: int = 768  # Embedding dimension
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12  # Number of attention heads
    dropout: float = (
        0.2  # Dropout probability (matching C++ default: dropout_prob = 0.2F)
    )
    bias: bool = True  # Use bias in linear layers and layer norm


class NanoGPT(AbstractModuleBase):
    """NanoGPT model implemented in Python using ttml operations."""

    def __init__(self, config: NanoGPTConfig) -> None:
        """Initialize NanoGPT model.

        Args:
            config: Configuration for the model
        """
        super().__init__()

        self.config = config
        # Note: RunMode is managed by AbstractModuleBase (defaults to TRAIN)
        # Use get_run_mode() to check, train()/eval() to set

        # Token and position embeddings
        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.wpe = Embedding(config.block_size, config.n_embd)

        # Transformer blocks
        self.blocks = []
        for i in range(config.n_layer):
            block = GPTBlock(config.n_embd, config.n_head, config.dropout, config.bias)
            self.blocks.append(block)
            # Register as submodule for parameter tracking
            setattr(self, f"block_{i}", block)

        # Final layer norm (use ml_dtypes.bfloat16)
        # Layer norm parameters must be in TILE layout
        ln_f_shape = (1, 1, 1, config.n_embd)
        gamma_f_np = np.ones(ln_f_shape, dtype=ml_dtypes.bfloat16)
        gamma_f_tensor = ttml.autograd.Tensor.from_numpy(
            gamma_f_np, layout=ttml.Layout.TILE
        )
        self.ln_f_gamma = Parameter(gamma_f_tensor)

        if config.bias:
            beta_f_np = np.zeros(ln_f_shape, dtype=ml_dtypes.bfloat16)
            beta_f_tensor = ttml.autograd.Tensor.from_numpy(
                beta_f_np, layout=ttml.Layout.TILE
            )
            self.ln_f_beta = Parameter(beta_f_tensor)
        else:
            self.ln_f_beta = None

        # Language model head (output projection)
        # Note: Weight tying with token embeddings will be handled in forward
        lm_head_shape = (1, 1, config.vocab_size, config.n_embd)
        # We'll use the same weight as wte for weight tying
        self.lm_head_weight = self.wte.weight

    # train() and eval() are inherited from AbstractModuleBase
    # They automatically propagate RunMode to all registered submodules

    def forward(
        self, idx: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None
    ) -> ttml.autograd.Tensor:
        """Forward pass of NanoGPT.

        Args:
            idx: Token indices, shape [batch_size, 1, 1, seq_len]
            mask: Optional causal attention mask, shape [1, 1, seq_len, seq_len]

        Returns:
            Logits tensor, shape [batch_size, 1, 1, seq_len, vocab_size]
        """
        # Token and position embeddings
        tok_emb = self.wte(idx)

        # Create position indices
        # Get sequence length from input
        idx_np = idx.to_numpy(ttml.autograd.DataType.UINT32)
        seq_len = idx_np.shape[-1]
        pos_np = np.arange(seq_len, dtype=np.uint32).reshape(1, 1, 1, seq_len)
        pos = ttml.autograd.Tensor.from_numpy(
            pos_np, layout=ttml.Layout.ROW_MAJOR, new_type=ttml.autograd.DataType.UINT32
        )
        pos_emb = self.wpe(pos)

        # Add embeddings
        x = ttml.ops.binary.add(tok_emb, pos_emb)

        # Apply dropout if in training mode (using RunMode from AbstractModuleBase)
        if self.get_run_mode() == RunMode.TRAIN and self.config.dropout > 0.0:
            x = ttml.ops.dropout.dropout(x, self.config.dropout)

        # Pass through transformer blocks with mask
        for block in self.blocks:
            x = block(x, mask=mask)

        # Final layer norm (use composite to avoid custom kernel include path issues)
        x = ttml.ops.layernorm.composite_layernorm(
            x, self.ln_f_gamma.tensor, self.ln_f_beta.tensor if self.ln_f_beta else None
        )

        # Language model head (weight tying: uses same weights as token embedding)
        logits = ttml.ops.linear.linear_op(x, self.lm_head_weight.tensor, None)

        # Reshape logits from [B, 1, 1, seq_len, vocab_size] to [B, 1, seq_len, vocab_size]
        # Use ttml's reshape operation which preserves the computation graph
        logits_shape = logits.shape()
        if len(logits_shape) == 5:
            # [B, 1, 1, seq_len, vocab_size] -> [B, 1, seq_len, vocab_size]
            new_shape = [logits_shape[0], 1, logits_shape[3], logits_shape[4]]
            logits = ttml.ops.reshape.reshape(logits, new_shape)

        return logits

    def __call__(
        self, idx: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None
    ) -> ttml.autograd.Tensor:
        """Call the forward method."""
        return self.forward(idx, mask)


def create_nanogpt(config: NanoGPTConfig) -> NanoGPT:
    """Factory function to create a NanoGPT model.

    Args:
        config: Configuration for the model

    Returns:
        A NanoGPT model instance
    """
    return NanoGPT(config)


__all__ = [
    "Embedding",
    "GPTBlock",
    "NanoGPT",
    "NanoGPTConfig",
    "create_nanogpt",
]
