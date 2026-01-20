# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""NanoGPT model package.

This package provides a Python implementation of NanoGPT (a small GPT model)
using ttml's C++ operations for computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import ml_dtypes

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, Parameter, ModuleList, RunMode, LinearLayer

from .embedding import Embedding
from .gpt_block import GPTBlock


def initialize_weights_gpt2(parameters: Dict[str, ttml.autograd.Tensor]) -> None:
    """Initialize weights matching C++ initialize_weights_gpt2.

    This function re-initializes all model parameters to match the C++ gpt2 model:
    - All "weight" parameters: normal(mean=0, stddev=0.02)
    - All "bias" parameters: constant 0
    - "gamma" and "beta" (LayerNorm) are NOT touched (they keep their initial values)

    This is called after model construction to ensure weights match the C++ initialization.

    Args:
        parameters: Dictionary of parameter name to tensor
    """
    for name, tensor in parameters.items():
        # Get current shape from tensor
        shape = tensor.shape()

        if "weight" in name:
            # Re-initialize weights with normal(0, 0.02)
            weight_np = np.random.normal(0.0, 0.02, size=shape).astype(
                ml_dtypes.bfloat16
            )
            new_tensor = ttml.autograd.Tensor.from_numpy(
                weight_np, layout=ttnn.Layout.TILE
            )
            tensor.assign(new_tensor)
        elif "bias" in name:
            # Re-initialize biases with 0
            bias_np = np.zeros(shape, dtype=ml_dtypes.bfloat16)
            new_tensor = ttml.autograd.Tensor.from_numpy(
                bias_np, layout=ttnn.Layout.TILE
            )
            tensor.assign(new_tensor)


@dataclass
class NanoGPTConfig:
    """Configuration for NanoGPT model.

    Note: The following C++ features are not yet implemented in Python:
    - TODO: runner_type (Default/MemoryEfficient) - for memory efficient block execution
    - TODO: weight_tying (Enabled/Disabled) - ties tok_emb and fc weights
    - TODO: positional_embedding_type (Trainable/Fixed) - only Trainable is implemented
    - TODO: experimental.use_composite_layernorm - use composite vs fused layernorm
    """

    vocab_size: int = 50304  # GPT-2 vocab size
    block_size: int = 1024  # Maximum sequence length
    n_embd: int = 768  # Embedding dimension
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12  # Number of attention heads
    dropout: float = (
        0.2  # Dropout probability (matching C++ default: dropout_prob = 0.2F)
    )
    bias: bool = True  # Use bias in linear layers and layer norm


class TrainablePositionalEmbedding(AbstractModuleBase):
    """Trainable positional embedding matching C++ TrainablePositionalEmbedding."""

    def __init__(
        self, sequence_length: int, embedding_dim: int, dropout_prob: float = 0.0
    ) -> None:
        """Initialize trainable positional embedding.

        Args:
            sequence_length: Maximum sequence length
            embedding_dim: Dimension of embeddings
            dropout_prob: Dropout probability
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.dropout_prob = dropout_prob

        weight_shape = (1, 1, sequence_length, embedding_dim)
        weight_np = np.random.normal(0.0, 0.02, size=weight_shape).astype(
            ml_dtypes.bfloat16
        )
        weight_tensor = ttml.autograd.Tensor.from_numpy(
            weight_np, layout=ttnn.Layout.TILE
        )
        self.weight = Parameter(weight_tensor)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass: add positional embeddings and apply dropout.

        Args:
            x: Input tensor (token embeddings), shape [batch, 1, seq_len, embedding_dim]

        Returns:
            Output tensor with positional embeddings added
        """
        # Simply add the positional weight tensor (matching C++ ops::add(input, m_weight))
        if len(x.shape()) != 4:
            raise ValueError(
                f"TrainablePositionalEmbedding: input tensor must have 4 dimensions. Got rank {len(x.shape())}"
            )
        if x.shape()[2] != self.sequence_length:
            raise ValueError(
                f"TrainablePositionalEmbedding: input tensor sequence length ({x.shape()[2]}) does not match the expected value ({self.sequence_length})"
            )
        out = ttml.ops.binary.add(x, self.weight.tensor)
        # Note: It's better to just use Dropout module here
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            out = ttml.ops.dropout.dropout(out, self.dropout_prob)
        return out

    def __call__(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Call the forward method."""
        return self.forward(x)


class NanoGPT(AbstractModuleBase):
    """NanoGPT model implemented in Python using ttml operations.

    This implementation matches the C++ ttml::models::gpt2::Transformer class.
    See tt-train/sources/ttml/models/gpt2.cpp for reference.
    """

    def __init__(self, config: NanoGPTConfig) -> None:
        """Initialize NanoGPT model.

        Args:
            config: Configuration for the model
        """
        super().__init__()

        self.config = config
        # Note: RunMode is managed by AbstractModuleBase (defaults to TRAIN)
        # Use get_run_mode() to check, train()/eval() to set

        # TODO: Implement weight_tying - when enabled, tok_emb shares weights with fc
        # C++ creates fc first, then passes fc->get_weight() to Embedding constructor
        self.fc = LinearLayer(
            config.n_embd, config.vocab_size, False
        )  # False - no bias
        vocab_size_divisible_by_32 = (config.vocab_size + 31) // 32 * 32
        self.tok_emb = Embedding(vocab_size_divisible_by_32, config.n_embd)
        self.pos_emb = TrainablePositionalEmbedding(
            config.block_size, config.n_embd, config.dropout
        )

        # Transformer blocks (ModuleList auto-registers all blocks)
        self.blocks = ModuleList(
            [
                GPTBlock(config.n_embd, config.n_head, config.dropout, config.bias)
                for _ in range(config.n_layer)
            ]
        )

        # Final layer norm (use ml_dtypes.bfloat16)
        # Layer norm parameters must be in TILE layout
        ln_f_shape = (1, 1, 1, config.n_embd)
        gamma_f_np = np.ones(ln_f_shape, dtype=ml_dtypes.bfloat16)
        gamma_f_tensor = ttml.autograd.Tensor.from_numpy(
            gamma_f_np, layout=ttnn.Layout.TILE
        )
        self.ln_f_gamma = Parameter(gamma_f_tensor)

        if config.bias:
            beta_f_np = np.zeros(ln_f_shape, dtype=ml_dtypes.bfloat16)
            beta_f_tensor = ttml.autograd.Tensor.from_numpy(
                beta_f_np, layout=ttnn.Layout.TILE
            )
            self.ln_f_beta = Parameter(beta_f_tensor)
        else:
            self.ln_f_beta = None

        # Initialize weights
        # This re-initializes all "weight" to normal(0, 0.02) and all "bias" to 0
        initialize_weights_gpt2(self.parameters())

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
            Logits tensor, shape [batch_size, 1, seq_len, vocab_size]
        """
        # Token embedding (matching C++ tok_emb_out = (*tok_emb)(x))
        tok_emb_out = self.tok_emb(idx)
        out = self.pos_emb(tok_emb_out)

        # TODO: Implement runner_type for memory efficient execution
        # C++ supports RunnerType::MemoryEfficient which uses memory_efficient_runner()
        for block in self.blocks:
            out = block(out, mask=mask)

        out = ttml.ops.layernorm.layernorm(
            out,
            self.ln_f_gamma.tensor,
            self.ln_f_beta.tensor if self.ln_f_beta else None,
        )

        # Output projection (matching C++ logits = (*fc)(out))
        logits = self.fc(out)

        return logits


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
    "initialize_weights_gpt2",
]
