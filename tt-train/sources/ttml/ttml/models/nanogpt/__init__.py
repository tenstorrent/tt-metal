# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""NanoGPT model package.

This package provides a Python implementation of NanoGPT (a small GPT model)
using ttml's C++ operations for computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Literal

import numpy as np
import ml_dtypes

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, Parameter, ModuleList, RunMode, LinearLayer

from .. import RunnerType, WeightTyingType, memory_efficient_runner
from .embedding import Embedding
from .pos_embedding import PositionalEmbedding, TrainablePositionalEmbedding
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
            weight_np = np.random.normal(0.0, 0.02, size=shape).astype(ml_dtypes.bfloat16)
            new_tensor = ttml.autograd.Tensor.from_numpy(weight_np, layout=ttnn.Layout.TILE)
            tensor.assign(new_tensor)
        elif "bias" in name:
            # Re-initialize biases with 0
            bias_np = np.zeros(shape, dtype=ml_dtypes.bfloat16)
            new_tensor = ttml.autograd.Tensor.from_numpy(bias_np, layout=ttnn.Layout.TILE)
            tensor.assign(new_tensor)


@dataclass
class NanoGPTExperimentalConfig:
    use_composite_layernorm: bool = False  # Use composite vs fused layernorm


@dataclass
class NanoGPTConfig:
    """Configuration for NanoGPT model."""

    vocab_size: int = 50304  # GPT-2 vocab size
    block_size: int = 1024  # Maximum sequence length
    n_embd: int = 768  # Embedding dimension
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12  # Number of attention heads
    dropout: float = 0.2  # Dropout probability (matching C++ default: dropout_prob = 0.2F)
    bias: bool = True  # Use bias in linear layers and layer norm
    runner_type: RunnerType = RunnerType.Default  # For memory efficient block execution
    weight_tying: WeightTyingType = WeightTyingType.Disabled  # Ties tok_emb and fc weights
    # Selects positional embedding type:
    # - trainable positional embedding (trainable)
    # - sinusoidal positional embedding (fixed)
    positional_embedding_type: Literal["trainable", "fixed"] = "trainable"
    experimental: NanoGPTExperimentalConfig = field(default_factory=NanoGPTExperimentalConfig)


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

        self.fc = LinearLayer(config.n_embd, config.vocab_size, False)  # False - no bias
        vocab_size_divisible_by_32 = (config.vocab_size + 31) // 32 * 32
        self.tok_emb = Embedding(vocab_size_divisible_by_32, config.n_embd)

        if config.weight_tying == ttml.models.WeightTyingType.Enabled:
            self.tok_emb.weight = self.fc.get_weight()

        if config.positional_embedding_type == "trainable":
            self.pos_emb = TrainablePositionalEmbedding(config.block_size, config.n_embd, config.dropout)
        elif config.positional_embedding_type == "fixed":
            self.pos_emb = PositionalEmbedding(config.block_size, config.n_embd, config.dropout)
        else:
            raise ValueError(
                f"Unsupported positional_embedding_type="
                f"{config.positional_embedding_type!r}; expected 'trainable' or 'fixed'."
            )

        # Transformer blocks (ModuleList auto-registers all blocks)
        self.blocks = ModuleList(
            [
                GPTBlock(
                    config.n_embd,
                    config.n_head,
                    config.dropout,
                    config.bias,
                    config.experimental.use_composite_layernorm,
                )
                for _ in range(config.n_layer)
            ]
        )

        # Final layer norm (use ml_dtypes.bfloat16)
        # Layer norm parameters must be in TILE layout
        ln_f_shape = (1, 1, 1, config.n_embd)
        gamma_f_np = np.ones(ln_f_shape, dtype=ml_dtypes.bfloat16)
        gamma_f_tensor = ttml.autograd.Tensor.from_numpy(gamma_f_np, layout=ttnn.Layout.TILE)
        self.ln_f_gamma = Parameter(gamma_f_tensor)

        if config.bias:
            beta_f_np = np.zeros(ln_f_shape, dtype=ml_dtypes.bfloat16)
            beta_f_tensor = ttml.autograd.Tensor.from_numpy(beta_f_np, layout=ttnn.Layout.TILE)
            self.ln_f_beta = Parameter(beta_f_tensor)
        else:
            self.ln_f_beta = None

        # Initialize weights
        # This re-initializes all "weight" to normal(0, 0.02) and all "bias" to 0
        initialize_weights_gpt2(self.parameters())

    # train() and eval() are inherited from AbstractModuleBase
    # They automatically propagate RunMode to all registered submodules

    def forward(self, idx: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None) -> ttml.autograd.Tensor:
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

        for block in self.blocks:
            if self.config.runner_type == ttml.models.RunnerType.MemoryEfficient:
                out = memory_efficient_runner(block, out, mask)
            else:
                out = block(out, mask)

        if self.config.experimental.use_composite_layernorm:
            layernorm_op = ttml.ops.layernorm.composite_layernorm
        else:
            layernorm_op = ttml.ops.layernorm.layernorm

        out = layernorm_op(
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
    "PositionalEmbedding",
    "TrainablePositionalEmbedding",
    "GPTBlock",
    "NanoGPT",
    "NanoGPTExperimentalConfig",
    "NanoGPTConfig",
    "create_nanogpt",
    "initialize_weights_gpt2",
]
