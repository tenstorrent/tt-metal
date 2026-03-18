# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""NanoGPT model package.

This package provides a Python implementation of NanoGPT (a small GPT model)
using ttml's C++ operations for computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal

import ttnn
import ttml
from ttml.modules import (
    AbstractModuleBase,
    Embedding,
    Parameter,
    ModuleList,
    LinearLayer,
)

from .. import RunnerType, WeightTyingType, memory_efficient_runner
from .pos_embedding import PositionalEmbedding, TrainablePositionalEmbedding
from .gpt_block import GPTBlock


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
    dropout: float = (
        0.2  # Dropout probability (matching C++ default: dropout_prob = 0.2F)
    )
    bias: bool = True  # Use bias in linear layers and layer norm
    runner_type: RunnerType = RunnerType.Default  # For memory efficient block execution
    weight_tying: WeightTyingType = (
        WeightTyingType.Disabled
    )  # Ties tok_emb and fc weights
    # Selects positional embedding type:
    # - trainable positional embedding (trainable)
    # - sinusoidal positional embedding (fixed)
    positional_embedding_type: Literal["trainable", "fixed"] = "trainable"
    experimental: NanoGPTExperimentalConfig = field(
        default_factory=NanoGPTExperimentalConfig
    )


class NanoGPT(AbstractModuleBase):
    """NanoGPT model implemented in Python using ttml operations.

    This implementation matches the C++ ttml::models::gpt2::Transformer class.
    See tt-train/sources/ttml/models/gpt2.cpp for reference.
    """

    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()

        self.config = config

        weight_init = ttml.init.normal(0.0, 0.02)
        bias_init = ttml.init.zeros()

        self.fc = LinearLayer(
            config.n_embd,
            config.vocab_size,
            False,
            weight_init=weight_init,
        )
        vocab_size_divisible_by_32 = (config.vocab_size + 31) // 32 * 32
        self.tok_emb = Embedding(
            vocab_size_divisible_by_32,
            config.n_embd,
            weight_init=weight_init,
        )

        if config.weight_tying == ttml.models.WeightTyingType.Enabled:
            self.tok_emb.weight = self.fc.weight.tensor

        if config.positional_embedding_type == "trainable":
            self.pos_emb = TrainablePositionalEmbedding(
                config.block_size,
                config.n_embd,
                config.dropout,
                weight_init=weight_init,
            )
        elif config.positional_embedding_type == "fixed":
            self.pos_emb = PositionalEmbedding(
                config.block_size, config.n_embd, config.dropout
            )
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
                    weight_init=weight_init,
                    bias_init=bias_init,
                )
                for _ in range(config.n_layer)
            ]
        )

        # Final layer norm parameters
        ln_f_shape = (1, 1, 1, config.n_embd)
        self.ln_f_gamma = Parameter(ttml.init.ones()(ln_f_shape))

        if config.bias:
            self.ln_f_beta = Parameter(ttml.init.zeros()(ln_f_shape))
        else:
            self.ln_f_beta = None

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

        for block in self.blocks:
            if self.config.runner_type == ttml.models.RunnerType.MemoryEfficient:
                out = memory_efficient_runner(block, out, mask)
            elif self.config.runner_type == ttml.models.RunnerType.Default:
                out = block(out, mask)
            else:
                raise ValueError(
                    "Unknown runner type. Supported runner types ['default', 'memory_efficient']"
                )

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
    """Factory function to create a NanoGPT model."""
    return NanoGPT(config)


__all__ = [
    "PositionalEmbedding",
    "TrainablePositionalEmbedding",
    "GPTBlock",
    "NanoGPT",
    "NanoGPTExperimentalConfig",
    "NanoGPTConfig",
    "create_nanogpt",
]
