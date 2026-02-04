# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""mLSTM Block implementation.

The mLSTM block wraps the mLSTM layer with pre-normalization
and residual (skip) connections, following the standard
Transformer-style block pattern.

Reference:
    - Paper: "xLSTM: Extended Long Short-Term Memory" (https://arxiv.org/abs/2405.04517)
"""

from dataclasses import dataclass, field
from typing import Optional

import ttnn

import ttml
from ttml.modules import AbstractModuleBase
from .components import LayerNorm
from .mlstm_layer import mLSTMLayer, mLSTMLayerConfig


@dataclass
class mLSTMBlockConfig:
    """Configuration for mLSTMBlock."""

    mlstm: mLSTMLayerConfig = field(default_factory=mLSTMLayerConfig)

    # Block tracking
    _num_blocks: int = 1
    _block_idx: int = 0

    def __post_init__(self):
        self.mlstm._num_blocks = self._num_blocks
        self.mlstm.__post_init__()


class mLSTMBlock(AbstractModuleBase):
    """mLSTM Block with pre-normalization and skip connection.

    The block follows the Pre-LN Transformer pattern:
        output = x + mLSTMLayer(LayerNorm(x))

    This design provides:
    - Better training stability (pre-normalization)
    - Easy gradient flow (residual connection)
    - Modular composition for stacking

    Args:
        config: mLSTMBlockConfig containing the mLSTM layer configuration

    Example:
        >>> config = mLSTMBlockConfig(
        ...     mlstm=mLSTMLayerConfig(
        ...         embedding_dim=512,
        ...         num_heads=8,
        ...         proj_factor=2.0,
        ...         context_length=1024,
        ...     )
        ... )
        >>> block = mLSTMBlock(config)
        >>> x = ttml.autograd.Tensor.from_numpy(np.random.randn(2, 128, 512).astype(np.float32))
        >>> output = block(x)  # (2, 128, 512)
    """

    def __init__(self, config: mLSTMBlockConfig) -> None:
        super().__init__()
        self.config = config

        embedding_dim = config.mlstm.embedding_dim

        # Pre-normalization layer
        self.xlstm_norm = LayerNorm(
            ndim=embedding_dim,
            weight=True,
            bias=False,
        )

        # mLSTM layer
        self.xlstm = mLSTMLayer(config.mlstm)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of mLSTM block.

        Args:
            x: Input tensor of shape (B, S, embedding_dim)

        Returns:
            Output tensor of shape (B, S, embedding_dim)
        """
        # Pre-norm + mLSTM layer
        x_norm = self.xlstm_norm(x)
        x_mlstm = self.xlstm(x_norm)

        # Residual connection
        x_val = x.get_value()
        x_mlstm_val = x_mlstm.get_value()
        output_val = ttnn.add(x_val, x_mlstm_val)

        return ttml.autograd.create_tensor(output_val, requires_grad=True)


class xLSTMStack(AbstractModuleBase):
    """Stack of mLSTM blocks for building full xLSTM models.

    Creates a sequential stack of mLSTM blocks that can be used
    as the core of a language model or other sequence model.

    Args:
        embedding_dim: Dimension of token embeddings
        num_blocks: Number of mLSTM blocks to stack
        num_heads: Number of attention heads per block
        proj_factor: Up-projection factor for inner dimension
        context_length: Maximum sequence length
        conv1d_kernel_size: Kernel size for causal convolution
        dropout: Dropout probability

    Example:
        >>> stack = xLSTMStack(
        ...     embedding_dim=512,
        ...     num_blocks=12,
        ...     num_heads=8,
        ...     context_length=2048,
        ... )
        >>> x = ttml.autograd.Tensor.from_numpy(np.random.randn(2, 128, 512).astype(np.float32))
        >>> output = stack(x)  # (2, 128, 512)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int = 4,
        proj_factor: float = 2.0,
        context_length: int = 2048,
        conv1d_kernel_size: int = 4,
        qkv_proj_blocksize: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks

        # Create blocks
        self.blocks = ttml.modules.ModuleList()
        for block_idx in range(num_blocks):
            layer_config = mLSTMLayerConfig(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                proj_factor=proj_factor,
                conv1d_kernel_size=conv1d_kernel_size,
                qkv_proj_blocksize=qkv_proj_blocksize,
                context_length=context_length,
                dropout=dropout,
                _num_blocks=num_blocks,
            )
            block_config = mLSTMBlockConfig(
                mlstm=layer_config,
                _num_blocks=num_blocks,
                _block_idx=block_idx,
            )
            self.blocks.append(mLSTMBlock(block_config))

        # Final layer norm
        self.final_norm = LayerNorm(
            ndim=embedding_dim,
            weight=True,
            bias=False,
        )

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass through all blocks.

        Args:
            x: Input tensor of shape (B, S, embedding_dim)

        Returns:
            Output tensor of shape (B, S, embedding_dim)
        """
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return x
