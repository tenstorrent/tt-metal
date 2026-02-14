# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""mLSTM Layer implementation.

The mLSTM layer is a complete processing unit that includes:
- Up-projection to inner dimension
- Causal convolution
- Q/K/V projections
- mLSTM cell computation
- Down-projection back to embedding dimension

Reference:
    - Paper: "xLSTM: Extended Long Short-Term Memory" (https://arxiv.org/abs/2405.04517)
"""

from dataclasses import dataclass
from math import sqrt
from typing import Optional

import numpy as np
import ttnn

import ttml
from ttml.modules import AbstractModuleBase, Parameter, LinearLayer, RunMode
from .components import (
    CausalConv1d,
    CausalConv1dConfig,
    LinearHeadwiseExpand,
    LinearHeadwiseExpandConfig,
)
from .mlstm_cell import mLSTMCell, mLSTMCellConfig


@dataclass
class mLSTMLayerConfig:
    """Configuration for mLSTMLayer."""

    embedding_dim: int
    num_heads: int = 4
    proj_factor: float = 2.0
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    bias: bool = False
    dropout: float = 0.0
    context_length: int = -1

    # Internal
    _num_blocks: int = 1
    _inner_embedding_dim: int = None

    def __post_init__(self):
        if self._inner_embedding_dim is None:
            self._inner_embedding_dim = round(self.proj_factor * self.embedding_dim)


class mLSTMLayer(AbstractModuleBase):
    """mLSTM Layer with projections, convolution, and mLSTM cell.

    This layer implements the full mLSTM processing pipeline:
    1. Up-projection: embedding_dim -> 2 * inner_dim
    2. Split into x_mlstm and z (output gate)
    3. Causal convolution + SiLU on x_mlstm
    4. Q/K/V projections
    5. mLSTM cell computation
    6. Skip connection with learnable scaling
    7. Output gating with z
    8. Down-projection: inner_dim -> embedding_dim

    Args:
        config: mLSTMLayerConfig
    """

    def __init__(self, config: mLSTMLayerConfig) -> None:
        super().__init__()
        self.config = config
        inner_dim = config._inner_embedding_dim

        # Up-projection: embedding_dim -> 2 * inner_dim
        self.proj_up = LinearLayer(
            config.embedding_dim, 2 * inner_dim, bias=config.bias
        )

        # Q/K/V projections (headwise)
        num_proj_heads = round(inner_dim / config.qkv_proj_blocksize)
        qkv_config = LinearHeadwiseExpandConfig(
            in_features=inner_dim,
            num_heads=num_proj_heads,
            expand_factor=1.0,
            bias=config.bias,
        )
        self.q_proj = LinearHeadwiseExpand(qkv_config)
        self.k_proj = LinearHeadwiseExpand(qkv_config)
        self.v_proj = LinearHeadwiseExpand(qkv_config)

        # Causal conv1d
        self.conv1d = CausalConv1d(
            CausalConv1dConfig(
                feature_dim=inner_dim,
                kernel_size=config.conv1d_kernel_size,
                bias=True,
            )
        )

        # mLSTM cell
        self.mlstm_cell = mLSTMCell(
            mLSTMCellConfig(
                context_length=config.context_length,
                embedding_dim=inner_dim,
                num_heads=config.num_heads,
            )
        )

        # Learnable skip connection scale
        skip_data = np.ones((inner_dim,), dtype=np.float32)
        self.learnable_skip = Parameter(ttml.autograd.Tensor.from_numpy(skip_data))

        # Down-projection: inner_dim -> embedding_dim
        self.proj_down = LinearLayer(inner_dim, config.embedding_dim, bias=config.bias)

        self.dropout_prob = config.dropout
        self.inner_dim = inner_dim

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of mLSTM layer.

        Args:
            x: Input tensor of shape (B, S, embedding_dim)

        Returns:
            Output tensor of shape (B, S, embedding_dim)
        """
        B = x.get_value().shape[0]
        S = x.get_value().shape[1]

        # Up-projection
        x_inner = self.proj_up(x)  # (B, S, 2 * inner_dim)

        # Split into x_mlstm and z
        x_inner_val = x_inner.get_value()
        x_mlstm_val = x_inner_val[:, :, : self.inner_dim]
        z_val = x_inner_val[:, :, self.inner_dim :]

        x_mlstm = ttml.autograd.create_tensor(x_mlstm_val, requires_grad=True)
        z = ttml.autograd.create_tensor(z_val, requires_grad=True)

        # Causal convolution + SiLU activation
        x_mlstm_conv = self.conv1d(x_mlstm)

        # Apply SiLU using ttml ops
        x_mlstm_conv_act = ttml.ops.unary.silu(x_mlstm_conv)

        # Q/K/V projections
        q = self.q_proj(x_mlstm_conv_act)  # (B, S, inner_dim)
        k = self.k_proj(x_mlstm_conv_act)  # (B, S, inner_dim)
        v = self.v_proj(
            x_mlstm
        )  # (B, S, inner_dim) - note: v uses x_mlstm, not conv output

        # mLSTM cell computation
        h_tilde_state = self.mlstm_cell(q, k, v)  # (B, S, inner_dim)

        # Skip connection with learnable scaling
        skip_val = self.learnable_skip.tensor.get_value()
        x_mlstm_conv_act_val = x_mlstm_conv_act.get_value()
        scaled_skip = ttnn.multiply(x_mlstm_conv_act_val, skip_val)
        h_tilde_state_skip_val = ttnn.add(h_tilde_state.get_value(), scaled_skip)

        # Output gating with SiLU(z)
        z_silu_val = ttnn.silu(z_val)
        h_state_val = ttnn.multiply(h_tilde_state_skip_val, z_silu_val)

        # Down-projection
        h_state = ttml.autograd.create_tensor(h_state_val, requires_grad=True)
        y = self.proj_down(h_state)  # (B, S, embedding_dim)

        # Apply dropout if training
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            y = ttml.ops.dropout.dropout(y, self.dropout_prob)

        return y
