# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTML Operations package.

This package provides custom operations implemented using the TTML autograd system.

mLSTM Components:
    - mlstm_parallel: Core mLSTM parallel computation with autograd
    - mLSTMCell: mLSTM cell with gates and normalization
    - mLSTMLayer: Full mLSTM layer with projections and convolution
    - mLSTMBlock: Pre-normed block with residual connection
    - xLSTMStack: Stack of mLSTM blocks for building models

Helper Components:
    - CausalConv1d: Causal 1D depthwise convolution
    - LayerNorm: Layer normalization
    - MultiHeadLayerNorm: Multi-head layer normalization
    - LinearHeadwiseExpand: Headwise linear projection
"""

# Core mLSTM operation
from .mlstm import mlstm_parallel, MLSTMParallel

# mLSTM modules
from .mlstm_cell import mLSTMCell, mLSTMCellConfig
from .mlstm_layer import mLSTMLayer, mLSTMLayerConfig
from .mlstm_block import mLSTMBlock, mLSTMBlockConfig, xLSTMStack

# Component layers
from .components import (
    CausalConv1d,
    CausalConv1dConfig,
    LayerNorm,
    MultiHeadLayerNorm,
    LinearHeadwiseExpand,
    LinearHeadwiseExpandConfig,
)

__all__ = [
    # Core mLSTM
    "mlstm_parallel",
    "MLSTMParallel",
    # mLSTM modules
    "mLSTMCell",
    "mLSTMCellConfig",
    "mLSTMLayer",
    "mLSTMLayerConfig",
    "mLSTMBlock",
    "mLSTMBlockConfig",
    "xLSTMStack",
    # Components
    "CausalConv1d",
    "CausalConv1dConfig",
    "LayerNorm",
    "MultiHeadLayerNorm",
    "LinearHeadwiseExpand",
    "LinearHeadwiseExpandConfig",
]
