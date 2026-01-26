# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Modules for roofline modeling.

This package contains module implementations for roofline estimation.
"""

from .module import MockParameter, MockModule, MockModuleList, MockModuleDict
from .linear import MockLinearLayer
from .embedding import MockEmbedding, MockTrainablePositionalEmbedding
from .layernorm import MockLayerNorm
from .dropout import MockDropout
from .attention import MockMultiHeadAttention
from .mlp import MockGPTMLP
from .gpt_block import MockGPTBlock
from .nanogpt import MockNanoGPT, MockNanoGPTConfig, create_mock_nanogpt

__all__ = [
    # Base classes
    "MockParameter",
    "MockModule",
    "MockModuleList",
    "MockModuleDict",
    # Linear
    "MockLinearLayer",
    # Embedding
    "MockEmbedding",
    "MockTrainablePositionalEmbedding",
    # Normalization
    "MockLayerNorm",
    # Dropout
    "MockDropout",
    # Attention
    "MockMultiHeadAttention",
    # MLP
    "MockGPTMLP",
    # Transformer
    "MockGPTBlock",
    # NanoGPT
    "MockNanoGPT",
    "MockNanoGPTConfig",
    "create_mock_nanogpt",
]
