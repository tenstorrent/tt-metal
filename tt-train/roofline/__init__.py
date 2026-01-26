# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline modeling system for Python operations.

This module provides a standalone roofline estimation framework for
modeling performance of training operations on Tenstorrent hardware
without requiring ttml/ttnn dependencies.

Key Components:
    - MockTensor: Tensor metadata without actual data
    - RooflineEstimate: Performance metrics for single operation
    - RooflineContext: Tracks cumulative performance across operations
    - RooflineFunction: Base class for roofline-aware operations
    - MockModule: Base class for modules with auto-registration

Hardware Configurations:
    - WORMHOLE_N150: Single n150 card (72 cores)
    - WORMHOLE_N300: Dual-chip n300 card (128 cores)
    - WORMHOLE_GALAXY: 32-chip Galaxy system
    - BLACKHOLE_CHIP: Single Blackhole chip
    - BLACKHOLE_GALAXY: 32-chip Blackhole Galaxy

Example:
    >>> from roofline import (
    ...     MockTensor, MockLinearLayer, MockModule, MockModuleList,
    ...     RooflineContext, WORMHOLE_N150
    ... )
    >>>
    >>> class MockMLP(MockModule):
    ...     def __init__(self, hidden_dim, ffn_dim):
    ...         super().__init__()
    ...         self.up_proj = MockLinearLayer(hidden_dim, ffn_dim)
    ...         self.down_proj = MockLinearLayer(ffn_dim, hidden_dim)
    ...
    ...     def forward(self, ctx, x):
    ...         x = self.up_proj(ctx, x)
    ...         x = self.down_proj(ctx, x)
    ...         return x
    >>>
    >>> # Create context and model
    >>> ctx = RooflineContext(WORMHOLE_N150)
    >>> mlp = MockMLP(hidden_dim=4096, ffn_dim=11008)
    >>>
    >>> # Run forward pass
    >>> x = MockTensor((1, 1, 8192, 4096), requires_grad=False)
    >>> y = mlp(ctx, x)
    >>>
    >>> # Run backward pass
    >>> y.backward(ctx)
    >>>
    >>> # Print analysis
    >>> print(ctx.summary(mlp))
"""

# Hardware specs
from .hardware import (
    # Enums
    MathFidelity,
    DataType,
    BottleneckType,
    # Hardware specs
    HardwareSpec,
    WORMHOLE_N150,
    WORMHOLE_N300,
    WORMHOLE_GALAXY,
    BLACKHOLE_CHIP,
    BLACKHOLE_GALAXY,
)

# Mock tensor with autograd
from .mock_tensor import (
    MockTensor,
    BackwardNode,
)

# Roofline estimation classes and functions
from .roofline import (
    RooflineEstimate,
    RooflineContext,
    matmul_roofline,
    reduction_roofline,
    elementwise_roofline,
    embedding_roofline,
    layernorm_roofline,
    softmax_roofline,
    attention_roofline,
    heads_creation_roofline,
    heads_fusion_roofline,
    dropout_roofline,
    cross_entropy_roofline,
)

# Module system
from .modules import (
    # Base classes
    MockParameter,
    MockModule,
    MockModuleList,
    MockModuleDict,
    # Linear
    MockLinearLayer,
    # Embedding
    MockEmbedding,
    MockTrainablePositionalEmbedding,
    # Normalization
    MockLayerNorm,
    # Dropout
    MockDropout,
    # Attention
    MockMultiHeadAttention,
    # MLP
    MockGPTMLP,
    # Transformer
    MockGPTBlock,
    # NanoGPT
    MockNanoGPT,
    MockNanoGPTConfig,
    create_mock_nanogpt,
)

# Operations
from .operations import (
    # Base classes
    RooflineFunctionContext,
    RooflineFunction,
    # Linear/MatMul
    MockLinearOp,
    MockMatMulOp,
    # Elementwise
    MockAddOp,
    MockMulOp,
    MockGELUOp,
    # Embedding
    MockEmbeddingOp,
    # Normalization
    MockLayerNormOp,
    # Dropout
    MockDropoutOp,
    # Attention
    MockHeadsCreationOp,
    MockHeadsFusionOp,
    MockScaledDotProductAttentionOp,
    # Loss
    MockCrossEntropyLossOp,
)

# Training utilities
from .training import (
    MockAdamW,
    mock_clip_grad_norm,
)

__all__ = [
    # Hardware
    "MathFidelity",
    "DataType",
    "BottleneckType",
    "HardwareSpec",
    "WORMHOLE_N150",
    "WORMHOLE_N300",
    "WORMHOLE_GALAXY",
    "BLACKHOLE_CHIP",
    "BLACKHOLE_GALAXY",
    # Roofline core
    "RooflineEstimate",
    "RooflineContext",
    # Roofline functions
    "matmul_roofline",
    "reduction_roofline",
    "elementwise_roofline",
    "embedding_roofline",
    "layernorm_roofline",
    "softmax_roofline",
    "attention_roofline",
    "heads_creation_roofline",
    "heads_fusion_roofline",
    "dropout_roofline",
    "cross_entropy_roofline",
    # Mock tensor
    "MockTensor",
    "BackwardNode",
    # Module base classes
    "MockParameter",
    "MockModule",
    "MockModuleList",
    "MockModuleDict",
    # Modules
    "MockLinearLayer",
    "MockEmbedding",
    "MockTrainablePositionalEmbedding",
    "MockLayerNorm",
    "MockDropout",
    "MockMultiHeadAttention",
    "MockGPTMLP",
    "MockGPTBlock",
    "MockNanoGPT",
    "MockNanoGPTConfig",
    "create_mock_nanogpt",
    # Operation base classes
    "RooflineFunctionContext",
    "RooflineFunction",
    # Operations
    "MockLinearOp",
    "MockMatMulOp",
    "MockAddOp",
    "MockMulOp",
    "MockGELUOp",
    "MockEmbeddingOp",
    "MockLayerNormOp",
    "MockDropoutOp",
    "MockHeadsCreationOp",
    "MockHeadsFusionOp",
    "MockScaledDotProductAttentionOp",
    "MockCrossEntropyLossOp",
    # Training utilities
    "MockAdamW",
    "mock_clip_grad_norm",
]
