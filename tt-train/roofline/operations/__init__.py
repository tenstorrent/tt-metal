# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Operations for roofline modeling.

This package contains roofline-aware operation implementations.
"""

from .operation import (
    RooflineFunctionContext,
    RooflineFunction,
    create_grad_tensor,
    create_activation_tensor,
)
from .linear import MockLinearOp
from .matmul import MockMatMulOp
from .elementwise import MockAddOp, MockMulOp, MockGELUOp
from .silu import MockSiLUOp
from .embedding import MockEmbeddingOp
from .layernorm import MockLayerNormOp
from .rmsnorm import MockRMSNormOp
from .dropout import MockDropoutOp
from .attention import (
    MockHeadsCreationOp,
    MockHeadsFusionOp,
    MockGroupedHeadsCreationOp,
    MockScaledDotProductAttentionOp,
    MockScaledDotProductAttentionFusedOp,
)
from .rope import MockRoPEOp
from .cross_entropy import MockCrossEntropyLossOp
from .swiglu_fused import MockSwiGLUFusedOp, SwiGLUFusedImpl

__all__ = [
    # Base classes
    "RooflineFunctionContext",
    "RooflineFunction",
    "create_grad_tensor",
    "create_activation_tensor",
    # Linear/MatMul
    "MockLinearOp",
    "MockMatMulOp",
    # Elementwise
    "MockAddOp",
    "MockMulOp",
    "MockGELUOp",
    # Activation
    "MockSiLUOp",
    # Embedding
    "MockEmbeddingOp",
    # Normalization
    "MockLayerNormOp",
    "MockRMSNormOp",
    # Dropout
    "MockDropoutOp",
    # Attention
    "MockHeadsCreationOp",
    "MockHeadsFusionOp",
    "MockGroupedHeadsCreationOp",
    "MockScaledDotProductAttentionOp",
    "MockScaledDotProductAttentionFusedOp",
    # Rotary Position Embedding
    "MockRoPEOp",
    # Loss
    "MockCrossEntropyLossOp",
    # Fused SwiGLU
    "MockSwiGLUFusedOp",
    "SwiGLUFusedImpl",
]
