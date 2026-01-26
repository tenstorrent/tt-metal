# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Operations for roofline modeling.

This package contains roofline-aware operation implementations.
"""

from .operation import RooflineFunctionContext, RooflineFunction
from .linear import MockLinearOp
from .matmul import MockMatMulOp
from .elementwise import MockAddOp, MockMulOp, MockGELUOp
from .embedding import MockEmbeddingOp
from .layernorm import MockLayerNormOp
from .dropout import MockDropoutOp
from .attention import (
    MockHeadsCreationOp,
    MockHeadsFusionOp,
    MockScaledDotProductAttentionOp,
)
from .cross_entropy import MockCrossEntropyLossOp

__all__ = [
    # Base classes
    "RooflineFunctionContext",
    "RooflineFunction",
    # Linear/MatMul
    "MockLinearOp",
    "MockMatMulOp",
    # Elementwise
    "MockAddOp",
    "MockMulOp",
    "MockGELUOp",
    # Embedding
    "MockEmbeddingOp",
    # Normalization
    "MockLayerNormOp",
    # Dropout
    "MockDropoutOp",
    # Attention
    "MockHeadsCreationOp",
    "MockHeadsFusionOp",
    "MockScaledDotProductAttentionOp",
    # Loss
    "MockCrossEntropyLossOp",
]
