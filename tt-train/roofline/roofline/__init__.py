# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline estimation core classes and functions.

This subpackage contains:
- RooflineEstimate: Performance metrics for a single operation
- RooflineContext: Tracks cumulative performance across operations
- Roofline functions for various operations
"""

from .roofline import RooflineEstimate, RooflineContext
from .matmul import matmul_roofline
from .reduction import reduction_roofline
from .elementwise import elementwise_roofline
from .embedding import embedding_roofline
from .layernorm import layernorm_roofline
from .rmsnorm import rmsnorm_roofline
from .softmax import softmax_roofline
from .attention import (
    attention_roofline,
    heads_creation_roofline,
    heads_fusion_roofline,
    grouped_heads_creation_roofline,
)
from .rope import rope_roofline
from .dropout import dropout_roofline
from .cross_entropy import cross_entropy_roofline

__all__ = [
    # Core classes
    "RooflineEstimate",
    "RooflineContext",
    # Roofline functions
    "matmul_roofline",
    "reduction_roofline",
    "elementwise_roofline",
    "embedding_roofline",
    "layernorm_roofline",
    "rmsnorm_roofline",
    "softmax_roofline",
    "attention_roofline",
    "heads_creation_roofline",
    "heads_fusion_roofline",
    "grouped_heads_creation_roofline",
    "rope_roofline",
    "dropout_roofline",
    "cross_entropy_roofline",
]
