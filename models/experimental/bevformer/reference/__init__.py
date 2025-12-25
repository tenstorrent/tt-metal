# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BEVFormer Reference Implementation.

This package contains PyTorch reference implementations of BEVFormer components
including:
- Multi-Scale Deformable Attention
- Spatial Cross Attention (SCA)
- Temporal Self Attention (TSA)
- Complete BEVFormer Encoder
"""

from .ms_deformable_attention import MSDeformableAttention
from .spatial_cross_attention import SpatialCrossAttention
from .temporal_self_attention import TemporalSelfAttention

__all__ = [
    "MSDeformableAttention",
    "SpatialCrossAttention",
    "TemporalSelfAttention",
]
