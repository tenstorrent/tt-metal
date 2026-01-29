# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Configuration classes for BEVFormer model components.

This module provides dataclass-based configuration objects for various BEVFormer
components including attention mechanisms and encoder configurations. These
configurations enable flexible model setup with validation and default values.

Key configurations:
- AttentionConfig: Base configuration for attention modules
- DeformableAttentionConfig: Configuration for multi-scale deformable attention
- SpatialCrossAttentionConfig: Configuration for spatial cross attention
- TemporalSelfAttentionConfig: Configuration for temporal self attention
"""

from dataclasses import dataclass
from typing import List


@dataclass
class AttentionConfig:
    """Base configuration class for attention modules"""

    embed_dims: int = 256
    num_heads: int = 8
    num_levels: int = 4
    num_points: int = 4
    dropout: float = 0.0
    batch_first: bool = True
    im2col_step: int = 64


@dataclass
class DeformableAttentionConfig(AttentionConfig):
    """Configuration for MultiScale Deformable Attention"""

    query_embed_dims: int = None

    def __post_init__(self):
        """
        Validate and set default values for deformable attention configuration.

        Sets query_embed_dims to embed_dims if not specified and validates
        that embed_dims is divisible by num_heads.
        """
        # Set query_embed_dims to embed_dims if not specified
        if self.query_embed_dims is None:
            self.query_embed_dims = self.embed_dims

        if self.embed_dims % self.num_heads != 0:
            raise ValueError(f"embed_dims ({self.embed_dims}) must be divisible by num_heads ({self.num_heads})")


@dataclass
class SpatialCrossAttentionConfig(AttentionConfig):
    """Extended config for spatial cross attention"""

    num_cams: int = 6
    pc_range: List[float] = None

    def __post_init__(self):
        """
        Set default point cloud range if not specified.

        Uses nuScenes default range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        """
        if self.pc_range is None:
            self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


@dataclass
class TemporalSelfAttentionConfig(AttentionConfig):
    """Extended config for temporal self attention"""

    num_frames: int = 2
    memory_len: int = 256
