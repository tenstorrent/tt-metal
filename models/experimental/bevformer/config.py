# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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


@dataclass
class SpatialCrossAttentionConfig(AttentionConfig):
    """Extended config for spatial cross attention"""

    num_cams: int = 6
    pc_range: List[float] = None

    def __post_init__(self):
        if self.pc_range is None:
            self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


@dataclass
class TemporalSelfAttentionConfig(AttentionConfig):
    """Extended config for temporal self attention"""

    num_frames: int = 2
    memory_len: int = 256
