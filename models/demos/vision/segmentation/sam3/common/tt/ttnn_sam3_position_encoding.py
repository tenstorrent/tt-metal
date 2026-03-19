# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Sinusoidal Position Encoding for SAM3.

Generates 2D sinusoidal position encodings used throughout SAM3:
- FPN neck position encodings
- Transformer encoder/decoder position encodings
- Geometry encoder position encodings

Reference: sam3.model.position_encoding.PositionEmbeddingSine
"""

import math
from typing import Tuple, Optional

import torch
import ttnn


def generate_position_encoding_sine(
    height: int,
    width: int,
    num_pos_feats: int = 256,
    temperature: float = 10000.0,
    normalize: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Generate 2D sinusoidal position encoding.

    Matches the output of sam3.model.position_encoding.PositionEmbeddingSine.

    Args:
        height: Height of the feature map.
        width: Width of the feature map.
        num_pos_feats: Number of positional features per dimension (half of total channels).
        temperature: Temperature for the sinusoidal encoding.
        normalize: Whether to normalize positions to [0, 1].
        scale: Scale factor for normalization. Default: 2*pi.

    Returns:
        Position encoding tensor (1, C, H, W) where C = 2 * num_pos_feats.
    """
    if scale is None:
        scale = 2 * math.pi

    # Create position grids
    y_embed = torch.arange(1, height + 1, dtype=torch.float32).unsqueeze(1).expand(height, width)
    x_embed = torch.arange(1, width + 1, dtype=torch.float32).unsqueeze(0).expand(height, width)

    if normalize:
        eps = 1e-6
        y_embed = y_embed / (height + eps) * scale
        x_embed = x_embed / (width + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t  # (H, W, num_pos_feats)
    pos_y = y_embed[:, :, None] / dim_t  # (H, W, num_pos_feats)

    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)

    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)  # (C, H, W)
    return pos.unsqueeze(0)  # (1, C, H, W)


def generate_position_encodings_for_fpn(
    feature_shapes: list,
    num_pos_feats: int = 256,
    temperature: float = 10000.0,
) -> list:
    """Generate position encodings for each FPN feature level.

    Args:
        feature_shapes: List of (H, W) tuples for each FPN level.
        num_pos_feats: Number of positional features.
        temperature: Temperature for sinusoidal encoding.

    Returns:
        List of position encoding tensors, each (1, 2*num_pos_feats, H, W).
    """
    pos_encodings = []
    for h, w in feature_shapes:
        pos = generate_position_encoding_sine(h, w, num_pos_feats, temperature)
        pos_encodings.append(pos)
    return pos_encodings


def position_encoding_to_ttnn(
    pos_encoding: torch.Tensor,
    device: object,
    dtype: object = ttnn.bfloat16,
) -> ttnn.Tensor:
    """Convert a position encoding tensor to ttnn format on device.

    Args:
        pos_encoding: torch tensor (1, C, H, W).
        device: ttnn device.
        dtype: ttnn data type.

    Returns:
        ttnn tensor on device.
    """
    return ttnn.from_torch(
        pos_encoding, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
