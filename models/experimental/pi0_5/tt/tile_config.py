# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pi0.5 device tile geometry — single source of truth for TILE-layout uploads."""
from __future__ import annotations

from typing import Any, Optional

import ttnn

# M-dimension tile height. Use 16 for M16 decode paths; 32 for legacy 32×32 geometry.
TILE_HEIGHT = 16
TILE_WIDTH = 32
# Packed bfloat8_b activations at the model tile geometry (tiny tiles now support blocked dtypes).
ACT_DTYPE = ttnn.bfloat8_b


def pi05_tile() -> ttnn.Tile:
    return ttnn.Tile((TILE_HEIGHT, TILE_WIDTH))


def from_torch_pi05(
    tensor,
    *,
    dtype: ttnn.DataType,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    device: Optional[ttnn.MeshDevice] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    tile: Optional[ttnn.Tile] = None,
    **kwargs: Any,
) -> ttnn.Tensor:
    """Upload a host tensor with the model tile shape (all dtypes, including blocked bf8_b/bf4_b,
    now support the tiny tile geometry on device). ROW-MAJOR uploads keep the default tile (a custom
    tile config is only valid for TILE layout)."""
    upload_tile = tile
    if upload_tile is None and layout == ttnn.TILE_LAYOUT:
        upload_tile = pi05_tile()
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
        tile=upload_tile,
        **kwargs,
    )


__all__ = ["TILE_HEIGHT", "TILE_WIDTH", "ACT_DTYPE", "pi05_tile", "from_torch_pi05"]
