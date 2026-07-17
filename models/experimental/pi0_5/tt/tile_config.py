# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pi0.5 device tile geometry — single source of truth for TILE-layout uploads."""
from __future__ import annotations

from typing import Any, Optional

import ttnn

# M-dimension tile height. Use 16 for M16 decode paths; 32 for legacy 32×32 geometry.
TILE_HEIGHT = 32
TILE_WIDTH = 32

_BLOCKED_DTYPES = frozenset({ttnn.bfloat8_b, ttnn.bfloat4_b})


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
    """Upload a host tensor with the model tile shape when the dtype supports it.

    Blocked dtypes (bf8_b / bf4_b) keep the default 32×32 tile — tiny tile heights are
    not supported for blocked layouts on device."""
    upload_tile = tile
    if upload_tile is None and dtype not in _BLOCKED_DTYPES:
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


__all__ = ["TILE_HEIGHT", "TILE_WIDTH", "pi05_tile", "from_torch_pi05"]
