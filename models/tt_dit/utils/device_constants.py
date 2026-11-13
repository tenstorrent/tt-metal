# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-chip constants for keying tables by device."""

from __future__ import annotations

import ttnn

# compute_with_storage_grid_size() of each chip, as (x, y).
BLACKHOLE_COMPUTE_GRID = (12, 10)
WORMHOLE_COMPUTE_GRID = (8, 9)


def compute_grid_key(grid: ttnn.CoreCoord) -> tuple[int, int]:
    """`grid` as the (x, y) tuple that per-grid tables are keyed by."""
    return (grid.x, grid.y)
