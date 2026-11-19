# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Which of the model's hand-sized plans fit the device's compute grid.

| Plan                                   | Needs                                    | Otherwise                         |
|----------------------------------------|------------------------------------------|-----------------------------------|
| Interleaved matmul program configs     | a core row per image, 8 or more columns  | ttnn chooses each program config  |
| L1 activations (L1, MIXED_PRECISION)   | a core row per image, 11 or more columns | activations in DRAM               |
| BFLOAT8_B-SHARDED                      | 8 x batch or batch x 8 cores             | the BFLOAT8_B-DRAM config         |
| Experimental QKV head split            | a core column per sequence tile, a core  | the generic head split            |
|                                        | row per image, batch 7 to 9              |                                   |
"""

import ttnn

# The interleaved matmul configs split each image's output columns over up to MAX_MATMUL_COLUMNS
# cores. Below MIN_MATMUL_COLUMNS their per-core output blocks grow past what L1 holds.
MAX_MATMUL_COLUMNS = 12
MIN_MATMUL_COLUMNS = 8

# The L1 configs' activations, with the matmul configs' blocks for that many columns, fit in L1 from
# MIN_L1_ACTIVATION_COLUMNS columns up: on 8 columns MIXED_PRECISION_BATCH9's FF1 blocks clash with
# the activations of earlier layers.
MIN_L1_ACTIVATION_COLUMNS = 11

# BFLOAT8_B-SHARDED shards each image's activations over this many cores.
SHARDED_CORES_PER_IMAGE = 8


def matmul_columns(grid):
    """The core columns the interleaved matmul configs split each image's output over."""
    return min(MAX_MATMUL_COLUMNS, grid.x)


def fits_matmul_configs(batch, grid):
    """Whether the interleaved matmul program configs fit: a core row per image and at least
    MIN_MATMUL_COLUMNS columns."""
    return batch <= grid.y and grid.x >= MIN_MATMUL_COLUMNS


def fits_l1_activations(batch, grid):
    """Whether the L1 configs' interleaved activations fit in L1, next to the matmul configs' blocks."""
    return batch <= grid.y and grid.x >= MIN_L1_ACTIVATION_COLUMNS


def sharded_grid(batch, grid):
    """The core grid and shard orientation of BFLOAT8_B-SHARDED, or None when the grid holds neither
    SHARDED_CORES_PER_IMAGE x batch nor batch x SHARDED_CORES_PER_IMAGE cores."""
    if SHARDED_CORES_PER_IMAGE <= grid.x and batch <= grid.y:
        return ttnn.CoreCoord(SHARDED_CORES_PER_IMAGE, batch), ttnn.ShardOrientation.ROW_MAJOR
    if batch <= grid.x and SHARDED_CORES_PER_IMAGE <= grid.y:
        return ttnn.CoreCoord(batch, SHARDED_CORES_PER_IMAGE), ttnn.ShardOrientation.COL_MAJOR
    return None


def fits_experimental_split_heads(batch, sequence, grid):
    """Whether ttnn.experimental.split_query_key_value_and_split_heads takes an interleaved input of
    this shape on this grid: it places one core per sequence tile and image, and only takes batch
    7 to 9."""
    return sequence // ttnn.TILE_SIZE <= grid.x and batch <= grid.y and 7 <= batch <= 9
