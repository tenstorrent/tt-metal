# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_sharded_to_interleaved.py — do NOT
# regenerate this file from a capture.
#
# Every captured case width-shards its input across a grid whose corner is core
# (7,3) (an 8x4 = 32-core grid). On a small device those cores do not exist, so the
# input tensor cannot be allocated and graph_case._shard_grid_fits skips the case at
# run time ("captured L1 shard grid needs cores up to (7,3); device compute grid is
# NxM"). That is the designed skip for oversized shard grids, not a bug.
#
# These variants width-shard the input across cores that DO fit a small device (a
# 2-core row and a 2x3 block), so sharded_to_interleaved actually runs there. Each
# still auto-skips via the same _shard_grid_fits gate on a device too small for its
# own (small) grid, so the file is safe to run anywhere.
# ---------------------------------------------------------------------------
"""Small-grid variants of ``ttnn.sharded_to_interleaved`` (input width-sharded over a small grid)."""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G

_OP = ttnn.sharded_to_interleaved

# Per-core shard is 32 x 256 (256 = 8 tiles wide), matching the captured cases; total
# width = 256 * (number of cores in the grid).
_SHARD_W = 256


def _case(case_id, grid, ncores):
    width = _SHARD_W * ncores
    return {
        "id": case_id,
        "op": "ttnn.sharded_to_interleaved",
        "count": 1,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, width],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [grid], "shape": [32, _SHARD_W], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "memory_config": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, width],
            },
        ],
    }


CASES = [
    # 2 cores: (0,0),(1,0) -> input [1,1,32,512]
    _case("2core_2x1_32x512_bf16_ws-l1", [0, 0, 1, 0], 2),
    # 2x3 block: (0,0)..(1,2) -> 6 cores, input [1,1,32,1536]
    _case("6core_2x3_32x1536_bf16_ws-l1", [0, 0, 1, 2], 6),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_sharded_to_interleaved_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
