# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_minimal_matmul.py — do NOT regenerate this file
# from a capture. See small_grid.py for the conventions these cases follow.
#
# The captured calls are prefill-sized ([2x2048 or 4x1024] x 2560..9728) with a bfloat8_b
# weight width-sharded over 12 DRAM banks and an 8x8 MinimalMatmulConfig grid. These keep
# the MinimalMatmulConfig path in bfloat16 at 128 rows on a 2x1 grid, with the weight both
# interleaved and width-sharded over 2 DRAM banks.
# ---------------------------------------------------------------------------
"""Small-grid cases for ``ttnn.experimental.minimal_matmul`` that fit a 1- or 2-compute-node device."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops import small_grid as S
from models.experimental.ops.quasar.tests.qwen3_vl_ops.small_grid import case, t, out, cfg

_OP = ttnn.experimental.minimal_matmul
_NAME = "ttnn.experimental.minimal_matmul"

CASES = [
    case(
        "small_int_128x256x512",
        _NAME,
        [t([1, 1, 128, 256]), t([1, 1, 256, 512])],
        {
            "config": cfg(
                "MinimalMatmulConfig",
                M_block_size=4,
                K_block_size=4,
                N_block_size=4,
                subblock_h=1,
                subblock_w=1,
                compute_with_storage_grid_size=[2, 1],
            )
        },
        [out([1, 1, 128, 512])],
    ),
    case(
        "small_dramws2_128x256x512",
        _NAME,
        [
            t([1, 1, 128, 256]),
            t([1, 1, 256, 512], mem=S.sharded("WIDTH_SHARDED", [256, 256], cores_x=2, buffer="DRAM")),
        ],
        {
            "config": cfg(
                "MinimalMatmulConfig",
                M_block_size=4,
                K_block_size=4,
                N_block_size=4,
                subblock_h=1,
                subblock_w=1,
                compute_with_storage_grid_size=[2, 1],
            )
        },
        [out([1, 1, 128, 512])],
    ),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case_spec", CASES, ids=[c["id"] for c in CASES])
def test_minimal_matmul_small_grid(ttnn_mesh_device, reset_seeds, case_spec):
    if not G._is_quasar(ttnn_mesh_device):
        # minimal_matmul accepts a 1-wide grid axis only on Quasar; Wormhole/Blackhole need at least 2x2.
        pytest.skip("the 2x1 MinimalMatmulConfig grid is Quasar-only; Wormhole/Blackhole need at least 2x2")
    S.skip_if_grid_too_small(case_spec, ttnn_mesh_device)
    G.run_case(_OP, case_spec, ttnn_mesh_device)
