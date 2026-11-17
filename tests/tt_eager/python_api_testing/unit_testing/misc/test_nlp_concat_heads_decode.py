# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from tt_py_test_utils_common.nlp_concat_heads_decode import num_to_corerange, run_test_concat_head

import pytest
from loguru import logger
import torch
from torch import nn
import ttnn
import ttnn.core

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)

from models.common.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    nearest_32,
)


@pytest.mark.parametrize(
    "n_local_heads, padded_local_heads, head_dim, batch_size",
    (
        # Single head-tile (padded_local_heads == 32)
        (8, 32, 128, 32),
        (17, 32, 96, 32),
        (32, 32, 64, 32),
        (8, 32, 128, 16),
        # Multiple head-tiles (padded_local_heads > 32)
        (8, 64, 64, 32),  # 2 head-tiles, low utilization
        (32, 64, 64, 32),  # 2 head-tiles, fully populated within first tile
        (32, 64, 128, 32),  # 2 head-tiles + multi-tile head_dim
        (16, 96, 64, 32),  # 3 head-tiles
        # n_local_heads > 32 — exercises head_tile_idx > 0 in the offset formula
        # (batch is capped at 32 by the op; n_local_heads uses device compute grid)
        (40, 64, 64, 8),  # heads 32..39 read from 2nd head-tile
        (64, 64, 64, 8),  # all 32 heads in 2nd head-tile fully populated
        (40, 96, 64, 8),  # 3 head-tiles, heads 32..39 from 2nd tile (non-power-of-2 padded_local_heads)
    ),
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 1), id="1x1_grid")], indirect=True)
def test_concat_head(
    n_local_heads,
    padded_local_heads,
    head_dim,
    batch_size,
    mesh_device,
):
    torch.manual_seed(0)

    for i in range(3):
        # multiple loops to test program caching
        run_test_concat_head(mesh_device, n_local_heads, padded_local_heads, head_dim, batch_size)


@pytest.mark.parametrize(
    "n_local_heads, padded_local_heads, head_dim, batch_size, input_sub_core_grids, compute_sub_core_grids",
    (
        (  # Test Case 0: Input is sharded on 8 cores, use all 8 cores for compute
            8,
            32,
            128,
            8,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
                ]
            ),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
                ]
            ),
        ),
        (  # Test Case 1: Input is sharded on only 1 core, use the compute sub core grids for compute
            8,
            32,
            128,
            1,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0)),
                ]
            ),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
                ]
            ),
        ),
        (  # Test Case 2: padded_local_heads=64 (multi head-tile path) on subcoregrids
            8,
            64,
            128,
            8,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
                ]
            ),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
                ]
            ),
        ),
        (  # Test Case 3: n_local_heads=64 — exercises head_tile_idx > 0 on subcoregrids
            64,
            64,
            64,
            8,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                ]
            ),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7)),
                ]
            ),
        ),
    ),
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 1), id="1x1_grid")], indirect=True)
def test_concat_head_subcoregrids(
    n_local_heads,
    padded_local_heads,
    head_dim,
    batch_size,
    input_sub_core_grids,
    compute_sub_core_grids,
    mesh_device,
):
    torch.manual_seed(0)

    for i in range(3):
        # multiple loops to test program caching
        run_test_concat_head(
            mesh_device,
            n_local_heads,
            padded_local_heads,
            head_dim,
            batch_size,
            input_sub_core_grids,
            compute_sub_core_grids,
        )
