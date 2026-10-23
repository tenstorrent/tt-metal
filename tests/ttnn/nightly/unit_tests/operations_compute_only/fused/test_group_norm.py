# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Validation-failure tests for group_norm checks added by PR #39330.

This file runs on a CPU-only runner using TT-Sim (software simulator) in the
L2 nightly workflow.  Tests that call host-side C++ helpers directly do not
need a device at all; tests that go through ``ttnn.group_norm`` use the
simulated device provided by the ``device`` fixture under TT-Sim.

"""

import pytest
import torch

import ttnn
from ttnn._ttnn.operations.normalization import (
    create_group_norm_input_mask,
    _find_expected_dram_grid,
)


# ---------------------------------------------------------------------------
# groupnorm_input_mask.cpp  create_group_norm_input_mask_impl():
#   TT_FATAL — num_cores_across_channel must be > 0
# ---------------------------------------------------------------------------
def test_input_mask_num_cores_across_channel_zero(expect_error):
    with expect_error(RuntimeError, "num_cores_across_channel must be > 0"):
        create_group_norm_input_mask(
            num_channel=256,
            num_groups=32,
            num_cores_across_channel=0,
            data_type=ttnn.DataType.BFLOAT16,
        )


# ---------------------------------------------------------------------------
# groupnorm_input_mask.cpp  create_group_norm_input_mask_impl():
#   TT_FATAL — num_groups must be divisible by num_cores_across_channel
# ---------------------------------------------------------------------------
def test_input_mask_num_groups_not_divisible_by_num_cores(expect_error):
    with expect_error(RuntimeError, "must be divisible by num_cores_across_channel"):
        create_group_norm_input_mask(
            num_channel=256,
            num_groups=32,
            num_cores_across_channel=3,
            data_type=ttnn.DataType.BFLOAT16,
        )


# ---------------------------------------------------------------------------
# groupnorm_nanobind.cpp  _find_expected_dram_grid (wrapper):
#   RuntimeError — no valid DRAM grid (calls find_expected_dram_grid)
# ---------------------------------------------------------------------------
def test_find_expected_dram_grid_no_valid_grid(expect_error):
    with expect_error(RuntimeError, "Cannot find a valid DRAM group-norm grid"):
        _find_expected_dram_grid(
            max_x=8,
            max_y=8,
            num_channels=33,
            num_groups=3,
            input_nhw=32,
        )


# ---------------------------------------------------------------------------
# normalization.py  dram_group_norm_virtual_columns():
#   AssertionError — no valid num_virtual_cols (wraps compute_num_virtual_cols)
#
# No num_virtual_cols can satisfy (num_channels / nvc) % 32 == 0 and
# num_groups % nvc == 0 for grid_x=1, num_channels=48, num_groups=16, so the
# helper must raise instead of returning 0.
# ---------------------------------------------------------------------------
def test_dram_group_norm_virtual_columns_none_valid(expect_error):
    with expect_error(AssertionError, "could not find a valid num_virtual_cols for"):
        ttnn.operations.normalization.dram_group_norm_virtual_columns(
            core_grid=ttnn.CoreGrid(x=1, y=1),
            num_channels=48,
            num_groups=16,
        )


# ---------------------------------------------------------------------------
# groupnorm.cpp  validate_dram_grid():
#   TT_THROW — invalid core_grid for input dimensions; suggests largest valid sub-grid
#     (param id: invalid_grid_with_suggestion)
#   TT_THROW — invalid core_grid for padded input dimensions; suggests largest valid sub-grid
#     (param id: invalid_grid_for_padded_channels)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "input_shape, num_groups, core_grid, msg_pattern",
    [
        pytest.param(
            (1, 1, 64, 64),
            1,
            ttnn.CoreGrid(x=8, y=8),
            r"Requested core_grid .* is invalid for the input dimensions",
            id="invalid_grid_with_suggestion",
        ),
        pytest.param(
            (1, 1, 32, 48),
            16,
            ttnn.CoreGrid(x=32, y=1),
            r"Requested core_grid .* is invalid for the input dimensions",
            id="invalid_grid_for_padded_channels",
        ),
    ],
)
def test_validate_dram_grid(input_shape, num_groups, core_grid, msg_pattern, device, expect_error):
    x = ttnn.empty(input_shape, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, msg_pattern):
        ttnn.group_norm(x, num_groups=num_groups, core_grid=core_grid, inplace=False)


# ---------------------------------------------------------------------------
# groupnorm_mcast_program_factory.cpp / groupnorm_no_mcast_program_factory.cpp:
#   TT_FATAL — num_out_blocks must be in [1, block_h]
#     (param id: mcast_num_out_blocks_exceeds_block_ht — mcast factory path)
#     (param id: no_mcast_num_out_blocks_exceeds_block_ht — no-mcast factory path)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "input_shape, core_grid, num_out_blocks",
    [
        pytest.param(
            (1, 1, 256, 256),
            ttnn.CoreGrid(y=2, x=4),
            5,
            id="mcast_num_out_blocks_exceeds_block_ht",
        ),
        pytest.param(
            (2, 1, 64, 256),
            ttnn.CoreGrid(y=1, x=1),
            3,
            id="no_mcast_num_out_blocks_exceeds_block_ht",
        ),
    ],
)
def test_num_out_blocks_exceeds_block_ht(input_shape, core_grid, num_out_blocks, device, expect_error):
    x = ttnn.from_torch(
        torch.randn(*input_shape, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    with expect_error(RuntimeError, r"num_out_blocks.*must be in"):
        ttnn.group_norm(
            x,
            num_groups=32,
            core_grid=core_grid,
            num_out_blocks=num_out_blocks,
            inplace=False,
        )
