# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Validation-failure tests for group_norm checks added by PR #39330.

This file runs on a CPU-only runner using TT-Sim (software simulator) in the
L2 nightly workflow.  Tests that call host-side C++ helpers directly do not
need a device at all; tests that go through ``ttnn.group_norm`` use the
simulated device provided by the ``device`` fixture under TT-Sim.

Covers the following checks:

  groupnorm_input_mask.cpp  create_group_norm_input_mask_impl():
    TT_FATAL  — num_cores_across_channel == 0
    TT_FATAL  — num_groups not divisible by num_cores_across_channel

  groupnorm_nanobind.cpp  _find_expected_dram_grid wrapper:
    RuntimeError  — no valid DRAM grid found

  groupnorm.cpp  get_mask_tensor():
    TT_FATAL  — compute_num_virtual_cols returns 0 (non-L1 buffer path)

  groupnorm.cpp  validate_dram_grid():
    TT_THROW  — invalid grid with a valid sub-grid suggestion
    TT_THROW  — no valid grid exists at all

  groupnorm_mcast_program_factory.cpp / groupnorm_no_mcast_program_factory.cpp:
    TT_FATAL  — num_out_blocks > block_ht
"""

import pytest
import torch

import ttnn
from ttnn._ttnn.operations.normalization import (
    create_group_norm_input_mask,
    _compute_num_virtual_cols,
    _find_expected_dram_grid,
)


# ===========================================================================
#  Pure CPU-only tests (no device fixture required)
# ===========================================================================


# ---------------------------------------------------------------------------
# create_group_norm_input_mask_impl: num_cores_across_channel must be > 0
# Source: groupnorm_input_mask.cpp
# ---------------------------------------------------------------------------
def test_input_mask_num_cores_across_channel_zero():
    with pytest.raises(RuntimeError, match="num_cores_across_channel must be > 0"):
        create_group_norm_input_mask(
            num_channel=256,
            num_groups=32,
            num_cores_across_channel=0,
            data_type=ttnn.DataType.BFLOAT16,
        )


# ---------------------------------------------------------------------------
# create_group_norm_input_mask_impl: num_groups % num_cores_across_channel != 0
# Source: groupnorm_input_mask.cpp
# ---------------------------------------------------------------------------
def test_input_mask_num_groups_not_divisible_by_num_cores():
    with pytest.raises(RuntimeError, match="must be divisible by num_cores_across_channel"):
        create_group_norm_input_mask(
            num_channel=256,
            num_groups=32,
            num_cores_across_channel=3,
            data_type=ttnn.DataType.BFLOAT16,
        )


# ---------------------------------------------------------------------------
# _find_expected_dram_grid: no valid grid exists
# Source: groupnorm_nanobind.cpp (RuntimeError wrapper around find_expected_dram_grid)
# ---------------------------------------------------------------------------
def test_find_expected_dram_grid_no_valid_grid():
    with pytest.raises(RuntimeError, match="Cannot find a valid DRAM group-norm grid"):
        _find_expected_dram_grid(
            max_x=8,
            max_y=8,
            num_channels=33,
            num_groups=3,
            input_nhw=32,
        )


# ===========================================================================
#  Device-dependent tests (use TT-Sim simulated device)
# ===========================================================================


# ---------------------------------------------------------------------------
# get_mask_tensor: compute_num_virtual_cols returns 0 (non-L1 buffer path)
# Source: groupnorm.cpp
# ---------------------------------------------------------------------------
def test_get_mask_tensor_num_virtual_cols_zero(device):
    torch_x = torch.randn(1, 1, 32, 48, dtype=torch.bfloat16)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        (32, 48),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_height_sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )
    x = ttnn.interleaved_to_sharded(
        ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT),
        dram_height_sharded,
    )

    with pytest.raises(RuntimeError, match="Cannot determine num_virtual_cols"):
        ttnn.group_norm(x, num_groups=16, core_grid=ttnn.CoreGrid(x=1, y=1), inplace=True)


# ---------------------------------------------------------------------------
# validate_dram_grid: invalid grid / no valid grid
# Source: groupnorm.cpp
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
            r"Cannot find any valid core grid",
            id="no_valid_grid",
        ),
    ],
)
def test_validate_dram_grid(input_shape, num_groups, core_grid, msg_pattern, device):
    x = ttnn.empty(input_shape, dtype=ttnn.DataType.BFLOAT16, device=device)
    with pytest.raises(RuntimeError, match=msg_pattern):
        ttnn.group_norm(x, num_groups=num_groups, core_grid=core_grid, inplace=False)


# ---------------------------------------------------------------------------
# num_out_blocks > block_ht in mcast and no_mcast program factories
# Source: groupnorm_mcast_program_factory.cpp, groupnorm_no_mcast_program_factory.cpp
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
def test_num_out_blocks_exceeds_block_ht(input_shape, core_grid, num_out_blocks, device):
    x = ttnn.from_torch(
        torch.randn(*input_shape, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    with pytest.raises(RuntimeError, match=r"num_out_blocks.*must be in"):
        ttnn.group_norm(
            x,
            num_groups=32,
            core_grid=core_grid,
            num_out_blocks=num_out_blocks,
            inplace=False,
        )
