# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Repro: sharded group_norm with a caller-supplied negative_mask is wrong when a core holds 1 or 2 tile rows.

Height-sharded ROW_MAJOR bf16 input, bfloat8_b input mask and negative mask. With a negative mask the
compute kernel rewrites the tilized input CB in place, and the rows that come out wrong equal the
correct result plus the original input (the negative-mask zeroing was not seen by the add).
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("grid_y, grid_x", [(1, 1), (8, 8)], ids=["1_core", "8x8_cores"])
@pytest.mark.parametrize("rows_per_core", [32, 64, 96, 128], ids=lambda r: f"rows_per_core={r}")
@pytest.mark.parametrize("use_negative_mask", [False, True], ids=["no_negative_mask", "negative_mask"])
def test_group_norm_negative_mask_short_shard(device, grid_y, grid_x, rows_per_core, use_negative_mask):
    if device.core_grid.y < grid_y or device.core_grid.x < grid_x:
        pytest.skip(f"device grid is smaller than {grid_x}x{grid_y}")
    torch.manual_seed(0)
    N, C, G = 1, 128, 8
    num_cores = grid_y * grid_x
    HW = rows_per_core * num_cores

    x = torch.rand((N, C, 1, HW), dtype=torch.bfloat16)
    ref = torch.nn.functional.group_norm(x.float(), G).permute(0, 2, 3, 1).reshape(N, 1, HW, C)

    tt_input = ttnn.from_torch(
        x.permute(0, 2, 3, 1).reshape(N, 1, HW, C),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    shard_spec = ttnn.ShardSpec(shard_grid, (N * HW // num_cores, C), ttnn.ShardOrientation.ROW_MAJOR)
    sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    tt_input = ttnn.to_memory_config(tt_input, sharded)

    input_mask = ttnn.to_device(ttnn.create_group_norm_input_mask(C, G, 1, ttnn.bfloat8_b), device)
    negative_mask = (
        ttnn.to_device(ttnn.create_group_norm_input_negative_mask(C, G, 1, ttnn.bfloat8_b), device)
        if use_negative_mask
        else None
    )

    tt_output = ttnn.group_norm(
        tt_input,
        num_groups=G,
        input_mask=input_mask,
        negative_mask=negative_mask,
        memory_config=sharded,
        core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
        inplace=False,
    )
    out = ttnn.to_torch(ttnn.from_device(ttnn.to_memory_config(tt_output, ttnn.DRAM_MEMORY_CONFIG))).float()

    assert_with_pcc(ref, out, 0.999)
