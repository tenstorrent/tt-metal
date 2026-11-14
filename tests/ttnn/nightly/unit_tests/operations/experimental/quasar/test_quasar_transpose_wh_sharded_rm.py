# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sharded row-major WH transpose (TransposeWHShardedRMProgramFactory), including output rows narrower than a tile.

Runs both the mainline op and its experimental/quasar copy, on Quasar only. The WH/BH sharded row-major path is
covered by tests/tt_eager/python_api_testing/unit_testing/misc/test_transpose.py."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp
from tt_py_test_utils_common.utility_functions import is_quasar

pytestmark = pytest.mark.skipif(not is_quasar(), reason="Quasar test")


def _height_sharded(device, num_cores, shard_shape):
    grid = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True)
    shard_spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


@pytest.mark.parametrize(
    "n, c, h, w",
    [
        (1, 4, 8, 64),  # 8-wide output rows
        (1, 4, 16, 64),  # 16-wide output rows (the ResNet fold width)
        (1, 4, 24, 128),  # 24-wide: not a face width
        (1, 4, 32, 64),  # tile-aligned output rows
        (1, 4, 40, 64),  # one full tile plus 8
        (1, 4, 16, 160),  # 5 tiles wide
        (1, 4, 32, 224),  # 7 tiles wide (the ResNet fold width)
    ],
    ids=["h8", "h16", "h24", "h32", "h40", "w160", "w224"],
)
@pytest.mark.parametrize("transpose", [ttnn.transpose, ttnn.experimental.quasar.transpose], ids=["main", "quasar"])
def test_quasar_transpose_wh_sharded_rm(device, transpose, n, c, h, w):
    # Shards hold a whole number of tiles' worth of elements, so the op takes the sharded factory rather than the
    # interleaved fallback.
    # Two cores, two HxW blocks per core, so each core runs more than one block.
    num_cores = 2
    blocks_per_core = n * c // num_cores
    in_mc = _height_sharded(device, num_cores, (blocks_per_core * h, w))
    out_mc = _height_sharded(device, num_cores, (blocks_per_core * w, h))

    torch.manual_seed(0)
    x = torch.randn((n, c, h, w), dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)
    tt_out = transpose(tt_in, 2, 3, memory_config=out_mc)

    assert tt_out.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert_with_ulp(expected_result=x.transpose(2, 3), actual_result=ttnn.to_torch(tt_out), ulp_threshold=0)


@pytest.mark.parametrize("case", ["width_sharded_output", "orientation_mismatch"])
@pytest.mark.parametrize("transpose", [ttnn.transpose, ttnn.experimental.quasar.transpose], ids=["main", "quasar"])
def test_quasar_transpose_wh_sharded_rm_fallback(device, transpose, case):
    # Shardings the sharded factory cannot write (a partial HxW block per shard, or input and output shards
    # assigned to cores in different orders) take the interleaved factory instead.
    grid_size = device.compute_with_storage_grid_size()
    if grid_size.x < 2 or grid_size.y < 2:
        pytest.skip("Needs a 2x2 core grid")
    if case == "width_sharded_output":
        x_shape = (1, 1, 64, 64)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        in_spec = ttnn.ShardSpec(grid, (32, 64), ttnn.ShardOrientation.ROW_MAJOR)
        out_layout, out_spec = ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.ShardSpec(
            grid, (64, 32), ttnn.ShardOrientation.ROW_MAJOR
        )
    else:
        x_shape = (1, 4, 32, 64)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        in_spec = ttnn.ShardSpec(grid, (32, 64), ttnn.ShardOrientation.ROW_MAJOR)
        out_layout, out_spec = ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.ShardSpec(
            grid, (64, 32), ttnn.ShardOrientation.COL_MAJOR
        )
    in_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in_spec)
    out_mc = ttnn.MemoryConfig(out_layout, ttnn.BufferType.L1, out_spec)

    # A ramp, so misplaced or unwritten rows show up as out-of-order values.
    x = (torch.arange(torch.Size(x_shape).numel()) % 251).reshape(x_shape).to(torch.bfloat16)
    tt_in = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)
    tt_out = transpose(tt_in, 2, 3, memory_config=out_mc)

    assert_with_ulp(expected_result=x.transpose(2, 3), actual_result=ttnn.to_torch(tt_out), ulp_threshold=0)
