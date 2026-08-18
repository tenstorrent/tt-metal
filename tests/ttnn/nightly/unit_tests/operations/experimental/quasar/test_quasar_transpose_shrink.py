# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Guards drift of the quasar generate_transpose_shard_spec twin from the DM helper."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp


L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)


def test_quasar_transpose_specless_sharded_output_grid_shrinks_height(device):
    """HEIGHT_SHARDED no-spec: baseline that the shrink is wired up at all on the quasar op."""
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x * compute_grid.y <= 8:
        pytest.skip("Device grid too small to observe shrink (need > 8 cores)")
    shape = (2, 2, 32, 64)
    out_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    torch.manual_seed(12345)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(
        x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=L1_INTERLEAVED
    )
    result = ttnn.experimental.quasar.transpose(ttnn_in, 2, 3, memory_config=out_mc)
    grid = result.memory_config().shard_spec.grid
    assert grid.num_cores() == 8, f"Expected 8 populated cores, got {grid.num_cores()}"
    expected = ttnn.num_cores_to_corerangeset(8, compute_grid, True)
    assert grid == expected, f"Expected row-wise CoreRangeSet {expected}, got {grid}"
    ref = x.transpose(2, 3)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_with_ulp(ref, got, ulp_threshold=0)


def test_quasar_transpose_specless_sharded_output_grid_shrinks_block_col_major(device):
    """BLOCK+COL_MAJOR mirror of the DM 6-core case; catches BLOCK divisor/TT_FATAL drift on the port."""
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x < 2 or compute_grid.y < 3:
        pytest.skip("Device grid too small for COL_MAJOR 2x3 BLOCK shrink test")
    shape = (1, 1, 96, 64)
    in_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            (64, 32),
            ttnn.ShardOrientation.COL_MAJOR,
        ),
    )
    out_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
    torch.manual_seed(12345)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=in_mc)
    result = ttnn.experimental.quasar.transpose(ttnn_in, 2, 3, memory_config=out_mc)
    ss = result.memory_config().shard_spec
    assert ss.orientation == ttnn.ShardOrientation.COL_MAJOR, f"Expected COL_MAJOR, got {ss.orientation}"
    assert ss.shape[0] == 32 and ss.shape[1] == 32, f"Expected shard=(32,32), got ({ss.shape[0]},{ss.shape[1]})"
    assert ss.grid.num_cores() == 6, f"Expected 2x3=6 populated cores, got {ss.grid.num_cores()}"
    expected = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 2))})
    assert ss.grid == expected, f"Expected COL_MAJOR rect (0,0)->(1,2), got {ss.grid}"
    ref = x.transpose(2, 3)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_with_ulp(ref, got, ulp_threshold=0)
