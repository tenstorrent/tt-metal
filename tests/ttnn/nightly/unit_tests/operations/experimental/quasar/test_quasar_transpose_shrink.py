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
    assert_with_ulp(expected_result=ref, actual_result=got, ulp_threshold=0)


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
    assert_with_ulp(expected_result=ref, actual_result=got, ulp_threshold=0)


def test_quasar_transpose_wh_sharded_rm_l1_budget_counts_tiles(device):
    """RM WH CB budget must count the streaming CBs in tiles, as TransposeWHProgramFactory allocates them.

    W is sized so the old element-count estimate fits in free L1 while the real allocation does not;
    the op must then fall back to permute instead of overflowing L1 at CB allocation.
    """
    tile_dim, tile_bytes, elem_bytes = 32, 2048, 2  # bfloat16
    H, num_cores = 64, 2
    Ht = H // tile_dim
    cb_limit = ttnn._ttnn.reports.get_device_info(device).cb_limit

    # Free L1 on a shard core is cb_limit minus the resident input shard, which is (H / num_cores) * W * elem_bytes.
    # Both estimates are affine in W: bytes = slope * W + const, so each bound on W is solvable in closed form.
    shard_slope = (H // num_cores) * elem_bytes
    old_slope = (2 + H) * elem_bytes  # (2W + 2H + H*W) elements
    old_const = 2 * H * elem_bytes
    real_slope = (2 + Ht) * tile_bytes // tile_dim  # (2Wt + 2Ht + Ht*Wt) tiles
    real_const = 2 * Ht * tile_bytes
    # Old estimate fits free L1 below w_old_fits; real allocation overflows it above w_real_overflows.
    w_old_fits = (cb_limit - old_const) // (old_slope + shard_slope)
    w_real_overflows = (cb_limit - real_const) // (real_slope + shard_slope)
    W = (w_old_fits + w_real_overflows) // 2 // tile_dim * tile_dim

    Wt = W // tile_dim
    free_l1 = cb_limit - shard_slope * W
    elementwise_estimate = old_slope * W + old_const
    allocated = (2 * Wt + 2 * Ht + Ht * Wt) * tile_bytes
    if not elementwise_estimate < free_l1 < allocated:
        pytest.skip(f"No W separates the estimates on this device (cb_limit={cb_limit})")

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    in_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, (H // num_cores, W), ttnn.ShardOrientation.ROW_MAJOR),
    )
    torch.manual_seed(0)
    x = torch.rand((1, 1, H, W), dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_in = ttnn.to_memory_config(ttnn_in, in_mc)
    result = ttnn.experimental.quasar.transpose(ttnn_in, 2, 3)
    got = ttnn.to_torch(result.cpu())
    assert_with_ulp(expected_result=x.transpose(2, 3), actual_result=got, ulp_threshold=0)
