# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_equal, assert_with_ulp


def test_transpose_with_reshape(device):
    # Create input tensor
    torch_input = torch.rand((1, 1, 2048, 512), dtype=torch.bfloat16)

    # TT operations
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_input = tt_input.reshape(1, 2048, 4, 128)
    tt_output = ttnn.transpose(tt_input, 1, 2)

    # Convert back to PyTorch for comparison
    tt_result = ttnn.to_torch(tt_output)

    # PyTorch reference operations
    torch_ref = torch_input.view(1, 2048, 4, 128)
    torch_ref = torch_ref.transpose(1, 2)

    # Compare results
    assert_equal(torch_ref, tt_result)


# ttnn.transpose RM WH must fall back to permute whenever the WH factory would overflow L1.
TILE_DIM, TILE_BYTES, ELEM_BYTES = 32, 2048, 2  # bfloat16


def _run_height_sharded_rm_transpose(device, shape, num_cores):
    N, C, H, W = shape
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    in_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, (N * C * H // num_cores, W), ttnn.ShardOrientation.ROW_MAJOR),
    )
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_in = ttnn.to_memory_config(ttnn_in, in_mc)
    result = ttnn.transpose(ttnn_in, 2, 3)
    got = ttnn.to_torch(result.cpu())
    assert_with_ulp(expected_result=x.transpose(2, 3), actual_result=got, ulp_threshold=0)


def test_transpose_wh_sharded_rm_l1_budget_counts_tiles(device):
    """The interleaved WH factory's streaming CBs must be counted in tiles, not elements."""
    H, num_cores = 64, 2
    Ht = H // TILE_DIM
    cb_limit = ttnn._ttnn.reports.get_device_info(device).cb_limit

    # Input shard (H / num_cores, W) is resident; the promoted WIDTH_SHARDED output shard (W, H / num_cores) is the
    # same size and is reserved by the budget too. Both bounds include it, so output reservation alone can never
    # trigger the fallback: only the tile-based CB estimate separates the old and real routes.
    shard_slope = (H // num_cores) * ELEM_BYTES
    old_slope = (2 + H) * ELEM_BYTES  # (2W + 2H + H*W) elements
    old_const = 2 * H * ELEM_BYTES
    real_slope = (2 + Ht) * TILE_BYTES // TILE_DIM  # (2Wt + 2Ht + Ht*Wt) tiles
    real_const = 2 * Ht * TILE_BYTES
    w_old_fits = (cb_limit - old_const) // (old_slope + 2 * shard_slope)
    w_real_overflows = (cb_limit - real_const) // (real_slope + 2 * shard_slope)
    W = (w_old_fits + w_real_overflows) // 2 // TILE_DIM * TILE_DIM

    Wt = W // TILE_DIM
    free_l1 = cb_limit - shard_slope * W
    output_shard = shard_slope * W
    old_estimate = old_slope * W + old_const
    real_estimate = (2 * Wt + 2 * Ht + Ht * Wt) * TILE_BYTES
    if not old_estimate + output_shard < free_l1 < real_estimate + output_shard:
        pytest.skip(f"No W separates the estimates on this device (cb_limit={cb_limit})")

    _run_height_sharded_rm_transpose(device, (1, 1, H, W), num_cores)


def test_transpose_wh_sharded_rm_l1_budget_reserves_l1_output(device):
    """An L1-sharded output is allocated after the budget samples free L1, so it must be reserved up front."""
    H, num_cores = 64, 2
    Ht = H // TILE_DIM
    cb_limit = ttnn._ttnn.reports.get_device_info(device).cb_limit

    # Input and promoted output shards each take (H / num_cores) * W bytes per core.
    shard_bytes_per_wt = (H // num_cores) * TILE_DIM * ELEM_BYTES
    cb_bytes_per_wt = (2 + Ht) * TILE_BYTES
    cb_const = 2 * Ht * TILE_BYTES
    # CBs alone fit next to the input shard, but not once the output shard is also placed.
    wt_lo = (cb_limit - cb_const) // (cb_bytes_per_wt + 2 * shard_bytes_per_wt) + 1
    wt_hi = (cb_limit - cb_const) // (cb_bytes_per_wt + shard_bytes_per_wt)
    Wt = (wt_lo + wt_hi) // 2

    free_l1 = cb_limit - shard_bytes_per_wt * Wt
    cbs = cb_bytes_per_wt * Wt + cb_const
    if not cbs < free_l1 < cbs + shard_bytes_per_wt * Wt:
        pytest.skip(f"No W separates the routes on this device (cb_limit={cb_limit})")

    _run_height_sharded_rm_transpose(device, (1, 1, H, Wt * TILE_DIM), num_cores)


def test_transpose_wh_sharded_rm_factory_l1_budget_counts_untilize_cbs(device):
    """For Ht > 8 the native sharded RM factory adds untilize + staging CBs (3*Ht tiles), not 2*Ht."""
    N, C, num_cores = 1, 2, 2
    cb_limit = ttnn._ttnn.reports.get_device_info(device).cb_limit

    def required(H, W, ht_tiles):
        Ht, Wt = -(-H // TILE_DIM), -(-W // TILE_DIM)
        # Input and output shards are both H*W per core; only the output is unallocated at check time.
        return (2 * Wt + Ht * Wt + ht_tiles) * TILE_BYTES + H * W * ELEM_BYTES

    def in_band(H, W):
        Ht = -(-H // TILE_DIM)
        # The native sharded RM path needs whole-tile shards; W % 8 keeps RM sticks 16B-aligned.
        native = (H * W) % (TILE_DIM * TILE_DIM) == 0
        free_l1 = cb_limit - H * W * ELEM_BYTES
        return native and required(H, W, 2 * Ht) <= free_l1 < required(H, W, 3 * Ht)

    shape = next(
        ((H, W) for H in range(9 * TILE_DIM, 17 * TILE_DIM, TILE_DIM) for W in range(8, 4096, 8) if in_band(H, W)),
        None,
    )
    if shape is None:
        pytest.skip(f"No shape separates the estimates on this device (cb_limit={cb_limit})")

    _run_height_sharded_rm_transpose(device, (N, C, *shape), num_cores)
