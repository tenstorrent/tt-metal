# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn


def assert_binary_shrink_h_or_w(device, result_mc, n_used):
    """Row-wise shrink: populated cores == n_used and CoreRangeSet == num_cores_to_corerangeset(n_used)."""
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x * compute_grid.y <= n_used:
        pytest.skip(f"Device grid too small to observe shrink (need > {n_used} cores)")
    grid = result_mc.shard_spec.grid
    assert grid.num_cores() == n_used, f"Expected {n_used} populated cores, got {grid.num_cores()}"
    expected = ttnn.num_cores_to_corerangeset(n_used, compute_grid, True)
    assert grid == expected, f"Expected row-wise CoreRangeSet {expected}, got {grid}"


def binary_add(a, b, device, out_mc):
    """ttnn.add(a, b) helper; late-binds ttnn.add so quasar's autouse monkeypatch takes effect at call time."""
    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.add(tt_a, tt_b, memory_config=out_mc)
