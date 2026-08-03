# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke coverage for the quasar generate_transpose_shard_spec cross-port; guards drift from the DM twin."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp


L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)


def test_quasar_transpose_specless_sharded_output_grid_shrinks_height(device):
    """HEIGHT_SHARDED no-spec output on quasar twin: ceil(tensor_h / shard_h) populated cores.
    shape=(2,2,32,64) WH → out=(2,2,64,32); tensor_h=256, shard_h=32 → 8 populated cores."""
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
