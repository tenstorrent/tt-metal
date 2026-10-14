# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sharded row-major WH transpose (TransposeWHShardedRMProgramFactory), including output rows narrower than a tile."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp


def _height_sharded(device, num_cores, shard_shape):
    grid = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True)
    shard_spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


@pytest.mark.parametrize(
    "n, c, h, w",
    [
        (1, 4, 8, 64),  # 8-wide output rows
        (1, 4, 16, 64),  # 16-wide output rows (the ResNet fold width)
        (1, 4, 24, 96),  # 24-wide: not a face width
        (1, 4, 32, 64),  # tile-aligned output rows
        (1, 4, 40, 64),  # one full tile plus 8
    ],
    ids=["h8", "h16", "h24", "h32", "h40"],
)
def test_quasar_transpose_wh_sharded_rm(device, n, c, h, w):
    is_quasar = device.arch() == ttnn.device.Arch.QUASAR
    if not is_quasar and h > 32 and h % 32 != 0:
        # The narrow-row pack strides rows by Ht * (H % 32) datums instead of H.
        pytest.skip("WH/BH narrow-row transpose is only correct for H < 32")
    # Two cores, two HxW blocks per core, so each core runs more than one block.
    num_cores = 2
    blocks_per_core = n * c // num_cores
    in_mc = _height_sharded(device, num_cores, (blocks_per_core * h, w))
    out_mc = _height_sharded(device, num_cores, (blocks_per_core * w, h))

    torch.manual_seed(0)
    x = torch.randn((n, c, h, w), dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)
    tt_out = ttnn.experimental.quasar.transpose(tt_in, 2, 3, memory_config=out_mc)

    assert tt_out.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert_with_ulp(expected_result=x.transpose(2, 3), actual_result=ttnn.to_torch(tt_out), ulp_threshold=0)
