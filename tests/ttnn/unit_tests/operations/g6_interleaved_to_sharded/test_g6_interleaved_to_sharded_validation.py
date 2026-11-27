# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""A shard taller (width) or wider (height) than the tensor cannot be allocated (TensorSpec requires
equality on the unsharded dim), so validate() must refuse it as malformed before planning, instead
of letting allocate_tensor_on_device fail after a program was built and cached."""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded_program_descriptor as descriptor


def _config(scheme, grid_end, shard_shape):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(*grid_end))})
    shard_spec = ttnn.ShardSpec(grid, tuple(shard_shape), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(scheme, ttnn.BufferType.L1, shard_spec)


CASES = {
    # p1 tensor, width shards of 2 tile rows on a 1-tile-row tensor.
    "width_shard_taller_than_tensor": ([1, 5120], ttnn.TensorMemoryLayout.WIDTH_SHARDED, (7, 3), (64, 160)),
    # p2 tensor, a height shard of 8 tile columns on a 4-tile-column tensor.
    "height_shard_wider_than_tensor": ([1, 1, 1, 128], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (0, 0), (32, 256)),
}


@pytest.mark.parametrize("case", list(CASES))
def test_mismatched_shard_is_malformed(case, device, expect_error):
    shape, scheme, grid_end, shard_shape = CASES[case]
    ttnn_input = ttnn.from_torch(
        torch.randn(shape).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    entries = len(descriptor._PROGRAM_CACHE)
    with expect_error(ValueError, "does not match"):
        g6_interleaved_to_sharded(ttnn_input, _config(scheme, grid_end, shard_shape))
    assert len(descriptor._PROGRAM_CACHE) == entries, "a refused request built a program"
