# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""cb_descriptor_from_sharded_tensor on ND-sharded tensors (https://github.com/tenstorrent/tt-metal/issues/46230).

A shard spec exists only when the sharding has a 2D form, so the default core range of an ND-sharded tensor must come
from its ND shard grid."""

import pytest
import torch

import ttnn


def _grid(x, y):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))])


def _nd_sharded(device, shape, shard_shape, grid, layout):
    memory_config = ttnn.MemoryConfig(ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape(shard_shape), grid))
    return ttnn.from_torch(
        torch.randn(shape), dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config
    )


@pytest.mark.parametrize(
    "shape, shard_shape, layout",
    [
        ((4, 64, 64), (1, 32, 32), ttnn.TILE_LAYOUT),
        ((2, 3, 64, 96), (1, 2, 32, 64), ttnn.TILE_LAYOUT),
        ((4, 8, 64), (2, 4, 64), ttnn.ROW_MAJOR_LAYOUT),
    ],
)
def test_cb_descriptor_from_nd_sharded_tensor(device, shape, shard_shape, layout):
    grid = _grid(2, 2)
    tensor = _nd_sharded(device, shape, shard_shape, grid, layout)
    assert tensor.memory_config().shard_spec is None, "case must exercise sharding with no 2D shard spec"

    desc = ttnn.cb_descriptor_from_sharded_tensor(0, tensor)
    explicit = ttnn.cb_descriptor_from_sharded_tensor(0, tensor, core_ranges=grid)

    assert desc.core_ranges == grid
    assert desc.total_size == explicit.total_size
    assert desc.total_size > 0


def test_cb_descriptor_core_ranges_override_on_nd_sharded_tensor(device):
    tensor = _nd_sharded(device, (4, 64, 64), (1, 32, 32), _grid(2, 2), ttnn.TILE_LAYOUT)
    override = _grid(1, 1)

    desc = ttnn.cb_descriptor_from_sharded_tensor(0, tensor, core_ranges=override)

    assert desc.core_ranges == override


def test_cb_descriptor_from_2d_sharded_tensor_unchanged(device):
    grid = _grid(2, 2)
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (64, 64), ttnn.ShardOrientation.ROW_MAJOR),
    )
    tensor = ttnn.from_torch(
        torch.randn(1, 1, 256, 64),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )

    desc = ttnn.cb_descriptor_from_sharded_tensor(0, tensor)

    assert desc.core_ranges == grid
