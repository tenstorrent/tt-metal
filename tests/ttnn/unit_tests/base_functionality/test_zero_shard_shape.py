# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


# A shard shape with a 0 in a sharded dim used to reach an integer division while the op built its
# output TensorSpec, which killed the process with SIGFPE instead of raising. Any op that takes a
# sharded memory_config hits the same check; add, embedding and to_memory_config are three of them.


def _one_core_shard(layout, shard_shape):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    return ttnn.MemoryConfig(
        layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    )


@pytest.mark.parametrize(
    "layout, shard_shape",
    [
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (0, 64)),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (64, 0)),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (0, 64)),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (64, 0)),
    ],
    ids=["height", "width", "block_zero_height", "block_zero_width"],
)
def test_add_zero_shard_dim(device, expect_error, layout, shard_shape):
    x = ttnn.from_torch(torch.randn(1, 1, 64, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Shard shape must be greater than 0 in each sharded dim"):
        ttnn.add(x, x, memory_config=_one_core_shard(layout, shard_shape))


def test_embedding_zero_shard_shape(device, expect_error):
    indices = ttnn.from_torch(torch.randint(0, 32, (1, 32)), dtype=ttnn.uint32, device=device)
    weights = ttnn.from_torch(torch.randn(32, 64).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "Shard shape must be greater than 0 in each sharded dim"):
        ttnn.embedding(
            indices,
            weights,
            layout=ttnn.TILE_LAYOUT,
            memory_config=_one_core_shard(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (0, 64)),
        )


# A row-major tensor takes its width alignment from the shard width, so a 0 width fails while the
# TensorLayout is built, ahead of the TensorSpec check.
@pytest.mark.parametrize(
    "layout", [ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED], ids=["width", "block"]
)
def test_row_major_zero_shard_width(device, expect_error, layout):
    with expect_error(RuntimeError, "Row Major width alignment must be greater than 0"):
        ttnn.from_torch(
            torch.randn(1, 1, 64, 64).bfloat16(),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=_one_core_shard(layout, (64, 0)),
        )


def _one_core_nd_shard(shard_shape):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    return ttnn.MemoryConfig(ttnn.BufferType.L1, ttnn.NdShardSpec(list(shard_shape), grid))


# An ND shard shape with a 0 dim reached `% 0` / `div_up(x, 0)` while TensorSpec derived the legacy shard spec.
@pytest.mark.parametrize("shard_shape", [(32, 0), (0, 32)], ids=["zero_width", "zero_height"])
def test_nd_zero_shard_dim(device, expect_error, shard_shape):
    with expect_error(RuntimeError, "ND shard shape must be greater than 0 in every dim"):
        ttnn.from_torch(
            torch.randn(1, 1, 64, 64).bfloat16(),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_one_core_nd_shard(shard_shape),
        )


# A zero last dim with a valid ND shard spec reached 0 / 0 while classifying the legacy shard layout.
def test_nd_sharded_zero_width_tensor_spec():
    spec = ttnn.TensorSpec((32, 0), ttnn.bfloat16, ttnn.TILE_LAYOUT, memory_config=_one_core_nd_shard((32, 32)))
    assert list(spec.shape) == [32, 0]


# A legacy shard spec with a 0 in its unsharded dim is valid, and so is reusing the spec built from it,
# even though that spec carries a derived ND shard shape with a 0.
@pytest.mark.parametrize(
    "layout, tensor_shape, shard_shape",
    [
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (0, 32), (0, 32)),
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (32, 0), (32, 0)),
    ],
    ids=["width_sharded", "height_sharded"],
)
def test_legacy_zero_shard_spec_can_be_reused(layout, tensor_shape, shard_shape):
    spec = ttnn.TensorSpec(
        tensor_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, memory_config=_one_core_shard(layout, shard_shape)
    )
    reused = ttnn.TensorSpec(spec.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, memory_config=spec.memory_config)
    assert list(reused.shape) == list(tensor_shape)


# Reshaping a row-major sharded tensor to a zero last dim divided the shard volume by it.
def test_reshape_sharded_to_zero_last_dim(device, expect_error):
    x = ttnn.from_torch(
        torch.zeros(0, 32).bfloat16(),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_one_core_shard(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (32, 32)),
    )
    with expect_error(RuntimeError, "Cannot reshape a sharded tensor to a zero last dimension"):
        ttnn.reshape(x, [32, 0])
