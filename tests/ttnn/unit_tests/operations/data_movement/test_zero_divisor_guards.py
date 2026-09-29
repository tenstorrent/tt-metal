# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


# Each of these zero arguments used to reach an integer division on the host, which killed the
# process with SIGFPE instead of raising.


def _rm(device, shape, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        torch.randn(shape).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _height_sharded(shape, shard_shape):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


def _fold(device):
    ttnn.fold(_rm(device, (1, 8, 8, 32)), 0, 1)


def _fold_transpose(device):
    ttnn.fold(
        _rm(device, (1, 8, 8, 32)), 0, 1, use_transpose_as_fold=True, output_shape=(1, 8, 8, 32), padding=[0, 0, 0, 0]
    )


def _slice(device):
    ttnn.slice(_rm(device, (1, 1, 64, 64)), [0, 0, 0, 0], [1, 1, 64, 64], [1, 1, 0, 1])


def _slice_write(device):
    ttnn.experimental.slice_write(
        _rm(device, (1, 8, 8, 32)), _rm(device, (1, 16, 8, 32)), [0, 0, 0, 0], [1, 8, 8, 32], [1, 0, 1, 1]
    )


def _concat(device):
    mem = _height_sharded((1, 1, 32, 64), (32, 64))
    out_mem = _height_sharded((1, 1, 32, 128), (32, 128))
    a = _rm(device, (1, 1, 32, 64), mem)
    b = _rm(device, (1, 1, 32, 64), mem)
    ttnn.concat([a, b], dim=-1, memory_config=out_mem, groups=0)


def _interleaved_to_sharded(device):
    x = ttnn.from_torch(torch.randn(1, 1, 64, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.interleaved_to_sharded(
        x, ttnn.CoreCoord(1, 1), [0, 64], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.ShardOrientation.ROW_MAJOR
    )


def _interleaved_to_sharded_partial(device):
    x = ttnn.from_torch(torch.randn(1, 1, 64, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.interleaved_to_sharded_partial(
        x,
        ttnn.CoreCoord(1, 1),
        [32, 64],
        0,
        0,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )


@pytest.mark.parametrize(
    "run, message",
    [
        (_fold, "stride_h and stride_w must be greater than 0"),
        (_fold_transpose, "stride_h and stride_w must be greater than 0"),
        (_slice, "Slice step must be greater than 0"),
        (_slice_write, "Step must be greater than 0"),
        (_concat, "groups must be greater than 0"),
        (_interleaved_to_sharded, "shard_shape must be greater than 0"),
        (_interleaved_to_sharded_partial, "num_slices must be greater than 0"),
    ],
    ids=["fold", "fold_transpose", "slice_step", "slice_write_step", "concat_groups", "i2s_shard_shape", "i2s_partial"],
)
def test_zero_divisor_raises(device, expect_error, run, message):
    with expect_error(RuntimeError, message):
        run(device)
