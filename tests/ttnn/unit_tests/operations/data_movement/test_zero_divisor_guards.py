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


def _interleaved_to_sharded_memory_config(device):
    x = _rm(device, (1, 1, 64, 64))
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (64, 0), ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn.interleaved_to_sharded(x, mem)


def _interleaved_to_sharded_partial(device, shard_shape=(32, 64), num_slices=0):
    x = ttnn.from_torch(torch.randn(1, 1, 64, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.interleaved_to_sharded_partial(
        x,
        ttnn.CoreCoord(1, 1),
        list(shard_shape),
        num_slices,
        0,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )


def _interleaved_to_sharded_partial_shard_shape(device):
    _interleaved_to_sharded_partial(device, shard_shape=(0, 64), num_slices=1)


def _interleaved_to_sharded_partial_wraps(device):
    _interleaved_to_sharded_partial(device, num_slices=2**32)


def _slice_tensor_args_num_devices(device):
    x = ttnn.from_torch(torch.randn(1, 1, 64, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    starts = ttnn.from_torch(torch.tensor([0, 0, 0, 0]), device=device)
    ends = ttnn.from_torch(torch.tensor([1, 1, 32, 64]), device=device)
    ttnn.slice(x, starts, ends, slice_dim=2, num_devices=0)


def _embedding_sharded_zero_shard(device):
    idx = ttnn.from_torch(torch.randint(0, 64, (1, 1, 1, 32)), dtype=ttnn.uint32, device=device)
    weights = ttnn.from_torch(torch.randn(1, 1, 64, 64).bfloat16(), device=device)
    out = ttnn.from_torch(torch.zeros(1, 1, 32, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.embedding(
        idx,
        weights,
        layout=ttnn.TILE_LAYOUT,
        memory_config=_height_sharded((1, 1, 32, 64), (0, 32)),
        output_tensor=out,
    )


def _reshape_volume_wraps(device):
    ttnn.reshape(_rm(device, (1, 1, 32, 32)), [-1, 65536, 65536, 65536, 65536])


def _interleaved_to_sharded_zero_width(device):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 0).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.interleaved_to_sharded(
        x, ttnn.CoreCoord(1, 1), [32, 32], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.ShardOrientation.ROW_MAJOR
    )


def _interleaved_to_sharded_partial_zero_width(device):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 0).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.interleaved_to_sharded_partial(
        x, ttnn.CoreCoord(1, 1), [32, 32], 1, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.ShardOrientation.ROW_MAJOR
    )


def _sharded_to_interleaved_partial_zero_width_cache(device):
    mem = _height_sharded((1, 1, 32, 32), (32, 32))
    x = ttnn.from_torch(torch.randn(1, 1, 32, 32).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    cache = ttnn.from_torch(torch.zeros(1, 1, 32, 0).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.sharded_to_interleaved_partial(x, cache, 1, 0)


@pytest.mark.parametrize(
    "run, message",
    [
        (_fold, "stride_h and stride_w must be greater than 0"),
        (_fold_transpose, "stride_h and stride_w must be greater than 0"),
        (_slice, "Slice step must be greater than 0"),
        (_slice_write, "Step must be greater than 0"),
        (_concat, "groups must be greater than 0"),
        (_interleaved_to_sharded, "shard_shape must be greater than 0"),
        (_interleaved_to_sharded_memory_config, "shard_shape must be greater than 0"),
        (_interleaved_to_sharded_partial, "num_slices must be greater than 0"),
        (_interleaved_to_sharded_partial_shard_shape, "shard_shape must be greater than 0"),
        (_interleaved_to_sharded_partial_wraps, "num_slices must be at most UINT32_MAX"),
        (_slice_tensor_args_num_devices, "num_devices must be greater than 0"),
        (_embedding_sharded_zero_shard, "Shard shape must be non-zero"),
        (_reshape_volume_wraps, "Invalid arguments to reshape"),
        (_interleaved_to_sharded_zero_width, "Input tensor width must be greater than 0"),
        (_interleaved_to_sharded_partial_zero_width, "Input tensor width must be greater than 0"),
        (_sharded_to_interleaved_partial_zero_width_cache, "Cache tensor width must be greater than 0"),
    ],
    ids=[
        "fold",
        "fold_transpose",
        "slice_step",
        "slice_write_step",
        "concat_groups",
        "i2s_shard_shape",
        "i2s_memory_config",
        "i2s_partial",
        "i2s_partial_shard_shape",
        "i2s_partial_num_slices_wraps",
        "slice_tensor_args_num_devices",
        "embedding_zero_shard",
        "reshape_volume_wraps",
        "i2s_zero_width",
        "i2s_partial_zero_width",
        "s2i_partial_zero_width_cache",
    ],
)
def test_zero_divisor_raises(device, expect_error, run, message):
    with expect_error(RuntimeError, message):
        run(device)


# Rolling an empty tensor is a no-op; it used to take `shift % 0`.
@pytest.mark.parametrize("dims", [(0,), None], ids=["dim0", "flattened"])
def test_roll_empty_tensor(device, dims):
    x = _rm(device, (0, 32))
    out = ttnn.roll(x, 1, dims[0]) if dims else ttnn.roll(x, 1)
    assert tuple(out.shape) == (0, 32)


# A zero-height height-sharded input took `% 0` while choosing the transpose factory and `Hts / Ht` in it.
def test_transpose_empty_height_sharded(device, expect_error):
    mem = _height_sharded((1, 1, 0, 32), (32, 32))
    x = ttnn.from_torch(torch.zeros(1, 1, 0, 32).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    with expect_error(RuntimeError, "Sharded WH transpose requires non-zero H and W"):
        ttnn.transpose(x, -2, -1)
