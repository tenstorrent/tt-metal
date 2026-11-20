# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

# Each of these zero-volume inputs used to reach an integer division by an empty dim on the host (SIGFPE), index an
# empty core list (SIGSEGV), fail to compile a kernel, or read input pages that do not exist. torch returns an empty
# result for the zero-volume ones, and pad of an empty tensor is all padding.

TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _grid(x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x1, y1))})


def _shards(layout, x1, y1, shape):
    return ttnn.MemoryConfig(
        layout, ttnn.BufferType.L1, ttnn.ShardSpec(_grid(x1, y1), shape, ttnn.ShardOrientation.ROW_MAJOR)
    )


HEIGHT_SHARDED = _shards(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 1, 0, [32, 64])
BLOCK_SHARDED = _shards(ttnn.TensorMemoryLayout.BLOCK_SHARDED, 1, 1, [32, 32])


def _to_device(device, src, layout=TILE, memory_config=DRAM, dtype=ttnn.bfloat16):
    return ttnn.from_torch(src, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _ints(shape, offset=0):
    n = 1
    for d in shape:
        n *= d
    return (torch.arange(n) % 251 - 125 + offset).reshape(shape).to(torch.bfloat16)


@pytest.mark.parametrize("shape", [[1, 1, 0, 16], [1, 1, 16, 0]], ids=["h0", "w0"])
def test_transpose_row_major_zero_volume(device, shape):
    out = ttnn.transpose(_to_device(device, _ints(shape), RM), -2, -1)
    assert list(out.shape) == [shape[0], shape[1], shape[3], shape[2]]


def test_permute_row_major_zero_last_dim_kept(device):
    out = ttnn.permute(_to_device(device, _ints([1, 2, 16, 0]), RM), (0, 2, 1, 3))
    assert list(out.shape) == [1, 16, 2, 0]


@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
def test_concat_last_dim_zero_height(device, layout):
    xs = [_to_device(device, _ints([1, 1, 0, 16], i), layout) for i in range(2)]
    assert list(ttnn.concat(xs, dim=-1).shape) == [1, 1, 0, 32]


def test_embedding_row_major_zero_width_weight(device):
    idx = ttnn.from_torch(
        (torch.arange(32) * 3 % 8).reshape(1, 32).to(torch.int32),
        dtype=ttnn.uint32,
        layout=RM,
        device=device,
        memory_config=DRAM,
    )
    weight = _to_device(device, _ints([8, 0]), RM)
    assert list(ttnn.embedding(idx, weight, memory_config=DRAM).shape) == [1, 32, 0]


def test_repeat_tile_zero_volume_into_l1(device):
    x = _to_device(device, _ints([2, 0, 32, 32]))
    assert list(ttnn.repeat(x, [2, 1, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG).shape) == [4, 0, 32, 32]


@pytest.mark.parametrize("memory_config", [HEIGHT_SHARDED, BLOCK_SHARDED], ids=["height", "block"])
def test_interleaved_to_sharded_zero_volume(device, memory_config):
    out = ttnn.interleaved_to_sharded(_to_device(device, _ints([1, 1, 0, 64])), memory_config)
    assert list(out.shape) == [1, 1, 0, 64]
    assert out.memory_config().memory_layout == memory_config.memory_layout


@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
def test_sharded_to_interleaved_zero_volume(device, layout):
    x = _to_device(device, _ints([1, 1, 0, 64]), layout, HEIGHT_SHARDED)
    out = ttnn.sharded_to_interleaved(x, DRAM)
    assert list(out.shape) == [1, 1, 0, 64]
    assert not out.is_sharded()


def test_sharded_to_interleaved_partial_zero_volume_leaves_cache(device):
    x = _to_device(device, _ints([1, 1, 0, 64]), TILE, HEIGHT_SHARDED)
    cache_src = torch.full((1, 1, 128, 64), 3.0, dtype=torch.bfloat16)
    cache = _to_device(device, cache_src)
    ttnn.sharded_to_interleaved_partial(x, cache, 2, 0, memory_config=DRAM)
    assert torch.equal(ttnn.to_torch(cache), cache_src)


@pytest.mark.parametrize(
    "layout, input_config, output_config",
    [
        (TILE, DRAM, None),
        (RM, DRAM, None),
        (TILE, HEIGHT_SHARDED, DRAM),
        (TILE, HEIGHT_SHARDED, None),
    ],
    ids=["tile_dram", "rm_dram", "tile_sharded_to_dram", "tile_sharded"],
)
def test_pad_zero_volume_is_all_padding(device, layout, input_config, output_config):
    x = _to_device(device, _ints([1, 1, 0, 64]), layout, input_config)
    out = ttnn.pad(x, [(0, 0), (0, 0), (0, 32), (0, 0)], 1.0, memory_config=output_config)
    assert out.is_sharded() == (output_config is None and input_config.is_sharded())
    assert torch.equal(ttnn.to_torch(out), torch.ones((1, 1, 32, 64), dtype=torch.bfloat16))
