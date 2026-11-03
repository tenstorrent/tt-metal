# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.alias_view: a zero-copy view of a paged cache with another dtype.

The input is a bfloat16 paged cache [num_blocks, 1, block_size, 256]. Writes go through paged_fill_cache, the op a
model uses to write one block. A valid view must cover exactly the input's block b when it writes its block b.
"""

import pytest
import torch

import ttnn

NUM_BLOCKS = 12
TARGET = 5
BF16_TILE_BYTES = 2048
FP32_TILE_BYTES = 4096


def _num_banks(device):
    grid = device.dram_grid_size()
    return grid.x * grid.y


def _fp32_view_is_valid(block_size, banks):
    """alias_view's rule for a float32 view of one bfloat16 [block_size, 256] block."""
    pages = (block_size // 32) * (256 // 32)
    return pages % banks == 0 and (pages // banks * BF16_TILE_BYTES) % FP32_TILE_BYTES == 0


def _valid_block_size(device):
    """A block size whose float32 view is valid on this device's bank count (7 on P100, 8 on P150, 12 on Wormhole)."""
    banks = _num_banks(device)
    return next(b for b in (1024, 896, 768) if _fp32_view_is_valid(b, banks))


def _owner(device, block_size, mesh_mapper=None, num_devices=1):
    host = torch.randn(NUM_BLOCKS, num_devices, block_size, 256).bfloat16()
    dev = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper)
    return host, dev


def _fill_block(view, data, block_id, device, mesh_mapper=None):
    page_table = ttnn.from_torch(
        torch.tensor([[block_id]], dtype=torch.int32),
        dtype=ttnn.int32,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if mesh_mapper else None,
    )
    x = ttnn.from_torch(data, dtype=view.dtype, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper)
    ttnn.experimental.paged_fill_cache(view, x, page_table, batch_idx=0)


def _tiles(x):
    """[a, r, c] -> tiles in TILE order, so a bfloat16 block can be compared with what an aliased write left."""
    a, r, c = x.shape
    return x.reshape(a, r // 32, 32, c // 32, 32).permute(0, 1, 3, 2, 4).reshape(-1, 32, 32)


def test_alias_view_shares_the_input_buffer(device):
    _, k = _owner(device, _valid_block_size(device))
    view = ttnn.experimental.alias_view(k, ttnn.Shape([NUM_BLOCKS, 1, k.shape[2], 128]), ttnn.float32)
    assert view.buffer_address() == k.buffer_address()
    assert view.dtype == ttnn.float32
    assert list(view.shape) == [NUM_BLOCKS, 1, k.shape[2], 128]


def test_alias_view_same_dtype_keeps_block_contents(device):
    """bfloat16 to bfloat16 with another shape: block b of the view holds exactly the tiles of input block b."""
    block_size = _valid_block_size(device)
    heads = 2 * block_size // 128  # bfloat16 [128, 128] tiles per bfloat16 [block_size, 256] block
    host, k = _owner(device, block_size)
    view = ttnn.experimental.alias_view(k, ttnn.Shape([NUM_BLOCKS, heads, 128, 128]), ttnn.bfloat16)

    state = torch.randn(1, heads, 128, 128).bfloat16()
    _fill_block(view, state, TARGET, device)
    got = ttnn.to_torch(k)

    assert torch.equal(_tiles(got[TARGET]), _tiles(state[0]))
    others = [i for i in range(NUM_BLOCKS) if i != TARGET]
    assert torch.equal(got[others], host[others])


@pytest.mark.parametrize("block_size", [896, 928, 960, 1024])
def test_alias_view_fp32_block_mapping(device, block_size, expect_error):
    """A float32 view either covers the input's blocks exactly or raises."""
    shape = ttnn.Shape([NUM_BLOCKS, 1, block_size, 128])
    host, k = _owner(device, block_size)
    if not _fp32_view_is_valid(block_size, _num_banks(device)):
        with expect_error(RuntimeError, "alias_view"):
            ttnn.experimental.alias_view(k, shape, ttnn.float32)
        return

    view = ttnn.experimental.alias_view(k, shape, ttnn.float32)
    before = ttnn.to_torch(view)
    state = torch.randn(1, 1, block_size, 128, dtype=torch.float32)
    _fill_block(view, state, TARGET, device)

    others = [i for i in range(NUM_BLOCKS) if i != TARGET]
    after_view = ttnn.to_torch(view)
    assert torch.equal(after_view[TARGET], state[0]), "a write and read through the view must round-trip"
    assert torch.equal(after_view[others], before[others]), "the write changed another block of the view"
    assert torch.equal(ttnn.to_torch(k)[others], host[others]), "the write changed another block of the input"


def test_alias_view_pool_pages_follow_blocks(device):
    """A rank-3 float32 pool [num_blocks * heads, 128, 128]: input block b is pool entries [b * heads, + heads)."""
    block_size = _valid_block_size(device)
    heads = block_size // 128
    _, k = _owner(device, block_size)
    blocks = ttnn.experimental.alias_view(k, ttnn.Shape([NUM_BLOCKS, heads, 128, 128]), ttnn.float32)
    pool = ttnn.experimental.alias_view(k, ttnn.Shape([NUM_BLOCKS * heads, 128, 128]), ttnn.float32)

    state = torch.randn(1, heads, 128, 128, dtype=torch.float32)
    _fill_block(blocks, state, TARGET, device)
    got = ttnn.to_torch(pool)[TARGET * heads : (TARGET + 1) * heads]
    assert torch.equal(got, state[0])


def test_alias_view_rejects_unsafe_inputs(device, expect_error):
    banks = _num_banks(device)
    block_size = _valid_block_size(device)
    _, k = _owner(device, block_size)

    with expect_error(RuntimeError, "per bank"):
        ttnn.experimental.alias_view(k, ttnn.Shape([NUM_BLOCKS + 1, 1, block_size, 128]), ttnn.float32)

    with expect_error(RuntimeError, "straddle"):
        ttnn.experimental.alias_view(k, ttnn.Shape([2, 5, 128, 128]), ttnn.float32)

    rank2 = ttnn.from_torch(
        torch.zeros(banks * 32, 32).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "rank"):
        ttnn.experimental.alias_view(rank2, ttnn.Shape([banks * 16, 32]), ttnn.float32)

    row_major = ttnn.from_torch(torch.zeros(2, 1, 32, 32).bfloat16(), dtype=ttnn.bfloat16, device=device)
    with expect_error(RuntimeError, "TILE"):
        ttnn.experimental.alias_view(row_major, ttnn.Shape([2, 1, 32, 16]), ttnn.float32)

    one_tile_blocks = ttnn.from_torch(
        torch.zeros(banks * 2, 1, 32, 32).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "multiple of the"):
        ttnn.experimental.alias_view(one_tile_blocks, ttnn.Shape([banks, 1, 32, 32]), ttnn.float32)


def test_alias_view_on_mesh(mesh_device):
    """Every device of the mesh sees its own shard through the view; block b stays block b on each of them."""
    n = mesh_device.get_num_devices()
    block_size = _valid_block_size(mesh_device)
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=1)
    concat = ttnn.ConcatMeshToTensor(mesh_device, dim=1)
    host, k = _owner(mesh_device, block_size, mesh_mapper=shard, num_devices=n)
    view = ttnn.experimental.alias_view(k, ttnn.Shape([NUM_BLOCKS, 1, block_size, 128]), ttnn.float32)
    assert view.buffer_address() == k.buffer_address()

    state = torch.randn(1, n, block_size, 128, dtype=torch.float32)
    _fill_block(view, state, TARGET, mesh_device, mesh_mapper=shard)

    others = [i for i in range(NUM_BLOCKS) if i != TARGET]
    assert torch.equal(ttnn.to_torch(view, mesh_composer=concat)[TARGET], state[0])
    assert torch.equal(ttnn.to_torch(k, mesh_composer=concat)[others], host[others])
