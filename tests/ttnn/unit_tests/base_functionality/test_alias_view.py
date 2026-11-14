# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.alias_view: a zero-copy view of a paged cache with another dtype.

The input is a bfloat16 or bfloat8_b paged cache [num_blocks, 1, block_size, 256]. Writes go through
paged_fill_cache, the op a model uses to write one block. A valid view must cover exactly the input's block b when it
writes its block b.
"""

import math

import pytest
import torch

import ttnn

NUM_BLOCKS = 12
TARGET = 5
BF8_TILE_BYTES = 1088  # 1,024 one-byte mantissas and 64 shared exponents
BF16_TILE_BYTES = 2048
FP32_TILE_BYTES = 4096

# Expected bfloat8_b results per DRAM bank count (8 on P150 and P300, 7 on P100).
# Smallest block size whose float32 view passes the default checks: 64 bfloat8_b pages per bank, 69,632 bytes.
BF8_ALIGNED_BLOCK = {7: 1792, 8: 2048}
# Fewest whole float32 pages one block holds per bank with check_slices=False. None: the block's pages do not split
# evenly over the banks, so alias_view raises in both modes.
BF8_MIN_WHOLE_FP32_PAGES = {7: {896: 8, 1024: None}, 8: {896: 6, 1024: 8}}


def _num_banks(device):
    grid = device.dram_grid_size()
    return grid.x * grid.y


def _pages_per_block(block_size):
    return (block_size // 32) * (256 // 32)


def _block_bytes_per_bank(block_size, banks, owner_tile):
    """Bytes one [block_size, 256] block of the input holds on each bank, when its pages split evenly over banks."""
    return _pages_per_block(block_size) // banks * owner_tile


def _view_is_valid(block_size, banks, owner_tile=BF16_TILE_BYTES, view_tile=FP32_TILE_BYTES):
    """alias_view's rule, with check_slices=True, for a view of one [block_size, 256] block of the input."""
    pages = _pages_per_block(block_size)
    return pages % banks == 0 and _block_bytes_per_bank(block_size, banks, owner_tile) % view_tile == 0


def _valid_block_size(device):
    """A block size whose float32 view is valid on this device's bank count (7 on P100, 8 on P150, 12 on Wormhole)."""
    banks = _num_banks(device)
    return next(b for b in (1024, 896, 768) if _view_is_valid(b, banks))


def _bf8_aligned_block_size(banks):
    """Smallest block size whose bfloat8_b pages are whole float32 pages on every bank."""
    return next(b for b in range(32, 8193, 32) if _view_is_valid(b, banks, BF8_TILE_BYTES, FP32_TILE_BYTES))


def _owner(device, block_size, mesh_mapper=None, num_devices=1):
    host = torch.randn(NUM_BLOCKS, num_devices, block_size, 256).bfloat16()
    dev = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper)
    return host, dev


def _bf8_owner(device, num_blocks, block_size):
    host = torch.randn(num_blocks, 1, block_size, 256).bfloat16()
    return ttnn.from_torch(host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)


def _fill_blocks(view, data, block_ids, device, mesh_mapper=None):
    """Writes data to view blocks block_ids, one block-size chunk of data per id, in order."""
    page_table = ttnn.from_torch(
        torch.tensor([block_ids], dtype=torch.int32),
        dtype=ttnn.int32,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if mesh_mapper else None,
    )
    x = ttnn.from_torch(data, dtype=view.dtype, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper)
    ttnn.experimental.paged_fill_cache(view, x, page_table, batch_idx=0)


def _fill_block(view, data, block_id, device, mesh_mapper=None):
    _fill_blocks(view, data, [block_id], device, mesh_mapper)


def _bits(x):
    """Bit pattern of a float tensor, so that equal NaNs compare equal."""
    return x.view({2: torch.int16, 4: torch.int32}[x.element_size()])


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
    if not _view_is_valid(block_size, _num_banks(device)):
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

    # a zero dimension, in dim 0 or elsewhere, leaves the view no pages to divide by
    for empty in ([0, 1, 32, 32], [1, 0, 32, 32], [1, 1, 0, 32]):
        for check_slices in (True, False):
            with expect_error(RuntimeError, "no elements"):
                ttnn.experimental.alias_view(k, ttnn.Shape(empty), ttnn.float32, check_slices=check_slices)

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


@pytest.mark.parametrize("view_dtype", [ttnn.float32, ttnn.bfloat16])
def test_alias_view_bf8_aligned_blocks(device, view_dtype):
    """A view of a bfloat8_b cache at the aligned block size: writing all of view block b leaves other blocks unchanged.

    The aligned block holds 64 bfloat8_b pages per bank, 69,632 bytes, the least common multiple of the 1,088 byte
    bfloat8_b page and the 4,096 byte float32 page: block 2048 on 8 banks, 1792 on 7. Head h of a view block
    [pages_per_bank, 32, 32 * banks] is page row h of the block, one page on each bank. A raw int32 view of the whole
    buffer is read before and after each write, so the other blocks are compared bit for bit.
    """
    banks = _num_banks(device)
    block_size = _bf8_aligned_block_size(banks)
    assert block_size == BF8_ALIGNED_BLOCK.get(banks, block_size)
    view_tile = FP32_TILE_BYTES if view_dtype == ttnn.float32 else BF16_TILE_BYTES
    bytes_per_bank = _block_bytes_per_bank(block_size, banks, BF8_TILE_BYTES)
    heads = bytes_per_bank // view_tile
    raw_heads = bytes_per_bank // FP32_TILE_BYTES

    k = _bf8_owner(device, NUM_BLOCKS, block_size)
    view = ttnn.experimental.alias_view(k, ttnn.Shape([NUM_BLOCKS, heads, 32, 32 * banks]), view_dtype)
    raw = ttnn.experimental.alias_view(k, ttnn.Shape([NUM_BLOCKS, raw_heads, 32, 32 * banks]), ttnn.int32)
    torch_dtype = torch.float32 if view_dtype == ttnn.float32 else torch.bfloat16

    written = [0, 1, TARGET, 6, NUM_BLOCKS - 1]
    decoded = ttnn.to_torch(k)
    before = ttnn.to_torch(raw)
    for b in written:
        state = torch.randn(1, heads, 32, 32 * banks, dtype=torch_dtype)
        _fill_block(view, state, b, device)
        after = ttnn.to_torch(raw)
        others = [i for i in range(NUM_BLOCKS) if i != b]
        assert torch.equal(after[others], before[others]), f"writing block {b} changed another block"
        assert torch.equal(ttnn.to_torch(view)[b], state[0]), "a write and read through the view must round-trip"
        if view_dtype == ttnn.float32:
            assert torch.equal(after[b], _bits(state[0])), "block b of the input must hold the state's bytes"
        before = after

    untouched = [i for i in range(NUM_BLOCKS) if i not in written]
    assert torch.equal(_bits(ttnn.to_torch(k)[untouched]), _bits(decoded[untouched]))


@pytest.mark.parametrize("block_size", [896, 1024, 1792, 2048])
def test_alias_view_bf8_default_checks(device, block_size, expect_error):
    """The default checks accept a float32 view of a bfloat8_b cache only at multiples of the aligned block size.

    On 8 banks, blocks of 896, 1024 and 1792 hold 7.4375, 8.5 and 14.875 float32 pages per bank, and 2048 holds 17.
    On 7 banks, 1024 and 2048 do not split evenly over the banks, 896 holds 8.5 float32 pages per bank and 1792
    holds 17.
    """
    banks = _num_banks(device)
    k = _bf8_owner(device, NUM_BLOCKS, block_size)
    shape = ttnn.Shape([NUM_BLOCKS, 1, 32, 32 * banks])
    if _view_is_valid(block_size, banks, BF8_TILE_BYTES, FP32_TILE_BYTES):
        assert block_size % BF8_ALIGNED_BLOCK.get(banks, block_size) == 0
        ttnn.experimental.alias_view(k, shape, ttnn.float32)
        return
    message = "not a multiple of the view's" if _pages_per_block(block_size) % banks == 0 else "banks"
    with expect_error(RuntimeError, message):
        ttnn.experimental.alias_view(k, shape, ttnn.float32)


@pytest.mark.parametrize("block_size", [896, 1024])
def test_alias_view_bf8_whole_pages_per_block(device, block_size, expect_error):
    """A flat float32 view (check_slices=False) of a bfloat8_b cache whose blocks are not whole float32 pages per bank.

    Block b holds bytes [b * c, (b + 1) * c) on every bank, with c = (block_size / 4 / banks) * 1088. Its state goes
    only to the float32 page rows wholly inside that range: rows ceil(b * c / 4096) to floor((b + 1) * c / 4096) - 1.
    Slice r of the view is page row r: one float32 page on each bank, at offset r * 4096. A raw int32 view of the
    same rows is read before and after each write, so every byte outside the written rows is compared bit for bit.
    """
    banks = _num_banks(device)
    expected = BF8_MIN_WHOLE_FP32_PAGES.get(banks, {})
    num_blocks = 16
    k = _bf8_owner(device, num_blocks, block_size)
    if _pages_per_block(block_size) % banks:
        assert expected.get(block_size) is None
        with expect_error(RuntimeError, "banks"):
            ttnn.experimental.alias_view(
                k, ttnn.Shape([num_blocks, 1, 32, 32 * banks]), ttnn.float32, check_slices=False
            )
        return
    c = _block_bytes_per_bank(block_size, banks, BF8_TILE_BYTES)
    rows = num_blocks * c // FP32_TILE_BYTES
    flat = ttnn.experimental.alias_view(k, ttnn.Shape([rows, 1, 32, 32 * banks]), ttnn.float32, check_slices=False)
    raw = ttnn.experimental.alias_view(k, ttnn.Shape([rows, 1, 32, 32 * banks]), ttnn.int32, check_slices=False)
    with expect_error(RuntimeError, "the input owns"):
        ttnn.experimental.alias_view(k, ttnn.Shape([rows + 1, 1, 32, 32 * banks]), ttnn.float32, check_slices=False)

    whole_rows = [range(math.ceil(b * c / FP32_TILE_BYTES), (b + 1) * c // FP32_TILE_BYTES) for b in range(num_blocks)]
    if block_size in expected:
        assert min(len(r) for r in whole_rows) == expected[block_size]

    decoded = ttnn.to_torch(k)
    before = ttnn.to_torch(raw)
    for b, block_rows in enumerate(whole_rows):
        n = len(block_rows)
        state = torch.randn(1, 1, n * 32, 32 * banks, dtype=torch.float32)
        _fill_blocks(flat, state, list(block_rows), device)
        after = ttnn.to_torch(raw)
        new_decoded = ttnn.to_torch(k)
        outside = [r for r in range(rows) if r not in block_rows]
        others = [i for i in range(num_blocks) if i != b]
        assert torch.equal(after[outside], before[outside]), f"writing block {b} changed bytes outside its pages"
        assert torch.equal(after[block_rows.start : block_rows.stop], _bits(state).reshape(n, 1, 32, 32 * banks))
        assert torch.equal(_bits(new_decoded[others]), _bits(decoded[others])), f"block {b} changed another block"
        before, decoded = after, new_decoded

    # A page row across a block boundary holds bytes of both blocks, which is why the runner must skip it.
    b = next(b for b in range(num_blocks - 1) if (b + 1) * c % FP32_TILE_BYTES)
    _fill_blocks(flat, torch.randn(1, 1, 32, 32 * banks), [(b + 1) * c // FP32_TILE_BYTES], device)
    new_decoded = ttnn.to_torch(k)
    changed = [i for i in range(num_blocks) if not torch.equal(_bits(new_decoded[i]), _bits(decoded[i]))]
    assert changed == [b, b + 1]


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
