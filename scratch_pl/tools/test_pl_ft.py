# Scratch, not for merge. Pack-lever round: `ttnn.tilize` callers of the fast tilize (bf16, 16-bit DEST) at row widths of
# 4 tiles (control) to 224 tiles, under the device profiler.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn


def _check(x, out):
    assert torch.equal(ttnn.to_torch(out).float(), x.float())


# width-sharded L1 tilize on 8 cores: 2 (paced init), 3, 4, 8 and 16 tiles per core.
@pytest.mark.parametrize("h", [64, 256])
@pytest.mark.parametrize("shard_tiles", [2, 3, 4, 8, 16])
def test_tilize_width_sharded(device, shard_tiles, h):
    w = 8 * shard_tiles * 32
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                            ttnn.ShardSpec(grid, [h, shard_tiles * 32], ttnn.ShardOrientation.ROW_MAJOR))
    torch.manual_seed(0)
    x = torch.randn(1, 1, h, w).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    out = ttnn.tilize(t, memory_config=mem)
    ttnn.synchronize_device(device)
    _check(x, out)


# the same with a bfp8_b output: 2 tiles per core take the W step now.
@pytest.mark.parametrize("h", [64, 256])
@pytest.mark.parametrize("shard_tiles", [2, 8])
def test_tilize_width_sharded_b8(device, shard_tiles, h):
    w = 8 * shard_tiles * 32
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                            ttnn.ShardSpec(grid, [h, shard_tiles * 32], ttnn.ShardOrientation.ROW_MAJOR))
    torch.manual_seed(0)
    x = torch.randn(1, 1, h, w).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    out = ttnn.tilize(t, memory_config=mem, dtype=ttnn.bfloat8_b)
    ttnn.synchronize_device(device)
    y = ttnn.to_torch(out).float()
    assert torch.corrcoef(torch.stack([y.flatten(), x.float().flatten()]))[0, 1] > 0.999


# height-sharded L1 tilize on 64 cores: 8 and 24 tiles per row.
@pytest.mark.parametrize("wt", [8, 24])
def test_tilize_height_sharded(device, wt):
    h, w = 64 * 64, wt * 32
    mem = ttnn.create_sharded_memory_config(
        (1, 1, h, w), core_grid=ttnn.CoreGrid(y=8, x=8), strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR)
    torch.manual_seed(0)
    x = torch.randn(1, 1, h, w).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    out = ttnn.tilize(t, memory_config=mem, use_multicore=True)
    ttnn.synchronize_device(device)
    _check(x, out)


# DRAM-interleaved multi-core tilize at LLM hidden widths (24, 64, 128, 224 tiles per row).
@pytest.mark.parametrize("shape", [(1, 1, 256, 768), (1, 1, 128, 2048), (1, 1, 64, 4096), (1, 1, 32, 7168)],
                         ids=["256x768", "128x2048", "64x4096", "32x7168"])
def test_tilize_interleaved(device, shape):
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.tilize(t, use_multicore=True)
    ttnn.synchronize_device(device)
    _check(x, out)


# to_layout with padding (tilize_with_val_padding), 2048 wide.
def test_to_layout_padded(device):
    torch.manual_seed(0)
    x = torch.randn(1, 1, 100, 2040).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
    ttnn.synchronize_device(device)
    assert torch.equal(ttnn.to_torch(out).float(), x.float())


# block-sharded 512x512 on 4x4 cores: 4 tiles per core (control).
def test_tilize_block_sharded(device):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                            ttnn.ShardSpec(grid, [128, 128], ttnn.ShardOrientation.ROW_MAJOR))
    torch.manual_seed(0)
    x = torch.randn(1, 1, 512, 512).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    out = ttnn.tilize(t, memory_config=mem)
    ttnn.synchronize_device(device)
    _check(x, out)
