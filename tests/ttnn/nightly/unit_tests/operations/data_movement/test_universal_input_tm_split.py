# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Universal I/O coverage for ttnn.split.

Tests the composite layer's transparent conversion of:
  * interleaved inputs in L1 and DRAM
  * mixed buffer locations (L1 in, DRAM out)
  * ROW_MAJOR inputs/outputs
  * sharded inputs (HEIGHT/WIDTH/BLOCK) with interleaved output
  * interleaved input with sharded output (equal splits)
  * program-cache reuse: same shape → no rebuild, different shape → rebuild
  * supported dtype combinations
  * various ranks and dimensions
"""

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device

TILE = 32

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

L1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mc=None):
    mc = mc or L1
    return ttnn.from_torch(t, dtype=dtype, device=device, memory_config=mc, layout=layout)


def _check(tt_outputs, torch_outputs):
    assert len(tt_outputs) == len(torch_outputs), f"chunk count {len(tt_outputs)} != {len(torch_outputs)}"
    for i, (tt_t, ref) in enumerate(zip(tt_outputs, torch_outputs)):
        out = ttnn.to_torch(tt_t).float()
        ref_f = ref.float()
        assert list(tt_t.shape) == list(ref.shape), f"chunk {i}: shape {list(tt_t.shape)} != {list(ref.shape)}"
        assert torch.allclose(out, ref_f, rtol=1e-2, atol=1e-2), f"chunk {i}: max_diff={(out - ref_f).abs().max():.4f}"


def _make_height_sharded_cfg(num_shards, shard_h, shard_w, buffer=ttnn.BufferType.L1):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_shards - 1))})
    spec = ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer, spec)


def _make_width_sharded_cfg(num_shards, shard_h, shard_w, buffer=ttnn.BufferType.L1):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_shards - 1, 0))})
    spec = ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer, spec)


def _make_block_sharded_cfg(grid_y, grid_x, shard_h, shard_w, buffer=ttnn.BufferType.L1):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    spec = ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer, spec)


# ---------------------------------------------------------------------------
# 1. Interleaved baseline — L1 and DRAM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, split_size, dim",
    [
        ([2 * TILE, 2 * TILE], TILE, -1),
        ([4 * TILE, 2 * TILE], TILE, 0),
        ([2 * TILE, 4 * TILE], 2 * TILE, -1),
    ],
)
def test_interleaved_l1_tile(device, shape, split_size, dim):
    """TILE layout, INTERLEAVED L1 in → INTERLEAVED L1 out."""
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=dim)
    tt_in = _from_torch(t, device, mc=L1)
    tt_out = ttnn.split(tt_in, split_size, dim=dim, memory_config=L1)
    _check(tt_out, ref)


@pytest.mark.parametrize(
    "shape, split_size, dim",
    [
        ([2 * TILE, 2 * TILE], TILE, -1),
        ([4 * TILE, 2 * TILE], TILE, 0),
    ],
)
def test_interleaved_dram_tile(device, shape, split_size, dim):
    """TILE layout, INTERLEAVED DRAM in → INTERLEAVED DRAM out."""
    torch.manual_seed(1)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=dim)
    tt_in = _from_torch(t, device, mc=DRAM)
    tt_out = ttnn.split(tt_in, split_size, dim=dim, memory_config=DRAM)
    _check(tt_out, ref)


def test_interleaved_l1_input_dram_output(device):
    """L1 input, DRAM output — memory-config cross-over."""
    torch.manual_seed(2)
    shape = [2 * TILE, 2 * TILE]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, TILE, dim=-1)
    tt_in = _from_torch(t, device, mc=L1)
    tt_out = ttnn.split(tt_in, TILE, dim=-1, memory_config=DRAM)
    _check(tt_out, ref)


# ---------------------------------------------------------------------------
# 2. ROW_MAJOR layout — interleaved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, split_size, dim",
    [
        ([64, 64], 32, -1),
        ([32, 64], 32, 0),
        ([1, 55, 48], 24, -1),
        ([3, 4, 8], 2, 1),
    ],
)
def test_row_major_interleaved(device, shape, split_size, dim):
    """ROW_MAJOR input, interleaved memory → output must be ROW_MAJOR."""
    torch.manual_seed(3)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=dim)
    tt_in = _from_torch(t, device, layout=ttnn.ROW_MAJOR_LAYOUT, mc=L1)
    tt_out = ttnn.split(tt_in, split_size, dim=dim)
    for tt_t in tt_out:
        assert tt_t.get_layout() == ttnn.ROW_MAJOR_LAYOUT, "Output layout must be ROW_MAJOR"
    _check(tt_out, ref)


# ---------------------------------------------------------------------------
# 3. Sharded inputs → interleaved output (de-sharding path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, split_size, dim",
    [
        ([4 * TILE, 2 * TILE], TILE, -1),
        ([4 * TILE, 2 * TILE], TILE, 0),
        ([8 * TILE, 2 * TILE], 2 * TILE, 0),
    ],
)
def test_height_sharded_input_interleaved_output(device, shape, split_size, dim):
    """HEIGHT_SHARDED TILE input → interleaved DRAM output."""
    torch.manual_seed(4)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=dim)

    num_shards = shape[0] // TILE
    sharded_cfg = _make_height_sharded_cfg(num_shards, TILE, shape[-1])

    tt_in = _from_torch(t, device, mc=sharded_cfg)
    tt_out = ttnn.split(tt_in, split_size, dim=dim, memory_config=DRAM)
    for tt_t in tt_out:
        assert not tt_t.memory_config().is_sharded(), "Output should be interleaved"
    _check(tt_out, ref)


@pytest.mark.parametrize(
    "shape, num_width_shards, split_size, dim",
    [
        ([2 * TILE, 4 * TILE], 2, TILE, -1),
        ([2 * TILE, 4 * TILE], 2, TILE, 0),
    ],
)
def test_width_sharded_input_interleaved_output(device, shape, num_width_shards, split_size, dim):
    """WIDTH_SHARDED TILE input → interleaved DRAM output."""
    torch.manual_seed(5)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=dim)

    shard_w = shape[-1] // num_width_shards
    sharded_cfg = _make_width_sharded_cfg(num_width_shards, shape[0], shard_w)

    tt_in = _from_torch(t, device, mc=sharded_cfg)
    tt_out = ttnn.split(tt_in, split_size, dim=dim, memory_config=DRAM)
    for tt_t in tt_out:
        assert not tt_t.memory_config().is_sharded(), "Output should be interleaved"
    _check(tt_out, ref)


@pytest.mark.parametrize(
    "shape, grid_y, grid_x, split_size, dim",
    [
        ([2 * TILE, 2 * TILE], 2, 2, TILE, -1),
        ([2 * TILE, 2 * TILE], 2, 2, TILE, 0),
    ],
)
def test_block_sharded_input_interleaved_output(device, shape, grid_y, grid_x, split_size, dim):
    """BLOCK_SHARDED TILE input → interleaved DRAM output."""
    torch.manual_seed(6)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=dim)

    shard_h = shape[0] // grid_y
    shard_w = shape[-1] // grid_x
    sharded_cfg = _make_block_sharded_cfg(grid_y, grid_x, shard_h, shard_w)

    tt_in = _from_torch(t, device, mc=sharded_cfg)
    tt_out = ttnn.split(tt_in, split_size, dim=dim, memory_config=DRAM)
    for tt_t in tt_out:
        assert not tt_t.memory_config().is_sharded(), "Output should be interleaved"
    _check(tt_out, ref)


# ---------------------------------------------------------------------------
# 4. ROW_MAJOR sharded inputs → interleaved output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, split_size, dim",
    [
        ([4 * TILE, 2 * TILE], TILE, -1),
        ([4 * TILE, 2 * TILE], TILE, 0),
    ],
)
def test_row_major_height_sharded_input(device, shape, split_size, dim):
    """ROW_MAJOR HEIGHT_SHARDED input → interleaved output."""
    torch.manual_seed(7)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=dim)

    num_shards = shape[0] // TILE
    sharded_cfg = _make_height_sharded_cfg(num_shards, TILE, shape[-1])

    tt_in = _from_torch(t, device, layout=ttnn.ROW_MAJOR_LAYOUT, mc=sharded_cfg)
    tt_out = ttnn.split(tt_in, split_size, dim=dim, memory_config=DRAM)
    for tt_t in tt_out:
        assert tt_t.get_layout() == ttnn.ROW_MAJOR_LAYOUT
        assert not tt_t.memory_config().is_sharded()
    _check(tt_out, ref)


@pytest.mark.parametrize(
    "shape, num_width_shards, split_size, dim",
    [
        ([2 * TILE, 4 * TILE], 2, TILE, -1),
        ([2 * TILE, 4 * TILE], 2, TILE, 0),
    ],
)
def test_row_major_width_sharded_input(device, shape, num_width_shards, split_size, dim):
    """ROW_MAJOR WIDTH_SHARDED input → interleaved DRAM output."""
    torch.manual_seed(17)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=dim)

    shard_w = shape[-1] // num_width_shards
    sharded_cfg = _make_width_sharded_cfg(num_width_shards, shape[0], shard_w)

    tt_in = _from_torch(t, device, layout=ttnn.ROW_MAJOR_LAYOUT, mc=sharded_cfg)
    tt_out = ttnn.split(tt_in, split_size, dim=dim, memory_config=DRAM)
    for tt_t in tt_out:
        assert tt_t.get_layout() == ttnn.ROW_MAJOR_LAYOUT
        assert not tt_t.memory_config().is_sharded()
    _check(tt_out, ref)


@pytest.mark.parametrize(
    "shape, grid_y, grid_x, split_size, dim",
    [
        ([2 * TILE, 4 * TILE], 2, 2, TILE, -1),
        ([2 * TILE, 4 * TILE], 2, 2, TILE, 0),
    ],
)
def test_row_major_block_sharded_input(device, shape, grid_y, grid_x, split_size, dim):
    """ROW_MAJOR BLOCK_SHARDED input → interleaved DRAM output."""
    torch.manual_seed(18)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=dim)

    shard_h = shape[0] // grid_y
    shard_w = shape[-1] // grid_x
    sharded_cfg = _make_block_sharded_cfg(grid_y, grid_x, shard_h, shard_w)

    tt_in = _from_torch(t, device, layout=ttnn.ROW_MAJOR_LAYOUT, mc=sharded_cfg)
    tt_out = ttnn.split(tt_in, split_size, dim=dim, memory_config=DRAM)
    for tt_t in tt_out:
        assert tt_t.get_layout() == ttnn.ROW_MAJOR_LAYOUT
        assert not tt_t.memory_config().is_sharded()
    _check(tt_out, ref)


# ---------------------------------------------------------------------------
# 5. Interleaved input → sharded output (equal splits)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sharding_type, shape, split_size, dim",
    [
        # HEIGHT: split along last dim so output height matches input height
        ("height", [4 * TILE, 2 * TILE], TILE, -1),
        # WIDTH: split along last dim; output width = input_width / num_shards
        ("width", [2 * TILE, 4 * TILE], 2 * TILE, -1),
        # BLOCK: split along last dim; output shard_w = input_shard_w / 2
        ("block", [2 * TILE, 4 * TILE], 2 * TILE, -1),
    ],
)
def test_interleaved_input_sharded_output(device, sharding_type, shape, split_size, dim):
    """Interleaved TILE input → sharded output (split along last dim)."""
    torch.manual_seed(8)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=dim)

    # Each output chunk has shape: shape with last dim = split_size
    out_shape = list(shape)
    out_shape[dim if dim >= 0 else len(shape) + dim] = split_size

    if sharding_type == "height":
        num_shards = out_shape[0] // TILE
        out_cfg = _make_height_sharded_cfg(num_shards, TILE, out_shape[-1])
    elif sharding_type == "width":
        # Two chunks → each chunk width = 2*TILE; shard on 2 cores
        num_shards = out_shape[-1] // TILE
        out_cfg = _make_width_sharded_cfg(num_shards, out_shape[0], TILE)
    else:
        out_cfg = _make_block_sharded_cfg(2, 2, out_shape[0] // 2, out_shape[-1] // 2)

    tt_in = _from_torch(t, device, mc=L1)
    tt_out = ttnn.split(tt_in, split_size, dim=dim, memory_config=out_cfg)

    assert tt_out[0].memory_config().is_sharded(), "Output should be sharded"
    _check(tt_out, ref)


# ---------------------------------------------------------------------------
# 6. Sharded input → sharded output (split along non-sharded dim)
# ---------------------------------------------------------------------------


def test_height_sharded_roundtrip(device):
    """HEIGHT_SHARDED input, split along last dim → HEIGHT_SHARDED output.

    The shard spec for the output is explicitly constructed for the output shape
    (each chunk has half the last-dim width of the input).
    """
    torch.manual_seed(9)
    shape = [4 * TILE, 2 * TILE]
    split_size = TILE
    dim = -1

    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=dim)

    # Input: HEIGHT_SHARDED [4*TILE, 2*TILE] with 4 shards of [TILE, 2*TILE]
    in_cfg = _make_height_sharded_cfg(shape[0] // TILE, TILE, shape[-1])
    # Output: each chunk is [4*TILE, TILE]; HEIGHT_SHARDED with 4 shards of [TILE, TILE]
    out_cfg = _make_height_sharded_cfg(shape[0] // TILE, TILE, split_size)

    tt_in = _from_torch(t, device, mc=in_cfg)
    tt_out = ttnn.split(tt_in, split_size, dim=dim, memory_config=out_cfg)

    assert tt_out[0].memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    _check(tt_out, ref)


# ---------------------------------------------------------------------------
# 7. Dtype coverage: bfloat16 and float32
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_dtype_coverage(device, torch_dtype, ttnn_dtype, layout):
    """bfloat16 and float32 must work with both layouts."""
    torch.manual_seed(10)
    shape = [2 * TILE, 2 * TILE]
    t = torch.randn(shape).to(torch_dtype)
    ref = torch.split(t, TILE, dim=-1)
    tt_in = ttnn.from_torch(t, dtype=ttnn_dtype, device=device, memory_config=L1, layout=layout)
    tt_out = ttnn.split(tt_in, TILE, dim=-1)
    out = [ttnn.to_torch(tt_t).to(torch_dtype) for tt_t in tt_out]
    for i, (o, r) in enumerate(zip(out, ref)):
        assert torch.allclose(
            o.float(), r.float(), rtol=1e-2, atol=1e-2
        ), f"chunk {i}: dtype={ttnn_dtype} layout={layout} mismatch"


# ---------------------------------------------------------------------------
# 8. Program-cache: same shape → reuse; different shape → rebuild
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_program_cache_same_shape(device, layout):
    """Two consecutive same-shape calls must produce correct results."""
    torch.manual_seed(11)
    shape = [4 * TILE, 2 * TILE]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, TILE, dim=-1)

    tt_in = _from_torch(t, device, layout=layout, mc=L1)
    for _ in range(2):
        tt_out = ttnn.split(tt_in, TILE, dim=-1, memory_config=L1)
        _check(tt_out, ref)


def test_program_cache_shape_change(device):
    """Different shape after first call must still produce correct results."""
    torch.manual_seed(12)
    shape_a = [4 * TILE, 2 * TILE]
    shape_b = [2 * TILE, 4 * TILE]

    t_a = torch.randn(shape_a, dtype=torch.bfloat16)
    t_b = torch.randn(shape_b, dtype=torch.bfloat16)
    ref_a = torch.split(t_a, TILE, dim=-1)
    ref_b = torch.split(t_b, TILE, dim=-1)

    tt_a = _from_torch(t_a, device, mc=L1)
    tt_out_a = ttnn.split(tt_a, TILE, dim=-1)
    _check(tt_out_a, ref_a)

    tt_b = _from_torch(t_b, device, mc=L1)
    tt_out_b = ttnn.split(tt_b, TILE, dim=-1)
    _check(tt_out_b, ref_b)


# ---------------------------------------------------------------------------
# 9. Sharded input with no explicit output config → DRAM interleaved default
# ---------------------------------------------------------------------------


def test_sharded_input_default_output_is_dram(device):
    """Sharded input with no memory_config arg must produce DRAM interleaved output."""
    torch.manual_seed(13)
    shape = [4 * TILE, 2 * TILE]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, TILE, dim=-1)

    in_cfg = _make_height_sharded_cfg(shape[0] // TILE, TILE, shape[-1])
    tt_in = _from_torch(t, device, mc=in_cfg)
    # No memory_config provided → should default to DRAM interleaved
    tt_out = ttnn.split(tt_in, TILE, dim=-1)
    for tt_t in tt_out:
        assert not tt_t.memory_config().is_sharded(), "Default output for sharded input should be interleaved"
        assert tt_t.memory_config().buffer_type == ttnn.BufferType.DRAM
    _check(tt_out, ref)


# ---------------------------------------------------------------------------
# 10. Various ranks and dimensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, split_size, dim",
    [
        ([2 * TILE, 2 * TILE], TILE, 0),  # 2D, dim 0
        ([2 * TILE, 2 * TILE], TILE, 1),  # 2D, dim 1
        ([2, 2 * TILE, 2 * TILE], TILE, 1),  # 3D, dim 1
        ([2, 2 * TILE, 2 * TILE], TILE, 2),  # 3D, dim 2
        ([1, 2, 2 * TILE, 2 * TILE], TILE, 2),  # 4D, dim 2
        ([1, 2, 2 * TILE, 2 * TILE], TILE, 3),  # 4D, dim 3
    ],
)
def test_various_ranks_and_dims(device, shape, split_size, dim):
    """Split across various ranks and dimensions with interleaved tensors."""
    torch.manual_seed(14)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=dim)
    tt_in = _from_torch(t, device, mc=L1)
    tt_out = ttnn.split(tt_in, split_size, dim=dim, memory_config=L1)
    _check(tt_out, ref)


# ---------------------------------------------------------------------------
# 11. Unequal splits (split_sizes list)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, split_sizes, dim",
    [
        ([2 * TILE, 3 * TILE], [TILE, 2 * TILE], -1),
        ([3 * TILE, 2 * TILE], [TILE, 2 * TILE], 0),
    ],
)
def test_unequal_splits_interleaved(device, shape, split_sizes, dim):
    """Unequal split sizes with interleaved input."""
    torch.manual_seed(15)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_sizes, dim=dim)
    tt_in = _from_torch(t, device, mc=L1)
    tt_out = ttnn.split(tt_in, split_sizes, dim=dim, memory_config=L1)
    _check(tt_out, ref)


@pytest.mark.parametrize(
    "shape, split_sizes, dim",
    [
        ([2 * TILE, 3 * TILE], [TILE, 2 * TILE], -1),
        ([3 * TILE, 2 * TILE], [TILE, 2 * TILE], 0),
    ],
)
def test_unequal_splits_sharded_input(device, shape, split_sizes, dim):
    """Unequal split sizes with HEIGHT_SHARDED input → DRAM interleaved output."""
    torch.manual_seed(16)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_sizes, dim=dim)

    num_shards = shape[0] // TILE
    in_cfg = _make_height_sharded_cfg(num_shards, TILE, shape[-1])
    tt_in = _from_torch(t, device, mc=in_cfg)
    tt_out = ttnn.split(tt_in, split_sizes, dim=dim, memory_config=DRAM)
    _check(tt_out, ref)


# ---------------------------------------------------------------------------
# 12. N-way equal TILE splits via native kernel (no slice fallback)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, num_chunks, dim",
    [
        ([2 * TILE, 3 * TILE], 3, -1),  # 2D, 3-way
        ([2 * TILE, 4 * TILE], 4, -1),  # 2D, 4-way
        ([1, 2, 2 * TILE, 4 * TILE], 4, -1),  # 4D, 4-way last dim
    ],
)
def test_tile_n_way_equal_split(device, shape, num_chunks, dim):
    """Equal N-way TILE split uses the native N-chunk kernel, not N slice calls."""
    torch.manual_seed(19)
    t = torch.randn(shape, dtype=torch.bfloat16)
    split_size = shape[dim if dim >= 0 else len(shape) + dim] // num_chunks
    ref = torch.split(t, split_size, dim=dim)
    tt_in = _from_torch(t, device, mc=L1)
    tt_out = ttnn.split(tt_in, split_size, dim=dim, memory_config=L1)
    assert len(tt_out) == num_chunks
    _check(tt_out, ref)


# ---------------------------------------------------------------------------
# 13. Slice-fallback 2-level batching boundary (num_chunks just above
#     SPLIT_BATCH_SIZE=64 triggers the sqrt-N batching path in
#     split_with_slice_impl). num_chunks > grid_dim_y also forces the slice
#     fallback rather than the native TILE kernel.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_chunks", [64, 65, 130])
def test_slice_fallback_batching_boundary(device, num_chunks):
    """Large equal-chunk splits across the SPLIT_BATCH_SIZE=64 boundary."""
    torch.manual_seed(20)
    split_size = TILE
    shape = [TILE, num_chunks * split_size]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=-1)
    tt_in = _from_torch(t, device, mc=L1)
    tt_out = ttnn.split(tt_in, split_size, dim=-1, memory_config=L1)
    assert len(tt_out) == num_chunks
    _check(tt_out, ref)
