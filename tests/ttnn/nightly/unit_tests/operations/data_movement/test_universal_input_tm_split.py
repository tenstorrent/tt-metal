# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Universal I/O coverage for ttnn.split: interleaved/sharded/DRAM MC combos, dtype/rank/dim matrices, program cache, no-spec synthesis, L1 CB-clash regression."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc

pytestmark = pytest.mark.use_module_device

TILE = 32

# Helpers.

L1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mc=None):
    mc = mc or L1
    return ttnn.from_torch(t, dtype=dtype, device=device, memory_config=mc, layout=layout)


def _check(tt_outputs, torch_outputs):
    assert len(tt_outputs) == len(torch_outputs), f"chunk count {len(tt_outputs)} != {len(torch_outputs)}"
    for i, (tt_t, ref) in enumerate(zip(tt_outputs, torch_outputs)):
        out = ttnn.to_torch(tt_t)
        assert list(tt_t.shape) == list(ref.shape), f"chunk {i}: shape {list(tt_t.shape)} != {list(ref.shape)}"
        try:
            assert_equal(ref, out)
        except AssertionError as e:
            raise AssertionError(f"chunk {i}: {e}") from e


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


# 1. Interleaved baseline — L1 and DRAM.


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


# 2. ROW_MAJOR layout — interleaved.


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


# 3. Sharded inputs → interleaved output (de-sharding path).


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


# 4. ROW_MAJOR sharded inputs → interleaved output.


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


# 5. Interleaved input → sharded output (equal splits).


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


# 6. Sharded input → sharded output (split along non-sharded dim).


def test_height_sharded_roundtrip(device):
    """HEIGHT_SHARDED input, split along last dim → HEIGHT_SHARDED output; output spec built for chunk shape."""
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


# 7. Dtype coverage: bfloat16 and float32.


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
        try:
            assert_equal(r, o)
        except AssertionError as e:
            raise AssertionError(f"chunk {i}: dtype={ttnn_dtype} layout={layout}: {e}") from e


# 8. Program-cache: same shape → reuse; different shape → rebuild.


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


# 9. Sharded input, default MC → sharded output with rescaled shard_spec.


def test_sharded_input_default_preserves_sharding(device):
    """HEIGHT_SHARDED input + no memory_config → HEIGHT_SHARDED output with adjusted spec."""
    torch.manual_seed(13)
    shape = [4 * TILE, 2 * TILE]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, TILE, dim=-1)

    in_cfg = _make_height_sharded_cfg(shape[0] // TILE, TILE, shape[-1])
    tt_in = _from_torch(t, device, mc=in_cfg)
    tt_out = ttnn.split(tt_in, TILE, dim=-1)
    for tt_t in tt_out:
        mc = tt_t.memory_config()
        assert mc.is_sharded(), "Default output for sharded input should preserve sharding"
        assert mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        assert mc.buffer_type == ttnn.BufferType.L1
        assert mc.shard_spec is not None
        assert list(mc.shard_spec.shape) == [TILE, TILE], f"expected shard [32,32], got {list(mc.shard_spec.shape)}"
    _check(tt_out, ref)


# 9b. Sharded input → sharded output with no shard_spec (synthesize_spec path).


@pytest.mark.parametrize(
    "input_layout, shape, split_size",
    [
        # HEIGHT: [128, 64] halved on last dim → adjusted shard [32, 32] (tile-aligned).
        pytest.param(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [4 * TILE, 2 * TILE], TILE, id="height_in"),
        # BLOCK: [128, 128] on 2x2 grid, halved on last dim → adjusted shard [64, 32] (tile-aligned).
        pytest.param(ttnn.TensorMemoryLayout.BLOCK_SHARDED, [4 * TILE, 4 * TILE], 2 * TILE, id="block_in"),
    ],
)
def test_sharded_input_to_sharded_no_spec(device, input_layout, shape, split_size):
    """User passes sharded output MC without shard_spec → split must synthesize (not FATAL)."""
    torch.manual_seed(21)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=-1)

    if input_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        in_cfg = _make_height_sharded_cfg(shape[0] // TILE, TILE, shape[-1])
    else:
        in_cfg = _make_block_sharded_cfg(2, 2, shape[0] // 2, shape[-1] // 2)

    out_no_spec = ttnn.MemoryConfig(input_layout, ttnn.BufferType.L1)
    tt_in = _from_torch(t, device, mc=in_cfg)
    tt_out = ttnn.split(tt_in, split_size, dim=-1, memory_config=out_no_spec)
    for tt_t in tt_out:
        mc = tt_t.memory_config()
        assert mc.is_sharded() and mc.shard_spec is not None, "shard_spec must be synthesized"
        assert mc.memory_layout == input_layout
    _check(tt_out, ref)


# 9c. Interleaved input → sharded output with no shard_spec (synthesize_spec path).


@pytest.mark.parametrize(
    "memory_layout, shape, split_size",
    [
        pytest.param(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [4 * TILE, 2 * TILE], TILE, id="H_out"),
        pytest.param(ttnn.TensorMemoryLayout.WIDTH_SHARDED, [2 * TILE, 4 * TILE], 2 * TILE, id="W_out"),
        pytest.param(ttnn.TensorMemoryLayout.BLOCK_SHARDED, [2 * TILE, 4 * TILE], 2 * TILE, id="B_out"),
    ],
)
def test_interleaved_to_sharded_no_spec(device, memory_layout, shape, split_size):
    """Interleaved input, sharded output MC without shard_spec → split must generate a valid spec."""
    torch.manual_seed(22)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=-1)
    tt_in = _from_torch(t, device, mc=L1)
    out_no_spec = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1)
    tt_out = ttnn.split(tt_in, split_size, dim=-1, memory_config=out_no_spec)
    for tt_t in tt_out:
        mc = tt_t.memory_config()
        assert mc.is_sharded() and mc.shard_spec is not None
        assert mc.memory_layout == memory_layout
    _check(tt_out, ref)


# 9d. Sub-tile rescale → graceful DRAM downgrade (tile-alignment guard on rescaled shard_w=16 < TILE).


def test_sub_tile_rescale_downgrade_to_dram(device):
    """Rescaled shard would be sub-tile → must fall back to DRAM (no crash, correct values)."""
    torch.manual_seed(23)
    shape = [2 * TILE, 2 * TILE]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, TILE, dim=-1)

    in_cfg = _make_block_sharded_cfg(2, 2, TILE, TILE)
    tt_in = _from_torch(t, device, mc=in_cfg)
    tt_out = ttnn.split(tt_in, TILE, dim=-1)
    for tt_t in tt_out:
        mc = tt_t.memory_config()
        assert not mc.is_sharded(), "Sub-tile rescaled shard must trigger DRAM downgrade"
        assert mc.buffer_type == ttnn.BufferType.DRAM
    _check(tt_out, ref)


# 9e. L1 CB-clash regression: large N-way L1 split must not crash; DRAM or L1 output both acceptable.


def test_l1_ci_regression_many_chunks_no_crash(device):
    """Large L1-interleaved input, 32-way split, no MC. L1 CB-clash regression."""
    torch.manual_seed(24)
    num_chunks = 32
    shape = [1, 1, TILE, num_chunks * TILE]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, TILE, dim=-1)
    tt_in = _from_torch(t, device, mc=L1)
    tt_out = ttnn.split(tt_in, TILE, dim=-1)
    assert len(tt_out) == num_chunks
    _check(tt_out, ref)


# 9f. DRAM-sharded input → rescaled DRAM-sharded output.


def test_dram_sharded_input_rescale(device):
    """DRAM-sharded input, no MC → DRAM-sharded output with rescaled spec."""
    torch.manual_seed(25)
    shape = [4 * TILE, 2 * TILE]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, TILE, dim=-1)

    num_cores = 4
    compute_grid = device.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    shard_shape = [shape[0] // num_cores, shape[-1]]
    spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, spec)

    tt_in = _from_torch(t, device, mc=in_cfg)
    tt_out = ttnn.split(tt_in, TILE, dim=-1)
    for tt_t in tt_out:
        mc = tt_t.memory_config()
        assert mc.is_sharded()
        assert mc.buffer_type == ttnn.BufferType.DRAM
        assert list(mc.shard_spec.shape) == [TILE, TILE]
    _check(tt_out, ref)


# 9g. COL_MAJOR shard orientation propagation.


def test_col_major_shard_orientation_preserved(device):
    """COL_MAJOR HEIGHT_SHARDED input → default output preserves COL_MAJOR orientation."""
    torch.manual_seed(26)
    shape = [4 * TILE, 2 * TILE]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, TILE, dim=-1)

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, shape[0] // TILE - 1))})
    spec = ttnn.ShardSpec(grid, [TILE, shape[-1]], ttnn.ShardOrientation.COL_MAJOR)
    in_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)

    tt_in = _from_torch(t, device, mc=in_cfg)
    tt_out = ttnn.split(tt_in, TILE, dim=-1)
    for tt_t in tt_out:
        mc = tt_t.memory_config()
        assert mc.is_sharded()
        assert mc.shard_spec.orientation == ttnn.ShardOrientation.COL_MAJOR
    _check(tt_out, ref)


# 10. Various ranks and dimensions.


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


# 11. Unequal splits (split_sizes list).


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


# 12. N-way equal TILE splits via native kernel (no fallback path).


@pytest.mark.parametrize(
    "shape, num_chunks, dim",
    [
        ([2 * TILE, 3 * TILE], 3, -1),  # 2D, 3-way
        ([2 * TILE, 4 * TILE], 4, -1),  # 2D, 4-way
        ([1, 2, 2 * TILE, 4 * TILE], 4, -1),  # 4D, 4-way last dim
    ],
)
def test_tile_n_way_equal_split(device, shape, num_chunks, dim):
    """Equal N-way TILE split uses the native N-chunk kernel."""
    torch.manual_seed(19)
    t = torch.randn(shape, dtype=torch.bfloat16)
    split_size = shape[dim if dim >= 0 else len(shape) + dim] // num_chunks
    ref = torch.split(t, split_size, dim=dim)
    tt_in = _from_torch(t, device, mc=L1)
    tt_out = ttnn.split(tt_in, split_size, dim=dim, memory_config=L1)
    assert len(tt_out) == num_chunks
    _check(tt_out, ref)


# 13. Fallback batching boundary: num_chunks > SPLIT_BATCH_SIZE=64 triggers the sqrt-N batching path.


@pytest.mark.parametrize("num_chunks", [64, 65, 130])
def test_fallback_batching_boundary(device, num_chunks):
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


# 14. Rank-5 input — verifies split's dim-normalization on higher ranks.


@pytest.mark.parametrize(
    "shape, split_size, dim",
    [
        ([1, 2, 3, 32, 64], TILE, -1),
        ([2, 1, 3, 64, 32], TILE, -2),
    ],
)
def test_rank5_interleaved(device, shape, split_size, dim):
    """Rank-5 TILE + interleaved input, split on tile-aligned dim."""
    torch.manual_seed(27)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=dim)
    tt_in = _from_torch(t, device, mc=L1)
    tt_out = ttnn.split(tt_in, split_size, dim=dim, memory_config=L1)
    _check(tt_out, ref)


# 15. N=1 single-chunk split (trivial identity) — must not crash and must return input unchanged.


def test_single_chunk_identity(device):
    """split_sizes=[dim_size] on dim yields one output identical to the input."""
    torch.manual_seed(28)
    shape = [2 * TILE, 2 * TILE]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, 2 * TILE, dim=-1)
    assert len(ref) == 1
    tt_in = _from_torch(t, device, mc=L1)
    tt_out = ttnn.split(tt_in, 2 * TILE, dim=-1, memory_config=L1)
    _check(tt_out, ref)


# 16. Multi-batch / multi-channel sharded inputs (N>1 or C>1) — verify rank-4 sharding is preserved through split.


@pytest.mark.parametrize(
    "shape, sharding, split_size",
    [
        pytest.param([2, 1, 64, 64], "height", TILE, id="N2_H_shard"),
        pytest.param([1, 2, 64, 64], "block", TILE, id="C2_B_shard"),
        pytest.param([2, 2, 32, 64], "height", TILE, id="N2C2_H_shard"),
    ],
)
def test_multi_batch_channel_sharded(device, shape, sharding, split_size):
    """N>1 and/or C>1 sharded input, split on last dim; validates rank-4 padded-shape flattening."""
    torch.manual_seed(29)
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, split_size, dim=-1)

    compute_grid = device.compute_with_storage_grid_size()
    flat_h = shape[0] * shape[1] * shape[2]
    if sharding == "height":
        ncores = flat_h // TILE
        shard_grid = ttnn.num_cores_to_corerangeset(ncores, compute_grid, True)
        spec = ttnn.ShardSpec(shard_grid, [TILE, shape[-1]], ttnn.ShardOrientation.ROW_MAJOR)
        in_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    else:
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        spec = ttnn.ShardSpec(shard_grid, [flat_h // 2, shape[-1] // 2], ttnn.ShardOrientation.ROW_MAJOR)
        in_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)

    tt_in = _from_torch(t, device, mc=in_cfg)
    tt_out = ttnn.split(tt_in, split_size, dim=-1)
    _check(tt_out, ref)


# 17. ROW_MAJOR sharded → sharded output no-spec (composite roundtrip through synthesize_spec).


def test_row_major_sharded_to_sharded_no_spec(device):
    """RM BLOCK_SHARDED input → BLOCK_SHARDED output no-spec; exercises RM composite path."""
    torch.manual_seed(30)
    shape = [1, 1, 64, 128]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, 64, dim=-1)

    in_cfg = _make_block_sharded_cfg(2, 2, 32, 64)
    out_no_spec = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)

    tt_in = _from_torch(t, device, layout=ttnn.ROW_MAJOR_LAYOUT, mc=in_cfg)
    tt_out = ttnn.split(tt_in, 64, dim=-1, memory_config=out_no_spec)
    for tt_t in tt_out:
        mc = tt_t.memory_config()
        assert mc.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
        assert mc.shard_spec is not None
    _check(tt_out, ref)


# 18. bfloat8_b dtype coverage (TILE only; block-quantized dtype requires TILE + PCC-based check).


def test_bfloat8_b_dtype_tile(device):
    """bfloat8_b split — must go through TILE path; PCC-based check (bf8_b is block-quantized)."""
    torch.manual_seed(31)
    shape = [2 * TILE, 2 * TILE]
    t = torch.randn(shape, dtype=torch.bfloat16)

    tt_in = ttnn.from_torch(t, dtype=ttnn.bfloat8_b, device=device, memory_config=L1, layout=ttnn.TILE_LAYOUT)
    tt_out = ttnn.split(tt_in, TILE, dim=-1, memory_config=L1)
    got = torch.cat([ttnn.to_torch(c) for c in tt_out], dim=-1).float()
    assert list(got.shape) == list(t.shape)
    assert_with_pcc(t.float(), got, 0.9999)


# 19. Native single-core sharded input — 1-core HEIGHT_SHARDED edge case.


def test_native_single_core_sharded_input(device):
    """1-core HEIGHT_SHARDED input, split on last dim; verifies degenerate shard grid."""
    torch.manual_seed(32)
    shape = [TILE, 2 * TILE]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.split(t, TILE, dim=-1)

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    spec = ttnn.ShardSpec(grid, [TILE, 2 * TILE], ttnn.ShardOrientation.ROW_MAJOR)
    in_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)

    tt_in = _from_torch(t, device, mc=in_cfg)
    tt_out = ttnn.split(tt_in, TILE, dim=-1)
    _check(tt_out, ref)
