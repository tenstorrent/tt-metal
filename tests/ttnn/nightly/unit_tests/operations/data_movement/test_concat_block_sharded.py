# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for block-sharded ttnn.concat.

Tests cover:
  - Width concat (dim=3) and height concat (dim=2) with block-sharded s2s
  - Batch (dim=0) and channel (dim=1) concat via S2I fallback
  - RM and TILE layouts
  - Various grid sizes (1x2, 2x1, 2x2, 2x4, 4x2)
  - 2, 3, and 4 input tensors
  - Different input shapes (symmetric and asymmetric along concat dim)
  - Block-sharded to interleaved (S2I) output
  - Interleaved to block-sharded (I2S) output

Restrictions of block-sharded concat:
  - s2s only for dim=2 (H) and dim=3 (W); dims 0/1 go through unshard path
  - Grid must be a single rectangular CoreRange
  - All inputs must share the same grid and memory layout
  - TILE layout requires tile-aligned shard dims (multiples of 32)
  - Max 16 input tensors (CB limit)
  - groups > 1 not supported for block sharding
"""

import pytest
import torch
import ttnn


def random_tensor(shape, dtype=torch.bfloat16):
    return torch.randn(shape, dtype=dtype)


def assert_equal(expected, actual):
    """Strict equality check — no PCC tolerance."""
    if not torch.equal(expected, actual):
        diff = (expected - actual).abs()
        max_atol = diff.max().item()
        raise AssertionError(f"Tensors not equal. Max abs diff: {max_atol}, " f"shape: {expected.shape}")


def make_block_sharded_config(shard_shape, grid):
    return ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=True,
    )


def to_block_sharded(torch_tensor, device, layout, shard_shape, grid):
    mem = make_block_sharded_config(shard_shape, grid)
    tt = ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=ttnn.bfloat16)
    return ttnn.to_memory_config(tt, mem)


# ---------------------------------------------------------------------------
# 1. Width concat (dim=3) — block-sharded s2s
# ---------------------------------------------------------------------------


class TestBlockShardedWidthConcat:
    """Block-sharded inputs → block-sharded output, concat on width (dim=3)."""

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_2x2_grid_2_tensors(self, device, layout):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard = (32, 32)
        out_shard = (32, 64)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=3)

        a = to_block_sharded(ta, device, layout, shard, grid)
        b = to_block_sharded(tb, device, layout, shard, grid)
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat([a, b], dim=3, memory_config=out_mem))
        assert_equal(expected, result)

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_2x2_grid_3_tensors(self, device, layout):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard = (32, 32)
        out_shard = (32, 96)

        inputs = [random_tensor(shape) for _ in range(3)]
        expected = torch.concat(inputs, dim=3)

        tt_inputs = [to_block_sharded(t, device, layout, shard, grid) for t in inputs]
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat(tt_inputs, dim=3, memory_config=out_mem))
        assert_equal(expected, result)

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_2x2_grid_4_tensors(self, device, layout):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard = (32, 32)
        out_shard = (32, 128)

        inputs = [random_tensor(shape) for _ in range(4)]
        expected = torch.concat(inputs, dim=3)

        tt_inputs = [to_block_sharded(t, device, layout, shard, grid) for t in inputs]
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat(tt_inputs, dim=3, memory_config=out_mem))
        assert_equal(expected, result)

    def test_1x2_grid(self, device):
        """1 row, 2 columns — height not split, only width split."""
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        shape = (1, 1, 32, 64)
        shard = (32, 32)
        out_shard = (32, 64)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=3)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat([a, b], dim=3, memory_config=out_mem))
        assert_equal(expected, result)

    def test_2x1_grid(self, device):
        """2 rows, 1 column — width not split, only height split."""
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
        shape = (1, 1, 64, 32)
        shard = (32, 32)
        out_shard = (32, 64)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=3)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat([a, b], dim=3, memory_config=out_mem))
        assert_equal(expected, result)

    def test_2x4_grid(self, device):
        """2 rows, 4 columns — wide grid."""
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1))})
        shape = (1, 1, 64, 128)
        shard = (32, 32)
        out_shard = (32, 64)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=3)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat([a, b], dim=3, memory_config=out_mem))
        assert_equal(expected, result)

    def test_4x2_grid(self, device):
        """4 rows, 2 columns — tall grid."""
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3))})
        shape = (1, 1, 128, 64)
        shard = (32, 32)
        out_shard = (32, 64)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=3)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat([a, b], dim=3, memory_config=out_mem))
        assert_equal(expected, result)

    def test_asymmetric_widths(self, device):
        """Inputs with different widths along the concat dim."""
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape_a = (1, 1, 64, 64)
        shape_b = (1, 1, 64, 128)
        shard_a = (32, 32)
        shard_b = (32, 64)
        out_shard = (32, 96)

        ta, tb = random_tensor(shape_a), random_tensor(shape_b)
        expected = torch.concat([ta, tb], dim=3)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard_a, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard_b, grid)
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat([a, b], dim=3, memory_config=out_mem))
        assert_equal(expected, result)

    def test_large_shard(self, device):
        """Larger shard shape — 64x64 per core on 2x2."""
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 128, 128)
        shard = (64, 64)
        out_shard = (64, 128)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=3)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat([a, b], dim=3, memory_config=out_mem))
        assert_equal(expected, result)

    def test_batch_gt_1(self, device):
        """Batch > 1 tensor, width concat."""
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (2, 1, 64, 64)
        shard = (64, 32)
        out_shard = (64, 64)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=3)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat([a, b], dim=3, memory_config=out_mem))
        assert_equal(expected, result)


# ---------------------------------------------------------------------------
# 2. Height concat (dim=2) — block-sharded s2s
# ---------------------------------------------------------------------------


class TestBlockShardedHeightConcat:
    """Block-sharded inputs → block-sharded output, concat on height (dim=2)."""

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_2x2_grid_2_tensors(self, device, layout):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard = (32, 32)
        out_shard = (64, 32)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=2)

        a = to_block_sharded(ta, device, layout, shard, grid)
        b = to_block_sharded(tb, device, layout, shard, grid)
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat([a, b], dim=2, memory_config=out_mem))
        assert_equal(expected, result)

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_2x2_grid_3_tensors(self, device, layout):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard = (32, 32)
        out_shard = (96, 32)

        inputs = [random_tensor(shape) for _ in range(3)]
        expected = torch.concat(inputs, dim=2)

        tt_inputs = [to_block_sharded(t, device, layout, shard, grid) for t in inputs]
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat(tt_inputs, dim=2, memory_config=out_mem))
        assert_equal(expected, result)

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_2x2_grid_4_tensors(self, device, layout):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard = (32, 32)
        out_shard = (128, 32)

        inputs = [random_tensor(shape) for _ in range(4)]
        expected = torch.concat(inputs, dim=2)

        tt_inputs = [to_block_sharded(t, device, layout, shard, grid) for t in inputs]
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat(tt_inputs, dim=2, memory_config=out_mem))
        assert_equal(expected, result)

    def test_1x2_grid(self, device):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        shape = (1, 1, 32, 64)
        shard = (32, 32)
        out_shard = (64, 32)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=2)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat([a, b], dim=2, memory_config=out_mem))
        assert_equal(expected, result)

    def test_2x1_grid(self, device):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
        shape = (1, 1, 64, 32)
        shard = (32, 32)
        out_shard = (64, 32)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=2)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat([a, b], dim=2, memory_config=out_mem))
        assert_equal(expected, result)

    def test_2x4_grid(self, device):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1))})
        shape = (1, 1, 64, 128)
        shard = (32, 32)
        out_shard = (64, 32)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=2)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat([a, b], dim=2, memory_config=out_mem))
        assert_equal(expected, result)

    def test_4x2_grid(self, device):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3))})
        shape = (1, 1, 128, 64)
        shard = (32, 32)
        out_shard = (64, 32)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=2)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat([a, b], dim=2, memory_config=out_mem))
        assert_equal(expected, result)

    def test_asymmetric_heights(self, device):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape_a = (1, 1, 64, 64)
        shape_b = (1, 1, 128, 64)
        shard_a = (32, 32)
        shard_b = (64, 32)
        out_shard = (96, 32)

        ta, tb = random_tensor(shape_a), random_tensor(shape_b)
        expected = torch.concat([ta, tb], dim=2)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard_a, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard_b, grid)
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat([a, b], dim=2, memory_config=out_mem))
        assert_equal(expected, result)

    def test_large_shard(self, device):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 128, 128)
        shard = (64, 64)
        out_shard = (128, 64)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=2)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat([a, b], dim=2, memory_config=out_mem))
        assert_equal(expected, result)


# ---------------------------------------------------------------------------
# 3. Batch dim (dim=0) and channel dim (dim=1) concat
#    These go through the S2I unshard path since sharded kernels only
#    support dim=2 and dim=3.
# ---------------------------------------------------------------------------


class TestBlockShardedBatchChannelConcat:
    """Block-sharded inputs, concat on batch/channel dims.
    Output is interleaved (fallback path: unshard → concat → optionally reshard)."""

    def test_batch_concat_dim0(self, device):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard = (32, 32)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=0)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)

        output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        result = ttnn.to_torch(ttnn.concat([a, b], dim=0, memory_config=output_mem))
        assert_equal(expected, result)

    def test_channel_concat_dim1(self, device):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard = (32, 32)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=1)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)

        output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        result = ttnn.to_torch(ttnn.concat([a, b], dim=1, memory_config=output_mem))
        assert_equal(expected, result)

    def test_batch_concat_3_tensors(self, device):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard = (32, 32)

        inputs = [random_tensor(shape) for _ in range(3)]
        expected = torch.concat(inputs, dim=0)

        tt_inputs = [to_block_sharded(t, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid) for t in inputs]
        output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

        result = ttnn.to_torch(ttnn.concat(tt_inputs, dim=0, memory_config=output_mem))
        assert_equal(expected, result)


# ---------------------------------------------------------------------------
# 4. Block-sharded to interleaved (S2I)
# ---------------------------------------------------------------------------


class TestBlockShardedS2I:
    """Block-sharded inputs → interleaved output."""

    @pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1], ids=["DRAM", "L1"])
    def test_s2i_width_concat(self, device, buffer_type):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard = (32, 32)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=3)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)

        out = ttnn.concat([a, b], dim=3, memory_config=output_mem)
        assert not out.is_sharded()
        result = ttnn.to_torch(out)
        assert_equal(expected, result)

    @pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1], ids=["DRAM", "L1"])
    def test_s2i_height_concat(self, device, buffer_type):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard = (32, 32)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=2)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)

        out = ttnn.concat([a, b], dim=2, memory_config=output_mem)
        assert not out.is_sharded()
        result = ttnn.to_torch(out)
        assert_equal(expected, result)

    def test_s2i_default_output_config(self, device):
        """No memory_config specified — should default to DRAM interleaved."""
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard = (32, 32)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=3)

        a = to_block_sharded(ta, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)
        b = to_block_sharded(tb, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid)

        out = ttnn.concat([a, b], dim=3)
        assert not out.is_sharded()
        result = ttnn.to_torch(out)
        assert_equal(expected, result)

    def test_s2i_3_tensors(self, device):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard = (32, 32)

        inputs = [random_tensor(shape) for _ in range(3)]
        expected = torch.concat(inputs, dim=3)

        tt_inputs = [to_block_sharded(t, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid) for t in inputs]
        output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

        out = ttnn.concat(tt_inputs, dim=3, memory_config=output_mem)
        assert not out.is_sharded()
        result = ttnn.to_torch(out)
        assert_equal(expected, result)


# ---------------------------------------------------------------------------
# 5. Interleaved to block-sharded (I2S)
# ---------------------------------------------------------------------------


class TestInterleavedToBlockSharded:
    """Interleaved inputs → block-sharded output."""

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_i2s_width_concat(self, device, layout):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        out_shard = (32, 64)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=3)

        input_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        a = ttnn.from_torch(ta, layout=layout, device=device, dtype=ttnn.bfloat16, memory_config=input_mem)
        b = ttnn.from_torch(tb, layout=layout, device=device, dtype=ttnn.bfloat16, memory_config=input_mem)
        out_mem = make_block_sharded_config(out_shard, grid)

        out = ttnn.concat([a, b], dim=3, memory_config=out_mem)
        assert out.is_sharded()
        result = ttnn.to_torch(out)
        assert_equal(expected, result)

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_i2s_height_concat(self, device, layout):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        out_shard = (64, 32)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=2)

        input_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        a = ttnn.from_torch(ta, layout=layout, device=device, dtype=ttnn.bfloat16, memory_config=input_mem)
        b = ttnn.from_torch(tb, layout=layout, device=device, dtype=ttnn.bfloat16, memory_config=input_mem)
        out_mem = make_block_sharded_config(out_shard, grid)

        out = ttnn.concat([a, b], dim=2, memory_config=out_mem)
        assert out.is_sharded()
        result = ttnn.to_torch(out)
        assert_equal(expected, result)

    def test_i2s_batch_concat(self, device):
        """Batch concat with block-sharded output."""
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 32, 64)
        out_shard = (32, 32)

        ta, tb = random_tensor(shape), random_tensor(shape)
        expected = torch.concat([ta, tb], dim=0)

        input_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        a = ttnn.from_torch(
            ta, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=input_mem
        )
        b = ttnn.from_torch(
            tb, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=input_mem
        )
        out_mem = make_block_sharded_config(out_shard, grid)

        out = ttnn.concat([a, b], dim=0, memory_config=out_mem)
        assert out.is_sharded()
        result = ttnn.to_torch(out)
        assert_equal(expected, result)


# ---------------------------------------------------------------------------
# 6. Parametrized sweep across shapes and grids
# ---------------------------------------------------------------------------


class TestBlockShardedSweep:
    """Parametrized sweep over multiple configurations."""

    @pytest.mark.parametrize(
        "grid_spec,shape,shard",
        [
            # (grid_end_coord, tensor_shape, shard_shape)
            ((1, 1), (1, 1, 64, 64), (32, 32)),
            ((1, 1), (1, 1, 128, 128), (64, 64)),
            ((3, 1), (1, 1, 64, 128), (32, 32)),
            ((1, 3), (1, 1, 128, 64), (32, 32)),
            ((3, 3), (1, 1, 128, 128), (32, 32)),
        ],
        ids=["2x2-small", "2x2-large", "4x2", "2x4", "4x4"],
    )
    @pytest.mark.parametrize("dim", [2, 3], ids=["H", "W"])
    @pytest.mark.parametrize("num_tensors", [2, 3], ids=["2T", "3T"])
    def test_sweep(self, device, grid_spec, shape, shard, dim, num_tensors):
        gx, gy = grid_spec
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx, gy))})

        out_shard_h = shard[0] * num_tensors if dim == 2 else shard[0]
        out_shard_w = shard[1] * num_tensors if dim == 3 else shard[1]
        out_shard = (out_shard_h, out_shard_w)

        inputs = [random_tensor(shape) for _ in range(num_tensors)]
        expected = torch.concat(inputs, dim=dim)

        tt_inputs = [to_block_sharded(t, device, ttnn.ROW_MAJOR_LAYOUT, shard, grid) for t in inputs]
        out_mem = make_block_sharded_config(out_shard, grid)

        result = ttnn.to_torch(ttnn.concat(tt_inputs, dim=dim, memory_config=out_mem))
        assert_equal(expected, result)


def test_manual_block_rm(device):
    """Height concat of 3 block-sharded tensors on a 2x2 grid.

    Each input is (2, 64) = 2 rows x 64 cols.
    With a 2x2 grid, block shard shape is (1, 32) per core.
    After height concat (dim=2), output is (6, 64) with shard (3, 32).
    Expected output rows: 1s, 0s, 3s (interleaved across cores).
    """
    dim = 2
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})  # 2 X 2
    shape = [1, 1, 2, 64]
    shard_shape = (1, 32)  # H / 2 , W / 2

    a = torch.ones(shape, dtype=torch.bfloat16)
    b = torch.zeros(shape, dtype=torch.bfloat16)
    c = torch.ones(shape, dtype=torch.bfloat16) * 3

    # output should be  11111....00000....33333
    expected = torch.cat([a, b, c], dim=dim)

    tt_a = to_block_sharded(a, device, ttnn.ROW_MAJOR_LAYOUT, shard_shape, grid)
    tt_b = to_block_sharded(b, device, ttnn.ROW_MAJOR_LAYOUT, shard_shape, grid)
    tt_c = to_block_sharded(c, device, ttnn.ROW_MAJOR_LAYOUT, shard_shape, grid)

    # expected output shape is (6, 64); block-sharded over 2X2 grid: shard shape is 6/2 , 64/2
    out_shard = (3, 32)

    out_mem = make_block_sharded_config(out_shard, grid)

    result = ttnn.to_torch(ttnn.concat([tt_a, tt_b, tt_c], dim=dim, memory_config=out_mem))
    assert_equal(expected, result)
    print("manual_block_test RM PASSED")


def test_manual_block_tiled(device):
    """Height concat of 3 block-sharded TILE tensors on a 2x2 grid.

    Each input is (64, 64) in the last two dims — tile-aligned.
    With a 2x2 grid, block shard shape is (32, 32) per core.
    After height concat (dim=2), output is (192, 64) with shard (96, 32).
    """
    dim = 2
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    shape = [1, 1, 64, 64]
    shard_shape = (32, 32)

    a = torch.ones(shape, dtype=torch.bfloat16)
    b = torch.zeros(shape, dtype=torch.bfloat16)
    c = torch.ones(shape, dtype=torch.bfloat16) * 3

    expected = torch.cat([a, b, c], dim=dim)

    tt_a = to_block_sharded(a, device, ttnn.TILE_LAYOUT, shard_shape, grid)
    tt_b = to_block_sharded(b, device, ttnn.TILE_LAYOUT, shard_shape, grid)
    tt_c = to_block_sharded(c, device, ttnn.TILE_LAYOUT, shard_shape, grid)

    out_shard = (96, 32)
    out_mem = make_block_sharded_config(out_shard, grid)

    result = ttnn.to_torch(ttnn.concat([tt_a, tt_b, tt_c], dim=dim, memory_config=out_mem))
    assert_equal(expected, result)
    print("manual_block_test TILE PASSED")
