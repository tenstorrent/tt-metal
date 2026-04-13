# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ttnn.concat across various memory configurations and layouts.

Covers:
- Interleaved DRAM / L1 with ROW_MAJOR and TILE layouts
- Height-sharded, width-sharded (sharded-to-sharded and sharded-to-interleaved)
- Mixed dtypes (bfloat16, int32, uint32, bfloat8_b)
- Various ranks (1D through 5D)
- Concat on every valid dimension
- Edge cases: single tensor, many tensors (batched concat), unaligned shapes
- Groups parameter for height-sharded width concat
- Known unsupported cases (mismatched ranks, non-rectangular grids, etc.)
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint16:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**15), 2**15, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**15, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


def run_concat_test(device, shapes, dim, layout, dtype, memory_config=None, pcc=0.99):
    torch_inputs = [random_torch_tensor(dtype, s) for s in shapes]
    torch_out = torch.concat(torch_inputs, dim=dim)

    ttnn_inputs = [
        ttnn.from_torch(t, layout=layout, device=device, dtype=dtype, memory_config=memory_config) for t in torch_inputs
    ]

    ttnn_out = ttnn.concat(ttnn_inputs, dim=dim, memory_config=memory_config)
    ttnn_out = ttnn.to_torch(ttnn_out)

    if dtype == ttnn.bfloat8_b:
        assert_with_pcc(torch_out, ttnn_out, pcc)
    else:
        assert_equal(torch_out, ttnn_out)


# ===========================================================================
# 1. Interleaved memory configs (DRAM and L1) with both layouts
# ===========================================================================


class TestInterleavedConcat:
    """Tests for non-sharded (interleaved) concat on DRAM and L1."""

    @pytest.mark.parametrize(
        "memory_config",
        [
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ],
        ids=["DRAM", "L1"],
    )
    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    @pytest.mark.parametrize("dim", [0, 1, 2, 3])
    @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
    def test_basic_4d(self, device, memory_config, layout, dim, dtype):
        shapes = [(1, 2, 32, 64), (1, 2, 32, 64)]
        run_concat_test(device, shapes, dim, layout, dtype, memory_config)

    @pytest.mark.parametrize(
        "memory_config",
        [
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ],
        ids=["DRAM", "L1"],
    )
    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_asymmetric_concat_dim(self, device, memory_config, layout):
        """Tensors differ only on the concat dimension."""
        shapes = [(1, 1, 32, 64), (1, 1, 32, 128)]
        run_concat_test(device, shapes, dim=3, layout=layout, dtype=ttnn.bfloat16, memory_config=memory_config)

    @pytest.mark.parametrize(
        "memory_config",
        [
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ],
        ids=["DRAM", "L1"],
    )
    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_three_tensors(self, device, memory_config, layout):
        """Concat 3 tensors on dim 2."""
        shapes = [(1, 1, 32, 64), (1, 1, 64, 64), (1, 1, 32, 64)]
        run_concat_test(device, shapes, dim=2, layout=layout, dtype=ttnn.bfloat16, memory_config=memory_config)


# ===========================================================================
# 2. Layout variations
# ===========================================================================


class TestLayoutConcat:
    """Tests exercising specific layout-related paths."""

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    @pytest.mark.parametrize(
        "shapes, dim",
        [
            ([(32,), (64,)], 0),
            ([(96,), (96,)], 0),
            ([(1,), (1,)], 0),
        ],
        ids=["32+64", "96+96", "1+1"],
    )
    def test_1d_concat(self, device, layout, shapes, dim):
        """1D concat exercises the unsqueeze/squeeze workaround path."""
        run_concat_test(device, shapes, dim, layout, ttnn.bfloat16)

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_2d_concat_dim0(self, device, layout):
        run_concat_test(device, [(32, 64), (32, 64)], dim=0, layout=layout, dtype=ttnn.bfloat16)

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_2d_concat_dim1(self, device, layout):
        run_concat_test(device, [(32, 64), (32, 128)], dim=1, layout=layout, dtype=ttnn.bfloat16)

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_3d_concat(self, device, layout):
        run_concat_test(device, [(2, 32, 64), (2, 32, 64)], dim=0, layout=layout, dtype=ttnn.bfloat16)

    @pytest.mark.parametrize("dim", [0, 1, 2, 3, 4])
    def test_5d_concat(self, device, dim):
        shapes = [(1, 1, 1, 32, 64)] * 2
        run_concat_test(device, shapes, dim, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)

    @pytest.mark.parametrize(
        "shapes, dim",
        [
            ([(1, 1, 20, 32), (1, 1, 20, 32)], 2),
            ([(1, 1, 20, 32), (1, 1, 20, 32)], 3),
        ],
        ids=["non_tile_aligned_h", "non_tile_aligned_w"],
    )
    def test_tile_layout_non_tile_aligned(self, device, shapes, dim):
        """Tile layout with non-tile-aligned dims triggers untilize/retilize path."""
        run_concat_test(device, shapes, dim, ttnn.TILE_LAYOUT, ttnn.bfloat16)


# ===========================================================================
# 3. Dtype variations
# ===========================================================================


class TestDtypeConcat:
    """Test all supported dtypes."""

    @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint32], ids=["bf16", "i32", "u32"])
    def test_dtype_tile(self, device, dtype):
        run_concat_test(device, [(1, 1, 32, 64), (1, 1, 32, 64)], dim=3, layout=ttnn.TILE_LAYOUT, dtype=dtype)

    @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint32], ids=["bf16", "i32", "u32"])
    def test_dtype_rm(self, device, dtype):
        run_concat_test(device, [(1, 1, 32, 64), (1, 1, 32, 64)], dim=3, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype)

    def test_bfloat8_b_tile(self, device):
        """bfloat8_b only works with TILE_LAYOUT."""
        run_concat_test(
            device, [(1, 1, 32, 64), (1, 1, 32, 64)], dim=3, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, pcc=0.99
        )


# ===========================================================================
# 4. Height-sharded concat (sharded-to-sharded)
# ===========================================================================


class TestHeightShardedConcat:
    """Height-sharded inputs with width concat (dim=-1) and height concat on width-sharded."""

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32], ids=["bf16", "i32"])
    def test_two_tensor_width_concat_height_sharded(self, device, layout, dtype):
        """2-tensor width concat on height-sharded tensors."""
        shape = (1, 1, 64, 64)
        shard_shape = (32, 64)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
        output_shard_shape = (32, 128)

        input_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        output_mem = ttnn.create_sharded_memory_config(
            output_shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        torch_a = random_torch_tensor(dtype, shape)
        torch_b = random_torch_tensor(dtype, shape)
        torch_out = torch.concat([torch_a, torch_b], dim=3)

        ttnn_a = ttnn.from_torch(torch_a, layout=layout, device=device, dtype=dtype)
        ttnn_a = ttnn.to_memory_config(ttnn_a, input_mem)
        ttnn_b = ttnn.from_torch(torch_b, layout=layout, device=device, dtype=dtype)
        ttnn_b = ttnn.to_memory_config(ttnn_b, input_mem)

        ttnn_out = ttnn.concat([ttnn_a, ttnn_b], dim=3, memory_config=output_mem)
        ttnn_out = ttnn.to_torch(ttnn_out)

        assert_equal(torch_out, ttnn_out)

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_two_tensor_asymmetric_width_concat(self, device, layout):
        """2-tensor width concat with different widths on height-sharded."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})

        shape_a = (1, 1, 64, 64)
        shape_b = (1, 1, 64, 32)
        shard_a = (32, 64)
        shard_b = (32, 32)
        output_shard = (32, 96)

        mem_a = ttnn.create_sharded_memory_config(
            shard_a, core_grid=shard_grid, strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
        )
        mem_b = ttnn.create_sharded_memory_config(
            shard_b, core_grid=shard_grid, strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
        )
        output_mem = ttnn.create_sharded_memory_config(
            output_shard,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        torch_a = random_torch_tensor(ttnn.bfloat16, shape_a)
        torch_b = random_torch_tensor(ttnn.bfloat16, shape_b)
        torch_out = torch.concat([torch_a, torch_b], dim=3)

        ttnn_a = ttnn.from_torch(torch_a, layout=layout, device=device, dtype=ttnn.bfloat16)
        ttnn_a = ttnn.to_memory_config(ttnn_a, mem_a)
        ttnn_b = ttnn.from_torch(torch_b, layout=layout, device=device, dtype=ttnn.bfloat16)
        ttnn_b = ttnn.to_memory_config(ttnn_b, mem_b)

        ttnn_out = ttnn.concat([ttnn_a, ttnn_b], dim=3, memory_config=output_mem)
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    def test_multi_tensor_height_sharded_rm(self, device):
        """3-tensor width concat on height-sharded RM tensors (multi path)."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
        shape = (1, 1, 16, 16)
        shard_shape = (8, 16)
        output_shard = (8, 48)

        input_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        output_mem = ttnn.create_sharded_memory_config(
            output_shard,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        torch_inputs = [random_torch_tensor(ttnn.bfloat16, shape) for _ in range(3)]
        torch_out = torch.concat(torch_inputs, dim=3)

        ttnn_inputs = [
            ttnn.to_memory_config(
                ttnn.from_torch(t, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16), input_mem
            )
            for t in torch_inputs
        ]
        ttnn_out = ttnn.concat(ttnn_inputs, dim=3, memory_config=output_mem)
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    def test_multi_tensor_height_sharded_tiled(self, device):
        """3-tensor width concat on height-sharded TILE tensors (multi path)."""
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 0)),
            }
        )
        shapes = [(1, 1, 256, 96), (1, 1, 256, 64), (1, 1, 256, 32)]
        shard_shapes = [(64, 96), (64, 64), (64, 32)]
        output_shard = (64, 192)

        torch_inputs = [random_torch_tensor(ttnn.bfloat16, s) for s in shapes]
        torch_out = torch.concat(torch_inputs, dim=3)

        ttnn_inputs = []
        for t, ss in zip(torch_inputs, shard_shapes):
            mem = ttnn.create_sharded_memory_config(
                ss, core_grid=shard_grid, strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
            )
            tt = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
            ttnn_inputs.append(ttnn.to_memory_config(tt, mem))

        output_mem = ttnn.create_sharded_memory_config(
            output_shard,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        ttnn_out = ttnn.concat(ttnn_inputs, dim=3, memory_config=output_mem)
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)


# ===========================================================================
# 5. Width-sharded concat (height concat on width-sharded)
# ===========================================================================


class TestWidthShardedConcat:
    """Width-sharded inputs with height concat (dim=-2)."""

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_two_tensor_width_sharded_height_concat(self, device, layout):
        """2-tensor height concat on width-sharded tensors — previously blocked, now routed to MultiProgramFactory."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})
        shape = (1, 1, 32, 128)
        shard_shape = (32, 32)
        output_shard = (64, 32)

        input_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        output_mem = ttnn.create_sharded_memory_config(
            output_shard,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)
        torch_b = random_torch_tensor(ttnn.bfloat16, shape)
        torch_out = torch.concat([torch_a, torch_b], dim=2)

        ttnn_a = ttnn.to_memory_config(
            ttnn.from_torch(torch_a, layout=layout, device=device, dtype=ttnn.bfloat16), input_mem
        )
        ttnn_b = ttnn.to_memory_config(
            ttnn.from_torch(torch_b, layout=layout, device=device, dtype=ttnn.bfloat16), input_mem
        )

        ttnn_out = ttnn.concat([ttnn_a, ttnn_b], dim=2, memory_config=output_mem)
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    def test_multi_tensor_width_sharded_rm(self, device):
        """3-tensor height concat on width-sharded RM tensors."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})
        shapes = [(1, 1, 8, 64), (1, 1, 7, 64), (1, 1, 23, 64)]
        shard_shapes = [(8, 16), (7, 16), (23, 16)]
        output_shard = (38, 16)

        torch_inputs = [random_torch_tensor(ttnn.bfloat16, s) for s in shapes]
        torch_out = torch.concat(torch_inputs, dim=2)

        ttnn_inputs = []
        for t, ss in zip(torch_inputs, shard_shapes):
            mem = ttnn.create_sharded_memory_config(
                ss, core_grid=shard_grid, strategy=ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
            )
            tt = ttnn.from_torch(t, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
            ttnn_inputs.append(ttnn.to_memory_config(tt, mem))

        output_mem = ttnn.create_sharded_memory_config(
            output_shard,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        ttnn_out = ttnn.concat(ttnn_inputs, dim=2, memory_config=output_mem)
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    def test_multi_tensor_width_sharded_tiled(self, device):
        """3-tensor height concat on width-sharded TILE tensors."""
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3)),
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 1)),
            }
        )
        shapes = [(1, 1, 32, 512), (1, 1, 64, 512), (1, 1, 96, 512)]
        shard_shapes = [(32, 64), (64, 64), (96, 64)]
        output_shard = (192, 64)

        torch_inputs = [random_torch_tensor(ttnn.bfloat16, s) for s in shapes]
        torch_out = torch.concat(torch_inputs, dim=2)

        ttnn_inputs = []
        for t, ss in zip(torch_inputs, shard_shapes):
            mem = ttnn.create_sharded_memory_config(
                ss, core_grid=shard_grid, strategy=ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
            )
            tt = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
            ttnn_inputs.append(ttnn.to_memory_config(tt, mem))

        output_mem = ttnn.create_sharded_memory_config(
            output_shard,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        ttnn_out = ttnn.concat(ttnn_inputs, dim=2, memory_config=output_mem)
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)


# ===========================================================================
# 5b. Block-sharded concat
# ===========================================================================


class TestBlockShardedConcat:
    """Block-sharded inputs with block-sharded output (s2s)."""

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_block_sharded_width_concat(self, device, layout):
        """2 block-sharded tensors, width concat (dim=3)."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard_shape = (32, 32)
        output_shard_shape = (32, 64)

        input_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )
        output_mem = ttnn.create_sharded_memory_config(
            output_shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)
        torch_b = random_torch_tensor(ttnn.bfloat16, shape)
        torch_out = torch.concat([torch_a, torch_b], dim=3)

        ttnn_a = ttnn.to_memory_config(
            ttnn.from_torch(torch_a, layout=layout, device=device, dtype=ttnn.bfloat16), input_mem
        )
        ttnn_b = ttnn.to_memory_config(
            ttnn.from_torch(torch_b, layout=layout, device=device, dtype=ttnn.bfloat16), input_mem
        )

        ttnn_out = ttnn.concat([ttnn_a, ttnn_b], dim=3, memory_config=output_mem)
        assert ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_block_sharded_height_concat(self, device, layout):
        """2 block-sharded tensors, height concat (dim=2)."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard_shape = (32, 32)
        output_shard_shape = (64, 32)

        input_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )
        output_mem = ttnn.create_sharded_memory_config(
            output_shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)
        torch_b = random_torch_tensor(ttnn.bfloat16, shape)
        torch_out = torch.concat([torch_a, torch_b], dim=2)

        ttnn_a = ttnn.to_memory_config(
            ttnn.from_torch(torch_a, layout=layout, device=device, dtype=ttnn.bfloat16), input_mem
        )
        ttnn_b = ttnn.to_memory_config(
            ttnn.from_torch(torch_b, layout=layout, device=device, dtype=ttnn.bfloat16), input_mem
        )

        ttnn_out = ttnn.concat([ttnn_a, ttnn_b], dim=2, memory_config=output_mem)
        assert ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    def test_block_sharded_three_tensors_width_concat(self, device):
        """3 block-sharded tensors, width concat."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard_shape = (32, 32)
        output_shard_shape = (32, 96)

        input_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )
        output_mem = ttnn.create_sharded_memory_config(
            output_shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )

        torch_inputs = [random_torch_tensor(ttnn.bfloat16, shape) for _ in range(3)]
        torch_out = torch.concat(torch_inputs, dim=3)

        ttnn_inputs = [
            ttnn.to_memory_config(
                ttnn.from_torch(t, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16), input_mem
            )
            for t in torch_inputs
        ]

        ttnn_out = ttnn.concat(ttnn_inputs, dim=3, memory_config=output_mem)
        assert ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    def test_block_sharded_three_tensors_height_concat(self, device):
        """3 block-sharded tensors, height concat."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard_shape = (32, 32)
        output_shard_shape = (96, 32)

        input_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )
        output_mem = ttnn.create_sharded_memory_config(
            output_shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )

        torch_inputs = [random_torch_tensor(ttnn.bfloat16, shape) for _ in range(3)]
        torch_out = torch.concat(torch_inputs, dim=2)

        ttnn_inputs = [
            ttnn.to_memory_config(
                ttnn.from_torch(t, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16), input_mem
            )
            for t in torch_inputs
        ]

        ttnn_out = ttnn.concat(ttnn_inputs, dim=2, memory_config=output_mem)
        assert ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    @pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1], ids=["DRAM", "L1"])
    def test_block_sharded_s2i_width_concat(self, device, buffer_type):
        """Block-sharded inputs → interleaved output, width concat."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard_shape = (32, 32)

        input_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )
        output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)
        torch_b = random_torch_tensor(ttnn.bfloat16, shape)
        torch_out = torch.concat([torch_a, torch_b], dim=3)

        ttnn_a = ttnn.to_memory_config(
            ttnn.from_torch(torch_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16), input_mem
        )
        ttnn_b = ttnn.to_memory_config(
            ttnn.from_torch(torch_b, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16), input_mem
        )

        ttnn_out = ttnn.concat([ttnn_a, ttnn_b], dim=3, memory_config=output_mem)
        assert not ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    @pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1], ids=["DRAM", "L1"])
    def test_block_sharded_s2i_height_concat(self, device, buffer_type):
        """Block-sharded inputs → interleaved output, height concat."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard_shape = (32, 32)

        input_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )
        output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)
        torch_b = random_torch_tensor(ttnn.bfloat16, shape)
        torch_out = torch.concat([torch_a, torch_b], dim=2)

        ttnn_a = ttnn.to_memory_config(
            ttnn.from_torch(torch_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16), input_mem
        )
        ttnn_b = ttnn.to_memory_config(
            ttnn.from_torch(torch_b, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16), input_mem
        )

        ttnn_out = ttnn.concat([ttnn_a, ttnn_b], dim=2, memory_config=output_mem)
        assert not ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)


# ===========================================================================
# 6. Sharded-to-interleaved and interleaved-to-sharded concat
# ===========================================================================


class TestShardedToInterleavedConcat:
    """Sharded inputs with interleaved output."""

    def test_s2i_sharded_inputs_default_output_config(self, device):
        """Sharded inputs with NO memory_config specified.
        Default is DRAM_MEMORY_CONFIG (interleaved). Should do s2s concat then convert output."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
        shape = (1, 1, 64, 32)
        shard_shape = (32, 32)

        input_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)
        torch_b = random_torch_tensor(ttnn.bfloat16, shape)
        torch_out = torch.concat([torch_a, torch_b], dim=3)

        ttnn_a = ttnn.to_memory_config(
            ttnn.from_torch(torch_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16), input_mem
        )
        ttnn_b = ttnn.to_memory_config(
            ttnn.from_torch(torch_b, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16), input_mem
        )

        # No memory_config specified — defaults to DRAM interleaved
        ttnn_out = ttnn.concat([ttnn_a, ttnn_b], dim=3)
        assert not ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    @pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1], ids=["DRAM", "L1"])
    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_s2i_height_sharded_width_concat(self, device, buffer_type, layout):
        """Height-sharded inputs → interleaved output, width concat."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
        shape = (1, 1, 64, 32)
        shard_shape = (32, 32)

        input_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)
        torch_b = random_torch_tensor(ttnn.bfloat16, shape)
        torch_out = torch.concat([torch_a, torch_b], dim=3)

        ttnn_a = ttnn.to_memory_config(
            ttnn.from_torch(torch_a, layout=layout, device=device, dtype=ttnn.bfloat16), input_mem
        )
        ttnn_b = ttnn.to_memory_config(
            ttnn.from_torch(torch_b, layout=layout, device=device, dtype=ttnn.bfloat16), input_mem
        )

        ttnn_out = ttnn.concat([ttnn_a, ttnn_b], dim=3, memory_config=output_mem)
        assert not ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    @pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1], ids=["DRAM", "L1"])
    def test_s2i_width_sharded_height_concat_3_tensors(self, device, buffer_type):
        """Width-sharded inputs → interleaved output, height concat, 3 tensors."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})
        shapes = [(1, 1, 8, 64), (1, 1, 8, 64), (1, 1, 8, 64)]
        shard_shape = (8, 16)

        output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)

        torch_inputs = [random_torch_tensor(ttnn.bfloat16, s) for s in shapes]
        torch_out = torch.concat(torch_inputs, dim=2)

        ttnn_inputs = []
        for t in torch_inputs:
            mem = ttnn.create_sharded_memory_config(
                shard_shape,
                core_grid=shard_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            tt = ttnn.from_torch(t, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
            ttnn_inputs.append(ttnn.to_memory_config(tt, mem))

        ttnn_out = ttnn.concat(ttnn_inputs, dim=2, memory_config=output_mem)
        assert not ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    @pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1], ids=["DRAM", "L1"])
    def test_s2i_width_sharded_height_concat_2_tensors(self, device, buffer_type):
        """2 width-sharded inputs → interleaved output, height concat."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})
        shape = (1, 1, 8, 64)
        shard_shape = (8, 16)

        output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)
        torch_b = random_torch_tensor(ttnn.bfloat16, shape)
        torch_out = torch.concat([torch_a, torch_b], dim=2)

        input_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )

        ttnn_a = ttnn.to_memory_config(
            ttnn.from_torch(torch_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16), input_mem
        )
        ttnn_b = ttnn.to_memory_config(
            ttnn.from_torch(torch_b, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16), input_mem
        )

        ttnn_out = ttnn.concat([ttnn_a, ttnn_b], dim=2, memory_config=output_mem)
        assert not ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)


class TestInterleavedToShardedConcat:
    """Interleaved inputs with sharded output.

    Implemented by performing interleaved concat first, then sharding the result.
    """

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_i2s_to_height_sharded(self, device, layout):
        """Interleaved inputs → height-sharded output, width concat."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
        shape = (1, 1, 64, 32)
        output_shard_shape = (32, 64)

        output_mem = ttnn.create_sharded_memory_config(
            output_shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        input_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)
        torch_b = random_torch_tensor(ttnn.bfloat16, shape)
        torch_out = torch.concat([torch_a, torch_b], dim=3)

        ttnn_a = ttnn.from_torch(torch_a, layout=layout, device=device, dtype=ttnn.bfloat16, memory_config=input_mem)
        ttnn_b = ttnn.from_torch(torch_b, layout=layout, device=device, dtype=ttnn.bfloat16, memory_config=input_mem)

        ttnn_out = ttnn.concat([ttnn_a, ttnn_b], dim=3, memory_config=output_mem)
        assert ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_i2s_to_width_sharded(self, device, layout):
        """Interleaved inputs → width-sharded output, height concat."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})
        shape = (1, 1, 32, 128)
        output_shard_shape = (64, 32)

        output_mem = ttnn.create_sharded_memory_config(
            output_shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        input_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)
        torch_b = random_torch_tensor(ttnn.bfloat16, shape)
        torch_out = torch.concat([torch_a, torch_b], dim=2)

        ttnn_a = ttnn.from_torch(torch_a, layout=layout, device=device, dtype=ttnn.bfloat16, memory_config=input_mem)
        ttnn_b = ttnn.from_torch(torch_b, layout=layout, device=device, dtype=ttnn.bfloat16, memory_config=input_mem)

        ttnn_out = ttnn.concat([ttnn_a, ttnn_b], dim=2, memory_config=output_mem)
        assert ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    @pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1], ids=["DRAM", "L1"])
    def test_i2s_to_height_sharded_l1_input(self, device, buffer_type):
        """Interleaved inputs from different buffer types → height-sharded output."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
        shape = (1, 1, 64, 32)
        output_shard_shape = (32, 64)

        output_mem = ttnn.create_sharded_memory_config(
            output_shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        input_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)
        torch_b = random_torch_tensor(ttnn.bfloat16, shape)
        torch_out = torch.concat([torch_a, torch_b], dim=3)

        ttnn_a = ttnn.from_torch(
            torch_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=input_mem
        )
        ttnn_b = ttnn.from_torch(
            torch_b, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=input_mem
        )

        ttnn_out = ttnn.concat([ttnn_a, ttnn_b], dim=3, memory_config=output_mem)
        assert ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    def test_i2s_three_tensors_height_sharded(self, device):
        """3 interleaved inputs → height-sharded output, width concat."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
        shape = (1, 1, 64, 32)
        output_shard_shape = (32, 96)

        output_mem = ttnn.create_sharded_memory_config(
            output_shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        input_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

        torch_inputs = [random_torch_tensor(ttnn.bfloat16, shape) for _ in range(3)]
        torch_out = torch.concat(torch_inputs, dim=3)

        ttnn_inputs = [
            ttnn.from_torch(
                t, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=input_mem
            )
            for t in torch_inputs
        ]

        ttnn_out = ttnn.concat(ttnn_inputs, dim=3, memory_config=output_mem)
        assert ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    def test_i2s_three_tensors_width_sharded(self, device):
        """3 interleaved inputs → width-sharded output, height concat."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})
        shape = (1, 1, 16, 64)
        output_shard_shape = (48, 16)

        output_mem = ttnn.create_sharded_memory_config(
            output_shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        input_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

        torch_inputs = [random_torch_tensor(ttnn.bfloat16, shape) for _ in range(3)]
        torch_out = torch.concat(torch_inputs, dim=2)

        ttnn_inputs = [
            ttnn.from_torch(
                t, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=input_mem
            )
            for t in torch_inputs
        ]

        ttnn_out = ttnn.concat(ttnn_inputs, dim=2, memory_config=output_mem)
        assert ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    def test_i2s_batch_dim_concat_height_sharded(self, device):
        """Interleaved inputs, batch dim concat (dim=0) → height-sharded output."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})
        shape = (1, 1, 32, 32)
        output_shard_shape = (16, 32)

        output_mem = ttnn.create_sharded_memory_config(
            output_shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        input_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)
        torch_b = random_torch_tensor(ttnn.bfloat16, shape)
        torch_out = torch.concat([torch_a, torch_b], dim=0)

        ttnn_a = ttnn.from_torch(
            torch_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=input_mem
        )
        ttnn_b = ttnn.from_torch(
            torch_b, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=input_mem
        )

        ttnn_out = ttnn.concat([ttnn_a, ttnn_b], dim=0, memory_config=output_mem)
        assert ttnn_out.is_sharded()
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)


# ===========================================================================
# 7. Single tensor and many-tensor edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge-case concat scenarios."""

    def test_single_tensor(self, device):
        """Single tensor concat is a no-op / memory_config change."""
        shape = (1, 1, 32, 64)
        torch_t = random_torch_tensor(ttnn.bfloat16, shape)
        ttnn_t = ttnn.from_torch(torch_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        result = ttnn.concat([ttnn_t], dim=0)
        result = ttnn.to_torch(result)
        assert_equal(torch_t, result)

    def test_single_tensor_memory_config_change(self, device):
        """Single tensor concat with different output memory config."""
        shape = (1, 1, 32, 64)
        torch_t = random_torch_tensor(ttnn.bfloat16, shape)
        dram_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        l1_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

        ttnn_t = ttnn.from_torch(
            torch_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=dram_mem
        )
        result = ttnn.concat([ttnn_t], dim=0, memory_config=l1_mem)
        assert result.memory_config().buffer_type == ttnn.BufferType.L1
        result = ttnn.to_torch(result)
        assert_equal(torch_t, result)

    @pytest.mark.parametrize("n_tensors", [5, 10, 20])
    def test_many_tensors(self, device, n_tensors):
        """Concat many tensors (tests batching for >50 tensors is indirectly covered by design)."""
        shape = (1, 1, 32, 32)
        torch_inputs = [random_torch_tensor(ttnn.bfloat16, shape) for _ in range(n_tensors)]
        torch_out = torch.concat(torch_inputs, dim=2)

        ttnn_inputs = [
            ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16) for t in torch_inputs
        ]
        ttnn_out = ttnn.concat(ttnn_inputs, dim=2)
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    def test_negative_dim(self, device):
        """Negative dimension indexing."""
        shapes = [(1, 1, 32, 64), (1, 1, 32, 64)]
        torch_inputs = [random_torch_tensor(ttnn.bfloat16, s) for s in shapes]
        torch_out = torch.concat(torch_inputs, dim=-1)

        ttnn_inputs = [
            ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16) for t in torch_inputs
        ]
        ttnn_out = ttnn.concat(ttnn_inputs, dim=-1)
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)

    def test_negative_dim_2(self, device):
        """Negative dimension -2."""
        shapes = [(1, 1, 32, 64), (1, 1, 64, 64)]
        torch_inputs = [random_torch_tensor(ttnn.bfloat16, s) for s in shapes]
        torch_out = torch.concat(torch_inputs, dim=-2)

        ttnn_inputs = [
            ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16) for t in torch_inputs
        ]
        ttnn_out = ttnn.concat(ttnn_inputs, dim=-2)
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_equal(torch_out, ttnn_out)


# ===========================================================================
# 8. Groups parameter (height-sharded only)
# ===========================================================================


class TestGroupsConcat:
    """Tests the groups parameter for interleaved concat on height-sharded tensors."""

    @pytest.mark.parametrize("groups", [2, 4])
    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_groups_two_tensor(self, device, groups, layout):
        """Groups > 1 with 2 height-sharded tensors on width dim."""
        core_grid = ttnn.CoreGrid(x=1, y=1)
        shape_a = (1, 1, 32, 64)
        shape_b = (1, 1, 32, 64)

        if layout == ttnn.TILE_LAYOUT and 64 // groups < 16:
            pytest.skip("Group size < 16 not supported for tiled inputs")

        torch_a = random_torch_tensor(ttnn.bfloat16, shape_a)
        torch_b = random_torch_tensor(ttnn.bfloat16, shape_b)
        expected = ttnn.concat.golden_function([torch_a, torch_b], 3, groups)

        mem_a = ttnn.create_sharded_memory_config(shape_a, core_grid, ttnn.ShardStrategy.HEIGHT)
        mem_b = ttnn.create_sharded_memory_config(shape_b, core_grid, ttnn.ShardStrategy.HEIGHT)
        output_mem = ttnn.create_sharded_memory_config(
            (1, 1, 32, shape_a[3] + shape_b[3]), core_grid, ttnn.ShardStrategy.HEIGHT
        )

        ttnn_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_a)
        ttnn_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem_b)

        result = ttnn.concat([ttnn_a, ttnn_b], dim=3, memory_config=output_mem, groups=groups)
        actual = ttnn.to_torch(result)
        assert_equal(expected, actual)


# ===========================================================================
# 9. Sub-core grids (interleaved only)
# ===========================================================================


class TestSubCoreGridsConcat:
    """Tests sub_core_grids parameter with interleaved outputs."""

    @pytest.mark.parametrize(
        "memory_config",
        [
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ],
        ids=["DRAM", "L1"],
    )
    @pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
    def test_sub_core_grids(self, device, memory_config, layout):
        sub_core_grids = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
        shapes = [(1, 1, 32, 64), (1, 1, 32, 64)]

        torch_inputs = [random_torch_tensor(ttnn.bfloat16, s) for s in shapes]
        torch_out = torch.concat(torch_inputs, dim=3)

        ttnn_inputs = [
            ttnn.from_torch(t, layout=layout, device=device, dtype=ttnn.bfloat16, memory_config=memory_config)
            for t in torch_inputs
        ]
        ttnn_out = ttnn.concat(ttnn_inputs, dim=3, memory_config=memory_config, sub_core_grids=sub_core_grids)
        ttnn_out = ttnn.to_torch(ttnn_out)
        assert_with_pcc(torch_out, ttnn_out, 0.99)


# ===========================================================================
# 10. Known unsupported cases (expected failures)
# ===========================================================================


class TestUnsupportedCases:
    """Tests that validate known unsupported configurations raise errors."""

    def test_empty_tensor_list(self, device):
        """Empty input list should raise."""
        with pytest.raises(RuntimeError):
            ttnn.concat([], dim=0)

    def test_block_sharded_non_tile_aligned_shard(self, device):
        """Non-tile-aligned shard dims are rejected by the infrastructure at tensor creation."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 48, 48)
        shard_shape = (24, 24)

        block_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)

        with pytest.raises(RuntimeError):
            ttnn.from_torch(
                torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=block_mem
            )

    def test_block_sharded_unsupported_dim(self, device):
        """Block-sharded concat only supports dim=2 and dim=3; dim=0 should raise."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shape = (1, 1, 64, 64)
        shard_shape = (32, 32)

        block_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)
        ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
        ttnn_a = ttnn.to_memory_config(ttnn_a, block_mem)

        with pytest.raises(RuntimeError):
            ttnn.concat([ttnn_a, ttnn_a], dim=0, memory_config=block_mem)

    def test_block_sharded_non_rectangular_grid(self, device):
        """Non-rectangular CoreRangeSet is rejected by the infrastructure at tensor creation."""
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(0, 1)),
            }
        )
        shape = (1, 1, 6, 64)
        shard_shape = (2, 64)

        block_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)

        with pytest.raises(RuntimeError):
            ttnn.from_torch(
                torch_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=block_mem
            )

    def test_mismatched_ranks(self, device):
        """Tensors with different ranks should raise."""
        torch_a = random_torch_tensor(ttnn.bfloat16, (1, 1, 32, 64))
        torch_b = random_torch_tensor(ttnn.bfloat16, (1, 32, 64))

        ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        with pytest.raises(RuntimeError):
            ttnn.concat([ttnn_a, ttnn_b], dim=0)

    def test_mismatched_non_concat_dims(self, device):
        """Tensors with different non-concat dimensions should raise."""
        torch_a = random_torch_tensor(ttnn.bfloat16, (1, 1, 32, 64))
        torch_b = random_torch_tensor(ttnn.bfloat16, (1, 1, 64, 128))

        ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        with pytest.raises(RuntimeError):
            ttnn.concat([ttnn_a, ttnn_b], dim=3)

    def test_dim_out_of_range(self, device):
        """Dimension out of range should raise."""
        shape = (1, 1, 32, 64)
        torch_t = random_torch_tensor(ttnn.bfloat16, shape)
        ttnn_t = ttnn.from_torch(torch_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        with pytest.raises(RuntimeError):
            ttnn.concat([ttnn_t, ttnn_t], dim=4)

    def test_optional_output_tensor_unsupported(self, device):
        """Preallocated output tensor is currently unsupported."""
        shape = (1, 1, 32, 64)
        torch_t = random_torch_tensor(ttnn.bfloat16, shape)
        ttnn_t = ttnn.from_torch(torch_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        out_t = ttnn.from_torch(
            random_torch_tensor(ttnn.bfloat16, (1, 1, 64, 64)),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.bfloat16,
        )

        with pytest.raises(RuntimeError):
            ttnn.concat([ttnn_t, ttnn_t], dim=2, output_tensor=out_t)

    def test_sharded_height_concat_on_height_sharded_not_supported(self, device):
        """Height concat (dim=-2) on height-sharded tensors is not supported.
        Only width concat (dim=-1) is valid for height-sharded."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
        shape = (1, 1, 64, 32)
        shard_shape = (32, 32)

        input_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        output_shard = (64, 32)
        output_mem = ttnn.create_sharded_memory_config(
            output_shard,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)
        ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
        ttnn_a = ttnn.to_memory_config(ttnn_a, input_mem)

        with pytest.raises(RuntimeError):
            ttnn.concat([ttnn_a, ttnn_a], dim=2, memory_config=output_mem)

    def test_width_concat_on_width_sharded_not_supported(self, device):
        """Width concat (dim=-1) on width-sharded tensors is not supported.
        Only height concat (dim=-2) is valid for width-sharded."""
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})
        shape = (1, 1, 32, 64)
        shard_shape = (32, 16)

        input_mem = ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        output_shard = (32, 32)
        output_mem = ttnn.create_sharded_memory_config(
            output_shard,
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )

        torch_a = random_torch_tensor(ttnn.bfloat16, shape)
        ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
        ttnn_a = ttnn.to_memory_config(ttnn_a, input_mem)

        with pytest.raises(RuntimeError):
            ttnn.concat([ttnn_a, ttnn_a], dim=3, memory_config=output_mem)
