# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Nightly tests for ttnn.concat with INT32/UINT32 data types, covering all compute
kernels that issue the MOVD2B instruction via transpose_wh_tile.

Code paths exercised:
  Path A – permute_rm_program_factory + transpose_xw_rm_single_tile_size.cpp
    Reached via interleaved TILE_LAYOUT with small (unaligned) last dim:
      untilize_rm_retilize_concat fires (logical_shape[-1] != padded_shape[-1])
      → tensors become interleaved ROW_MAJOR
      → non_aligned_last_dim_concat fires (padded_shape[-1]*4 < 32-byte alignment)
      → ttnn::transpose(rm_tensor, -2, -1)
      → prim::permute({0,1,3,2}) on RM tensor
      → permute_rm_program_factory + transpose_xw_rm_single_tile_size.cpp
    Also reached via interleaved ROW_MAJOR with unaligned last dim directly.

  Path B – concat_s2s_tiled_program_factory (sharded TILE_LAYOUT concat)
    Reached via sharded TILE_LAYOUT tensors concatenated along last dim.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

torch.manual_seed(0)


def _rand_torch(dtype, shape):
    if dtype == ttnn.int32:
        return torch.randint(-(2**30), 2**30, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


# ──────────────────────────────────────────────────────────────────────────────
# Path A – permute_rm_program_factory via interleaved TILE_LAYOUT
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize(
    "shapes,dim",
    [
        # last dim < 32 tiles → untilize + non-aligned last-dim transpose
        ([[1, 1, 4, 4], [1, 1, 4, 4]], -1),
        ([[1, 1, 4, 4], [1, 1, 4, 8]], -1),  # different last dims
        ([[1, 1, 8, 4], [1, 1, 8, 4]], -1),
        ([[1, 1, 16, 4], [1, 1, 16, 4]], -1),
        ([[2, 3, 4, 4], [2, 3, 4, 4]], -1),
        ([[1, 1, 4, 4], [1, 1, 4, 4]], 2),  # dim=H (also unaligned)
        # non-last-dim concat with small tiles (goes via untilize path, no transpose)
        ([[1, 1, 4, 4], [1, 1, 4, 4]], -2),
    ],
)
def test_tiled_interleaved_concat_int(device, dtype, shapes, dim):
    """
    Concat of interleaved TILE_LAYOUT INT32/UINT32 tensors with small last dim.
    For dim=-1 with width<32: triggers untilize_rm_retilize_concat followed by
    non_aligned_last_dim_concat, exercising permute_rm_program_factory +
    transpose_xw_rm_single_tile_size.cpp (MOVD2B path).
    """
    torch_tensors = [_rand_torch(dtype, s) for s in shapes]
    torch_out = torch.cat(torch_tensors, dim=dim)

    ttnn_tensors = [ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype) for t in torch_tensors]
    out = ttnn.concat(ttnn_tensors, dim=dim)
    assert_equal(torch_out, ttnn.to_torch(out))


# ──────────────────────────────────────────────────────────────────────────────
# Path A – permute_rm_program_factory via interleaved ROW_MAJOR directly
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize(
    "shapes,dim",
    [
        # padded_shape[-1] * 4 bytes not aligned to 32 bytes  →  non_aligned_last_dim_concat
        ([[1, 1, 32, 4], [1, 1, 32, 4]], -1),  # 4*4=16 bytes, unaligned
        ([[1, 1, 32, 5], [1, 1, 32, 5]], -1),  # 5*4=20 bytes, unaligned
        ([[1, 1, 32, 7], [1, 1, 32, 7]], -1),  # 7*4=28 bytes, unaligned
        ([[1, 1, 32, 4], [1, 1, 32, 8]], -1),  # different last dims, both unaligned
        ([[2, 3, 32, 4], [2, 3, 32, 4]], -1),
    ],
)
def test_rm_interleaved_concat_unaligned_int(device, dtype, shapes, dim):
    """
    Concat of interleaved ROW_MAJOR INT32/UINT32 tensors where the last dim is
    not 32-byte-aligned.  Exercises non_aligned_last_dim_concat →
    permute_rm_program_factory + transpose_xw_rm_single_tile_size.cpp (MOVD2B path).
    """
    torch_tensors = [_rand_torch(dtype, s) for s in shapes]
    torch_out = torch.cat(torch_tensors, dim=dim)

    ttnn_tensors = [ttnn.from_torch(t, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=dtype) for t in torch_tensors]
    out = ttnn.concat(ttnn_tensors, dim=dim)
    assert_equal(torch_out, ttnn.to_torch(out))


# ──────────────────────────────────────────────────────────────────────────────
# Path B – concat_s2s_tiled_program_factory via sharded TILE_LAYOUT
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize(
    "shapes,shard_shape,core_grid,strategy,dim",
    [
        # HEIGHT-sharded tiled concat along width (dim=3)
        (
            [[[1, 1, 192, 32], (96, 32)], [[1, 1, 192, 32], (96, 32)]],
            (96, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
            ttnn.ShardStrategy.HEIGHT,
            3,
        ),
        (
            [[[1, 1, 256, 64], (64, 64)], [[1, 1, 256, 128], (64, 128)]],
            (64, 192),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 0)),
                }
            ),
            ttnn.ShardStrategy.HEIGHT,
            3,
        ),
        # WIDTH-sharded tiled concat along height (dim=2)
        (
            [[[1, 1, 32, 512], (32, 64)], [[1, 1, 64, 512], (64, 64)]],
            (96, 64),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 1)),
                }
            ),
            ttnn.ShardStrategy.WIDTH,
            2,
        ),
    ],
)
def test_sharded_tiled_concat_int(device, dtype, shapes, shard_shape, core_grid, strategy, dim):
    """
    Concat of sharded TILE_LAYOUT INT32/UINT32 tensors.
    Exercises concat_s2s_tiled_program_factory.
    """
    torch_tensors = []
    ttnn_tensors = []
    for shape, shd in shapes:
        t = _rand_torch(dtype, shape)
        mem_cfg = ttnn.create_sharded_memory_config(
            shd, core_grid=core_grid, strategy=strategy, use_height_and_width_as_shard_shape=True
        )
        tt = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
        tt = ttnn.to_memory_config(tt, mem_cfg)
        torch_tensors.append(t)
        ttnn_tensors.append(tt)

    out_mem_cfg = ttnn.create_sharded_memory_config(
        shard_shape, core_grid=core_grid, strategy=strategy, use_height_and_width_as_shard_shape=True
    )
    out = ttnn.concat(ttnn_tensors, dim=dim, memory_config=out_mem_cfg)
    assert_equal(torch.cat(torch_tensors, dim=dim), ttnn.to_torch(out))
