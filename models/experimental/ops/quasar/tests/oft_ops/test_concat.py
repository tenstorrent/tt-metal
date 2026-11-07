# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.concat`` — the ResNet stem re-joins its two GroupNorm halves, one height slice
at a time (``TTResNetFeatures.forward``):

    slice_0 = interleaved_to_sharded_partial(gn_half_0, (5, 4), [3072, 32], 2, i, HEIGHT_SHARDED, ROW_MAJOR)
    slice_1 = interleaved_to_sharded_partial(gn_half_1, ...)
    slice_1 = ttnn.concat([slice_0, slice_1], dim=3, memory_config=HS(3072, 64))

Shape source: second kernel cache (in-tree OFT, factory ``concat_s2s_tiled`` at ``628a4797bf7``):
``height_sharded_width_concat_two_tensors/832626`` + ``reader/writer_height_sharded_width_concat_
two_tensors_tiled`` (141352 / 450657), ct = [cb 0, 1, 3, 4, 5, 6, 2, in0 96 x 1 tiles, in1 96 x 1
tiles, 2048 B, groups 1, batch 8]: two bf16 TILE HEIGHT_SHARDED (3072, 32) shards per core on the
5x4 grid -> (3072, 64); a slice is 192*640 / 2 = 61440 rows.                                  [H]

Value-preserving: checked bit-exact. The mainline s2s tiled concat factory is Metal 2.0
(ProgramSpec), so it is called directly.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U


def _hs(mesh, shard, grid):
    spec = ttnn.ShardSpec(U.model_grid(mesh, *grid), list(shard), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def _run(mesh, shape, shard_h, grid):
    rows, half = shape[2], shape[3]
    a = U.torch_rand(shape)
    b = U.torch_rand(shape)
    in_mem = _hs(mesh, (shard_h, half), grid)
    out_mem = _hs(mesh, (shard_h, 2 * half), grid)
    tt_a = U.op("to_memory_config")(U.to_tt(a, mesh), in_mem)
    tt_b = U.op("to_memory_config")(U.to_tt(b, mesh), in_mem)
    out = ttnn.concat([tt_a, tt_b], dim=3, memory_config=out_mem)
    assert out.is_sharded(), "concat output lost its sharded memory layout"
    U.assert_lossless(torch.cat([a, b], dim=3), out, mesh_device=mesh)
    assert tuple(out.shape) == (1, 1, rows, 2 * half)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, shard_h, grid",
    [
        pytest.param(
            (1, 1, U.STEM_H * U.STEM_W // 2, 32),
            3072,
            (U.MODEL_GRID_X, U.MODEL_GRID_Y),
            id="stem-gn-halves-hs-3072x32x2-832626",
        )
    ],
)
def test_concat_height_sharded_width(ttnn_mesh_device, reset_seeds, shape, shard_h, grid):
    _run(ttnn_mesh_device, shape, shard_h, grid)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, shard_h, grid",
    [pytest.param((1, 1, 128, 32), 64, (2, 1), id="proxy-hs-64x32x2-2cores")],
)
def test_concat_height_sharded_width_small(ttnn_mesh_device, reset_seeds, shape, shard_h, grid):
    _run(ttnn_mesh_device, shape, shard_h, grid)
