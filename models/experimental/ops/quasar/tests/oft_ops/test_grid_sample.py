# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.grid_sample`` — the OFT voxel lookup into the integral image (one call
per bbox corner per BEV slice: 4 x 44 calls).

Shape source (kernel cache): ``writer_grid_sample_nearest_sharded`` 134738 / 883676 (the two
split-reader halves of one op; they differ only in ``reader_id``). Compile-time args:

  * mode ``nearest`` (no bilinear reader / compute kernel exists), ``align_corners=True``,
    ``use_precomputed_grid=True``, grid dtype bf16                                     [H]
  * input = integral image [1, 48, 160, 256] float32 ROW_MAJOR (1024-B sticks), L1
    interleaved                                                                         [H]
  * grid batching factor 7 (``batch_output_channels``: the 7 height points of each BEV
    cell in one stick), grid_hw 576 -> grid [1, 1, 576, 14] after the reshape            [H]
  * grid HEIGHT_SHARDED, 32 sticks per core on the 5x4 grid (and the interleaved->sharded
    copy ``128759`` that put it there; 28-B logical rows)                             [H/M]
  * the grid shard width is padded 14 -> 16 bf16 (32 B, L1-aligned), as upstream's
    ``prepare_sharded_grid_memory_config`` does: the sharded factory strides the grid shard by
    ``padded_shape[-1] * elem_size`` while the buffer stores rows at the L1-aligned page size,
    so an unaligned shard width (28 B) reads every row after the first at the wrong offset.
  * output [1, 1, 576, 1792] float32 ROW_MAJOR, sharded like the grid (32, 1792)          [H]

The precomputed grid is made on host with ``ttnn.prepare_grid_sample_grid`` from a
[1, 576, 7, 2] float32 grid of normalized (x, y) coords. Nearest sampling is a pure gather,
so the check is exact: points are placed on pixel centres (unambiguous rounding), plus some
out-of-bounds points that must sample zero (``padding_mode="zeros"``).
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

K = U.GRID_POINTS
L1_ALIGNMENT_BYTES = 16


def _make_grid(rows, h, w):
    """[1, rows, K, 2] normalized grid on exact pixel positions (align_corners=True), ~5% OOB."""
    ix = torch.randint(0, w, (1, rows, K))
    iy = torch.randint(0, h, (1, rows, K))
    gx = ix.float() / (w - 1) * 2 - 1
    gy = iy.float() / (h - 1) * 2 - 1
    oob = torch.rand(1, rows, K) < 0.05
    gx = torch.where(oob, torch.full_like(gx, 3.0), gx)
    return torch.stack((gx, gy), dim=-1), ix, iy, oob


def _expected(inp_nhwc, ix, iy, oob):
    c = inp_nhwc.shape[-1]
    gathered = inp_nhwc[0, iy.clamp(max=inp_nhwc.shape[1] - 1), ix.clamp(max=inp_nhwc.shape[2] - 1)]  # [1, rows, K, C]
    gathered[oob] = 0
    rows = ix.shape[1]
    return gathered.reshape(1, 1, rows, K * c)


def _run(mesh, in_shape, rows, sharded):
    _, h, w, c = in_shape
    inp = U.rand_nonneg(in_shape) * 100  # integral-image-like magnitudes
    grid, ix, iy, oob = _make_grid(rows, h, w)
    expected = _expected(inp, ix, iy, oob)

    grid_host = ttnn.prepare_grid_sample_grid(
        ttnn.from_torch(grid, dtype=ttnn.float32),
        list(in_shape),
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
        output_dtype=ttnn.bfloat16,
    )
    grid_host = ttnn.reshape(grid_host, (1, 1, rows, 2 * K))
    tt_grid = ttnn.to_device(grid_host, mesh)
    kwargs = {}
    if sharded:
        # L1-align the shard row (14 -> 16 bf16 = 32 B); see module docstring.
        grid_shard_w = -(-2 * K * 2 // L1_ALIGNMENT_BYTES) * L1_ALIGNMENT_BYTES // 2
        tt_grid = ttnn.to_memory_config(tt_grid, U.resolve_memcfg(("hs", (U.TILE, grid_shard_w)), mesh))
        kwargs["memory_config"] = U.resolve_memcfg(("hs", (U.TILE, K * c)), mesh)

    tt_inp = U.to_tt(inp, mesh, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.grid_sample(
        tt_inp,
        tt_grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
        use_precomputed_grid=True,
        batch_output_channels=True,
        **kwargs,
    )
    U.assert_shape_dtype(out, shape=(1, 1, rows, K * c), dtype=ttnn.float32, finite=False)
    U.assert_lossless(expected, out, mesh_device=mesh)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "in_shape",
    [pytest.param((1, U.L2_H, U.L2_W, U.OFT_C), id="oft8-intimg-1x48x160x256-grid576x7-134738")],
)
def test_grid_sample_oft_sharded(ttnn_mesh_device, reset_seeds, in_shape):
    """Model-faithful: grid and output HEIGHT_SHARDED 32 rows per core on the 5x4 grid."""
    _run(ttnn_mesh_device, in_shape, U.SLICE_ROWS, sharded=True)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "in_shape, rows",
    [
        pytest.param((1, U.L2_H, U.L2_W, U.OFT_C), U.SLICE_ROWS, id="oft8-1x48x160x256-grid576x7-interleaved"),
        pytest.param((1, 12, 40, U.OFT_C), 64, id="proxy-1x12x40x256-grid64x7"),
    ],
)
def test_grid_sample_oft(ttnn_mesh_device, reset_seeds, in_shape, rows):
    """Same op (nearest, precomputed, 7-point channel batching, fp32 input) with an interleaved grid."""
    _run(ttnn_mesh_device, in_shape, rows, sharded=False)
