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
    copy that put it there; 28-B logical rows)                                        [H/M]
  * output [1, 1, 576, 1792] float32 ROW_MAJOR, sharded like the grid (32, 1792)          [H]

The precomputed grid is made on host with ``ttnn.prepare_grid_sample_grid`` from a
[1, 576, 7, 2] float32 grid of normalized (x, y) coords. Nearest sampling is a pure gather,
so the check is exact: points are placed on pixel centres (unambiguous rounding), plus some
out-of-bounds points that must sample zero (``padding_mode="zeros"``).

Second cache (in-tree OFT at ``628a4797bf7``, 3 scales): ``reader_grid_sample_interleaved_start_id``
176844 / 144953 / 158470 + ``writer_grid_sample_interleaved`` 169986 / 415859 / 109611, compute
``compute_pool_2d`` (AVG_POOL2D defines, HiFi4, ``total_interpolations`` = 7 x {70|71, 105|106,
115|116} sticks per core over a 20-core ``split_work_to_cores``):

  * mode ``bilinear`` (pool compute kernel present), ``align_corners=False`` (the default; the model
    passes neither), ``use_precomputed_grid=True`` (reader ct[11]=1), grid dtype bf16          [H/M]
  * input = integral image [1, H, W, 256] bf16 ROW_MAJOR, L1 interleaved (512-B sticks) with
    (H, W) = (48, 160) / (24, 80) / (12, 40)                                                  [H]
  * grid [1, 1, rows, 42] bf16 ROW_MAJOR DRAM interleaved (7 points x 6 precomputed values,
    84-B sticks padded to 128 B), rows = 25344 / num_slices = 1408 / 2112 / 2304 (18 / 12 / 11
    slices); made on device by ``ttnn.reshape`` of the [1, rows, 7, 6] prepared grid
    (``rm_reshape_interleaved`` 107626 / 832696: 12-B -> 84-B sticks)                         [H]
  * output [1, 1, rows, 1792] bf16 HEIGHT_SHARDED (96 | 128, 1792) on the 5x4 grid
    (writer TensorAccessor: 20 banks, shard 96 / 128 rows), ``batch_output_channels=True``     [H]
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

K = U.GRID_POINTS


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
        tt_grid = ttnn.to_memory_config(tt_grid, U.resolve_memcfg(("hs", (U.TILE, 2 * K)), mesh))
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


# =============================================================================
# Cache #2: bilinear, precomputed grid in DRAM, bf16, sharded output, 3 OFT scales
# =============================================================================

# (H, W, rows per slice) of the 1/8, 1/16 and 1/32 OFT scales (18 / 12 / 11 slices of 25344 rows).
OFT8_2 = (U.L2_H, U.L2_W, U.BEV_ROWS // 18)  # 48 x 160, 1408 rows
OFT16_2 = (U.L2_H // 2, U.L2_W // 2, U.BEV_ROWS // 12)  # 24 x 80, 2112 rows
OFT32_2 = (U.L2_H // 4, U.L2_W // 4, U.BEV_ROWS // 11)  # 12 x 40, 2304 rows


def _bilinear_ref(inp_nhwc, grid):
    """torch reference: [1, rows, K, C] samples flattened to the batched [1, 1, rows, K*C] output."""
    rows = grid.shape[1]
    out = F.grid_sample(
        inp_nhwc.permute(0, 3, 1, 2).float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )  # [1, C, rows, K]
    return out.permute(0, 2, 3, 1).reshape(1, 1, rows, -1)


def _run_bilinear(mesh, in_shape, rows, out_mem):
    _, h, w, c = in_shape
    inp = (U.rand_nonneg(in_shape) * 100).to(torch.bfloat16)  # integral-image-like magnitudes
    # Normalized bbox-corner coords, clamped to [-1, 1] as the model does.
    grid = torch.rand(1, rows, K, 2) * 2 - 1
    expected = _bilinear_ref(inp, grid)

    # Model: prepare on host (bilinear, align_corners=False defaults), upload to DRAM, reshape on device.
    grid_host = ttnn.prepare_grid_sample_grid(
        ttnn.from_torch(grid, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32),
        list(in_shape),
        padding_mode="zeros",
        output_dtype=ttnn.bfloat16,
    )
    tt_grid = ttnn.to_device(grid_host, mesh)  # [1, rows, K, 6] bf16 RM DRAM
    tt_grid = U.op("reshape")(tt_grid, (1, 1, rows, 6 * K))

    tt_inp = U.to_tt(inp, mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.grid_sample(
        tt_inp,
        tt_grid,
        use_precomputed_grid=True,
        batch_output_channels=True,
        memory_config=U.resolve_memcfg(out_mem, mesh),
    )
    U.assert_shape_dtype(out, shape=(1, 1, rows, K * c), dtype=ttnn.bfloat16, finite=False)
    U.assert_pcc(expected, out, pcc=0.999, mesh_device=mesh)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "in_shape, rows",
    [
        pytest.param((1, OFT8_2[0], OFT8_2[1], U.OFT_C), OFT8_2[2], id="oft8-bilinear-1x48x160x256-grid1408x7-176844"),
        pytest.param(
            (1, OFT16_2[0], OFT16_2[1], U.OFT_C), OFT16_2[2], id="oft16-bilinear-1x24x80x256-grid2112x7-144953"
        ),
        pytest.param(
            (1, OFT32_2[0], OFT32_2[1], U.OFT_C), OFT32_2[2], id="oft32-bilinear-1x12x40x256-grid2304x7-158470"
        ),
    ],
)
def test_grid_sample_oft_bilinear_sharded_out(ttnn_mesh_device, reset_seeds, in_shape, rows):
    """Model-faithful (cache #2): bilinear, precomputed bf16 grid in DRAM, bf16 L1 input, output
    HEIGHT_SHARDED (``hs_rows(rows)``, 1792) on the 5x4 grid."""
    _run_bilinear(ttnn_mesh_device, in_shape, rows, ("hs", (U.hs_rows(rows), K * in_shape[-1])))


@U.with_default_mesh()
@pytest.mark.parametrize(
    "in_shape, rows",
    [pytest.param((1, OFT32_2[0], OFT32_2[1], U.OFT_C), 64, id="proxy-bilinear-1x12x40x256-grid64x7")],
)
def test_grid_sample_oft_bilinear(ttnn_mesh_device, reset_seeds, in_shape, rows):
    """Same op (bilinear, precomputed DRAM grid, bf16, 7-point batching) with an L1-interleaved output."""
    _run_bilinear(ttnn_mesh_device, in_shape, rows, "l1")
