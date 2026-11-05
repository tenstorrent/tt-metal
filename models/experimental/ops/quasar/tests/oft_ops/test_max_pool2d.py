# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.max_pool2d`` — the two pools the OFT model ran.

Shape source (kernel cache, factories at ``1088b10b014``):
  * Stem maxpool: ``compute_pool_2d/491125`` + ``reader_pool_2d`` (126035 / 650649). Padded
    input 194x642 (= 192x640 + 2x pad 1), 64 channels, 3x3 / s2 / p1, input and output
    bfloat8_b TILE, 1536 output sticks per core on 20 cores, pad value -inf.             [H]
  * Encoder NMS pool: ``compute_mpwi/357674`` + ``reader_mpwi`` (100453 / 820616), i.e.
    ``return_indices=True``. Input = NMS conv output 1x159x159x1 bf16 ROW_MAJOR
    (padded 161x161), 3x3 / s1 / p1; values bf16 ROW_MAJOR, indices uint16; 1265 output
    sticks per core.                                                                         [H]

Both: HiFi4, fp32_dest_acc off, HEIGHT_SHARDED on 20 cores.

Second cache (in-tree model, factories at main ``628a4797bf7``):
  * Stem maxpool: ``compute_pool_2d_quant`` 159781 / 321590 (a fork-only rename of mainline
    ``compute_pool_2d.cpp`` that adds ``IS_INT8`` / ``OUTPUT_INT8`` defines, both 0 here, and 3
    extra compile-time args; the MAX path is mainline's) + ``reader_pool_2d`` (300145 /
    800173 / 678885 / 117334). The pool ran through the **DRAM-slice path in 2 width slices**:
    padded input 194 x 322 / 194 x 321 (one slice per variant), 768 output sticks per core
    (96 x 160 per slice / 20 cores), 64 channels, 3x3 / s2 / p1, pad -inf, output bfloat8_b
    TILE (written by ``slice_write`` with 1088-B bfp8 tiles).                              [H]
    The input is **bf16 TILE in DRAM** (``padded_slice_reader_tiled`` 103914: 2 tiles per row,
    2048-B pages), not the bfloat8_b the model code typecasts to.                          [H]
  * Encoder NMS pool: ``compute_mpwi/163020`` + ``reader_mpwi`` (158066 / 461432) carry the same
    args as cache #1 (1265 sticks, 161 x 161 padded) -> covered by ``nms-pool-...-357674``.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

# Same pre-existing Quasar limitation the VAD-variant suite tracks (tiled pool output).
_QUASAR_TILED_POOL_XFAIL = (
    "Quasar max_pool2d tiled-output path is broken (pre-existing; see test_pool2d.py tiled_out_*)"
)


@U.with_mesh_l1small(U.OFT_L1_SMALL_SIZE)
@pytest.mark.parametrize(
    "c, h, hw",
    [
        pytest.param(64, U.STEM_H, U.STEM_W, id="stem-maxpool-192x640x64-491125"),
        pytest.param(64, 24, 40, id="proxy-stem-24x40x64"),
    ],
)
def test_max_pool2d_stem(ttnn_mesh_device, reset_seeds, c, h, hw):
    """bfloat8_b TILE in -> bfloat8_b TILE out, 3x3 / s2 / p1 (stem maxpool)."""
    U.xfail_on_quasar(_QUASAR_TILED_POOL_XFAIL)
    mesh = ttnn_mesh_device
    w = hw
    x_nhwc = U.torch_rand((1, h, w, c))
    x = U.to_tt(x_nhwc.reshape(1, 1, h * w, c), mesh, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    x_seen = U.seen(x, x_nhwc, mesh).permute(0, 3, 1, 2)

    out = U.op("max_pool2d")(
        input_tensor=x,
        batch_size=1,
        input_h=h,
        input_w=w,
        channels=c,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1],
        dilation=[1, 1],
        ceil_mode=False,
        applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        reallocate_halo_output=True,
        output_layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )

    ref = torch.nn.functional.max_pool2d(x_seen.float(), 3, 2, 1)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c)
    # Max of bfp8 values re-packed to bfp8: shared exponents can shift, so PCC rather than exact.
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)


@U.with_mesh_l1small(U.OFT_L1_SMALL_SIZE)
@pytest.mark.parametrize(
    "h, hw",
    [
        pytest.param(U.BEV, U.BEV, id="nms-pool-159x159x1-357674"),
        pytest.param(32, 32, id="proxy-nms-pool-32x32x1"),
    ],
)
def test_max_pool2d_with_indices(ttnn_mesh_device, reset_seeds, h, hw):
    """Encoder NMS: 3x3 / s1 / p1 max pool on the smoothed heatmap, returning indices."""
    mesh = ttnn_mesh_device
    w, c = hw, 1
    x_nhwc = U.torch_rand((1, h, w, c))
    x = U.to_tt(x_nhwc, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)

    out, idx = U.op("max_pool2d")(
        input_tensor=x,
        batch_size=1,
        input_h=h,
        input_w=w,
        channels=c,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        ceil_mode=False,
        applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_input=False,
        reallocate_halo_output=True,
        return_indices=True,
    )

    x_nchw = x_nhwc.permute(0, 3, 1, 2).float()
    ref = torch.nn.functional.max_pool2d(x_nchw, 3, 1, 1)
    got = U.from_tt(out, mesh).reshape(-1)[: h * w]
    assert torch.equal(got, ref.reshape(-1)), "max_pool2d values differ from torch (max of bf16 is exact)"

    # Ties are possible in bf16, so check that each index points at a maximal input element
    # inside the 3x3 window rather than comparing indices to torch's tie-break.
    got_idx = ttnn.to_torch(idx).reshape(-1)[: h * w].to(torch.int64)
    flat = x_nchw.reshape(-1)
    assert torch.equal(flat[got_idx], ref.reshape(-1)), "returned indices do not point at the window maxima"
    rows, cols = got_idx // w, got_idx % w
    out_r = torch.arange(h).repeat_interleave(w)
    out_c = torch.arange(w).repeat(h)
    assert ((rows - out_r).abs() <= 1).all() and ((cols - out_c).abs() <= 1).all(), "index outside the 3x3 window"


@U.with_mesh_l1small(U.OFT_L1_SMALL_SIZE)
@pytest.mark.parametrize(
    "c, h, hw, num_slices",
    [
        pytest.param(64, U.STEM_H, U.STEM_W, 2, id="stem-maxpool-dram-wslice2-159781"),
        # TILE output width-slices are whole tiles: out width 64 (in 128, s2) -> 2 x 32.
        pytest.param(64, 24, 128, 2, id="proxy-stem-dram-wslice2-24x128x64"),
    ],
)
def test_max_pool2d_stem_dram_sliced(ttnn_mesh_device, reset_seeds, c, h, hw, num_slices):
    """bf16 TILE DRAM in -> bfloat8_b TILE out, 3x3 / s2 / p1, DRAM path in 2 width slices.

    As ``TtMaxPool2d`` calls it (no ``applied_shard_scheme``, ``deallocate_input=True``); the
    slice count / direction the auto slicer picked in the cached run is pinned explicitly. [H]
    """
    U.xfail_on_quasar(_QUASAR_TILED_POOL_XFAIL)
    mesh = ttnn_mesh_device
    w = hw
    x_nhwc = U.torch_rand((1, h, w, c))
    x = U.to_tt(x_nhwc.reshape(1, 1, h * w, c), mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x_seen = U.seen(x, x_nhwc, mesh).permute(0, 3, 1, 2)

    out = U.op("max_pool2d")(
        input_tensor=x,
        batch_size=1,
        input_h=h,
        input_w=w,
        channels=c,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1],
        dilation=[1, 1],
        ceil_mode=False,
        dram_slice_config=ttnn.Op2DSliceConfig(num_slices=num_slices, slice_type=ttnn.Op2DDRAMSliceWidth),
        deallocate_input=True,
        reallocate_halo_output=True,
        output_layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )

    ref = torch.nn.functional.max_pool2d(x_seen.float(), 3, 2, 1)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c)
    # bf16 maxima re-packed to bfp8 (shared exponents): PCC rather than exact.
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)
