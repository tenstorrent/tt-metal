# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.slice`` / tensor slicing, ``ttnn.experimental.padded_slice`` and
``ttnn.experimental.slice_write`` — as the OFT model ran them.

Shape source (kernel cache, factories at ``1088b10b014``):
  * ``ortho_feats[:, :, :25281, :]`` (drop the 63 pad rows after the 44 OFT slices):
    bf16 TILE DRAM [1, 1, 25344, 128] (``reader_unary_unpad_dims_interleaved_start_id/723853``) [M]
  * head output split ``outputs[:, :, s:s+n]`` for n in [1, 3, 3, 2]: bf16 ROW_MAJOR L1
    [1, 1, 9, 159, 159] (``slice_reader_unary_unpad_dims_rm_interleaved_start_id/148027``)     [H/M]
  * ``ang_offsets[:, 0]`` / ``[:, 1]``: bf16 TILE L1 [1, 2, 159, 159] (119790)                 [M]
  * ``grid[1:, 1:]`` / ``grid[:-1, :-1]`` (positions decode): ROW_MAJOR DRAM -> DRAM
    (422600 / 426284)                                                                         [L]
  * conv DRAM-slicing internals for the head conv (2 height slices): ``padded_slice`` of the
    bf16 TILE DRAM [1, 159, 159, 128] input into ROW_MAJOR HEIGHT_SHARDED shards — rows
    [0, 81) at 644 sticks/core and rows [79, 159) at 636 (halo rows included)
    (``padded_slice_reader_tiled_interleaved_start_id/115525``)                                [H]
  * ``slice_write`` of conv output slices into the full DRAM output: head (bf16 ROW_MAJOR
    sharded, C=9, 18-B pages; 620394) and stem (bfloat8_b TILE sharded (1536, 64) ->
    [1, 192, 640, 64] DRAM, 4 slices of 48 rows; 112021)                                     [H/M]

Second cache (in-tree model, factories at ``628a4797bf7``). All three ops below are launched
by conv2d DRAM slicing (conv-internal; the conv cases live in ``test_conv2d.py``) but exercise
regimes the cache-1 cases do not:
  * topdown convs (``HSLICE_4``, ``HS_ABH_64_RM``) take a ROW_MAJOR input: ``padded_slice`` of a
    bf16 ROW_MAJOR DRAM [1, 159, 159, 256] tensor into HEIGHT_SHARDED shards
    (``padded_slice_reader_rm_interleaved_start_id/401385``: aligned reads, 64-B DRAM
    alignment, 2 trids, 512-B pages). Slice rows / shard heights are a 4-slice estimate     [H op / L rows]
  * the same convs write ROW_MAJOR output slices: ``slice_write`` bf16 ROW_MAJOR sharded,
    C=256 (512-B pages; ``slice_write_writer_interleaved/240705``)                            [H op / M shape]
  * a bf16 TILE ``slice_write`` (2048-B pages, ``312461``): no bf16 TILE-output sliced conv was
    identified in the in-tree configs; the stem geometry of the bfloat8_b case is reused     [L]

All value-preserving: checked bit-exact.
"""

import math

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

BEV, C = U.BEV, U.TD_C
LAT_C = 256  # in-tree model topdown channels (second cache)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, begins, ends, layout_rm, mem",
    [
        pytest.param((1, 1, U.BEV_ROWS, C), (0, 0, 0, 0), (1, 1, U.BEV_NHW, C), False, "dram", id="ortho-tail-723853"),
        pytest.param((1, 1, 640, C), (0, 0, 0, 0), (1, 1, 577, C), False, "dram", id="proxy-ortho-tail"),
        pytest.param((1, 1, 9, BEV, BEV), (0, 0, 0, 0, 0), (1, 1, 1, BEV, BEV), True, "l1", id="head-score-0:1-148027"),
        pytest.param((1, 1, 9, BEV, BEV), (0, 0, 1, 0, 0), (1, 1, 4, BEV, BEV), True, "l1", id="head-pos-1:4-148027"),
        pytest.param((1, 1, 9, BEV, BEV), (0, 0, 4, 0, 0), (1, 1, 7, BEV, BEV), True, "l1", id="head-dim-4:7-148027"),
        pytest.param((1, 1, 9, BEV, BEV), (0, 0, 7, 0, 0), (1, 1, 9, BEV, BEV), True, "l1", id="head-ang-7:9-148027"),
        pytest.param((1, 2, BEV, BEV), (0, 0, 0, 0), (1, 1, BEV, BEV), False, "l1", id="ang-cos-119790"),
        pytest.param((1, 2, BEV, BEV), (0, 1, 0, 0), (1, 2, BEV, BEV), False, "l1", id="ang-sin-119790"),
        pytest.param((BEV + 1, BEV + 1, 3), (1, 1, 0), (BEV + 1, BEV + 1, 3), True, "dram", id="grid-1:,1:-422600"),
        pytest.param((BEV + 1, BEV + 1, 3), (0, 0, 0), (BEV, BEV, 3), True, "dram", id="grid-:-1,:-1-422600"),
    ],
)
def test_slice(ttnn_mesh_device, reset_seeds, shape, begins, ends, layout_rm, mem):
    mesh = ttnn_mesh_device
    x = U.torch_rand(shape)
    layout = ttnn.ROW_MAJOR_LAYOUT if layout_rm else ttnn.TILE_LAYOUT
    tt_x = U.to_tt_mem(x, mesh, mem, layout=layout)
    out = U.op("slice")(tt_x, list(begins), list(ends), memory_config=U.resolve_memcfg(mem, mesh))
    ref = x[tuple(slice(b, e) for b, e in zip(begins, ends))]
    U.assert_shape_dtype(out, shape=ref.shape, finite=False)
    U.assert_lossless(ref, out, mesh_device=mesh)


def _two_core_hs(rows, width):
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [math.ceil(rows / 2), width], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _run_padded_slice(mesh, shape, r0, r1, out_mem, layout=ttnn.TILE_LAYOUT):
    x = U.torch_rand(shape)
    tt_x = U.to_tt(x, mesh, layout=layout)  # DRAM, as the conv receives it
    _, _, w, c = shape
    out = U.op("padded_slice", experimental=True)(
        tt_x, [0, r0, 0, 0], [1, r1, w, c], [1, 1, 1, 1], memory_config=out_mem
    )
    U.assert_lossless(x[:, r0:r1].reshape(1, 1, -1, c), out, mesh_device=mesh)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, r0, r1, shard_h",
    [
        pytest.param((1, BEV, BEV, C), 0, 81, 644, id="head-in-rows0:81-115525"),
        pytest.param((1, BEV, BEV, C), 79, BEV, 636, id="head-in-rows79:159-115525"),
    ],
)
def test_padded_slice(ttnn_mesh_device, reset_seeds, shape, r0, r1, shard_h):
    mesh = ttnn_mesh_device
    _run_padded_slice(mesh, shape, r0, r1, U.resolve_memcfg(("hs", (shard_h, shape[-1])), mesh))


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, r0, r1, shard_h",
    [
        # 4 height slices of 40 output rows; 3x3 conv: +1 halo row each side (estimate).
        pytest.param((1, BEV, BEV, LAT_C), 0, 41, 326, id="topdown-in-rm-rows0:41-401385"),
        pytest.param((1, BEV, BEV, LAT_C), 39, 81, 334, id="topdown-in-rm-rows39:81-401385"),
    ],
)
def test_padded_slice_rm(ttnn_mesh_device, reset_seeds, shape, r0, r1, shard_h):
    """``padded_slice`` of a ROW_MAJOR DRAM input (cache 1 only had TILE inputs)."""
    mesh = ttnn_mesh_device
    out_mem = U.resolve_memcfg(("hs", (shard_h, shape[-1])), mesh)
    _run_padded_slice(mesh, shape, r0, r1, out_mem, layout=ttnn.ROW_MAJOR_LAYOUT)


@U.with_default_mesh()
@pytest.mark.parametrize("shape, r0, r1", [pytest.param((1, 8, 32, 64), 3, 8, id="proxy-rm-rows3:8-2cores")])
def test_padded_slice_rm_small(ttnn_mesh_device, reset_seeds, shape, r0, r1):
    _, _, w, c = shape
    _run_padded_slice(ttnn_mesh_device, shape, r0, r1, _two_core_hs((r1 - r0) * w, c), layout=ttnn.ROW_MAJOR_LAYOUT)


@U.with_default_mesh()
@pytest.mark.parametrize("shape, r0, r1", [pytest.param((1, 8, 32, 64), 3, 8, id="proxy-rows3:8-2cores")])
def test_padded_slice_small(ttnn_mesh_device, reset_seeds, shape, r0, r1):
    _, _, w, c = shape
    _run_padded_slice(ttnn_mesh_device, shape, r0, r1, _two_core_hs((r1 - r0) * w, c))


def _run_slice_write(mesh, out_shape, num_slices, dtype, layout, in_mem_fn):
    """Write ``num_slices`` height slices into a zeroed DRAM output, as conv DRAM slicing does."""
    n, h, w, c = out_shape
    x = U.torch_rand(out_shape)
    tt_out = U.to_tt(torch.zeros(out_shape, dtype=torch.bfloat16), mesh, dtype=dtype, layout=layout)
    expected = torch.zeros(out_shape)
    step = math.ceil(h / num_slices)
    for r0 in range(0, h, step):
        r1 = min(r0 + step, h)
        part = x[:, r0:r1]
        tt_part = U.to_tt(part, mesh, dtype=dtype, layout=layout)
        tt_part = ttnn.to_memory_config(tt_part, in_mem_fn(r1 - r0))
        expected[:, r0:r1] = U.seen(tt_part, part, mesh)  # bfloat8_b: what the device holds
        U.op("slice_write", experimental=True)(tt_part, tt_out, [0, r0, 0, 0], [n, r1, w, c], [1, 1, 1, 1])
    U.assert_lossless(expected, tt_out, mesh_device=mesh)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "out_shape, num_slices, dt, layout_rm",
    [
        pytest.param((1, BEV, BEV, U.HEAD_OUT), 2, "bf16", True, id="head-out-rm-c9-2slices-620394"),
        pytest.param((1, U.STEM_H, U.STEM_W, 64), 4, "bfp8", False, id="stem-out-bfp8-tile-4slices-112021"),
        # second cache
        pytest.param((1, BEV, BEV, LAT_C), 4, "bf16", True, id="topdown-out-rm-c256-4slices-240705"),
        pytest.param((1, U.STEM_H, U.STEM_W, 64), 4, "bf16", False, id="bf16-tile-4slices-312461"),
    ],
)
def test_slice_write_sharded(ttnn_mesh_device, reset_seeds, out_shape, num_slices, dt, layout_rm):
    mesh = ttnn_mesh_device
    _, _, w, c = out_shape
    layout = ttnn.ROW_MAJOR_LAYOUT if layout_rm else ttnn.TILE_LAYOUT

    # Each slice's rows * W sticks spread over the 20 cores (stem: 48 * 640 / 20 = 1536).
    def in_mem(rows):
        return U.resolve_memcfg(("hs", (math.ceil(rows * w / U.MODEL_NUM_CORES), c)), mesh)

    _run_slice_write(mesh, out_shape, num_slices, U.DTYPES[dt], layout, in_mem)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "out_shape",
    [
        pytest.param((1, 8, 32, U.HEAD_OUT), id="proxy-head-out-rm-c9-l1"),
        pytest.param((1, 8, 32, LAT_C), id="proxy-topdown-out-rm-c256-l1"),
    ],
)
def test_slice_write(ttnn_mesh_device, reset_seeds, out_shape):
    """Same op with an L1-interleaved ROW_MAJOR source (emulator-sized)."""
    _run_slice_write(
        ttnn_mesh_device, out_shape, 2, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, lambda rows: ttnn.L1_MEMORY_CONFIG
    )
