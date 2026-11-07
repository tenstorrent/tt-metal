# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: memory-config moves — ``to_memory_config`` (interleaved <-> sharded, DRAM <-> L1),
``sharded_to_interleaved``, ``sharded_to_interleaved_partial`` and ``ttnn.move`` — as the
OFT model ran them.

Shape source (kernel cache, factories at ``1088b10b014``). Sharded readers/writers carry
``num_readers=20`` and the shard shape; interleaved accessors give DRAM vs L1 and page size:
  * integral image fp32 ROW_MAJOR DRAM -> L1 interleaved, 1024-B sticks
    (``reader/writer_unary_stick_layout_interleaved_start_id`` 177182 / 462779)          [H]
  * area slice fp32 TILE DRAM -> HEIGHT_SHARDED (32, 1792) (``reader_unary_sharded_blocks_
    interleaved_start_id/502856``, 4096-B tiles)                                        [H/M]
  * conv inputs bf16 / bfloat8_b TILE DRAM -> HEIGHT_SHARDED on 20 cores (154358 / 291869);
    shard (1536, 64) = layer1 input [M]
  * lat8 conv output fp32 TILE HEIGHT_SHARDED (384, 256) -> DRAM (159108 / 261104)        [L]
  * NMS pool indices uint16 ROW_MAJOR HEIGHT_SHARDED (1265, 16) -> L1 interleaved
    (writer stick 32 B, 178576)                                                           [H]
  * ``ttnn.move`` of sharded tensors: fp32 vox slice (32, 1792) (``reader_unary_local_l1_copy_
    backwards/119210``) and bf16 (``312368``, topdown activations (1280, 128) [L])         [H]
  * per-slice ``sharded_to_interleaved_partial(vox, ortho, 44, i)``: bf16 TILE (32, 128)
    shards into a [1, 1, 25344, 128] TILE DRAM tensor (532985 / 729250)                  [H/M]

All value-preserving: checked bit-exact.

Second cache (in-tree OFT at ``628a4797bf7``, all bf16; ids carry the cache-#2 hashes):
  * ``to_memory_config`` DRAM -> L1 interleaved of the bf16 ROW_MAJOR integral image (``tt_oft.py``):
    ``redistribute_pages_row_major_reader/writer`` 936427 / 424582 (copy factory, 512-B rows)       [H]
  * DRAM ROW_MAJOR -> HEIGHT_SHARDED (lateral GroupNorm input, 512-B rows,
    ``reader_unary_stick_layout_sharded_blocks_interleaved_start_id/190297``)            [H page / M shape]
  * L1 interleaved TILE -> 20-core shard (``reader_unary_sharded_blocks_interleaved_start_id/226960``,
    source not identified; layer2-sized representative)                                           [L]
  * OFT area slices DRAM -> HEIGHT_SHARDED (96 | 128, 1792) (``create_sharded_tensor``): 1408 / 2112 /
    2304 rows over 15 / 17 / 18 used cores of the 20-core shard grid (readers 115190 / 136900 / 137435)
  * ``sharded_to_interleaved`` -> DRAM: feats8 (384, 128) / feats32 (32, 512) on 15 cores TILE
    (``writer_unary_sharded_blocks_interleaved_start_id/164400``), lat8 HS (384, 256) / lat32 BS
    (96, 64) ROW_MAJOR (``writer_unary_stick_layout_sharded_blocks_interleaved_start_id/165319``) [H mem / M shape]
  * ``interleaved_to_sharded_partial`` (new op) + ``sharded_to_interleaved_partial`` of the stem:
    2 slices of 61440 rows, HS (3072, 32) out of each GroupNorm half and (3072, 64) back into conv1
    (DRAM TILE) after the concat + relu (readers 465767 / writer 164400)                     [H/M]
  * per-scale OFT ``sharded_to_interleaved_partial(linear_out, ortho_feats, 18 | 12 | 11, i)``: bf16
    (96 | 128, 256) shards into [1, 1, 25344, 256] DRAM (writer 164400)                               [M]
  * ``ttnn.move`` of the bf16 vox slice HS (96, 1792) (``reader_unary_local_l1_copy_backwards/159159``)
    and of a DRAM-interleaved TILE tensor (``reader_unary_start_id/128139`` [cb 0, DRAM, 2048 B]:
    the topdown ``ttnn.move(out)`` after the DRAM-sliced conv, [1, 1, 25344, 256])     [H kernel / M shape]
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

_DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32, "uint16": ttnn.uint16}


def _rand(shape, dt):
    if dt == "uint16":
        return torch.randint(0, U.BEV_NHW, shape, dtype=torch.int32)
    return U.torch_rand(shape, dtype=torch.float32 if dt == "fp32" else torch.bfloat16)


# fmt: off
# (shape, dtype, layout_rm, in_mem, out_mem)
MOVE_CASES = [
    pytest.param((1, U.L2_H, U.L2_W, U.OFT_C), "fp32", True, "dram", "l1", id="intimg-rm-fp32-dram->l1-177182"),
    pytest.param((1, 12, 40, U.OFT_C), "fp32", True, "dram", "l1", id="proxy-intimg-rm-fp32-dram->l1"),
    pytest.param((1, U.L2_H, U.L2_W, U.OFT_C), "bf16", True, "dram", "l1", id="intimg-rm-bf16-dram->l1-936427"),
    pytest.param((1, 12, 40, U.OFT_C), "bf16", True, "dram", "l1", id="proxy-intimg-rm-bf16-dram->l1"),
]
SHARDED_MOVE_CASES = [
    pytest.param((1, 1, U.SLICE_ROWS, U.VOX_C), "fp32", False, "dram", ("hs", (U.TILE, U.VOX_C)), id="area-fp32-dram->hs-502856"),
    pytest.param((1, 1, U.L1_H * U.L1_W, 64), "bf16", False, "dram", ("hs", (1536, 64)), id="conv-in-bf16-dram->hs-154358"),
    pytest.param((1, 1, U.L1_H * U.L1_W, 64), "bfp8", False, "dram", ("hs", (1536, 64)), id="conv-in-bfp8-dram->hs-291869"),
    pytest.param((1, 1, U.L2_H * U.L2_W, U.OFT_C), "fp32", False, ("hs", (384, U.OFT_C)), "dram", id="lat8-fp32-hs->dram-159108"),
    pytest.param((1, 1, U.BEV_NHW, 16), "uint16", True, ("hs", (1265, 16)), "l1", id="nms-indices-uint16-hs->l1-178576"),
    # cache #2 (bf16)
    pytest.param((1, 1, U.L2_H * U.L2_W, 256), "bf16", True, "dram", ("hs", (384, 256)), id="bn8-in-rm-bf16-dram->hs-190297"),
    pytest.param((1, 1, U.L2_H * U.L2_W, 128), "bf16", False, "l1", ("hs", (384, 128)), id="bf16-l1->hs-226960"),
    pytest.param((1, 1, 1408, U.VOX_C), "bf16", False, "dram", ("hs", (96, U.VOX_C)), id="area8-bf16-dram->hs-15cores-115190"),
    pytest.param((1, 1, 2112, U.VOX_C), "bf16", False, "dram", ("hs", (128, U.VOX_C)), id="area16-bf16-dram->hs-17cores-136900"),
    pytest.param((1, 1, 2304, U.VOX_C), "bf16", False, "dram", ("hs", (128, U.VOX_C)), id="area32-bf16-dram->hs-18cores-137435"),
]
# fmt: on


def _run_move(mesh, shape, dt, layout_rm, in_mem, out_mem):
    x = _rand(shape, dt)
    layout = ttnn.ROW_MAJOR_LAYOUT if layout_rm else ttnn.TILE_LAYOUT
    tt_x = U.to_tt_mem(x, mesh, in_mem, dtype=_DT[dt], layout=layout)
    ref = U.seen(tt_x, x, mesh)  # bfloat8_b: the quantized values must survive
    out = U.op("to_memory_config")(tt_x, U.resolve_memcfg(out_mem, mesh))
    U.assert_lossless(ref, out, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize("shape, dt, layout_rm, in_mem, out_mem", MOVE_CASES)
def test_to_memory_config(ttnn_mesh_device, reset_seeds, shape, dt, layout_rm, in_mem, out_mem):
    _run_move(ttnn_mesh_device, shape, dt, layout_rm, in_mem, out_mem)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape, dt, layout_rm, in_mem, out_mem", SHARDED_MOVE_CASES)
def test_to_memory_config_sharded(ttnn_mesh_device, reset_seeds, shape, dt, layout_rm, in_mem, out_mem):
    _run_move(ttnn_mesh_device, shape, dt, layout_rm, in_mem, out_mem)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, dt, shard",
    [
        pytest.param((1, 1, U.SLICE_ROWS, U.VOX_C), "fp32", (U.TILE, U.VOX_C), id="vox-fp32-hs-119210"),
        pytest.param((1, 1, U.BEV_NHW, U.TD_C), "bf16", (1280, U.TD_C), id="topdown-bf16-hs-312368"),
        pytest.param((1, 1, 1408, U.VOX_C), "bf16", (96, U.VOX_C), id="vox8-bf16-hs-159159"),
    ],
)
def test_move_sharded(ttnn_mesh_device, reset_seeds, shape, dt, shard):
    mesh = ttnn_mesh_device
    x = _rand(shape, dt)
    # Free a buffer allocated below the input so move has somewhere to go: if the output lands
    # on the input's own address, move returns without launching the copy kernel.
    tt_dummy = U.to_tt_mem(x, mesh, ("hs", shard), dtype=_DT[dt])
    tt_x = U.to_tt_mem(x, mesh, ("hs", shard), dtype=_DT[dt])
    input_address = tt_x.buffer_address()
    ttnn.deallocate(tt_dummy)
    out = U.op("move")(tt_x)
    assert out.buffer_address() != input_address, "move did not relocate the input buffer"
    assert out.is_sharded(), "move output lost its sharded memory layout"
    U.assert_lossless(x, out, mesh_device=mesh)


def _run_partial_s2i(mesh, shard_mem, num_slices, slice_rows, width):
    """Each slice is height-sharded, then written into its rows of a DRAM tensor (the ortho feats)."""
    total = num_slices * slice_rows
    full = U.torch_rand((1, 1, total, width))
    ortho = U.to_tt(torch.zeros(1, 1, total, width, dtype=torch.bfloat16), mesh)
    for i in range(num_slices):
        part = full[:, :, i * slice_rows : (i + 1) * slice_rows, :]
        tt_part = ttnn.to_memory_config(U.to_tt(part, mesh), shard_mem)
        ttnn.sharded_to_interleaved_partial(tt_part, ortho, num_slices, i, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    U.assert_lossless(full, ortho, mesh_device=mesh)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [pytest.param((1, 1, U.BEV_ROWS, U.TD_C), id="ortho-44x576x128-532985")],
)
def test_sharded_to_interleaved_partial(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device
    shard_mem = U.resolve_memcfg(("hs", (U.TILE, U.TD_C)), mesh)
    _run_partial_s2i(mesh, shard_mem, U.NUM_SLICES, U.SLICE_ROWS, U.TD_C)


@U.with_default_mesh()
@pytest.mark.parametrize("shape", [pytest.param((1, 1, 4 * 64, U.TD_C), id="proxy-ortho-4x64x128-2cores")])
def test_sharded_to_interleaved_partial_small(ttnn_mesh_device, reset_seeds, shape):
    """Same op with 4 slices of 64 rows, each sharded over 2 cores (emulator-sized)."""
    mesh = ttnn_mesh_device
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    shard_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [U.TILE, U.TD_C], ttnn.ShardOrientation.ROW_MAJOR),
    )
    _run_partial_s2i(mesh, shard_mem, 4, 64, U.TD_C)


# =============================================================================
# Second cache (in-tree OFT at 628a4797bf7)
# =============================================================================


def _memcfg(mesh, spec):
    """ "dram" / "l1" / ("hs" | "bs", shard, (grid_x, grid_y)): HS ROW_MAJOR, BS COL_MAJOR (GroupNorm BS)."""
    if spec in ("dram", "l1"):
        return U.resolve_memcfg(spec, mesh)
    kind, shard, grid = spec
    cores = U.model_grid(mesh, *grid)
    if kind == "hs":
        shard_spec = ttnn.ShardSpec(cores, list(shard), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    shard_spec = ttnn.ShardSpec(cores, list(shard), ttnn.ShardOrientation.COL_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)


G5x4, G5x3 = (U.MODEL_GRID_X, U.MODEL_GRID_Y), (5, 3)


# fmt: off
S2I_CASES = [
    pytest.param((1, 1, U.L2_H * U.L2_W, 128), False, ("hs", (384, 128), G5x4), id="feats8-bf16-hs->dram-164400"),
    pytest.param((1, 1, 12 * 40, 512), False, ("hs", (32, 512), G5x3), id="feats32-bf16-hs15->dram-164400"),
    pytest.param((1, 1, U.L2_H * U.L2_W, 256), True, ("hs", (384, 256), G5x4), id="lat8-rm-bf16-hs->dram-165319"),
    pytest.param((1, 1, 12 * 40, 256), True, ("bs", (96, 64), G5x4), id="lat32-rm-bf16-bs->dram-165319"),
]
S2I_PROXY_CASES = [
    pytest.param((1, 1, 128, 128), False, ("hs", (64, 128), (2, 1)), id="proxy-bf16-hs->dram"),
    pytest.param((1, 1, 128, 256), True, ("hs", (64, 256), (2, 1)), id="proxy-rm-bf16-hs->dram"),
]
# fmt: on


def _run_s2i(mesh, shape, rm, spec):
    x = U.torch_rand(shape)
    layout = ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT
    tt_x = U.op("to_memory_config")(U.to_tt(x, mesh, layout=layout), _memcfg(mesh, spec))
    out = U.op("sharded_to_interleaved")(tt_x, ttnn.DRAM_MEMORY_CONFIG)
    assert not out.is_sharded(), "sharded_to_interleaved output is still sharded"
    U.assert_lossless(x, out, mesh_device=mesh)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape, rm, spec", S2I_CASES)
def test_sharded_to_interleaved(ttnn_mesh_device, reset_seeds, shape, rm, spec):
    """``ttnn.sharded_to_interleaved(x, DRAM)``: ResNet feature maps and lateral GroupNorm outputs."""
    _run_s2i(ttnn_mesh_device, shape, rm, spec)


@U.with_default_mesh()
@pytest.mark.parametrize("shape, rm, spec", S2I_PROXY_CASES)
def test_sharded_to_interleaved_small(ttnn_mesh_device, reset_seeds, shape, rm, spec):
    _run_s2i(ttnn_mesh_device, shape, rm, spec)


def _run_partial_i2s(mesh, shape, num_slices, shard, grid):
    """``interleaved_to_sharded_partial`` of each height slice of a DRAM TILE tensor (stem GN halves)."""
    U.model_grid(mesh, *grid)
    x = U.torch_rand(shape)
    tt_x = U.to_tt(x, mesh)
    slice_rows = shape[2] // num_slices
    for i in range(num_slices):
        part = ttnn.interleaved_to_sharded_partial(
            tt_x,
            ttnn.CoreCoord(*grid),
            list(shard),
            num_slices,
            i,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        assert part.is_sharded(), "interleaved_to_sharded_partial output is not sharded"
        U.assert_lossless(x[:, :, i * slice_rows : (i + 1) * slice_rows, :], part, mesh_device=mesh)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, num_slices, shard, grid",
    [pytest.param((1, 1, U.STEM_H * U.STEM_W, 32), 2, (3072, 32), G5x4, id="stem-gn-half-2x61440x32-465767")],
)
def test_interleaved_to_sharded_partial(ttnn_mesh_device, reset_seeds, shape, num_slices, shard, grid):
    _run_partial_i2s(ttnn_mesh_device, shape, num_slices, shard, grid)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, num_slices, shard, grid",
    [pytest.param((1, 1, 256, 32), 2, (64, 32), (2, 1), id="proxy-2x128x32-2cores")],
)
def test_interleaved_to_sharded_partial_small(ttnn_mesh_device, reset_seeds, shape, num_slices, shard, grid):
    _run_partial_i2s(ttnn_mesh_device, shape, num_slices, shard, grid)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, num_slices, shard_h",
    [
        pytest.param((1, 1, U.STEM_H * U.STEM_W, 64), 2, 3072, id="stem-conv1-2x61440x64-164400"),
        pytest.param((1, 1, U.BEV_ROWS, 256), 18, 96, id="ortho8-18x1408x256-15cores-164400"),
        pytest.param((1, 1, U.BEV_ROWS, 256), 12, 128, id="ortho16-12x2112x256-17cores-164400"),
        pytest.param((1, 1, U.BEV_ROWS, 256), 11, 128, id="ortho32-11x2304x256-18cores-164400"),
    ],
)
def test_sharded_to_interleaved_partial_oft2(ttnn_mesh_device, reset_seeds, shape, num_slices, shard_h):
    """bf16 TILE slices sharded over the 5x4 grid (partly empty shards for the OFT scales) into DRAM."""
    mesh = ttnn_mesh_device
    width = shape[3]
    shard_mem = U.resolve_memcfg(("hs", (shard_h, width)), mesh)
    _run_partial_s2i(mesh, shard_mem, num_slices, shape[2] // num_slices, width)


def _run_move_dram(mesh, shape):
    x = U.torch_rand(shape)
    # As in test_move_sharded: free a buffer next to the input so move has somewhere to go
    # (move returns without launching the copy when the output lands on the input's address).
    tt_dummy = U.to_tt(x, mesh)
    tt_x = U.to_tt(x, mesh)
    input_address = tt_x.buffer_address()
    ttnn.deallocate(tt_dummy)
    out = U.op("move")(tt_x)
    assert out.buffer_address() != input_address, "move did not relocate the input buffer"
    assert not out.is_sharded(), "move changed the memory layout"
    U.assert_lossless(x, out, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 25344, 256), id="topdown-conv-out-bf16-dram-128139"),
        pytest.param((1, 1, 1024, 256), id="proxy-bf16-dram"),
    ],
)
def test_move_dram(ttnn_mesh_device, reset_seeds, shape):
    """``ttnn.move`` of a DRAM-interleaved bf16 TILE tensor (the topdown conv output)."""
    _run_move_dram(ttnn_mesh_device, shape)
