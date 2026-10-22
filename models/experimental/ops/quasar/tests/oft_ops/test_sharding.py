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
]
SHARDED_MOVE_CASES = [
    pytest.param((1, 1, U.SLICE_ROWS, U.VOX_C), "fp32", False, "dram", ("hs", (U.TILE, U.VOX_C)), id="area-fp32-dram->hs-502856"),
    pytest.param((1, 1, U.L1_H * U.L1_W, 64), "bf16", False, "dram", ("hs", (1536, 64)), id="conv-in-bf16-dram->hs-154358"),
    pytest.param((1, 1, U.L1_H * U.L1_W, 64), "bfp8", False, "dram", ("hs", (1536, 64)), id="conv-in-bfp8-dram->hs-291869"),
    pytest.param((1, 1, U.L2_H * U.L2_W, U.OFT_C), "fp32", False, ("hs", (384, U.OFT_C)), "dram", id="lat8-fp32-hs->dram-159108"),
    pytest.param((1, 1, U.BEV_NHW, 16), "uint16", True, ("hs", (1265, 16)), "l1", id="nms-indices-uint16-hs->l1-178576"),
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
    U.xfail_no_quasar_port("sharded_to_interleaved_partial")
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
