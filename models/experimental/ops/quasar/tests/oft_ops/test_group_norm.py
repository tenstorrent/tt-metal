# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.group_norm`` — every GroupNorm the in-tree OFT model ran on device
(``GroupNorm`` / ``GroupNormDRAM`` in ``oft/tt/common.py``; 16 groups everywhere, eps 1e-5).

Shape source: second kernel cache (in-tree OFT, factories at ``628a4797bf7``). Cache #1 had no
group_norm kernels (GroupNorm ran on host there).

Sharded path (``groupnorm_sharded_v2`` compute, 8 configs x {mcast sender, receiver}). The compute
compile-time args give num_cores_per_mcast_group, groups per core, channels per group, per-core
M / N in tiles and the TILIZE_IN / UNTILIZE_OUT / FUSE_NEGATIVE_MASK defines. All bf16 (every CB
fp16b), gamma / beta present (ROW_MAJOR DRAM, 64-B sticks), HiFi4, approx on, no fp32 dest acc
(the op's default compute config: the model passes none), not Welford.          [H]

  ===========  =====  ======  =========  ===================  ==========  ===============  =========
  model        C      groups  H*W        sharding             shard       layout           hash
  ===========  =====  ======  =========  ===================  ==========  ===============  =========
  stem bn1 x2  32     8       192*640    HS, 20 cores         (6144, 32)  RM, neg. mask    176183
  layer1       64     16      96*320     HS, 20 cores         (1536, 64)  TILE             125128
  layer2       128    16      48*160     HS, 20 cores         (384, 128)  TILE             181657
  layer3       256    16      24*80      HS, 20 cores         (96, 256)   TILE             175741
  layer4       512    16      12*40      BS 5x4, COL_MAJOR    (96, 128)   TILE             127123
  bn8 (lat8)   256    16      48*160     HS, 20 cores         (384, 256)  RM (in-place)    531458
  bn16         256    16      24*80      HS, 20 cores         (96, 256)   RM (in-place)    323219
  bn32         256    16      12*40      BS 5x4, COL_MAJOR    (96, 64)    RM (in-place)    120528
  ===========  =====  ======  =========  ===================  ==========  ===============  =========

  * The stem GroupNorm(16, 64) runs as two ``ttnn.split`` halves of 32 channels (``num_splits=2``:
    8 groups each), with ``negative_mask`` and the input untilized + ``ttnn.move``-d first, so it
    runs ``inplace`` on ROW_MAJOR data.                                                     [H]
  * The model passes ``core_grid = CoreGrid(5, 4)`` for both HS and BS; masks / gamma / beta are
    built for 1 core across channels (HS) or 4 (BS), exactly as ``GroupNorm.__call__`` does.
  * Model bug, not reproduced: the stem passes the full 64-channel gamma / beta to
    ``create_group_norm_weight_bias_rm(w, 32, 1)``, which keeps only the first 32 values for both
    halves. The test gives each call its own 32-channel gamma / beta.

DRAM path (``GroupNormDRAM``, topdown BasicBlocks): ``groupnorm`` compute (sender + receiver),
``reader_mcast_{sender,receiver}_unary_gn``, ``writer_unary_gn_rm_gb``. Input and output DRAM
interleaved bf16 TILE (accessors [DRAM, 2048 B]), gamma / beta sticks 64 B, bf16 mask tile, HiFi4,
no TILIZE_IN (input already TILE).                                                           [H]
Its arguments are named compile-time args that the ELF does not keep, so the shape comes from the
model: 256 channels, 16 groups, 159*159 rows tilized to 25344 (``_nearest_32_per_core(25281, 4)``),
``core_grid = CoreGrid(y=4, x=4)``, ``num_out_blocks = 2`` (``num_splits=2``).                [M]
"""

import pytest
import torch.nn.functional as F

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

EPS = 1e-5
STEM_ROWS = U.STEM_H * U.STEM_W  # 122880
BEV_PADDED = 25344  # 159 * 159 = 25281 tilized to 4 cores x 32-row multiples


def _shard_memcfg(mesh, shape, shard, grid):
    """The model's GroupNorm memory config: HS (ROW_MAJOR over all cores) or BS (COL_MAJOR)."""
    gx, gy = grid
    cores = U.model_grid(mesh, gx, gy)  # skips when the device grid is smaller
    hw, c = shape[2], shape[3]
    if shard == "hs":
        spec = ttnn.ShardSpec(cores, [hw // (gx * gy), c], ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    spec = ttnn.ShardSpec(cores, [hw // gx, c // gy], ttnn.ShardOrientation.COL_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)


def _torch_group_norm(x, groups, weight, bias):
    """torch reference on a [1, 1, H*W, C] (NHWC-flattened) tensor."""
    _, _, hw, c = x.shape
    nchw = x.float().reshape(1, hw, c).permute(0, 2, 1)
    out = F.group_norm(nchw, groups, weight.float(), bias.float(), eps=EPS)
    return out.permute(0, 2, 1).reshape(x.shape)


def _run_sharded(mesh, shape, groups, shard, rm, neg, grid):
    c = shape[3]
    gx, gy = grid
    memcfg = _shard_memcfg(mesh, shape, shard, grid)
    cores_across_channel = 1 if shard == "hs" else gy  # GroupNorm.__call__: HS folds the grid to (x*y, 1)

    x = U.torch_rand(shape)
    weight = U.torch_rand((c,))
    bias = U.torch_rand((c,))

    if neg:
        # stem: DRAM TILE split -> HS -> untilize -> move, then in-place GN on ROW_MAJOR
        tt_x = U.op("to_memory_config")(U.to_tt(x, mesh), memcfg)
        tt_x = U.op("to_layout")(tt_x, ttnn.ROW_MAJOR_LAYOUT)
        tt_x = U.op("move")(tt_x)
    else:
        layout = ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT
        tt_x = U.op("to_memory_config")(U.to_tt(x, mesh, layout=layout), memcfg)

    mask = ttnn.to_device(ttnn.create_group_norm_input_mask(c, groups, cores_across_channel, ttnn.bfloat16), mesh)
    nmask = None
    if neg:
        nmask = ttnn.create_group_norm_input_negative_mask(c, groups, cores_across_channel, ttnn.bfloat16)
        nmask = ttnn.to_device(nmask, mesh)
    gamma = U.to_tt(
        ttnn.create_group_norm_weight_bias_rm(weight, c, cores_across_channel), mesh, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    beta = U.to_tt(
        ttnn.create_group_norm_weight_bias_rm(bias, c, cores_across_channel), mesh, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    out = ttnn.group_norm(
        tt_x,
        num_groups=groups,
        input_mask=mask,
        negative_mask=nmask,
        weight=gamma,
        bias=beta,
        memory_config=memcfg,
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        epsilon=EPS,
        inplace=tt_x.layout == ttnn.ROW_MAJOR_LAYOUT,
    )
    U.assert_pcc(_torch_group_norm(x, groups, weight, bias), out, pcc=0.99, mesh_device=mesh)


_ARGS = "shape, groups, shard, rm, neg, grid"
G5x4 = (U.MODEL_GRID_X, U.MODEL_GRID_Y)

# fmt: off
SHARDED_CASES = [
    pytest.param((1, 1, STEM_ROWS, 32), 8, "hs", True, True, G5x4, id="stem-split-c32-g8-hs-rm-negmask-176183"),
    pytest.param((1, 1, U.L1_H * U.L1_W, 64), 16, "hs", False, False, G5x4, id="layer1-c64-hs-1536-125128"),
    pytest.param((1, 1, U.L2_H * U.L2_W, 128), 16, "hs", False, False, G5x4, id="layer2-c128-hs-384-181657"),
    pytest.param((1, 1, 24 * 80, 256), 16, "hs", False, False, G5x4, id="layer3-c256-hs-96-175741"),
    pytest.param((1, 1, 12 * 40, 512), 16, "bs", False, False, G5x4, id="layer4-c512-bs-96x128-127123"),
    pytest.param((1, 1, U.L2_H * U.L2_W, 256), 16, "hs", True, False, G5x4, id="bn8-c256-hs-384-rm-531458"),
    pytest.param((1, 1, 24 * 80, 256), 16, "hs", True, False, G5x4, id="bn16-c256-hs-96-rm-323219"),
    pytest.param((1, 1, 12 * 40, 256), 16, "bs", True, False, G5x4, id="bn32-c256-bs-96x64-rm-120528"),
]

# Same op configs on 2 cores (emulator-sized).
SHARDED_PROXY_CASES = [
    pytest.param((1, 1, 128, 32), 8, "hs", True, True, (2, 1), id="proxy-c32-g8-hs-rm-negmask"),
    pytest.param((1, 1, 128, 64), 16, "hs", False, False, (2, 1), id="proxy-c64-hs-tile"),
    pytest.param((1, 1, 128, 256), 16, "hs", True, False, (2, 1), id="proxy-c256-hs-rm"),
    pytest.param((1, 1, 64, 128), 16, "bs", False, False, (1, 2), id="proxy-c128-bs-tile"),
    pytest.param((1, 1, 64, 128), 16, "bs", True, False, (1, 2), id="proxy-c128-bs-rm"),
]
# fmt: on


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(_ARGS, SHARDED_CASES)
def test_group_norm_sharded(ttnn_mesh_device, reset_seeds, shape, groups, shard, rm, neg, grid):
    _run_sharded(ttnn_mesh_device, shape, groups, shard, rm, neg, grid)


@U.with_default_mesh()
@pytest.mark.parametrize(_ARGS, SHARDED_PROXY_CASES)
def test_group_norm_sharded_small(ttnn_mesh_device, reset_seeds, shape, groups, shard, rm, neg, grid):
    _run_sharded(ttnn_mesh_device, shape, groups, shard, rm, neg, grid)


def _run_dram(mesh, shape, groups, grid, num_out_blocks, pad_rows):
    gx, gy = grid
    U.model_grid(mesh, gx, gy)
    c = shape[3]
    core_grid = ttnn.CoreGrid(y=gy, x=gx)

    x = U.torch_rand(shape)
    if pad_rows:
        x[:, :, -pad_rows:, :] = 0  # tilize_with_val_padding(pad_value=0) rows: they count in the stats
    weight = U.torch_rand((c,))
    bias = U.torch_rand((c,))

    tt_x = U.to_tt(x, mesh)  # DRAM interleaved bf16 TILE
    [gamma, beta], mask = ttnn.dram_group_norm_params_from_torch(
        [weight, bias], c, groups, mesh, core_grid=core_grid, return_mask=True
    )
    out = ttnn.group_norm(
        tt_x,
        num_groups=groups,
        input_mask=mask,
        weight=gamma,
        bias=beta,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_layout=ttnn.TILE_LAYOUT,
        core_grid=core_grid,
        inplace=False,
        num_out_blocks=num_out_blocks,
        epsilon=EPS,
    )
    U.assert_pcc(_torch_group_norm(x, groups, weight, bias), out, pcc=0.99, mesh_device=mesh)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, groups, grid, num_out_blocks, pad_rows",
    [
        pytest.param(
            (1, 1, BEV_PADDED, 256), 16, (4, 4), 2, BEV_PADDED - U.BEV_NHW, id="topdown-c256-dram-4x4-nob2-240754"
        ),
    ],
)
def test_group_norm_dram(ttnn_mesh_device, reset_seeds, shape, groups, grid, num_out_blocks, pad_rows):
    if ttnn.device.is_wormhole_b0(ttnn_mesh_device):
        # The model's config (4x4 grid, num_out_blocks=2) puts ~1.48 MiB of CBs on the mcast
        # sender core: fits Blackhole's 1.5 MiB L1, not Wormhole's 1464 KiB.
        pytest.skip("model-faithful DRAM group_norm CBs exceed Wormhole L1")
    _run_dram(ttnn_mesh_device, shape, groups, grid, num_out_blocks, pad_rows)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, groups, grid, num_out_blocks, pad_rows",
    [pytest.param((1, 1, 512, 256), 16, (2, 1), 2, 0, id="proxy-c256-dram-2x1-nob2")],
)
def test_group_norm_dram_small(ttnn_mesh_device, reset_seeds, shape, groups, grid, num_out_blocks, pad_rows):
    _run_dram(ttnn_mesh_device, shape, groups, grid, num_out_blocks, pad_rows)
