# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: sharded -> sharded ``ttnn.to_memory_config`` (the reshard op) as the in-tree OFT
ResNet ran it.

Shape source: second kernel cache (in-tree OFT, reshard factories at ``628a4797bf7``):

  * ``reshard_reader/140579`` (+ 112306, the same kernel on the other RISC), generic factory,
    ct = [cb 16, grid 5, 4, 2048, 2048]: bf16 TILE, equal input / output page sizes, L1 on the
    5x4 grid. The one generic reshard in ``tt_resnet.py`` is layer4's block-sharded GroupNorm
    output (and identity) moved to the 15-core height-sharded config of the residual add:
    ``create_sharded_memory_config([480 // 15, 512], CoreGrid(y=3, x=5), HEIGHT, ROW_MAJOR)``;
    BS (96, 128) COL_MAJOR on 5x4 -> HS (32, 512) on 5x3.                              [H kernel / M shape]
  * ``reshard_same_width_reader/156531`` (+ 321848), ct = [cb 0, not DRAM, aligned, 2048, 2048,
    2048, scratch cb 1]: height-sharded -> height-sharded, same width, bf16 TILE, L1. The shard
    heights are runtime args; a conv-output -> GroupNorm-input reshard of layer3 is used as a
    representative (HS (128, 256) on 15 cores -> HS (96, 256) on 20 cores).                 [L]

Value-preserving: checked bit-exact. ``to_memory_config`` has a Quasar port (it dispatches to the
Quasar reshard), so it is called through ``U.op``.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U


def _memcfg(mesh, spec):
    """("hs" | "bs", shard, (grid_x, grid_y)): HS is ROW_MAJOR, BS is COL_MAJOR (the model's GroupNorm BS)."""
    kind, shard, grid = spec
    cores = U.model_grid(mesh, *grid)
    if kind == "hs":
        shard_spec = ttnn.ShardSpec(cores, list(shard), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    shard_spec = ttnn.ShardSpec(cores, list(shard), ttnn.ShardOrientation.COL_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)


def _run(mesh, shape, in_spec, out_spec):
    x = U.torch_rand(shape)
    tt_x = U.op("to_memory_config")(U.to_tt(x, mesh), _memcfg(mesh, in_spec))
    out_mem = _memcfg(mesh, out_spec)
    out = U.op("to_memory_config")(tt_x, out_mem)
    assert out.memory_config().memory_layout == out_mem.memory_layout, "reshard produced the wrong layout"
    U.assert_lossless(x, out, mesh_device=mesh)


G5x4, G5x3 = (5, 4), (5, 3)

# fmt: off
RESHARD_CASES = [
    pytest.param((1, 1, 12 * 40, 512), ("bs", (96, 128), G5x4), ("hs", (32, 512), G5x3), id="layer4-bs5x4->hs15-140579"),
    pytest.param((1, 1, 24 * 80, 256), ("hs", (128, 256), G5x3), ("hs", (96, 256), G5x4), id="layer3-hs15->hs20-same-width-156531"),
]
RESHARD_PROXY_CASES = [
    pytest.param((1, 1, 64, 128), ("bs", (64, 64), (1, 2)), ("hs", (32, 128), (2, 1)), id="proxy-bs1x2->hs2"),
    pytest.param((1, 1, 128, 64), ("hs", (128, 64), (1, 1)), ("hs", (64, 64), (2, 1)), id="proxy-hs1->hs2-same-width"),
]
# fmt: on


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape, in_spec, out_spec", RESHARD_CASES)
def test_reshard(ttnn_mesh_device, reset_seeds, shape, in_spec, out_spec):
    _run(ttnn_mesh_device, shape, in_spec, out_spec)


@U.with_default_mesh()
@pytest.mark.parametrize("shape, in_spec, out_spec", RESHARD_PROXY_CASES)
def test_reshard_small(ttnn_mesh_device, reset_seeds, shape, in_spec, out_spec):
    _run(ttnn_mesh_device, shape, in_spec, out_spec)
