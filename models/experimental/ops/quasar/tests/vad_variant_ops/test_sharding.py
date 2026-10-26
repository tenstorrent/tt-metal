# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: interleaved <-> sharded conversions and reshard (``ttnn.to_memory_config``).

Shape source (kernel cache):
  * interleaved_to_sharded (``reader_unary_sharded_blocks_interleaved_start_id`` ct
    [cb, num_shard_cores=20, TA]) from bf16 DRAM, bf16 L1 and bfp8 L1  [H cores / dtype]
  * sharded_to_interleaved (``writer_unary_sharded_blocks_interleaved_start_id``) to bf16
    DRAM, bf16 L1 and bfp8 L1  [H dtype]
  * generic reshard (``reshard_reader`` ct [16, grid.x=5, grid.y=4, page, unit]) on bfp8  [H grid]
Shapes are runtime args [L]: BLOCK sharding of (1, 1, 640, 256) on the 5x4 grid. The BEV
height-sharded layout (1, 1, 10000, 256) / 512-row shards is covered as the model's main
sharded tensor. Value-preserving: checked bit-exact against the (quantized) source.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

Q, E = U.NUM_QUERY, U.EMBED

BLOCK_SHAPE = (1, 1, 640, E)
BLOCK_SPEC = ("bs", (128, 64))  # 5 x 4 grid, COL_MAJOR: height over x (5), width over y (4)

# fmt: off
# (shape, dtype, interleaved_mem, sharded_spec)
CASES = [
    pytest.param(BLOCK_SHAPE, "bf16", "dram", BLOCK_SPEC, id="SH1-bf16-dram"),
    pytest.param(BLOCK_SHAPE, "bf16", "l1", BLOCK_SPEC, id="SH2-bf16-l1"),
    pytest.param(BLOCK_SHAPE, "bfp8", "l1", BLOCK_SPEC, id="SH3-bfp8-l1"),
    pytest.param((1, 1, Q, E), "bf16", "dram", ("hs", (512, E)), id="SH-bev-hs20-dram"),
    pytest.param((1, 1, Q, E), "bf16", "l1", ("hs", (512, E)), id="SH-bev-hs20-l1"),
]
# fmt: on


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape, dt, mem, shard_spec", CASES)
def test_interleaved_sharded_roundtrip(ttnn_mesh_device, reset_seeds, shape, dt, mem, shard_spec):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt_mem(x_torch, mesh, mem, dtype=U.DTYPES[dt])
    ref = U.seen(x, x_torch, mesh)

    sharded = ttnn.to_memory_config(x, U.resolve_memcfg(shard_spec, mesh))  # interleaved_to_sharded
    assert sharded.is_sharded()
    U.assert_lossless(ref, sharded, mesh_device=mesh)

    back = ttnn.to_memory_config(sharded, U.resolve_memcfg(mem, mesh))  # sharded_to_interleaved
    assert not back.is_sharded()
    U.assert_lossless(ref, back, mesh_device=mesh)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape", [pytest.param(BLOCK_SHAPE, id="SH7-bfp8-block-to-height")])
def test_reshard(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt_mem(x_torch, mesh, BLOCK_SPEC, dtype=ttnn.bfloat8_b)
    ref = U.seen(x, x_torch, mesh)
    out = ttnn.to_memory_config(x, U.resolve_memcfg(("hs", (32, E)), mesh))
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    U.assert_lossless(ref, out, mesh_device=mesh)
